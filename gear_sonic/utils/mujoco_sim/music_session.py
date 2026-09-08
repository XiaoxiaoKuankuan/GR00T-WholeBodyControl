"""G1 音乐演示的 MuJoCo 会话、统一时间轴与故障现场记录。

仅在显式指定本机 music_endpoint 时启用。网络线程处理幂等 JSON 控制消息，
所有 reset、弹力带切换和动力学状态访问均在仿真线程执行。固定步长保持不变，
跌倒、心跳超时和时间偏差超过 100 ms 会冻结物理推进，保留窗口及真实 qpos/qvel。
JSONL 日志按 50 Hz 保存实际状态，可用于诊断和复现，而非替代实时验收。
常驻模式不在换歌时 reset 或自动松绳；窗口的 ]、9、P 经过事件队列传入仿真
线程，分别请求策略起控、切换吊绳及停止当前表演，起控与停止计数由 GENMO 确认。
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import numpy as np
import zmq


class MusicSimSession:
    """控制消息与仿真状态之间通过短时锁交换，网络不阻塞物理计算。"""

    def __init__(self, endpoint: str, log_path: str = ""):
        if not endpoint.startswith("tcp://127.0.0.1:"):
            raise ValueError("音乐仿真接口只能绑定本机")
        self.lock = threading.RLock()
        self.state, self.session_id, self.error = "idle", "", ""
        self.epoch_ns = self.heartbeat_ns = 0
        self.sim_origin = None
        self.sim_time = self.skew_ms = self.tilt_deg = 0.0
        self.band_enabled, self.reset_pending = True, False
        self.resident = False
        self.enable_requests = self.stop_requests = 0
        self.pending_keys = []
        self.hang_height = None
        self.replies, self.last_seq = {}, -1
        self.running = True
        self.ready = threading.Event()
        self.server_error, self.log = None, None
        self.last_log_time = -1.0
        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.log = path.open("x", encoding="utf-8")
        self.thread = threading.Thread(target=self._serve, args=(endpoint,), daemon=True)
        self.thread.start()
        self.ready.wait(5)
        if self.server_error or not self.ready.is_set():
            raise RuntimeError(self.server_error or "MuJoCo 音乐接口启动超时")

    def status(self):
        """返回仿真线程最近一次采样，网络线程不直接读取 mj_data。"""
        return dict(
            session_id=self.session_id,
            state=self.state,
            error=self.error,
            epoch_ns=self.epoch_ns,
            sim_time=self.sim_time,
            sim_elapsed=None if self.sim_origin is None else self.sim_time - self.sim_origin,
            skew_ms=self.skew_ms,
            tilt_deg=self.tilt_deg,
            band_enabled=self.band_enabled,
            resident=self.resident,
            hang_height=self.hang_height,
            enable_requests=self.enable_requests,
            stop_requests=self.stop_requests,
            monotonic_ns=time.monotonic_ns(),
        )

    def request(self, req):
        """幂等控制入口；只有新 prepare 才清除旧会话的故障。"""
        with self.lock:
            op, sid, seq = req["op"], req.get("session_id", ""), req["seq"]
            if not isinstance(seq, int) or seq < 0:
                raise ValueError("请求序号非法")
            key = json.dumps(req, sort_keys=True)
            if sid == self.session_id and seq in self.replies:
                old, reply = self.replies[seq]
                if old != key:
                    raise ValueError("相同请求序号的内容冲突")
                self.heartbeat_ns = time.monotonic_ns()
                return reply
            if op == "resident":
                if not sid or self.session_id:
                    raise ValueError("常驻初始化需要空闲的新仿真")
                height = float(req.get("hang_height", 0.82))
                if not np.isfinite(height) or not 0.78 <= height <= 1.2:
                    raise ValueError("常驻吊绳锚点高度必须位于 0.78 至 1.2 米")
                self.hang_height = height
                self.session_id, self.state, self.resident = sid, "standing", True
                self.replies.clear()
                self.last_seq = -1
            elif op == "prepare":
                continuing = self.resident and sid == self.session_id
                if not sid or (sid == self.session_id and not continuing) or self.state in ("playing", "armed", "prepared"):
                    raise ValueError("无法在当前状态准备此会话")
                if self.resident and (not continuing or self.state != "standing"):
                    raise ValueError("常驻音乐只能从站姿开始，故障不得通过换歌清除")
                if continuing and seq <= self.last_seq:
                    raise ValueError("请求序号已过期")
                self.session_id, self.state, self.error = sid, "prepared", ""
                self.epoch_ns, self.sim_origin = 0, None
                if not continuing:
                    self.replies.clear()
                    self.last_seq, self.reset_pending = -1, True
            elif op != "status" or sid:
                if sid != self.session_id or seq <= self.last_seq:
                    raise ValueError("会话或请求序号不匹配")
                if op == "start":
                    epoch = int(req["epoch_ns"])
                    if self.state != "prepared" or self.reset_pending:
                        raise ValueError("仿真尚未准备完成")
                    if self.resident and self.band_enabled:
                        raise ValueError("请先在 MuJoCo 按 9 松开吊绳")
                    if not time.monotonic_ns() + 100_000_000 <= epoch <= time.monotonic_ns() + 10_000_000_000:
                        raise ValueError("起播时间必须位于未来 0.1 至 10 秒")
                    self.epoch_ns, self.state = epoch, "armed"
                elif op == "stop":
                    self.state = "fault" if req.get("fault") else "stopped"
                    self.error = req.get("reason", "用户停止")
                elif op == "finish":
                    if self.state in ("fault", "stopped"):
                        raise ValueError("故障会话不能恢复站姿")
                    self.state = "standing" if self.resident else "finished"
                    if self.resident:
                        self.epoch_ns, self.sim_origin, self.skew_ms = 0, None, 0.0
                    if self.log:
                        self.log.flush()
                elif op == "key":
                    if not self.resident or req.get("key") not in ("]", "9", "p"):
                        raise ValueError("不支持的常驻仿真按键")
                    self.pending_keys.append(req["key"])
                elif op != "status":
                    raise ValueError("未知仿真音乐操作")
            if sid == self.session_id:
                self.heartbeat_ns = time.monotonic_ns()
            reply = dict(self.status(), ok=True, seq=seq)
            if sid == self.session_id:
                self.last_seq = max(seq, self.last_seq)
                self.replies[seq] = key, reply
                while len(self.replies) > 16:
                    del self.replies[next(iter(self.replies))]
            return reply

    def keyboard(self, key):
        """GLFW 回调只投递事件，避免在窗口线程操作动力学或网络。"""
        with self.lock:
            if not self.resident or key not in ("]", "9", "p"):
                return False
            self.pending_keys.append(key)
            return True

    def _serve(self, endpoint):
        """REP 线程对非法消息也回复错误，避免请求端永久等待。"""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 100)
        socket.setsockopt(zmq.SNDTIMEO, 100)
        socket.setsockopt(zmq.MAXMSGSIZE, 65536)
        try:
            socket.bind(endpoint)
            self.ready.set()
            while self.running:
                try:
                    parts = socket.recv_multipart()
                except zmq.Again:
                    continue
                message = None
                try:
                    if len(parts) != 1:
                        raise ValueError("仿真会话只接受一段 JSON")
                    message = json.loads(parts[0])
                    reply = self.request(message)
                except Exception as exc:
                    reply = dict(ok=False, error=str(exc))
                    with self.lock:
                        if self.state in ("playing", "armed") and (
                            not isinstance(message, dict) or message.get("session_id") == self.session_id
                        ):
                            self.state, self.error = "fault", str(exc)
                socket.send_json(reply)
        except Exception as exc:
            self.server_error = str(exc)
            self.ready.set()
        finally:
            socket.close()
            context.term()

    def before_step(self, env):
        """仿真线程执行准备和弹力带释放；故障后继续绘制但停止步进。"""
        with self.lock:
            now = time.monotonic_ns()
            if self.resident and hasattr(getattr(env, "elastic_band", None), "point"):
                # 原 1 米悬挂下，策略长时间空踩后的自由落体对释放相位敏感。
                # 只降低常驻准备锚点，仍由用户按 9 真正解除；不重置姿态或保留隐形拉力。
                point = np.asarray(env.elastic_band.point, dtype=float)
                point[2] += np.clip(self.hang_height - point[2], -0.001, 0.001)
                env.elastic_band.point = point
            for key in self.pending_keys:
                if key == "]":
                    self.enable_requests += 1
                elif key == "p":
                    self.stop_requests += 1
                elif key == "9" and hasattr(env, "elastic_band"):
                    env.elastic_band.enable = not env.elastic_band.enable
            self.pending_keys.clear()
            if self.reset_pending:
                env.reset()
                if hasattr(env, "elastic_band"):
                    env.elastic_band.enable = True
                self.reset_pending, self.last_log_time = False, -1
            if (self.state in ("playing", "armed") or self.resident) and now - self.heartbeat_ns > 1_500_000_000:
                self.state, self.error = "fault", "协调器心跳超时"
            if self.state == "armed" and now >= self.epoch_ns:
                self.state, self.sim_origin = "playing", float(env.mj_data.time)
                if not self.resident and hasattr(env, "elastic_band"):
                    env.elastic_band.enable = False
            self.band_enabled = bool(getattr(getattr(env, "elastic_band", None), "enable", False))
            return self.state not in ("fault", "stopped")

    def after_step(self, env):
        """记录真实姿态并锁存失稳或时间偏差，不用重置机器人掩盖失败。"""
        with self.lock:
            self.sim_time = float(env.mj_data.time)
            now, qpos = time.monotonic_ns(), env.mj_data.qpos
            self.tilt_deg = float(np.degrees(np.arccos(np.clip(1 - 2 * (qpos[4] ** 2 + qpos[5] ** 2), -1, 1))))
            if self.state == "playing" or (self.resident and not self.band_enabled):
                if self.state == "playing":
                    self.skew_ms = ((now - self.epoch_ns) / 1e9 - (self.sim_time - self.sim_origin)) * 1000
                if not np.isfinite(qpos).all() or float(qpos[2]) < 0.2:
                    self.state, self.error = "fault", "仿真跌倒或状态非有限"
                elif abs(self.skew_ms) > 100:
                    self.state, self.error = "fault", "仿真与媒体时间偏差超过 100 ms"
            if self.log and self.sim_time - self.last_log_time >= 0.019999:
                row = dict(self.status(), qpos=qpos.tolist(), qvel=env.mj_data.qvel.tolist())
                self.log.write(json.dumps(row, ensure_ascii=False) + "\n")
                self.last_log_time = self.sim_time
                if self.state == "fault":
                    self.log.flush()

    def close(self):
        """只关闭本实例的线程和日志。"""
        self.running = False
        self.thread.join(timeout=2)
        if self.log:
            self.log.close()
