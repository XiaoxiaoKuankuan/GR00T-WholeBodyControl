"""MuJoCo 音乐会话的时间、故障冻结和重启隔离测试。

使用轻量仿真状态替身，不启动控制策略、DDS 或动力学；通过显式单调时钟验证
预约起播才释放弹力带，跌倒/时钟偏差锁存故障，只有新会话能清除旧的播放状态。
真实物理推进、实时因子与音乐同步另外由三进程集成会话记录验证。
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import zmq

from gear_sonic.utils.mujoco_sim.music_session import MusicSimSession


@pytest.fixture
def session():
    with zmq.Context() as context:
        with context.socket(zmq.REP) as socket:
            port = socket.bind_to_random_port("tcp://127.0.0.1")
    service = MusicSimSession(f"tcp://127.0.0.1:{port}")
    yield service
    service.close()


def fake_env():
    env = SimpleNamespace(
        mj_data=SimpleNamespace(time=0.0, qpos=np.array([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]), qvel=np.zeros(6)),
        elastic_band=SimpleNamespace(enable=True),
    )
    env.reset = lambda: None
    return env


def test_scheduled_release_and_clock_fault(session):
    env = fake_env()
    session.request(dict(op="prepare", session_id="a", seq=0))
    with patch("time.monotonic_ns", return_value=1_000_000_000):
        assert session.before_step(env)
        session.request(dict(op="start", session_id="a", seq=1, epoch_ns=1_200_000_000))
        assert session.before_step(env)
        assert env.elastic_band.enable
    with patch("time.monotonic_ns", return_value=1_200_000_000):
        assert session.before_step(env)
        assert not env.elastic_band.enable
    with patch("time.monotonic_ns", return_value=1_350_000_000):
        env.mj_data.time = 0.005
        session.after_step(env)
        assert session.state == "fault"
        assert not session.before_step(env)


def test_fall_and_new_session_do_not_reuse_epoch(session):
    env = fake_env()
    session.request(dict(op="prepare", session_id="a", seq=0))
    session.before_step(env)
    session.state = "playing"
    session.epoch_ns, session.sim_origin = 1, 0.0
    env.mj_data.qpos[2] = 0.1
    session.after_step(env)
    assert session.state == "fault" and "跌倒" in session.error
    assert not session.before_step(env)
    session.request(dict(op="prepare", session_id="b", seq=0))
    assert session.state == "prepared" and session.epoch_ns == 0
    assert session.reset_pending
    with pytest.raises(ValueError):
        session.request(dict(op="start", session_id="a", seq=1, epoch_ns=1))
