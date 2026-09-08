# SONIC G1 MuJoCo 音乐会话输入

`g1_deploy_onnx_ref --input-type music` 为 GENMO 文件音乐部署提供连续可靠参考。当前配置固定为 release `model_encoder.onnx` 的 1762→64 和 `model_decoder.onnx` 的 994→29，使用同目录 `observation_config.yaml`，强制 `--encoder-mode 2`。本入口只接受 `lo` 仿真和 `--disable-crc-check`。

## 使用方式

推荐在 GENMO 运行统一入口：

```bash
cd /home/weili/GENMO
.venv/bin/python -B scripts/demo/demo_music_sonic.py --audio /path/song.wav --launch-local
```

如果需要手动保持三个进程，先在 SONIC 根目录启动仿真：

```bash
.venv_sim/bin/python -B gear_sonic/scripts/run_sim_loop.py \
  --music-endpoint tcp://127.0.0.1:5561 \
  --music-log-path /absolute/new/sim_state.jsonl
```

然后在 `gear_sonic_deploy` 中启动控制器：

```bash
target/release/g1_deploy_onnx_ref lo \
  policy/release/model_decoder.onnx reference/example \
  --encoder-file policy/release/model_encoder.onnx \
  --obs-config policy/release/observation_config.yaml \
  --input-type music --encoder-mode 2 --disable-crc-check \
  --music-endpoint tcp://127.0.0.1:5560 \
  --logs-dir /absolute/new/sonic_logs
```

最后运行 GENMO 入口并省略 `--launch-local`。PD 初始化、准备参考、朝向校准、弹力带释放和起舞全部由会话协调，不需要 Enter/T/R。默认自然结束后保持站立；播放中 Ctrl+C 经统一入口淡出和收尾。故障冻结仿真，重新创建会话，不自动重播、追帧或回退到其他编码器模式。

## 本机请求协议

控制器 REP 默认 `tcp://127.0.0.1:5560`，MuJoCo REP 默认 `tcp://127.0.0.1:5561`。请求第一段为 JSON，含 `op`、`session_id`、非负整数 `seq`；动作请求第二段为既有 Protocol v3 的 `pose` 前缀、1280 字节 JSON 头和小端数组。应答为 JSON，失败具有 `ok=false` 和 `error`。

| 操作 | 控制器参数和语义 |
|---|---|
| `prepare` | 新 session_id、audio_frames、audio_start_frame=100；清除旧缓冲；活动会话不可覆盖 |
| `append` | start_frame、end_frame 和动作附件；必须紧接已接收末帧 |
| `start` | epoch_ns，未来 0.1～10 秒；需已预热且收到足够连续参考 |
| `status` | 实际 used_frame、received_frame、buffer_seconds、mode=2、控制时间戳和耗时；匿名查询不续心跳 |
| `finish` | 必须含全部音乐、50 帧收尾、10 帧站立保护；标记参考完整 |
| `stop` | fault=true 时锁存故障；普通立即停止可冻结会话 |
| `stop` + graceful=true | start_frame 至少领先消费帧 10 帧，并附 60 帧收尾；仅显式用户停止可以替换未消费未来段 |

MuJoCo 支持 `prepare/start/status/stop/finish`，不接受动作附件。两端保留最近 16 个请求的精确应答，客户端一次只允许一个未完成请求，超时重试使用相同编号和内容；更旧请求明确拒绝。缺帧、重叠、错误会话、错误类型、非有限数和非单位四元数均拒绝。C++ release 启用 fast-math，因此浮点有效性按 IEEE754 位模式检查。

| 字段 | 形状 | 类型/约定 |
|---|---|---|
| smpl_pose | N×21×3 | f32，轴角 |
| smpl_joints | N×24×3 | f32，SONIC 根局部人体 FK，含附加手部点 |
| body_quat | N×4 | f32，wxyz，单位四元数 |
| joint_pos | N×29 | f32，IsaacLab 顺序，仅六腕有效 |
| joint_vel | N×29 | f32，首版为零，SMPL mode 2 不消费 |
| frame_index | N | i64，会话 50 Hz 连续帧号 |

当前参考窗口为当前至未来第 9 帧，跨度 180 ms。网络线程验证并发布不可变快照；新的块只扩展未来，不改变消费位置。未来最多缓存 750 帧，另保留当前帧与十帧历史。低于 25 帧且尚未完整收尾时故障；心跳超时 1.5 秒同样锁存故障。播放中禁用原有跌倒自动 reset。

`status` 带 `audit=true` 返回实际编码器输入及对应帧、机器人四元数和校准 heading；`metrics=true` 返回本次起播后所有实际控制周期的计算耗时 P99 和墙钟频率。普通状态采样中的 compute_ms 是最近周期，不能用它代替完整周期 P99。

## 编译与测试

在 SONIC 根目录：

```bash
cmake -S gear_sonic_deploy -B gear_sonic_deploy/build
cmake --build gear_sonic_deploy/build --target g1_deploy_onnx_ref run_tests -j4
gear_sonic_deploy/target/release/run_tests --gtest_filter='MusicSession.*'
.venv_sim/bin/python -B -m pytest -q -p no:cacheprovider gear_sonic/tests/test_music_sim_session.py
```

MuJoCo reset 后紧接 `mj_forward`，避免读取尚未刷新的零四元数。仿真按 5 ms 固定步长和绝对墙钟期限运行；SONIC 复用本地 Unitree SDK 的 CLOCK_MONOTONIC 周期 timerfd。主线程的小数 `sleep(0.02)` 已改成 `sleep_for(20ms)`，消除整型截断形成的忙循环。

完整固定音频、模型指纹、三进程回归和故障证据位于 GENMO 的 `outputs/sonic_music/`；详细使用及模型构建参见 GENMO `docs/sonic_music_deployment.md`。本次只验证 G1 MuJoCo。用户提供的 `g1.tar.gz` 和其他既有工作保持原样，未启动训练。
