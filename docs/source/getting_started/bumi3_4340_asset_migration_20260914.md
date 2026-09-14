# 2026-09-14 BUMI3 SONIC 切换 4340 资产记录

本次按用户明确要求，SONIC 的训练 URDF、MuJoCo XML 和参考运动学 XML 改用
legged_lab 的 bumi3_4340。用户再次确认左右肩 roll 的新限位同时写入 XML 与 URDF。
已运行的 noetix-volc 100000→200000 续训保持原进程和旧资产；本次修改的是新进程默认配置。

## 来源、修改与副本

源目录：`/home/weili/legged_lab/source/NoetixRobot/NoetixRobot/assets/robots/bumi3_4340/`。
legged_lab 分支 feature/amp、起始 HEAD `df265177f4c9a4cd2d9dbc31d305d45f551a6cd8`；
用户 E1 等其他未提交修改保留。SONIC 分支 feature/bumi-native-sonic-full-training，
起始 HEAD `22ecc509e146f6e87bbf0b8e65955e42e8db7db6`。

| 文件 | 修改前 SHA256 | 修改后 SHA256 |
|---|---|---|
| 源 MJCF | `94ac99adf5f4512ac11903f521d5ec2f2fddb0413cfccec3e31a73d852a37719` | `4bc25c1d1ce3650f1c3c7c46bc31fe4fce54672f45e47d74bc773120da0d5ddb` |
| 源 URDF | `98da93ccb02c4955b04cb1eba5e0ae64d48ab40c8188123d2fa7b72c1e33cc90` | `e9bff4e0d81f9d876ef5f636e6ff3cc214e79b9d701c4b06b104d6aa61a4d1c6` |
| SONIC 新 MJCF | 新增 | `f4e11d57715fe6dc5ce867130425b823c38ff3940de12fcd60c9fbf118913aec` |
| SONIC 新 URDF | 新增 | `098f46181b75db4d965cdafc5901f4d78eeedfa4a63a2af2eab2a3ef88dbaf0d` |

SONIC 新文件为 `gear_sonic/data/assets/robot_description/mjcf/bumi3_4340.xml` 和
`gear_sonic/data/assets/robot_description/urdf/bumi3_4340/bumi3_4340.urdf`。
副本只调整相对网格路径并清除一处行尾空白，以便服务器独立运行和通过 Git 空白检查。22 个同名 STL 全部逐文件 SHA256 相同，
复用已跟踪的 `meshes/bumi3/`。源和副本 27 项 MuJoCo 编译数组完全相同，含质量、惯量、
连杆/关节变换、碰撞形状/掩码/摩擦/求解属性、阻尼/armature 和执行器参数。
URDF 归一化 mesh 路径并忽略排版空白后 XML 树完全相同。旧 SONIC XML/URDF 保留原字节。

## 实际参数与边界

| 项目 | 当前值 |
|---|---|
| 左肩 roll | `[-0.14, 1.94]` rad，XML/URDF 同步 |
| 右肩 roll | `[-1.94, 0.14]` rad，XML/URDF 同步 |
| XML 全部接触 geom + 地面 | `condim=6`、`friction="1 0.05 0.01"` |
| XML 脚底/地面 | `solref="0.01 1"`、`solimp="0.9 0.95 0.001"` |
| XML 求解器 | Newton，iterations=80，elliptic，impratio=10 |
| XML 关节被动参数 | damping=0.001、armature=0.03、frictionloss=0.1 |
| SONIC MuJoCo 运行时覆盖 | 21 驱动关节 armature=0.01，浮动根=0 |
| PD/动作缩放/默认角/限矩 | 从 ONNX 的 sonic_bumi3_control 元数据读取 |
| 物理/策略 | 默认 200/50 Hz；physics-substeps=5 时 1000/50 Hz |
| MuJoCo 刚体质量合计 | 20.7073094 kg |

摩擦和 condim 是 MJCF 接触设置，不会直接把三项摩擦或 condim 写入 URDF/PhysX。
训练的名义执行器和环境随机化保留 SONIC 原配置，未换成 Mimic 的 PD。
XML 原生 24 个机器人 geom（22 可视、18 接触，共用其中部分）；不再使用旧 SONIC
的 5 capsule + 9 mesh 近似碰撞。源 URDF 有 19 个碰撞 mesh，不包括 base 和双膝；
源 XML 有 18 个，不包括双膝和双踝 pitch。这些源模型差别仍存在；新配置不等于
Isaac Lab 与 MuJoCo 完全相同，也不等于旧策略重新完成了 4340 训练。

## 代码修改及理由

- `robots/bumi3.py` 指向新 URDF；部署 YAML 和实验 MotionLib 指向新 XML。
- 新增 `utils/bumi3_assets.py`：在训练配置快照前及 `create_manager_env` 前迁移已知旧
  MJCF 路径，避免旧 checkpoint 的物理资产与参考 FK 混用；原 checkpoint 配置不写回。
  G1/H2 及旧 G1 默认行为保持不变，未知自定义 BUMI3 路径报错。
- MuJoCo 运行器按 ankle_roll 刚体的唯一触地 mesh 定位足底；按可见 mesh 复制影子。
  适配 4340 geom 名和 group=0，保留原模型所有碰撞属性及惯量。
- 数据准备工具当前目标切到 4340，旧 SONIC 精确数据来源指纹仍可读取；21 个关节
  名称、顺序、轴和新限位与旧 SONIC 一致。数据兼容不作动力学兼容保证。
- 两个验证器锁定新来源的指纹、真实碰撞布局、接触参数与部署契约；补充旧配置迁移、
  G1/H2 兼容、质量和肩限位回归。角速度测试改为编译前构造惯性主轴旋转的专用 XML，
  保留坐标系错误的检出能力，避免依赖旧资产根惯性主轴不对齐。
- `.gitignore` 放行新增代码资产；修改记录与当前使用指南同步更新。

## 验证结果

- 源/副本一致性、XML/URDF 21 关节/22 刚体、网格和 Hydra 解析检查通过。
- 88 项 pytest 通过，11 条既有 tracer/弃用提示；数据来源白名单的后续定点修改另做
  对应工具回归 12 项通过，完整汇总写入根修改记录。
- Isaac Lab 实际单环境 `reset + 10 步` 零动作通过；日志打印新 4340 skeleton 路径和
  21 DoF/22 bodies；19 条本机 CPU core 拓扑探测提示之后环境正常运行。没有执行 PPO。
- MuJoCo 静态初始化无自接触、无地面穿透，显示 22 个参考 mesh。零策略仅验证有限值，
  会摔倒，不能将该检查解释为控制质量通过。
- 真实旧资产训练的 100000 Robot ONNX（`*_g1_control.onnx`，SHA256
  `af600733d013d8925f5083806c26cfba60c121ed6a83d34c371478bb986d95fd`）在新 XML 上
  做 40 组回放：十条参考 × 200/1000 Hz × 30 s 首帧保持/完整播放。全部完成、无 NaN/Inf、
  MuJoCo warning=0。本轮诊断倾覆阈值为最大根倾角>65° 或根高<0.15 m，40 组均未触发。
  完整播放每频率 9067 帧；未做 GUI 观感、SMPL 全轨迹、实机或新资产训练质量验收。
- 本轮没有统计承重接触点切向速度，因此不报告脚滑已消失。完整播放的根位移含目标运动，
  不能把该值当作漂移；首帧保持的位移才用于本轮静态漂移对照。

| 物理频率 | 模式 | 组数 | 最大根倾角 | 最低根高 | 首帧保持最大结束净位移 |
|---|---|---:|---:|---:|---:|
| 200 Hz | hold30 | 10 | 7.4151° | 0.464029 m | 5.8033 cm |
| 200 Hz | full | 10 | 32.2997° | 0.243388 m | 不作漂移指标 |
| 1000 Hz | hold30 | 10 | 7.3538° | 0.464418 m | 3.3981 cm |
| 1000 Hz | full | 10 | 33.5899° | 0.240435 m | 不作漂移指标 |

逐轨迹结果（全部无错误、无 warning、未触发上述诊断阈值）：

| Hz | 模式 | 轨迹 | 步数 | 最大倾角 ° | 最低根高 m | 结束净位移 m |
|---:|---|---|---:|---:|---:|---:|
| 200 | hold30 | Idle_Right_001__A018 | 1500 | 3.19671 | 0.464029 | 0.019653 |
| 200 | hold30 | walk_forward_amateur_003__A001 | 1500 | 5.16307 | 0.468673 | 0.017651 |
| 200 | hold30 | walk_backward_loop_001__A021 | 1500 | 3.03201 | 0.470681 | 0.058033 |
| 200 | hold30 | wave_R_001__A431 | 1500 | 5.41784 | 0.471638 | 0.024315 |
| 200 | hold30 | squat_003__A361 | 1500 | 2.69259 | 0.471007 | 0.021164 |
| 200 | hold30 | Jump_002__A018 | 1500 | 5.90980 | 0.468111 | 0.022650 |
| 200 | hold30 | dance_basic_chaines_180_R_fast_001__A309 | 1500 | 7.41511 | 0.468365 | 0.043144 |
| 200 | hold30 | Neutral_kick_trash_004__A057 | 1500 | 4.54653 | 0.465782 | 0.025970 |
| 200 | hold30 | pels_air_punch_001__A493 | 1500 | 2.61259 | 0.468558 | 0.026509 |
| 200 | hold30 | run_start_180_R_001__A327 | 1500 | 5.31365 | 0.466742 | 0.035205 |
| 200 | full | Idle_Right_001__A018 | 2984 | 5.74747 | 0.428968 | 0.042532 |
| 200 | full | walk_forward_amateur_003__A001 | 1509 | 16.35640 | 0.420933 | 0.181430 |
| 200 | full | walk_backward_loop_001__A021 | 789 | 11.65718 | 0.439903 | 4.899231 |
| 200 | full | wave_R_001__A431 | 297 | 9.46290 | 0.456763 | 0.025636 |
| 200 | full | squat_003__A361 | 424 | 31.34088 | 0.243388 | 0.033282 |
| 200 | full | Jump_002__A018 | 1965 | 26.42520 | 0.348665 | 0.182755 |
| 200 | full | dance_basic_chaines_180_R_fast_001__A309 | 384 | 15.38723 | 0.405529 | 0.943813 |
| 200 | full | Neutral_kick_trash_004__A057 | 399 | 12.84403 | 0.415735 | 2.612048 |
| 200 | full | pels_air_punch_001__A493 | 179 | 21.48237 | 0.402956 | 0.119218 |
| 200 | full | run_start_180_R_001__A327 | 137 | 32.29970 | 0.399319 | 2.938722 |
| 1000 | hold30 | Idle_Right_001__A018 | 1500 | 3.34718 | 0.464418 | 0.016955 |
| 1000 | hold30 | walk_forward_amateur_003__A001 | 1500 | 5.22698 | 0.468499 | 0.017792 |
| 1000 | hold30 | walk_backward_loop_001__A021 | 1500 | 3.00602 | 0.470665 | 0.028994 |
| 1000 | hold30 | wave_R_001__A431 | 1500 | 5.54206 | 0.471595 | 0.011094 |
| 1000 | hold30 | squat_003__A361 | 1500 | 2.66625 | 0.471037 | 0.012164 |
| 1000 | hold30 | Jump_002__A018 | 1500 | 5.87297 | 0.468651 | 0.021468 |
| 1000 | hold30 | dance_basic_chaines_180_R_fast_001__A309 | 1500 | 7.35382 | 0.468313 | 0.019521 |
| 1000 | hold30 | Neutral_kick_trash_004__A057 | 1500 | 4.51198 | 0.465823 | 0.017136 |
| 1000 | hold30 | pels_air_punch_001__A493 | 1500 | 2.67050 | 0.468127 | 0.010503 |
| 1000 | hold30 | run_start_180_R_001__A327 | 1500 | 5.31540 | 0.467107 | 0.033981 |
| 1000 | full | Idle_Right_001__A018 | 2984 | 6.14817 | 0.431822 | 0.037165 |
| 1000 | full | walk_forward_amateur_003__A001 | 1509 | 16.02320 | 0.429204 | 0.130375 |
| 1000 | full | walk_backward_loop_001__A021 | 789 | 10.23453 | 0.439141 | 4.882358 |
| 1000 | full | wave_R_001__A431 | 297 | 8.16998 | 0.462359 | 0.012407 |
| 1000 | full | squat_003__A361 | 424 | 31.34467 | 0.240435 | 0.034661 |
| 1000 | full | Jump_002__A018 | 1965 | 26.74102 | 0.344999 | 0.234361 |
| 1000 | full | dance_basic_chaines_180_R_fast_001__A309 | 384 | 13.27087 | 0.404518 | 1.268117 |
| 1000 | full | Neutral_kick_trash_004__A057 | 399 | 12.87869 | 0.406964 | 2.692679 |
| 1000 | full | pels_air_punch_001__A493 | 179 | 21.50400 | 0.404784 | 0.100117 |
| 1000 | full | run_start_180_R_001__A327 | 137 | 33.58987 | 0.399942 | 2.872838 |

## 复现命令

```bash
cd /home/weili/GR00T-WholeBodyControl
BUMI_PY=/home/weili/miniconda3/envs/env_isaaclab/bin/python
BUMI_RUN="$PWD/models/sonic_bumi3/sonic_bumi3_uniform90_lr2e5_critic1e3_ee040_scratch_100k-20260909_141204"
BUMI_DATA="$PWD/data/noetix_bumi3_bigset_10pairs_20260910"

"$BUMI_PY" gear_sonic/scripts/run_bumi3_sim2sim.py \
  --encoder robot --policy "$BUMI_RUN/exported/model_step_100000_g1_control.onnx" \
  --dataset "$BUMI_DATA/dataset.json" --physics-substeps 5

"$BUMI_PY" -m gear_sonic.tools.validate_bumi3_sim2sim
"$BUMI_PY" -m gear_sonic.tools.validate_bumi3_integration \
  --smoke --motion-file "$BUMI_DATA/robot" --smpl-motion-file "$BUMI_DATA/smpl" \
  --num-envs 1 --iterations 10 --device cuda:0
"$BUMI_PY" -m pytest \
  gear_sonic/tests/test_bumi3_sim2sim.py gear_sonic/tests/test_bumi3_control_metadata.py \
  gear_sonic/tests/test_bumi3_smpl_sim2sim.py gear_sonic/tests/test_bumi3_motion_playlist.py \
  gear_sonic/tests/test_bumi3_assets.py gear_sonic/tools/test_build_bumi3_three_source_dataset.py \
  gear_sonic/tools/test_prepare_bumi3_sonic_dataset.py
```

回放方法：`load_motion_dataset(..., policy.contract)`；对 substeps=1/5 分别用
`Bumi3SonicSim2Sim(..., start_paused=True/False)`，保持模式执行 1500 次 `step_control()`，
播放模式执行该动作的 `num_frames` 次。每步读取根高及 base 的世界旋转矩阵 `R[2,2]`
计算 `acos` 倾角，累计 warning 及位移；不改动作或 ONNX。

临时验证产物在关键命令、失败和指标归档后，确认进程结束且无进程引用，再按精确路径
清理。旧模型、原始数据、正式训练日志和用户文件保留。回滚通过新的修复提交恢复旧默认
资产路径与旧验证契约，禁止 reset/clean/force push 或覆盖用户未提交工作。
