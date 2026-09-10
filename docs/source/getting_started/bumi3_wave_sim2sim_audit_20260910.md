# BUMI3 wave 首帧摔倒诊断与修复（2026-09-10）

**当前版本说明：** 第 1～9 节保留提交 `9ff1e1a` 的历史定位和验证，当时手臂
运行时 armature 为 0。之后用户要求采用 G1 式显式 PD、手臂 armature=0.03、
XML 被动关节阻尼 0.05，对应验收见第 10 节。随后用户要求手臂惯量降为 0.01，
当前参数与本轮验收记于第 13 节；第 10～12 节保留各自试验时的参数和结果。
历史实验不表示所有参数下的显式 PD 都不稳定，也不代表当前仍使用原生位置伺服。

用户现场为 `wave_R_001__A428.pkl` 尚未按 T 就摔倒，Isaac Lab 的 Robot/SMPL
两个分支正常。本次在本地复现了该故障，确认主要原因是 MuJoCo 部署端显式 PD
阻尼的数值不稳定，已经修复。使用同一 30000 轮模型和同一动作，修复后可以稳定等待、
播放完整动作并继续保持。没有为解决此次故障更换模型、调整训练奖励或改动碰撞资产。

## 1. 固定的模型、动作和环境

- 仓库：`/home/weili/GR00T-WholeBodyControl`。
- 分支：`feature/bumi-native-sonic-full-training`，诊断起始 HEAD：
  `21a3b5fb4c0892a1a2ce92f4d1291f2fb06b00f8`。
- 模型目录：
  `models/sonic_bumi3/sonic_bumi3_uniform90_lr2e5_critic1e3_ee040_scratch_100k-20260909_141204`。
- 动作：`data/noetix_bumi3_5pairs_20260910/robot/wave_R_001__A428.pkl`，219 帧、
  50Hz、21 个关节；配对 SMPL 在同级 `smpl/` 目录。
- Python：`/home/weili/miniconda3/envs/env_isaaclab/bin/python`；MuJoCo 3.3.2，
  ONNX Runtime 1.27.0，SciPy 1.15.3；Isaac Lab 位于 `/home/weili/IsaacLab`。
- 正式控制参数保持 `sim_dt=0.005`、`decimation=4`，即物理 200Hz、策略 50Hz。

| 对象 | SHA-256（本次修复未改动） |
| --- | --- |
| `model_step_030000.pt` | `281dac49d29a2f816ecb71538bbc1ecf15190399d4dd8992c97ec8c28a59e495` |
| `exported/model_step_030000_g1.onnx` | `e884db48c3d5d222821c0816f9c4c81fbcd8e3b9e9129cefcd507ab463466a9a` |
| `exported/model_step_030000_smpl.onnx` | `6f7e75978cff5e90e117036457d42c50e77061e37c4f6b4c033c92c71da85cf7` |
| `robot/wave_R_001__A428.pkl` | `c2e1d7d944918249d1407d753673bf7a93a6a63649e04ea6afa541663733fcc2` |
| `gear_sonic/data/assets/robot_description/mjcf/bumi3.xml` | `28d55b3b460c2731ba478c083c780948b5175132cd3b7b1a73e8d6cbe6fd6547` |
| `gear_sonic/data/assets/robot_description/urdf/bumi3/bumi.urdf` | `0e08c15fe2226fedeac967c06a7910701935fc6de8fca2d4664a76c9ac41e955` |
| `gear_sonic/config/sim2sim/bumi3_sonic.yaml` | `843756aef9332faa81f0f5ea71e95869e1c0d4604b0f4a11c64974773ead0d17` |

## 2. 为什么没按 T 也会摔倒

等待 T 时，程序固定的是参考动作，真实机器人仍由网络输出动作、由物理引擎执行。
原实现先在 Python 中按当前角度和速度计算 `Kp * (目标角度 - 当前角度) - Kd * 当前速度`，
再把结果当作固定力矩交给 MuJoCo 积分 5ms。对于本资产低惯量的手臂关节，阻尼力矩
在一个物理步内可能纠正过头，导致速度反向，再在下一步被反向纠正，形成放大的振荡。
因此，即使目标姿态不动，实际机器人也会被控制计算弄得抖动并摔倒。

Lab 的 `ImplicitActuator` 由物理求解器处理关节驱动；原来的 MuJoCo 外部力矩实现
没有提供同样的阻尼求解方式。只改积分器名称仍不能让求解器识别已经算完的 motor
力矩中包含的速度反馈，需要把 PD 配为 MuJoCo 原生位置执行器。
[MuJoCo 3.3.2 位置执行器文档](https://mujoco.readthedocs.io/en/3.3.2/XMLreference.html#actuator-position)
也建议有速度反馈时配合 `implicitfast` 或 `implicit`。

固定模型、动作及其它参数，每组运行 5s；摔倒判据为根高度低于 0.22m 或根上轴与
世界竖直方向夹角超过 60°。以下为定位阶段的单变量对照，不混入后续参考速度修复：

| 定位方案 | 首帧等待 | 自动播放 | 说明 |
| --- | --- | --- | --- |
| 原显式 PD，5ms | 1.16s 摔倒 | 1.38s 摔倒 | 原故障复现；5s 内关节速度峰值 57.83/59.57rad/s |
| 仅将机器人抬高 18mm | 1.38s 摔倒 | 1.28s 摔倒 | 去掉初始脚底穿地仍失败 |
| 仅设 implicitfast，仍外部显式 PD | 1.16s 摔倒 | 1.38s 摔倒 | 求解器仍看不到外部 PD 的速度导数 |
| 显式 PD，物理改为 1ms，策略仍 50Hz | 5s 未倒 | 5s 未倒 | 减小物理步长也消除了本次不稳定 |
| 原生位置 PD + implicitfast，仍 5ms | 5s 未倒 | 5s 未倒 | 采用此修复，保留原控制频率与参数 |
| 原生位置 PD 再抬高 18mm | 5s 未倒 | 5s 未倒 | 额外对照；正式代码不抬高机器人 |

又做了不使用神经网络的隔离实验：关闭重力、机器人离地、无接触、固定目标角度，
只给左 arm-yaw 初速度 1rad/s。原实现 0.5s 内关节速度峰值达到 **54.1699rad/s**；
修复版按控制周期采样的峰值为 **0.17419rad/s**，0.5s 后降至 **0.0031986rad/s**。
没有网络和接触仍能复现失稳，直接支持 PD 数值实现是主因。

## 3. 实际修改

1. `gear_sonic/utils/mujoco_sim/bumi3_sim2sim.py`：将运行时执行器配为原生位置 PD，
   保留 Kp、Kd、关节惯量、armature、动作缩放和力矩上限。Python 的 `ctrl` 现在是
   目标角度，输出力矩限制放到 `forcerange`，积分器用 `implicitfast`。reset 初始化
   ctrl 为当前关节角度，力矩统计读取实际 `actuator_force`，避免角度/力矩单位混淆。
2. 同文件：每个控制周期末执行 `mj_forward`，刷新根姿态和速度等派生数据，避免
   下一次推理把最新关节状态与滞后约一个 5ms 物理步的根状态混用。
3. 同文件：PKL 缺省关节速度从中心差分改为训练 MotionLib 使用的前向差分，并匹配
   训练末帧复用倒数第二段速度的约定。修复前 wave 参考关节速度最大差为
   **12.31266785rad/s**。两帧短动作复用唯一速度段；显式速度、NPZ/CSV 契约保留。
4. 同文件：PKL 根线/角速度采用训练中心差分与 `sigma=2` 高斯滤波，自动播放/reset
   时使用该初速度，世界角速度转为 MuJoCo freejoint 局部角速度。等待阶段仍置零。
   与训练实现对照，角速度最大差 **3.94432654e-6rad/s**，线速度最大差
   **1.38237764e-8m/s**。
5. `gear_sonic/scripts/run_bumi3_sim2sim.py`：启动时打印 PD 实现与积分器，方便辨认
   是否加载了修复版本。
6. `gear_sonic/tests/test_bumi3_sim2sim.py`：增加 PD 力矩及饱和、无接触阻尼衰减、
   积分后根状态刷新、PKL 差分和根速度滤波/坐标转换回归。
7. `gear_sonic/tools/bumi3_lab_audit.py`：新增可选的单环境有限步采集回调，记录真实
   策略输入、动作、状态、运行时物理参数与失败/超时标记，用于可复核的部署对照。
   通过已有回调接口加载，不更改训练或评估入口行为。

第 3、4 项属于播放/初始化契约差异；首帧等待的参考速度本来为零，它们不是按 T 前
摔倒的主要解释。所有修复均不改写动作 PKL、模型权重或 XML/URDF。

## 4. 确认模型导出和观测排列

在 Lab 单环境真实运行 wave，采集每一步实际送给策略的输入，再将**同一输入**送到
原有 ONNX。这样比较不会混入两个仿真器各自演化导致的状态差异。

| 分支与输入 | 样本数 | 输入维数 | PT/ONNX 最大动作差 | 动作 RMSE |
| --- | --- | --- | --- | --- |
| Robot Encoder（g1），默认观测噪声 | 219 | 1170 | 7.62939453e-6 | 8.01893009e-7 |
| Robot Encoder（g1），关观测噪声 | 219 | 1170 | 7.62939453e-6 | 7.77117123e-7 |
| SMPL Encoder，默认观测噪声 | 219 | 1470 | 9.53674316e-6 | 8.02561431e-7 |

对无观测噪声的 Lab 状态，在 MuJoCo 中重建对应状态与历史，并使用该次 Lab 实际
随机化后的关节零位偏差：690 维本体观测最大差 **1.34110451e-7**，480 维参考观测
最大差 **1.78813934e-5**。实际本体输入顺序是 `base_ang_vel → joint_pos → joint_vel
→ actions → gravity_dir`；应核对运行时 ObservationManager，不能按 YAML 展示顺序判断。
没有发现 ONNX 导出错误、关节顺序错位或本体观测排列错误。

Lab 默认带观测噪声的 g1 完成 219 步，仅索引 218 的最后一步 done。另两次带完整
终止分类的采集结果为：g1 无噪声、SMPL 默认噪声均 `terminated=[]`、`timeout=[218]`，
完整动作期间没有失败终止。根高度范围分别为 0.460697～0.474412m、
0.464103～0.474376m。这里不把最后一帧动作自然结束算作摔倒。

## 5. 碰撞和资产还存在哪些差异

原始 wave 首帧双脚确实约有 **17.37/17.84mm** 的地面穿透，属于需要显式处理的
数据/地面基准问题。单独消除穿透仍会摔倒；不消除穿透、只修复 PD 就能稳定，
因此该穿透不是本次直接摔倒的主要原因。本次没有偷偷加 root Z 偏置、移动地面或
修改碰撞体；红色参考也没有跟随实际机器人降低高度。

将同一 Lab 关节姿态装入 MuJoCo 后，22 个刚体 FK 位置最大绝对差
**1.1818364e-5m**，四元数误差 **5.5834812e-7**。MuJoCo 总质量为
**20.24093728kg**，Lab 默认总质量为 **20.2409373671kg**，单刚体默认质量最大差
**1.4917755e-7kg**。没有发现整体尺寸、关节连接或默认质量错误。

仍保留以下物理差异，不能声称两个仿真器动力学完全相同：

- Lab 此次评估仍执行启动随机化：材质、腰部质量/重心、关节零位。无观测噪声
  对照只关了观测噪声，没有关闭这些随机化。本次腰质量相对默认值多
  **0.1335056859kg**，腰重心偏移为 **[-0.0089866261, 0.0421744308, 0.0180803732]m**；
  这是一次采样值，不是配置范围。关节零位范围 ±0.01rad，本次最大绝对偏差
  0.0099692922rad。MuJoCo 使用固定默认物理参数。
- Lab 开启自碰撞；当前 MuJoCo XML 保留机器人与地面接触，但关闭 link 间自碰撞。
- MuJoCo `frictionloss=0.1`，Lab joint friction coefficient 为 0；二者参数语义也不同。
- PhysX 与 MuJoCo 的接触、约束和积分实现仍不同。

这些差异可能影响其它姿态和精细跟踪质量，但不应替代已经由隔离实验确认的本次主因。
本轮没有全数据集评估或重新训练，结论仅覆盖指定模型和 wave 场景。

## 6. 全部修复后的长时验收

同一真实 ONNX、原始 wave、原有碰撞/增益/惯量、200Hz 物理/50Hz 策略。
等待和末帧保持期间物理与策略持续运行，没有逐步强写机器人 qpos。

| 场景 | 控制步数/时长 | 结果 | 最终根高度 | 最大根倾角 |
| --- | --- | --- | --- | --- |
| 不按 T，持续保持首帧 | 1500 / 30s | 未倒，仍为第 0 帧 | 0.47250969m | 5.202718° |
| 自动播放完整 wave，再保持末帧 | 1500 / 共 30s | 未倒，第 218 帧，停止播放 | 0.46921058m | 9.409625° |
| 等待 10s→发出 T→完整播放→保持 10s | 1219 / 24.38s | 未倒，第 218 帧，停止播放 | 0.46933021m | 9.135653° |

自动测试：

```bash
/home/weili/miniconda3/envs/env_isaaclab/bin/python -m pytest -q \
  gear_sonic/tests/test_bumi3_sim2sim.py \
  gear_sonic/tests/test_bumi3_motion_playlist.py -p no:cacheprovider
```

结果：**34 passed in 8.65s**。覆盖资产/输入契约、真实 MuJoCo 阻尼回归、首帧等待、
T/P 状态机与窗口回调。上述长时测试额外使用真实 checkpoint，不依赖零策略。
`validate_bumi3_sim2sim.py` 的零策略 100 步检查只证明接口和有限值，零策略实际根高度
会下降，不能把该工具的 PASS 当作已训练策略站稳。此前 1s GUI 和短时交互验证不足以
发现 1.16s 才触发的故障，本次补充了直接针对该故障的回归与长时测试。
本轮完整动作证据来自无窗口真实物理仿真，没有进行人工长时 GUI 视觉验收或实机测试。

## 7. 用户运行命令

退出之前的 sim2sim 窗口，重新运行以下命令即可；原 ONNX 可以直接继续使用。

```bash
cd /home/weili/GR00T-WholeBodyControl
conda activate env_isaaclab
BUMI_RUN="$PWD/models/sonic_bumi3/sonic_bumi3_uniform90_lr2e5_critic1e3_ee040_scratch_100k-20260909_141204"
BUMI_DATA="$PWD/data/noetix_bumi3_5pairs_20260910"

python gear_sonic/scripts/run_bumi3_sim2sim.py \
  --policy "$BUMI_RUN/exported/model_step_030000_g1.onnx" \
  --motion "$BUMI_DATA/robot/wave_R_001__A428.pkl"
```

先保持首帧，T 开始播放。原 `--dataset "$BUMI_DATA/dataset.json"` 的五条轨迹及
P 切换行为同样保留。启动日志应显示 `mujoco_native_position_servo` 和 `implicitfast`。

## 8. 复现 Lab 的有限步采集

使用上面的环境和变量，创建唯一诊断目录；回调拒绝覆盖已有采集文件。输出有
`lab_contract.json`、`lab_trace.npz` 和成功标记 `BUMI3_LAB_AUDIT`。需要同时检查
标记、数据文件与终止分类，不能只凭 Isaac 进程退出码 0 判定成功。

```bash
BUMI_AUDIT_DIR=$(mktemp -d /tmp/bumi-wave-lab-XXXXXXXX)
TMPDIR=/tmp PYTHONUNBUFFERED=1 python gear_sonic/eval_agent_trl.py \
  checkpoint="$BUMI_RUN/model_step_030000.pt" \
  ++headless=true ++num_envs=1 ++use_encoder=g1 \
  ++max_render_steps=230 \
  '++eval_callbacks=[bumi_audit]' \
  ++callbacks.bumi_audit._target_=gear_sonic.tools.bumi3_lab_audit.Bumi3LabAuditCallback \
  ++callbacks.bumi_audit.output_dir="$BUMI_AUDIT_DIR" \
  ++callbacks.bumi_audit.max_steps=219 \
  eval_base_dir="$BUMI_AUDIT_DIR/hydra" \
  ++output_dir="$BUMI_AUDIT_DIR/output" \
  ++algo.trl.output_dir="$BUMI_AUDIT_DIR/trl" \
  ++manager_env.config.save_rendering_dir="$BUMI_AUDIT_DIR/renderings" \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file="$BUMI_DATA/robot" \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="$BUMI_DATA/smpl" \
  '++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[wave_R_001__A428]' \
  ++manager_env.commands.motion.start_from_first_frame=true
```

将 `++use_encoder=g1` 改为 `++use_encoder=smpl` 复现 SMPL 分支。为了比较无噪声
的同状态观测，g1 另加 `++manager_env.observations.policy.enable_corruption=false`
和 `++manager_env.observations.tokenizer.enable_corruption=false`。

## 9. 修改记录与产物边界

本次详细执行记录、临时文件清理和 Git/服务器同步结果写在根目录
`BUMI3_SONIC_修改记录.md`。一次性诊断原始 trace/log 只用于本次检查，关键命令、
数值和失败原因归档后按仓库要求清理；正式模型与五对数据保留。回滚使用新的反向
提交，只撤销本次部署/诊断代码与说明，不改写已推送历史。

## 10. 按用户要求恢复显式 PD 后的当前配置与验收

用户要求控制方式和 G1 保持一致，并恢复 BUMI XML 中原有的手臂附加转动惯量。
G1 实际链路为 C++ 生成目标角度及 PD 参数，通过 DDS 发送；Python 的
`BaseSimulator.compute_body_torques` 计算力矩，`sim_step` 限幅后写入 motor。
其默认物理频率为 200Hz、策略频率为 50Hz，积分器为 Euler；500Hz 是 DDS
命令发布频率。现在 BUMI 的目标角度、外部 PD 力矩计算和积分方式与之相同。

| 参数或行为 | 本轮修改前 | 当前 |
| --- | --- | --- |
| 运行时执行器 | 原生位置 PD | XML motor，Python 显式 PD |
| 积分器 | implicitfast | Euler |
| 八个肩/肘关节的运行时 armature | 0 | 0.03，与 XML 原值相同 |
| 21 个电机关节的 XML damping | 0.001 | 0.05，与 G1 对应关节相同 |
| 手臂 Kp / Kd / 力矩上限 | 8 / 0.4 / 4Nm | 保持原值 |
| 物理步长 / 策略频率 | 5ms / 50Hz | 保持原值 |
| PKL 速度与训练对齐、派生状态刷新 | 已修复 | 保留 |

“被动关节阻尼”不是无电机关节：这里指每个有电机 hinge 关节自身的速度阻力。
PD 阻尼来自主动控制的 Kd，XML damping 则由物理引擎独立处理。BUMI 的八个手臂
关节具体为左右各 `arm_pitch`、`arm_roll`、`arm_yaw`、`elbow_pitch`。
XML 的 armature 原本就是 0.03，原运行器按 YAML 覆盖成零；本轮修正 YAML，
使 0.03 在实际运行中生效。其余关节 armature、模型权重、动作文件、碰撞几何和
根高度保持原值，训练侧的手臂惯量仍为 0，未修改正在进行的 Lab 训练。

原始 wave 首帧约 18mm 穿地仍保留；本轮没有发现上述指定场景继续摔倒，因此没有
为了通过测试再改碰撞、初始化或观测。历史隔离实验描述的是旧参数组合，当前结果
进一步说明可以采用显式 PD 配合用户指定的部署惯量完成这条动作。

| 同一 30000 轮 ONNX 与 wave 的场景 | 步数 / 时长 | 结果 | 最终根高度 | 最大根倾角 |
| --- | --- | --- | --- | --- |
| 首帧等待 | 1500 / 30s | 未倒，保持第 0 帧 | 0.47212930m | 5.305660° |
| 自动播放完整动作后保持 | 1500 / 共 30s | 未倒，保持第 218 帧 | 0.46838502m | 9.587809° |
| 等待 10s→T→完整播放→再保持 10s | 1219 / 24.38s | 未倒，保持第 218 帧 | 0.46906875m | 9.188391° |

测试仍采用根高 <0.22m 或根倾角 >60° 作为摔倒判据，全程持续推理与物理积分。
三组关节速度峰值分别为 4.735219、21.501643、5.998213rad/s；自动播放组会带入
参考初速度，不能与零初速度等待组直接等同解释。没有以人为冻结姿态维持站立。

使用原测试命令，本轮结果为 **34 passed in 8.18s**。力矩测试现验证 motor 的
ctrl/实际力矩与 PD 公式和限幅一致，额外确认 Euler、运行时 0.03 手臂惯量及
0.05 关节被动阻尼；无接触扰动衰减回归仍通过。启动日志新增实际关节阻尼和
armature 数组，避免只看 XML 而漏掉运行时覆盖。

当前 BUMI XML SHA-256：
`f7a7c25565f410a54b01f5a9ba94c0dc7535eb26a75a94a427504251c9eac546`；
当前 sim2sim YAML SHA-256：
`7cf5b6fb31855540bca50fb048fe64cf61f3b2c2a12c200422a0468552604863`。
G1 被动阻尼的来源为实际默认场景包含的
`gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml`，SHA-256 为
`58c82f77753db54f8a6ca0a8e020e142b015d8587db6cfc442f9a75f2bc444c6`。

第 7 节的模型/动作运行命令仍适用，只需关闭旧窗口后重新启动；当前启动日志应为
`pd_implementation=python_explicit_pd_motor`、`integrator=Euler`。

## 11. 当前显式 PD 配置下将 armature 清零的对照实验

2026-09-10，用户进一步询问将 BUMI sim2sim 的 armature 改为 0 是否会摔倒。
本次在 HEAD `056c46293bf8d326ba34f82196c11e49c268abbe` 上实际测试，正式
YAML、XML、运行器、ONNX 和动作文件均未修改。使用第 10 节同一 30000 轮模型、
`wave_R_001__A428.pkl`、MuJoCo 3.3.2 / ONNX Runtime 1.27.0 / CPU 推理。
保持显式 PD、Euler、5ms 物理步长、50Hz 策略和 0.05 被动阻尼，Kp/Kd、
力矩上限、碰撞和初始姿态也保持原值，仅在内存中替换 armature 数组。

对照方法：加载正式 `Bumi3Contract` 后，用 `dataclasses.replace` 设置
`armature_mujoco`；直接调用正式 `Bumi3SonicSim2Sim.step_control()`，无窗口、
不按墙钟限速。每组使用新建的仿真实例；等待组 `start_paused=True`，自动播放
组为 False，因此分别复现零初速度等待和带参考初速度播放。第三场景在第 500
个控制步调用 `enqueue_key(ord("T"))`，计划等待 10s、完整播放、再保持约 10s。
每组初始化时检查真实运行时的 armature、被动阻尼和积分器；每个控制步以
20ms 间隔记录根高度、倾角和关节速度。摔倒判据沿用根高 <0.22m 或根倾角 >60°。

| armature 设置 | 首帧等待 | 自动播放 | 等待 10s→T→播放→保持 |
| --- | --- | --- | --- |
| 当前：八个肩肘 0.03，踝 pitch 0.012574 / roll 0.009608，其余 0 | 30s 未倒 | 完整播放后保持至 30s，未倒 | 24.38s 未倒 |
| 仅八个肩肘改为 0，踝保持原值 | 3.82s 达到摔倒阈值 | 2.60s 达到摔倒阈值，第 130 帧 | 3.82s 倒，尚未触发 T |
| 全部 21 个关节改为 0 | 1.60s 达到摔倒阈值 | 0.98s 达到摔倒阈值，第 49 帧 | 1.60s 倒，尚未触发 T |

当前配置三组结果与第 10 节一致。仅肩肘清零时，等待组的手臂速度峰值为
24.475161rad/s（左 arm_yaw），当前 0.03 对照组为 2.778115rad/s；自动播放
清零组为 31.994544rad/s（右 arm_yaw）。这些是每 20ms 采样的峰值，不是物理
子步的完整峰值。全部清零组最大速度出现在左 ankle_roll，等待/播放分别为
40.700312 / 46.904142rad/s。九组测试均无 MuJoCo 数值警告或 NaN/Inf。

为排除短暂蹲低后恢复的情况，对两种清零配置的等待和自动播放各继续仿真到 10s：

| 配置 / 场景 | 首次倾角 >60° | 10s 时根高度 | 10s 时根倾角 |
| --- | --- | --- | --- |
| 仅肩肘清零 / 等待 | 3.94s | 0.050560m | 63.794836° |
| 仅肩肘清零 / 自动播放 | 2.70s | 0.076259m | 89.614416° |
| 全部清零 / 等待 | 1.70s | 0.052768m | 83.432963° |
| 全部清零 / 自动播放 | 1.02s | 0.052671m | 83.179767° |

因此本次“摔倒”不只是短时高度越界。实验表明，在当前完整部署参数组合下，
去掉肩肘附加惯量足以让该动作等待和播放失稳；继续去掉踝惯量则更早失稳。
这不单独证明 PD 公式错误或训练失败，也不代表所有模型、动作、步长下清零都
必然摔倒。本次未更改控制实现或正式配置，正式手臂 armature 继续为 0.03。

本次运行前后验证 YAML、XML、运行器、ONNX、wave 文件 SHA-256 一致，输出
`FORMAL_FILES_UNCHANGED=PASS`。XML/YAML 与第 10 节指纹一致，模型/动作与
第 1 节指纹一致；本轮只追加本节及根目录实验记录，没有修改训练或服务器任务。

## 12. 与用户指定 Mimic 部署对照：实际惯量与 5ms 反馈更新的影响

用户进一步指定 `/home/weili/legged_lab/scripts/sim2sim_mimic_vision_4340.py`，
询问为什么此前 Mimic 可以用零 armature。本次核对该脚本当前默认路径，不推断
用户历史命令通过 `--model` 指定其他资产时的值。

### 12.1 Mimic 当前默认部署并未使用零手臂 armature

实际加载链为 `sim2sim_mimic_vision_4340.py` 的 `--model` 默认值，经父类
`sim2sim_mimic.py:269` 调用 `MjModel.from_xml_path`，得到：

`/home/weili/legged_lab/source/NoetixRobot/NoetixRobot/assets/robots/bumi3_4340/mjcf/bumi3_4340.xml`

该 XML 第 5 行为 `<joint damping="0.001" armature="0.03" frictionloss="0.1" />`。
两个脚本没有向 `model.dof_armature` 写入其他值。实际用 MuJoCo 编译后检查，21 个
hinge 关节全部为 armature=0.03、被动 damping=0.001；根 freejoint 不继承这些值。
资产 SHA-256 为 `94ac99adf5f4512ac11903f521d5ec2f2fddb0413cfccec3e31a73d852a37719`。

Mimic 父类从 ONNX metadata 读取 `joint_stiffness`、`joint_damping`、
`default_joint_pos`、`action_scale` 并按关节名重排；它没有将训练 armature
恢复到 MuJoCo。实际读取默认 `bumi3_0806_lalacao_dance_jingjian_6.onnx`，
八个肩肘 Kp=8、Kd=0.4、action_scale=0.125，与 SONIC 相同。

| 项目 | 当前 Mimic 默认部署 | 当前 SONIC 部署 |
| --- | --- | --- |
| PD 方式 | Python 计算目标角度误差及速度反馈，写 motor 力矩 | 相同 |
| 物理 / 策略周期 | 5ms / 20ms | 5ms / 20ms |
| 积分器 | XML 默认 Euler | 明确设置 Euler |
| 肩肘 Kp / Kd / 力矩上限 | 8 / 0.4 / 4Nm | 8 / 0.4 / 4Nm |
| 肩肘 armature | 0.03，继承 XML | 0.03，由 YAML 显式覆盖 |
| 其余关节 armature | 全部 0.03，继承 XML | 四踝按训练值，其余 0 |
| XML 被动 damping | 0.001 | 0.05 |

两份资产的质量和碰撞配置也不同：Mimic 4340 编译总质量 20.7073094kg、25 个
geom；SONIC BUMI3 为 20.24093728kg、37 个 geom。本轮不互换资产，且 geom 总数
不等于参与碰撞的几何体数量。默认姿态下左肩 yaw 的关节空间惯量矩阵对角项却
几乎相同：无附加惯量时 SONIC 0.0003508188858、Mimic 0.0003508187656kg·m²。
该数值对应其他关节加速度为零时的该轴惯量，不是整条手臂质量，也不忽略连杆
原有惯量。增加 0.03 后 SONIC 此项为 0.0303508188858，约为清零值的 86.5 倍。

### 12.2 无网络、无接触的隔离实验

关闭重力及约束（接触、关节限位、干摩擦），目标设为默认关节角度，给左肩 yaw
注入 1rad/s；保持原 PD 增益和力矩限幅，每个物理步重算力矩。第一轮保留各组 PD，
第二轮仅启用八个肩肘 PD，以排除悬空腿部驱动耦合。两轮都复现零手臂惯量的
逐步反向放大；以下报告第二轮仅肩肘 PD 的结果，时间长度为 0.5s：

| SONIC 肩肘 armature / 物理步长 | 左肩 yaw 前三个物理步速度（rad/s） | 0.5s 末八臂最大速度 |
| --- | --- | --- |
| 0.03 / 5ms | +0.9264 → +0.8522 → +0.7778 | 0.005393rad/s |
| 0 / 5ms | -3.1053 → +10.8259 → -30.7411 | 24.775654rad/s |
| 0 / 1ms | -0.2950 → +0.0991 → -0.0221 | 0.000125rad/s |

用 Mimic 的实际资产和默认 ONNX 元数据做第一轮同类隔离：默认 armature=0.03
时初始速度逐步衰减；只清零肩肘时，在 5ms 下同样出现
`-5.6076 → +35.2877 → -31.8883rad/s` 的速度交替放大。该实验不运行 Mimic
网络，不是 Mimic 完整动作验收，但表明清零后的振荡不需要 SONIC 策略或地面碰撞。

机理是低惯量下角加速度对力矩很敏感；外部 PD 在步开始根据当前速度计算一次
力矩，并在该物理步内保持，5ms 足以让该小惯量轴的速度越过零点并反向放大。
下一步再反向纠正，产生高频振荡。力矩限幅只能限制力矩，不能保证速度振荡衰减。
这是参数与数值更新周期的组合问题，不是目标角度加 PD 这一公式本身错误。
MuJoCo 的 Euler 为半隐式积分，并隐式处理 XML 关节 damping；Python 已计算好
写入 motor 的力矩不会自动变成求解器内部的速度伺服。
参考 [MuJoCo 3.3.2 积分器说明](https://mujoco.readthedocs.io/en/3.3.2/computation/index.html#integrators)
和 [armature / damping 定义](https://mujoco.readthedocs.io/en/3.3.2/XMLreference.html#body-joint-armature)。

### 12.3 保留零手臂惯量和原 PD，只缩小步长的真实模型验证

随后恢复全部真实动力学、接触、重力和 SONIC 网络，保持第 11 节的零手臂惯量，
只在临时 contract 中将 `sim_dt=0.001`、`decimation=20`，使策略周期仍为 20ms。
与第 11 节共用同一 30000 轮 ONNX 和 wave；其他增益、力矩限制、被动阻尼、
踝惯量、碰撞、动作和初始姿态均保持原值。

| 1ms 物理步长、零手臂惯量的场景 | 结果 | 最终根高度 | 最大倾角 |
| --- | --- | --- | --- |
| 首帧等待 30s | 未倒，第 0 帧 | 0.4728370615m | 5.242422° |
| 完整播放后保持到 30s | 未倒，第 218 帧 | 0.4697831458m | 10.053520° |

因此不能把第 11 节解释为“armature 清零一定摔倒”。准确结论是：当前小惯量、
PD 增益与 5ms 更新间隔组合失稳；该动作可在保持原 PD 公式、零手臂惯量的情况下，
通过更细的物理/PD 更新间隔稳定运行。1ms 是本轮验证通过的试验点，未搜索临界
步长，也未做全数据集验收。正式 YAML 仍限制 5ms / decimation=4，本轮没有修改
这些校验和默认参数，更没有更改训练任务。

本轮诊断数据位于 `/tmp/bumi-mimic-armature-mechanism-20260910-ob9psbrj`。
仅追加本节和根目录实验日志，不修改 SONIC/Mimic 正式实现、XML 或模型。

## 13. 当前正式参数：肩肘 armature=0.01，保持 200Hz 显式 PD

用户要求将 0.03 降为 0.01。本次仅修改左右 arm_pitch、arm_roll、arm_yaw、
elbow_pitch 八个关节：YAML 的 arms 组改为 0.01，XML 的对应 joint 显式设为
0.01，确保原始资产和运行时覆盖一致。XML 其余关节默认值和运行器中的腿腰、
踝部惯量均保持原值，训练端手臂惯量也保持原值。

控制继续采用网络目标角度→Python PD 力矩→限幅→motor，Euler、5ms 物理步长、
decimation=4、策略 50Hz。手臂 Kp=8、Kd=0.4、力矩上限 4Nm，XML 被动阻尼
0.05 均保持不变。没有修改碰撞、质量、连杆惯量、初始姿态、ONNX 或参考动作。

同一 30000 轮 ONNX 与 `wave_R_001__A428.pkl` 的新参数验收结果：

| 场景 | 控制步数 / 仿真时长 | 结果 | 最终根高度 | 最大倾角 |
| --- | --- | --- | --- | --- |
| 首帧等待 | 1500 / 30s | 未倒，第 0 帧 | 0.4720431696m | 5.236921° |
| 完整播放后保持 | 1500 / 30s | 未倒，第 218 帧 | 0.4706900855m | 9.835809° |
| 等待 10s→T→完整播放→保持约 10s | 1219 / 24.38s | 未倒，第 218 帧 | 0.4695242554m | 9.554793° |

三组均无 MuJoCo 数值警告或 NaN/Inf。以控制步采样的关节速度峰值分别为
4.712578 / 15.063709 / 5.832744rad/s；自动播放带参考初速度，不能与等待场景
直接等同比较。摔倒阈值继续为根高 <0.22m 或倾角 >60°。

`test_bumi3_sim2sim.py` 和 `test_bumi3_motion_playlist.py` 共 34 项测试通过，
包含新惯量下的无网络扰动衰减与显式 PD 检查。两个资产检查器同步新 XML 指纹；
原 XML 去注释后的语义对照确认只有八个肩肘新增 armature=0.01，YAML 也只有
arms 组数值变化，运行器除说明文字外的 AST 完全一致。

当前 XML SHA-256：`1ef8da2e76be03430ba7f022e49309f194a289174db0275d7d3a123197cac3e3`。
当前 YAML SHA-256：`f52be29ca85a264273dc5ab75055ea52b8c297a362d093fd27f25ca90d9865f8`。
正式模型和动作指纹保持第 1 节记录。使用者退出旧窗口并重新执行原 sim2sim
命令即可加载新值，无需重导出模型。这里证明的是这条 wave 的上述场景稳定，
不将 0.01 视为已辨识的真实电机参数，也未完成整个动作集验收。
