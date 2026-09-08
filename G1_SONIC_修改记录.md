# G1 原生 SONIC 修改记录

本文档记录 G1 原生 SONIC 训练分支的实际代码修改、运行边界和验证证据。静态检查、
单元验证、Isaac Sim 运行和正式训练必须分别陈述，未执行的验证不得描述为已经通过。

## 2026-09-03：修复 TensorBoard 标量日志并改为从零训练

### 1. 修改背景与工作区保护

- 所属分支：`feature/g1-native-sonic-training`。
- 起始 HEAD：`570264710c4f0a122dd8b09eb8db836a9272501c`，修改前与
  `origin/feature/g1-native-sonic-training`、`origin/main` 完全一致。
- 修改前仅有用户已有的未跟踪文件 `g1.tar.gz`；本次不读取、不修改、不暂存该文件。
- 已固定并停止旧的官方 checkpoint 微调实验。保留的第 10000 轮模型为
  `/data/sonic_g1/runs/TRL_G1_Track/g1_sonic_bones_seed_8gpu_finetune-20260902_212036/model_step_010000.pt`，
  SHA256 为 `3b58d0342952a1b5c3a47f56f48730cdac1efe59c7a007f2f4bb1471e063bb61`。

### 2. 故障原因

- 原 `TRLPPOTrainer.log()` 虽然把零维 `torch.Tensor` 转成 Python 标量并保存到
  `state.log_history`，但仍将未经转换的原始 `logs` 传给 Hugging Face callback。
- 当前 `transformers==4.57.6` 的 TensorBoard callback 因类型不符合要求而丢弃了
  86 类环境指标，包括 reward 分项、termination、运动误差和 adaptive sampling；
  旧日志累计约 88 万条警告，TensorBoard 只保留 36 类 Trainer 原生指标。
- 该问题只影响日志写入，不改变 PPO、奖励、终止条件、自适应采样或模型参数更新。

### 3. 修改内容

- `gear_sonic/trl/trainer/ppo_trainer.py`：在统一日志边界生成 `sanitized_logs`，将单元素
  Tensor、NumPy 数组和 NumPy 标量转换为 Python 数字；Trainer 历史和 callback 共用
  同一份转换结果。若误传非标量 Tensor/数组则立即给出包含指标名和形状的明确异常，
  避免静默丢失或错误聚合训练指标。
- `agent.md`：补充 G1 分支的中文记录、提交同步、用户文件保护、验证产物清理和
  `noetix-12` 训练环境约束。
- `G1_SONIC_修改记录.md`：新增本记录文件。

### 4. 训练边界

- 新正式实验必须显式设置 `checkpoint=null`、`resume=false`、
  `auto_load_latest=false`，确保 Actor、Critic、optimizer、scheduler 和 trainer 计数均从零开始。
- 保持原始 `sonic_release` 的 G1、Teleop、SMPL 三编码器及原 PPO、奖励、终止、控制频率；
  仅通过命令行接入现有 Robot/SMPL 数据目录并指定独立实验目录。

### 5. 验证结果

- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m compileall -q
  gear_sonic/trl/trainer/ppo_trainer.py`：通过。
- 使用不创建持久测试文件的最小 mock 调用 `TRLPPOTrainer.log()`：零维 Tensor、单元素
  NumPy 数组、NumPy 标量和 Python 数字均转换为 Python 标量并传入 callback；非标量
  Tensor 按预期抛出包含指标名和形状的 `ValueError`，结果为
  `TENSORBOARD_LOG_SANITIZE_TEST=PASS`。
- `git diff --check`：通过。
- Isaac Sim reset/step 和短回合训练未在本地运行：本次只改变 Trainer 日志边界，不改变
  环境、观测、动作或动力学；正式服务器启动后将以首轮真实 event 中是否出现 reward、
  termination、运动误差和 adaptive sampling 标签作为运行时门禁。
- Git 提交 `e03a07b` 完成日志修复与记录文件，提交 `5bfd6fc` 修正 `agent.md` 文件尾部
  格式；二者已推送到 GitHub 的 `feature/g1-native-sonic-training`。`noetix-12` 工作区从
  `5702647` 通过 `git pull --ff-only` 快进到 `5bfd6fc`，拉取前后均为同名分支且工作区干净。
- `/root/miniconda3/envs/jump/bin/python` 在服务器执行同一最小日志回调测试：通过，输出
  `REMOTE_TENSORBOARD_LOG_SANITIZE_TEST=PASS`。
- 正式八卡训练于 `2026-09-03 11:04:40 CST` 启动，tmux 会话为
  `sonic_g1_scratch_8gpu_20260903_110440`，实验目录为
  `/data/sonic_g1/runs/TRL_G1_Track/g1_sonic_bones_seed_8gpu_scratch_100k-20260903_110440`。
  启动参数显式包含 `checkpoint=null`、`resume=false`、`auto_load_latest=false`、
  `use_wandb=false` 和 `algo.trl.report_to=tensorboard`；日志中没有
  `Loading checkpoint from`，实际初始化 G1、Teleop、SMPL 三个原生 Encoder。
- 正式训练首轮后，8 个 rank 均存活且 8 张 GPU 均进入计算；训练日志中 TensorBoard 类型
  警告为 0。实际 event 文件包含 122 个 scalar 标签，并确认
  `train/objective/rewards`、`train/Episode_Reward/tracking_anchor_pos`、
  `train/Episode_Termination/time_out`、`train/Metrics/motion/error_joint_pos` 和
  `train/adp_samp/failure_rate_mean` 均存在，结果为
  `TENSORBOARD_RUNTIME_TAG_GATE=PASS`。本次按用户要求没有启动 TensorBoard 服务进程。
- 正式训练不是短回合测试，因此保留其实验目录、日志、event 和后续 checkpoint；没有产生
  需要清理的临时训练目录或临时 tmux 会话。

### 6. 回滚方法

- 代码回滚只需撤销本次提交中 `TRLPPOTrainer.log()` 的日志转换改动；正式训练使用独立
  实验目录，不覆盖旧微调模型。不得通过 `git reset --hard` 或删除正式训练目录回滚。

## 2026-09-03：修正 sonic_release 的 C++ Encoder 部署观测契约

### 1. 问题现象与根因

- `model_step_010000_encoder.onnx` 的实际输入是 `1751` 维，但使用通用
  `policy/release/observation_config.yaml` 时，C++ 部署端组装了 `1762` 维。
- 多出的 `11` 维来自旧通用配置中的十帧根高度和单帧根高度；当前
  `sonic_release` checkpoint 的 Encoder 并未使用这两项。
- 原 `observation_config_sonic_release.yaml` 写的是 Python tokenizer term 名，而 C++
  运行时只能识别部署观测注册表名称，因此不能直接使用。

### 2. 修改内容

- 只修改 `gear_sonic_deploy/policy/release/observation_config_sonic_release.yaml`，不改动
  为其他历史模型保留的通用 `observation_config.yaml`。
- 用 `encoder_mode_4` 表示 1 维动态模式选择与 3 维 encoder index，并按
  Python ONNX 导出器的实际展平顺序排列 Robot、Teleop 和 SMPL 观测。
- 保留 G1/Teleop/SMPL 的模式编号 `0/1/2`，每个模式只计算它需要的观测；
  观测并集总维度严格为 `1751`。

### 3. 影响边界

- 本次修改只影响导出后的 G1 `sonic_release` C++ sim2sim/部署配置，不改变
  训练、checkpoint、ONNX 权重、MuJoCo 动力学或正在 noetix-12 运行的八卡训练。

### 4. 验证结果

- 直接读取 `model_step_010000_encoder.onnx` 的输入 shape，并按 C++ 观测注册表维度
  求和：配置和 ONNX 均为 `1751`，模式顺序为 `g1=0, teleop=1, smpl=2`，
  输出 `SONIC_RELEASE_DEPLOY_OBS_CONTRACT=PASS`。
- 使用本机已运行的 MuJoCo 进程启动一次短暂的 C++ 控制器初始化：13 条部署动作
  全部加载，Decoder `994 -> 29`、Encoder `1751 -> 64` 均通过运行时维度检查，
  成功进入 `Init Done`。验证后通过 `O` 正常退出控制器，未停止用户的 MuJoCo
  进程，未留下临时 tmux 或测试进程。
- 本次没有执行 `] -> 9 -> T` 的落地跟踪，因为验证目标是排除观测契约崩溃；
  实际动作效果由用户在可视化窗口中继续检查。
- `git diff --check`：通过。

### 5. 回滚方法

- 若需回滚，只撤销 `observation_config_sonic_release.yaml` 的本次更改；不应修改或删除
  通用 `observation_config.yaml`、ONNX/TRT 模型和训练 checkpoint。

## 2026-09-08：取回十对 G1/SMPL PKL 并打通双编码器 MuJoCo 部署验证

### 1. 任务范围与工作区保护

- 所属分支：`feature/g1-native-sonic-training`；起始 HEAD：
  `edc80bb06ad2c473607e363712652c03de712081`。
- 修改前工作区只有用户已有的未跟踪文件 `g1.tar.gz`；本次没有读取、修改、暂存或删除它。
- 本次只处理数据传输、离线格式转换和本机 MuJoCo 回环验证；没有启动、停止或修改
  noetix-12 上的训练任务，没有连接真实 G1，也没有做真机下发。
- 已核对 noetix-12 仓库为 `/root/home/liwei/GR00T-WholeBodyControl`，分支为
  `feature/g1-native-sonic-training`，检查时 HEAD 与本地起始 HEAD 同为 `edc80bb06ad2`。

### 2. 数据来源与十对样本

- Robot 源目录：
  `/data/datasets/bones-seed/sonic/motion_lib_bones_seed/robot_filtered`；单条文件是
  30 Hz、外层仅含一个同名运动键的 joblib 字典，核心字段为 `root_trans_offset`、
  `pose_aa(T,30,3)`、`dof(T,29)`、`root_rot(T,4)` 和 `fps`。
- SMPL 源目录：
  `/data/datasets/bones-seed/sonic/training_assets/data/smpl_filtered`；单条文件包含
  `pose_aa(T,72)`、`transl(T,3)`、`smpl_joints(T,24,3)` 和 50 Hz `fps`。
- 取回动作覆盖舞蹈、深蹲、弓步、挥手、行走、慢跑、跳跃、爬行和跪姿：
  `dance_in_da_party_001__A464`、`macarena_001__A545`、`squat_001__A359`、
  `forward_lunge_R_001__A359`、`wave_R_001__A430`、
  `walking_quip_180_R_002__A428`、`jog_ff_loop_360_R_normal_pace_001__A454`、
  `jump_and_land_light_001__A001`、`crawl_ff_loop_270_R_002__A232`、
  `kneeling_start_001__A036`。
- 原始数据保存在忽略目录
  `data/noetix12_g1_smpl_10pairs_20260908/raw/{g1,smpl}`，共 20 个 PKL、约 4.1 MiB；
  使用 rsync 经现有 noetix-12 SSH 隧道下载，远端与本地逐文件 SHA256 完全一致。
- 生成的部署数据保存在同目录的 `deploy/{g1,smpl}`，`manifest.json` 记录每个源 PKL、
  每个 CSV 和 `g1_29dof_rev_1_0.xml` 的 SHA256；这些数据产物受 `.gitignore` 保护，
  不进入代码提交。

### 3. 转换与部署代码修改

- 新增 `tools_local/convert_paired_g1_smpl_pkl_to_deploy.py`：
  - 使用训练 MotionLib 的 `Humanoid_Batch.fk_batch` 和同一 G1 29-DoF MJCF，将 Robot
    `pose_aa` 从 30 Hz 按训练公式插值到 50 Hz；
  - 将 29 自由度及完整刚体从 MuJoCo 顺序转换为 IsaacLab 顺序，再选取 `motion.yaml`
    的 14 个跟踪刚体；FK 输出的 xyzw 四元数转换为部署 reader 要求的 wxyz；
  - 对 SMPL 根执行 `axis-angle -> wxyz -> Y-up 到 Z-up -> remove_smpl_base_rot`，然后将
    原始 `smpl_joints` 乘以逐帧根四元数逆，严格复现训练 mode 2 的观测生成；
  - mode 2 保留配对 G1 的关节位置/速度，为 release encoder 的六个手腕关节条件提供数据；
    SMPL 根位置采用配对 G1 pelvis，根朝向与根角速度来自处理后的 SMPL；
  - 写盘前严格检查同名唯一配对、必需字段、维度、有限值、FPS、原始 `dof/root_rot`
    自洽性、插值后帧数和四元数范数；已有输出目录一律拒绝覆盖。
- 修改 `g1_deploy_onnx_ref.cpp`：新增 `--encoder-mode 0|1|2`，在本地 Encoder 成功加载后
  将显式模式设置到所有静态参考轨迹及 planner motion。默认仍为 mode 0，不改变历史行为；
  非整数或范围外值在模型加载前报错退出。
- 修改 `gear_sonic_deploy/deploy.sh`：增加同名参数的帮助、校验、配置回显和命令透传。
- 修改 `docs/source/references/motion_reference.md`：补充双 PKL 契约、转换命令、输出结构、
  mode 0/mode 2 启动方式，以及 SMPL 模式仍依赖配对 G1 手腕条件的边界说明。
- 生成但不提交 `data/noetix12_g1_smpl_10pairs_20260908/README_zh.md` 与
  `mujoco_validation.json`，保存本地操作说明和本轮定量验证证据。

### 4. 静态、格式与加载验证

- Python `py_compile`、`ruff check`、`ruff format --check`、
  `bash -n gear_sonic_deploy/deploy.sh`、`git diff --check`：通过。
- 实际转换 10 对成功，得到 G1 mode 0 与 SMPL mode 2 共 20 条部署轨迹。Robot 插值后
  帧数与同名 SMPL 帧数逐对完全相等，范围为 274 至 1375 帧，没有截断、补帧或静默对齐。
- 十对处理后 SMPL 根与配对 G1 根的平均四元数角差为 1.185° 至 5.093°，单帧最大值为
  12.317°；该结果支持当前 Y-up/Z-up 与 base rotation 方向，没有出现明显二次旋转。
- `DEPLOY_MANIFEST_HASH_AND_ROW_CHECK=PASS files=180 pairs=10`：manifest 中 180 个输出文件
  哈希全部复算一致，所有 CSV 数据行数都等于该运动声明的部署帧数。
- 临时 C++ reader 验证程序直接调用生产 `MotionDataReader`：G1 根下 10 条均为
  29 joints、14 bodies、14 body quaternions；SMPL 根下 10 条均为 29 joints、1 body、
  1 body quaternion、24 SMPL joints、21 SMPL poses；输出
  `CPP_MOTION_READER_g1=PASS motions=10` 与 `CPP_MOTION_READER_smpl=PASS motions=10`。
- `just build` 完整通过并重新链接 `target/release/g1_deploy_onnx_ref`；底层程序和
  `deploy.sh --help` 均显示新参数，`--encoder-mode 3` 按预期以退出码 1 拒绝。
- 转换器对已有输出目录的覆盖保护按预期以退出码 1 抛出 `FileExistsError`，且在加载
  G1 网格或写任何文件前结束。
- 格式化后的最终转换器用单对舞蹈源文件重新生成临时输出，G1 与 SMPL 两个运动目录均与
  正式输出逐文件一致，结果为 `FORMATTED_CONVERTER_REPRODUCIBILITY=PASS`。
- 仓库已有 `FK.TestFKAndGlobalVelocities` 测试被调用，但其固定输入目录
  `reference/bones_072925_test/` 在当前 checkout 不存在，因此该测试没有执行有效断言，
  不把进程退出码 0 记作本轮通过证据。

### 5. MuJoCo 同轨迹双模式运行证据

- 使用当前 release 模型：Decoder SHA256
  `c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed`，实际维度
  `994 -> 29`；Encoder SHA256
  `013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3`，实际维度
  `1762 -> 64`；观测配置为 `policy/release/observation_config.yaml`。
- 为保证公平性，mode 0 与 mode 2 分别从全新 MuJoCo 进程启动，均使用 `lo` 回环接口、
  200 Hz 仿真目标频率、50 Hz 控制目标频率、`--disable-crc-check`，未打开真机设备。
- 代表轨迹为 `dance_in_da_party_001__A464`，两侧均完整播放 497 帧。mode 0 日志中的
  Encoder 模式始终为 0，首末帧墙钟间隔 9.920000 s、实测 50.000000 Hz；mode 2 始终
  为 2，首末帧间隔 9.919991 s、实测 50.000045 Hz。两侧 action、token 和状态均为有限值，
  仿真电机错误码非零计数均为 0，控制器均通过 `O` 正常退出。
- 以日志中的实测关节转换回 IsaacLab 顺序，并与同帧配对 G1 参考比较：mode 0 直接
  joint RMSE 为 0.6480 rad、MAE 为 0.4466 rad；mode 2 分别为 0.6537 rad、0.4526 rad。
  mode 2 的直接 RMSE 高约 0.0057 rad，本单轨迹差异很小，但两个绝对误差都不低，不能
  据此宣称达到高质量精确跟踪。
- mode 0 的基座倾角 mean/p95/max/final 为 5.19°/10.20°/17.11°/3.19°；mode 2 为
  4.43°/8.44°/12.19°/5.47°。两侧均完成整段且末帧仍接近直立；这只能说明本次仿真
  没有在该指标上出现明显倒地，不能外推为十条轨迹整体质量或真机安全性。
- 本轮采用无窗口运行并记录定量日志，没有保存视频；十条动作逐条视觉质量、接触质量、
  脚滑和长时稳定性仍需在 MuJoCo 窗口中人工核验。未运行 Isaac Sim、训练或真机测试。

### 6. 临时产物、风险与回滚

- 两个临时 MuJoCo 进程和两个临时控制器均已正常停止；临时 CSV 日志、单动作软链接目录、
  第一版转换备份、临时 C++ reader 源码与二进制在证据汇总后已移入系统回收站。
- 保留用户需要的 20 个原始 PKL、20 条部署轨迹、manifest、中文说明和验证 JSON；没有
  删除服务器源数据、正式训练数据、checkpoint、ONNX 或历史 TensorRT engine。
- 风险边界：本轮只完整运行一条代表轨迹的两种模式；SMPL mode 2 的 G1 手腕条件来自
  配对 Robot 轨迹，这是当前 release encoder 的固有输入契约，不是纯 SMPL-only 控制。
- 代码回滚应只撤销转换工具、`--encoder-mode` 参数、启动脚本透传和文档修改；本地数据
  位于独立忽略目录，可保留审计。不得使用 `git reset --hard`，也不得删除用户的
  `g1.tar.gz`、远端原始数据或训练产物。

### 7. Git 提交与 noetix-12 同步

- 功能提交为 `a60adeddbcf9dc2067f2d2a0ee811c76af849515`，已推送到 GitHub 的
  `feature/g1-native-sonic-training`；提交只包含上述五个代码、脚本与文档文件，没有
  暂存 `g1.tar.gz` 或 `data/` 下的原始/生成数据。
- 拉取前再次确认 noetix-12 位于同名分支、HEAD 为 `edc80bb06ad2` 且工作区干净；随后
  使用 `git pull --ff-only origin feature/g1-native-sonic-training` 快进到 `a60aded`，
  拉取后工作区仍干净。该同步只更新服务器代码 checkout，没有启动或重启训练与部署进程。


## 2026-09-08：GENMO 音乐会话部署实现（进行中）

- 分支 feature/g1-native-sonic-training，起始 HEAD cf47eaced1f8ad7b4d7564112663085efc430d03，保留未跟踪 g1.tar.gz。
- 用户授权实现 physics_v3 s100000、SONIC release SMPL mode 2、本机 MuJoCo 三进程同步播放。
- 首批新增 music_session.hpp，提供请求去重、连续帧确认、不可变缓冲快照；控制器新增 music 输入与统一单调时钟帧选择，固定 mode 2，避免旧流接口追赶重置和模式回退。
- 当前为源码阶段，尚未完成编译、仿真和同步验收。无训练和真机操作。
- 回滚仅涉及新增音乐接口及显式 music 分支，不涉及模型、训练配置或历史轨迹。

本次实现进展：新增 `gear_sonic/utils/mujoco_sim/music_session.py`；在 `configs.py`、`base_sim.py` 增加本机会话、固定步长绝对期限、播放时释放弹力带、故障冻结、音乐模式关闭跌倒自动重置及真实 qpos/qvel 日志；C++ 音乐输入锁定 mode 2、以单调时钟选帧，用户显式停止允许在未消费未来段加入 1 秒收尾。修正主线程 sleep(0.02) 被整型截断引起的忙循环，改为 20 ms sleep_for，降低与推理/仿真的 CPU 竞争。以上尚待端到端及故障验证。

进展与验证：新增 `unit_tests/test_music_session.cpp`；4 项 C++ 协议测试通过。测试发现 release fast-math 消除了普通 isfinite 检查，已改为 IEEE754 位检查，NaN/Inf 明确拒绝；MuJoCo 会话 reset 后零四元数问题通过紧接 mj_forward 修复。新增实际控制周期统计及编码器输入审计，8 秒真实音频会话 436 次 Python/C++ 观测最大误差 8.47e-16，控制完整周期 P99 1.801 ms；无弹力带、无 reset，仿真实际步进 200.007 Hz，RTF 1.000037，基座最低 0.7133 m、最大倾角 17.072 度。新的请求类型校验、活动会话隔离及容量上下文修正仍需最终回归编译和故障注入。

最终交付整理进展：补齐 `docs/sonic_music_input.md`、7 项 C++ 协议测试与 2 项 MuJoCo 会话单元测试；新 Python 文件经 Ruff 格式化。C++ 新会话清零使用帧与时间戳，完成操作拒绝已锁存故障，完整控制周期墙钟频率使用首尾实测控制时间。三进程 heartbeat/generation_delay/user_stop 注入通过（GENMO `outputs/sonic_music/fault_validation_20260908/results.json`）。读取本地 SDK 的 recurrent_thread.cpp.o 确认 CLOCK_MONOTONIC timerfd 周期调度，因此保留现有控制线程实现。初次 180 秒在线 FineDance 已完整结束，无跌倒/重置/弹力带；30 秒 Compas3D 也完成。最终帧与声卡回归、提交前检查进行中。

最终验证完成：`outputs/sonic_music/20260908_192304_56097e25`，源曲 `outputs/server_music_wav_4set_10_20260818/finedance/100.wav` 的前 180 秒，seed 42。5400 帧 30 Hz SMPL、9000 帧音乐参考，加前后缀共 9160 帧，编号完全连续；17 个代码/模型/资产指纹与当前文件一致。

| 指标 | 最终实测 |
|---|---|
| 播放中生产 | 58 窗，平均 75.02 ms，P95 77.62 ms |
| 音频/实际参考时间误差 | P95 10.53 ms，最大 17.39 ms |
| 完整控制周期计算 | P99 1.304 ms，墙钟 50.00006 Hz |
| 物理步进和实时因子 | 200.00055 Hz，RTF 1.000003 |
| GPU 总已用显存峰值 | 3.041 GiB，含显示及同时运行的模型 |
| 控制状态 | 无弹力带、无跌倒、无 reset；最低基座 0.368 m，最大倾角 23.66° |
| 双膝跟踪相位 | 约 40 ms 滞后，相关 0.959，只代表双膝信号 |
| 足底接触点切向速度 | RMS 0.108 m/s，P95 0.117 m/s，存在脚滑 |

测试与边界：GENMO 17 项、MuJoCo 会话 2 项、C++ 协议 7 项通过；Ruff 新文件检查、git diff --check 通过；三个实际三进程故障用例通过。8 秒固定观测审计、30 秒 Compas3D、两次 180 秒 FineDance 与 1.037 秒短尾窗完成。未经人工主观视听复核、未验证真机、未启动训练。部署资料/录像/模型和用户要求的失败现场保留在 GENMO outputs，未纳入 Git。回滚仅撤销本次新增音乐模式、适配/协调/评估/测试/文档及对应窄范围改动，不改原 checkpoint、数据、用户 agent.md 或 g1.tar.gz；模式默认值仍不启用音乐。源代码将在当前 feature 提交推送并只对同名服务器分支执行 ff-only。

交付复核：最终完整录像为 GENMO `outputs/sonic_music/20260908_192304_56097e25/dance_with_music.mp4`，H.264 1280×720、50 FPS，AAC 48 kHz，总长 183 秒（两秒准备＋180 秒音乐＋一秒收尾），已检查实际帧画面与媒体流信息。评估工具将未开启观测审计的会话标记为 null，避免把未运行审计误记失败；原 8 秒已开启审计的通过证据保留。所有本次联调子进程已经退出；模型与媒体/故障归档按用户要求保留，其余临时包、编译审计对象和重复日志精确清理。
