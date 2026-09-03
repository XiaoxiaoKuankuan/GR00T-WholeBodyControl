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
