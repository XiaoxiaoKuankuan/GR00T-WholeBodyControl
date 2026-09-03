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
