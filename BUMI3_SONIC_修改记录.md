# BUMI3 原生 SONIC 修改记录

本文档记录 BUMI3 原生 SONIC 支持的实际修改、机器人参数来源、兼容性边界和验证证据。所有结论区分静态检查、Isaac Sim 配置导入、环境 reset/step 与真实训练；未执行的测试不会表述为已通过。

## 2026-08-28：修复五集合根坐标契约、足底验证和脚部训练闭环

### 1. 修改目标、分支与工作区保护

- 所属分支：`feature/bumi-native-sonic-full-training`；起始 HEAD：
  `b1c3606ce96f00a01745cb8382f8bfa0b9b4d780`，与同名 origin 分支 ahead `0`、behind
  `0`。本轮没有执行 commit、push、pull、merge、rebase、stash、reset 或分支切换。
- 修改前工作区已有 sim2sim 代码、配置、测试、文档、`gear_sonic/pyproject.toml`、
  `agent.md` 和本记录文件等未提交工作；这些内容全部视为用户受保护修改。本轮没有覆盖、
  回退或暂存它们，`agent.md` 未由本轮修改。
- 目标是修复已停止训练所使用的 `hq_all_v1` 数据中公开四库根姿态坐标错误，补足会让
  横躺/穿地动作静默通过的验证，并让“5point”奖励实际包含双脚；不启动服务器训练，
  不原地覆盖旧数据，不改 SONIC 网络、PPO、控制频率或其他奖励数值。

### 2. 故障证据与坐标契约结论

- 对服务器现有 3,162 条公开动作和 99 条 Mine 动作进行了只读全帧根倾角审计。旧输出中：
  - AIOZ-GDANCE 1,978 条：中位根倾角 `87.693°`，`>45°` 帧占 `99.789%`；
  - AIST++ 963 条：中位 `79.847°`，`>45°` 占 `96.339%`；
  - CoMPAS3D 72 条：中位 `89.742°`，`>45°` 占 `99.885%`；
  - FineDance 149 条：中位 `88.348°`，`>45°` 占 `99.088%`；
  - Mine 99 条：中位 `5.503°`，`>45°` 仅 `0.058%`。
- 每个源文件本身已携带可区分的契约，不能通过倾角猜测来源：公开四库为
  `genmo.bumi_legacy_motion.v1`，Mine 为 `genmo.bumi_csv_qpos_xyzw.v1`。原转换器读取根
  `wxyz` 四元数后直接写出，没有处理公开库的 legacy Y-up 根姿态。
- 已验证的公开库修正是世界系左乘 `Rx(+90°)`：
  `q_zup = [sqrt(0.5), sqrt(0.5), 0, 0] ⊗ q_legacy`。全量只读统计应用该修正后，五集合
  中位倾角依次为 AIOZ `8.441°`、AIST `15.477°`、CoMPAS `9.444°`、FineDance
  `10.187°`，Mine identity 后仍为 `5.503°`；公开库 `>45°` 比例降为 `0.586%`、
  `8.770%`、`0.646%`、`2.600%`。AIST 中仍有真实大倾角动作，因此验证采用数据集聚合
  阈值，不粗暴删除任意单帧 `>45°` 的动作。
- 只修四元数还不够：公开库 Root-Z 是修正前足底 QP 的结果；Mine 虽然姿态正确，但源
  元数据明确采用 `legacy_body_origin_min_zero`。抽样 Mine 在当前 BUMI3 MJCF 下足底最低
  到 `-0.051115 m`。因此 Mine 四元数保持 identity，但所有集合的 Root-Z 都必须在当前
  SONIC BUMI3 MJCF 下重新对地，不能为让旧数据通过而放宽穿地检查。

### 3. 修改文件与具体内容

- `gear_sonic/tools/prepare_bumi3_sonic_dataset.py`：
  - 将输出 provenance 升级为 `sonic.bumi3_hq_all.v2`，要求每个源文件存在
    `source_motion_contract_version`；公开库只接受 legacy 契约并做固定 `Rx(+90°)`，Mine
    只接受 CSV Z-up 契约并做 identity，契约和数据集不匹配时直接失败，不使用启发式猜测；
  - 同时更新 `root_rot` 和 `pose_aa[:, 0]`，写出最终 root frame、修正四元数、Root-Z
    policy 和优化诊断，验证二者的四元数 round trip 一致；
  - 用当前 `bumi3.xml` 的关节名称和 `jnt_qposadr` 执行 MuJoCo FK，使用
    `mj_geomDistance(ground, foot_geom)` 计算实际几何足底距离，不用 body origin 或虚构
    包围盒点替代脚底；
  - 以足底接触软目标、修正的一/二阶平滑项和每帧硬下界重新求 Root-Z；二阶项只作用于
    新增修正，不抹平源动作本身的起跳/落地加速度；最终每帧足底穿透最多 `0.002 m`；
  - 全量 `validate` 新增 source/root 契约、四元数范数、根姿态与 axis-angle 一致性、每段
    17 帧 FK 抽检、完整序列穿地诊断，以及每数据集根倾角中位数 `<=30°`、`>45°` 比例
    `<=20%`；shape、有限值、FPS、配对、计数和 SHA256 原检查继续保留；
  - manifest/provenance 记录 root correction、Root-Z policy、倾角阈值和穿地容差；示例输出
    改为新目录 `hq_all_v2`，避免误覆盖或继续使用已知错误的 `hq_all_v1`。
- `gear_sonic/tools/test_prepare_bumi3_sonic_dataset.py`：从 4 项扩展到 6 项；Mine fixture
  显式声明 Z-up 契约并验证 identity/足底，新增公开 legacy `Rx(+90°)` + Root-Z 回归测试，
  新增“公开数据伪装成 Mine 契约必须失败”测试。
- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bumi3.yaml`：
  - `reward_point_body` 从实际 3 点扩展为腰、双肘、双踝 5 点，双脚 offset 为零；继续调用
    原 `tracking_local_vr_5point_error`，其 weight `2.0`、std `0.1` 和实现均未修改；
  - `foot_pos_xyz` 从 BUMI3 覆盖值 `0.15 m` 恢复到 SONIC/G1 基础值 `0.20 m`，让训练早期
    能先通过五点奖励纠正脚部，再触发三维脚误差终止；
  - `ee_body_pos=0.12` 的自适应高度检查、`anchor_pos=0.12`、`anchor_ori_full=0.20`、
    `feet_acc=-2.5e-6`、力矩限制奖励、其他奖励和全部 domain randomization 保持不变。
- `gear_sonic/tools/validate_bumi3_integration.py`：锁定最终 5 个 reward body 和 `0.20 m`
  foot threshold；smoke 不再强制使用依赖可选 Nucleus USD 的 `plane`，改用正式 BUMI3
  配置的本地生成 `trimesh`。静态集成仍同时 compose `sonic_release` 和 `sonic_h2`。
- `BUMI3_SONIC_修改记录.md`：新增本节，记录故障证据、变更、实际测试、失败项、风险与
  回滚方法。

### 4. 真实源动作抽样结果

- 从服务器只读拉取 AIST、AIOZ、FineDance、CoMPAS 和 Mine 各一条源动作，在本地临时
  目录用最终代码转换并对全部帧做 MuJoCo 几何 FK：
  - AIST：倾角中位数 `78.001° -> 16.677°`，P95 `27.545°`，足底最低 `-0.002 m`；
  - AIOZ：`86.894° -> 6.864°`，P95 `15.304°`，足底最低 `-0.002 m`；
  - FineDance：`83.788° -> 9.018°`，P95 `27.169°`，`>45°` 为 `3.245%`，足底最低
    `-0.002 m`；
  - CoMPAS：`82.140° -> 12.992°`，P95 `21.124°`，足底最低 `-0.002 m`；
  - Mine：倾角保持 `5.963°`，P95 `12.765°`，足底从已知最低 `-0.051115 m` 修正到
    `-0.002 m`，没有对 Mine 应用公开库姿态旋转。
- 五条动作的额外动态 Root-Z 最大修正分别为 `0.045675`、`0.071957`、`0.104962`、
  `0.072836`、`0.033079 m`；最大修正加速度分别为 `4.476`、`3.538`、`6.267`、
  `5.514`、`1.962 m/s²`。这些值被写入每段输出诊断，便于全量构建后审计离群段。
- 远端源文件没有被修改；本地临时源副本、单段转换产物和临时日志验证后均已逐文件删除，
  未进入 Git。

### 5. 实际运行的验证与结果

- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m pytest -q
  gear_sonic/tools/test_prepare_bumi3_sonic_dataset.py`：`6 passed`。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m pytest -q
  gear_sonic/tools/test_prepare_bumi3_sonic_dataset.py gear_sonic/tests/test_bumi3_sim2sim.py`：
  `11 passed`；只有既有 `<unknown>:4 invalid escape sequence` DeprecationWarning。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python
  gear_sonic/tools/validate_bumi3_integration.py`：通过；实际解析 `21 DoF/22 bodies`，
  `sim_dt=0.005`、`decimation=4`、control/target `50 Hz`、action `21`、FSQ token `64`、
  actor proprioception `690`、tokenizer flat `1262`、critic `1245`、dynamic decoder
  `754 -> 21`，并通过 G1/H2 compose 与 mapping 检查。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m compileall -q gear_sonic`：通过。
- `git diff --check`：通过。
- 本地环境没有安装独立 Ruff，因此没有把行长人工检查描述为 Ruff 已通过；变更 Python
  文件按仓库 `115` 字符限制检查，无超长行。

### 6. 动态 smoke 失败项与未运行项

- 使用修正后的 FineDance `001` Robot/SMPL 单段数据运行 1-env、10-step smoke。第一次在
  scene 创建前失败，因为验证器原先强制 `plane`，本机 `ISAAC_NUCLEUS_DIR=None`，请求了
  `None/Isaac/Environments/Grid/default_environment.usd`；改用本地 `trimesh` 后该问题消失。
- 第二次到达 BUMI3 URDF 导入时失败：本地 Isaac Lab 的 `UrdfConverter` 调用
  `ImportConfig.set_merge_fixed_ignore_inertia`，但当前 Isaac Sim URDF importer 没有该 API；
  这是本机 Isaac Lab/Isaac Sim 版本不配套。环境未完成创建，因而本轮没有把 reset、step
  或 NaN/Inf smoke 写成通过；该兼容问题需在正式服务器 `liwei_lab` 环境复核/修复。
- 未在服务器构建完整 `hq_all_v2`、未运行 3,261/3,162 全量 `validate`、未运行 16-env
  replay、100 iteration smoke training 或八卡训练：本轮代码尚未获得 commit/push 授权，
  旧 `hq_all_v1` 又是本次确认的错误产物，不能拿它冒充修复后验证。服务器训练保持停止。

### 7. 兼容性、后续边界与回滚

- `sonic_release`、H2、G1/H2 mapping、Robot+SMPL 双编码器、FSQ、PPO、critic、trainer、
  `sim_dt`、decimation、history/future frame、SMPL 参数均未修改。只改变 BUMI3 活跃配置的
  五点名称集合和脚部三维终止阈值。
- 旧 `hq_all_v1/built` 不得继续用于训练或 ONNX 导出；修复代码默认示例将新产物放到
  `hq_all_v2`，构建器本身也拒绝覆盖已有 `robot_all/smpl_all`。全量构建通过后仍需先检查
  每集合倾角摘要、Root-Z 离群诊断和 Isaac reset/step，再允许重新训练。
- 回滚时只需撤销本节涉及的四个代码/配置/测试文件和本记录节；数据修正未写入服务器，
  没有远端产物需要删除。回滚到旧转换器只代表恢复代码，不代表旧 `hq_all_v1` 数据正确。

## 2026-08-28：服务器构建 hq_all_v2 并从零重启八卡训练

### 1. 授权、保护边界和服务器现状

- 用户明确要求“在服务器上重新开始训练”，因此本轮只向
  `noetix-volc` 的 `/home/liwei/GR00T-WholeBodyControl` 同步本次 BUMI3 修复相关文件、
  构建新数据并启动新实验；没有 commit、push、pull、merge、rebase、stash 或 reset。
- 本地和服务器分支均为 `feature/bumi-native-sonic-full-training`，HEAD 为
  `b1c3606ce96f00a01745cb8382f8bfa0b9b4d780`。服务器同步前是干净工作区；本地既有
  sim2sim、`agent.md`、`pyproject.toml` 等受保护未提交修改没有被同步、回退或覆盖。
- 服务器原代码备份为
  `/data/sonic_bumi3/code_snapshots/pre_hq_all_v2_20260828_131927`；旧 `hq_all_v1`、旧
  `hq_all_scratch_100k-20260827_170205` 训练目录及 16k checkpoint 均保留，但新训练不读取它们。
- 服务器 Conda 环境是 `/root/miniconda3/envs/liwei_lab`，PyTorch
  `2.7.0+cu126`、MuJoCo `3.3.2`、Isaac Sim `5.1.0`。启动前确认 8 张 RTX 4090 D
  均无训练进程、显存约 `2 MiB`。

### 2. 开训前追加修正

- `gear_sonic/data/assets/robot_description/urdf/bumi3/bumi.urdf`：复核发现当前提交仍是
  较大的旧腿部圆柱，与用户先前明确参数不一致；因此在正式训练前落实最终值：
  - 左右 `leg_roll_link`：origin `[0, 0, -0.02]`、radius `0.03`、length `0.08`；
  - 左右 `knee_pitch_link`：保留之前 origin `[0.008475, 0, -0.0894694]`，radius `0.025`、
    length `0.13`；
  - `base_link` 保持 length `0.12`，左右 `leg_pitch/yaw` 继续无 collision。
  URDF 已重新统一为 CRLF，避免因换行符造成全文件假 diff。
- `gear_sonic/tools/validate_bumi3_integration.py`：参考资产路径不再写死为
  `/home/weili`；先读 `BUMI3_REFERENCE_ROOT`，再按仓库同级、`/home/weili`、`/home/liwei`和
  `/home/listao` 候选路径查找，仍使用原 SHA256 锁定权威版本。服务器自有
  `legged_lab` 的 `bumi.py/MJCF` SHA 与本集成权威版不同，未修改该参考仓库；只把本机
  权威快照复制到 `/data/sonic_bumi3/reference_assets/bumi3` 供验证使用。
- `gear_sonic/tools/prepare_bumi3_sonic_dataset.py`：`_git_commit` 使用精确仓库路径的
  `git -c safe.directory=...` 读取 HEAD，修复 root 用户读取 `/home/liwei` 普通用户仓库时
  provenance 被记录为 `unknown` 的问题。该修正不改动作数值，只重写 metadata 和 SHA 清单。
- `BUMI3_SONIC_修改记录.md`：新增本节，记录服务器同步、新数据、smoke、正式训练与
  尚未证明的质量边界。

### 3. hq_all_v2 构建与独立全量验证

- 构建命令读取 `hq_all_v1/source_{bumi,smpl,mine}`，只向新目录
  `/data/sonic_bumi3/datasets/hq_all_v2` 写入，使用 32 workers；日志为
  `/data/sonic_bumi3/logs/prepare_hq_all_v2_20260828_132309.log`。完整转换、内置全量验证和
  原子发布均通过，随后再独立执行 `validate` 一次，输出
  `BUMI3_SONIC_DATASET_VALIDATE=PASS`。
- 最终数据为 3,261 条 robot、3,162 条 SMPL、99 条 Mine-only，数据集约 `3.7G`。
  provenance 为 `sonic.bumi3_hq_all.v2`，代码 HEAD 为 `b1c3606...`，公开库修正是
  `[0.70710678, 0.70710678, 0, 0]`，运行目标 50Hz，足底穿透容差 `0.002 m`。
- 全量修正后根倾角聚合结果：AIST++ `15.477° / 8.770%`、AIOZ-GDANCE
  `8.441° / 0.586%`、FineDance `10.187° / 2.600%`、CoMPAS3D `9.444° / 0.646%`、
  Mine `5.503° / 0.058%`；每组分别为中位倾角和 `>45°` 帧占比。
- 元数据 SHA256：
  - `meta/SHA256SUMS`：`2aa75a3ab0c95b999978be4fc29d56d261aee2cf3ad5a3cc39b3f6175c4bd427`；
  - `meta/manifest.jsonl`：`7c388ae40874d06d195afcf336f5dbc5b2a2de5a48a8e1b0d1f80290c83058da`；
  - `meta/provenance.json`：`bc8debdb3604164acfeaa8a801438008c86784a8dc3593e339da26b03cca66b6`。

### 4. 实际验证命令和结果

- 本机 `env_isaaclab`：`compileall -q gear_sonic`、数据工具 6 项 pytest 和完整
  `validate_bumi3_integration.py` 均通过；后者实际解析 21 DoF/22 bodies、映射、执行器、
  Robot+SMPL 网络和 Hydra 兼容配置。
- 服务器 `liwei_lab`：`compileall -q gear_sonic`、`pytest
  gear_sonic/tools/test_prepare_bumi3_sonic_dataset.py` 为 `6 passed`；不启动 Isaac 的资产/Hydra
  静态部分通过，resolved 为 `sim_dt=0.005`、`decimation=4`、50Hz、action 21、FSQ 64、
  actor proprioception 690、tokenizer 1262、critic 1245、decoder `754 -> 21`。
- 服务器直接执行验证器的裸 `SimulationApp` 时，会因 pip IsaacLab 的 source 路径和
  headless Vulkan 上下文报 `isaaclab.sim`/图形插件错误；未把这次失败写成通过。改用项目
  正式 `train_agent_trl.py` 入口自身的 `AppLauncher` 后，16 env、2 iterations 真实 smoke
  完成 768 timesteps，两次 PPO 更新均成功，无 Traceback、OOM、NCCL、NaN/Inf。
- smoke 配置是 `checkpoint:null`、`auto_load_latest:false`，数据路径为 `hq_all_v2`；
  日志为 `/data/sonic_bumi3/logs/smoke_hq_all_v2_20260828_134953.log`，产物目录为
  `/data/sonic_bumi3/smoke/TRL_BUMI3_Track/manager/universal_token/all_modes/
  sonic_bumi3_hq_all_v2_coordfix_smoke-20260828_134957`。smoke 权重没有作为正式训练初始化。

### 5. 八卡正式从零训练

- tmux：`bumi3_sonic_v2`；日志：
  `/data/sonic_bumi3/logs/train_hq_all_v2_20260828_135112.log`；实验目录：
  `/data/sonic_bumi3/runs/TRL_BUMI3_Track/manager/universal_token/all_modes/
  sonic_bumi3_hq_all_v2_coordfix_scratch_100k-20260828_135120`。
- resolved 配置明确为 8 processes、每 rank `num_envs=4096`、`100000` iterations、
  `checkpoint:null`、`auto_load_latest:false`、`resume:false`、`sim_dt=0.005`、`decimation=4`，
  robot/SMPL 都只读 `hq_all_v2/built`。旧 16k checkpoint 和 smoke 权重均未加载。
- 截至 `iteration 105`，进程仍在运行；每轮 786,432 timesteps，累计
  `82,575,360` timesteps，最近吞吐 `220,558 steps/s`、iteration `3.57s`，8 卡各占约
  `19.7–20.2 GiB`。日志无 Traceback、OOM、NCCL timeout/error、NaN/Inf，`last.pt`
  已保存，大小 `385,843,368` bytes。
- 当时 mean reward `0.46244`、mean length `9.5775` steps；主要 termination 分解为
  `ee_body_pos=0.6108`、`anchor_ori_full=0.2627`、`foot_pos_xyz=0.2028`、
  `anchor_pos=0.0067`。这只证明从零随机策略的训练通道正常；前 100 iterations 的
  mean length 仍在约 9–10 steps 波动，没有证明中后期策略质量已恢复。
- 根据当时 ETA `353,339 s`，动态估计剩余约 `98.1 h`（约 4.1 天）；该值会随
  采样/更新吞吐变化，不是完成承诺。应在 500/1000/5000 iterations 继续检查 mean length、
  termination 分解和固定动作回放，不能只看训练 reward 就判定物理质量或真机安全。
- 服务器仍会打印已知的 `VkResult: ERROR_INCOMPATIBLE_DRIVER`/图形插件告警；
  本次 headless smoke 和 8 卡正式训练均继续执行 PhysX/CUDA。这不等价于 GUI/相机/渲染
  已验证，以后若需可视化仍必须单独修复 Vulkan ICD/驱动环境。

### 6. 回滚与继续监控

- 停止新训练可向 `tmux` 会话 `bumi3_sonic_v2` 发送 `Ctrl-C`；不要删除新数据或
  旧实验来“回滚”代码。代码回滚目标是本节的 URDF、验证器路径解析和 provenance
  safe-directory 三处追加修正，覆盖前服务器文件可从上述 code snapshot 恢复。
- 本轮交付时既没有自动停止正式训练，也没有 commit/push 当前未提交工作区。

## 2026-08-28：新增 BUMI3 原生 SONIC MuJoCo sim2sim

### 1. 修改目标、分支与起始状态

- 所属分支：`feature/bumi-native-sonic-full-training`。
- 起始 HEAD：`b1c3606ce96f00a01745cb8382f8bfa0b9b4d780`。
- 修改前本分支相对 `origin/feature/bumi-native-sonic-full-training` 为 ahead `0`、behind
  `0`；没有执行 pull、commit、push、merge、rebase、stash 或分支切换。
- 修改前已有且受保护的用户工作区内容为 `BUMI3_SONIC_修改记录.md` 和 `agent.md`；本轮
  只在记录文档新增本节，没有撤销、覆盖或改写原有记录，`agent.md` 未由本轮修改。
- 目标是参考现有 G1 SONIC sim2sim 的动作参考、历史观测、ONNX 推理和 MuJoCo PD 闭环，
  增加可独立运行的 BUMI3 版本；不把 BUMI3 强行接入 G1 专用的 29 电机、Unitree DDS、
  C++ 硬件 order 和 robot model。

### 2. 配置与资产来源

- 权威参考仓库：`/home/weili/legged_lab`，分支 `main`，HEAD
  `d555c76e5977af66ef55a104b98e1be486349996`。
- 参考工作区当前有未提交修改，本轮只读、未修改；其中：
  - `assets/robots/bumi3/bumi.py` SHA256：
    `74aaeca9da615c50e3749e4f103bbf713b83443d9cb16fab08edfd320227c03e`；
  - `assets/robots/bumi3/mjcf/bumi3.xml` SHA256：
    `041c81e8176c7f375302796deca28b141891a3c097d8e341e8d967b735466edf`。
- 本仓库实际加载的 BUMI3 MJCF SHA256：
  `02874afebbe30ba1f90218394c8f9953f5d7a808e6b9950e7964c731da6dfbfe`。验证脚本会把
  本地 MJCF 与当前参考做完整 XML 语义比较，当前唯一允许的差异是仓库布局导致的
  `compiler.meshdir`：参考为 `../meshes/`，本地为 `../meshes/bumi3/`。
- PD、effort、velocity、armature、初始姿态和 action scale 来源于当前本仓库
  `gear_sonic/envs/manager_env/robots/bumi3.py`，该文件此前已按上述参考 `bumi.py`
  逐字段验证。本轮没有修改 BUMI3 URDF、MJCF、mesh 或训练配置。

### 3. 修改文件与具体目的

- `gear_sonic/config/sim2sim/bumi3_sonic.yaml`：新增集中、可审计的 BUMI3 sim2sim
  契约，记录 21 关节 MuJoCo/IsaacLab 顺序、初始姿态、PD、effort/velocity、踝关节
  armature、时间参数和网络维度。动作缩放不写死，运行时始终按
  `0.25 * effort_limit / stiffness` 计算。
- `gear_sonic/utils/mujoco_sim/bumi3_sim2sim.py`：新增核心实现：
  - 通过名称查询每个关节的 `jnt_qposadr`、`jnt_dofadr` 和 motor，不使用
    `qpos[7:]` 这类隐式顺序；
  - 支持 SONIC PKL、NPZ 和 G1 `MotionDataReader` 风格 CSV clip；PKL 自动采用
    MuJoCo DoF + `xyzw` root quaternion，CSV 自动采用策略 DoF + `wxyz`，也可由 CLI
    显式覆盖；只接受 50 FPS，不在部署端静默重采样；
  - 复刻 SONIC Robot Encoder 的 10 个 future frames、0.1 秒间隔和训练端
    `command_multi_future_nonflat` 的实际 flatten 布局；
  - 默认计算 `robot_start_heading * inverse(reference_start_heading)`，把参考动作起始 yaw
    对齐到机器人，与 G1 sim2sim 的 `ComputeApplyDeltaHeading` 行为一致，并支持 CLI 关闭；
  - 按训练 `PolicyCfg` 顺序构造 10 帧 `base_ang_vel / joint_pos_rel / joint_vel /
    last_action / gravity_dir`，proprioception 为 690 维；
  - 加载 `eval_agent_trl.py` 导出的 `*_g1.onnx` 联合 Robot Encoder + dynamic decoder，
    严格验证 `1170 -> 21`；这里 `g1` 仅为网络内部兼容键名；
  - 用 BUMI3 action scale、PD 和 effort clip 在 `0.005` 秒 MuJoCo step、decimation `4`
    下闭环执行，并逐步检查 observation/action/torque/qpos/qvel/ctrl 有限值。
- `gear_sonic/scripts/run_bumi3_sim2sim.py`：新增 Tyro CLI，支持 GUI/headless、实时/最快
  执行、动作选择、顺序覆盖、起始帧、时长、循环和 validate-only。
- `gear_sonic/tools/validate_bumi3_sim2sim.py`：新增一键验证；检查参考 MJCF、全部 mesh、
  `nq/nv/nu/body`、关节和 actuator 顺序、mapping round trip、action scale、armature、
  480/690/1170/21 维度以及 100 控制周期有限值；可选真实 ONNX 和动作输入。
- `gear_sonic/tests/test_bumi3_sim2sim.py`：新增 4 个回归测试，覆盖映射/维度/action scale、
  SONIC PKL 顺序、G1 风格 CSV 顺序和无界面 MuJoCo 闭环。
- `docs/source/getting_started/bumi3_sim2sim.md`：新增环境安装、checkpoint 导出、GUI/
  headless 运行、动作格式、验证命令和能力边界说明。
- `gear_sonic/pyproject.toml`：在既有 `sim` extra 中加入 `onnxruntime`，使新的推理入口按
  文档安装后具备完整依赖；不修改 training/teleop/inference extra。
- `BUMI3_SONIC_修改记录.md`：新增本次来源、实现、测试、风险和回滚记录。

### 4. 最终顺序、映射与 resolved 契约

BUMI3 MuJoCo 顺序：

`[waist_yaw, l_arm_pitch, l_arm_roll, l_arm_yaw, l_elbow_pitch, r_arm_pitch,
r_arm_roll, r_arm_yaw, r_elbow_pitch, l_leg_pitch, l_leg_roll, l_leg_yaw,
l_knee_pitch, l_ankle_pitch, l_ankle_roll, r_leg_pitch, r_leg_roll, r_leg_yaw,
r_knee_pitch, r_ankle_pitch, r_ankle_roll]`，完整名称均带 `_joint`。

策略/IsaacLab 顺序：

`[l_leg_pitch, r_leg_pitch, waist_yaw, l_leg_roll, r_leg_roll, l_arm_pitch,
r_arm_pitch, l_leg_yaw, r_leg_yaw, l_arm_roll, r_arm_roll, l_knee_pitch,
r_knee_pitch, l_arm_yaw, r_arm_yaw, l_ankle_pitch, r_ankle_pitch,
l_elbow_pitch, r_elbow_pitch, l_ankle_roll, r_ankle_roll]`，完整名称均带 `_joint`。

- IsaacLab → MuJoCo：
  `[2,5,9,13,17,6,10,14,18,0,3,7,11,15,19,1,4,8,12,16,20]`。
- MuJoCo → IsaacLab：
  `[9,15,0,10,16,1,5,11,17,2,6,12,18,3,7,13,19,4,8,14,20]`。
- 实际 MJCF：`nq=28`、`nv=27`、`nu=21`、robot bodies `22`（MuJoCo `nbody=23`
  还包含 world body）、mesh `22`。
- `sim_dt=0.005`、`decimation=4`、control frequency `50 Hz`、target FPS `50`、history
  `10`、future frames `10`、future stride `5`（0.1 秒）。
- Robot tokenizer `480`、actor proprioception `690`、联合 ONNX input `1170`、FSQ token
  `64`、action/output `21`。

### 5. 实际运行的验证与结果

- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m compileall -q gear_sonic`：通过。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m pytest -q
  gear_sonic/tests/test_bumi3_sim2sim.py`：`5 passed`，其中新增 90 度参考 yaw 对齐回归测试。
- `.venv_sim/bin/python -m compileall -q gear_sonic`、`.venv_sim/bin/python -m pytest -q
  gear_sonic/tests/test_bumi3_sim2sim.py` 和 CLI `--help`：通过；证明独立 Python 3.10
  MuJoCo 环境可编译、可运行全部 5 个测试并正确生成命令帮助。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python
  gear_sonic/tools/validate_bumi3_sim2sim.py --steps 100`：通过；实际执行 100 个 50 Hz
  control steps / 400 个 MuJoCo steps，仿真时间 `2.0 s`，全链路无 NaN/Inf。
- 使用 ONNX helper 临时生成严格 `1170 -> 21` 的零输出 ONNX，并生成带显式
  `joint_order=mujoco`、`quaternion_convention=wxyz` 的 50 FPS BUMI3 NPZ：
  - `.venv_sim/bin/python gear_sonic/tools/validate_bumi3_sim2sim.py --policy
    /tmp/bumi3-sim2sim-ucYE2d/mock_g1.onnx --motion
    /tmp/bumi3-sim2sim-ucYE2d/motion.npz --steps 100`：通过，验证了真实 ONNX Runtime
    session、真实文件加载和 100 周期闭环；
  - `.venv_sim/bin/python gear_sonic/scripts/run_bumi3_sim2sim.py ... --duration 0.1
    --headless --no-real-time`：通过，实际运行 5 个控制周期，仿真时间 `0.1 s`。
  - 上述 `/tmp` 文件仅为接口测试产物，不是训练数据或可交付 checkpoint；验证后已逐文件
    删除并移除空临时目录，未加入 Git，也没有保留生成数据。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python
  gear_sonic/tools/validate_bumi3_integration.py`：通过；重新确认 `sonic_bumi3` 与现有 G1/H2
  Hydra compose、执行器、mapping、时间参数和网络维度兼容检查未回归。
- `git diff --check`：通过。
- 使用独立临时 index 显式加入全部已跟踪修改和 6 个未跟踪新增文件后运行
  `git diff --cached --check`：通过；首次完整统计为 `9 files changed, 1962 insertions,
  2 deletions`。真实 Git index 在审计前后均为空，未替用户暂存任何文件。
- 为运行独立 sim 环境验证，在未跟踪的 `.venv_sim` 中安装了 `onnxruntime==1.23.2` 和
  `pytest==9.1.1`；项目依赖声明只新增 `onnxruntime`，pytest 仅为本地测试工具。

### 6. 未通过/未运行项与原因

- 合并运行新增测试和现有 `gear_sonic/tests/test_input_readers.py` 时，现有测试在 collection
  阶段失败：它从当前 `gear_sonic.utils.teleop.input_readers` 导入
  `build_body_pose_sample`，但该函数当前不存在。该失败发生在任何 BUMI3 测试执行前，属于
  现有 Teleop 测试/实现漂移；本轮不修改无关 Teleop 通用代码。单独运行全部 4 个新增测试
  已通过。
- 未使用 `model_step_016000.pt` 导出并回放真实 BUMI3 policy：本机 checkpoint 的 resolved
  config 指向 `/data/sonic_bumi3/datasets/hq_all_v1/built/robot_all` 和 `smpl_all`，这些训练数据
  本机不存在；本轮不擅自从其他 BUMI3/G1 数据替代，也不生成训练数据。因而本轮 ONNX
  Runtime 证据是接口级临时 ONNX，不是训练效果验证。
- 未运行 GUI viewer：本轮自动验证使用 headless，CLI 的 viewer 分支已完成导入与参数路径
  检查，但没有把无界面 smoke 描述为 GUI 验证。
- 未做实机测试：该入口明确不包含 DDS/CAN/硬件安全映射，只用于 MuJoCo sim2sim。

### 7. 兼容性、已知风险与回滚

- 未修改 `gear_sonic/scripts/run_sim_loop.py`、`gear_sonic_deploy` 或 G1/H2 配置；现有 G1
  C++/DDS sim2sim 行为保持不变。BUMI3 使用新入口，避免 21 DoF 被错误塞入 29 DoF 映射。
- 本地 BUMI3 MJCF 保持当前 `legged_lab` 数值，只调整了既有 meshdir；训练使用的 URDF
  有用户指定的简化碰撞，而 MuJoCo MJCF 当前仍是 mesh geometry。该差异是 sim2sim 的一个
  明确物理域差异，对爬行/跪地效果的影响必须用真实动作和真实 checkpoint 继续评估。
- 100 周期零策略 smoke 中机器人最终跌倒但数值保持有限；这是零策略不产生平衡动作的预期
  结果，不能当作 policy 稳定性结论。
- 回滚时删除本轮 5 个新增代码/配置/测试文件及 1 个新增文档，移除 `pyproject.toml` 中
  `sim` extra 的 `onnxruntime` 一行，并删除本记录节即可；BUMI3/G1/H2 既有资产和训练代码
  均不需要回滚。本轮未 commit/push，Git 历史和远端没有变化。

### 8. 统一改用 Conda `env_isaaclab`

- 用户明确要求不使用 `.venv_sim`，后续 BUMI3 训练、ONNX 导出和 sim2sim 统一使用
  `/home/weili/miniconda3/envs/env_isaaclab`。sim2sim 实现本身从未绑定 `.venv_sim`；此前该
  环境只用于隔离测试。
- 检查时 `env_isaaclab` 已有 Python `3.11.15`、MuJoCo `3.3.2`、ONNX Runtime
  `1.27.0`、NumPy `1.26.4`、PyYAML `6.0.2` 和 joblib `1.5.3`，唯一缺少 CLI 依赖
  Tyro。
- 第一次直接安装最新版 `tyro==1.0.16` 时，它把 `typing_extensions` 从 Isaac Sim 5.1
  要求的 `4.12.2` 升级到 `4.16.0`。发现冲突后立即卸载该 Tyro 和其新增的 typeguard，
  恢复 `typing_extensions==4.12.2`，并安装与其兼容的 `tyro==0.8.14`；没有把环境留在
  已知冲突状态。
- `gear_sonic/pyproject.toml` 的 `sim` extra 同步将 Tyro 固定为 `0.8.14`，避免以后按
  extra 安装时再次升级 Isaac Sim 的 typing-extensions。文档的全部训练、导出、运行和
  验证命令也统一改为先 `conda activate env_isaaclab`。
- 在最终 Conda 环境中实际运行：
  - CLI `--help`：通过；
  - `python -m compileall -q gear_sonic`：通过；
  - `python -m pytest -q gear_sonic/tests/test_bumi3_sim2sim.py`：`5 passed`；
  - `python gear_sonic/tools/validate_bumi3_sim2sim.py --steps 100`：通过，400 个 MuJoCo
    steps 无 NaN/Inf；
  - `python gear_sonic/tools/validate_bumi3_integration.py`：在恢复 Isaac Sim 依赖版本后再次
    通过，确认 BUMI3/G1/H2 Hydra、执行器、mapping 和网络维度检查未受 Tyro 安装影响；
  - 使用临时 `1170 -> 21` ONNX 和 50 FPS NPZ 运行正式 CLI 5 个控制周期：通过，仿真
    时间 `0.1 s`；临时 ONNX/NPZ 随后已删除。
- `pip check` 仍报告两个本轮之前就存在的环境问题：IsaacSim kernel 声明
  `numpy==1.26.0` 而环境为 `1.26.4`，FastAPI 要求 `starlette<0.46.0` 而环境为
  `0.49.1`。本轮没有改动 NumPy、FastAPI 或 Starlette；当前 Isaac Lab 集成验证和
  BUMI3 sim2sim 均能运行，但这两个历史依赖漂移不能被描述为整个 Conda 环境完全无冲突。

## 2026-08-27：建立 Git 分支模型与 Agent 安全操作规范

### 修改目标与所属分支

- 所属分支：`feature/bumi-native-sonic-full-training`。
- 起始 HEAD：`b1c3606ce96f00a01745cb8382f8bfa0b9b4d780`。
- 用户要求为 SONIC 和 GENMO 统一建立 `feature/*`、`release/*`、`main` 的职责边界，
  并特别确认用户已有未提交修改均视为正确修改，Agent 不得擅自丢弃。

### 修改文件与具体内容

- `agent.md`：保留原有 BUMI3 来源、兼容性和强制记录规则，新增以下仓库级约束：
  - `main` 保持可运行、已验证、可交付；功能开发进入 `feature/*`，发布准备进入
    `release/*`，线上紧急修复进入 `hotfix/*`。
  - 规定分支来源、命名、允许修改范围、release 修复回流和语义版本 Tag 生命周期。
  - 将用户已有未提交修改定义为正确且受保护的内容，禁止 reset、clean、stash、覆盖、
    回退或混入无关提交。
  - commit、push、merge、rebase、Tag 和删分支等共享历史操作必须获得用户明确授权。
  - 强制修改前审计分支、HEAD、工作区和远端差异；逐文件暂存并区分代码、配置、资产、
    模型及生成数据。
  - 细化修改记录字段、验证等级、合并审计、发布说明和 GitHub 分支保护要求。
- `BUMI3_SONIC_修改记录.md`：新增本节，记录规则修改的来源、范围、理由和验证边界。

### 修改理由与兼容性边界

- 本仓库后续主要由 Agent 执行修改，仅定义分支用途不能防止覆盖用户工作区、误提交数据、
  未经批准改写历史或夸大验证结果，因此将工作区保护、权限边界、提交审计和发布证据纳入
  同一套规则。
- 本次只修改 Agent 工作规范和记录文档，不修改 SONIC/BUMI3/G1/H2 的代码、配置、机器人
  资产、训练数据、Checkpoint、网络结构或运行行为。
- 回滚时只需撤销本节和 `agent.md` 对应规则文本；不会影响任何模型或训练产物。

### 实际验证结果

- 修改前确认当前分支为 `feature/bumi-native-sonic-full-training`、起始工作区干净且已跟踪
  同名 origin 分支。
- `git diff --check`：通过。
- 未运行 Python、Isaac Sim、动作 replay 或训练测试：本次没有修改任何代码、配置或资产，
  这些运行级验证与文档规则修改无直接关系，因此不将文档检查描述为功能验证。

## 2026-08-27：接入五集合 Robot+SMPL 高质量训练数据

### 修改文件与内容

- 新增 `gear_sonic/tools/prepare_bumi3_sonic_dataset.py`：读取四个配对数据集和
  Mine-only 数据集；依照源 `joint_names` 将 BUMI3 qpos 重排到当前 SONIC MJCF，
  输出 30Hz robot motion-lib PKL；将 SMPL `pose_aa/transl/smpl_joints` 同步转换到
  50Hz；执行 3261/3162/99 计数、全量维度、有限值、配对、四元数和 SHA256 校验。
- 新增 `gear_sonic/tools/test_prepare_bumi3_sonic_dataset.py`：覆盖 MJCF 顺序解析、
  名称重排、axis-angle 生成以及与 SONIC 相同的 30Hz→50Hz 末帧排除时间网格。
- `validate` 读取包含输出路径和运行时帧数的扩展 manifest 时，只提取
  `SampleRecord` 契约字段，避免元数据扩展字段被误传给数据类构造函数。
- robot/SMPL 单文件先写入带 PID 的隐藏临时文件，joblib 完成后再原子替换为目标
  文件名；中断恢复不会把半文件误判为已完成。
- 锁定源 BUMI 文件实际携带的 MJCF SHA256
  `482138b437dbdabd6171fa8d44b55db5d7125a228c95b69ce3d1159cafe8537c`，并将其与
  当前 SONIC BUMI3 MJCF 指纹分别写入 provenance。两者不相同，因此只允许通过
  每段文件的 21 个 `joint_names` 做名称集合验证和显式重排，禁止按源列位置直拷。
- 实施前复核发现 `bumi.urdf` 的修改时间晚于验证脚本，工作区已经存在一组未记录
  的后续碰撞参数。为保留用户现有修改，本轮不改 URDF，只同步更新
  `validate_bumi3_integration.py` 的锁定值：左右 leg-roll 为 origin Z `-0.08`、
  radius `0.045`、length `0.1`；左右 knee 为 origin Z `-0.0694694`、radius
  `0.046`、length `0.15`。base 圆柱和四个无碰撞 link 保持不变。

### 修改理由与兼容性

- 机器人源文件保留 30Hz，由 motion-lib 在 FK 加载时统一转换到目标 50Hz；SMPL
  当前加载器只会重采样 pose，不能同步处理 joints/transl，因此三项必须离线共同
  转成 50Hz，且目标帧数必须与机器人运行时帧数完全一致。
- 机器人源数据采用另一份 MJCF 生成，不能按列位置假设关节顺序；转换器只信任
  每个文件的 `joint_names`，并要求其名称集合与当前 BUMI3 MJCF 精确相等。
- 工具仅新增离线数据入口，不修改 G1/H2 数据、训练网络、奖励、termination、
  domain randomization 或 checkpoint 行为。
- 碰撞验证器调整只是追踪当前较新的 URDF 工作区状态，不把碰撞参数回退为日志中
  较早的缩小版本；对应资产仍需通过实际 Isaac 导入和训练 smoke。

### 实际验证结果

- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m compileall -q gear_sonic`：通过。
- `gear_sonic/tools/validate_bumi3_integration.py`：通过；当前较新的 URDF 碰撞值、
  21 DoF/22 bodies、双编码器维度、50Hz 控制契约和 BUMI3 映射均通过静态及
  Isaac 配置导入检查；本次未请求该脚本的动作 smoke。
- `python -m pytest -q gear_sonic/tools/test_prepare_bumi3_sonic_dataset.py`：4 项通过；
  覆盖当前 MJCF、名称重排、轴角生成和目标帧时间网格。
- `git diff --check`：通过。
- 本地 Isaac Lab 环境未安装独立 `ruff` 可执行文件，因此没有把 Ruff 描述为已运行；
  将在 noetix 的正式训练环境再次检查可用性。
- noetix 全量数据转换及 SONIC 单卡/八卡训练 smoke 已于本节后续“服务器落盘与
  正式规模训练可行性验证”中完成并记录。正式 100k 训练未启动。

## 2026-08-27：服务器落盘与正式规模训练可行性验证

### 1. noetix 2TiB 数据盘

- 操作前再次确认 `/dev/vdb` 为精确的 2TiB 空盘，无分区、文件系统、挂载点、
  残留签名、占用进程或 `/etc/fstab` 条目；系统盘是 `/dev/vda2`，未对其执行
  分区或格式化操作。
- 为 `/dev/vdb` 创建 GPT 和单个 ext4 分区 `/dev/vdb1`，卷标为 `SONIC_DATA`，
  reserved blocks 为 0；实际 UUID 为
  `68f60019-f39c-44d1-8e14-320d25755dd6`。
- `/etc/fstab` 使用 `defaults,noatime,nofail 0 2` 挂载到 `/data`；`findmnt`、
  `findmnt --verify`、`df -hT` 和写入/删除测试均通过。原 fstab 备份为
  `/etc/fstab.codex-before-sonic-data`。
- 数据固定放在 `/data/sonic_bumi3/datasets/hq_all_v1`，训练和 smoke 分别放在
  `/data/sonic_bumi3/runs`、`/data/sonic_bumi3/smoke`，日志放在
  `/data/sonic_bumi3/logs`。全部数据和 smoke 验证结束时数据盘使用约 6.2GiB，
  仍有约 2.0TiB 可用；其中 datasets 约 5.5GiB、smoke 约 737MiB。

### 2. 服务器间直传和源数据校验

- SMPL 服务器直接向 noetix 推送 2,202 个 motion/curation 文件，共
  `1,503,420,296` bytes；BUMI 服务器直接推送四个配对集合 3,174 个文件，共
  `418,741,956` bytes，以及 Mine 集合 103 个文件，共 `17,722,804` bytes。
- 数据没有经过本地机器，也没有传 WAV 或 35 维音乐特征。传输使用临时受限
  SSH 公钥和可续传 rsync；完成后源端私钥和 noetix 临时授权条目均已删除，
  密码未写入仓库、脚本或日志。
- 源端/目标端逐文件排序 SHA256 清单的聚合指纹完全一致：SMPL 为
  `e934012a7c4b81adaa69d821df147fe2024ca3eda320549a32acefcdfa9bd23d`，
  四个 BUMI 配对集合为
  `46f04748306a8b9e473525394c13ceda10972bb4555d560a2ca509be49a4e25c`，
  Mine 为 `7cebe8f2404e271900760cac18dbb8bfaef90aa1bb2ca938c61cef52b78ff123`。
- 目标端逐文件清单保存在 `meta/source_smpl_files.sha256`、
  `meta/source_bumi_files.sha256` 和 `meta/source_mine_files.sha256`。

### 3. 代码、LFS 与转换产物

- noetix 仓库固定在提交 `b3cd0699a04ac31aef0a1f2ce76b8e06082ae30f`；分支为
  `feature/bumi-native-sonic-full-training`，拉取后工作区干净。
- BUMI3 的 22 个 STL 和 `human/human_joints_info.pkl` 共 23 个 LFS 文件均不再是
  指针，并逐文件通过“实体 SHA256 等于 Git LFS OID”检查。全仓库 `git lfs fsck`
  仍会报告两个与 BUMI3 无关且未实体化的大文件，因此不把全仓库 fsck 结果误写成
  BUMI3 资产失败。
- noetix 的 `compileall` 和转换器 4 项 pytest 均通过；本地完整
  `validate_bumi3_integration.py` 再次通过实际 Isaac Sim 配置导入、Hydra 组合、
  资产追溯、映射、执行器、碰撞和双编码器维度检查。
- 全量 `build` 后又独立运行一次 `validate`，两次均通过。最终是 3,261 个 robot、
  3,162 个 SMPL、3,162 个配对、99 个 Mine-only；所有 key 唯一，Mine-only 的
  `dataset=mine`、key 前缀为 `mine__`，并且 `smpl_file=null`。
- robot motion-lib 保持 30Hz，SMPL 的 pose/transl/joints 全部共同离线转为 50Hz，
  两者运行目标均为 50Hz。`meta/SHA256SUMS` 共 6,425 行，执行
  `sha256sum -c` 全部通过。
- 关键元数据指纹：`SHA256SUMS` 为
  `2fecb9ed7f70430c8d86a9b261c3c4d3862e032b31fb7b0dc06cfceffbf01c99`，
  `manifest.jsonl` 为
  `5871ec25ff70786763bb31d0f70b177bed6b278e8b7abdadec46e8b2020593b6`，
  `provenance.json` 为
  `e21d9572ee922a7db7d524e0197369f921f8b5b0dbef546db5faf24a8f3930cd`。

### 4. 实际训练 smoke

所有 smoke 都使用同一份正式数据、`checkpoint=null`、`auto_load_latest=False`、
`resume=False` 从零初始化。未使用 smoke 权重作为后一档 smoke 的初始化；最后的
checkpoint 重载是单独的兼容性检查。

| 阶段 | 结果 | 关键证据 |
|---|---|---|
| 单卡、64 env、10 iterations | 通过 | 完成 15,360 timesteps 和 10 次 PPO 更新；无 OOM、Traceback、NaN/Inf；首批实际采到 `mine__...` |
| 八卡、每 rank 512 env、100 iterations | 通过 | 8 个进程正常退出；9,830,400 timesteps；约 41k steps/s；每卡约 10.6–11.0GiB；无 OOM、NCCL timeout、Traceback、NaN/Inf |
| 八卡、每 rank 4,096 env、100 iterations | 通过 | 8 个进程正常退出；78,643,200 timesteps；稳定约 195k–202k steps/s；每卡约 20.3–20.9GiB、约 86–90% GPU 负载；无 OOM、NCCL timeout、Traceback、NaN/Inf |

- 4,096/rank 已完整通过，因此没有执行 2,048 或 1,024 回退，正式 `num_envs`
  取 4,096。
- 512/rank 和 4,096/rank 的日志均显示八个 rank 各自初始化 Robot（内部兼容键名
  `g1`）与 SMPL encoder，并在抽样 motion key 中实际出现 Mine-only。Mine 数据没有
  SMPL 文件，因此只能走 Robot encoder；配对数据同时提供两个 encoder 的输入。
- 4,096/rank 的 `last.pt` 在 step 50 和 100 均成功保存，大小约 368MiB。直接加载
  得到 global step 100、45 个 policy tensors、17 个 value tensors，并包含 optimizer、
  LR scheduler 和 env state。随后通过 SONIC 训练入口打印
  `Loaded checkpoint from step 100`，在单卡 64 env 上再完成 1 次 PPO 更新，证明
  不只是 pickle 可读，模型权重形状和加载路径也兼容。
- noetix 的 Isaac Sim 在纯 headless 启动时每 rank 会打印
  `VkResult: ERROR_INCOMPATIBLE_DRIVER` 和图形插件不可用；4,096/rank 日志共 24 次。
  这是当前服务器 Vulkan/渲染环境告警，不影响 PhysX/CUDA 无渲染训练：三档 smoke
  均继续完成，八卡负载稳定且进程以 0 退出。若未来需要相机或渲染，必须先修复
  Vulkan ICD/驱动环境，不能用本次 headless 训练通过替代渲染验证。

### 5. 正式训练边界

- 本轮没有启动 100k 正式训练，也没有执行 ONNX 导出；只证明当前提交、当前数据和
  当前 noetix 软硬件环境可从零训练到 4,096 env/rank。
- 正式训练必须继续使用 `checkpoint=null`、`auto_load_latest=False`、`resume=False`，
  不得添加 smoke checkpoint。正式命令的 `num_envs=4096` 已由完整 100 iterations
  实测确定，其余网络、PPO、环境和数据参数保持活跃配置不变。

## 2026-08-26：将误用的 BUMI2 集成完整迁移为 BUMI3

### 1. 修改原因与处理原则

- 用户确认实际训练机器人是 BUMI3，此前 BUMI2 机器人资产和参数选择错误。
- 本次不是简单字符串改名：BUMI2 的 URDF/MJCF、质量惯量、执行器上限、KP/KD、armature 和 action scale 全部废弃，按指定 BUMI3 参考目录重新复制和实现。
- 旧 BUMI2 资产目录、机器人模块、Hydra 实验入口和验证脚本已从当前集成中删除；这些文件此前均未提交，仍可从原参考仓库重新复制恢复。
- SONIC 的网络主体、Robot/SMPL 双编码器、PPO、critic、trainer、50 Hz 控制契约和 G1/H2 支持继续保留。

### 2. BUMI3 参考来源与版本状态

唯一机器人参数来源：

`/home/weili/legged_lab/source/NoetixRobot/NoetixRobot/assets/robots/bumi3/`

参考仓库当前提交：`d555c76e5977af66ef55a104b98e1be486349996`。

参考仓库当前存在未提交修改，因此本集成明确采用“当前工作区版本”，不回退到提交版：

- `bumi.py` SHA256：`74aaeca9da615c50e3749e4f103bbf713b83443d9cb16fab08edfd320227c03e`。
  - 相对当前 Git HEAD 的有效差异：arms `velocity_limit_sim` 从 `30` 改为 `12`；本集成采用 `12`。
- `urdf/bumi.urdf` SHA256：`174c1747019ced64267e74244bf89f3746856c90c30f88e4f162582ebc486476`。
- `mjcf/bumi3.xml` SHA256：`041c81e8176c7f375302796deca28b141891a3c097d8e341e8d967b735466edf`。
  - 相对当前 Git HEAD 的有效差异：`waist_yaw_joint` axis 使用 `0 0 1`；左右 `arm_roll` 限位分别为 `[-0.14, 1.94]`、`[-1.94, 0.14]`。这些当前值已原样纳入。
- `meshes/BUMI(1)_5.26.urdf` 和 `meshes/BUMI_V3.0_260119GG.urdf` 也有参考工作区修改；mesh 目录按当前文件集合逐文件复制并由验证脚本比较 SHA256。

验证脚本锁定上述三个关键 SHA。参考文件若再次变化，验证会明确失败，要求重新审计，而不会静默继续使用旧参数。

### 3. 机器人资产迁移

- 新增 `gear_sonic/data/assets/robot_description/urdf/bumi3/bumi.urdf`：复制 BUMI3 权威 URDF，只把 mesh 相对路径从 `../meshes/` 改为 `../../meshes/bumi3/`；质量、质心、惯量、碰撞、joint origin/axis/limit 均不修改。
- 新增 `gear_sonic/data/assets/robot_description/meshes/bumi3/*`：复制当前 BUMI3 `meshes/` 全部 26 个文件，包括 22 个 STL 和 4 个附属 URDF。
- 新增 `gear_sonic/data/assets/robot_description/mjcf/bumi3.xml`：复制当前 BUMI3 MJCF，只把 `meshdir` 从 `../meshes/` 改为 `../meshes/bumi3/`；body/joint 名称、轴、限位和全部数值均不修改。
- 删除本次集成产生的 `urdf/bumi2/`、`meshes/bumi2/` 和 `mjcf/bumi2.xml`。
- `.gitignore` 的资产白名单由 BUMI2 改为 BUMI3；本地 CSV、motion、训练输出和其他 `data/` 内容仍保持忽略。
- `.gitattributes` 对上述 BUMI3 URDF 设置路径级 `whitespace=cr-at-eol`：参考文件原生使用 CRLF，该规则只让 Git 将 `CR` 视为换行的一部分，不转换文件内容，从而同时保持参考字节与有效的代码 whitespace 检查。

### 4. BUMI3 原生机器人配置

新增 `gear_sonic/envs/manager_env/robots/bumi3.py`，不依赖 `NoetixRobot` Python 包，使用项目内 `DelayedImplicitActuatorCfg`。

基础配置与参考 `Bumi_CFG` 一致：

- floating base，contact sensors 开启，cylinder-to-capsule 开启。
- self collision 开启，solver position/velocity iterations 为 `8/4`。
- 初始 root position 为 `(0, 0, 0.65)`，soft joint position limit factor 为 `0.9`。
- 腿部初始姿态：左右 hip pitch `-0.1495`、knee `0.3215`、ankle pitch `-0.1720`，其余腿关节为零。
- 上身初始姿态：左 arm roll `0.3`、右 arm roll `-0.3`，腰和其他手臂关节为零。
- 四组执行器 delay 均为 `min_delay=0, max_delay=4`。

实际执行器参数如下，其中 effort/velocity 分别为仿真力矩和速度上限：

| 关节组 | effort | velocity | KP | KD | armature |
|---|---:|---:|---:|---:|---:|
| leg yaw | 12 | 12 | 20 | 1.0 | 未启用 |
| leg roll | 50 | 12 | 45 | 3.0 | 未启用 |
| leg pitch | 50 | 12 | 45 | 3.0 | 未启用 |
| knee pitch | 50 | 12 | 45 | 2.0 | 未启用 |
| waist yaw | 27 | 9 | 53 | 3.4 | 未启用 |
| ankle pitch | 9 | 12 | 8 | 0.5 | 0.012574 |
| ankle roll | 9 | 12 | 8 | 0.5 | 0.009608 |
| arm pitch/roll/yaw/elbow | 4 | 12 | 8 | 0.4 | 未启用 |

`BUMI3_ACTION_SCALE` 未写死，继续按 `0.25 * effort_limit_sim / stiffness` 生成：

- leg yaw：`0.15`。
- leg roll/pitch/knee：`0.2777777777777778`。
- waist yaw：`0.12735849056603774`。
- ankle pitch/roll：`0.28125`。
- arms/elbows：`0.125`。

### 5. 关节、body 顺序与映射

BUMI3 MuJoCo DoF 顺序来自当前 MJCF：

`[waist_yaw, l_arm_pitch, l_arm_roll, l_arm_yaw, l_elbow_pitch, r_arm_pitch, r_arm_roll, r_arm_yaw, r_elbow_pitch, l_leg_pitch, l_leg_roll, l_leg_yaw, l_knee_pitch, l_ankle_pitch, l_ankle_roll, r_leg_pitch, r_leg_roll, r_leg_yaw, r_knee_pitch, r_ankle_pitch, r_ankle_roll]`，各项完整名称均带 `_joint`。

BUMI3 Isaac Lab DoF 顺序来自参考 `bumi.py:joint_names`：

`[l_leg_pitch, r_leg_pitch, waist_yaw, l_leg_roll, r_leg_roll, l_arm_pitch, r_arm_pitch, l_leg_yaw, r_leg_yaw, l_arm_roll, r_arm_roll, l_knee_pitch, r_knee_pitch, l_arm_yaw, r_arm_yaw, l_ankle_pitch, r_ankle_pitch, l_elbow_pitch, r_elbow_pitch, l_ankle_roll, r_ankle_roll]`，各项完整名称均带 `_joint`。

映射由名称生成并在导入时验证：

- Isaac Lab → MuJoCo DoF：`[2, 5, 9, 13, 17, 6, 10, 14, 18, 0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20]`。
- MuJoCo → Isaac Lab DoF：`[9, 15, 0, 10, 16, 1, 5, 11, 17, 2, 6, 12, 18, 3, 7, 13, 19, 4, 8, 14, 20]`。
- Isaac Lab → MuJoCo body：`[0, 3, 6, 10, 14, 18, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21]`。
- MuJoCo → Isaac Lab body：`[0, 10, 16, 1, 11, 17, 2, 6, 12, 18, 3, 7, 13, 19, 4, 8, 14, 20, 5, 9, 15, 21]`。
- lower-body MuJoCo indices：`[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]`。

### 6. SONIC 注册、converter 与数据接口

- `robots/__init__.py` 和 `modular_tracking_env_cfg.py`：只注册新 robot type `bumi3`；G1/H2 条目不变。
- `order_converter.py`：将原错误 converter 替换为 lazy-import `Bumi3Converter`；factory 支持 `bumi3`，默认仍为 G1。
- `commands.py`：BUMI3 lower-body index 检查固定为 `9..20`，通用默认仍为 G1 `range(12)`。
- `motion_lib_base.py`：wrist indices 继续可配置；BUMI3 设为空列表，G1 默认 `[19,20,21,26,27,28]` 不变。
- BUMI3 实验的 motion/SMPL 路径保持 `null`，必须由 CLI 指定，避免把任何 G1 或未核验 BUMI3 数据静默用于训练。

### 7. Robot + SMPL 双编码器 SONIC

- `all_mlp_v1_no_teleop.yaml`：只实例化内部键名 `g1` 的 Robot Encoder 和 `smpl` Encoder；MLP、FSQ、token、dynamic/kinematic decoder 不变。
- `unitoken_robot_smpl_noz.yaml`：只含 encoder index、Robot multi-future joint/anchor 和 SMPL multi-future local joint/root 输入。
- `g1_recon_and_smpl_latent.yaml`：只含 Robot reconstruction、Robot-SMPL latent 和 reencoded SMPL-Robot latent 三项损失。
- 内部 `g1` 键保留用于 SONIC 网络/checkpoint 兼容，不代表使用 G1 机器人资产。

### 8. BUMI3 专用训练配置

`sonic_bumi3.yaml` 保持：

- `sim_dt=0.005`、`decimation=4`，控制频率 50 Hz。
- `target_fps=50`，actor/critic history length 10。
- Robot/SMPL future frames 均为 10，步长分别 `0.1/0.02`。
- action dim 21，motion library asset 为 `bumi3.xml`，robot type 为 `bumi3`。
- 沿用 SONIC 14-body tracking subset、奖励函数和 termination 实现，只使用 BUMI3 当前模型中存在的 body 名称。

用户后续明确授权的 BUMI3 动力学差异：

- 质量 scale：选中 waist 和左右 elbow，范围 `[0.8, 1.2]`。
- 踝 armature：在各自 BUMI3 名义值上按 `[0.9, 1.1]` scale；pitch 实际范围 `[0.0113166, 0.0138314]`，roll 实际范围 `[0.0086472, 0.0105688]`。
- 全部 21 关节 KP/KD：分别按 `[0.8, 1.2]` scale，每次 reset 从名义默认值重新采样。
- 力矩限制奖励：全部关节，`limit_ratio=0.85`、`weight=-0.01`，惩罚平方超额和。

### 9. 验证脚本

`gear_sonic/tools/validate_bumi3_integration.py` 已迁移并增强：

- 锁定当前参考 `bumi.py`、URDF、MJCF SHA256。
- 比较 URDF/MJCF 允许的 mesh 路径改动以及 mesh 文件 SHA256。
- 验证 21 DoF、22 bodies、名称唯一、URDF/MJCF 全关节 axis/range 一致。
- 验证 DoF/body mapping 完整排列和 round trip。
- 验证完整初始姿态、rigid/solver 参数、全部 BUMI3 effort/velocity/KP/KD/armature/delay。
- 在隔离的模块命名空间内直接执行当前参考工作区的 `bumi.py`，逐字段比较本地 `BUMI3_CFG` 与参考 `Bumi_CFG`；这样不仅验证手工抄录的期望值，也验证本地实现与当前参考 Python 配置本身一致。
- 由公式验证 action scale，不依赖复制的常量。
- 验证质量、armature、KP/KD 随机化和 torque-limit reward 的 resolved 配置。
- 验证双编码器、无 Teleop 活跃输入/损失、FSQ 64、actor proprioception 690、tokenizer 1262、critic 1245、dynamic decoder `754 -> 21`。
- 保留可选 `--smoke` 环境 reset/step 和 NaN/Inf 检查。

### 10. 当前实际验证结果

- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m compileall -q gear_sonic`：通过。
- 静态 BUMI3 资产、拓扑、轴/限位、Hydra 和维度验证：通过。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python gear_sonic/tools/validate_bumi3_integration.py`：通过；实际导入 BUMI3 `ArticulationCfg` 并验证全部机器人参数和映射。
- 当前参考 `bumi.py` 直接执行与本地 `BUMI3_CFG` 逐字段一致性检查：通过；覆盖执行器 joint selector、effort/velocity、KP/KD、armature、delay、初始关节姿态和 action scale。
- MuJoCo 直接加载 `bumi3.xml`：通过；模型得到 `nq=28`、`nv=27`、`nu=21`、`nbody=23`（其中 `nbody` 包含 MuJoCo world body，对应机器人自身 22 bodies）。
- 活跃代码和配置负向搜索：`gear_sonic`、`.gitignore`、`agent.md` 中不存在 BUMI2 标识；`sonic_bumi3` 活跃配置中不存在 Teleop encoder、Teleop tokenizer input、Teleop auxiliary loss 或 G1 wrist index。
- resolved 数值：`sim_dt=0.005`、`decimation=4`、control frequency `50.0`、target FPS `50`、action dim `21`、FSQ `64`、actor proprioception `690`、tokenizer `1262`、critic `1245`、dynamic decoder `754 -> 21`。
- 包含全部未跟踪新资产/配置的临时索引 `git diff --cached --check`：通过；在本次碰撞修改前统计为 45 files changed、5319 insertions、7 deletions，真实 Git 暂存区未改动。
- 现有测试检索只发现 `gear_sonic/tests/test_input_readers.py`，没有覆盖机器人配置、converter、mapping 或 Hydra compose 的相关单元测试；1-env ManagerBasedRLEnv reset/step 和训练 smoke 也尚未运行。当前 BUMI3 配置要求用户提供相互匹配的 robot motion 与 SMPL motion；本轮未擅自选择、生成或转换数据。本机先前还存在 URDF importer extension 依赖冲突，不能把配置导入检查表述为环境 spawn 已通过。

## 2026-08-26：修复 BUMI3 双编码器 ONNX 导出硬编码 Teleop

### 问题

- `gear_sonic/eval_agent_trl.py` 的 universal-token ONNX 导出入口原先无条件导出 `smpl`、`g1`和 `teleop` 三个 encoder。
- BUMI3 活跃网络只注册 `g1` Robot Encoder 和 `smpl` Encoder，因此原入口会在读取 `encoder_input_features["teleop"]` 时触发 `KeyError: 'teleop'`，并在导出组合 encoder 和 decoder 之前异常退出。

### 修改内容与理由

- 删除 `smpl/g1/teleop` 三段硬编码调用，改为遍历当前模型的 `actor_module.encoders_to_iterate`。
- 导出前检查 encoder 列表非空，并确认每个 encoder 都已注册 `encoder_input_features`，避免缺失配置被静默忽略。
- 输出文件名仍为 `_<encoder_name>.onnx`；BUMI3 自动生成 `_g1.onnx` 和 `_smpl.onnx`，不再尝试生成 `_teleop.onnx`。
- 组合 `_encoder.onnx` 和 `_decoder.onnx` 继续导出，并复用已校验的 `actor_module` 引用。

### 兼容性

- BUMI3 双编码器模型只导出 `g1/smpl`，修复确定的 Teleop `KeyError`。
- 原 G1 SONIC 的 `encoders_to_iterate` 仍包含 `g1/teleop/smpl`，因此依然会导出三个 encoder 模型，不删减原有能力。
- 不修改训练网络、checkpoint、观测维度、FSQ、decoder 或 PPO 参数。

### 实际验证结果

- `compileall` 和 `git diff --check` 通过。
- Hydra 解析后，BUMI3 的 encoder 顺序为 `['g1', 'smpl']`，原 G1 SONIC v1.1 为 `['g1', 'teleop', 'smpl']`，证明动态导出同时覆盖双编码器和三编码器配置。
- 使用当前 BUMI3 resolved 网络和随机初始化权重实际执行同一导出逻辑，精确生成四个模型：
  - `_g1.onnx`：输入 `[1, 1170]`，输出 `[1, 21]`；
  - `_smpl.onnx`：输入 `[1, 1470]`，输出 `[1, 21]`；
  - `_encoder.onnx`：输入 `[1, 1263]`，输出 `[1, 64]`；
  - `_decoder.onnx`：输入 `[1, 754]`，输出 `[1, 21]`。
- 四个模型均通过 ONNX checker 和 ONNX Runtime CPU 有限值推理；组合 encoder 的 `g1/smpl` 两个 selector 分支均通过。
- 导出目录中没有 `_teleop.onnx`，因此 BUMI3 不再进入会触发 `KeyError: 'teleop'` 的路径。
- 未运行“真实训练 checkpoint + 完整仿真环境”的 `eval_agent_trl.py` 全流程：当前没有本次从零训练生成的 BUMI3 checkpoint 及成对动作数据。本轮已验证的是本次修改直接影响的网络构建与 ONNX 导出边界。

## 2026-08-26：为爬行和跪地动作补齐 BUMI3 简化碰撞体

### 1. 修改原因和范围

- SONIC 训练动作包含爬行、跪地等非站立接触，原 BUMI3 URDF 的 `base_link`、左右 `leg_roll_link` 和左右 `knee_pitch_link` collision 被注释，无法产生相应身体接触。
- 用户明确要求上述 link 使用圆柱简化碰撞，并取消左右 `leg_pitch_link`、左右 `leg_yaw_link` 的碰撞。
- 本轮只修改 SONIC 使用的 `gear_sonic/data/assets/robot_description/urdf/bumi3/bumi.urdf`；参考 `legged_lab`、BUMI3 MJCF、mesh、质量、质心、惯量、visual、joint origin/axis/limit、执行器和训练配置均未修改。

### 2. 圆柱尺寸和坐标依据

使用当前仓库 BUMI3 STL 的局部轴对齐包围盒确定圆柱长度和中心，圆柱轴均沿 link 局部 Z 轴。半径取横向主半宽并小幅取整，没有采用包围盒对角半径，避免爬行时碰撞体过度膨胀：

| Link | STL 局部 AABB extents | collision origin xyz | radius | length |
|---|---|---|---:|---:|
| `base_link` | `[0.101807, 0.096937, 0.131050]` | `[-0.0013853, 0, 0.065525]` | `0.052` | `0.13105` |
| `l/r_leg_roll_link` | `[0.115000, 0.083960, 0.156461]` | `[0, 0, -0.0402693]` | `0.058` | `0.1564614` |
| `l/r_knee_pitch_link` | `约 [0.09297, 0.10500, 0.255315]` | `[0.008475, 0, -0.0894694]` | `0.053` | `0.2553155` |

最终 URDF collision 策略：

- 圆柱 collision：`base_link`、`l_leg_roll_link`、`r_leg_roll_link`、`l_knee_pitch_link`、`r_knee_pitch_link`。
- 无 collision：`l_leg_pitch_link`、`r_leg_pitch_link`、`l_leg_yaw_link`、`r_leg_yaw_link`。
- 其余 link 的 collision 与当前 BUMI3 参考 URDF 保持一致。
- `BUMI3_CFG.spawn.replace_cylinders_with_capsules=True` 保持不变，所以文件层面是用户要求的 URDF cylinder，Isaac Lab 训练运行时会将这五个 cylinder 转成接触更平滑的 Capsule。

### 3. 验证脚本修改

- 修改 `gear_sonic/tools/validate_bumi3_integration.py` 的资产追溯逻辑：结构化比较本地与参考 URDF，并忽略且单独审计 collision 节点；由此证明除 mesh 相对路径和本次指定 collision 外，其他 URDF 结构完全未改。
- 新增五个圆柱的 link、origin、RPY、radius、length 和唯一 geometry 检查。
- 新增四个无碰撞 link 的零 collision 检查。
- 对其余全部 link 逐项比较本地和参考 collision，防止意外扩大修改范围。

### 4. 实际验证结果

- 使用 `trimesh` 读取 9 个相关 BUMI3 STL 并检查 bounds/centroid：通过；上述尺寸均来自当前 BUMI3 mesh。
- XML 资产追溯、碰撞策略、mesh 存在性、21 DoF 和 22 bodies 静态检查：通过，输出 `COLLISION_STATIC_VALIDATION=PASS`。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m compileall -q gear_sonic`：通过。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python gear_sonic/tools/validate_bumi3_integration.py`：通过；原 SONIC 时间、网络、映射、执行器和 Hydra 兼容性检查均保持通过。
- 使用 Isaac Lab `UrdfConverterCfg(replace_cylinders_with_capsules=True)` 实际执行 URDF→USD：通过；生成主 USD 共 14 个 collider，五个目标 link 均为 Capsule，四个禁用 link 均没有 collider，输出 `ISAAC_URDF_TO_USD_COLLISION_SMOKE=PASS`。
- 第一次 USD 检查只使用普通 `stage.Traverse()`，没有遍历 instance proxy，因此误报缺少 `base_link` collider；生成的 USD 本身成功。改用 `Usd.TraverseInstanceProxies()` 后重新断言并通过。
- 未运行含动作数据的 1-env reset/step 或爬行/跪地 replay：BUMI3 robot-motion/SMPL 数据路径仍按训练安全要求保持 `null`，本轮没有擅自选择训练数据。因此碰撞资产和 Isaac 导入已验证，但具体动作的接触质量仍需在指定动作数据上回放确认。
- 该轮完成时的完整临时索引 `git diff --cached --check`：通过；当时累计统计为 45 files changed、5461 insertions、7 deletions，真实 Git 暂存区未改动。

## 2026-08-26：按训练需求缩小 BUMI3 圆柱碰撞体

### 1. 用户指定参数

本次参数是用户对上一版 STL 包围盒初始方案的明确覆盖，不再把圆柱描述为完整包络 mesh。碰撞 link 集合和局部 Z 轴方向不变，只修改以下尺寸与位置：

| Link | 修改项 | 新值 | 保持不变项 |
|---|---|---|---|
| `base_link` | length | `0.12` | radius `0.052`，origin `[-0.0013853, 0, 0.065525]` |
| `l/r_leg_roll_link` | radius、length、origin | `0.03`、`0.08`、`[0, 0, -0.02]` | RPY `[0, 0, 0]` |
| `l/r_knee_pitch_link` | radius、length | `0.025`、`0.13` | origin `[0.008475, 0, -0.0894694]`，RPY `[0, 0, 0]` |

左右 `leg_pitch_link` 和左右 `leg_yaw_link` 继续保持无 collision；其他 link 的碰撞不变。`replace_cylinders_with_capsules=True` 也不变，因此 Isaac Lab 运行时仍会把五个 URDF cylinder 转为 Capsule。

### 2. 修改文件和原因

- `gear_sonic/data/assets/robot_description/urdf/bumi3/bumi.urdf`：应用用户指定的新圆柱参数；未修改 visual、inertial、joint、mesh 路径或其他 collision。
- `gear_sonic/tools/validate_bumi3_integration.py`：同步更新五个圆柱的期望值，并明确这些值来自用户训练调参，而不是继续声称完全来自 STL AABB。
- `BUMI3_SONIC_修改记录.md`：记录覆盖关系、最终参数、未改范围和测试证据。

### 3. 兼容性边界

- 不修改 BUMI3 MJCF、执行器、action scale、域随机化、奖励、termination、SONIC 网络、控制频率或数据路径。
- 不修改参考 `/home/weili/legged_lab`。
- 21 DoF、22 bodies、关节/body 顺序和 Isaac Lab/MuJoCo mapping 不变。

### 4. 实际验证结果

- XML 资产追溯、五个圆柱精确参数、四个无碰撞 link、mesh 存在性、21 DoF 和 22 bodies 静态检查：通过，输出 `COLLISION_STATIC_VALIDATION=PASS`。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m compileall -q gear_sonic`：通过。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python gear_sonic/tools/validate_bumi3_integration.py`：通过；SONIC 时间参数、网络维度、执行器和映射仍保持原值。
- 使用 `UrdfConverterCfg(replace_cylinders_with_capsules=True)` 强制重新生成 USD：通过；随后通过 `Usd.TraverseInstanceProxies()` 读取实际碰撞 prim，五个目标 link 的 Capsule origin/radius/height 与本节表格逐项一致，轴均为 Z。
- 生成 USD 共 14 个 collider；左右 `leg_pitch_link`、左右 `leg_yaw_link` 仍为零 collider，输出 `ISAAC_TUNED_COLLISION_SMOKE=PASS`。
- 未运行带爬行/跪地动作数据的环境 replay，原因仍是 BUMI3 robot-motion/SMPL 路径保持 `null`，本轮未擅自选择数据。

## 2026-08-31：修正双肘终止与 BUMI3 sim2sim 参考状态、锚点及碰撞语义

### 1. 起点、授权范围和保护措施

- 分支：`feature/bumi-native-sonic-full-training`。
- 修改前 HEAD：`b1c3606ce96f00a01745cb8382f8bfa0b9b4d780`。
- 用户明确要求：`ee_body_pos` 只检查双肘并保留 `0.12 m`；降低 sim2sim 固定悬空回退高度；修复 Robot Encoder 把 root quaternion 当成 `waist_yaw_link` 锚点姿态的问题；解释终止条件和 Robot/SMPL 异常配对。
- 工作树在本轮开始前已有 BUMI3 资产、数据准备、sim2sim 和记录文件的未提交改动；本轮没有 reset、clean、stash、checkout、rebase，也没有提交或推送。

### 2. 训练 termination 修改

- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bumi3.yaml`：从 `ee_body_pos.params.body_names` 删除左右 `ankle_roll_link`，只保留 `l_elbow_pitch_link`、`r_elbow_pitch_link`；threshold 继续为 `0.12`，adaptive、down threshold、root height threshold 和 termination 实现均未修改。
- 左右脚仍由独立的 `foot_pos_xyz` 检查，不再同时进入 `ee_body_pos`。该改动减少重复截断，但不被描述为训练失败的唯一根因。
- `gear_sonic/tools/validate_bumi3_integration.py`：同步锁定 resolved 双肘列表，防止 Hydra 配置回退到双脚+双肘。

### 3. sim2sim 参考状态和初始化高度

- `ReferenceMotion` 新增可选 `root_position_world`。SONIC PKL/NPZ 会读取 `root_trans_offset/root_pos/root_position/root_translation`，含 `qpos` 的 NPZ 可回退到 `qpos[:, :3]`；CSV 若提供 `body_pos.csv` 则读取第一个 body 的位置。
- reset 不再固定写入 `[0,0,0.65] + 默认关节 + 零速度`，而是使用所选参考帧的 root position、root quaternion、21 关节位置和关节速度；浮动根速度由相邻 MuJoCo qpos 用 `mj_differentiatePos` 求得。
- `gear_sonic/config/sim2sim/bumi3_sonic.yaml` 的旧动作缺 root translation 时的回退位置改为 `[0,0,0.4744]`。真实 FineDance 首帧实际使用动作自带的约 `0.529/0.538 m`，不是强行改成回退高度。
- reset 的 10 帧 proprioception history 改为按 Isaac Lab `CircularBuffer` 首次 append 行为用当前状态复制填满，不再以 9 帧零状态开始。

### 4. Robot Encoder 锚点语义修复

- 训练端 `motion_anchor_ori_b_mf_nonflat` 实际读取 `TrackingCommand.root_rot_dif_l_multi_future`，其参考 quaternion 来自 MotionLib 中 `motion_anchor_body_index` 指向的 `waist_yaw_link`，不是浮动根 `base_link`。
- 旧 sim2sim 在 tokenizer 和起始 heading 对齐中直接使用 `motion.root_quat_wxyz`，因此腰部 yaw 非零时输入语义错误。
- 新实现逐帧把参考 root 和 21 关节写入一个独立 `MjData`，调用 BUMI3 MJCF FK，缓存 `waist_yaw_link` 的世界 quaternion；未来 10 帧 6D orientation 和起始 heading 对齐均使用该缓存。
- checkpoint/ONNX 只保存网络权重和张量接口，不保存动作文件、anchor body 名称或 FK 结果。本地 `model_step_070000.pt` 为 368 MiB，含 policy/value/optimizer/trainer state；`env_state_dict` 只有 motion-lib state。`model_step_070000_g1.onnx` 为 53 MiB、输入 1170、输出 21，因此部署端必须自行正确重建观测。

### 5. sim2sim 碰撞语义修复

- 真实 FineDance 首帧在旧 MJCF mesh collision 下 reset 即出现左右 `knee_pitch_link` 与 `ankle_roll_link` 自碰撞，穿透分别约 `24.62/24.87 mm`。首帧 observation 最大绝对值为 `1.08`，一次控制后碰撞冲击把该值推到约 `31`，证明问题不只是初始高度。
- 运行器现在仅在内存 `MjModel` 中关闭训练 URDF 未启用的 arm pitch、arm yaw、leg pitch、leg yaw mesh collision；把 base、左右 leg roll、左右 knee 的 mesh geom 改为训练 importer 实际使用的 capsule。
- capsule 的位置、半径和半长严格对应当前 URDF cylinder 与 `replace_cylinders_with_capsules=True`：base `[-0.0013853,0,0.065525], r=0.052, half=0.06`；leg roll `[0,0,-0.02], r=0.03, half=0.04`；knee `[0.008475,0,-0.0894694], r=0.025, half=0.065`。
- 原始 `bumi3.xml` 未改写；GUI 中上述五个单 geom 会显示成简化 capsule。该取舍避免改变 MotionLib 使用的参考 MJCF，同时优先保证 sim2sim 碰撞动力学接近训练 URDF。

### 6. Robot/SMPL 精确配对审计结论

- 在服务器 `/data/sonic_bumi3/datasets/hq_all_v2/built` 对 3162 个公开 Robot/SMPL 同名配对重新审计；99 个 Mine robot-only 动作没有 SMPL，不计入“异常配对”。
- 审计严格使用 MotionLib 的 `arange(0,duration,1/50)` 独占末帧时间网格、机器人 root+waist quaternion Slerp、BUMI3 `waist_yaw_link` FK 等价姿态，以及训练端 SMPL Y-up→Z-up 和 base rotation removal。3162 对全部帧数一致，没有时间轴长度错配。
- 以整段 `median(angle(inv(robot_waist) * smpl_root)) > 45°` 为初筛后得到 55 条，而不是旧的 root-only 近似统计 57 条。55 条中 51 条来自 AIST++、3 条 AIOZ-GDance、1 条 COMPAS；AIST++ 中 33 条为 `gBR` breakdance。
- 55 条中 41 条至少一侧中位倾角超过 45°：23 条双方均超过、13 条只有机器人 waist 超过、5 条只有 SMPL root 超过。这些主要是倒地、翻滚、breaking 等动作在人类 pelvis 与重定向机器人 base/waist 上的姿态差，不是整库仍存在统一 90° 坐标轴错误。
- 剩余 14 条双方中位倾角都不超过 45°，但多数已接近 30–45°；只有 `aistpp__gWA_sBM_cAll_d26_mWA4_ch08.pkl` 双方都低于 30°（约 23.4°/23.9°）而相对角中位数仍约 48.6°，应列为优先人工同步回放对象。该审计不自动删除或改写训练数据。

### 7. 修改文件及目的

- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bumi3.yaml`：`ee_body_pos` 仅双肘。
- `gear_sonic/config/sim2sim/bumi3_sonic.yaml`：旧动作固定回退根高降至 `0.4744 m`。
- `gear_sonic/utils/mujoco_sim/bumi3_sim2sim.py`：root translation 加载、参考状态 reset、正确 history 初始化、waist FK anchor、训练 URDF collision runtime override。
- `gear_sonic/tests/test_bumi3_sim2sim.py`：新增 root translation、参考 reset、waist FK、history 和 collision 回归测试。
- `gear_sonic/tools/validate_bumi3_sim2sim.py`：验证回退根高、FK anchor、collision override，并打印实际 reference reset 来源。
- `gear_sonic/tools/validate_bumi3_integration.py`：验证 resolved `ee_body_pos` 仅双肘。
- `docs/source/getting_started/bumi3_sim2sim.md`：说明 checkpoint/ONNX 边界、参考初始化、anchor FK、history 和碰撞契约。
- `BUMI3_SONIC_修改记录.md`：记录本轮原因、实现、实测数据、测试边界和回滚信息。

### 8. 实际运行的验证及结果

- `git diff --check`：通过。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m pytest -q gear_sonic/tests/test_bumi3_sim2sim.py gear_sonic/tools/test_prepare_bumi3_sonic_dataset.py`：`15 passed`；另有 1 个既有 invalid escape sequence DeprecationWarning。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python gear_sonic/tools/validate_bumi3_integration.py`：通过；resolved `sim_dt=0.005`、`decimation=4`、control 50 Hz、action 21、actor proprioception 690、dynamic decoder `754 -> 21`。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m compileall -q gear_sonic`：通过。
- 真实 `model_step_070000_g1.onnx` + `finedance__001_50fps.pkl` 执行 100 个控制周期：接口、1170/21 维度、FK anchor、碰撞覆盖和 NaN/Inf 检查通过；2 秒后 root height 仍降到 `0.066842 m`。
- 最后一项明确表明当前 70k checkpoint 仍未学会稳定跟踪；本轮已修复部署语义，但不能把有限值 smoke 误报为动作质量通过。训练曲线此前平均 episode 约 `0.79 s`、`ee_body_pos` 聚合触发约 70%，所以该 checkpoint 摔倒与训练质量一致，需要在新 termination 配置下做受控重训/对照实验后再导出新 ONNX。

### 9. 未执行项、风险和回滚

- 未在本轮停止、重启或续训服务器任务；代码配置改变不会追溯修改已有 checkpoint。
- 未自动筛除 55 条高差异配对，因为其中大量是用户需要的爬行、倒地和 breakdance；应先对 14 条非明显倒地项，尤其唯一双方低于 30° 的样本做 Robot/SMPL 同步可视化，再决定 quarantine。
- 未证明删除脚部重复 termination 单独即可解决训练。正确验证方法是同一初始权重/seed 做旧配置与新配置短程 A/B，记录左右脚、左右肘的独立误差和首次触发 body；当前聚合 `ee_body_pos` TensorBoard tag 无法反推是哪一个 body。
- 回滚范围是本节列出的配置、sim2sim、测试、验证和文档局部 diff；工作树无本轮 commit，不能使用破坏其他未提交修改的全局 reset。

## 2026-08-31：隔离 55 条异常配对、增加坐标启动门禁与 TensorBoard

### 1. 起点与授权边界

- 分支：`feature/bumi-native-sonic-full-training`。
- 修改前 HEAD：`b1c3606ce96f00a01745cb8382f8bfa0b9b4d780`。
- 用户明确要求：55 条 Robot/SMPL 异常配对不参与训练；确认原始/训练坐标契约；在服务器从头启动 8 卡训练；必须产生 TensorBoard event 文件。
- 本轮开始前工作树已有 BUMI3 数据、碰撞、sim2sim 和记录文件的未提交改动。本轮未执行 reset、clean、stash、checkout、rebase、commit 或 push，也未覆盖无关改动。

### 2. 55 条异常动作的精确隔离

- `sonic_bumi3.yaml` 在 `motion_lib_cfg.exclude_motion_keys` 中逐条保存 55 个完整 key：AIST++ 51 条、AIOZ-GDance 3 条、CoMPAS3D 1 条；不含 99 条 Mine-only 动作。
- `motion_lib_base.py` 新增 `exclude_motion_data_by_exact_keys`，在正则过滤、前缀删除和随机限量之前执行完整 key 匹配。它不复用 `remove_motion_keys` 的前缀语义，也不原地修改完整动作索引。
- MotionLib 会记录 requested、matched、missing、remaining。完整训练集的预期日志为 `requested=55, matched=55, missing=0, remaining=3206`；对单动作导出/回放则允许清单中其他 key 缺失并明确 warning，避免破坏既有评估入口。
- `validate_bumi3_integration.py` 锁定数量、唯一性、来源计数和排序后 SHA256 指纹 `808786f5202af4c8cef08c0aee8ff025468b99d3e3a5ade83f273e2d4aacfd88`。
- `test_prepare_bumi3_sonic_dataset.py` 新增精确匹配、无前缀误删、原字典不被修改、重复 key 拒绝测试。

### 3. 原始数据和训练输入的坐标契约

- 四个公开 Robot 库源契约为 `genmo.bumi_legacy_motion.v1`：世界上轴是 `+Y`。数据准备阶段对根四元数世界系左乘 `Rx(+90°)`，统一写成 SONIC/MuJoCo `+Z` 上轴；21 个关节按 BUMI3 MuJoCo actuator 名称重排；根高按当前 BUMI3 MJCF 双脚 FK 优化。Robot 文件仍以 30 Hz 保存，MotionLib 按独占末帧的时间网格插值为 50 Hz。
- Mine Robot 源契约为 `genmo.bumi_csv_qpos_xyzw.v1`：已经是 `+Z` 上轴，因此根姿态使用 identity 修正；关节顺序和根高仍按同一 BUMI3 契约校验。Mine 只有 Robot 动作，没有 SMPL 配对。
- 公开 SMPL 的 `pose_aa` 保留源 `+Y` 上轴并以 50 Hz 保存。训练命令项在构造 SMPL 根姿态 token 时左乘 `Rx(+90°)` 一次，再移除 `[0.5,0.5,0.5,0.5]` SMPL base rotation；`smpl_y_up=true` 因此是正确且必要的，不能改成 false。
- 同一个 SMPL 文件中的 `smpl_joints` 已在数据准备阶段使用转换后的 Z-up 根姿态做 FK，训练观测不再对 joint positions 做第二次 Y→Z 转换。也就是说，源文件保留可追溯契约，但进入两个 Encoder 的实际姿态/关节张量都已经在 Z-up 语义下。

### 4. 训练前坐标门禁

- 新增 `gear_sonic/tools/validate_bumi3_training_coordinates.py`。该脚本只读数据，不生成、改写、筛选或重采样任何文件。
- Robot 侧复现 MotionLib 30→50 Hz 时间网格，并对 root 与 `waist_yaw_link` 局部姿态分别 Slerp 后组合腰部世界姿态；SMPL 侧复现训练时 Y-up→Z-up 和 base rotation removal。
- 门禁要求 3261 Robot、3162 同名配对、99 Mine-only、55 个隔离 key 全部存在；重新计算的 `median(angle(inv(robot_waist) * processed_smpl_root)) > 45°` 集合必须与配置中的 55 条完全一致。通过后预期训练候选为 3206 Robot，其中 3107 条有 SMPL。

### 5. TensorBoard 原生写入

- `ppo_trainer.py` 在 rank 0 为每个实验创建 `<experiment_dir>/tensorboard/` 的 `SummaryWriter`，初始化时立即 flush，保证即使环境或首轮 rollout 失败也能定位 event 文件。
- 每次 PPO 日志迭代把已有的数值指标按 `global_step` 写入并 flush；字符串路径等非标量不会写入。正常结束和提前停止都会关闭 writer；其他 GPU 不重复写 event。
- 该改动不改变 PPO、网络、奖励、控制频率、batch 或优化器，只新增旁路可视化记录。

### 6. 已完成的本地验证

- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m compileall -q gear_sonic`：通过。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m pytest -q gear_sonic/tools/test_prepare_bumi3_sonic_dataset.py`：`8 passed`；有 1 个既有 invalid escape sequence DeprecationWarning。
- `git diff --check`：通过。

### 7. 待本节后续补录

- 尚需把本轮文件同步到 `noetix-volc:/home/liwei/GR00T-WholeBodyControl`，运行全量坐标门禁和 BUMI3 集成验证，再启动新的 8 卡 scratch 训练。
- 只有确认 MotionLib 日志命中 55/55、首轮真实 PPO 指标、8 张 4090 均有计算负载、训练日志无 traceback/NaN/Inf、且新实验目录内存在可解析 TensorBoard event 后，才把“训练已重启”记录为完成。

### 8. 服务器全量门禁和 8 卡 scratch 启动结果

- 同步目标：`noetix-volc:/home/liwei/GR00T-WholeBodyControl`。仅同步本轮明确涉及的配置、MotionLib、trainer、测试、验证脚本和记录文件；没有目录级删除、Git reset/clean 或参考仓库修改。
- 首次多源 `rsync` 错误地把 6 个代码文件平铺到远端仓库根目录。核对文件名/大小后仅删除这 6 个由本轮刚创建的根目录副本，再以 `rsync -R` 同步到正确包路径；实际包内文件在纠正前未被该错误传输覆盖，其他根目录文件未动。
- 服务器 `/root/miniconda3/envs/liwei_lab/bin/python -m compileall -q gear_sonic`：通过。
- 服务器数据契约单测：`8 passed`，同一既有 invalid escape sequence warning。
- 服务器全量坐标门禁：通过。实际输出为 `robot_total=3261 smpl_paired=3162 mine_only=99 excluded=55 training_robot=3206 training_smpl=3107`；保留配对相对角中位数 `13.636874°`、最大 `44.710664°`，隔离集合最小 `45.181285°`。
- 远端 `validate_bumi3_integration.py` 未通过外部参考 provenance 检查：`/home/liwei/legged_lab` 的 `bumi.py` 和 MJCF 已不同于本地锁定参考版本。只读核对确认本仓库实际训练 URDF/MJCF 哈希与本地通过集成验证的文件完全一致；训练不动态导入远端参考仓库。没有为了通过检查而修改 `legged_lab` 或放宽哈希门禁。

实际启动命令使用 Accelerate 8 个进程、每个 rank 4096 个环境、100000 PPO iterations：

```bash
/root/miniconda3/envs/liwei_lab/bin/accelerate launch --num_processes=8 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_bumi3 \
  +resume=false checkpoint=null auto_load_latest=false \
  use_wandb=false headless=True num_envs=4096 \
  base_dir=/data/sonic_bumi3/runs \
  exp_var=hq_all_v2_coordfix_q55_scratch_100k \
  algo.config.num_learning_iterations=100000 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/data/sonic_bumi3/datasets/hq_all_v2/built/robot_all \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/data/sonic_bumi3/datasets/hq_all_v2/built/smpl_all
```

- tmux：`sonic_bumi3_q55_8gpu`。
- 启动日志：`/data/sonic_bumi3/launch_logs/sonic_bumi3_q55_8gpu_20260831_135526.log`。
- 实验目录：`/data/sonic_bumi3/runs/TRL_BUMI3_Track/manager/universal_token/all_modes/sonic_bumi3_hq_all_v2_coordfix_q55_scratch_100k-20260831_135533`。
- resolved config 已确认 `checkpoint: null`、`resume: false`、`auto_load_latest: false`，日志没有 checkpoint 加载记录，属于从头训练。
- 8 个 rank 均打印 `requested=55, matched=55, missing=0, remaining=3206`，证明隔离实际生效而非只写进 YAML。
- 观察到第 12 iteration：约 `226842–231005 steps/s`；第 12 轮 mean reward `0.50039`、mean length `11.18375`。这是随机初始化最早期指标，只能证明训练循环工作，不能据此判断收敛质量。
- 检查时 8 张 RTX 4090 显存约 `19.7–20.7 GiB`，GPU 利用率约 `74%–89%`；8 个 worker 均存活。
- 训练日志未发现 traceback、AssertionError、RuntimeError、OOM、NCCL error 或 NaN/Inf。Isaac headless 初始化存在既有 Vulkan/GPU Foundation renderer 报错，但各 rank 均继续完成场景、MotionLib、DDP 和 PPO 训练初始化。
- TensorBoard event：`<实验目录>/tensorboard/events.out.tfevents.1788155827.noetix.3046464.0`。EventAccumulator 成功解析 `122` 个 scalar tags；检查时已有 13 个 step，`objective/rewards=0.5022393`、`objective/length=11.48`、`fps=229372`，证明不是空文件或仅创建目录。

## 2026-08-31：BUMI3 SONIC sim2sim 参考影子与训练 termination 复查

### 1. 起点、参考范围和保护措施

- 分支：`feature/bumi-native-sonic-full-training`；起始 HEAD：`b1c3606ce96f00a01745cb8382f8bfa0b9b4d780`。
- 本轮开始前 sim2sim 核心、入口、测试、验证和文档均为工作区内已有的未跟踪文件；本轮在其现有实现上增量加入参考影子，没有 reset、clean、stash、覆盖其他用户文件，也没有 commit/push。
- 用户指定参考 `/home/weili/legged_lab/scripts/sim2sim_mimic_vision_4340.py`。本轮只复用“独立参考 FK data + 半透明 decorative geom”的可视化思路；没有复制 BUMI3_4340 的执行器、关节限位、初始高度、映射、奖励或其他机器人参数。实际影子继续使用本仓库 BUMI3 SONIC contract、21 DoF 名称映射和当前 `bumi3.xml`。

### 2. 服务器 termination 实际复查

- 训练会话仍为 `sonic_bumi3_q55_8gpu`，检查时 8 个 worker 存活，日志无 traceback/OOM/NCCL/NaN。
- TensorBoard rank-0 曲线：`objective/length` 从 step 1 的 `11.865` 增至 step 1000 的 `34.19`，但 step 4503 又降至 `24.10`，即平均 episode 约 `0.482 s`；`objective/rewards` 同期为 `0.52251 -> 2.14561 -> 1.50061`，没有保持单调改善。
- step 4503 termination：`anchor_pos=0.26003`、`anchor_ori_full=0.06537`、`ee_body_pos=0.53400`、`foot_pos_xyz=0.44933`。当前 `ee_body_pos` 已只包含双肘，因此高触发率不能再归因于双脚重复计数；双肘和双脚位置仍是主要失败项，anchor position 也在恶化，而 anchor orientation 已相对较低。
- termination term 可以同一 episode 同时触发，所以上述比例不可相加当成概率分布；但短 episode、双肘/双脚高触发以及 length/reward 回落共同证明当前训练仍未稳定学会跟踪。

### 3. 参考影子实现

- `gear_sonic/utils/mujoco_sim/bumi3_sim2sim.py`：新增独立 `reference_visual_data`。每帧用动作文件的 root position、root quaternion 和 21 个关节构造完整 MuJoCo qpos，只施加与 Robot Encoder 一致的可选起始 heading 对齐，再执行 `mj_forward`。
- 参考 root Z 不跟随真实机器人，也不在显示层做贴地或高度覆盖。如果训练参考横躺/穿地，红色影子会原样显示，避免参考脚本中 `align_reference_height_to_robot=True` 掩盖问题。
- 参考 geom 写入 passive viewer 的 `user_scn`，统一红色 tint、默认 alpha `0.32`，category 为 `mjCAT_DECOR`；不进入 contact、collision、actuator 或 MuJoCo 积分。
- 新增 `reference_pose_diagnostics`：打印参考帧 root height、base/`waist_yaw_link` 上轴对世界 +Z 的倾角和最低 body origin 高度。站立通常接近 0°，侧躺通常接近 90°。
- `gear_sonic/scripts/run_bumi3_sim2sim.py`：默认开启影子，新增 `--reference-alpha` 和 `--no-show-reference`，启动前打印 `BUMI3_REFERENCE_POSE`。
- `gear_sonic/tests/test_bumi3_sim2sim.py`：新增直立/90°侧躺诊断、参考高度不跟随实际机器人、marker 数量/有限值/透明度/decorative category 回归测试。
- `gear_sonic/tools/validate_bumi3_sim2sim.py`：验证独立参考 FK、22 个影子 geom、alpha 和 physics=false 契约。
- `docs/source/getting_started/bumi3_sim2sim.md`：记录不透明实际机器人/红色参考影子的视觉含义、判读方式和 CLI。

### 4. 真实参考和旧 checkpoint 验证

- `finedance__001_50fps.pkl` 首帧：root height `0.538330 m`，base/waist tilt 均为 `0.755295°`，明确为直立参考，不是横躺。
- `finedance__001` 全 4879 帧：base/waist tilt 中位数 `9.019°`、P95 `27.0182°`、最大 `93.8133°`、超过 45° 帧占 `3.2794%`。少量高倾角帧属于动作内容，整段不是统一横躺。
- `finedance__002` 全 5179 帧：中位数 `11.4468°`、P95 `54.1611°`、最大 `81.7806°`、超过 45° 帧占 `8.0711%`；该动作包含更多大倾角片段，但中位参考仍为直立。
- 使用旧 `model_step_070000_g1.onnx` + `finedance__001_50fps.pkl` 做 100 控制步真实 smoke：参考首帧 tilt `0.7553°`，实际机器人 2 秒后 root height `0.066842 m`。至少对该轨迹，摔倒不能归因于参考首帧横躺，说明需要继续检查 checkpoint 学习质量、SONIC 双编码器/解码器训练语义、奖励/termination 和部署动力学，而不是继续做统一 90° 根坐标修正。

### 5. 实际测试及结果

- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m compileall -q gear_sonic`：通过。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m pytest -q gear_sonic/tests/test_bumi3_sim2sim.py`：`11 passed`。
- `git diff --check`：通过。
- `validate_bumi3_sim2sim.py --skip-smoke`：通过；影子 `22` geoms、alpha `0.32`、physics false，静态参考 base/waist tilt `0°`。
- 真实 ONNX+参考 100-step validation：通过接口、顺序、影子 FK 和有限值检查；不把 2 秒后摔倒误报为动作质量通过。
- GUI passive viewer 实际启动 2 个控制步：通过，`viewer.user_scn` 影子写入和同步没有 API/runtime error；该短运行只验证渲染链路，不代表人工完成整段视觉验收。
- Ruff 未运行：当前 `env_isaaclab` 没有安装 `ruff`（`No module named ruff`），未为本轮擅自改变环境依赖。

### 6. 风险和回滚

- 当前只对本地已有两条 FineDance 50 Hz 轨迹做完整倾角统计；全库训练坐标门禁此前已通过 3162 对配对检查，但“动作语义是否合理”仍需选取代表性站立、爬行、跪地和翻滚片段人工观察红色影子。
- 影子只影响 GUI user scene；关闭方式为 CLI `--no-show-reference`，代码回滚范围仅是本节列出的 sim2sim、测试、验证和文档增量。训练进程未重启，影子修改不会改变正在运行的 Isaac Lab 训练。

## 2026-08-31：参考影子“散架”渲染修正

### 1. 问题现象与确认原因

- 用户实际截图显示红色参考机器人各 link 视觉上分离，并出现大量 capsule 形状；这不是指定参考脚本的显示效果。
- 第一个实现只建了独立 `MjData`，但仍复用已执行 `_apply_training_collision_contract()` 的动力学 `self.model`。该模型已把 base、左右 leg-roll 和左右 knee 的唯一 mesh geom 替换成训练 URDF capsule，因而不能用来画完整参考外观。
- 第一个实现还直接使用 `geom_xpos/geom_xmat` 和模型数组重建 marker，没有像参考脚本一样先调用 `mjv_updateScene` 获取渲染器已解析的 `MjvGeom`。
- 对照 `/home/weili/legged_lab/scripts/sim2sim_mimic_vision_4340.py` 后确认，参考实现的关键是独立 `ref_model + ref_data + ref_scene`，而不只是第二个 data。

### 2. 代码修改

- `gear_sonic/utils/mujoco_sim/bumi3_sim2sim.py`：从原始 BUMI3 MJCF 单独重新加载 `reference_visual_model`，不对它施加训练碰撞体或 armature 运行时覆盖；参考 FK、base/`waist_yaw_link` 姿态诊断和 anchor 四元数均改为使用该独立模型。
- `gear_sonic/utils/mujoco_sim/bumi3_sim2sim.py`：每帧调用 `mjv_updateScene(reference_visual_model, reference_visual_data, ...)`，只复制其中的 22 个 robot `MjvGeom`，不自行推断 mesh 姿态/尺寸。
- `gear_sonic/utils/mujoco_sim/bumi3_sim2sim.py`：写入 viewer scene 时复制 `type/pos/mat/size/rgba/dataid/matid` 及其他渲染字段，再单独设为 `mjCAT_DECOR`。不再调用会二次解释 capsule size 的 `mjv_initGeom`。
- 离屏像素检查又发现预分配 `MjvGeom.label` 未清空时会把随机字节渲染为字符；现已显式复制参考 label（当前为空字符串）。
- `gear_sonic/tests/test_bumi3_sim2sim.py`：回归测试改为与独立 `mjv_updateScene` 结果逐 geom 对比 `pos/mat/size/dataid/matid/label`；同时断言动力学 base 为 capsule、红色参考 base 仍为原始 mesh。
- `gear_sonic/tools/validate_bumi3_sim2sim.py`：新增独立 ref model、完整有限 marker 字段和 base mesh/capsule 隔离校验；输出明确标记 `render_source:independent_ref_model_mjv_updateScene`。
- `docs/source/getting_started/bumi3_sim2sim.md`：记录独立 ref model 与最终 `MjvGeom` 复制契约，说明为何不能复用已替换碰撞体的动力学 model。

### 3. 实际验证

- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m pytest -q gear_sonic/tests/test_bumi3_sim2sim.py`：`11 passed`。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python gear_sonic/tools/validate_bumi3_sim2sim.py --skip-smoke`：`PASS`；输出 `geoms:22` 且 `render_source:independent_ref_model_mjv_updateScene`。
- 使用真实 `model_step_070000_g1.onnx` + `finedance__001_50fps.pkl` 运行 100 个控制步，再用 MuJoCo EGL 离屏渲染同一 user-scene geom 复制链。修正后红色参考为完整、连通、直立的 22 个原始 mesh，无随机字符；白色机器人同时已倒地，root height 为 `0.0668425 m`。这个画面现在能正确区分“参考姿态”与“policy/部署闭环失败”。
- 此修改只影响 sim2sim 可视化，没有改动 ONNX 观测、控制、训练数据、奖励、termination 或服务器训练进程。

## 2026-08-31：纠正 sim2sim 碰撞来源为 BUMI3 XML

### 1. 用户纠正与最终契约

- 用户明确纠正：sim2sim 在 MuJoCo 内运行，碰撞必须使用 `bumi3.xml`，与 Isaac Lab URDF 无关。
- 本节取代本文档前面“sim2sim 在内存中复刻训练 URDF collision”的旧结论。旧结论是错误的，不再是当前代码行为。
- 最终白色 policy 机器人就是 `self.model/self.data`：MuJoCo 从 `bumi3.xml` 加载的同一组 geom 同时用于碰撞、动力学和渲染，不隐藏、不替换、不叠加 policy 可视代理。
- 红色参考从同一 `bumi3.xml` 单独加载 `ref_model/ref_data/ref_scene`，只用于持有参考 qpos 和绘制 decorative marker，不参与物理。
- Isaac Lab URDF 的 cylinder→capsule、禁用 collision link 等契约不再进入 sim2sim Python。如需改变 MuJoCo 碰撞，应显式修改并审核 XML。

### 2. 代码修改

- `gear_sonic/utils/mujoco_sim/bumi3_sim2sim.py`：删除 `BUMI3_URDF_DISABLED_COLLISION_BODIES`、`BUMI3_URDF_CAPSULE_COLLISIONS`、`_apply_training_collision_contract()` 及初始化调用。运行时不再改写 `geom_type/dataid/pos/quat/size/contype/conaffinity`。
- `gear_sonic/utils/mujoco_sim/bumi3_sim2sim.py`：撤销中途尝试的“隐藏动力学 geom + 用 user scene 额外绘制白色 policy mesh”方案。GUI 现在由 MuJoCo 直接渲染 XML 动力学模型，user scene 只包含红色参考。
- `gear_sonic/tests/test_bumi3_sim2sim.py`：用 `test_runtime_visual_and_collision_geoms_match_original_xml` 取代 URDF collision 测试。测试重新加载 XML，逐数组比较 `geom_type/bodyid/contype/conaffinity/condim/dataid/group/priority/pos/quat/size/friction/solref/solimp/margin/gap/rgba`，并断言 22 个 robot geom 均保留 XML mesh。
- `gear_sonic/tools/validate_bumi3_sim2sim.py`：删除 capsule/禁用 link 断言和白色可视代理断言，改为同样的 XML geom 数组指纹校验；输出 `collision_source:bumi3_xml` 和 `render_source:dynamics_bumi3_xml`。
- `gear_sonic/scripts/run_bumi3_sim2sim.py`：resolved 输出改为 `policy_visual_and_collision: direct_bumi3_xml_dynamics_model`。
- `docs/source/getting_started/bumi3_sim2sim.md`：删除“运行时复刻 URDF 碰撞”说明，明确 sim2sim 不读取、不复刻、不覆盖 URDF collision。

### 3. 实际验证和结果

- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m compileall -q gear_sonic`：通过。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m pytest -q gear_sonic/tests/test_bumi3_sim2sim.py`：`11 passed`。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python gear_sonic/tools/validate_bumi3_sim2sim.py --skip-smoke`：通过；22 个 policy geom 全部来自 XML，输出 `physics:true` 和 `collision_source:bumi3_xml`。
- 真实 `model_step_070000_g1.onnx` + `finedance__001_50fps.pkl` 100 控制步：接口/有限值/XML 碰撞指纹验证通过；2 秒后 policy root height 为 `0.084035 m`，仍然摔倒。
- MuJoCo EGL 离屏像素检查：白色 policy 是完整 XML mesh 且直接由动力学 scene 渲染；红色参考是完整 XML mesh 的 decorative 影子。画面不再出现 URDF capsule 代理。
- GUI 真实入口运行 5 个控制步并正常退出；X11 在程序结束后打印 `NV-GLX missing` 环境警告，本次命令退出码仍为 0，不把该警告当作 GPU GUI 稳定性证明。
- `git diff --check`：通过；搜索已确认 sim2sim 代码和文档中不再存在 `_apply_training_collision_contract`、`BUMI3_URDF_CAPSULE_COLLISIONS`、`training_urdf_runtime_override` 或 policy 可视代理路径。本轮未 commit/push，未改动服务器训练。

## 2026-09-01：2790 条 HQ4 PASS-only BUMI3 严格 50 Hz 数据准备（阶段一）

### 1. 起点、授权范围与保护措施

- 分支仍为 `feature/bumi-native-sonic-full-training`，起始 HEAD 仍为 `b1c3606ce96f00a01745cb8382f8bfa0b9b4d780`；工作区已有的大量用户修改全部保留，没有 reset、clean、stash、commit 或 push。
- 本轮数据源只使用服务器2的 `/data0/user/liwei/robot_retargeter_bumi3_hq4_zup_v1`；唯一白名单只接受质量报告 `/data0/user/liwei/datasets/bumi_quality_robot_retargeter_30hz_v1/quality_report.jsonl` 中同时满足 `status=PASS` 和 `quality_accepted=true` 的条目，数量必须恰好为 2790。REVIEW/REJECT 即使文件存在也不会进入输出。
- 4090 原目录在本阶段尚未替换；先构建独立数据集、全量验证和安全传输，最终只会通过保留时间戳备份的 rename/symlink 发布，保证可回滚。
- SSH 密码只用于本轮交互式认证，没有写入仓库、转换脚本、日志或数据 provenance。

### 2. 新增代码与目的

- 新增 `gear_sonic/tools/prepare_bumi3_pass50_dataset.py`：
  - 从质量报告建立严格、唯一、哈希锁定的 2790 条 PASS manifest；
  - 比较 retargeter MJCF 与本仓库 SONIC MJCF 的 22-body/21-joint 核心名称、树结构、body pose、joint origin/axis/range 和 actuator 顺序，不一致立即拒绝构建；
  - 采用 `arange(0,(T-1)/30,1/50)` 的独占末帧时间网格，把位置/关节位置线性插值到 50 Hz，把 wxyz body 四元数做最短弧 SLERP；
  - 从 50 Hz 结果重新计算 joint velocity/acceleration/jerk、body linear/angular velocity，并按双 `ankle_roll_link` 的低位与水平/垂直速度滞回重新计算接触；
  - 输出 SONIC 实际读取的 50 Hz robot PKL，同时把导数量和接触保存到独立 audit NPZ；PASS 原始 NPZ 只允许建立硬链接，跨文件系统或退化成复制会失败；
  - `pair-smpl` 不只要求同名、50 Hz 和严格同帧数，还复现训练的 SMPL Y-up→Z-up 与 base rotation removal，并比较 Robot `waist_yaw_link` 世界姿态。中位相对角超过 45 度的条目保留为 robot-only，避免把坐标语义异常配对送入 SMPL Encoder。
- 新增 `gear_sonic/tools/test_prepare_bumi3_pass50_dataset.py`：用合成的匀速平移/绕 Z 轴旋转动作覆盖完整 build/validate/pair 路径，验证 PASS-only、硬链接 inode、线性插值、SLERP、四元数范数、派生速度以及 SMPL 坐标配对门禁。

### 3. 服务器2真实构建结果

- 质量报告实际为 3154 行；严格白名单为 2790 条，组成：`aioz_gdance=1884`、`aistpp=750`、`finedance=88`、`compas3d=68`。
- retargeter MJCF SHA256 为 `fe93472dd764704fe8389b0f82052ae84ed8bc90f6d71b1467872f86e08a9ad3`，SONIC MJCF SHA256 为 `02874afebbe30ba1f90218394c8f9953f5d7a808e6b9950e7964c731da6dfbfe`；二者整体文件不同，但过滤 retargeter 辅助 marker 后，核心运动学门禁通过，核心 SHA256 为 `b65b51c4775a91658aa95e48eda220b3ff75b99451491b4ef175ce18d7d0bed2`。
- 发布目录：`/data0/user/liwei/datasets/sonic_bumi3_hq4_pass50_v1`。原始源帧总数 `3,043,605`，严格 50 Hz 目标总帧数 `5,069,508`；2790 条全部 finite，root quaternion 最大单位范数误差 `2.220446049250313e-16`。
- 左右脚平均接触占比分别为 `0.5534127` 和 `0.5653995`。PASS raw 硬链接目录表观大小约 3.4 GiB、robot PKL 约 1.1 GiB、audit NPZ 约 3.3 GiB、meta 约 4.2 MiB；抽样 inode 和全量 validator 都确认 raw 是原文件硬链接，不是复制。
- provenance 固定质量报告 SHA256 `d3bc24fb62600a71339625ef233bf0eb267d05296d45a3701a298d92b5cfb798`，并记录源/目标 MJCF 哈希、时间网格、插值方法、派生量和接触重算规则。

### 4. 阶段一实际验证

- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m pytest -q gear_sonic/tools/test_prepare_bumi3_pass50_dataset.py`：`3 passed`。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m compileall -q gear_sonic`：通过。
- `git diff --check`：通过。
- 服务器2构建内置全量验证：`BUMI3_PASS50_BUILD=PASS`，`robot_count=2790`，`total_target_frames=5069508`。
- 服务器2独立再次执行 full validate：`BUMI3_PASS50_VALIDATE=PASS`，逐条反序列化 2790 个 robot PKL，并逐条检查 2790 个 raw 硬链接和 2790 个 audit NPZ。

### 5. 仍待阶段二补录

- 4090 端训练子集传输、逐条 training-only 校验、SMPL key/frame/坐标语义配对、旧目录时间戳备份与替换、8 卡 scratch 启动及 TensorBoard 真实 event/首轮指标将在传输和运行验证结束后补录；在这些证据出现前不把训练写成“已成功重启”。

## 2026-09-01：2790 条 HQ4 PASS-only 发布、SMPL 配对与八卡重训（阶段二）

### 1. 传输、校验和发布

- 只把服务器2输出中的 `built/robot_all`（约 1.1 GiB）和 `meta`（约 4.2 MiB）传到 4090；3.4 GiB raw 硬链接和 3.3 GiB audit NPZ 留在服务器2，避免重复跨机传输。传输目标先使用 `/data/sonic_bumi3/datasets/.hq4_pass50_v1.transfer.20260901` staging。
- 最初在单流传输仍运行时过早执行了 validation，因只看到 24 个已落盘文件而产生预期的 `manifest.jsonl` missing。检查本地与两端 tar PID 后确认是校验竞态，不是 tar 已成功结束；没有发布不完整目录。随后按目标端实际文件大小生成缺失清单，用四个独立 SSH master 传输不重叠且总字节均衡的分片。
- 四个分片均返回 `MULTI_SHARD_0..3=PASS`。4090 端独立运行 training-only validator，逐条反序列化 2790 个 PKL，结果为 `BUMI3_PASS50_VALIDATE=PASS {"robot_count": 2790, "total_target_frames": 5069508}`。
- 独立发布目录为 `/data/sonic_bumi3/datasets/hq4_pass50_v1`。旧 `/data/sonic_bumi3/datasets/hq_all_v1/built/robot_all` 的 3261 条文件完整移动到 `/data/sonic_bumi3/datasets/hq_all_v1/built/robot_all.pre_hq4_pass50_20260901_110141`；原路径现为指向新 2790 条目录的符号链接。没有删除旧数据，回滚只需停止新训练、删除该符号链接并把备份目录改回 `robot_all`。

### 2. SMPL 严格配对结果

- 配对源仍为 `/data/sonic_bumi3/datasets/hq_all_v1/built/smpl_all`，新目录为 `/data/sonic_bumi3/datasets/hq4_pass50_v1/built/smpl_all`。通过项使用同文件系统硬链接，抽样确认源/目标 `st_dev + inode` 完全相同。
- 2790 条 Robot 中 2788 条通过 key、`fps=50`、三项 SMPL 字段严格同帧数以及 Robot-waist/processed-SMPL-root 中位相对角不超过 45 度的全部门禁；保留配对的中位角总体中位数 `16.112373°`，最大 `42.886849°`。
- 两条仅因坐标语义门禁降为 robot-only：`aistpp__gLO_sBM_cAll_d15_mLO5_ch04=48.422173°`、`aistpp__gWA_sBM_cAll_d27_mWA5_ch08=61.399928°`。训练代码对无 SMPL 配对的 Robot 动作使用既有零 SMPL fallback，没有删除这两条 PASS Robot 数据。
- 完整逐条结果保存在新数据集 `meta/smpl_pairing_report.jsonl`，汇总保存在 `meta/smpl_pairing_summary.json`。

### 3. Hydra resolved 配置与启动命令

- 实际 compose 输出保存为 `/data/sonic_bumi3/datasets/hq4_pass50_v1/meta/resolved_training_config.yaml`。解析确认：`checkpoint=null`、`resume=false`、`auto_load_latest=false`、`num_envs=4096/rank`、`num_learning_iterations=100000`、`target_fps=50`、`exclude_motion_keys=[]`、`sim_dt=0.005`、`decimation=4`，控制频率为 50 Hz。
- 新质量报告已重新验证机器人本体，因此旧 55 条 Robot/SMPL 异常清单不再作为 Robot 过滤器；其与新 2790 PASS 的交集为 11 条。启动时显式覆盖为空，保证 Robot 唯一白名单确实是 2790 PASS；SMPL 是否启用只由本轮新配对门禁决定。
- 八卡从头训练命令：

```bash
/root/miniconda3/envs/liwei_lab/bin/accelerate launch --num_processes=8 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_bumi3 \
  +resume=false checkpoint=null auto_load_latest=false \
  use_wandb=false headless=True num_envs=4096 \
  base_dir=/data/sonic_bumi3/runs \
  exp_var=hq4_pass50_v1_scratch_100k \
  algo.config.num_learning_iterations=100000 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/data/sonic_bumi3/datasets/hq4_pass50_v1/built/robot_all \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/data/sonic_bumi3/datasets/hq4_pass50_v1/built/smpl_all \
  ++manager_env.commands.motion.motion_lib_cfg.exclude_motion_keys=[]
```

- 训练 tmux：`sonic_bumi3_hq4_pass50_8gpu`；启动日志：`/data/sonic_bumi3/launch_logs/sonic_bumi3_hq4_pass50_8gpu_20260901_110307.log`；实验目录：`/data/sonic_bumi3/runs/TRL_BUMI3_Track/manager/universal_token/all_modes/sonic_bumi3_hq4_pass50_v1_scratch_100k-20260901_110314`。

### 4. 真实运行证据和当前边界

- 4090 `liwei_lab` 环境运行 `python -m pytest -q gear_sonic/tools/test_prepare_bumi3_pass50_dataset.py`：`3 passed`；本地 `env_isaaclab` 同一测试为 `3 passed`，本地 `compileall -q gear_sonic` 与 `git diff --check` 均通过。
- 8 个 rank 均打印 `Loaded 2790 motions` 和 `requested=0, matched=0, missing=0, remaining=2790`；运行时 Action Manager shape 为 `21`，policy observation 为 `690`，critic observation 为 `1245`。
- tokenizer 只有 `encoder_index`、Robot 两项输入和 SMPL 两项输入；运行时只初始化 `g1` 与 `smpl` encoder。FSQ 输出为 `64`（2 tokens × 32），没有 Teleop encoder/tokenizer/loss。
- 运行时 physics step `0.005 s`、environment step `0.02 s`；奖励表确认 `feet_acc=-2.5e-6`、`torque_limits=-0.01`，termination 表确认 `ee_body_pos`、`foot_pos_xyz` 是两个独立项。
- TensorBoard event 已真实生成并解析：`tensorboard/events.out.tfevents.1788231887.noetix.3236527.0`，含 122 个 scalar tags。step 1→51：reward `0.61743→0.57356`、mean length `10.97→10.326`、value loss `0.08377→0.01805`；step 51 吞吐约 `239275 steps/s`。早期 reward/length 尚未改善，必须继续观察而不能把训练链路正常等同于策略已开始收敛。
- 检查时 8 张 RTX 4090 显存约 `19.9–20.3 GiB`，利用率约 `85%–90%`；日志未发现 traceback、AssertionError、RuntimeError、CUDA OOM、NCCL 或 NaN。已有 headless Vulkan/renderer 报错与该服务器此前多卡训练相同，所有 rank 在这些日志后继续完成场景、MotionLib、网络、DDP 和 PPO iteration。
- 这些 51 个 iteration 只证明数据加载、Isaac 环境、双 Encoder、PPO、TensorBoard 和八卡计算链路工作；随机初始化早期 termination 仍高，不能据此宣称动作已收敛或 sim2sim 不再摔倒，后续必须持续看 length/reward/termination 曲线并用新 checkpoint 做参考影子 sim2sim。

### 5. TensorBoard 服务和查看方式

- 服务器原 6006 端口已有其他 TensorBoard，未终止或覆盖用户进程。本实验单独在 tmux `tensorboard_bumi3_hq4_pass50` 的 `127.0.0.1:6016` 运行 TensorBoard 2.20.0。
- 本地端口转发：`ssh -N -L 6016:127.0.0.1:6016 noetix-volc`，浏览器打开 `http://127.0.0.1:6016/`。
- 训练查看：`ssh noetix-volc -t 'tmux attach -t sonic_bumi3_hq4_pass50_8gpu'`；脱离 tmux 使用 `Ctrl-b` 后按 `d`。

## 2026-09-01：补充 Git 中文提交说明与 SONIC 训练服务器约束

### 1. 修改范围与理由

- 所属分支：`feature/bumi-native-sonic-full-training`；起始 HEAD：
  `b1c3606ce96f00a01745cb8382f8bfa0b9b4d780`；修改前与上游 ahead `0`、behind `0`。
- `agent.md`：把原有“推荐提交消息”规则收紧为 Git 提交标题和正文必须使用详细中文说明，
  并要求明确写出具体改动、修改理由、影响边界、实际验证、未执行项及原因，避免模糊或仅列
  文件名的提交记录。
- `agent.md`：新增 SONIC 训练服务器章节，固定使用 SSH Host 别名 `noetix-volc`，并记录
  用户指定的 HostName、用户、端口、IdentityFile、IdentitiesOnly 和 keepalive 参数。
- 修改理由：保证 Git 历史可由中文直接审计，并避免后续 SONIC 训练误连其他服务器。

### 2. 兼容性、验证与回滚

- 本轮只修改规则文档和本记录，没有修改训练代码、配置、数据、模型或服务器状态；没有连接
  `noetix-volc`，也没有执行 commit、push、merge、rebase、stash、reset 或训练操作。
- 实际验证：使用文本检索核对新增规则和 7 项 SSH 配置字段，并执行 `git diff --check`。
- 未运行代码测试：本轮没有代码行为变化，单元测试、仿真和训练均不适用。
- 风险：SSH 配置中的私钥路径要求执行环境已有 `~/.ssh/noetix-8.pem` 且权限正确；本轮未验证
  密钥存在性或远端可达性。
- 回滚：只需删除 `agent.md` 的对应规则增量和本节记录；不得影响同文件内其他已有未提交内容。

## 2026-09-01：补充代码注释必须使用详细中文的约束

### 1. 修改内容与理由

- 所属分支：`feature/bumi-native-sonic-full-training`；起始 HEAD：
  `b1c3606ce96f00a01745cb8382f8bfa0b9b4d780`。
- `agent.md`：新增强制规则，要求新增、改写及受当次代码修改影响的注释全部使用详细中文，
  并准确说明代码目的、关键逻辑、输入输出、边界条件或必要的设计理由。
- 同时禁止新增仅复述代码字面行为、含义模糊或只有英文的注释；未涉及的历史注释不进行批量
  改写，避免把无关格式变化混入功能修改。
- 修改理由：让后续代码的关键意图和约束可以直接用中文审查、维护与追溯。

### 2. 验证、影响与回滚

- 本轮只修改规则文档和本记录，没有修改任何运行代码、训练配置、数据、模型或服务器状态；
  没有执行 commit、push、远程连接或训练操作。
- 实际验证：文本检索确认规则已写入 `agent.md`，并执行 `git diff --check`。
- 未运行代码测试：本轮没有代码行为变化，单元测试、仿真和训练均不适用。
- 回滚：删除 `agent.md` 第 6 条规则及本节记录即可，不应改动同文件内其他已有内容。

## 2026-09-01：修正 Robot Encoder 的 BUMI3 waist 锚点姿态语义

### 1. 现象、根因与修正边界

- 所属分支：`feature/bumi-native-sonic-full-training`；起始 HEAD：
  `b1c3606ce96f00a01745cb8382f8bfa0b9b4d780`；修改前与上游 ahead `0`、behind `0`。
- 工作区已有 BUMI3 资产、数据处理、sim2sim 和训练调整等未提交改动；本轮仅编辑
  锚点调用链、对应回归测试及本记录，没有 reset、stash、clean 或覆盖其他现有改动。
- `sonic_bumi3.yaml` 虽然已将 `anchor_body` 配成 `waist_yaw_link`，但 Robot Encoder 的姿态
  输入实际经过 `motion_anchor_ori_b_mf_nonflat -> root_rot_dif_l_multi_future`，
  原实现在参考侧直接调用 `MotionLib.get_root_quat_w()`，得到的是第 0 个刚体
  `base_link` 的浮动根四元数；仿真侧则使用 `robot_anchor_quat_w`，即
  `waist_yaw_link`。因此原始计算实际是
  `inverse(sim_waist) * reference_base`，当 `waist_yaw_joint` 非零时两侧刚体语义不一致。
- 本记录早期第 886 行曾误写“参考 quaternion 已来自
  `motion_anchor_body_index`”；该结论是根据配置和辅助属性做的错误推断，没有跟到
  `root_rot_dif_l_multi_future` 末端实现。本节的生产代码差异和回归测试证据取代
  那条旧记录，不再把原始状态说成已修正。
- 本修正不改网络层、token 数、观测维度、控制频率、奖励、PPO 或数据，只把
  Robot Encoder 的参考姿态改为与仿真侧同名的 `waist_yaw_link` FK 姿态。

### 2. 文件级修改

- `gear_sonic/envs/manager_env/mdp/commands.py`：
  - `root_rot_dif_l` 从 `get_root_quat_w(...)` 改为读取 `self.anchor_quat_w`；
  - `root_rot_dif_l_multi_future` 从多未来帧根四元数改为读取
    `self.anchor_quat_w_multi_future`，并恢复为 `[num_envs, num_future_frames, 4]`；
  - 修正后单帧和 Robot Encoder 实际使用的多未来帧均计算
    `inverse(sim_waist) * reference_waist`；
  - 保留 `root_rot_dif_*` 属性名以兼容旧配置，只修正其内部语义；新增中文注释
    明确 BUMI3 禁止把 `base_link` 根姿态代替 `waist_yaw_link` 的 FK 姿态。
- `gear_sonic/tests/test_tracking_anchor_semantics.py`：新增不需启动 Isaac Sim 的 AST 契约测试，
  锁定单帧和多未来帧实现必须读取命名 anchor，且不得重新调用
  `get_root_quat_w`。文件开头已详细说明 BUMI3 `base_link`/`waist_yaw_link` 差异和
  采用 AST 的原因。
- sim2sim 生产代码本轮无需再改：
  `gear_sonic/utils/mujoco_sim/bumi3_sim2sim.py` 已使用 `bumi3.xml` 独立 FK 每帧
  `waist_yaw_link` 的参考世界姿态，且用当前 policy robot 的同名 waist 姿态求相对旋转；
  `gear_sonic/config/sim2sim/bumi3_sonic.yaml` 中 `anchor_body_name` 也已是
  `waist_yaw_link`。

### 3. 兼容性与 checkpoint 影响

- G1 的配置 anchor 与其根刚体语义等价，所以返回数值不变；H2 和其他机器人从此也统一
  遵循各自配置的 `anchor_body`，旧的配置键和观测属性名没有变化。
- 当前正在服务器上运行的训练进程以及已生成 checkpoint 不会自动获得本地修正；
  其 Robot Encoder 已经按旧的 base/waist 混合语义训练。要评估这个修正，必须将当前
  代码同步到 4090 服务器后从头训练；本轮没有连接服务器、停止训练或启动新任务。
- 维度契约保持不变：`sim_dt=0.005`、`decimation=4`、控制/target FPS `50 Hz`、
  `action_dim=21`、FSQ 总维度 `64`、actor proprioception `690`、dynamic decoder
  `754 -> 21`。

### 4. 实际验证和未执行项

- 首次尝试在普通 `pytest` 中直接导入 `TrackingCommand` 做行为测试，收集阶段因未启动
  Isaac Sim `SimulationApp` 而报 `ModuleNotFoundError: No module named 'pxr'`。这是测试运行环境
  限制，不是锚点逻辑断言失败；因此把新测试改为可在普通 Python 中运行的 AST
  契约检查。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m pytest -q
  gear_sonic/tests/test_tracking_anchor_semantics.py`：`2 passed in 0.06s`。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m pytest -q
  gear_sonic/tests/test_bumi3_sim2sim.py`：`11 passed in 2.52s`；其中现有数值测试会将
  `waist_yaw_joint` 设为 `0.4 rad`，验证 sim2sim Robot Encoder 参考姿态来自 waist FK，
  而非 identity root quaternion。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m compileall -q
  gear_sonic/envs/manager_env/mdp/commands.py gear_sonic/tests/test_tracking_anchor_semantics.py`：通过。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python
  gear_sonic/tools/validate_bumi3_integration.py`：通过，输出 `BUMI3 原生 SONIC 集成验证通过`，
  并确认 21 DoF、22 bodies、上述频率和网络维度契约均未变。
- 未执行 1-env Isaac Lab reset/step：当前本地 BUMI3 训练数据路径为 `null`/由 CLI 指定，
  validation 脚本明确输出 `smoke: 未请求（需显式提供现有 BUMI3/SMPL 数据）`；本轮不把未运行
  的仿真 smoke 写为通过。
- 本轮没有 commit、push、merge、rebase、stash、reset，也没有修改远程数据或训练进程。

### 5. 回滚方式

- 如需回滚本轮锚点修正，只恢复 `commands.py` 中两个 `root_rot_dif_l*` 属性的原实现，
  删除 `test_tracking_anchor_semantics.py` 和本节记录即可；不得回滚或覆盖工作区中其他
  BUMI3 资产、数据、配置、sim2sim 和训练改动。

## 2026-09-01：强制中文注释与修改后 GitHub/训练服务器同步闭环

### 1. 用户规则与文件修改

- 所属分支：`feature/bumi-native-sonic-full-training`；起始 HEAD：
  `b1c3606ce96f00a01745cb8382f8bfa0b9b4d780`；修改前本地与上游 ahead `0`、behind `0`。
- `agent.md` 将代码注释规则明确扩展到行内注释、块注释、docstring 和 TODO；说明
  标识符、API 名和必要专有名词可保留原文，但解释句必须使用完整中文，不得
  新增只有英文或没有中文语义的注释。
- `agent.md` 记录用户对当前开发分支的持续非破坏性授权：每次修改完成后，必须
  更新记录、验证、逐文件暂存、使用详细中文提交、推送 GitHub，并在
  `noetix-volc` 同分支执行 `git pull --ff-only`。该授权不包含 merge、rebase、Tag、
  删分支、force push 或改写历史。
- 闭环规则显式保留安全边界：如果本地或服务器有用户独立未提交修改、分支不一致、
  非快进历史或网络故障，必须保留现场并报告，禁止通过 stash、reset、force push
  或自行合并来强行完成。

### 2. 服务器同步前的只读核对

- 已通过 SSH Host `noetix-volc` 连接，服务器仓库为
  `/home/liwei/GR00T-WholeBodyControl`，分支为
  `feature/bumi-native-sonic-full-training`，HEAD 同样为
  `b1c3606ce96f00a01745cb8382f8bfa0b9b4d780`。
- 服务器仓库所有者不是 SSH 的 `root` 用户，Git 报 `dubious ownership`。本轮不修改全局
  Git 配置，远端检查与同步命令只使用单次参数
  `-c safe.directory=/home/liwei/GR00T-WholeBodyControl`。
- 服务器存在 8 个已跟踪改动和 3 个未跟踪数据工具文件。逐文件 SHA256 核对后，
  10 个代码/配置/测试文件与本地完全相同；服务器的
  `BUMI3_SONIC_修改记录.md` 为本地记录前 `107075` 字节的严格字节前缀，
  无服务器独有后缀。这些文件是之前部署但未提交的同一批改动，不是新的服务器分叉。
- 正式同步时必须在 GitHub 推送后再根据远端目标 commit 复核所有这些路径；只有确认
  工作树内容已完整包含在目标 commit 中、没有独有修改后，才可清除重复工作树状态
  并执行快进拉取。

### 3. 本地验证结果

- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m compileall -q gear_sonic`：通过。
- 下列四组测试合并运行：`24 passed in 3.92s`；存在 1 条历史
  `DeprecationWarning: invalid escape sequence '\\*'`，未造成测试失败。
  - `gear_sonic/tools/test_prepare_bumi3_sonic_dataset.py`；
  - `gear_sonic/tools/test_prepare_bumi3_pass50_dataset.py`；
  - `gear_sonic/tests/test_bumi3_sim2sim.py`；
  - `gear_sonic/tests/test_tracking_anchor_semantics.py`。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python gear_sonic/tools/validate_bumi3_integration.py`：
  通过，输出 `BUMI3 原生 SONIC 集成验证通过`；实测为 21 DoF、22 bodies、
  `sim_dt=0.005`、`decimation=4`、控制/target FPS `50 Hz`、`action_dim=21`、
  FSQ `64`、actor proprioception `690`、dynamic decoder `754 -> 21`。
- integration validation 启动 Isaac Sim 时输出了已知的 CPU topology、powersave、IOMMU 和
  deprecated extension 警告，但进程退出码为 `0`并完成全部静态/Hydra 契约验证。
- validation 中 1-env smoke 仍显式未请求，原因是本地没有通过 CLI 提供现有
  BUMI3/SMPL 数据目录；本轮不把它记为已通过。

### 4. 待本轮操作完成后补录

- 当前累积 BUMI3 原生 SONIC、数据准备、训练坐标校验、sim2sim、waist 锚点修正、
  规则与详细记录将逐文件暂存，使用详细中文提交并推送 GitHub；实际 commit SHA、
  push 结果、服务器重复改动处置和 `git pull --ff-only` 证据将在操作完成后另行补录。

### 5. 实际提交、GitHub 推送和服务器快进结果

- 本地逐文件暂存了 21 个已核对的源码、配置、URDF、测试和 Markdown 文件；
  没有暂存 checkpoint、ONNX、PKL、NPZ、CSV、视频、日志、缓存或训练输出。
  `git diff --cached --check` 通过，staged stat 为 `21 files changed, 5864 insertions(+),
  64 deletions(-)`。
- 主提交 SHA 为 `c287ac97808dfe0511ed00b875e6fbcfc3313499`，中文标题为
  `feat: 完成 BUMI3 原生 SONIC 训练与 sim2sim 闭环`。提交正文详细记录了数据、训练、
  sim2sim、waist 锚点、兼容边界、实际验证和未执行 1-env smoke 的原因。
- `git push origin feature/bumi-native-sonic-full-training` 成功，GitHub 远端由
  `b1c3606` 快进到 `c287ac9`，没有 force push 或历史改写。
- GitHub 推送后，服务器执行 `git fetch` 并确认目标 ref 精确为
  `c287ac97808dfe0511ed00b875e6fbcfc3313499`。10 个重复代码/配置/测试文件再次与
  目标 commit 逐字节 SHA256 一致，服务器记录再次通过目标记录严格前缀检查。
- 为使三个已被目标 commit 完整包含的未跟踪数据工具可恢复，先将它们移动到
  `/tmp/bumi3_git_sync_backup.FqkxYc`；已跟踪重复改动只在确认目标 commit 已保存同样内容后
  恢复到旧 HEAD，随后执行非破坏性快进拉取。拉取后三个新文件与备份逐字节
  `cmp` 通过，因此没有丢失服务器内容。
- 服务器实际执行
  `git -c safe.directory=/home/liwei/GR00T-WholeBodyControl pull --ff-only origin
  feature/bumi-native-sonic-full-training`，结果为 `Updating b1c3606..c287ac9` 和
  `Fast-forward`。执行后分支正确、HEAD 为 `c287ac9`、`git status --short` 无输出，
  `SYNC_VERIFY=PASS`。
- 本次只完成代码 Git 同步，没有停止、重启或恢复服务器上已运行的训练进程。
  旧进程已加载的 Python 模块不会因工作树 `git pull` 自动替换；如要让新的
  `waist_yaw_link` 训练锚点生效，仍需用新代码从头启动新训练。

## 2026-09-01：修复 noetix-volc 的 Git 目录信任并统一分支跟踪

### 1. 问题含义和根因

- 所属分支：`feature/bumi-native-sonic-full-training`；起始 HEAD：
  `5d97f452751fbe5645929af9bb57e7b92e6a00bf`；本地工作区干净，与上游 ahead `0`、behind `0`。
- “服务器仓库非 root 用户所有”并不表示没有关联 GitHub。服务器目录
  `/home/liwei/GR00T-WholeBodyControl` 的顶层所有者是数值 UID/GID `14000032`，
  当前系统没有该 UID 的用户名映射，因此 `stat` 显示 `owner=UNKNOWN`；`.git` 目录本身
  属于 root。root 访问顶层所有者不同的仓库时，Git 按安全策略报
  `detected dubious ownership`。
- 服务器在修复前已存在正确 GitHub 远端：`origin` 的 fetch/push URL 均为
  `git@github.com:XiaoxiaoKuankuan/GR00T-WholeBodyControl.git`；服务器并非未关联仓库。

### 2. 实际修复和分支统一

- 在服务器 root 的全局 Git 配置中精确新增：
  `safe.directory=/home/liwei/GR00T-WholeBodyControl`。没有设置通配符，没有把其他目录
  加入信任范围，也没有 chown 整个仓库。修复后 root 可以直接运行普通 `git` 命令，
  不再需要每次传入 `-c safe.directory=...`。
- 本地和服务器均显式执行了
  `git branch --set-upstream-to=origin/feature/bumi-native-sonic-full-training
  feature/bumi-native-sonic-full-training`。两端当前分支、upstream 和 GitHub ref 现在一致：
  - 当前分支：`feature/bumi-native-sonic-full-training`；
  - upstream：`origin/feature/bumi-native-sonic-full-training`；
  - 本地、GitHub、服务器核对时 HEAD：
    `5d97f452751fbe5645929af9bb57e7b92e6a00bf`；
  - 两端 ahead/behind：`0/0`；
  - 两端 `git status --short`：无输出。

### 3. 端到端验证和影响边界

- 本地实际执行普通 `git push --dry-run`，结果为 `Everything up-to-date`，证明当前本地
  分支可以通过 `origin` SSH URL 访问 GitHub 并使用正确的默认推送目标。
- 服务器在不传 `-c safe.directory`、不指定远端和分支的情况下，实际执行普通
  `git pull --ff-only`，结果为 `Already up to date.`，证明 root 目录信任、GitHub SSH
  访问、origin 和 upstream 全部有效。
- 本轮修改的是服务器 root Git 配置、两端分支跟踪关系和本记录；没有修改
  SONIC 源码、训练配置、数据、checkpoint 或运行中的训练进程，因此代码单元测试和仿真
  不适用。
- 回滚服务器信任配置时，只需精确删除 root Git config 中这一条
  `safe.directory=/home/liwei/GR00T-WholeBodyControl`；但删除后 root 会再次遇到 ownership 安全拦截。

## 2026-09-01：实现 BUMI3 SONIC 三数据源只读索引与可选尾帧对齐

### 1. 修改起点、授权范围和数据来源

- 所属分支：`feature/bumi-native-sonic-full-training`；起始 HEAD：
  `4efe77e2bee5fd2752376e59e302761d688b32c2`。修改前本地与
  `origin/feature/bumi-native-sonic-full-training` 的 ahead/behind 为 `0/0`，
  `git status --short` 无输出。
- 用户明确要求实现三数据源联合训练方案、提交并推送当前 feature 分支，再让
  `noetix-volc` 同分支执行 `git pull --ff-only`。本轮不获得停止或覆盖现有 hq4
  八卡训练的授权，因此只读核对其进程，不改变 PID、tmux、日志、checkpoint 或运行目录。
- 服务器只读核对确认新索引目标
  `/data/sonic_bumi3/datasets/bumi3_sonic_three_source_v1` 尚不存在，2 TB 数据盘
  约剩余 1.9 TB；三项实际来源路径为：
  - `/data/sonic_bumi3/datasets/bumi3_smpl_97660_v1`；
  - `/data/sonic_bumi3/datasets/hq4_pass50_v1`；
  - `/data/sonic_bumi3/datasets/hq_all_v2/built/robot_all` 中精确的 `mine__*`。
- 服务器构建前仍在运行 8 个训练 worker，启动参数指向旧 hq4 PASS50 数据和独立
  `hq4_pass50_v1_scratch_100k` 实验。本轮提交和后续快进不会替换已经载入进程内存的
  Python 代码，也不会自动重启该任务。

### 2. MotionLib 尾帧和随机截段修正

- `gear_sonic/utils/motion_lib/motion_lib_base.py` 新增
  `resolve_paired_frame_alignment`：
  - 默认 `strict` 保留 G1/H2 的严格行为；同为目标帧率时，在统一切片前要求已有
    SMPL 时间字段与 Robot 原始长度完全一致，不能把较长 SMPL 静默裁短；
  - 可选 `trim_trailing` 只接受 Robot/SMPL 都已经是目标帧率、
    `pose_aa/smpl_joints/transl` 内部长度一致且尾差不超过配置上限的配对；
  - 返回两侧共同的较短长度，只裁末尾，不重复帧、不插值、不改变起始时间；非法模式、
    布尔型/负数上限、错帧率、缺字段、内部错长或超过上限均在 FK 前失败。
- 同名 SMPL 现在先于 Robot 随机截段载入。Robot 和 50 Hz SMPL 共用同一
  `[start:end]`，修复旧实现在 `max_len` 随机截取 Robot 后仍把整段 SMPL 输入网络的
  时间错位；`pose_aa`、`smpl_joints`、`transl` 三个字段使用相同窗口。
- freeze-frame augmentation 继续使用同一 Robot 源帧索引换算到目标帧率，并同步冻结
  三个 SMPL 字段。Robot-only 动作保持 `curr_smpl_data=None`，不会进入配对对齐逻辑。
- BUMI3 `sonic_bumi3.yaml` 显式启用 `mode=trim_trailing`、
  `max_frame_delta=2`。因此 50 Hz 大集可裁 0–2 个尾帧，30 Hz Mine Robot-only 仍由
  既有 Robot FK 插值到 50 Hz；若未来误把 30 Hz Robot 和 SMPL 配在一起会立即失败，
  不会用该开关掩盖帧率契约错误。

### 3. 三来源全量构建与审计工具

- 新增 `gear_sonic/tools/build_bumi3_three_source_dataset.py`，文件头以中文详细说明
  输入、字段级坐标契约、降级边界、原子发布和验证能力。工具默认锁定计划中的来源
  计数：大集 train `92443`、test `5217`，hq4 Robot `2790`、SMPL `2788`，
  Mine `99`。
- 发现阶段递归建立 basename 唯一索引，要求大集 train/test 完整配对、hq4 SMPL 是
  Robot 子集、三训练来源 key 不冲突、train/test 不交叉。整个 `hq_all_v2` 不进入索引，
  只接受精确 `mine__` 前缀，避免和 hq4 PASS50 重复训练。
- 每条 Robot 全量检查外层 key、必要字段、有限值、50/30 Hz 来源契约、
  `(T,22,3)` pose、`(T,21)` DoF、`(T,4)` root、四元数单位范数和 xyzw 顺序；
  还统计根倾角和 Z 高度，以来源级中位数/横躺帧比例阻止系统性错误坐标。
- 每条 SMPL 全量检查 `pose_aa/transl/smpl_joints/fps`、50 Hz、有限值和内部长度，
  并根据 24 个关节的 XYZ extent 判断 `smpl_joints` 是否以 Z 为人体主轴。
  `pose_aa/transl` 仍按 Y-up 源字段处理，不对整份 PKL 做错误的统一旋转。
- 同名配对复现 SONIC 的 SMPL Y-up 到 Z-up 左乘、SMPL base rotation removal，
  并用 BUMI3 `pose_aa` 的 index 1 计算 `waist_yaw_link` 世界姿态；对全部共同帧计算
  waist/SMPL 根姿态中位差。帧差超过 2、SMPL 契约错误或中位差超过 45 度时，
  只把该 SMPL 降级为 Robot-only；Robot 不合格则整次构建失败。
- 每个实际发布的源文件均写入 SHA256。输出先在目标同级唯一 staging 目录构建，
  只创建绝对软链接，不复制、不重写、不重采样源 PKL；同时生成 train/test JSONL
  manifest、`summary.json` 和 `provenance.json`，结构验证通过后才用 `os.replace`
  原子发布。目标已存在时拒绝覆盖，失败时只清理本工具创建的 staging 目录。
- `validate` 子命令验证清单、软链接目标、计数、配对状态和 train/test 隔离；正式模式
  重新计算全部源 SHA256，`--skip-hash-verification` 仅允许调试结构。
- 配置删除旧 `hq_all_v2` 的 55 条静态 key 清单，设为 `exclude_motion_keys: []`。
  新三源构建根据当前全量数据重新决定配对状态，避免用旧名单误删同名新数据。
- `validate_bumi3_training_coordinates.py` 继续作为旧 hq_all_v2 的历史审计工具：新配置
  清单为空时仍要求从旧目录独立检出恰好 55 条；若未来传入非空完整清单，则继续要求
  检测集合和配置完全相同。它不再把旧 55 条描述为新三源训练的活跃隔离名单。

### 4. 新增测试与兼容性门禁

- `gear_sonic/tools/test_bumi3_paired_frame_alignment.py` 覆盖 strict 默认值、同帧率
  strict 错长拒绝、0/1/2 帧双方向裁剪、超过两帧拒绝、帧率错误、缺字段、字段内部
  错长和非法配置。测试还用轻量假 FK 真正执行 `load_motion_with_skeleton`，验证随机
  截段和 freeze-frame 后 Robot/SMPL 四项时间数据逐帧一致。
- `gear_sonic/tools/test_build_bumi3_three_source_dataset.py` 构造缩小的三来源数据，
  验证 2 帧尾差配对、hq4/Mine Robot-only、旧公开动作不重新进入索引、自然按动作数
  采样、train/test 隔离、绝对软链接和 SHA256 复核。
- `validate_bumi3_integration.py` 将 resolved 配置门禁更新为：活跃隔离列表为空，
  `paired_frame_alignment` 必须精确为 `trim_trailing/2`；G1 `sonic_release`、H2 compose、
  网络层、FSQ、token、PPO、奖励和控制参数的原有校验保持不变。

### 5. 本地实际验证结果

- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m pytest -q` 后接本轮两个新测试、
  两个 BUMI3 数据准备测试、`test_bumi3_sim2sim.py` 和
  `test_tracking_anchor_semantics.py`：`36 passed in 3.92s`；仅有 1 条已有
  `DeprecationWarning: invalid escape sequence '\\*'`，未造成失败。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m compileall -q gear_sonic`：通过。
- `/home/weili/miniconda3/envs/env_isaaclab/bin/python
  gear_sonic/tools/validate_bumi3_integration.py`：退出码 `0`，输出
  `BUMI3 原生 SONIC 集成验证通过`；同时完成 G1/H2/BUMI3 Hydra compose 和 Isaac Sim
  动态机器人配置导入。实测 resolved 数值为 `sim_dt=0.005`、`decimation=4`、控制频率
  `50 Hz`、`target_fps=50`、`action_dim=21`、FSQ 总维度 `64`、actor proprioception
  `690`、tokenizer flat `1262`、critic `1245`、dynamic decoder `754 -> 21`。
- `git diff --check`：通过。
- 尝试运行当前 `env_isaaclab` 内的 Black/Ruff，两个模块均未安装，因此没有把格式检查
  写成通过；新增文件人工限制在 Black 100 字符行宽，语法和 compileall 已通过。

### 6. 尚未执行项、风险与回滚

- 首次代码提交前尚未在服务器执行 100,431 条 train/test 全量数据审计、索引发布和
  第二次全量哈希复核；这些操作必须先让服务器快进到包含本工具的提交，实际结果会在
  后续记录中补录，当前不能写成通过。
- 本地没有三来源真实数据，所以尚未运行 MotionLib 三源真实加载、1-env reset/step、
  16-env 100-step 或 8 卡 100-iteration smoke。服务器 8 张 GPU 正被用户要求保留的
  hq4 正式训练占用；不会为了 smoke 停止该任务，最终会明确记录能执行和不能执行的层级。
- 全量 SHA256 会产生明显磁盘读取；绝对软链接节省容量但依赖三个源目录保持原路径。
  后续 `validate` 会在源文件内容或链接目标漂移时失败。
- 本轮没有修改 PPO、网络主体、奖励、事件、终止、机器人资产、关节顺序、
  `sim_dt`、`decimation` 或控制频率。回滚代码时应同时恢复 MotionLib 对齐函数、
  BUMI3 两项配置、两个新测试、构建工具及两个验证脚本的对应门禁；发布的数据索引
  只是源文件软链接，可在确认没有训练使用后单独处理，但不得删除或重写三个源目录。

### 7. 首次服务器全量审计、SHA 复核和历史审计结果

- 提交 `851eda706649ec8296736bbfc654668d6b2b00be` 已推送 GitHub；服务器仓库在
  同一 feature 分支、工作区干净的前提下由 `4efe77e` 执行普通
  `git pull --ff-only`，正常 fast-forward 到 `851eda7`。两端均未执行 force push、
  merge、rebase、stash、reset 或 clean。
- 服务器 `liwei_lab` 同步后执行和本地相同的 6 组相关测试：
  `36 passed in 6.62s`；`compileall -q gear_sonic` 与 `git diff --check` 通过。
- 在独立 tmux `bumi3_three_source_build` 执行全量 build。发现记录数精确为
  `100549 = 95332 train + 5217 test`，逐条审计完成后原子发布到用户指定目录；
  构建内结构验证输出 `BUMI3_THREE_SOURCE_VALIDATE=PASS`。
- 首次真实汇总不是计划前的估计值：train Robot `95332` 全部通过；可配对 SMPL
  `95132`，Robot-only `200`。其中预期无 SMPL 为 hq4 `2` 加 Mine `99`，另有
  大集 train `99` 条 waist/SMPL 根姿态中位差超过 45 度而降级。它们的帧差为
  `0:44`、`-1:50`、`-2:5`，所以不是尾帧裁剪制造的异常；角差范围约
  `45.0421°` 至 `177.3349°`。hq4 的 2788 条配对全部通过，test 的 5217 条
  全部通过。
- 自然按动作条数的实测来源概率为：大集 `0.96969538035497`、hq4
  `0.029266143582427726`、Mine `0.0010384760626022743`，没有增加来源权重。
- 来源级坐标实测：Robot 根倾角 clip 中位数分别约为大集 train `6.0823°`、
  test `6.0210°`、hq4 `8.2705°`、Mine `5.6613°`；各含 SMPL 来源的
  `smpl_joints` 人体主轴均为 Z。通过配对的根姿态中位差数据集内中位数为
  大集 train `9.6749°`、test `8.8959°`、hq4 `16.1124°`。
- 在独立 tmux `bumi3_three_source_hash` 调用不带跳过参数的正式 `validate`，
  第二遍重新计算 `100549/100549` 个 Robot/test manifest 条目的源 SHA256；输出
  `BUMI3_THREE_SOURCE_VALIDATE=PASS`。该结果证明发布后的链接、清单、第一次记录
  的哈希和第二次独立读取一致。
- 更新后的旧数据历史工具对 `hq_all_v2` 全量 3162 对运行，输出
  `BUMI3_TRAINING_COORDINATES=PASS`：仍独立检出 55 条历史异常，活跃配置隔离数为 0，
  正常保留集合中位角 `13.636874°`、最大 `44.710664°`，历史异常最小
  `45.181285°`。这证明清空新配置静态名单没有破坏旧数据审计能力。

### 8. 数据资产/关节顺序门禁补强及原因

- 服务器当前 `/home/liwei/legged_lab` 并不是本集成锁定的参考快照：其 `bumi.py`
  SHA256 为 `41e39f9c...5037`，手臂 velocity limit 已由 12 改为 30；其
  `bumi3.xml` SHA256 为 `8639f0da...580`，waist 轴为 `-Z` 且 arm-roll 限位不同。
  本地用户明确指定的 `/home/weili/legged_lab` 对应 SHA 仍是验证器锁定的
  `74aaeca9...03e`、`041c81e8...edf`。本轮没有修改任何 legged_lab 文件。
- 大集自身归档了 `/meta/bumi3.source.xml`，SHA256 为
  `db4f51fc64030a99a69f0592852c4436e5495eebf8941f89c727a1410a20c1a4`；其报告明确
  记录 DoF/body 双重重排和 waist_yaw 取反。程序化解析确认它使用 waist `+Z`、
  正确 arm-roll 限位，并与当前 SONIC MJCF 的 21 关节遍历顺序、轴、限位完全一致；
  两份 XML 的总体 SHA 不同仅因为归档版使用 `pelvis` 名称和另一套 mesh 相对路径。
- hq4 provenance 的 `target_mjcf_sha256` 和 hq_all_v2 Mine provenance 的
  `mjcf_sha256` 均为当前 SONIC MJCF
  `02874afebbe30ba1f90218394c8f9953f5d7a808e6b9950e7964c731da6dfbfe`。
- 根据上述现场，继续只检查 `dof.shape == 21` 不足以抵御另一套 BUMI3 轴符号或
  同维度错序。本轮再次修改 `build_bumi3_three_source_dataset.py`：
  - 从当前 MJCF body 遍历自动读取 21 个名称、单位轴和限位，并断言名称精确等于
    BUMI3 MuJoCo 顺序；
  - 构建前验证大集归档 MJCF 的关节顺序/轴/限位，验证 hq4 和 Mine provenance
    的目标 MJCF SHA，并把全部路径、SHA、顺序、轴和限位写入 summary/provenance；
  - 每条 Robot 对全部帧执行
    `pose_aa[:,1:,:] == dof[:,:,None] * current_mjcf_axes`，容差 `1e-6`。这会同时锁住
    pose 节点顺序、dof 顺序、关节轴和 waist 取反结果；真实三来源分层抽查误差均为 0；
  - 检查可选 `start_time/time_offset/timestamps` 等字段。实际 PKL 没有显式时间字段，
    因此两侧都按“数组 index 0 等于 0 秒”解释，只允许裁末尾；未来若任一来源声明
    不同非零起点，该 SMPL 会降级为 Robot-only，不能用尾帧对齐掩盖头部偏移。
- `test_build_bumi3_three_source_dataset.py` 增加真实 MJCF/provenance 的缩小版门禁、
  waist 使用负 Z 时 Robot 致命拒绝、同名 Robot/SMPL 时间起点差 0.02 秒时只降级
  SMPL 的测试。补强后六组本地相关测试为 `38 passed in 3.99s`，compileall 和
  `git diff --check` 再次通过。

### 9. 仿真 smoke 现场边界

- 首次服务器 1-env smoke 尚未创建环境，就在 `_validate_asset_provenance` 被服务器
  漂移后的 `/home/liwei/legged_lab/bumi.py` SHA 拦截，退出码为 1。因此这次结果只能
  记为“参考路径错误，仿真未执行”，不能记为 MotionLib、reset 或 step 失败。
- 本轮将以不修改 legged_lab 的方式，把本地锁定参考文件复制到独立临时验证目录并通过
  `BUMI3_REFERENCE_ROOT` 显式指定后重跑。临时目录只服务验证，不参与训练资产加载；
  训练和 MotionLib 仍以仓库 `gear_sonic/.../bumi3.xml` 为准。
- 原 hq4 八卡任务在本轮开始时确实有 8 个 worker；之后于 iteration 4855、日志时间
  `15:31:43` 停止，`last.pt` 时间为 `15:31:26`。日志末尾没有 Python traceback，
  dmesg/journal 未检出 OOM、killed process 或 GPU Xid。本 Agent 没有调用 kill、发送
  信号、关闭其 tmux、删除进程或覆盖实验目录；停止原因目前不能从日志确认。

### 10. 增强门禁重建结果与 AppLauncher 启动修正

- 关节/时间门禁提交 `02dd0131e0873d2ffb027f681902e6a6702b1f58` 已推送并在
  服务器同分支再次 `git pull --ff-only`。同步后服务器新增测试结果为
  `14 passed in 2.24s`，`py_compile` 通过。
- 重建前确认没有进程引用新三源索引，将首次由 `851eda7` 生成的目录完整移动到
  `/data/sonic_bumi3/datasets/bumi3_sonic_three_source_v1.pre_joint_contract_851eda7`。
  这是可恢复改名，不是删除；其清单和报告仍保留。随后用增强门禁重新构建用户指定的
  原目标 `/data/sonic_bumi3/datasets/bumi3_sonic_three_source_v1`。
- 增强重建再次审计 `100549/100549` 条并通过；所有真实 Robot 的全帧
  `pose_aa/dof/current-MJCF-axis` 最大误差均未超过 `1e-6`，否则构建会整体失败。
  配对状态和首次结果完全一致：train Robot `95332`、paired SMPL `95132`、
  Robot-only `200`、test paired `5217`。
- 新 summary/provenance 实际写入：当前 MJCF SHA
  `02874afb...bfe`、大集归档 MJCF SHA `db4f51fc...c1a4`、hq4 provenance SHA
  `246464d36e92b41372105ecb7577bada7f47837c822f28d36075b6505ad6ebbc`、Mine
  provenance SHA
  `bc8debdb3604164acfeaa8a801438008c86784a8dc3593e339da26b03cca66b6`，以及完整
  21 关节名称、轴和限位。重建后正式 rehash 又完成 `100549/100549`，输出
  `BUMI3_THREE_SOURCE_VALIDATE=PASS`。
- 使用临时锁定参考目录重跑 smoke 后，参考 SHA 门禁已通过，但服务器 pip 版
  `isaaclab` 顶层包只暴露 `isaaclab.app`，第一次报
  `ModuleNotFoundError: isaaclab.sim`；手工只加入 core source 后又报
  `ModuleNotFoundError: isaaclab_contrib`。两次均发生在环境创建前，不能记为 reset/step
  失败。这也证明用手工拼接单个 PYTHONPATH 不是可靠的训练等价启动方式。
- `gear_sonic/tools/validate_bumi3_integration.py` 改为使用项目训练入口同款的
  `isaaclab.app.AppLauncher`，而不是直接构造 `isaacsim.SimulationApp`。AppLauncher
  负责注册 pip 安装内的 core/contrib/assets 等 source 路径，并注入 physics CUDA
  device 和 headless kit 参数；验证脚本增加 `--no-window` 且不启用 camera。
- 本地 `env_isaaclab` 使用 AppLauncher 重新运行完整 integration validation，退出码 0，
  21 DoF、22 bodies、G1/H2/BUMI3 Hydra 和全部 resolved 网络/频率数值仍通过；输出
  `smoke: 未请求`。该修改不涉及环境、奖励、训练算法或数据契约，服务器真实 1/16-env
  smoke 必须等本提交同步后再执行。

### 11. 为近十万动作补充 MotionLib 启动元数据

- AppLauncher 修复提交 `ef90ad488d41ce1d428186e10778ae339e879e0a` 已推送并在
  服务器快进同步。真实 1-env smoke 随后成功创建场景、启动仿真，MotionLib 识别到
  `95332` 条动作，说明 AppLauncher、IsaacLab 模块、BUMI3 场景和三源目录均已打通。
- 该 smoke 没有继续到 reset/step：启用 adaptive sampling 时，MotionLib 要为全库建立
  稳定 bin，但新软链接目录没有 `metadata.pkl`，因此 `init_adaptive_sampling` 退化为
  逐条打开全部 95332 个 Robot PKL 读取 `length/fps`。进程在约 80 秒时仍以 200% 以上
  CPU 执行文件扫描；这不是死锁或数据错误，但 8 卡正式训练会让每个 rank 重复扫描，
  启动延迟和磁盘压力不可接受。
- 本 Agent 只向自己创建的 tmux `bumi3_three_source_smoke1_final` 发送 Ctrl-C，停止该次
  未完成 smoke；没有影响用户训练进程。shell 因 Isaac Sim 的信号处理最终记录
  `SMOKE_EXIT=0`，但本记录明确不把它当作 reset/step 通过证据。
- `build_bumi3_three_source_dataset.py` 现在根据已经全量审计的 manifest，在 train/test
  各自 `robot_all/metadata.pkl` 写入最小 `{key: {length, fps}}`。该文件只缓存整数帧数和
  浮点帧率，不包含动作数组、不复制源文件、不改变采样权重，也不改变任何源 PKL SHA。
- `validate_index` 新增 metadata 顶层类型、完整 key 集、逐动作 length/fps 与 manifest
  一致性检查；Robot 软链接计数显式排除 `metadata.pkl`，防止把元数据误算成动作。
  provenance 增加 `motionlib_metadata=generated_length_and_fps_only`。
- 缩小版构建测试新增 train metadata 精确内容断言。相关六组本地回归仍为
  `38 passed in 3.88s`，`compileall -q gear_sonic` 与 `git diff --check` 通过。
- 该修正需要再次生成目标索引后才能生效；仍采用“完整移动旧索引为可恢复备份，再构建
  原目标路径”的方式，不会删除已有报告或源数据。完成后的真实 smoke 必须看到
  MotionLib 从 metadata 获取全库长度，再实际完成 reset/step 才能记为通过。

### 12. 修正配对尾帧裁剪后的 adaptive sampling 有效长度

- metadata 提交 `86faf55f2f4b59bef17f82a7d136944d48aaca36` 已推送并同步服务器。
  重建前把上一版增强索引完整移动到可恢复目录
  `/data/sonic_bumi3/datasets/bumi3_sonic_three_source_v1.pre_metadata_02dd013`，随后
  重新构建目标路径。train/test metadata 大小约为 5.1 MB/279 KB，配对计数保持
  `95332/95132/200/5217` 不变，正式第二遍 SHA 再次完成 `100549/100549` 并通过。
- 带 metadata 的 1-env smoke 退出码为 0，明确输出 `smoke: 通过`：完成场景创建、
  MotionLib 全库索引、1 条动作加载、环境 setup、reset 和 1 次 step；验证器递归检查
  observation/action/reward 没有 NaN/Inf。该轮加载日志从 `Loaded 95332 motions` 到
  `Loading motions with 1 jobs` 不再出现近十万次源 PKL 文件扫描。
- 16-env、100-step smoke 随机选中 `walk_ff_stop_180_R_002__A047` 后，在 manager
  初始化阶段 fail-fast：`Adaptive sampling frame count mismatch`，具体为
  `adp_samp=375, loaded=374`。该条是允许的一帧尾差配对；FK 按
  `trim_trailing` 使用共同 374 帧，而初版 metadata 错误写入 Robot 原始 375 帧。
  因此该失败准确定位在 curriculum 长度缓存，不是 NaN、坐标错误或 PhysX 摔倒。
- `build_bumi3_three_source_dataset.py` 对通过配对新增 `aligned_source_frames`，值为
  `min(robot_frames, smpl_frames)`；MotionLib metadata 中 PAIRED 使用该有效长度。
  所有 Robot-only 条目继续使用完整 Robot 帧数，尤其不能让缺失、错坐标或被降级的
  SMPL 缩短仍可训练的 Robot 数据。
- 构建测试把两帧尾差合成配对的 metadata 期望从 Robot 原始 10 改为共同 8，并锁定
  manifest 的 `aligned_source_frames=8`。修正后相关测试 `38 passed in 3.90s`，
  compileall 和 `git diff --check` 通过。
- 该修正必须再次重建目标索引并重新运行 16-env 100-step；在看到实际退出码 0 前，
  本记录不会把 16-env 写成通过。1-env 的既有通过证据仍有效，因为它随机选中的配对
  没有触发 metadata/FK 长度差。

### 13. 最终三源索引重建、元数据和哈希门禁

- 有效长度修正提交 `ab5dc9db0d346f5f284067e9acd8cc51980f0b71` 已推送 GitHub，
  `noetix-volc` 在同一 `feature/bumi-native-sonic-full-training` 分支执行普通
  `git pull --ff-only` 后，本地、GitHub 和服务器 HEAD 完全一致，服务器工作区干净。
- 最终重建前确认没有构建、验证或训练进程引用目标索引。上一版目录没有删除，而是完整
  改名保留为
  `/data/sonic_bumi3/datasets/bumi3_sonic_three_source_v1.pre_aligned_metadata_86faf55`；
  另外两次历史重建也分别保存在 `.pre_joint_contract_851eda7` 和
  `.pre_metadata_02dd013`，需要回滚时仍可核对原报告。
- 最终 build 在 tmux `bumi3_three_source_final_build` 中完成 `100549/100549` 条全量
  数值审计并原子发布。正式路径仍为
  `/data/sonic_bumi3/datasets/bumi3_sonic_three_source_v1`，日志为
  `/data/sonic_bumi3/build_logs/bumi3_three_source_final_build_ab5dc9d.log`。
- 最终计数为 train Robot `95332`、train paired SMPL `95132`、Robot-only `200`、
  test Robot/SMPL 配对 `5217`。Robot-only 中 `101` 条原本就没有 SMPL，另 `99` 条
  是 Robot 合格但 SMPL 坐标配对中位角超过 45 度；没有丢弃这些 Robot 动作。
- `61346` 条合法配对存在末尾网格差，其中 SMPL 相对 Robot 少 1 帧为 `53569` 条、
  少 2 帧为 `7777` 条。逐条读取 `train_manifest.jsonl` 和 `metadata.pkl` 后确认：
  所有 PAIRED metadata `length` 均等于 `aligned_source_frames=min(robot, smpl)`；
  所有 Robot-only metadata 仍等于完整 Robot 帧数，输出 `metadata_alignment=PASS`。
- 独立 tmux `bumi3_three_source_final_hash` 再次调用不带跳过参数的正式 `validate`，
  重新读取并计算 `100549/100549` 条 manifest 源文件 SHA256，输出
  `BUMI3_THREE_SOURCE_VALIDATE=PASS`。日志为
  `/data/sonic_bumi3/build_logs/bumi3_three_source_final_hash_ab5dc9d.log`。
- 最终来源自然采样概率仍为：大集 `0.96969538035497`、hq4 PASS50
  `0.029266143582427726`、Mine Robot-only `0.0010384760626022743`；代码没有增加
  来源权重，也没有把旧 `hq_all_v2` 四库动作再次加入训练。

### 14. 16 环境和八卡 100 轮真实运行结果

- 服务器使用仓库训练同款 `isaaclab.app.AppLauncher` 执行 16-env、100-step：

```bash
BUMI3_REFERENCE_ROOT=/tmp/bumi3_reference_02dd013 \
/root/miniconda3/envs/liwei_lab/bin/python \
  gear_sonic/tools/validate_bumi3_integration.py \
  --smoke \
  --motion-file /data/sonic_bumi3/datasets/bumi3_sonic_three_source_v1/train/robot_all \
  --smpl-motion-file /data/sonic_bumi3/datasets/bumi3_sonic_three_source_v1/train/smpl_all \
  --num-envs 16 --iterations 100 --device cuda:0
```

  实际完成场景、MotionLib、reset 和 100 次 step，输出 `smoke: 通过`；验证器递归检查
  observation、action 和 reward 均无 NaN/Inf。日志为
  `/data/sonic_bumi3/build_logs/bumi3_three_source_smoke16_ab5dc9d.log`。
- 16-env resolved 实测：URDF/MJCF 为 `21 DoF/22 bodies`，`sim_dt=0.005`、
  `decimation=4`、控制频率 `50 Hz`、`target_fps=50`、`action_dim=21`、FSQ 总维度
  `64`、actor proprioception `690`、tokenizer flat `1262`、critic observation
  `1245`、dynamic decoder `754 -> 21`；Isaac/MuJoCo 双向关节和 body mapping 门禁通过。
- GPU 空闲后，在独立 tmux `sonic_bumi3_three_source_smoke_8gpu` 执行生产规模
  `8 rank x 4096 env/rank`、从随机初始化开始的 100-iteration smoke；命令没有使用
  checkpoint、resume 或正式训练目录。8 个 rank 都打印 `Loaded 95332 motions` 并完成
  environment setup，最终 tmux pane `dead_status=0`，保存了 `last.pt`。
- 八卡 smoke 累计 `78,643,200` timesteps、`3,276,800` episodes，总耗时
  `362.55 s`；首轮/末轮吞吐分别为 `159254/218651 steps/s`，运行中 8 卡显存约
  `15.7--16.6 GiB`、利用率通常约 `87%--89%`。完整日志未发现 traceback、
  AssertionError、RuntimeError、CUDA OOM、NCCL 或 NaN。
- smoke 实验主目录为
  `/data/sonic_bumi3/smoke_runs/TRL_BUMI3_Track/manager/universal_token/all_modes/`
  `sonic_bumi3_three_source_smoke_100iter_ab5dc9d-20260901_162058`；`last.pt` 大小约
  391 MB，TensorBoard event 含 `122` 个 scalar tags，100 个 step 的所有 scalar 均有限。
- TensorBoard 首轮到末轮：reward `0.918509 -> 0.905722`、episode length
  `14.69625 -> 14.23250`、value loss `0.149846 -> 0.022008`；三项 auxiliary loss tag
  都真实存在。末轮 termination 分量为 anchor-pos `0.000203`、anchor-ori
  `0.363922`、双肘 ee-body-pos `0.216858`、双脚 foot-pos `0.494853`、timeout
  `0.002024`。100 轮只证明端到端计算和数值稳定，不能据此宣称策略已收敛。
- resolved `config.yaml` 现场解析确认：anchor 是 `waist_yaw_link`；encoder keys 仅
  `g1/smpl`；tokenizer 数据项仅 `encoder_index`、Robot 两项和 SMPL 两项；aux loss
  仅 `g1_recon/g1_smpl_latent/reencoded_smpl_g1_latent`；
  `wrist_mujoco_dof_indices=[]`。活动配置中不存在 Teleop encoder、Teleop tokenizer、
  Teleop auxiliary loss 或 G1 腕部索引 `[19,20,21,26,27,28]`。
- 服务器无 X Server 时仍打印既有 Vulkan/GPU Foundation renderer 错误，但随后各 rank
  都完成 headless PhysX、MotionLib、网络、DDP 和 PPO 100 轮并以 0 退出；因此这些日志
  记录为无窗口渲染噪声，不伪装成已修复，也不把它们误判为训练失败。

### 15. 正式八卡训练与 TensorBoard 命令

- smoke 发现各 rank 若分别解析 `${timestamp}`，秒边界可能生成两个候选目录；DDP 计算
  不受影响，但正式任务必须显式给出同一个 `experiment_dir`，保证 checkpoint、Hydra
  配置和 TensorBoard 只落入一个目录。下面命令先在当前 shell 生成一次唯一目录，再把
  该固定字符串传给所有 rank：

```bash
cd /home/liwei/GR00T-WholeBodyControl
BUMI3_RUN_ID="$(date +%Y%m%d_%H%M%S)"
BUMI3_EXP_DIR="/data/sonic_bumi3/runs/TRL_BUMI3_Track/manager/universal_token/all_modes/sonic_bumi3_three_source_scratch_100k-${BUMI3_RUN_ID}"
BUMI3_LOG="/data/sonic_bumi3/launch_logs/sonic_bumi3_three_source_scratch_100k-${BUMI3_RUN_ID}.log"
mkdir -p /data/sonic_bumi3/launch_logs

tmux new-session -d -s sonic_bumi3_three_source_8gpu \
  "/root/miniconda3/envs/liwei_lab/bin/accelerate launch --num_processes=8 \
    gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_bumi3 \
    +resume=false checkpoint=null auto_load_latest=false \
    use_wandb=false headless=True num_envs=4096 \
    base_dir=/data/sonic_bumi3/runs \
    exp_var=three_source_scratch_100k \
    experiment_dir=${BUMI3_EXP_DIR} \
    algo.config.num_learning_iterations=100000 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=/data/sonic_bumi3/datasets/bumi3_sonic_three_source_v1/train/robot_all \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/data/sonic_bumi3/datasets/bumi3_sonic_three_source_v1/train/smpl_all \
    ++manager_env.commands.motion.motion_lib_cfg.exclude_motion_keys=[] \
    2>&1 | tee ${BUMI3_LOG}"
```

- 查看训练 tmux：
  `ssh noetix-volc -t 'tmux attach -t sonic_bumi3_three_source_8gpu'`；脱离会话使用
  `Ctrl-b` 后按 `d`。本轮只运行了独立 100 轮 smoke，没有替用户启动 100000 轮正式长训练。
- 正式训练启动后，可另建 TensorBoard tmux（6017 避开现有 6006/6016）：

```bash
tmux new-session -d -s tensorboard_bumi3_three_source \
  "/root/miniconda3/envs/liwei_lab/bin/tensorboard \
    --logdir /data/sonic_bumi3/runs/TRL_BUMI3_Track \
    --host 127.0.0.1 --port 6017"
```

  本地建立隧道：`ssh -N -L 6017:127.0.0.1:6017 noetix-volc`，浏览器访问
  `http://127.0.0.1:6017/`。
- 回滚边界：代码可按本轮提交逐项反向提交；数据索引是软链接和元数据，只有确认没有训练
  使用时才可把当前目录改名并恢复某个 `.pre_*` 备份。三个源数据、现有 hq4 历史实验、
  本轮 smoke checkpoint 和日志均未删除或覆盖。
- 最终交付前，本地 `env_isaaclab` 再次运行六组相关 pytest，结果为
  `38 passed in 3.92s`，仅保留一条已有 invalid escape sequence DeprecationWarning；
  `python -m compileall -q gear_sonic`、BUMI3/G1/H2 完整 Hydra/资产/网络静态集成门禁和
  `git diff --check` 均通过。集成门禁输出仍为 `smoke: 未请求`，这是本地命令没有传真实
  服务器数据的准确说明；真实 1/16-env 和八卡 smoke 证据以上述服务器日志为准。

## 2026-09-01：增加一次性验证产物完成后强制清理规则

- 所属分支：`feature/bumi-native-sonic-full-training`；起始 HEAD：
  `dccf1328a4b791034e5a64d379348e0248bd00e6`。修改前本地工作区干净，服务器正式
  BUMI3 三源八卡训练正在独立目录运行，本次规则修改不停止、不重启、不覆盖该正式任务。
- 修改 `agent.md` 的“验证、合并与发布”章节，新增规则：Agent 为验证代码创建的 smoke、
  测试运行、短回合训练、临时 replay/导出在完成且记录证据后，必须删除该次验证专用的
  运行目录、checkpoint、event、导出、渲染、日志及已结束 tmux，保持目录结构清晰。
- 为避免“测试后删除”被错误扩大为破坏性清理，规则同时锁定删除前置条件：必须先确认进程
  结束，精确核对目标只属于该次测试且未被正式训练或用户任务引用；禁止宽泛路径、未解析
  变量、递归通配符和 `git clean`。测试源码、fixture、原始数据、正式训练/checkpoint、
  用户产物和共享缓存不属于清理范围；用户明确要求保留时记录路径和原因后保留。
- 本次只修改仓库治理文档，没有启动新的测试任务或生成新测试产物，也没有追溯删除历史
  产物，因此代码单元测试、仿真和训练验证不适用。提交前执行 Markdown diff 人工检查和
  `git diff --check`；回滚时只需反向提交本节及 `agent.md` 对应规则，不影响训练代码和数据。

## 2026-09-02：直接修正 BUMI3 MJCF 自碰撞与初始陷地

### 1. 变更边界与参考版本

- 所属分支：`feature/bumi-native-sonic-full-training`；起始 HEAD：
  `98717f7ee21114fb6ba668a4206c891578cade1e`。开始修改前本地与 GitHub 同步且工作区
  干净；本轮不改训练 URDF、SONIC 网络、观测、奖励、控制频率、执行器、PPO 或数据。
- 权威参考为
  `/home/weili/legged_lab/source/NoetixRobot/NoetixRobot/assets/robots/bumi3/mjcf/bumi3.xml`，
  SHA256 为 `041c81e8176c7f375302796deca28b141891a3c097d8e341e8d967b735466edf`。
  本地修正后 `bumi3.xml` SHA256 为
  `c4521504388c6eba296b8070fd80d73bb85c506b7346722031cefa3bcea11c04`。
- 验证器不再要求本地 MJCF 与参考逐字相同，而是排除获准修改的 `geom`、碰撞 default
  和 `meshdir` 后，继续严格锁定 body 层级、质量、质心、惯量、关节位置/轴/限位、site、
  actuator 和 sensor。这样既允许修复接触，又防止碰撞修改误伤参考动力学参数。

### 2. `bumi3.xml` 的实际修改

- 22 个原始 STL 全部保留为 `group=1` 的可视 mesh，并设置
  `contype=0/conaffinity=0`；白色 policy 机器人和红色参考影子仍使用完整 BUMI3 原始
  外观，不把 capsule 当作渲染模型。
- 新增 14 个 `group=3` 碰撞体。`base_link`、左右 `leg_roll_link`、左右
  `knee_pitch_link` 使用审核后的 5 个 capsule；位置、半径和长度严格对应训练 URDF：
  base 为 `pos=(-0.0013853,0,0.065525), radius=0.052, length=0.12`，leg-roll 为
  `pos=(0,0,-0.02), radius=0.03, length=0.08`，knee 为
  `pos=(0.008475,0,-0.0894694), radius=0.025, length=0.13`。
- `waist_yaw_link`、双侧 arm-roll、双肘、双侧 ankle-pitch 和 ankle-roll 共 9 个 link
  保留 mesh collision；arm-pitch、arm-yaw、leg-pitch 和 leg-yaw 只渲染、不碰撞，避免
  高精度相邻网格在动作 reset 时互相嵌入。
- 机器人碰撞体设置 `contype=1/conaffinity=0`，地面设置互补的
  `contype=0/conaffinity=1`。MuJoCo 因此仍计算机器人与地面的接触，但不计算机器人
  link 之间的自碰撞；没有关闭膝、肘、脚或机身用于爬行/跪地的触地能力。
- 地面从 `Z=0` 调整为 `Z=-0.02 m`。这是按已取回大集动作的脚底基准消除首帧约
  1--2 cm 陷地；FineDance 首帧会相对地面留出约 1--2 cm 间隙并自然落地，本轮没有
  在 Python 中偷偷抬高/压低 policy 或让参考影子跟随机器人。

### 3. sim2sim 与验证代码同步

- `gear_sonic/utils/mujoco_sim/bumi3_sim2sim.py` 允许一个 body 同时拥有 visual 和
  collision geom，并让红色参考影子只复制 `group=1` 的 22 个原始 mesh；动力学模型
  仍直接加载本轮 `bumi3.xml`，没有运行时碰撞覆盖。
- `gear_sonic/tests/test_bumi3_sim2sim.py` 锁定 `36=22 visual+14 collision`、5 个
  capsule、9 个碰撞 mesh、碰撞名称集合、地面高度和 bitmask；静态 reset 必须无自碰撞
  与地面穿透，主动下移浮动根后又必须只产生地面接触。
- `gear_sonic/tools/validate_bumi3_sim2sim.py` 和
  `gear_sonic/tools/validate_bumi3_integration.py` 增加同样的资产来源保护、碰撞数量/类型/
  尺寸/名称/地面门禁。输出显式区分可视 geom、碰撞 geom、自碰撞开关和初始接触。
- `docs/source/getting_started/bumi3_sim2sim.md` 更新 XML 是唯一 MuJoCo 接触来源的说明，
  并记录 FineDance 与大集地面基准差异，避免以后又在 sim2sim Python 中隐式复刻 URDF。

### 4. 实际验证结果与边界

- 本地 `env_isaaclab` 执行 `pytest -q gear_sonic/tests/test_bumi3_sim2sim.py`：
  最终复验 `12 passed in 3.09s`；`python -m compileall -q gear_sonic` 和
  `git diff --check` 同时通过。
- 执行 `validate_bumi3_sim2sim.py --skip-smoke`：通过；实测 `nq=28,nv=27,nu=21`、
  22 bodies、22 visual mesh、14 collision geom、静态 reset `contacts=0`、自碰撞 `0`、
  地面穿透 `0`，控制契约仍为 `sim_dt=0.005,decimation=4,50 Hz,input=1170,action=21`。
- 对本地取回的 5 条 FineDance 和 5 条大集动作逐帧执行 MuJoCo FK/接触扫描，共
  `26844` 帧；修正前这些动作出现肘-髋、肘-腰、膝-踝等碰撞，修正后自碰撞计数为
  `0`。10 条动作首帧均为 `ncon=0`，没有初始地面穿透；整段仍有 `702` 个机器人-地面
  接触，证明触地碰撞没有被误关。爬行/落地帧可能有较深地面接触，这是动作本身的运行
  姿态，不等同于首帧陷地。
- 完整 `validate_bumi3_integration.py` 通过；实际组合结果仍为 `sim_dt=0.005`、
  `decimation=4`、`50 Hz`、`action_dim=21`，锚点仍是 `waist_yaw_link`，本轮未请求
  Isaac 1-env 数据 smoke。Isaac 启动日志中的 platforminfo/Vulkan 信息不影响静态门禁
  退出码 0。
- 使用现有 `model_step_018000_g1.onnx` 分别对 `axe_idle_R_102__A355.pkl` 和
  `finedance__001.pkl` 执行 100 控制周期真实回放，观测、动作和状态均有限且验证器通过；
  但 2 秒末 root 高度仍分别只有约 `0.0295 m` 和 `0.0545 m`。因此本轮可以确认
  “XML 初始自碰撞/陷地”已消除，不能把旧 checkpoint 仍会摔倒伪装成已解决；后者还需
  继续核对训练/部署动力学一致性或用修正后契约重新训练，不能仅靠资产门禁下结论。
- 本轮验证只使用已有模型和动作，不创建持久 checkpoint、event、导出、渲染或临时运行
  目录，因此没有一次性测试产物需要删除。回滚时反向提交本节列出的 7 个文件即可；参考
  `legged_lab` 仓库、训练数据和已有正式模型均未修改。

## 2026-09-02：统一 base_link 锚点并收敛 BUMI3 训练变量

### 1. 修改边界与最终决策

- 所属分支：`feature/bumi-native-sonic-full-training`；起始 HEAD：
  `7cf7616afaecf198199e60453e062d986083db8a`。开始修改前，本地与 GitHub ahead/behind
  为 `0/0` 且工作区干净；服务器同分支、同 HEAD、工作区干净，没有正在运行的 SONIC
  trainer。历史 TensorBoard 和数据预检 tmux 不属于训练进程。
- 用户最终决定将 BUMI3 训练、数据配对审计和 sim2sim 的命名锚点统一为浮动根
  `base_link`，不再混用 `waist_yaw_link`。所有锚点位置、姿态、线速度、角速度都沿用
  `TrackingCommand` 的命名 body 索引通路；`waist_yaw_link` 仍是正常机器人刚体和全身
  tracking body，不做全局重命名或删除。
- 当前只保留 `base_link` 的 COM 随机化。质量随机化、全关节 KP/KD 随机化、踝关节
  armature 随机化全部设为 `null`；四组 actuator 从项目自定义的
  `DelayedImplicitActuatorCfg(min_delay=0,max_delay=4)` 改为与 G1 相同的
  `ImplicitActuatorCfg`，彻底取消 0～4 个 physics-step 随机延迟。BUMI3 名义质量、
  COM、KP/KD、armature、力矩/速度限制和 action scale 数值没有改写。
- termination 不再保留 BUMI3 的严格覆盖：脚部位置继承 G1 的 `0.20 m`，锚点位置继承
  `0.15 m`，双肘位置继承 `0.15 m`，锚点完整姿态继承 `0.20`；adaptive、
  `down_threshold` 和 `root_height_threshold` 均继承原 G1 配置。`ee_body_pos` 仍只检查
  双肘，不重复检查双脚。
- 强跟踪点从 base/双肘/双脚五个 body 减为 `base_link` 加双肘三个 body，偏移均为零。
  双脚仍参与原有脚部 termination、feet acceleration 和其他发布版奖励，但不进入该强
  point-tracking reward，避免同一脚部误差被过度约束。奖励函数、权重、std、PPO、网络、
  `sim_dt=0.005`、`decimation=4` 和 50 Hz 控制频率没有修改。
- 明确不修改 MuJoCo Euler 积分器或 Python 显式 PD，因为它们与现有 G1 sim2sim 的
  基础实现一致。本轮只修正 BUMI3 非对角 `fullinertia` 暴露出的确定性坐标问题：
  `mj_objectVelocity(local=1)` 返回惯性主轴表达，而训练 `root_ang_vel_b` 使用 link 轴；
  sim2sim 现在先读世界角速度，再乘 `base_link` 世界旋转转置得到 link 局部角速度。

### 2. 修改文件与目的

- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bumi3.yaml`：统一 base
  anchor、COM/anti-shake/VR/reward 名称，关闭三类域随机化，继承 G1 termination 阈值，
  将 reward points 缩为三点。
- `gear_sonic/envs/manager_env/robots/bumi3.py`：四组执行器改用无延迟
  `ImplicitActuatorCfg`；保留 BUMI3 参考执行器的其余名义参数和按公式生成的 action scale。
- `gear_sonic/envs/manager_env/mdp/commands.py`：更新通用锚点姿态接口说明，明确 BUMI3
  配置选择 base，但不在通用实现硬编码机器人类型。
- `gear_sonic/trl/utils/order_converter.py`：BUMI3 三点 body 名称的第三项改为
  `base_link`；G1/H2 converter 未修改。
- `gear_sonic/config/sim2sim/bumi3_sonic.yaml`、
  `gear_sonic/utils/mujoco_sim/bumi3_sim2sim.py`：部署锚点与训练一致，并修正非对角惯量时
  base 角速度的 link 坐标表达；没有修改 Euler、PD、力矩裁剪、MJCF 或碰撞。
- `gear_sonic/tools/build_bumi3_three_source_dataset.py`：配对门禁直接比较 Robot
  `root_rot/base_link` 与训练处理后的 SMPL 根姿态，不再乘 waist 局部旋转；契约版本升级为
  `sonic.bumi3.three_source_base_anchor.v2`，默认输出改到独立的
  `/data/sonic_bumi3/datasets/bumi3_sonic_three_source_base_anchor_v2`，避免覆盖旧索引。
- `gear_sonic/tools/validate_bumi3_training_coordinates.py`：旧 hq_all_v2 只读审计也改为
  base 根锚点；旧 waist 契约下的固定 55 条不再被当作新契约真值，可用
  `--expected-bad-count` 显式设置门禁。
- `gear_sonic/tools/validate_bumi3_integration.py`、
  `gear_sonic/tools/validate_bumi3_sim2sim.py`：锁定 resolved base anchor、三点奖励、G1
  termination、仅 COM 随机化、无延迟 actuator 和 sim2sim base 锚点。
- `gear_sonic/tests/test_bumi3_sim2sim.py`、
  `gear_sonic/tests/test_tracking_anchor_semantics.py`、
  `gear_sonic/tools/test_build_bumi3_three_source_dataset.py`：锁定腰关节旋转不改变 base
  锚点/配对结果，并增加非对角惯量下 link 轴角速度回归。
- `docs/source/getting_started/bumi3_sim2sim.md`：把训练/部署共同锚点、诊断输出和观测来源
  更新为 base 契约。历史记录中的 waist 结论作为当时版本的审计证据保留，由本节明确取代。

### 3. 本地实际验证结果

- `/home/weili/miniconda3/envs/env_isaaclab/bin/python -m pytest -q` 运行本次相关的六组
  BUMI 测试文件，结果为 `41 passed in 4.80s`，仅有一条已有的 invalid escape sequence
  DeprecationWarning。单独首轮锚点/sim2sim/构建定向测试为 `19 passed in 3.55s`。
- 直接收集整个 `gear_sonic/tests` 时，被未安装的可选依赖 `msgpack` 阻断于
  `test_input_readers.py` 导入阶段；该错误发生在测试收集、与本轮 BUMI 修改无关，因此
  未把它伪装成代码失败，也未擅自改动环境依赖。
- `validate_bumi3_integration.py --device cpu` 退出码为 0；实际启动 headless Isaac Sim，
  验证 URDF/MJCF `21 DoF/22 bodies`、无延迟 actuator、BUMI3/G1/H2 Hydra compose、
  `sim_dt=0.005`、`decimation=4`、50 Hz、`action_dim=21`、FSQ `64`、actor proprioception
  `690`、tokenizer flat `1262`、critic obs `1245`、dynamic decoder `754 -> 21`。本地未提供
  服务器三源数据，所以该命令准确输出 `smoke: 未请求`。
- `validate_bumi3_sim2sim.py --steps 100` 退出码为 0：100 个控制周期 observation、action、
  torque、qpos、qvel 均为有限值；resolved anchor 为 `base_link`，模型仍为
  `nq=28,nv=27,nu=21`、22 bodies、22 个可视 mesh 和 14 个 XML 碰撞体。
- `python -m compileall -q gear_sonic` 与 `git diff --check` 均通过。上述命令未创建训练
  目录、checkpoint、TensorBoard event、导出或渲染文件，没有本轮一次性测试产物需要清理。

### 4. 待服务器闭环、删除边界与回滚

- 代码提交推送并由服务器 `git pull --ff-only` 后，必须用新 v2 默认路径全量构建和验证
  base-anchor 三源软链接索引。新索引通过前不得删除三个源数据：
  `bumi3_smpl_97660_v1`、`hq4_pass50_v1`、`hq_all_v2`。
- 新索引通过后，按用户授权精确删除旧三源派生索引及其 `.pre_*` 备份、未再引用的
  `hq_all_v1`、旧 BUMI3 训练 run/checkpoint，以及本地 `models/sonic_bumi3`。删除前必须
  再次核对无进程引用，实际删除路径、容量和结果将在本节后续记录，不能用计划冒充完成。
- 代码回滚可反向提交本节对应提交；数据回滚边界以新索引发布和旧目录实际删除记录为准。
  原始三源 PKL 始终保留，因此即使旧软链接索引被删除，也可用记录的构建命令重新生成。

### 5. 服务器首次 v2 构建的资产指纹门禁修正

- 上述代码提交 `13645ea066720f3b881967f49c867500388a3b19` 已推送 GitHub，服务器
  同分支通过 `git pull --ff-only` 快进到同一提交且工作区干净。首次执行 v2 build 时，
  构建器在发现来源阶段、创建目标目录前 fail-fast；日志为
  `/data/sonic_bumi3/build_logs/bumi3_base_anchor_v2_build_13645ea.log`。
- 失败原因是 hq4/Mine provenance 保存的目标 MJCF SHA256 为
  `02874afebbe30ba1f90218394c8f9953f5d7a808e6b9950e7964c731da6dfbfe`，而当前碰撞
  修正后 MJCF 为 `c4521504388c6eba296b8070fd80d73bb85c506b7346722031cefa3bcea11c04`。
  通过 `git show 7cf7616^:.../bumi3.xml | sha256sum` 确认前一个指纹恰好就是碰撞修正前
  仓库资产；提交 `7cf7616` 只修改已审核的 geom/地面，受保护的质量、惯量、关节、执行器
  和传感器签名仍由集成验证器逐项锁定。因此该失败是完整 XML 哈希把碰撞层变化误判为
  轨迹运动学不兼容，不是数据坐标、帧率或关节契约失败。
- `build_bumi3_three_source_dataset.py` 新增精确白名单：只接受当前完整指纹和上述已审核的
  碰撞修正前指纹。未知旧版本仍立即失败；大集归档 XML 仍逐项比较 21 个关节名称、遍历
  顺序、轴和限位，Robot PKL 仍逐条检查 dof/pose 轴符号。summary/provenance 额外记录
  当前指纹、允许指纹、hq4 实际指纹和 Mine 实际指纹，不能静默放宽。
- `test_build_bumi3_three_source_dataset.py` 新增当前指纹、碰撞修正前指纹通过以及任意未知
  指纹拒绝的三向回归。定向测试结果为 `5 passed in 0.15s`，相关文件 compileall 与
  `git diff --check` 通过。失败 build 未生成目标索引、checkpoint、event 或临时 staging，
  只保留上面的诊断日志作为本次正式数据构建审计证据。

### 6. 锁定服务器当前 2816 条 hq4 PASS 白名单

- 指纹修正提交 `99397122a4d54b4492c2334d0401ec05792ba7c3` 推送并同步服务器后，
  第二次 build 在计数门禁 fail-fast：当前 `hq4_pass50_v1` 含 Robot `2816`、SMPL
  `2815`，而旧代码仍锁定历史 `2790/2788`。日志为
  `/data/sonic_bumi3/build_logs/bumi3_base_anchor_v2_build_9939712.log`，目标 v2 目录仍未
  创建，tmux 已结束且退出码为 1。
- 现场检查确认这不是在旧目录里临时多放 26 个文件：当前 `meta/provenance.json` 和
  `meta/manifest.jsonl` 均于 2026-09-02 13:53 整套更新，provenance 明确声明
  `pass_count=2816`，质量报告 SHA256 为
  `2fc2c5865b86d38f656832985a50cd61611cc5979a533bb9a71f4fd65c2c3b20`。与旧 2790
  manifest 比较，旧集合有 10 条已不在新发布集，新集合新增 36 条，净增 26；因此不能把
  旧 key 列表直接套到已经替换的源目录，也不能假装仍是原 2790 资产。
- 按“旧数据删除、使用当前新数据”的边界，服务器从当前 hq4 manifest 原子生成固定
  `/data/sonic_bumi3/datasets/hq4_pass50_v1/meta/sonic_train_whitelist.txt`，包含 2816 个
  唯一 key，SHA256 为
  `85355027e47112b61201e5debe1d581a016bc4a597a99208cd33a1f75e1398f5`；相邻
  `sonic_train_whitelist.provenance.json` 保存 manifest/provenance/质量报告 SHA 和生成时
  `2816/2815` 计数。原 Robot/SMPL PKL 没有改写。
- 构建器现在必须读取上述白名单后再索引 hq4；白名单缺失、空行、重复 key、缺少 Robot
  或计数不符均立即失败。目录内未来新增但未进入固定 manifest 的 PKL 只记录到
  `hq4_ignored_*`，不会自动进入训练。默认 hq4 计数随当前发布契约更新为
  `2816 Robot/2815 SMPL`。
- 缩小版构建测试额外放入一条白名单外 hq4 Robot/SMPL，并确认最终 manifest 不包含它；
  定向结果为 `5 passed in 0.17s`，`git diff --check` 通过。该改动不改变大集 92443/5217
  或 Mine 99 条来源，也不改变逐 PKL 坐标、帧率、关节、配对和 SHA256 门禁。

### 7. v2 全量发布与移除外部机器人仓库验证依赖

- 提交 `a1665a4c5763b7b9cc1619c8029f3e7a2b84b8c2` 推送并同步服务器后，第三次
  build 对 `100575` 条训练/test 动作完成全量 PKL 数值审计，退出码 0，并原子发布
  `/data/sonic_bumi3/datasets/bumi3_sonic_three_source_base_anchor_v2`。独立 hash validate
  再次逐条复核 `100575/100575` 个源文件 SHA256，输出
  `BUMI3_THREE_SOURCE_VALIDATE=PASS` 且 tmux dead status 为 0。正式日志分别为
  `/data/sonic_bumi3/build_logs/bumi3_base_anchor_v2_build_a1665a4.log` 和
  `bumi3_base_anchor_v2_hash_a1665a4.log`。
- v2 最终静态计数：train Robot `95358`、paired SMPL `95222`、Robot-only `136`、test
  Robot/SMPL `5217/5217`。Robot-only 包含 `100` 条没有 SMPL 和 `36` 条 base/SMPL
  根姿态中位差超过 45 度的配对降级；没有丢弃合格 Robot。来源自然采样比例为大集
  `0.9694309863881373`、hq4 `0.0295308206967428`、Mine `0.001038192915119864`。
- 全量坐标统计中，大集 train 合格配对根姿态差中位数总体为约 `7.615°`、最大
  `44.441°`，Robot clip 根倾角中位数约 `6.082°`；test 对应约 `7.547°/28.310°` 和
  `6.021°`。hq4 配对根姿态差中位数总体约 `8.824°`、最大 `36.729°`，Robot 根倾角
  中位数约 `6.880°`。这些结果未发现数据集整体横躺；它们是数据静态门禁，不代替策略
  收敛证明。
- 新 v2 执行 1-env、1-step Isaac Lab smoke，实际加载 `95358` 条 MotionLib 元数据并抽取
  `walking_quip_360_R_002__A430_M`，完成场景、reset 和 step；action shape `21`、policy
  obs `690`、critic obs `1245`，tokenizer 只有 `g1/smpl` 五项，event 现场显示质量、
  armature、KP/KD 均为 `None`，只保留 `base_link` COM。输出 `smoke: 通过`，观测、动作、
  reward 无 NaN/Inf。服务器无图形设备的 Vulkan 报错是既有 headless 渲染噪声，PhysX 和
  smoke 仍完成。
- 该 smoke 首次调用暴露出验证器会在启动前读取 `/home/liwei/legged_lab` 的外部参考哈希。
  这不属于训练运行时依赖：实际 MotionLib、URDF、MJCF 和机器人配置始终从当前 SONIC
  仓库加载；但把外部仓库放在 smoke 前置门禁仍是错误耦合。按用户要求，
  `validate_bumi3_integration.py` 已删除外部路径发现、`BUMI3_REFERENCE_ROOT`、动态导入
  `NoetixRobot/bumi.py` 及逐文件外部比较，改为锁定 SONIC 仓库内 URDF、MJCF 和 mesh
  bundle 指纹，并继续用显式数值断言验证全部执行器、初始姿态、mapping 和 action scale。
- `validate_bumi3_sim2sim.py` 同样删除外部 MJCF 参数和比较，只读取仓库内
  `gear_sonic/data/assets/robot_description/mjcf/bumi3.xml`；sim2sim 使用文档同步说明没有
  `legged_lab` 或 `NoetixRobot` 依赖。外部仓库当前内容不会被复制、接受或用于训练。
- 去耦后本地重新运行集成验证，无任何外部环境变量即可通过；sim2sim 100 控制周期通过，
  BUMI 相关 pytest 为 `42 passed in 5.14s`，compileall 与 `git diff --check` 通过。服务器
  还需在同步该去耦提交后，不设置任何参考路径重跑 1-env smoke，结果将在后续记录补全。
