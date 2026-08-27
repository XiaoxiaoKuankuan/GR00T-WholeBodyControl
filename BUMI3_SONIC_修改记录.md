# BUMI3 原生 SONIC 修改记录

本文档记录 BUMI3 原生 SONIC 支持的实际修改、机器人参数来源、兼容性边界和验证证据。所有结论区分静态检查、Isaac Sim 配置导入、环境 reset/step 与真实训练；未执行的测试不会表述为已通过。

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
