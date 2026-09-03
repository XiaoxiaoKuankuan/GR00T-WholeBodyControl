# BUMI3 SONIC sim2sim

BUMI3 使用独立的 Python MuJoCo sim2sim 入口，不复用 G1 C++ 部署程序中的 29 电机、
Unitree DDS 和硬件映射。训练网络仍保留内部键名 `g1` 作为 Robot Encoder 的 checkpoint
兼容名称，因此实际部署文件应是 `model_step_XXXXXX_g1.onnx`；这不表示机器人是 G1。

## 1. 使用 `env_isaaclab` Conda 环境

```bash
conda activate env_isaaclab
python -m pip install "tyro==0.8.14" "typing_extensions==4.12.2"
```

本机该环境已经配置好 MuJoCo、ONNX Runtime、YAML、joblib 和上述兼容版 Tyro，后续
训练、ONNX 导出和 sim2sim 都可以使用同一个 `env_isaaclab`。固定 Tyro `0.8.14` 是为了
保留 Isaac Sim 5.1 要求的 `typing_extensions==4.12.2`；不要在此环境直接安装最新版
Tyro，否则会升级该依赖并破坏 Isaac Sim 的版本契约。

BUMI3 配置位于
`gear_sonic/config/sim2sim/bumi3_sonic.yaml`，默认加载
`gear_sonic/data/assets/robot_description/mjcf/bumi3.xml`。

如果在另一台机器建立不含 Isaac Lab 的纯 MuJoCo 环境，仍可使用
`python -m pip install -e "gear_sonic[sim]"`；项目的 `sim` extra 已固定兼容版 Tyro。

## 2. 从训练 checkpoint 导出联合 ONNX

在 Isaac Lab 环境中运行：

```bash
conda activate env_isaaclab
python gear_sonic/eval_agent_trl.py \
  +checkpoint=/absolute/path/to/model_step_016000.pt \
  +num_envs=1 \
  +headless=true \
  +export_onnx_only=true \
  +manager_env.commands.motion.motion_lib_cfg.motion_file=/absolute/path/to/bumi3_robot_motion \
  +manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/absolute/path/to/paired_smpl_motion
```

导出目录中的 `model_step_016000_g1.onnx` 是本入口需要的联合模型，输入为 Robot
tokenizer `480` 维加 actor proprioception `690` 维，总计 `1170` 维，输出为 BUMI3
IsaacLab 顺序的 `21` 维动作。

## 3. 运行 sim2sim

GUI 实时播放：

```bash
conda activate env_isaaclab
python gear_sonic/scripts/run_bumi3_sim2sim.py \
  --policy /absolute/path/to/model_step_016000_g1.onnx \
  --motion /absolute/path/to/bumi3_motion.pkl
```

GUI 默认同时显示两套完整原始 XML mesh 机器人：

- 不透明白色机器人：由 MuJoCo 直接加载 BUMI3 XML，XML geom 同时用于
  碰撞、动力学和渲染，qpos/qvel 就是 ONNX policy 的实际状态。
- 红色半透明机器人：动作文件的 root position、root quaternion 和 21 个关节经过同一
  BUMI3 MJCF FK 后得到的参考影子。

只有红色参考写入 MuJoCo viewer 的 decorative user scene，不参与接触、
碰撞、力矩或积分。白色 policy 不是 marker 或额外视觉代理，就是实际
MuJoCo 动力学模型。sim2sim 不读取、不复刻、不覆盖 Isaac Lab URDF
碰撞体契约。

参考状态的根高度始终来自动作文件，不会为了贴近已摔倒的实际机器人而下降。因此红色
影子若从开头就横躺，说明传给 sim2sim/训练的 Robot 参考仍有坐标或数据问题；红色影子
直立而不透明机器人快速摔倒，则应继续检查 checkpoint 学习质量、SONIC 观测/控制和
sim2sim 动力学契约。可以用 ``--reference-alpha 0.5`` 调整透明度，或使用
``--no-show-reference`` 关闭影子。

影子的 mesh 变换严格沿用参考脚本的渲染路径：从同一 BUMI3 XML
单独加载 ``ref_model``，配合独立 ``MjData`` 持有参考 qpos 并执行 FK 后
调用 ``mjv_updateScene``，再复制其中已经解析完成的 ``MjvGeom``（包括 ``pos``、``mat``、
``size``、``dataid`` 和 ``matid``）。不能从 ``geom_xpos/geom_xmat`` 自行重建 mesh
marker，否则会丢失渲染级 mesh 变换，并出现各 link 分离的“炸开”画面。

入口还会在推进仿真前打印 ``BUMI3_REFERENCE_POSE``，其中
``base_tilt_degrees``/``anchor_tilt_degrees`` 都以当前统一的 ``base_link`` 为对象，
表示其上轴与世界 +Z 的夹角：站立通常接近 0°，侧躺通常接近 90°。两项暂时保留是为了
兼容既有诊断输出；它们在当前契约下应一致。这项数值检查不替代完整动作可视化，但能
避免只凭相机角度误判。

服务器无显示、尽快运行 10 秒：

```bash
conda activate env_isaaclab
python gear_sonic/scripts/run_bumi3_sim2sim.py \
  --policy /absolute/path/to/model_step_016000_g1.onnx \
  --motion /absolute/path/to/bumi3_motion.pkl \
  --duration 10 \
  --headless \
  --no-real-time
```

加载器会把含根平移的每条动作整体做水平归零：整段 ``root x/y`` 减去首帧
``root x/y``，所以第一帧位于世界水平原点，后续每帧相对首帧的运动轨迹保持不变。
这个操作不修改 ``root z``，也不改变关节、根旋转或帧率。多动作 PKL/NPZ 或包含多个
clip 子目录的 CSV 根目录使用 `--motion-key NAME`。顺序默认值：

- SONIC 训练 PKL：`dof` 为 MuJoCo 顺序，`root_rot` 为 `xyzw`。
- G1 `MotionDataReader` 风格 CSV clip：`joint_pos.csv` 为策略/IsaacLab 顺序，
  `body_quat.csv` 第一个 body 为 root 且 quaternion 为 `wxyz`。
- NPZ：优先读取 `joint_order` 和 `quaternion_convention` 元数据；缺失时使用策略顺序和
  `wxyz`。

若实际文件不符合默认值，显式指定 `--joint-order mujoco|policy` 和
`--quaternion-order xyzw|wxyz`。加载器要求动作已经是 50 FPS，不会静默重采样。
默认还会像 G1 sim2sim 一样，把参考动作起始 yaw 对齐到机器人当前 yaw；需要观察原始
世界朝向差时可传入 `--no-align-reference-heading`。

运行器会优先用动作中的 `root_trans_offset/root_pos/qpos[:3]`、root quaternion、关节
位置和关节速度初始化 MuJoCo。旧 CSV 不含 `body_pos.csv` 时才回退到配置中的
`[0, 0, 0.4744]`，不再固定悬空在 `0.65 m`。Robot Encoder 的参考锚点明确使用
``base_link``：它等于动作中的浮动根世界四元数；当前 policy 状态也读取 MuJoCo
``base_link`` 的世界姿态和速度。训练端、数据配对审计和 sim2sim 因而使用同一语义。
reset 后的 10 帧 proprioception history 会按 Isaac Lab `CircularBuffer` 的首次写入规则，
用当前状态复制填满，而不是以 9 帧零值开头。

sim2sim 是 MuJoCo 闭环，所有碰撞完全以 `bumi3.xml` 为准。XML 里保留 22 个原始
link mesh 作为 `group=1` 的可视 geom，并把 14 个审核后的接触几何单独设为
`group=3`：base、双侧 leg-roll 和双侧 knee 使用简化 capsule，其余 9 个需要接触的
link 使用 mesh；arm-pitch/arm-yaw、leg-pitch/leg-yaw 不参与碰撞。地面 Z 基准为
`0 m`，与重定向数据和世界坐标原点统一；加载器不会改动动作原始根高度。机器人碰撞体使用
`contype=1/conaffinity=0`，地面使用互补的 `contype=0/conaffinity=1`，因此保留
机器人与地面的接触，但不会计算机器人 link 之间的自碰撞。运行器不会根据
Isaac Lab URDF 或其他仓库规则再次覆盖这些定义；启动验证会将运行时
`geom_type/bodyid/contype/conaffinity/pos/quat/size/friction/solref/solimp` 与重新加载
的 XML 编译结果逐数组比较，并检查静态 reset 无自碰撞和地面穿透。

旧大集若按 `Z=-0.02 m` 地面制作，切换到当前 XML 后可能出现约 1--2 cm 的初始穿地；
这是数据地面基准差异，应在数据转换或专用资产中显式处理，不应通过 sim2sim 自动修改
``root z`` 或让参考影子跟随 policy 根高度来掩盖。

ONNX 只保存网络权重与 1170→21 的张量接口，不包含参考轨迹、锚点 body 名称或 FK
结果；这些观测语义由 sim2sim 运行器负责重建。因此换动作文件或部署实现时仍必须使用
本配置和运行器，不能只凭 ONNX 文件名推断观测正确。

## 4. 验证

无需 checkpoint 和动作数据的 100 控制周期接口/有限值 smoke：

```bash
conda activate env_isaaclab
python gear_sonic/tools/validate_bumi3_sim2sim.py
```

真实 ONNX 和真实参考动作 smoke：

```bash
conda activate env_isaaclab
python gear_sonic/tools/validate_bumi3_sim2sim.py \
  --policy /absolute/path/to/model_step_016000_g1.onnx \
  --motion /absolute/path/to/bumi3_motion.pkl \
  --steps 100
```

`validate_bumi3_sim2sim.py` 只读取并锁定当前 SONIC 仓库内的 BUMI3 MJCF，不依赖
`legged_lab`、`NoetixRobot` 或其他外部机器人仓库。它检查所有 mesh、21 DoF、
22 robot bodies、映射、动作缩放、armature、输入输出维度和 NaN/Inf。

## 5. 边界

- 实际参数固定为 `sim_dt=0.005`、`decimation=4`、控制频率 50 Hz、参考 FPS 50。
- 动作经过 `default + action_scale * policy_action`，其中 action scale 始终由
  `0.25 * effort_limit / stiffness` 计算；PD torque 按 BUMI3 effort limit 截断。
- Python 入口只用于 MuJoCo sim2sim，不连接 BUMI3 实机总线。
- 零策略 smoke 只证明接口、顺序、维度和有限值，不证明训练 checkpoint 的动作质量；
  真实效果仍需使用对应训练数据、真实 ONNX 和指定动作回放确认。
