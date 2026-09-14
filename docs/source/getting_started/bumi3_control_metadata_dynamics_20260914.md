# BUMI3 ONNX 控制参数与 SONIC / Mimic 4340 动力学核对

2026-09-14，本地 MuJoCo 3.3.2、Python `/home/weili/miniconda3/envs/env_isaaclab/bin/python`。
起始代码 `fc67431b8093068b58be2d57e945ec6a2413485a`；所有实验使用原 100000 模型的
计算图及 `noetix_bumi3_bigset_10pairs_20260910/dataset.json` 十条动作。
本次用户要求：21 个驱动关节运行时 armature=0.01，PD 从 ONNX 读取。

## 1. 频率和稳定性

| 入口 | 物理 / 显式 PD | 策略推理 | 每策略周期物理步数 |
| --- | --- | --- | --- |
| G1 `run_sim_loop.py` + `g1_deploy_onnx_ref` 默认配置 | 200 Hz / 5 ms | 50 Hz / 20 ms | 通常 4，两个进程异步通信 |
| Mimic `sim2sim_mimic_vision_4340.py` | 200 Hz / 5 ms | 50 Hz / 20 ms | 4 |
| SONIC BUMI3 默认 | 200 Hz / 5 ms | 50 Hz / 20 ms | 4 |
| SONIC BUMI3 `--physics-substeps 5` | 1000 Hz / 1 ms | 50 Hz / 20 ms | 20 |

G1 的 C++ `publish_dt_=0.002` 是 DDS 命令发布频率 500 Hz；`control_dt_=0.02` 是
策略频率。MuJoCo 内每个 `SIMULATE_DT=0.005` 才根据最新 q/dq 重算一次 PD。
不要把 DDS 500 Hz 当成 MuJoCo 积分频率。两份 BUMI XML 编译默认 timestep 都是
0.002，实际运行器覆盖为 0.005 或 0.001，不能只读 XML 推断运行频率。

外部 PD 每个物理小步重新计算
`tau = clip(Kp * (q_default + action_scale * action - q) - Kd * dq, -limit, limit)`。
动作仍每 20 ms 更新，但缩短物理步长减少 PD 力矩保持期间的状态变化和接触离散误差，
使快速关节和触地冲击更容易数值稳定。不能仅用 Kp 的大小跨机器人比较；有效惯量、Kd、
armature、碰撞形状与软接触共同决定闭环行为。原 BUMI 部分关节运行时 armature=0，
低惯量自由度对显式控制更敏感；统一 0.01 后，本次 200 Hz 也通过了全部首帧保持。
这支持“物理步长和关节等效惯量共同影响数值闭环”的解释，不证明单一病因。

Lab 使用 PhysX 的 ImplicitActuatorCfg；外部 Python PD 的反馈项是在积分前算好的力矩，
与物理引擎内部求解的隐式位置驱动有差别。仅把 MuJoCo 积分器改为 implicitfast，并不会
自动把 Python 的 Kp/Kd 变成引擎内的隐式位置驱动。

参考：[MuJoCo 时间步与 armature](https://mujoco.readthedocs.io/en/3.3.2/XMLreference.html#option-timestep)。

## 2. 两份 XML 的实际质量、惯量和限位

SONIC：`gear_sonic/data/assets/robot_description/mjcf/bumi3.xml`。
Mimic：`/home/weili/legged_lab/source/NoetixRobot/NoetixRobot/assets/robots/bumi3_4340/mjcf/bumi3_4340.xml`。
它们确实是不同资产版本，不能因为都是 21 个同名关节就互换动力学参数。

| 项目 | SONIC BUMI3 | Mimic 4340 | 差异 |
| --- | ---: | ---: | ---: |
| 总质量 kg | 20.24093728 | 20.70730940 | +0.46637212，约 +2.3041% |
| 左小腿 `l_knee_pitch_link` 质量 kg | 1.529376 | 1.762560 | +0.233184，约 +15.2470% |
| 右小腿 `r_knee_pitch_link` 质量 kg | 1.529376 | 1.762560 | +0.233184，约 +15.2470% |
| 左肩 roll 限位 rad | [-0.14, 1.94] | [-1.04, 1.40] | 内外侧活动范围不同 |
| 右肩 roll 限位 rad | [-1.94, 0.14] | [-1.40, 1.04] | 内外侧活动范围不同 |

其余 19 个驱动关节的限位完全相同，包括所有腿部关节。其余连杆的质量差别只有
文本精度舍入，合计约 4.12e-6 kg。所有关节轴与局部 joint position 相同，肩部
body quaternion 等有约 1e-6 rad 量级的文本舍入。根 body 初始高度分别为 0.55 和 0，
属于 XML 初始状态；sim2sim reset 会写入参考根状态，不是腿长差异。

惯量不能直接逐项对比 `diaginertia`：它位于各自惯性主轴坐标系，还必须结合
`inertial quat`。下面把惯量统一变换为各连杆局部坐标系、关于质心的张量，单位 kg·m²。

| 左小腿张量项 | SONIC BUMI3 | Mimic 4340 |
| --- | ---: | ---: |
| Ixx | 0.0053171830 | 0.0056605857 |
| Iyy | 0.0052861310 | 0.0056188346 |
| Izz | 0.0016575900 | 0.0017942897 |
| Ixy | -0.0000107799 | -0.0000121917 |
| Ixz | -0.0001574639 | -0.0001696543 |
| Iyz | -0.0000752748 | -0.0000527548 |

右小腿对角项和 Ixz 与左侧相同，Ixy/Iyz 符号镜像。小腿张量差的 Frobenius 范数相对
SONIC 为 6.4901%；小腿 COM 只有小数位舍入差别。额外小腿质量会改变重力矩、摆腿惯性、
触地冲量和所需关节力矩，惯量变化会改变角加速度和耦合。

`base_link` 的 SONIC 局部惯量对角为 `[0.004581786, 0.002116897, 0.004196393]`，
Ixy/Ixz/Iyz 为 `[-1.10233e-5, -6.8942e-6, -2.89974e-5]`；4340 的对角约相同，
但三个交叉项均为零。张量相对差约 0.6847%。其他连杆主要是精度舍入。

左右足底 STL 的 SHA256 在两个目录中分别完全相同，编译后的碰撞网格位置也相同；
所以足底不是换成了另一种尺寸的脚，但质量、惯量、碰撞启用范围和接触参数不同。
G1 则为另一种机器人，当前带手模型总质量 36.1652352 kg、armature=0.01；
其脚底主要碰撞体为 box，不能仅凭 G1 在 200 Hz 稳定推断 BUMI 必然稳定。

## 3. 接触参数及影响

| 项目 | SONIC BUMI3 | Mimic 4340 | 影响 |
| --- | --- | --- | --- |
| 足底形状 | STL mesh 凸包 | 同一 STL mesh 凸包 | 几何轮廓相同 |
| 地面高度 | z=0 | z=0 | 相同 |
| 足底/地面 condim | 3 / 3 | 6 / 6 | 是否有独立扭转及滚动摩擦力矩 |
| 足底 friction | 1, 0.005, 0.0001 | 1.2, 0.05, 0.01 | 滑动摩擦上限及扭转、滚动阻力 |
| 地面 friction | 1, 0.005, 0.0001 | 1, 0.05, 0.01 | 与脚组合成接触参数 |
| solref | 0.02, 1 | 0.01, 1 | 4340 接触恢复时间更短，通常更硬 |
| solimp | 0.9, 0.95, 0.001, 0.5, 2 | 相同 | 两者阻抗形状参数相同 |
| cone | pyramidal | elliptic | 摩擦锥的表示不同 |
| impratio | 1 | 10 | 4340 的摩擦约束相对法向更硬 |
| solver / iterations | Newton / 100 | Newton / 80 | 最大迭代次数，不等于实际每步次数 |
| 机器人碰撞体 | 14 个；mask=1/0 | 18 个；mask=1/1 | SONIC 禁止自身碰撞；4340 允许非过滤对自碰撞 |
| 被动 joint damping | 0.05 | 0.001 | 前者关节本身粘性阻尼更大，独立于 PD Kd |
| joint frictionloss | 0.1 | 0.1 | 两者相同，不是足底接触摩擦 |

4340 的小腿和 ankle_pitch 碰撞关闭，而多处肩、髋、leg_yaw 碰撞启用；SONIC 的
碰撞集合不同。这会改变身体其他部位触地及自碰撞行为，不能把整套设置的效果都归因于足底。

`friction="1 0.005 0.0001"` 三个数并非静摩擦、动摩擦、恢复系数：

1. `1` 是切平面两方向的滑动摩擦系数，无量纲，控制脚水平滑动的摩擦能力。
2. `0.005` 是绕接触法线的扭转摩擦系数，单位 m，控制脚在地上转向的阻力矩。
3. `0.0001` 是绕切平面两轴的滚动摩擦系数，单位 m，控制接触面的滚动阻力矩。

单独考虑一个分量时，对应上限尺度为 `μ_slide * Fn`（N）、`μ_torsion * Fn`（N·m）、
`μ_roll * Fn`（N·m）；多分量同时作用还受摩擦锥耦合限制。
`condim=3` 只启用法向力和两维切向力，后两个摩擦力矩分量不启用；`condim=4` 再加
扭转，`condim=6` 再加两维滚动。多接触点的切向力依然可以合成扭矩，不能把 condim=3
解释成整只脚完全没有抗转能力。

同 priority 的动态接触逐项取两个 geom 摩擦系数的最大值。本次创建真实 MuJoCo
接触核验：SONIC `dim=3, friction=[1,1,0.005,0.0001,0.0001]`，4340
`dim=6, friction=[1.2,1.2,0.05,0.01,0.01]`。存储了五维数组不代表五项都启用。
调高滑动摩擦通常增加可承受的切向力；调高扭转/滚动摩擦也可能压制参考动作需要的
转脚、脚跟到脚尖的转换。调高系数不保证脚滑消失，且可能增加跟踪误差和关节负荷。
`impratio=10` 增强的是摩擦约束相对阻抗，不等于把摩擦系数乘十。

参考：[接触维度和单位](https://mujoco.readthedocs.io/en/3.3.2/computation/index.html#contact)、
[参数混合及 solref](https://mujoco.readthedocs.io/en/3.3.2/modeling.html#contact-parameters)、
[impratio](https://mujoco.readthedocs.io/en/3.3.2/XMLreference.html#option-impratio)。

## 4. 本次控制修改与模型

原 `*_100000_g1.onnx`、`*_100000_smpl.onnx` metadata 均为空。
新增 `sonic_bumi3_control` JSON 元数据版本 1，包含：机器人类型、参数关节名、策略
关节顺序、joint_stiffness、joint_damping、default_joint_pos、action_scale、
joint_effort_limit、control_dt、action_clip、来源说明。

导出读取 `robot.cfg.actuators`、`robot.cfg.init_state`、实际动作项 `cfg.scale`，不读取
域随机化后的第零个环境状态；加载时按关节名重排参数，同时严格核对网络输入输出顺序。
真实策略路径拒绝无元数据/坏数据模型，YAML 中的名义控制值仅供无模型的静态验证。
元数据只是控制参数的保存方式，ONNX 网络本身不执行 MuJoCo PD，也不会学习 Kp/Kd。

本次真实导出元数据与原 YAML 的名义 PD、默认角、动作缩放和限矩逐项完全相等。

| 关节组 | Kp N·m/rad | Kd N·m·s/rad | 力矩上限 N·m |
| --- | ---: | ---: | ---: |
| 腰 yaw | 53 | 3.4 | 27 |
| 髋 pitch / roll | 45 | 3 | 50 |
| 髋 yaw | 20 | 1 | 12 |
| 膝 pitch | 45 | 2 | 50 |
| 踝 pitch / roll | 8 | 0.5 | 9 |
| 肩 pitch / roll / yaw、肘 pitch | 8 | 0.4 | 4 |

运行时 armature：腰和髋/膝 9 关节从 0→0.01；踝 pitch 从 0.012574→0.01；
踝 roll 从 0.009608→0.01；八肩肘保持 0.01。21 个驱动关节最终统一 0.01，浮动根为 0。
armature 是关节空间的附加等效转动惯量（kg·m²），不是连杆质量，也不修改 XML 的
刚体惯量张量；会改变力矩对应的加速度和数值条件。原 XML、训练端和 Mimic/G1 资产维持原值。

已从原 checkpoint 实际运行新版 eval 导出到临时目录，再将导出的控制元数据附加到
原 ONNX 的副本，生成正式保留的 `model_step_100000_g1_control.onnx` 和
`model_step_100000_smpl_control.onnx`。计算图序列化字节与原文件完全相同；每个编码器
100 组固定种子随机输入验证新副本、原 ONNX、实际新导出 ONNX，最大绝对误差均为 0。

## 5. 实际验证

单元与回归：68 passed、11 warnings。十个警告来自既有 torch.onnx 固定维度 trace 与
弃用提示，一个是已有 SMPL 文档非法转义；不是数值失败。
静态资产与 sim2sim 验证器 PASS（零策略只验证接口与有限值，不用于稳定性结论）。
真实 Robot 执行下列对照；真实 SMPL CLI 在 1000/50 Hz 完成 3 秒动态运行，根高
0.464486 m，未做 SMPL 全十条质量评估。本轮未进行 GUI 人工观感评估或实机试验。

判据：倾角 >65° 或根高 <0.15 m 记为诊断倾覆，有限值与 MuJoCo warning 另查。
首帧保持关闭参考推进并清零参考速度，策略与物理仍持续执行。完整播放为当前加载器
解析的全十条共 9067 控制帧。完整播放的根平移包含参考本身移动，不作为漂移指标。
接触滑动速度由真实足底接触点 `mj_jac @ qvel` 的切向分量计算；只统计足底与地面接触、
法向力 >1 N 的样本，按法向力加权，50 Hz 采样，排除前 2 秒。短起跑样本总长 2.74 s，
仅剩 0.74 s 统计窗口，不足以代表持续跑步质量。

| 模式 | 200 Hz | 1000 Hz |
| --- | ---: | ---: |
| 10 条首帧各保持 30 s | 10/10 完成，warning=0 | 10/10 完成，warning=0 |
| 10 条完整播放 | 10/10 完成，warning=0 | 10/10 完成，warning=0 |
| 首帧保持最大根倾角 | 6.14225° | 6.10574° |
| 保持结束各轨迹最大水平净位移 | 5.75471 cm | 2.33422 cm |
| 保持过程中最大水平位移 | 7.32503 cm | 3.65608 cm |

前置 200 Hz 首帧保持 10 s 另有 10/10 完成；总计 50 个独立回放，无诊断倾覆及 MuJoCo warning。
同条件 30 s 的逐轨迹结果如下；数值是结束时净位移，不是全过程峰值。

| 轨迹 | 200 Hz 最大倾角 ° / 净位移 cm | 1000 Hz 最大倾角 ° / 净位移 cm |
| --- | ---: | ---: |
| Idle_Right_001__A018 | 2.6094 / 3.4520 | 2.9232 / 2.0160 |
| walk_forward_amateur_003__A001 | 5.3057 / 2.1306 | 5.3569 / 2.0690 |
| walk_backward_loop_001__A021 | 3.0274 / 3.8679 | 3.0305 / 1.5011 |
| wave_R_001__A431 | 4.9964 / 2.4743 | 4.3983 / 1.2981 |
| squat_003__A361 | 2.6841 / 2.5168 | 2.6496 / 0.2206 |
| Jump_002__A018 | 3.8281 / 5.7547 | 3.7917 / 1.8664 |
| dance_basic_chaines_180_R_fast_001__A309 | 6.1422 / 2.0695 | 6.1057 / 0.5820 |
| Neutral_kick_trash_004__A057 | 5.0941 / 4.4606 | 4.7757 / 1.1000 |
| pels_air_punch_001__A493 | 2.6490 / 3.2710 | 2.6963 / 0.4009 |
| run_start_180_R_001__A327 | 3.8940 / 3.4629 | 3.8813 / 2.3342 |

完整播放实际接触点滑动速度，单位 cm/s：

| 轨迹 | 200 Hz 左 / 右 | 1000 Hz 左 / 右 |
| --- | ---: | ---: |
| Idle_Right_001__A018 | 0.5324 / 0.2250 | 0.1964 / 0.1498 |
| walk_forward_amateur_003__A001 | 5.0839 / 5.3451 | 2.1020 / 2.4631 |
| walk_backward_loop_001__A021 | 5.4407 / 5.6490 | 3.6562 / 3.1941 |
| wave_R_001__A431 | 0.5632 / 0.4968 | 0.2033 / 0.2898 |
| squat_003__A361 | 0.5903 / 0.9259 | 0.4754 / 0.5271 |
| Jump_002__A018 | 3.0275 / 2.2340 | 1.9054 / 1.4084 |
| dance_basic_chaines_180_R_fast_001__A309 | 16.0096 / 12.7544 | 10.3354 / 12.3715 |
| Neutral_kick_trash_004__A057 | 4.6750 / 2.5885 | 1.4977 / 1.5871 |
| pels_air_punch_001__A493 | 1.0146 / 1.3900 | 1.1391 / 2.2966 |
| run_start_180_R_001__A327 | 14.6953 / 19.6710 | 18.3585 / 14.4713 |

1000 Hz 改善多数片段的滑动，但出拳两侧和起跑左脚并未更优；舞蹈、起跑仍有明显接触滑动。
完整播放未触发倾覆阈值不等同于零脚滑或跟踪质量合格。

## 6. 模型与源文件指纹

| 文件 | SHA256 |
| --- | --- |
| model_step_100000_g1_control.onnx | `af600733d013d8925f5083806c26cfba60c121ed6a83d34c371478bb986d95fd` |
| model_step_100000_g1.onnx 原件 | `03da87b9c8d8fcf0a231a43affcbf7feb64fa230e150f46d459b1038cd00dc01` |
| model_step_100000_smpl_control.onnx | `003287b9f4361834e279d55e776fc58bf7c2d94a181ff33ffa0fa959ba43c9e5` |
| model_step_100000_smpl.onnx 原件 | `70dcf313f1b6f3354746b70c67c7631d0f38a9784b78d3d33921a47736f6dfe1` |
| /home/weili/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/bumi3.xml | `1ef8da2e76be03430ba7f022e49309f194a289174db0275d7d3a123197cac3e3` |
| /home/weili/legged_lab/source/NoetixRobot/NoetixRobot/assets/robots/bumi3_4340/mjcf/bumi3_4340.xml | `94ac99adf5f4512ac11903f521d5ec2f2fddb0413cfccec3e31a73d852a37719` |
| BUMI3 Lab robots/bumi3.py | `53bc574948e4faabf8887a8d552e5d1ca1fdf71cb50f1f2a59c3c50f17f0ff8c` |
| model_step_100000.pt | `60b499e2173fb0c17f004adac0083887c0dff3d009f8c4f074ddad54d400f6fd` |
| 左脚 STL，两版本相同 | `02ffd67b382031735f1749d89f1b573f1cff1d198b7dd3036da410ebda4f83dd` |
| 右脚 STL，两版本相同 | `931cfc248e090d6aeddbc4ca35a2f1523aac8fa4f58a7d7db257bf48eff7c70e` |

legged_lab 当前 HEAD `df265177f4c9a4cd2d9dbc31d305d45f551a6cd8`，有其他用户未提交工作，
本轮均只读。4340 脚本 SHA256 `85cbedd4a310f40a58f4bdb6c7308d99ce67c5825c32158f4ba93c326927b7c8`。
该脚本当前已逐关节裁剪 PD 力矩；不能继续引用早期“无显式裁剪”的旧观察。

## 7. 使用命令

本机已生成可直接使用的带元数据模型：

```bash
cd /home/weili/GR00T-WholeBodyControl
BUMI_PY=/home/weili/miniconda3/envs/env_isaaclab/bin/python
BUMI_RUN="$PWD/models/sonic_bumi3/sonic_bumi3_uniform90_lr2e5_critic1e3_ee040_scratch_100k-20260909_141204"
BUMI_DATA="$PWD/data/noetix_bumi3_bigset_10pairs_20260910"

"$BUMI_PY" gear_sonic/scripts/run_bumi3_sim2sim.py \
  --encoder robot \
  --policy "$BUMI_RUN/exported/model_step_100000_g1_control.onnx" \
  --dataset "$BUMI_DATA/dataset.json" \
  --physics-substeps 5
```

T 开始当前轨迹，P 切换下一条并保持首帧；`--autoplay` 可在打开时自动播放。
比较 200 Hz 时把倍率改成 `--physics-substeps 1`。SMPL 模式将 encoder 改成 smpl，
policy 改为 `model_step_100000_smpl_control.onnx`。启动日志输出
`control_parameters_source`、控制关节顺序、Kp/Kd、默认角、缩放、限矩和实际 armature。

后续 checkpoint 使用新版 `eval_agent_trl.py` 正常导出，即会自动写入控制元数据，
输出仍为标准 `model_step_XXXXXX_g1.onnx` 和 `model_step_XXXXXX_smpl.onnx`，无需手工附加。
本轮 `_control` 后缀只用于保留现有 100000 ONNX 原件。

验证命令：

```bash
"$BUMI_PY" -m pytest -q \
  gear_sonic/tests/test_bumi3_control_metadata.py \
  gear_sonic/tests/test_bumi3_sim2sim.py \
  gear_sonic/tests/test_bumi3_smpl_sim2sim.py \
  gear_sonic/tests/test_bumi3_motion_playlist.py
"$BUMI_PY" gear_sonic/tools/validate_bumi3_sim2sim.py
```

实际导出用临时 checkpoint 目录的 config.yaml、meta.yaml、model_config.yaml 副本和
原 checkpoint 硬链接，CLI 为 `checkpoint=<临时目录>/export_run/model_step_100000.pt`
加 `++headless=true ++num_envs=1 ++export_onnx_only=true ++run_eval_loop=false`
及 `++eval_callbacks=[]`；Robot/SMPL motion_file 指向本机上述十条数据。
临时 `algo.trl.output_dir`、`eval_output_dir`、`hydra.run.dir` 全部指向本次审计目录。
导出 exit=0，四个联合/独立 ONNX 均完成；正式保留两个带元数据的联合模型副本。

回放使用上述模型及数据加载器，分别构造 `with_physics_substeps(1/5)` 契约，
对每条 motion 创建独立 `Bumi3SonicSim2Sim`，保持测试为 `start_paused=True`
并运行 1500 次 `step_control()`；完整播放测试为 `start_paused=False`，
运行 `motion.num_frames` 次。动力学、观测、历史与推理均使用正式实现。
