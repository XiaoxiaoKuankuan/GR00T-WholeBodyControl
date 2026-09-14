# BUMI3 SONIC 100000 轮 sim2sim 与 Mimic/G1 对照记录

本文记录 2026-09-14 对“全部动作初始站不稳、走路脚滑”的代码、实际加载资产和闭环诊断。
测试固定本地 100000 轮 Robot 联合模型及十条参考；结论只覆盖下列条件。
起始分支为 `feature/bumi-native-sonic-full-training`，起始提交为
`b5af01cc97211583b5c59c3057cdc81ffa57bcc8`。本次只增加可选物理细分入口和对应测试、文档。

## 1. 结论与当前使用方式

在相同模型、参考、XML、PD 增益、armature、动作缩放和力矩限制下，只将 MuJoCo
物理步长由 5 ms 改为 1 ms、将每次策略动作的物理步数从 4 增加到 20，首帧保持
明显稳定，正常播放的承重脚滑速也下降。策略推理、历史观测、动作播放仍然是 50 Hz。
这表明当前问题对外部显式 PD 的离散更新和接触求解步长高度敏感；并非“Lab 与
MuJoCo 写了相同的 KP/KD、相同 200 Hz，所以动力学就相同”。

PT/ONNX 同输入误差与 Lab/MuJoCo 同状态观测误差均很小，暂未发现导出、关节映射、
当前本体观测或 Robot 未来参考排列的大尺度错误。该排除只针对本次采样；快速动作
仍有脚滑，不宣称已达到 Lab 接触质量，也不把无倾覆视为完整动作跟踪成功。

```bash
cd /home/weili/GR00T-WholeBodyControl
BUMI_RUN="$PWD/models/sonic_bumi3/sonic_bumi3_uniform90_lr2e5_critic1e3_ee040_scratch_100k-20260909_141204"
BUMI_DATA="$PWD/data/noetix_bumi3_bigset_10pairs_20260910"
BUMI_PY=/home/weili/miniconda3/envs/env_isaaclab/bin/python

"$BUMI_PY" gear_sonic/scripts/run_bumi3_sim2sim.py \
  --encoder robot \
  --policy "$BUMI_RUN/exported/model_step_100000_g1.onnx" \
  --dataset "$BUMI_DATA/dataset.json" \
  --physics-substeps 5
```

窗口默认保持第一帧，`T` 开始播放，`P` 切换下一条并重新等待 `T`。
直接启动播放可加 `--autoplay`；单独检查步行可加
`--motion-name walk_forward_amateur_003__A001`。`g1.onnx` 是 BUMI3 模型的
Robot Encoder 内部键名，不表示加载 G1 机器人。

新增参数默认值为 `1`，因此旧命令仍是 5 ms / 4 步。必须显式增加
`--physics-substeps 5` 才启用本次验证的 1 ms / 20 步。该选择保留此前约定的默认
200 Hz 物理 / 50 Hz 策略方案，便于对照；没有修改训练频率或要求重新导出。
应在启动输出中确认 `sim_dt=0.001`、`decimation=20`、
`physics_frequency_hz=1000`、`control_frequency_hz=50`。

## 2. 实际入口、资产和控制差异

| 项目 | 当前 SONIC BUMI3 | legged_lab Mimic 4340 | G1 默认 SONIC 部署链路 |
| --- | --- | --- | --- |
| 入口 | `gear_sonic/scripts/run_bumi3_sim2sim.py` | `/home/weili/legged_lab/scripts/sim2sim_mimic_vision_4340.py`，复用 `sim2sim_mimic.py` | `gear_sonic/scripts/run_sim_loop.py` → `base_sim.py`；C++ `g1_deploy_onnx_ref` |
| MJCF | `gear_sonic/data/assets/robot_description/mjcf/bumi3.xml` | `source/NoetixRobot/NoetixRobot/assets/robots/bumi3_4340/mjcf/bumi3_4340.xml` | `gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml` → `g1_29dof_with_hand.xml` |
| 编译总质量 | 20.24093728 kg | 20.7073094 kg | 36.1652352 kg，包含手部 |
| 主要脚底碰撞 | ankle-roll mesh | ankle-roll mesh，与 BUMI3 脚网格几何相同，但其它质量、惯量、限位不同 | ankle-roll box，半尺寸 `0.085 0.03 0.005`，平底长宽约 170 × 60 mm |
| 接触求解 | Newton；pyramidal；impratio=1；iterations=100 | Newton；elliptic；impratio=10；iterations=80 | Newton；pyramidal；impratio=1；iterations=100 |
| 足底接触维度 | condim=3 | 足底与地面 condim=6 | condim=3 |
| 足底 friction | `1 0.005 0.0001` | `1.2 0.05 0.01`；地面为 `1 0.05 0.01` | `1 0.005 0.0001` |
| 足底 solref | `0.02 1` | `0.01 1` | `0.02 1` |
| 运行时 armature | 肩肘 .01；踝 pitch .012574、roll .009608；腰/髋/膝 0 | 21 个关节 .03 | 关节 .01 |
| 被动 damping | .05 | .001 | .05 |
| 关节 frictionloss | .1 | .1 | 腿等 .2，腕/手指 .1 |
| 默认控制形式 | 目标角 → Python 显式 PD → 按关节限幅 → motor；物理 200 Hz / 策略 50 Hz | 外部 PD，增益、默认角和 action scale 从对应 ONNX metadata 读取；物理 200 Hz / 策略 50 Hz | DDS 目标角/速度/前馈及 KP/KD → Python PD 与力矩限幅；默认物理 200 Hz / 策略 50 Hz |
| 观测契约 | Robot `1170→21`，480 维参考 + 690 维本体历史，锚点 base_link | 对应 21 关节配置 `570→21`，114 维 × 5 帧，锚点 waist_yaw_link | release 为独立 encoder/decoder 和 29 电机契约，不能直接替换 BUMI3 模型 |
| 初始流程 | reset 到所选参考首帧；GUI 立即运行冻结参考的策略，等 T 播放 | 对应 Mimic 模型和参考流程 | C++ `InitControl()` 用约 3 秒插值到 default_angles，再进入策略/参考播放状态 |

G1 默认 YAML 是 `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml`；
其中还启用了可切换的弹性绳。这里只证明默认配置存在该选项，没有本轮用户正在运行的
G1 进程来确认绳是否开启，不能据此断言用户观察的稳定是由绳造成。

4340 并不是同一 BUMI3 资产的可视化版本，其总质量约多 0.466 kg，且运行时关节
惯量、接触约束、模型观测和控制参数来源不同。G1 的外部 PD 形式与 BUMI3 相同，
但平底 box、质量/惯量、增益和初始化不同，因此 G1 可在 5 ms 稳定，不能推出
当前 BUMI3 策略也必然在 5 ms 稳定。本次没有替换 BUMI3 资产，也没有修改 G1/Mimic。

`condim=3` 已包含切向摩擦；`condim=6` 还增加绕法向及切向轴的转动摩擦。
MuJoCo 官方也将 elliptic/impratio 与软接触滑动联系起来，但这些参数不能替代
具体机器人对照。参见 [MuJoCo 3.3.2 接触维度](https://mujoco.readthedocs.io/en/3.3.2/XMLreference.html#body-geom-condim)
及 [求解器与滑移说明](https://mujoco.readthedocs.io/en/3.3.2/modeling.html#solvers)。

## 3. 与本次实际 Lab 状态的差异

Lab 入口仍是 `gear_sonic/eval_agent_trl.py`，机器人配置为
`gear_sonic/envs/manager_env/robots/bumi3.py`，实际资产是
`gear_sonic/data/assets/robot_description/urdf/bumi3/bumi.urdf`。
它采用 PhysX 的 `ImplicitActuatorCfg`，物理 5 ms / decimation=4；MuJoCo
使用外部显式 PD。相同增益和时间间隔并不能保证两者的离散更新、约束和速度限制相同。

Lab 和 BUMI3 MuJoCo 脚部都来自 BUMI3 脚 STL，不能将此处解释为“Lab 是 box、
MuJoCo 是 mesh”。Lab URDF 导入使用 convex hull，而两套引擎的实际接触约束和
求解不同；本轮未逐接触点对比 PhysX 导入后的凸包形状与 MuJoCo 接触流形。
Lab 开启自碰撞，MuJoCo 保留机器人对地碰撞而关闭 link 自碰撞。

实际采集的 Lab 还保留了启动域随机化：默认关节角有 ±0.01 rad 范围的偏置、
身体质量与摩擦随机化；一次样本 waist_yaw_link 质量为 5.40517 kg，名义为
5.27167 kg。Lab 肩肘 armature 为 0，当前 MuJoCo 为 .01；Lab 关节 friction
为 0，MuJoCo frictionloss 为 .1。Lab 执行器有物理关节速度上限（分组 9/12 rad/s），
当前 MuJoCo 控制路径主要按力矩限幅。这些差异都不能靠“两个 YAML 看起来相同”排除。

## 4. 模型和观测排除实验

对 `walk_forward_amateur_003__A001` 采集两次单环境 Lab 回放：原带噪观测
450 步（9 s）以及只关闭 policy/tokenizer 观测噪声的 300 步（6 s）。两次均无
termination 或 timeout。未重新训练，未修改模型、原始轨迹或正式训练配置。

| 对照项 | 关闭观测噪声的 300 步最大绝对误差 |
| --- | ---: |
| 同一 Lab 输入，PT 与 Robot ONNX 动作 | 4.7683716e-6 |
| Robot 未来关节位置 | 2.3841858e-7 |
| Robot 未来关节速度 | 1.7881393e-5 |
| Robot 未来姿态 | 7.9721212e-7 |
| 当前局部根角速度 | 2.0489097e-7 |
| 当前关节位置偏差 | 0 |
| 当前关节速度 | 0 |
| 当前重力方向 | 1.3411045e-7 |

原带噪 450 步的 PT/ONNX 最大动作误差为 4.2915344e-6。初次将带噪 Lab
观测与无噪声运动学重建值比较时，差异落在配置噪声幅度内，不能称为代码错误；
随后用关闭观测噪声的采样单独验证。参考帧按 `0,1,2,...` 推进，未来窗口为
`0,5,...,45`，没有发现当前帧错一位。

同状态实验把 Lab 根位姿、根速度、21 关节位置/速度放入仅做 `mj_forward`
的 MuJoCo 数据，使用 Lab 实际随机化后的默认角，并关闭参考 heading 二次对齐；
不推进 MuJoCo 动力学。Robot ONNX 输入由 Lab tokenizer 的 `[2:482]` 与
完整 690 维 actor history 拼接。当前本体项逐项比较；完整历史更新与顺序还由
已有回归测试覆盖。本实验排除的是采样中的数据/模型契约问题，不是所有分布下的
完整等价性证明，也没有据此声称 Lab 的所有动作都没有脚滑。

## 5. 闭环测量方法

MuJoCo 3.3.2，解释器 `/home/weili/miniconda3/envs/env_isaaclab/bin/python`。
对每条动作独立 reset：保持首帧 10 s 或自动播放至最后一帧（T 帧共 T−1 次
参考推进）。1 ms 方案再单独延长首帧保持至 30 s。所有参考为 50 Hz，共 9067 帧。
自动播放采用参考初速度；保持首帧采用零速度，与 GUI 等待逻辑一致。

诊断只在正常 `step_control()` 后读状态，未额外强写 root/joint pose、脚位置、
地面高度或动作输出。基准 ONNX 使用 CPU 单线程，以固定推理执行条件；正式入口
另以原 ONNX Runtime 设置完成 Robot/SMPL 回放。参考影子不参与碰撞。

脚滑指标取真实 ankle-roll 与地面的接触，法向力大于 1 N，使用接触点雅可比
乘刚体广义速度，并投影到接触切平面；跨接触点/控制帧以法向力加权。为排除
reset 冲击，前 2 s 不计入脚滑统计，采样频率 50 Hz。该指标是承重接触点的
滑速，不是踝 link 位移，也不是物理 1000 Hz 全量积分值。
正常运动的根部位移不能直接叫脚滑；首帧保持中的根漂移也可能包含策略迈步。

倾角大于 65° 或根高低于 0.15 m 提前停止，称为“诊断倾覆”；该判据不是训练
termination，也不是动作质量合格线。下蹲时约 31° 的根倾角不直接等于失稳。
每组只测固定初值的一次确定性轨迹，无随机种子统计置信区间。

<!-- 本次实际数值表由固定诊断结果生成，下面各表均标明单位与观测阶段。 -->

### 5.1 相同 10 秒首帧保持

根位移为结束时相对 reset 的水平净位移，不是全程最大值。带“倾覆”的原方案提前停止，时间见括号。

| 动作 | 5 ms 最大倾角 / 根净位移 | 1 ms 最大倾角 / 根净位移 | 5 ms 状态 |
| --- | ---: | ---: | --- |
| `Idle_Right_001__A018` | 16.48° / 101.21 cm | 2.78° / 1.53 cm | 完成 10 s |
| `walk_forward_amateur_003__A001` | 16.59° / 180.16 cm | 5.38° / 2.01 cm | 完成 10 s |
| `walk_backward_loop_001__A021` | 80.91° / 218.31 cm | 3.00° / 2.56 cm | 倾覆（4.08 s） |
| `wave_R_001__A431` | 4.80° / 2.71 cm | 4.42° / 0.69 cm | 完成 10 s |
| `squat_003__A361` | 23.67° / 138.04 cm | 2.66° / 1.04 cm | 完成 10 s |
| `Jump_002__A018` | 15.48° / 18.59 cm | 3.78° / 2.74 cm | 完成 10 s |
| `dance_basic_chaines_180_R_fast_001__A309` | 6.19° / 2.24 cm | 6.14° / 1.58 cm | 完成 10 s |
| `Neutral_kick_trash_004__A057` | 71.53° / 58.89 cm | 4.92° / 1.20 cm | 倾覆（1.22 s） |
| `pels_air_punch_001__A493` | 82.41° / 143.05 cm | 2.66° / 0.40 cm | 倾覆（3.24 s） |
| `run_start_180_R_001__A327` | 74.76° / 121.77 cm | 3.86° / 3.00 cm | 倾覆（3.48 s） |

1 ms 的十组均完成 10 s；原方案的静态 Idle 尽管未倾覆，也产生约 1 m 水平漂移。

### 5.2 完整自动播放的真实承重接触滑速

单位 cm/s，按左脚 / 右脚列出，排除前 2 s。三组方案均未触发诊断倾覆。
“4340 接触”只是在原 5 ms BUMI3 内存模型里采用下节说明的接触参数，未加载 4340 权重/资产。

| 动作 | 播放时长 s | 5 ms 原方案 | 1 ms 细分 | 5 ms + 4340 接触 |
| --- | ---: | ---: | ---: | ---: |
| `Idle_Right_001__A018` | 59.66 | 0.68 / 0.21 | 0.20 / 0.13 | 0.01 / 0.02 |
| `walk_forward_amateur_003__A001` | 30.16 | 7.76 / 7.85 | 2.84 / 3.07 | 2.72 / 3.54 |
| `walk_backward_loop_001__A021` | 15.76 | 8.53 / 7.74 | 3.60 / 3.91 | 5.88 / 4.10 |
| `wave_R_001__A431` | 5.92 | 1.11 / 0.96 | 0.23 / 0.39 | 0.12 / 0.06 |
| `squat_003__A361` | 8.46 | 6.19 / 15.57 | 0.50 / 0.51 | 1.03 / 5.17 |
| `Jump_002__A018` | 39.28 | 2.55 / 2.68 | 1.85 / 1.85 | 1.45 / 0.79 |
| `dance_basic_chaines_180_R_fast_001__A309` | 7.66 | 15.65 / 15.96 | 8.62 / 8.87 | 7.41 / 9.54 |
| `Neutral_kick_trash_004__A057` | 7.96 | 5.72 / 3.03 | 1.97 / 1.95 | 0.46 / 2.39 |
| `pels_air_punch_001__A493` | 3.56 | 15.68 / 3.51 | 1.08 / 3.44 | 3.24 / 8.82 |
| `run_start_180_R_001__A327` | 2.72 | 16.96 / 19.17 | 12.44 / 14.36 | 2.18 / 24.69 |

前走左右脚约降低 63% / 61%；1 ms 快舞仍约 8.6 / 8.9 cm/s，急转跑仍约 12.4 / 14.4 cm/s。
空击和急转跑片段很短，排除前 2 s 后的统计窗口分别只有约 1.56 s 和 0.72 s，应谨慎解释。

### 5.3 1 ms 首帧保持延长至 30 秒

| 动作 | 最大倾角 | 末时刻根水平净位移 cm | 左 / 右承重脚滑速 cm/s |
| --- | ---: | ---: | ---: |
| `Idle_Right_001__A018` | 2.85° | 0.93 | 0.15 / 0.12 |
| `walk_forward_amateur_003__A001` | 5.38° | 2.20 | 0.03 / 0.04 |
| `walk_backward_loop_001__A021` | 3.00° | 1.44 | 0.09 / 0.11 |
| `wave_R_001__A431` | 4.42° | 1.62 | 0.10 / 0.07 |
| `squat_003__A361` | 2.66° | 0.08 | 0.08 / 0.06 |
| `Jump_002__A018` | 3.78° | 1.71 | 0.15 / 0.14 |
| `dance_basic_chaines_180_R_fast_001__A309` | 6.14° | 0.55 | 0.05 / 0.10 |
| `Neutral_kick_trash_004__A057` | 4.92° | 1.42 | 0.07 / 0.10 |
| `pels_air_punch_001__A493` | 2.66° | 0.63 | 0.06 / 0.07 |
| `run_start_180_R_001__A327` | 3.86° | 2.21 | 0.04 / 0.09 |

全部完成 1500 策略步，最大倾角 6.14352°；十条结束时根水平净位移最大 2.20749 cm，MuJoCo warning 总数为 0。
该净位移不代表机器人全程从未偏离 2.21 cm，也不代表每只脚完全不动。

### 5.4 正式细分接口下的 T 按键混合流程

使用 `with_physics_substeps(5)`，逐条独立 reset，先保持 500 步，调用
`enqueue_key(ord("T"))`，播放 T−1 步，再保持 500 步。各段连续沿用同一动力学状态。

| 动作 | 等待最大倾角 | 播放最大倾角 | 结束保持最大倾角 |
| --- | ---: | ---: | ---: |
| `Idle_Right_001__A018` | 2.78° | 3.30° | 2.60° |
| `walk_forward_amateur_003__A001` | 5.38° | 15.94° | 3.50° |
| `walk_backward_loop_001__A021` | 3.00° | 11.67° | 3.21° |
| `wave_R_001__A431` | 4.42° | 6.68° | 4.76° |
| `squat_003__A361` | 2.66° | 30.68° | 7.44° |
| `Jump_002__A018` | 3.78° | 26.65° | 3.78° |
| `dance_basic_chaines_180_R_fast_001__A309` | 6.14° | 15.88° | 5.86° |
| `Neutral_kick_trash_004__A057` | 4.92° | 16.54° | 3.89° |
| `pels_air_punch_001__A493` | 2.66° | 21.53° | 2.45° |
| `run_start_180_R_001__A327` | 3.86° | 37.65° | 5.78° |

全部十组、三段流程均完成且没有诊断倾覆或 MuJoCo warning。这是无窗口按键队列流程验证，未声称人工观看了十段 GUI 视频。

## 6. 其它对照为何未保留为正式参数

4340 接触对照只将 BUMI3 内存模型的 cone 改 elliptic、impratio=10、iterations=80，
左右 ankle-roll 与地面 condim=6、solref=`0.01 1`，足摩擦=`1.2 0.05 0.01`、
地面摩擦=`1 0.05 0.01`。模型质量、关节惯量、增益、力矩上限和 5 ms 物理步不改。
虽然十组首帧保持未达到倾覆阈值，以下动作仍明显漂移：

| 动作 | 10 s 根水平净位移 | 最大倾角 |
| --- | ---: | ---: |
| `walk_forward_amateur_003__A001` | 78.46 cm | 14.25° |
| `squat_003__A361` | 39.24 cm | 26.35° |
| `pels_air_punch_001__A493` | 68.27 cm | 28.99° |

前走接触点虽很少滑动，机器人仍可能通过迈步漂移；因此“脚摩擦增加”不是本轮全部问题的充分解释。
急转跑右脚接触滑速还从原约 19.17 增至 24.69 cm/s。该方案没有写入正式 XML。

仅把积分器改为 `implicitfast`、仍由 Python 显式计算 PD，在 Idle/前走/下蹲的
首帧与全程六组对照中与原方案结果完全一致。不能把“换积分器”当成本轮已经有效的修复，
也不能把这项试验解释成把 PD 阻尼项改成了 MuJoCo 原生隐式执行器。

## 7. 文件修改、验证与复现信息

- `gear_sonic/utils/mujoco_sim/bumi3_sim2sim.py`：新增 `Bumi3Contract.with_physics_substeps()`，
  验证正整数倍率并返回步长/decimation 同比变化的新契约；不改原配置对象。
- `gear_sonic/scripts/run_bumi3_sim2sim.py`：增加默认 1 的 CLI 参数和实际物理频率日志。
- `gear_sonic/tests/test_bumi3_sim2sim.py`：用真实 MuJoCo 计数策略/物理步、仿真时间及参考帧；
  检查 0、负数、小数、布尔值被拒绝。
- 使用文档、本报告、根目录 `BUMI3_SONIC_修改记录.md` 同步记录原因、指标、边界和回滚方式。

回归命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$BUMI_PY" -m pytest -q \
  gear_sonic/tests/test_bumi3_sim2sim.py \
  gear_sonic/tests/test_bumi3_motion_playlist.py \
  gear_sonic/tests/test_bumi3_smpl_sim2sim.py -p no:cacheprovider
"$BUMI_PY" gear_sonic/tools/validate_bumi3_sim2sim.py --skip-smoke
git diff --check
```

结果为 **56 passed，1 warning，7.85 s**；warning 来自已有训练观测源码中的非法转义 `\*`，
与本次细分无关。静态资产/契约验证输出 `BUMI3_SIM2SIM_VALIDATION=PASS`，
`git diff --check` 通过。静态验证依然报告默认 5 ms，符合保留旧默认的预期。

正式 Robot CLI 加 `--physics-substeps 5 --headless --no-real-time --duration 10`，
完成 500 步、仿真时间 10 s、末帧根高 0.465304 m；SMPL 入口改
`--encoder smpl --policy "$BUMI_RUN/exported/model_step_100000_smpl.onnx"`
并运行 5 s，完成 250 步、末帧根高 0.464022 m。两者使用各自 1170/1470 维
模型。SMPL 只验证了默认 Idle 短回放，没有对 SMPL 全十条做同规模矩阵。

本次一次性诊断目录为 `/tmp/bumi100k-sim-audit-20260914-oFicT7`，临时脚本、
Lab 采样、日志和 JSON 在证据归档且核验无进程引用后清理，不作为正式工具提交。
临时诊断实际使用以下调用；下列是本次执行记录，清理后脚本路径不再存在：

```bash
"$BUMI_PY" "$AUDIT_TMP/audit.py" --variant baseline --output "$AUDIT_TMP/baseline_all.json"
"$BUMI_PY" "$AUDIT_TMP/audit.py" --variant dt001 --output "$AUDIT_TMP/dt001_all.json"
"$BUMI_PY" "$AUDIT_TMP/audit.py" --variant dt001 --phases hold --hold-duration 30 --output "$AUDIT_TMP/dt001_hold30.json"
"$BUMI_PY" "$AUDIT_TMP/audit.py" --variant mimic_contact --output "$AUDIT_TMP/mimic_contact_all.json"
"$BUMI_PY" "$AUDIT_TMP/audit.py" --variant implicitfast --motions Idle_Right_001__A018,walk_forward_amateur_003__A001,squat_003__A361 --output "$AUDIT_TMP/implicitfast.json"
"$BUMI_PY" "$AUDIT_TMP/mixed.py"
"$BUMI_PY" "$AUDIT_TMP/parity.py" lab_walk
"$BUMI_PY" "$AUDIT_TMP/parity.py" lab_walk_clean
```

`AUDIT_TMP` 指上述精确目录。`audit.py` 直接复用正式 `step_control()`；
`dt001` 诊断采用 `dataclasses.replace(sim_dt=.001, decimation=20)`，
正式入口与 mixed 验收则使用新契约方法。脚滑公式和停止判据见第 5 节。
Lab 采样复用 `gear_sonic.tools.bumi3_lab_audit.Bumi3LabAuditCallback` 的临时子类，
仅额外记录刚体速度、接触力；实际配置为 checkpoint=100000、num_envs=1、headless=true、
use_encoder=g1、export_onnx_only=false、eval_callbacks=[bumi_audit]、
start_from_first_frame=true、filter_motion_keys=[walk_forward_amateur_003__A001]，
robot/smpl 路径与本报告第 1 节相同。第二次只增加 policy/tokenizer 的
`enable_corruption=false` 并将 max_steps 从 450 改为 300。hydra、回调与算法输出
均指向上述目录的对应 lab_walk/lab_walk_clean 子目录。

没有启动训练、改写正式模型/数据、连接真机、完整重跑 G1/4340 策略，
也没有对 Lab 全十条完成接触定量回放。后续若继续缩小差异，应固定同一模型和参考
分别检查足底凸包接触、执行器/速度限制及域随机化，不能一次替换全部资产参数。
恢复旧行为只需省略细分参数或传 `--physics-substeps 1`；源码回滚用新的反向提交，
不覆盖正式训练产物或改写历史。

## 8. 固定输入指纹

| 输入 | SHA256 |
| --- | --- |
| 100000 PT | `60b499e2173fb0c17f004adac0083887c0dff3d009f8c4f074ddad54d400f6fd` |
| 100000 Robot ONNX | `03da87b9c8d8fcf0a231a43affcbf7feb64fa230e150f46d459b1038cd00dc01` |
| 100000 SMPL ONNX | `70dcf313f1b6f3354746b70c67c7631d0f38a9784b78d3d33921a47736f6dfe1` |
| BUMI3 MJCF | `1ef8da2e76be03430ba7f022e49309f194a289174db0275d7d3a123197cac3e3` |
| BUMI3 URDF | `0e08c15fe2226fedeac967c06a7910701935fc6de8fca2d4664a76c9ac41e955` |
| BUMI3 sim2sim YAML | `f52be29ca85a264273dc5ab75055ea52b8c297a362d093fd27f25ca90d9865f8` |
| 十对清单 | `8d11e47a2a20412edd8c4ad32ba04f5708eca060e8bcf3eeebf7c3be15c3f8f1` |
| Mimic 4340 脚本 | `85cbedd4a310f40a58f4bdb6c7308d99ce67c5825c32158f4ba93c326927b7c8` |
| Mimic 基类 | `50ae1da00b5418ee7922ac20952ac8707ce246e43c54e0d68e8691d6724346bc` |
| 4340 MJCF | `94ac99adf5f4512ac11903f521d5ec2f2fddb0413cfccec3e31a73d852a37719` |
| G1 scene | `274e7b3755dc1eedcf210af54d1c97749f31457d77ae2c499ec234e804065b21` |
