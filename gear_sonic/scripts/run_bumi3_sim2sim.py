# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""运行 BUMI3 原生 SONIC Robot/SMPL 编码器的 MuJoCo sim2sim。

入口读取单条 PKL/NPZ/CSV 或包含多条动作的 JSON/YAML 数据集清单，构造与
``sonic_bumi3.yaml`` 一致的 1170 维联合 ONNX 输入，以 50 Hz 推理 21 维动作，
再用 BUMI3 的 PD 参数在 200 Hz MuJoCo 中执行。实际模型应使用
``eval_agent_trl.py`` 导出的 ``model_step_XXXXXX_g1.onnx``；文件名中的 ``g1``
代表为 checkpoint 兼容保留的 Robot Encoder 内部键名，不代表 G1 机器人。
指定 --encoder smpl 后，--motion 改读含 pose_aa/smpl_joints 的人体 PKL/NPZ，
--dataset 改读每项 smpl 路径，模型必须换成 *_smpl.onnx（1470 维）。配对的
robot 只用于初始化和红色影子；单条 SMPL 可用 --robot-motion 提供同帧配对，
省略时从默认站姿开始并关闭不存在的机器人参考影子。

默认打开 MuJoCo viewer 并保持第一帧参考，按 T 开始、按 P 切下一条并重新等待；
保持阶段策略与物理继续运行，GUI 不指定时长则一直运行到关闭窗口。
每次初始化会按实际脚部碰撞体最小上移真实机器人，消除穿地并保留 0.1 mm 余量；
终端输出校正前后高度与上移量，参考数据和红色影子的原始高度保持不变。
窗口内叠加红色半透明参考影子：不透明机器人
是 ONNX policy 实际控制结果，红色影子是训练 Robot PKL 的 root+21 关节经同一个
BUMI3 MJCF FK 得到的参考姿态。影子不参与碰撞或动力学，也不会跟随真实机器人降低
高度，因此可直接判断参考本身是直立还是横躺。服务器无显示时使用 ``--headless``；
``--validate-only`` 只核对配置、动作和 ONNX 契约，不推进仿真。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import tyro

from gear_sonic.utils.mujoco_sim.bumi3_motion_dataset import (
    load_motion_dataset,
    load_smpl_reference_motion,
)
from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import (
    Bumi3Contract,
    Bumi3SonicSim2Sim,
    DEFAULT_BUMI3_SIM2SIM_CONFIG,
    OnnxRobotPolicy,
    load_reference_motion,
)
from gear_sonic.utils.mujoco_sim.bumi3_smpl_reference import SMPL_FUTURE_FRAMES, SMPL_FUTURE_STRIDE


@dataclass
class Args:
    """BUMI3 sim2sim 命令行参数。"""

    policy: Path
    """联合 ONNX 路径：robot 使用 *_g1.onnx，smpl 使用 *_smpl.onnx。"""

    encoder: Literal["robot", "smpl"] = "robot"
    """参考编码器；默认保留机器人轨迹输入，smpl 改为真正的人体参考输入。"""

    motion: Path | None = None
    """单条 50 FPS 参考；robot 为 PKL/NPZ/CSV，smpl 为人体 PKL/NPZ，与 dataset 二选一。"""

    robot_motion: Path | None = None
    """单条 SMPL 的可选配对 Robot，供初始化和影子显示；数据集在每项 robot 中声明。"""

    robot_motion_key: str | None = None
    """robot-motion 容器中的动作键；不指定时读取唯一动作。"""

    dataset: Path | None = None
    """包含有序 motions 列表的 JSON/YAML 清单；相对路径按清单目录解析。"""

    motion_name: str | None = None
    """指定从数据集哪条 name 开始，后续按 P 仍按清单顺序循环切换。"""

    config: Path = DEFAULT_BUMI3_SIM2SIM_CONFIG
    """BUMI3 sim2sim YAML；通常不需要覆盖。"""

    motion_key: str | None = None
    """多动作 PKL/NPZ 或 CSV 根目录中的动作名称。"""

    joint_order: Literal["auto", "policy", "isaaclab", "mujoco"] = "auto"
    """参考动作关节顺序；auto 会按文件格式和元数据选择。"""

    quaternion_order: Literal["auto", "wxyz", "xyzw"] = "auto"
    """参考 root quaternion 顺序；auto 会按文件格式和元数据选择。"""

    provider: Literal["cpu", "cuda"] = "cpu"
    """ONNX Runtime provider；小型 MLP 通常使用 CPU 即可。"""

    start_frame: int = 0
    """从参考动作的哪一帧开始。"""

    duration: float | None = None
    """运行秒数；GUI 不指定则常驻等待按键，无窗口默认运行一条动作的时长。"""

    autoplay: bool = False
    """GUI 启动后自动播放；默认保持第一帧，T 开始、P 切下一条后重新等待。"""

    loop_motion: bool = False
    """到动作末尾后循环播放。"""

    align_reference_heading: bool = True
    """把参考动作起始 yaw 对齐到机器人当前 yaw，与 G1 sim2sim 默认行为一致。"""

    headless: bool = False
    """不创建 MuJoCo viewer，适合服务器 smoke。"""

    real_time: bool = True
    """按 50 Hz 墙钟节拍运行；关闭后尽快执行。"""

    show_reference: bool = True
    """在 GUI 中叠加红色半透明参考机器人；使用 ``--no-show-reference`` 关闭。"""

    reference_alpha: float = 0.32
    """参考影子透明度，必须位于 (0, 1]。"""

    validate_only: bool = False
    """只验证配置、动作和 ONNX 维度，不推进 MuJoCo。"""


def main(args: Args) -> None:
    if (args.motion is None) == (args.dataset is None):
        raise ValueError("必须且只能指定 --motion 或 --dataset 其中一个")
    if args.robot_motion is not None and (args.encoder != "smpl" or args.dataset is not None):
        raise ValueError("--robot-motion 仅用于 --encoder smpl --motion；数据集请填写每项 robot")
    if args.robot_motion_key is not None and args.robot_motion is None:
        raise ValueError("--robot-motion-key 必须同时指定 --robot-motion")
    if args.motion_name is not None and args.dataset is None:
        raise ValueError("--motion-name 仅用于选择数据集起始轨迹")
    if args.dataset is not None and (
        args.motion_key is not None or args.joint_order != "auto" or args.quaternion_order != "auto"
    ):
        raise ValueError("数据集模式请在清单每项中指定 motion_key/关节顺序/四元数顺序")
    contract = Bumi3Contract.from_yaml(args.config)
    if args.dataset is not None:
        motions = load_motion_dataset(args.dataset, contract, encoder=args.encoder)
        if args.motion_name is not None:
            names = [item.name for item in motions]
            if args.motion_name not in names:
                raise ValueError(f"数据集没有轨迹 {args.motion_name!r}；可选: {names}")
            start = names.index(args.motion_name)
            motions = motions[start:] + motions[:start]
    elif args.encoder == "smpl":
        motions = [load_smpl_reference_motion(
            args.motion, contract, motion_key=args.motion_key,
            robot_path=args.robot_motion, robot_motion_key=args.robot_motion_key,
            joint_order=args.joint_order, quaternion_order=args.quaternion_order,
        )]
    else:
        motions = [load_reference_motion(
            args.motion, contract, motion_key=args.motion_key,
            joint_order=args.joint_order, quaternion_order=args.quaternion_order,
        )]
    motion = motions[0]
    policy = OnnxRobotPolicy(args.policy, contract, provider=args.provider, encoder=args.encoder)
    runner = Bumi3SonicSim2Sim(
        contract,
        motion,
        policy,
        loop_motion=args.loop_motion,
        start_frame=args.start_frame,
        align_reference_heading=args.align_reference_heading,
        motions=motions,
        start_paused=not (args.headless or args.autoplay),
        encoder=args.encoder,
    )

    resolved = {
        "robot_type": "bumi3",
        "encoder": args.encoder,
        "model_path": str(contract.model_path),
        "policy_path": str(args.policy.expanduser().resolve()),
        "motion_name": motion.name,
        "motion_frames": motion.num_frames,
        "motion_count": len(motions),
        "motion_names": [item.name for item in motions],
        "sim_dt": contract.sim_dt,
        "decimation": contract.decimation,
        "pd_implementation": "python_explicit_pd_motor",
        "integrator": "Euler",
        "joint_passive_damping": runner.model.dof_damping[runner.dof_addresses].tolist(),
        "joint_armature": runner.model.dof_armature[runner.dof_addresses].tolist(),
        "control_frequency_hz": 1.0 / contract.control_dt,
        "target_fps": contract.target_fps,
        "history_length": contract.history_length,
        "future_frames": SMPL_FUTURE_FRAMES if args.encoder == "smpl" else contract.num_future_frames,
        "future_frame_stride": SMPL_FUTURE_STRIDE if args.encoder == "smpl" else contract.future_frame_stride,
        "align_reference_heading": args.align_reference_heading,
        "action_dim": contract.action_dim,
        "policy_input_dim": runner.policy_input_dim,
        "initialization": "paired_robot_reference" if motion.has_robot_reference else "default_robot_pose_smpl_yaw",
        "reference_kind": "paired_robot_shadow" if motion.has_robot_reference else "none",
        "policy_visual_and_collision": "direct_bumi3_xml_dynamics_model",
        "show_reference": args.show_reference,
        "reference_alpha": args.reference_alpha,
    }
    print("BUMI3_SIM2SIM_RESOLVED=" + json.dumps(resolved, ensure_ascii=False, indent=2))
    if not motion.has_robot_reference:
        print("未提供配对机器人轨迹：从 BUMI 默认站姿初始化，朝向采用 SMPL 当前帧 yaw；不显示机器人参考影子。")
    reference_diagnostics = runner.reference_pose_diagnostics(args.start_frame)
    print(
        "BUMI3_REFERENCE_POSE="
        + json.dumps(reference_diagnostics, ensure_ascii=False, indent=2)
    )
    if args.validate_only:
        print("BUMI3_SIM2SIM_VALIDATE_ONLY=PASS")
        return

    if args.duration is None:
        remaining_frames = motion.num_frames if args.loop_motion else motion.num_frames - args.start_frame
        control_steps = max(1, remaining_frames) if args.headless else None
    else:
        if args.duration <= 0.0:
            raise ValueError("duration 必须大于 0")
        control_steps = max(1, round(args.duration / contract.control_dt))
    print("BUMI3_PLAYBACK=" + json.dumps(runner.playback_status(), ensure_ascii=False), flush=True)
    if not args.headless:
        print("请在 MuJoCo 窗口按 T 开始播放，按 P 切换下一条并保持第一帧；关闭窗口退出。", flush=True)
    stats = runner.run(
        control_steps,
        headless=args.headless,
        real_time=args.real_time,
        show_reference=args.show_reference,
        reference_alpha=args.reference_alpha,
    )
    print("BUMI3_SIM2SIM_STATS=" + json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
