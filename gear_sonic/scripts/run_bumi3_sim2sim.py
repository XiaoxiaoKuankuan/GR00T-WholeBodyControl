# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""运行 BUMI3 原生 SONIC Robot Encoder 的 MuJoCo sim2sim。

入口读取训练数据 PKL/NPZ 或 G1 ``MotionDataReader`` 风格 CSV 目录，构造与
``sonic_bumi3.yaml`` 一致的 1170 维联合 ONNX 输入，以 50 Hz 推理 21 维动作，
再用 BUMI3 的 PD 参数在 200 Hz MuJoCo 中执行。实际模型应使用
``eval_agent_trl.py`` 导出的 ``model_step_XXXXXX_g1.onnx``；文件名中的 ``g1``
代表为 checkpoint 兼容保留的 Robot Encoder 内部键名，不代表 G1 机器人。

默认打开 MuJoCo viewer、按实时速度播放，并叠加红色半透明参考影子：不透明机器人
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

from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import (
    Bumi3Contract,
    Bumi3SonicSim2Sim,
    DEFAULT_BUMI3_SIM2SIM_CONFIG,
    OnnxRobotPolicy,
    load_reference_motion,
)


@dataclass
class Args:
    """BUMI3 sim2sim 命令行参数。"""

    policy: Path
    """``eval_agent_trl.py`` 导出的 ``*_g1.onnx`` 路径。"""

    motion: Path
    """50 FPS BUMI3 动作 PKL/NPZ、单个 CSV clip 目录或 CSV 根目录。"""

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
    """运行秒数；不指定时播放到动作末帧。"""

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
    contract = Bumi3Contract.from_yaml(args.config)
    motion = load_reference_motion(
        args.motion,
        contract,
        motion_key=args.motion_key,
        joint_order=args.joint_order,
        quaternion_order=args.quaternion_order,
    )
    policy = OnnxRobotPolicy(args.policy, contract, provider=args.provider)
    runner = Bumi3SonicSim2Sim(
        contract,
        motion,
        policy,
        loop_motion=args.loop_motion,
        start_frame=args.start_frame,
        align_reference_heading=args.align_reference_heading,
    )

    resolved = {
        "robot_type": "bumi3",
        "model_path": str(contract.model_path),
        "policy_path": str(args.policy.expanduser().resolve()),
        "motion_name": motion.name,
        "motion_frames": motion.num_frames,
        "sim_dt": contract.sim_dt,
        "decimation": contract.decimation,
        "control_frequency_hz": 1.0 / contract.control_dt,
        "target_fps": contract.target_fps,
        "history_length": contract.history_length,
        "future_frames": contract.num_future_frames,
        "align_reference_heading": args.align_reference_heading,
        "action_dim": contract.action_dim,
        "policy_input_dim": contract.combined_policy_input_dim,
        "policy_visual_and_collision": "direct_bumi3_xml_dynamics_model",
        "show_reference": args.show_reference,
        "reference_alpha": args.reference_alpha,
    }
    print("BUMI3_SIM2SIM_RESOLVED=" + json.dumps(resolved, ensure_ascii=False, indent=2))
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
        control_steps = max(1, remaining_frames)
    else:
        if args.duration <= 0.0:
            raise ValueError("duration 必须大于 0")
        control_steps = max(1, round(args.duration / contract.control_dt))
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
