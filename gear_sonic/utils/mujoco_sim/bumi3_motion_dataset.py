"""读取 BUMI3 sim2sim 的多轨迹数据集清单。

数据集使用 JSON 或 YAML 保存有序的 ``motions`` 列表，每项提供唯一 ``name``、
机器人动作路径 ``robot``，以及可选的配对 ``smpl`` 和容器内 ``motion_key``。
相对路径以清单所在目录为基准，不依赖启动程序时的工作目录。Robot 模式保持只读
机器人参考的原行为；SMPL 模式读取人体参考，配对机器人用于初始化和影子显示，
并要求同帧数、同帧率。SMPL 模式可省略 robot，此时明确采用默认站姿初始化，
不显示虚构的机器人参考。动作数值、帧率和关节顺序在打开窗口之前校验。
本模块只读原始数据，不执行仿真、不重采样、不改写 root 高度或源文件。
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import (
    Bumi3Contract,
    EncoderMode,
    JointOrder,
    QuaternionOrder,
    ReferenceMotion,
    load_reference_motion,
    quaternion_heading,
)
from gear_sonic.utils.mujoco_sim.bumi3_smpl_reference import load_smpl_reference


def load_smpl_reference_motion(
    path: str | Path, contract: Bumi3Contract, *, motion_key: str | None = None,
    robot_path: str | Path | None = None, robot_motion_key: str | None = None,
    joint_order: JointOrder = "auto", quaternion_order: QuaternionOrder = "auto",
) -> ReferenceMotion:
    """连接人体参考和可选机器人初始化；仅 SMPL 时用机器人默认站姿及人体参考 yaw。"""

    smpl = load_smpl_reference(path, motion_key=motion_key, target_fps=contract.target_fps)
    if robot_path is not None:
        robot = load_reference_motion(
            robot_path, contract, motion_key=robot_motion_key,
            joint_order=joint_order, quaternion_order=quaternion_order,
        )
        if robot.num_frames != smpl.num_frames or not np.isclose(robot.fps, smpl.fps):
            raise ValueError(
                f"Robot/SMPL 配对帧数或 FPS 不一致: robot={robot.num_frames}/{robot.fps}, "
                f"smpl={smpl.num_frames}/{smpl.fps}；请使用同一动作的已对齐参考"
            )
        return replace(robot, smpl_reference=smpl, name=smpl.name)
    count = smpl.num_frames
    return ReferenceMotion(
        joint_pos_policy=np.repeat(contract.default_policy[None], count, axis=0).astype(np.float32),
        joint_vel_policy=np.zeros((count, contract.action_dim), dtype=np.float32),
        root_position_world=np.repeat(contract.initial_root_position[None], count, axis=0),
        root_quat_wxyz=quaternion_heading(smpl.root_quat_wxyz),
        root_lin_vel_world=np.zeros((count, 3)), root_ang_vel_world=np.zeros((count, 3)),
        fps=smpl.fps, name=smpl.name, smpl_reference=smpl, has_robot_reference=False,
    )


def load_motion_dataset(
    path: str | Path, contract: Bumi3Contract, *, encoder: EncoderMode = "robot",
) -> list[ReferenceMotion]:
    """按清单顺序加载所有轨迹；格式、重复名称或缺失路径均在打开窗口前报错。"""

    path = Path(path).expanduser().resolve()
    contract.policy_input_dim(encoder)
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or content.get("version") != 1:
        raise ValueError("数据集清单必须是含 version: 1 的 JSON/YAML 对象")
    if content.get("robot_type") != "bumi3":
        raise ValueError("数据集 robot_type 必须为 bumi3")
    entries = content.get("motions")
    if not isinstance(entries, list) or not entries:
        raise ValueError("数据集 motions 必须为非空列表")
    names = set()
    motions = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"motions[{index}] 必须为对象")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError(f"motions[{index}] 的 name 必须非空且不能重复")
        names.add(name)
        paths = {}
        for field in ("robot", "smpl"):
            value = entry.get(field)
            required_field = "smpl" if encoder == "smpl" else "robot"
            if field != required_field and value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"轨迹 {name} 缺少有效的 {field} 路径")
            target = Path(value).expanduser()
            target = (path.parent / target).resolve() if not target.is_absolute() else target.resolve()
            if not target.exists():
                raise FileNotFoundError(f"轨迹 {name} 的 {field} 不存在: {target}")
            paths[field] = target
        if encoder == "smpl":
            motion = load_smpl_reference_motion(
                paths["smpl"], contract,
                motion_key=entry.get("smpl_motion_key", entry.get("motion_key")),
                robot_path=paths.get("robot"), robot_motion_key=entry.get("motion_key"),
                joint_order=entry.get("joint_order", "auto"),
                quaternion_order=entry.get("quaternion_order", "auto"),
            )
        else:
            motion = load_reference_motion(
                paths["robot"], contract,
                motion_key=entry.get("motion_key"),
                joint_order=entry.get("joint_order", "auto"),
                quaternion_order=entry.get("quaternion_order", "auto"),
            )
        motions.append(replace(motion, name=name))
    return motions
