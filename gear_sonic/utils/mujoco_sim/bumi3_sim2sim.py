# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BUMI3 原生 SONIC 的 MuJoCo sim2sim 核心实现。

本模块把现有 G1 部署链路中的关键契约改写为可独立验证的 BUMI3 Python 实现：
按名称读取 21 个关节的 ``jnt_qposadr/jnt_dofadr``，构造与训练一致的 10 帧
proprioception 历史、10 个 0.1 秒间隔的 Robot Encoder 参考帧；训练与 sim2sim
统一使用浮动根 ``base_link`` 作为 Robot Encoder、位置、姿态和重力方向锚点，避免
在基础闭环中混用根连杆与腰部连杆。运行器优先从动作加载 root
position/quaternion 和关节状态进行 reset，
并兼容 BUMI3 重定向器导出的 Mimic NPZ：当文件只提供全身
``body_pos_w/body_quat_w`` 时，必须依据 ``body_names`` 精确定位配置中的参考根 body，
不得假设根 body 永远位于数组第 0 项。
只在旧动作缺少 root translation 时使用接地附近的固定高度，然后调用
``eval_agent_trl.py`` 导出的 ``*_g1.onnx`` 联合模型得到 21 维动作。这里的 ``g1``
只是 SONIC 为 checkpoint 兼容保留的 Robot Encoder 内部键名，并不使用 G1 资产。

实现刻意不接入 G1 专用的 29 电机、Unitree DDS 或 C++ 硬件映射。仿真端使用
BUMI3 配置中的 PD、力矩上限和 ``0.25 * effort / stiffness`` 动作缩放，在
``sim_dt=0.005``、``decimation=4`` 下运行。所有顺序、维度、ONNX 输入输出和有限值
都会在启动时检查；任何不一致都会直接报错，而不是截断或补齐数据。GUI 默认把同一
Robot 参考 qpos 经 MJCF FK 后作为红色半透明 decorative 影子叠加显示；影子使用独立
``MjData``、不参与物理，也不以实际机器人高度覆盖参考根高，便于直接区分“参考数据
横躺”和“策略/部署跟踪失败”。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Literal, Protocol

import joblib
import mujoco
import numpy as np
import yaml


JointOrder = Literal["auto", "policy", "isaaclab", "mujoco"]
QuaternionOrder = Literal["auto", "wxyz", "xyzw"]

DEFAULT_BUMI3_SIM2SIM_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "sim2sim" / "bumi3_sonic.yaml"
)

def _target_order_indices(source_names: list[str], target_names: list[str]) -> np.ndarray:
    """按名称生成把 source 数组重排成 target 数组的索引。"""

    if len(source_names) != len(set(source_names)):
        raise ValueError("source 关节名称存在重复")
    if len(target_names) != len(set(target_names)):
        raise ValueError("target 关节名称存在重复")
    if set(source_names) != set(target_names):
        missing = sorted(set(target_names) - set(source_names))
        extra = sorted(set(source_names) - set(target_names))
        raise ValueError(f"关节集合不一致: missing={missing}, extra={extra}")
    return np.asarray([source_names.index(name) for name in target_names], dtype=np.int64)


def _normalize_quaternion(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quaternion_wxyz, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("检测到零范数 quaternion")
    return quaternion_wxyz / norm


def _quaternion_to_wxyz(quaternion: np.ndarray, order: str) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1] != 4:
        raise ValueError(f"quaternion 最后一维必须为 4，实际为 {quaternion.shape}")
    if order == "wxyz":
        result = quaternion
    elif order == "xyzw":
        result = quaternion[..., [3, 0, 1, 2]]
    else:
        raise ValueError(f"不支持 quaternion 顺序: {order}")
    return _normalize_quaternion(result)


def quaternion_conjugate(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """返回 wxyz quaternion 的共轭。"""

    result = np.asarray(quaternion_wxyz, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply(left_wxyz: np.ndarray, right_wxyz: np.ndarray) -> np.ndarray:
    """计算两个 wxyz quaternion 的 Hamilton 乘积。"""

    lw, lx, ly, lz = np.moveaxis(np.asarray(left_wxyz), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(right_wxyz), -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quaternion_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """把 wxyz quaternion 转成旋转矩阵，公式与 Isaac Lab 一致。"""

    quaternion_wxyz = _normalize_quaternion(quaternion_wxyz)
    w, x, y, z = np.moveaxis(quaternion_wxyz, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion_wxyz.shape[:-1] + (3, 3))


def quaternion_to_rotation_6d(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """返回旋转矩阵前两列按行展开的 6D 表示。"""

    matrix = quaternion_to_matrix(quaternion_wxyz)
    return matrix[..., :2].reshape(matrix.shape[:-2] + (6,))


def quaternion_heading(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """只保留 wxyz quaternion 的 world-Z heading，复刻 G1 动作起点对齐。"""

    matrix = quaternion_to_matrix(quaternion_wxyz)
    yaw = np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])
    half_yaw = 0.5 * yaw
    zeros = np.zeros_like(half_yaw)
    return np.stack((np.cos(half_yaw), zeros, zeros, np.sin(half_yaw)), axis=-1)


@dataclass(frozen=True)
class Bumi3Contract:
    """从 YAML 解析出的 BUMI3 仿真、网络和执行器契约。"""

    config_path: Path
    model_path: Path
    sim_dt: float
    decimation: int
    target_fps: float
    align_reference_heading: bool
    history_length: int
    num_future_frames: int
    future_frame_stride: int
    action_clip: float
    anchor_body_name: str
    reference_root_body_name: str
    mujoco_joint_names: tuple[str, ...]
    policy_joint_names: tuple[str, ...]
    policy_to_mujoco: np.ndarray
    mujoco_to_policy: np.ndarray
    default_mujoco: np.ndarray
    default_policy: np.ndarray
    effort_mujoco: np.ndarray
    velocity_mujoco: np.ndarray
    stiffness_mujoco: np.ndarray
    damping_mujoco: np.ndarray
    armature_mujoco: np.ndarray
    action_scale_mujoco: np.ndarray
    action_scale_policy: np.ndarray
    initial_root_position: np.ndarray
    initial_root_quaternion_wxyz: np.ndarray
    actor_proprioception_dim: int
    robot_tokenizer_dim: int
    combined_policy_input_dim: int
    action_dim: int
    token_dim: int

    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.decimation

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Bumi3Contract":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if raw.get("robot_type") != "bumi3":
            raise ValueError(f"robot_type 必须为 bumi3: {raw.get('robot_type')!r}")

        simulation = raw["simulation"]
        sonic = raw["sonic"]
        mujoco_names = list(raw["mujoco_joint_names"])
        policy_names = list(raw["policy_joint_names"])
        policy_to_mujoco = _target_order_indices(policy_names, mujoco_names)
        mujoco_to_policy = _target_order_indices(mujoco_names, policy_names)

        if policy_to_mujoco.tolist() != [
            2, 5, 9, 13, 17, 6, 10, 14, 18, 0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20
        ]:
            raise ValueError("BUMI3 IsaacLab -> MuJoCo 映射与训练契约不一致")
        if mujoco_to_policy.tolist() != [
            9, 15, 0, 10, 16, 1, 5, 11, 17, 2, 6, 12, 18, 3, 7, 13, 19, 4, 8, 14, 20
        ]:
            raise ValueError("BUMI3 MuJoCo -> IsaacLab 映射与训练契约不一致")

        defaults = raw["default_joint_positions"]
        if set(defaults) != set(mujoco_names):
            raise ValueError("default_joint_positions 未精确覆盖 21 个 BUMI3 关节")
        default_mujoco = np.asarray([defaults[name] for name in mujoco_names], dtype=np.float64)

        parameters: dict[str, dict[str, float]] = {}
        for group in raw["actuator_groups"].values():
            for joint_name in group["joint_names"]:
                if joint_name in parameters:
                    raise ValueError(f"关节 {joint_name} 被多个 actuator group 覆盖")
                parameters[joint_name] = {
                    key: float(group[key])
                    for key in (
                        "effort_limit",
                        "velocity_limit",
                        "stiffness",
                        "damping",
                        "armature",
                    )
                }
        if set(parameters) != set(mujoco_names):
            missing = sorted(set(mujoco_names) - set(parameters))
            extra = sorted(set(parameters) - set(mujoco_names))
            raise ValueError(f"actuator group 覆盖错误: missing={missing}, extra={extra}")

        def parameter_array(name: str) -> np.ndarray:
            return np.asarray([parameters[joint][name] for joint in mujoco_names], dtype=np.float64)

        effort = parameter_array("effort_limit")
        velocity = parameter_array("velocity_limit")
        stiffness = parameter_array("stiffness")
        damping = parameter_array("damping")
        armature = parameter_array("armature")
        if np.any(stiffness <= 0.0):
            raise ValueError("stiffness 必须全部大于零")
        action_scale_mujoco = 0.25 * effort / stiffness

        model_path = (config_path.parent / raw["model_path"]).resolve()
        contract = cls(
            config_path=config_path,
            model_path=model_path,
            sim_dt=float(simulation["sim_dt"]),
            decimation=int(simulation["decimation"]),
            target_fps=float(simulation["target_fps"]),
            align_reference_heading=bool(simulation["align_reference_heading"]),
            history_length=int(sonic["history_length"]),
            num_future_frames=int(sonic["num_future_frames"]),
            future_frame_stride=int(sonic["future_frame_stride"]),
            action_clip=float(sonic["action_clip"]),
            anchor_body_name=str(sonic["anchor_body_name"]),
            reference_root_body_name=str(sonic["reference_root_body_name"]),
            mujoco_joint_names=tuple(mujoco_names),
            policy_joint_names=tuple(policy_names),
            policy_to_mujoco=policy_to_mujoco,
            mujoco_to_policy=mujoco_to_policy,
            default_mujoco=default_mujoco,
            default_policy=default_mujoco[mujoco_to_policy],
            effort_mujoco=effort,
            velocity_mujoco=velocity,
            stiffness_mujoco=stiffness,
            damping_mujoco=damping,
            armature_mujoco=armature,
            action_scale_mujoco=action_scale_mujoco,
            action_scale_policy=action_scale_mujoco[mujoco_to_policy],
            initial_root_position=np.asarray(
                simulation["initial_root_position"], dtype=np.float64
            ),
            initial_root_quaternion_wxyz=_normalize_quaternion(
                np.asarray(simulation["initial_root_quaternion_wxyz"], dtype=np.float64)
            ),
            actor_proprioception_dim=int(sonic["actor_proprioception_dim"]),
            robot_tokenizer_dim=int(sonic["robot_tokenizer_dim"]),
            combined_policy_input_dim=int(sonic["combined_policy_input_dim"]),
            action_dim=int(sonic["action_dim"]),
            token_dim=int(sonic["token_dim"]),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        """验证不可被 sim2sim 静默改变的 SONIC/BUMI3 训练常量。"""

        if not self.model_path.is_file():
            raise FileNotFoundError(f"BUMI3 MJCF 不存在: {self.model_path}")
        if not np.isclose(self.sim_dt, 0.005):
            raise ValueError(f"sim_dt 必须为 0.005，实际为 {self.sim_dt}")
        if self.decimation != 4:
            raise ValueError(f"decimation 必须为 4，实际为 {self.decimation}")
        if not np.isclose(1.0 / self.control_dt, 50.0):
            raise ValueError(f"控制频率必须为 50 Hz，实际为 {1.0 / self.control_dt}")
        if not np.isclose(self.target_fps, 50.0):
            raise ValueError(f"target_fps 必须为 50，实际为 {self.target_fps}")
        if self.history_length != 10 or self.num_future_frames != 10:
            raise ValueError("SONIC history_length 和 num_future_frames 必须均为 10")
        if self.future_frame_stride != 5:
            raise ValueError("0.1 秒 future frame 在 50 FPS 下必须使用 stride=5")
        if len(self.mujoco_joint_names) != 21 or self.action_dim != 21:
            raise ValueError("BUMI3 必须为 21 DoF / 21 维动作")
        expected_proprio = self.history_length * (3 + 21 + 21 + 21 + 3)
        expected_tokenizer = self.num_future_frames * (21 + 21 + 6)
        if self.actor_proprioception_dim != expected_proprio:
            raise ValueError(
                f"actor proprioception 应为 {expected_proprio}，实际为 "
                f"{self.actor_proprioception_dim}"
            )
        if self.robot_tokenizer_dim != expected_tokenizer:
            raise ValueError(
                f"Robot Encoder tokenizer 应为 {expected_tokenizer}，实际为 "
                f"{self.robot_tokenizer_dim}"
            )
        if self.combined_policy_input_dim != expected_proprio + expected_tokenizer:
            raise ValueError("联合 ONNX 输入维度必须为 tokenizer 480 + proprioception 690")
        if self.token_dim != 64:
            raise ValueError("SONIC FSQ token 总维度必须为 64")


@dataclass(frozen=True)
class ReferenceMotion:
    """已转换到 50 FPS、策略关节顺序和 wxyz 浮动根状态的动作。

    ``root_position_world`` 对旧 CSV 可以为空；正常 SONIC PKL/NPZ 应提供训练时使用的
    root translation。加载器会把整段水平根轨迹平移为首帧 ``x=y=0``，但保持每帧
    相对位移和原始 ``z`` 不变。Robot Encoder 仍通过加载 BUMI3 MJCF 后的命名刚体获取
    锚点；当前契约将该刚体固定为 ``base_link``，因此腰关节姿态不再改变参考锚点。
    """

    joint_pos_policy: np.ndarray
    joint_vel_policy: np.ndarray
    root_position_world: np.ndarray | None
    root_quat_wxyz: np.ndarray
    fps: float
    name: str

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos_policy.shape[0])

    def future_indices(
        self,
        frame: int,
        count: int,
        stride: int,
        loop: bool,
    ) -> np.ndarray:
        indices = frame + np.arange(count, dtype=np.int64) * stride
        if loop:
            return indices % self.num_frames
        return np.minimum(indices, self.num_frames - 1)


def _read_csv(path: Path) -> np.ndarray:
    """读取 G1 MotionDataReader 兼容 CSV，并自动处理可选表头。"""

    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    try:
        [float(value) for value in first_line.split(",")]
        skip_header = 0
    except ValueError:
        skip_header = 1
    value = np.loadtxt(path, delimiter=",", skiprows=skip_header, ndmin=2)
    return np.asarray(value, dtype=np.float64)


def _unwrap_motion_mapping(data: Any, motion_key: str | None) -> tuple[dict[str, Any], str]:
    """从单动作或 ``{motion_key: motion}`` 容器中选择一个动作。"""

    if not isinstance(data, dict):
        raise TypeError(f"动作文件顶层必须为 dict，实际为 {type(data).__name__}")
    direct_fields = {"dof", "dof_pos", "joint_pos", "qpos"}
    if direct_fields.intersection(data):
        return data, motion_key or "motion"
    keys = [key for key, value in data.items() if isinstance(value, dict)]
    if motion_key is None:
        if len(keys) != 1:
            raise ValueError(f"动作文件含 {len(keys)} 个动作，请用 --motion-key 选择: {keys[:20]}")
        motion_key = keys[0]
    if motion_key not in data or not isinstance(data[motion_key], dict):
        raise KeyError(f"motion_key={motion_key!r} 不存在或不是动作 dict")
    return data[motion_key], str(motion_key)


def _extract_first(mapping: dict[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _extract_named_body_series(
    mapping: dict[str, Any],
    field_name: str,
    body_name: str,
    value_width: int,
) -> np.ndarray | None:
    """从 Mimic 全身 body 数组中按名称提取一个 body 的逐帧数据。

    BUMI3 重定向器生成的 Mimic NPZ 保存 ``body_pos_w[T,B,3]`` 和
    ``body_quat_w[T,B,4]``，但不重复保存顶层 root 字段。这里必须使用同文件内的
    ``body_names`` 定位配置指定的 ``reference_root_body_name``；不能默认使用索引 0，
    因为 body 顺序属于数据契约，未来导出器或资产调整后可能改变。缺少名称、名称重复、
    根 body 不存在或数组 shape 不匹配都会立即报错，避免把腰部等非根连杆静默当作浮动根。
    """

    if field_name not in mapping:
        return None
    if "body_names" not in mapping:
        raise ValueError(f"{field_name} 兼容格式必须同时提供 body_names")

    body_names_array = np.asarray(mapping["body_names"])
    if body_names_array.ndim != 1 or body_names_array.size == 0:
        raise ValueError(f"body_names 必须是一维非空数组，实际为 {body_names_array.shape}")
    body_names = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in body_names_array.tolist()
    ]
    body_indices = [index for index, name in enumerate(body_names) if name == body_name]
    if len(body_indices) != 1:
        raise ValueError(
            f"body_names 必须恰好包含一个 reference_root_body_name={body_name!r}，"
            f"实际匹配 {len(body_indices)} 个"
        )

    values = np.asarray(mapping[field_name])
    expected_tail = (len(body_names), value_width)
    if values.ndim != 3 or values.shape[1:] != expected_tail:
        raise ValueError(
            f"{field_name} 必须为 [T,{len(body_names)},{value_width}]，实际为 {values.shape}"
        )
    return values[:, body_indices[0], :]


def load_reference_motion(
    path: str | Path,
    contract: Bumi3Contract,
    *,
    motion_key: str | None = None,
    joint_order: JointOrder = "auto",
    quaternion_order: QuaternionOrder = "auto",
) -> ReferenceMotion:
    """加载 SONIC PKL/NPZ 或 G1 MotionDataReader 风格 CSV 目录。

    自动模式下，SONIC PKL 的 ``dof`` 视为 MuJoCo 顺序、``root_rot`` 视为 xyzw；
    CSV 的 ``joint_pos.csv`` 视为策略顺序、``body_quat.csv`` 视为 wxyz。NPZ 会读取
    ``joint_order`` 和 ``quaternion_convention/quaternion_order`` 元数据；若顶层没有
    root pose，则从 ``body_pos_w/body_quat_w`` 中按 ``body_names`` 精确提取配置指定的
    参考根 body。元数据缺失时分别采用策略顺序和 wxyz。所有输入必须已经是 50 FPS，
    以避免部署端隐式重采样改变训练时间契约。若输入含根平移，返回前会让整段
    ``root x/y`` 减去首帧水平位置；该刚体平移不改变相对运动轨迹，也不会改动根高度。
    """

    motion_path = Path(path).expanduser().resolve()
    source_kind: str
    name = motion_path.stem

    if motion_path.is_dir():
        source_kind = "csv"
        joint_path = motion_path / "joint_pos.csv"
        if not joint_path.is_file():
            subdirs = sorted(
                child for child in motion_path.iterdir() if child.is_dir() and (child / "joint_pos.csv").is_file()
            )
            if motion_key is None:
                if len(subdirs) != 1:
                    raise ValueError(
                        f"目录含 {len(subdirs)} 个 CSV 动作，请用 --motion-key 选择: "
                        f"{[item.name for item in subdirs[:20]]}"
                    )
                motion_path = subdirs[0]
            else:
                motion_path = motion_path / motion_key
            joint_path = motion_path / "joint_pos.csv"
        joint_pos = _read_csv(joint_path)
        velocity_path = motion_path / "joint_vel.csv"
        joint_vel = _read_csv(velocity_path) if velocity_path.is_file() else None
        quaternion_path = motion_path / "body_quat.csv"
        if not quaternion_path.is_file():
            raise FileNotFoundError(f"CSV 动作缺少 body_quat.csv: {motion_path}")
        body_quat = _read_csv(quaternion_path)
        if body_quat.shape[1] < 4 or body_quat.shape[1] % 4 != 0:
            raise ValueError(f"body_quat.csv 列数必须是 4 的倍数: {body_quat.shape}")
        root_quat = body_quat[:, :4]
        position_path = motion_path / "body_pos.csv"
        if position_path.is_file():
            body_position = _read_csv(position_path)
            if body_position.shape[1] < 3 or body_position.shape[1] % 3 != 0:
                raise ValueError(f"body_pos.csv 列数必须是 3 的倍数: {body_position.shape}")
            root_position = body_position[:, :3]
        else:
            root_position = None
        fps = contract.target_fps
        name = motion_path.name
        metadata_joint_order = "policy"
        metadata_quaternion_order = "wxyz"
    elif motion_path.suffix.lower() == ".npz":
        source_kind = "npz"
        with np.load(motion_path, allow_pickle=True) as loaded:
            mapping = {key: loaded[key] for key in loaded.files}
        mapping, name = _unwrap_motion_mapping(mapping, motion_key)
        joint_pos = _extract_first(mapping, ("joint_pos", "dof_pos", "dof"))
        qpos = _extract_first(mapping, ("qpos",))
        if joint_pos is None and qpos is not None:
            qpos = np.asarray(qpos)
            if qpos.shape[-1] != 28:
                raise ValueError(f"BUMI3 qpos 必须为 root7+joint21，实际为 {qpos.shape}")
            joint_pos = qpos[:, 7:]
        joint_vel = _extract_first(mapping, ("joint_vel", "dof_vel"))
        root_quat = _extract_first(mapping, ("root_quat", "root_rot"))
        if root_quat is None and qpos is not None:
            root_quat = qpos[:, 3:7]
        root_position = _extract_first(
            mapping,
            ("root_trans_offset", "root_pos", "root_position", "root_translation"),
        )
        if root_position is None and qpos is not None:
            root_position = qpos[:, :3]
        if root_position is None:
            root_position = _extract_named_body_series(
                mapping,
                "body_pos_w",
                contract.reference_root_body_name,
                3,
            )
        if root_quat is None:
            root_quat = _extract_named_body_series(
                mapping,
                "body_quat_w",
                contract.reference_root_body_name,
                4,
            )
        fps = float(np.asarray(mapping.get("fps", contract.target_fps)).reshape(-1)[0])
        metadata_joint_order = str(np.asarray(mapping.get("joint_order", "policy")).item())
        metadata_quaternion_order = str(
            np.asarray(
                mapping.get(
                    "quaternion_convention",
                    mapping.get("quaternion_order", "wxyz"),
                )
            ).item()
        )
    elif motion_path.suffix.lower() in {".pkl", ".joblib"}:
        source_kind = "pkl"
        mapping, name = _unwrap_motion_mapping(joblib.load(motion_path), motion_key)
        joint_pos = _extract_first(mapping, ("dof", "dof_pos", "joint_pos"))
        joint_vel = _extract_first(mapping, ("dof_vel", "joint_vel"))
        root_quat = _extract_first(mapping, ("root_rot", "root_quat"))
        root_position = _extract_first(
            mapping,
            ("root_trans_offset", "root_pos", "root_position", "root_translation"),
        )
        fps = float(mapping.get("fps", contract.target_fps))
        metadata_joint_order = str(mapping.get("joint_order", "mujoco"))
        metadata_quaternion_order = str(mapping.get("quaternion_convention", "xyzw"))
    else:
        raise ValueError(f"不支持的动作路径: {motion_path}")

    if joint_pos is None or root_quat is None:
        raise ValueError(f"{source_kind} 动作必须同时提供 joint position 和 root quaternion")
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    root_quat = np.asarray(root_quat, dtype=np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != contract.action_dim:
        raise ValueError(f"joint position 必须为 [T,21]，实际为 {joint_pos.shape}")
    if root_quat.ndim == 3:
        root_quat = root_quat[:, 0]
    if root_quat.ndim != 2 or root_quat.shape[1] != 4:
        raise ValueError(f"root quaternion 必须为 [T,4]，实际为 {root_quat.shape}")
    if joint_pos.shape[0] != root_quat.shape[0] or joint_pos.shape[0] < 2:
        raise ValueError("joint/root frame 数必须一致且至少为 2")
    if root_position is not None:
        root_position = np.asarray(root_position, dtype=np.float64)
        if root_position.ndim != 2 or root_position.shape != (joint_pos.shape[0], 3):
            raise ValueError(
                f"root position 必须为 [T,3] 且与动作等长，实际为 {root_position.shape}"
            )
    if not np.isclose(fps, contract.target_fps):
        raise ValueError(
            f"动作必须预先转换为 {contract.target_fps:g} FPS，实际为 {fps:g} FPS；"
            "sim2sim 不隐式重采样训练参考"
        )

    resolved_joint_order = metadata_joint_order if joint_order == "auto" else joint_order
    if resolved_joint_order == "isaaclab":
        resolved_joint_order = "policy"
    if resolved_joint_order == "mujoco":
        joint_pos = joint_pos[:, contract.mujoco_to_policy]
        if joint_vel is not None:
            joint_vel = np.asarray(joint_vel, dtype=np.float64)[:, contract.mujoco_to_policy]
    elif resolved_joint_order != "policy":
        raise ValueError(f"不支持 joint_order={resolved_joint_order!r}")

    if joint_vel is None:
        joint_vel = np.gradient(joint_pos, 1.0 / fps, axis=0, edge_order=1)
    joint_vel = np.asarray(joint_vel, dtype=np.float64)
    if joint_vel.shape != joint_pos.shape:
        raise ValueError(f"joint velocity shape 错误: {joint_vel.shape} != {joint_pos.shape}")

    resolved_quaternion_order = (
        metadata_quaternion_order if quaternion_order == "auto" else quaternion_order
    )
    root_quat_wxyz = _quaternion_to_wxyz(root_quat, resolved_quaternion_order)
    for label, value in (
        ("joint_pos", joint_pos),
        ("joint_vel", joint_vel),
        ("root_quat", root_quat_wxyz),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"{label} 含 NaN/Inf")
    if root_position is not None:
        if not np.isfinite(root_position).all():
            raise ValueError("root_position 含 NaN/Inf")
        # MuJoCo 默认视角和 XML 世界原点都以 (0, 0) 为场景中心。对整段水平根轨迹
        # 施加同一个刚体平移，既能让首帧稳定出现在画面中心，又严格保留逐帧位移、
        # 速度和转向路径；Z 轴必须保持数据原值，避免把地面接触问题隐藏成高度修正。
        root_position = root_position.copy()
        root_position[:, :2] -= root_position[0, :2]
    return ReferenceMotion(
        joint_pos_policy=joint_pos.astype(np.float32),
        joint_vel_policy=joint_vel.astype(np.float32),
        root_position_world=(
            None if root_position is None else root_position.astype(np.float64)
        ),
        root_quat_wxyz=root_quat_wxyz.astype(np.float64),
        fps=fps,
        name=name,
    )


class Policy(Protocol):
    """联合 Robot Encoder + dynamic decoder 的最小推理协议。"""

    input_dim: int
    output_dim: int

    def __call__(self, observation: np.ndarray) -> np.ndarray: ...


class OnnxRobotPolicy:
    """加载 ``*_g1.onnx``，并验证其输入为 1170、输出为 21。"""

    def __init__(self, path: str | Path, contract: Bumi3Contract, provider: str = "cpu"):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                '缺少 onnxruntime；请执行 pip install -e "gear_sonic[sim]"'
            ) from error
        policy_path = Path(path).expanduser().resolve()
        if not policy_path.is_file():
            raise FileNotFoundError(f"ONNX policy 不存在: {policy_path}")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(policy_path), providers=providers)
        if len(self.session.get_inputs()) != 1 or len(self.session.get_outputs()) != 1:
            raise ValueError("BUMI3 联合 ONNX 必须只有一个输入和一个输出")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        self.input_dim = int(input_shape[-1])
        self.output_dim = int(output_shape[-1])
        if self.input_dim != contract.combined_policy_input_dim:
            raise ValueError(
                f"ONNX 输入应为 {contract.combined_policy_input_dim}，实际为 {self.input_dim}"
            )
        if self.output_dim != contract.action_dim:
            raise ValueError(f"ONNX 输出应为 {contract.action_dim}，实际为 {self.output_dim}")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        batch = np.asarray(observation, dtype=np.float32).reshape(1, self.input_dim)
        action = self.session.run([self.output_name], {self.input_name: batch})[0]
        return np.asarray(action, dtype=np.float32).reshape(self.output_dim)


class ZeroPolicy:
    """只供静态/动力学 smoke 使用的零动作策略，不用于效果评估。"""

    def __init__(self, contract: Bumi3Contract):
        self.input_dim = contract.combined_policy_input_dim
        self.output_dim = contract.action_dim

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if np.asarray(observation).size != self.input_dim:
            raise ValueError("ZeroPolicy 输入维度错误")
        return np.zeros(self.output_dim, dtype=np.float32)


class Bumi3SonicSim2Sim:
    """执行 BUMI3 参考动作、SONIC ONNX 推理与 MuJoCo PD 控制闭环。"""

    def __init__(
        self,
        contract: Bumi3Contract,
        motion: ReferenceMotion,
        policy: Policy,
        *,
        loop_motion: bool = False,
        start_frame: int = 0,
        align_reference_heading: bool | None = None,
    ):
        self.contract = contract
        self.motion = motion
        self.policy = policy
        self.loop_motion = loop_motion
        self.align_reference_heading = (
            contract.align_reference_heading
            if align_reference_heading is None
            else align_reference_heading
        )
        if policy.input_dim != contract.combined_policy_input_dim:
            raise ValueError("policy input_dim 与配置不一致")
        if policy.output_dim != contract.action_dim:
            raise ValueError("policy output_dim 与配置不一致")
        if not 0 <= start_frame < motion.num_frames:
            raise ValueError(f"start_frame 越界: {start_frame}/{motion.num_frames}")
        self.motion_frame = start_frame

        self.model = mujoco.MjModel.from_xml_path(str(contract.model_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = contract.sim_dt
        self._resolve_model_contract()
        self._apply_armature_contract()
        # MuJoCo sim2sim 的白色 policy 机器人直接使用 self.model 中
        # bumi3.xml 定义的 geom 进行碰撞和渲染，不引入任何 URDF 碰撞覆盖。
        # 红色参考只为了持有与 policy 不同的 qpos，才从同一 XML 重新加载
        # 独立 ref_model + ref_data + ref_scene；它不参与物理积分。
        self.reference_visual_model = mujoco.MjModel.from_xml_path(
            str(contract.model_path)
        )
        self.reference_visual_model.opt.timestep = contract.sim_dt
        if (
            self.reference_visual_model.nq != self.model.nq
            or self.reference_visual_model.nv != self.model.nv
            or self.reference_visual_model.ngeom != self.model.ngeom
        ):
            raise ValueError("参考可视化模型与 BUMI3 动力学模型拓扑不一致")
        self.reference_base_body_id = mujoco.mj_name2id(
            self.reference_visual_model,
            mujoco.mjtObj.mjOBJ_BODY,
            "base_link",
        )
        self.reference_anchor_body_id = mujoco.mj_name2id(
            self.reference_visual_model,
            mujoco.mjtObj.mjOBJ_BODY,
            contract.anchor_body_name,
        )
        if self.reference_base_body_id < 0 or self.reference_anchor_body_id < 0:
            raise ValueError("参考可视化模型缺少 base/anchor body")
        self.reference_visual_data = mujoco.MjData(self.reference_visual_model)
        self.reference_visual_scene = mujoco.MjvScene(
            self.reference_visual_model, maxgeom=1000
        )
        self.reference_visual_camera = mujoco.MjvCamera()
        self.reference_visual_option = mujoco.MjvOption()
        self.reference_visual_perturb = mujoco.MjvPerturb()
        self.reference_anchor_quat_wxyz = self._compute_reference_anchor_quaternions()

        shape_by_name = {
            "base_ang_vel": 3,
            "joint_pos": contract.action_dim,
            "joint_vel": contract.action_dim,
            "actions": contract.action_dim,
            "gravity_dir": 3,
        }
        self.histories = {
            name: deque(
                [np.zeros(width, dtype=np.float32) for _ in range(contract.history_length)],
                maxlen=contract.history_length,
            )
            for name, width in shape_by_name.items()
        }
        self.last_action_policy = np.zeros(contract.action_dim, dtype=np.float32)
        self.last_observation = np.zeros(contract.combined_policy_input_dim, dtype=np.float32)
        self.last_torque_mujoco = np.zeros(contract.action_dim, dtype=np.float64)
        self.reference_heading_delta_wxyz = np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.reset(start_frame=start_frame)

    def _resolve_model_contract(self) -> None:
        if self.model.nq != 28 or self.model.nv != 27 or self.model.nu != 21:
            raise ValueError(
                f"BUMI3 MJCF 应为 nq=28,nv=27,nu=21，实际为 "
                f"{self.model.nq},{self.model.nv},{self.model.nu}"
            )
        self.joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.contract.mujoco_joint_names
            ],
            dtype=np.int64,
        )
        if np.any(self.joint_ids < 0):
            missing = [
                name
                for name, joint_id in zip(
                    self.contract.mujoco_joint_names, self.joint_ids, strict=True
                )
                if joint_id < 0
            ]
            raise ValueError(f"MJCF 缺少 BUMI3 关节: {missing}")
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids].copy()
        self.dof_addresses = self.model.jnt_dofadr[self.joint_ids].copy()
        if len(set(self.qpos_addresses.tolist())) != 21 or len(set(self.dof_addresses.tolist())) != 21:
            raise ValueError("BUMI3 qpos/dof address 存在重复")

        self.actuator_ids = np.empty(21, dtype=np.int64)
        for index, joint_id in enumerate(self.joint_ids):
            matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint_id)
            if len(matches) != 1:
                raise ValueError(
                    f"关节 {self.contract.mujoco_joint_names[index]} 应有且只有一个 motor"
                )
            self.actuator_ids[index] = matches[0]
        self.root_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "root"
        )
        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.contract.reference_root_body_name
        )
        self.anchor_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.contract.anchor_body_name
        )
        if min(self.root_joint_id, self.base_body_id, self.anchor_body_id) < 0:
            raise ValueError("MJCF 缺少 root joint、base body 或配置指定的 anchor body")
        self.root_qpos_address = int(self.model.jnt_qposadr[self.root_joint_id])

    def _apply_armature_contract(self) -> None:
        self.model.dof_armature[self.dof_addresses] = self.contract.armature_mujoco
        actual = self.model.dof_armature[self.dof_addresses]
        if not np.allclose(actual, self.contract.armature_mujoco, atol=1e-12):
            raise ValueError("未能应用 BUMI3 armature 配置")

    def _body_geom_ids(self, body_name: str) -> np.ndarray:
        """按 body 名称返回全部直接所属 geom，禁止依赖 XML 中的隐式编号。

        BUMI3 MJCF 将原始 mesh 保留为可视 geom，并为需要参与接触的 link 另设
        collision geom；同一个 body 因此可以合法拥有一个或两个 geom。调用方必须再按
        geom 名称或 ``group`` 区分用途，不能假设每个 body 只有一个 mesh。
        """

        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id < 0:
            raise ValueError(f"BUMI3 MJCF 缺少 collision body: {body_name}")
        start = int(self.model.body_geomadr[body_id])
        count = int(self.model.body_geomnum[body_id])
        if count <= 0:
            raise ValueError(f"{body_name} 没有直接所属 geom")
        return np.arange(start, start + count, dtype=np.int64)

    def _reference_qpos(self, frame: int) -> np.ndarray:
        """按动作帧构造完整 MuJoCo qpos；旧动作缺根平移时才使用固定回退位置。"""

        qpos = np.zeros(self.model.nq, dtype=np.float64)
        root = self.root_qpos_address
        root_position = (
            self.contract.initial_root_position
            if self.motion.root_position_world is None
            else self.motion.root_position_world[frame]
        )
        qpos[root : root + 3] = root_position
        qpos[root + 3 : root + 7] = self.motion.root_quat_wxyz[frame]
        qpos[self.qpos_addresses] = self.motion.joint_pos_policy[frame][
            self.contract.policy_to_mujoco
        ]
        return qpos

    def _aligned_reference_qpos(self, frame: int) -> np.ndarray:
        """构造用于观测和影子显示的参考 qpos，并只施加可选的起始 heading 对齐。

        根高度始终来自动作文件（或明确的旧数据回退值），不会跟随已经摔倒的真实
        robot。这样如果数据中的参考本身横躺，红色影子也会原样横躺，不能被渲染
        层的高度修正掩盖。
        """

        qpos = self._reference_qpos(frame)
        if not self.align_reference_heading:
            return qpos

        root = self.root_qpos_address
        alignment_frame = getattr(self, "reference_alignment_frame", frame)
        alignment_origin = self._reference_qpos(alignment_frame)[root : root + 3]
        root_delta = qpos[root : root + 3] - alignment_origin
        heading_matrix = quaternion_to_matrix(self.reference_heading_delta_wxyz)
        qpos[root : root + 3] = alignment_origin + heading_matrix @ root_delta
        qpos[root + 3 : root + 7] = _normalize_quaternion(
            quaternion_multiply(
                self.reference_heading_delta_wxyz,
                qpos[root + 3 : root + 7],
            )
        )
        return qpos

    def _update_reference_visual_data(self, frame: int) -> None:
        """把当前参考帧写入独立 FK data；不调用 ``mj_step``。"""

        self.reference_visual_data.qpos[:] = self._aligned_reference_qpos(frame)
        self.reference_visual_data.qvel[:] = 0.0
        self.reference_visual_data.ctrl[:] = 0.0
        mujoco.mj_forward(self.reference_visual_model, self.reference_visual_data)

    def reference_pose_diagnostics(self, frame: int) -> dict[str, float]:
        """返回参考 base 与配置锚点相对世界 Z 轴的倾角，辅助判断是否横躺。"""

        if not 0 <= frame < self.motion.num_frames:
            raise ValueError(f"reference diagnostics frame 越界: {frame}")
        self._update_reference_visual_data(frame)
        world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

        def tilt_degrees(body_id: int) -> float:
            rotation = quaternion_to_matrix(self.reference_visual_data.xquat[body_id])
            up_dot = float(np.clip(rotation[:, 2] @ world_up, -1.0, 1.0))
            return float(np.degrees(np.arccos(up_dot)))

        root = self.root_qpos_address
        return {
            "frame": float(frame),
            "root_height": float(self.reference_visual_data.qpos[root + 2]),
            "base_tilt_degrees": tilt_degrees(self.reference_base_body_id),
            "anchor_tilt_degrees": tilt_degrees(self.reference_anchor_body_id),
            "minimum_body_origin_height": float(
                np.min(self.reference_visual_data.xpos[1:, 2])
            ),
        }

    def reference_marker_specs(
        self,
        frame: int,
        *,
        alpha: float = 0.32,
    ) -> list[dict[str, Any]]:
        """复制 ``mjv_updateScene`` 解析后的 geom 为红色半透明影子。

        这里故意与 ``sim2sim_mimic_vision_4340.py`` 保持同一条渲染路径。
        ``MjvGeom`` 已包含 mesh 的渲染级变换、尺寸、``dataid`` 和 ``matid``；
        直接使用 ``MjData.geom_xpos/geom_xmat`` 重建 marker 会丢失这些信息，
        对 mesh geom 表现为各 link 视觉位姿错乱。复制后的 geom 仍只是
        ``mjCAT_DECOR``，不参与接触或动力学。
        """

        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"reference alpha 必须位于 (0, 1]，实际为 {alpha}")
        self._update_reference_visual_data(frame)
        tint = np.asarray([1.0, 0.15, 0.15], dtype=np.float32)
        return self._resolved_visual_marker_specs(
            self.reference_visual_data,
            self.reference_visual_scene,
            self.reference_visual_option,
            self.reference_visual_perturb,
            self.reference_visual_camera,
            alpha=alpha,
            tint=tint,
        )

    def _resolved_visual_marker_specs(
        self,
        visual_data: mujoco.MjData,
        visual_scene: mujoco.MjvScene,
        visual_option: mujoco.MjvOption,
        visual_perturb: mujoco.MjvPerturb,
        visual_camera: mujoco.MjvCamera,
        *,
        alpha: float,
        tint: np.ndarray | None,
    ) -> list[dict[str, Any]]:
        """从原始 XML 的 resolved MjvScene 复制 22 个 robot 可视 mesh。

        group=3 的 collision geom 只服务 MuJoCo 接触计算，不能被重新着色后加入红色参考
        影子；否则 capsule/透明碰撞网格会覆盖原始 link 外观，使影子与 policy 机器人看起来
        使用了不同模型。这里固定只复制 XML 中 group=1 的 22 个可视 geom。
        """

        mujoco.mjv_updateScene(
            self.reference_visual_model,
            visual_data,
            visual_option,
            visual_perturb,
            visual_camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            visual_scene,
        )
        markers: list[dict[str, Any]] = []
        for scene_geom_id in range(visual_scene.ngeom):
            resolved_geom = visual_scene.geoms[scene_geom_id]
            if int(resolved_geom.objtype) != int(mujoco.mjtObj.mjOBJ_GEOM):
                continue
            model_geom_id = int(resolved_geom.objid)
            if not 0 <= model_geom_id < self.reference_visual_model.ngeom:
                continue
            if int(self.reference_visual_model.geom_bodyid[model_geom_id]) == 0:
                continue
            if int(self.reference_visual_model.geom_group[model_geom_id]) != 1:
                continue

            rgba = np.asarray(resolved_geom.rgba, dtype=np.float32).copy()
            if tint is not None:
                rgba[:3] = 0.65 * rgba[:3] + 0.35 * tint
            rgba[3] = alpha
            markers.append(
                {
                    "type": int(resolved_geom.type),
                    "size": np.asarray(resolved_geom.size, dtype=np.float64).copy(),
                    "pos": np.asarray(resolved_geom.pos, dtype=np.float64).copy(),
                    # Python binding 中 MjvGeom.mat 是 (3, 3)，mjv_initGeom 接口
                    # 要求连续 9 元素数组；数值和行主序不变。
                    "mat": np.asarray(resolved_geom.mat, dtype=np.float64)
                    .reshape(9)
                    .copy(),
                    "rgba": rgba,
                    "dataid": int(resolved_geom.dataid),
                    "matid": int(resolved_geom.matid),
                    "objtype": int(resolved_geom.objtype),
                    "objid": model_geom_id,
                    "segid": int(resolved_geom.segid),
                    "texcoord": int(resolved_geom.texcoord),
                    "transparent": int(resolved_geom.transparent),
                    "emission": float(resolved_geom.emission),
                    "specular": float(resolved_geom.specular),
                    "shininess": float(resolved_geom.shininess),
                    "reflectance": float(resolved_geom.reflectance),
                    "modelrbound": float(resolved_geom.modelrbound),
                    "camdist": float(resolved_geom.camdist),
                    "label": str(resolved_geom.label),
                }
            )
        return markers

    def _write_reference_markers_to_scene(
        self,
        scene: mujoco.MjvScene,
        frame: int,
        *,
        alpha: float,
    ) -> int:
        """刷新 viewer user scene，只写参考影子并返回 geom 数量。"""

        markers = self.reference_marker_specs(frame, alpha=alpha)
        scene.ngeom = 0
        self._append_visual_markers_to_scene(scene, markers)
        return scene.ngeom

    @staticmethod
    def _append_visual_markers_to_scene(
        scene: mujoco.MjvScene,
        markers: list[dict[str, Any]],
    ) -> int:
        """把 resolved marker 追加到 user scene，不清空既有 geom。"""

        if scene.ngeom + len(markers) > scene.maxgeom:
            raise RuntimeError(
                f"可视化需要 {scene.ngeom + len(markers)} geoms，"
                f"但 viewer 只允许 {scene.maxgeom}"
            )
        for marker in markers:
            geom = scene.geoms[scene.ngeom]
            # marker 已经是 mjv_updateScene 产生的最终 MjvGeom。这里逐字段
            # 复制，不再调 mjv_initGeom；后者会重新解释 capsule/mesh
            # size，导致与参考 scene 不一致。
            geom.type = marker["type"]
            geom.size[:] = marker["size"]
            geom.pos[:] = marker["pos"]
            geom.mat[:] = np.asarray(marker["mat"]).reshape(3, 3)
            geom.rgba[:] = marker["rgba"]
            # MjvScene 预分配 geom 的 label 字符缓冲不保证为空。
            # 不显式覆盖会把未初始化字节当作随机文字画出。
            geom.label = marker["label"]
            for field in (
                "dataid",
                "matid",
                "objtype",
                "objid",
                "segid",
                "texcoord",
                "transparent",
                "emission",
                "specular",
                "shininess",
                "reflectance",
                "modelrbound",
                "camdist",
            ):
                setattr(geom, field, marker[field])
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            scene.ngeom += 1
        return scene.ngeom

    def _compute_reference_anchor_quaternions(self) -> np.ndarray:
        """用 BUMI3 MJCF FK 计算每帧配置锚点的世界姿态。"""

        scratch = mujoco.MjData(self.reference_visual_model)
        result = np.empty((self.motion.num_frames, 4), dtype=np.float64)
        for frame in range(self.motion.num_frames):
            scratch.qpos[:] = self._reference_qpos(frame)
            mujoco.mj_kinematics(self.reference_visual_model, scratch)
            result[frame] = scratch.xquat[self.reference_anchor_body_id]
        return _normalize_quaternion(result)

    def _reference_qvel(self, frame: int) -> np.ndarray:
        """从相邻参考 qpos 求浮动根速度，并保留动作文件给出的关节速度。"""

        next_frame = frame + 1
        if next_frame >= self.motion.num_frames:
            next_frame = 0 if self.loop_motion else frame
        qvel = np.zeros(self.model.nv, dtype=np.float64)
        if next_frame != frame:
            mujoco.mj_differentiatePos(
                self.model,
                qvel,
                1.0 / self.motion.fps,
                self._reference_qpos(frame),
                self._reference_qpos(next_frame),
            )
        qvel[self.dof_addresses] = self.motion.joint_vel_policy[frame][
            self.contract.policy_to_mujoco
        ]
        return qvel

    def reset(self, *, start_frame: int = 0) -> None:
        """按参考动作重置浮动根、关节状态、历史缓冲和播放位置。"""

        if not 0 <= start_frame < self.motion.num_frames:
            raise ValueError(f"start_frame 越界: {start_frame}/{self.motion.num_frames}")

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._reference_qpos(start_frame)
        self.data.qvel[:] = self._reference_qvel(start_frame)
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._update_reference_heading_alignment(start_frame)
        self.last_action_policy.fill(0.0)
        self.last_observation.fill(0.0)
        self.last_torque_mujoco.fill(0.0)
        initial_state = self._current_state()
        for name, history in self.histories.items():
            history.clear()
            # Isaac Lab CircularBuffer 首次 append 会把当前值复制到完整 history；这里必须
            # 复刻该行为，不能让 episode 开头的前 9 帧变成训练中未出现的全零状态。
            history.extend(
                initial_state[name].copy() for _ in range(self.contract.history_length)
            )
        self.motion_frame = start_frame

    def _update_reference_heading_alignment(self, start_frame: int) -> None:
        """把参考 ``base_link`` 起始 yaw 对齐到当前根锚点 yaw。"""

        self.reference_alignment_frame = start_frame
        if not self.align_reference_heading:
            self.reference_heading_delta_wxyz = np.asarray(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
            return
        robot_heading = quaternion_heading(self.data.xquat[self.anchor_body_id])
        reference_heading = quaternion_heading(
            self.reference_anchor_quat_wxyz[start_frame]
        )
        self.reference_heading_delta_wxyz = _normalize_quaternion(
            quaternion_multiply(robot_heading, quaternion_conjugate(reference_heading))
        )

    def _joint_state_policy(self) -> tuple[np.ndarray, np.ndarray]:
        q_mujoco = self.data.qpos[self.qpos_addresses]
        dq_mujoco = self.data.qvel[self.dof_addresses]
        return (
            q_mujoco[self.contract.mujoco_to_policy],
            dq_mujoco[self.contract.mujoco_to_policy],
        )

    def _base_angular_velocity_local(self) -> np.ndarray:
        """返回 ``base_link`` 连杆坐标系中的根角速度。

        BUMI3 的 ``fullinertia`` 含非对角项，MuJoCo 的
        ``mj_objectVelocity(..., flg_local=1)`` 会把角速度表达在对角化后的惯性
        主轴中，而 Isaac Lab 的 ``root_ang_vel_b`` 使用根连杆轴。这里先读取世界
        角速度，再显式乘以 ``base_link`` 世界旋转的转置，确保训练与 sim2sim
        使用同一坐标系；该修正不改变积分器、控制频率或 PD 控制方式。
        """
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body_id,
            velocity,
            0,
        )
        base_rotation_world = self.data.xmat[self.base_body_id].reshape(3, 3)
        return (base_rotation_world.T @ velocity[:3]).astype(np.float32)

    def _anchor_gravity_direction(self) -> np.ndarray:
        anchor_quat = self.data.xquat[self.anchor_body_id]
        world_down = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        return (quaternion_to_matrix(anchor_quat).T @ world_down).astype(np.float32)

    def _current_state(self) -> dict[str, np.ndarray]:
        """读取与训练端 PolicyCfg 同顺序、同语义的当前 proprioception term。"""

        q_policy, dq_policy = self._joint_state_policy()
        return {
            "base_ang_vel": self._base_angular_velocity_local(),
            "joint_pos": (q_policy - self.contract.default_policy).astype(np.float32),
            "joint_vel": dq_policy.astype(np.float32),
            "actions": self.last_action_policy.copy(),
            "gravity_dir": self._anchor_gravity_direction(),
        }

    def _append_current_state(self) -> None:
        current_state = self._current_state()
        for name, value in current_state.items():
            self.histories[name].append(value)

    def _build_proprioception(self) -> np.ndarray:
        # PolicyCfg 的定义顺序固定为 base_ang_vel, joint_pos, joint_vel, actions, gravity_dir。
        value = np.concatenate(
            [
                np.stack(self.histories[name], axis=0).reshape(-1)
                for name in (
                    "base_ang_vel",
                    "joint_pos",
                    "joint_vel",
                    "actions",
                    "gravity_dir",
                )
            ]
        ).astype(np.float32)
        if value.size != self.contract.actor_proprioception_dim:
            raise ValueError(f"proprioception 维度错误: {value.size}")
        return value

    def _build_robot_tokenizer(self) -> np.ndarray:
        indices = self.motion.future_indices(
            self.motion_frame,
            self.contract.num_future_frames,
            self.contract.future_frame_stride,
            self.loop_motion,
        )
        joint_pos = self.motion.joint_pos_policy[indices]
        joint_vel = self.motion.joint_vel_policy[indices]
        anchor_quat = self.data.xquat[self.anchor_body_id]
        anchor_inverse = quaternion_conjugate(anchor_quat)
        reference_quat = quaternion_multiply(
            np.broadcast_to(
                self.reference_heading_delta_wxyz,
                self.reference_anchor_quat_wxyz[indices].shape,
            ),
            self.reference_anchor_quat_wxyz[indices],
        )
        relative_quat = quaternion_multiply(
            np.broadcast_to(anchor_inverse, reference_quat.shape),
            reference_quat,
        )
        orientation_6d = quaternion_to_rotation_6d(relative_quat).astype(np.float32)

        # 训练端先 cat(pos_flat, vel_flat)，再 reshape 为 [future, -1]；这里必须保留该布局。
        value = np.concatenate(
            (joint_pos.reshape(-1), joint_vel.reshape(-1), orientation_6d.reshape(-1))
        ).astype(np.float32)
        if value.size != self.contract.robot_tokenizer_dim:
            raise ValueError(f"Robot Encoder tokenizer 维度错误: {value.size}")
        return value

    def build_observation(self) -> np.ndarray:
        """构造联合 ``*_g1.onnx`` 所需的 480+690=1170 维输入。"""

        self._append_current_state()
        value = np.concatenate((self._build_robot_tokenizer(), self._build_proprioception()))
        if value.size != self.contract.combined_policy_input_dim:
            raise ValueError(f"联合 policy 输入维度错误: {value.size}")
        if not np.isfinite(value).all():
            raise FloatingPointError("policy observation 含 NaN/Inf")
        self.last_observation = value.astype(np.float32, copy=False)
        return self.last_observation

    def infer_action(self) -> np.ndarray:
        action = np.asarray(self.policy(self.build_observation()), dtype=np.float32).reshape(-1)
        if action.size != self.contract.action_dim:
            raise ValueError(f"policy action 维度错误: {action.size}")
        if not np.isfinite(action).all():
            raise FloatingPointError("policy action 含 NaN/Inf")
        self.last_action_policy = np.clip(
            action, -self.contract.action_clip, self.contract.action_clip
        )
        return self.last_action_policy

    def _apply_pd_control(self) -> None:
        target_policy = (
            self.contract.default_policy
            + self.contract.action_scale_policy * self.last_action_policy
        )
        target_mujoco = target_policy[self.contract.policy_to_mujoco]
        q_mujoco = self.data.qpos[self.qpos_addresses]
        dq_mujoco = self.data.qvel[self.dof_addresses]
        torque = self.contract.stiffness_mujoco * (target_mujoco - q_mujoco)
        torque -= self.contract.damping_mujoco * dq_mujoco
        torque = np.clip(
            torque, -self.contract.effort_mujoco, self.contract.effort_mujoco
        )
        if not np.isfinite(torque).all():
            raise FloatingPointError("PD torque 含 NaN/Inf")
        self.data.ctrl[self.actuator_ids] = torque
        self.last_torque_mujoco = torque

    def step_control(self) -> np.ndarray:
        """执行一次 50 Hz policy inference 和 4 次 200 Hz MuJoCo step。"""

        action = self.infer_action()
        for _ in range(self.contract.decimation):
            self._apply_pd_control()
            mujoco.mj_step(self.model, self.data)
            for label, value in (
                ("qpos", self.data.qpos),
                ("qvel", self.data.qvel),
                ("ctrl", self.data.ctrl),
            ):
                if not np.isfinite(value).all():
                    raise FloatingPointError(f"MuJoCo {label} 含 NaN/Inf")
        self.motion_frame += 1
        if self.loop_motion:
            self.motion_frame %= self.motion.num_frames
        else:
            self.motion_frame = min(self.motion_frame, self.motion.num_frames - 1)
        return action

    def run(
        self,
        control_steps: int,
        *,
        headless: bool = True,
        real_time: bool = False,
        show_reference: bool = True,
        reference_alpha: float = 0.32,
    ) -> dict[str, float]:
        """运行控制闭环，policy 由 XML 动力学模型直接渲染。"""

        if control_steps <= 0:
            raise ValueError("control_steps 必须大于零")
        if not 0.0 < reference_alpha <= 1.0:
            raise ValueError("reference_alpha 必须位于 (0, 1]")
        viewer = None
        if not headless:
            from mujoco import viewer as mujoco_viewer

            viewer = mujoco_viewer.launch_passive(self.model, self.data)
            if show_reference:
                if not hasattr(viewer, "user_scn"):
                    raise RuntimeError(
                        "当前 MuJoCo passive viewer 不支持 user_scn 参考影子"
                    )
                with viewer.lock():
                    self._write_reference_markers_to_scene(
                        viewer.user_scn,
                        self.motion_frame,
                        alpha=reference_alpha,
                    )
                viewer.sync()
        start_reference_diagnostics = self.reference_pose_diagnostics(self.motion_frame)
        started = time.monotonic()
        try:
            for _ in range(control_steps):
                tick = time.monotonic()
                self.step_control()
                if viewer is not None:
                    if not viewer.is_running():
                        break
                    if show_reference:
                        with viewer.lock():
                            self._write_reference_markers_to_scene(
                                viewer.user_scn,
                                self.motion_frame,
                                alpha=reference_alpha,
                            )
                    viewer.sync()
                if real_time:
                    remaining = self.contract.control_dt - (time.monotonic() - tick)
                    if remaining > 0.0:
                        time.sleep(remaining)
        finally:
            if viewer is not None:
                viewer.close()
        elapsed = time.monotonic() - started
        final_reference_diagnostics = self.reference_pose_diagnostics(self.motion_frame)
        return {
            "requested_control_steps": float(control_steps),
            "elapsed_seconds": elapsed,
            "control_steps_per_second": control_steps / max(elapsed, 1e-12),
            "simulation_time": float(self.data.time),
            "root_height": float(self.data.qpos[self.root_qpos_address + 2]),
            "max_abs_torque": float(np.max(np.abs(self.last_torque_mujoco))),
            "max_abs_action": float(np.max(np.abs(self.last_action_policy))),
            "reference_start_base_tilt_degrees": start_reference_diagnostics[
                "base_tilt_degrees"
            ],
            "reference_start_anchor_tilt_degrees": start_reference_diagnostics[
                "anchor_tilt_degrees"
            ],
            "reference_final_base_tilt_degrees": final_reference_diagnostics[
                "base_tilt_degrees"
            ],
            "reference_final_anchor_tilt_degrees": final_reference_diagnostics[
                "anchor_tilt_degrees"
            ],
        }


def make_static_reference_motion(
    contract: Bumi3Contract,
    *,
    num_frames: int = 64,
) -> ReferenceMotion:
    """为无数据 smoke 构造默认姿态参考；不写文件，也不作为训练数据。"""

    if num_frames < 2:
        raise ValueError("num_frames 必须至少为 2")
    return ReferenceMotion(
        joint_pos_policy=np.repeat(
            contract.default_policy[None].astype(np.float32), num_frames, axis=0
        ),
        joint_vel_policy=np.zeros((num_frames, contract.action_dim), dtype=np.float32),
        root_position_world=np.repeat(
            contract.initial_root_position[None], num_frames, axis=0
        ),
        root_quat_wxyz=np.repeat(
            contract.initial_root_quaternion_wxyz[None], num_frames, axis=0
        ),
        fps=contract.target_fps,
        name="static_validation_only",
    )
