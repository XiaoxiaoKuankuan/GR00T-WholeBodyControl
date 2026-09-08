#!/usr/bin/env python3
"""
将成对的 G1 Robot PKL 与 SMPL PKL 转换为 SONIC C++ 部署可直接读取的 CSV 目录。

这个工具面向训练数据的“原始双文件契约”，而不是
``gear_sonic_deploy/reference/convert_motions.py`` 所处理的已完成 FK 的单一 PKL：

1. G1 Robot PKL 保存 30 Hz 的 ``pose_aa``、根平移和 MuJoCo 顺序关节数据；脚本调用
   训练时同一个 ``Humanoid_Batch.fk_batch``，按训练规则插值到 50 Hz，并转换为
   IsaacLab 的 29 自由度顺序及 14 个跟踪刚体顺序。
2. SMPL PKL 保存 50 Hz 的 Y-up ``pose_aa`` 和 ``smpl_joints``；脚本复现训练端
   ``smpl_root_ytoz_up -> remove_smpl_base_rot -> inverse-root quat_apply`` 的观测链，
   生成 SMPL 编码器实际接收的 24x3 逐帧根坐标关节。
3. 每对输入分别输出 ``g1/<motion>``（编码器 mode 0）与 ``smpl/<motion>``
   （编码器 mode 2）。SMPL 模式仍携带配对 G1 的 29 关节数据，因为当前模型的
   SMPL 模式明确需要六个 G1 手腕关节作为条件；这不是可省略的冗余字段。
4. 脚本在写文件前完成字段、维度、有限值、FPS、四元数、帧数和配对关系校验；
   输出目录已存在时直接拒绝，避免覆盖用户已有的转换结果。最终 ``manifest.json``
   记录源文件与输出文件 SHA256、帧数、持续时间、坐标约定和根朝向配对误差。

典型调用（在仓库根目录）：

    /home/weili/miniconda3/envs/env_isaaclab/bin/python \
      tools_local/convert_paired_g1_smpl_pkl_to_deploy.py \
      --robot-root data/noetix12_g1_smpl_10pairs_20260908/raw/g1 \
      --smpl-root data/noetix12_g1_smpl_10pairs_20260908/raw/smpl \
      --output-root data/noetix12_g1_smpl_10pairs_20260908/deploy

原始 PKL 只读，脚本不会修改、移动或删除任何输入文件。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from omegaconf import OmegaConf
import torch

from gear_sonic.isaac_utils.rotations import (
    remove_smpl_base_rot,
    smpl_root_ytoz_up,
    wxyz_to_xyzw,
)
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    quat_apply,
    quat_inv,
)
from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MJCF = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"

# 以下映射来自 gear_sonic/envs/manager_env/robots/g1.py。这里显式保存常量，避免转换工具
# 为了读取纯数据而导入 IsaacLab 仿真模块；启动时仍会校验它们是完整、不重复的排列。
G1_MUJOCO_TO_ISAACLAB_DOF = [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
]
G1_MUJOCO_TO_ISAACLAB_BODY = [
    0,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    6,
    12,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
    22,
    29,
]

# motion.yaml 中 14 个跟踪刚体在完整 IsaacLab 刚体顺序中的索引。
TRACKED_BODY_INDEXES = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]
TRACKED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

ROBOT_REQUIRED_FIELDS = {
    "root_trans_offset",
    "pose_aa",
    "dof",
    "root_rot",
    "fps",
}
SMPL_REQUIRED_FIELDS = {"pose_aa", "transl", "smpl_joints", "fps"}


@dataclass
class SourcePair:
    """一条同名 Robot/SMPL 轨迹的只读源文件定位。"""

    name: str
    robot_path: Path
    smpl_path: Path


@dataclass
class ConvertedPair:
    """完成全部内存校验、尚未写盘的一对部署数组与审计信息。"""

    source: SourcePair
    g1_arrays: dict[str, np.ndarray]
    smpl_arrays: dict[str, np.ndarray]
    audit: dict[str, Any]


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA256，避免因大文件一次性读入造成额外内存峰值。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_array(
    value: Any,
    *,
    field: str,
    ndim: int,
    tail_shape: tuple[int, ...],
) -> np.ndarray:
    """将字段规范为 float32，并严格检查维度、尾部形状和有限值。"""

    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim or tuple(array.shape[-len(tail_shape) :]) != tail_shape:
        raise ValueError(f"字段 {field} 形状错误：得到 {array.shape}，要求 (*, {', '.join(map(str, tail_shape))})")
    if array.shape[0] < 2:
        raise ValueError(f"字段 {field} 至少需要 2 帧，实际为 {array.shape[0]}")
    if not np.isfinite(array).all():
        raise ValueError(f"字段 {field} 含 NaN 或 Inf")
    return np.ascontiguousarray(array)


def index_pkls(root: Path, label: str) -> dict[str, Path]:
    """递归按文件 stem 建索引；同名文件会立即报错，防止错误配对。"""

    if not root.is_dir():
        raise FileNotFoundError(f"{label} 根目录不存在：{root}")
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*.pkl")):
        if path.stem in {"metadata", "shape_metadata"}:
            continue
        if path.stem in result:
            raise ValueError(f"{label} 中存在重复轨迹名 {path.stem}：{result[path.stem]} 与 {path}")
        result[path.stem] = path
    if not result:
        raise ValueError(f"{label} 根目录下没有可转换的 PKL：{root}")
    return result


def pair_sources(robot_root: Path, smpl_root: Path) -> list[SourcePair]:
    """要求两个目录的运动名称集合完全一致，并生成稳定排序的配对列表。"""

    robot_files = index_pkls(robot_root, "Robot")
    smpl_files = index_pkls(smpl_root, "SMPL")
    robot_only = sorted(set(robot_files) - set(smpl_files))
    smpl_only = sorted(set(smpl_files) - set(robot_files))
    if robot_only or smpl_only:
        raise ValueError(f"Robot/SMPL 名称集合不一致：仅 Robot={robot_only[:10]}，仅 SMPL={smpl_only[:10]}")
    return [SourcePair(name, robot_files[name], smpl_files[name]) for name in sorted(robot_files)]


def load_robot_entry(pair: SourcePair) -> dict[str, Any]:
    """读取单运动 Robot PKL，并验证外层字典键与文件名一致。"""

    container = joblib.load(pair.robot_path)
    if not isinstance(container, dict) or len(container) != 1:
        raise ValueError(f"Robot PKL 必须是只含一条运动的字典：{pair.robot_path}")
    motion_name, entry = next(iter(container.items()))
    if motion_name != pair.name:
        raise ValueError(f"Robot PKL 内外名称不一致：文件={pair.name}，字典键={motion_name}")
    if not isinstance(entry, dict):
        raise TypeError(f"Robot 运动条目不是字典：{pair.robot_path}")
    missing = sorted(ROBOT_REQUIRED_FIELDS - set(entry))
    if missing:
        raise KeyError(f"Robot PKL 缺字段 {missing}：{pair.robot_path}")
    return entry


def load_smpl_entry(pair: SourcePair) -> dict[str, Any]:
    """读取 SMPL PKL，并校验部署转换所需字段。"""

    entry = joblib.load(pair.smpl_path)
    if not isinstance(entry, dict):
        raise TypeError(f"SMPL PKL 不是字典：{pair.smpl_path}")
    missing = sorted(SMPL_REQUIRED_FIELDS - set(entry))
    if missing:
        raise KeyError(f"SMPL PKL 缺字段 {missing}：{pair.smpl_path}")
    return entry


def build_humanoid(mjcf_path: Path) -> Humanoid_Batch:
    """使用训练 MotionLib 的同一 FK 实现与同一 G1 MJCF 创建 CPU 解析器。"""

    if not mjcf_path.is_file():
        raise FileNotFoundError(f"G1 MJCF 不存在：{mjcf_path}")
    cfg = OmegaConf.create(
        {
            "asset": {
                "assetRoot": str(mjcf_path.parent.resolve()),
                "assetFileName": mjcf_path.name,
            },
            "extend_config": [],
        }
    )
    return Humanoid_Batch(cfg, device=torch.device("cpu"))


def validate_mappings() -> None:
    """验证硬编码映射是完整排列，避免静默重复或漏掉关节/刚体。"""

    if sorted(G1_MUJOCO_TO_ISAACLAB_DOF) != list(range(29)):
        raise RuntimeError("G1 自由度映射不是 0..28 的完整排列")
    if sorted(G1_MUJOCO_TO_ISAACLAB_BODY) != list(range(30)):
        raise RuntimeError("G1 刚体映射不是 0..29 的完整排列")
    if len(TRACKED_BODY_INDEXES) != len(TRACKED_BODY_NAMES):
        raise RuntimeError("14 个跟踪刚体的索引与名称数量不一致")


def convert_smpl_observation(smpl_pose: np.ndarray, smpl_joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """复现训练 mode 2 的根朝向与逐帧根坐标 SMPL 关节计算。"""

    pose_tensor = torch.from_numpy(smpl_pose)
    joints_tensor = torch.from_numpy(smpl_joints)
    root_quat_wxyz = angle_axis_to_quaternion(pose_tensor[:, :3])
    root_quat_wxyz = smpl_root_ytoz_up(root_quat_wxyz)
    root_quat_wxyz = remove_smpl_base_rot(root_quat_wxyz, w_last=False)

    root_quat_inv = quat_inv(root_quat_wxyz).unsqueeze(1).expand(-1, 24, -1)
    joints_local = quat_apply(root_quat_inv.reshape(-1, 4), joints_tensor.reshape(-1, 3)).reshape(-1, 24, 3)
    return (
        np.ascontiguousarray(root_quat_wxyz.numpy().astype(np.float32)),
        np.ascontiguousarray(joints_local.numpy().astype(np.float32)),
    )


def quaternion_alignment_degrees(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """以 q/-q 等价方式统计两条 wxyz 根四元数序列的夹角误差。"""

    dots = np.abs(np.sum(first * second, axis=-1))
    angles = np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))
    return {
        "mean_deg": float(np.mean(angles)),
        "p95_deg": float(np.percentile(angles, 95)),
        "max_deg": float(np.max(angles)),
    }


@torch.no_grad()
def convert_pair(
    pair: SourcePair,
    humanoid: Humanoid_Batch,
    target_fps: int,
) -> ConvertedPair:
    """转换一对输入；任何契约不一致都会在写盘前终止整个批次。"""

    robot = load_robot_entry(pair)
    smpl = load_smpl_entry(pair)

    robot_pose = require_array(robot["pose_aa"], field=f"{pair.name}.robot.pose_aa", ndim=3, tail_shape=(30, 3))
    robot_trans = require_array(
        robot["root_trans_offset"],
        field=f"{pair.name}.robot.root_trans_offset",
        ndim=2,
        tail_shape=(3,),
    )
    robot_dof = require_array(robot["dof"], field=f"{pair.name}.robot.dof", ndim=2, tail_shape=(29,))
    robot_root_xyzw = require_array(
        robot["root_rot"], field=f"{pair.name}.robot.root_rot", ndim=2, tail_shape=(4,)
    )
    robot_frames = robot_pose.shape[0]
    if any(array.shape[0] != robot_frames for array in (robot_trans, robot_dof, robot_root_xyzw)):
        raise ValueError(f"{pair.name} 的 Robot 字段帧数不一致")
    robot_fps = float(robot["fps"])
    if not np.isfinite(robot_fps) or robot_fps <= 0:
        raise ValueError(f"{pair.name} 的 Robot fps 非法：{robot_fps}")

    # 原始 dof 与 pose_aa 是同一信息的两种表示；先核对它们，避免把错误关节轴送入 FK。
    raw_dof_error = float(np.max(np.abs(robot_pose[:, 1:, :].sum(axis=-1) - robot_dof)))
    if raw_dof_error > 1e-5:
        raise ValueError(f"{pair.name} 的 pose_aa 与 dof 不一致，最大误差 {raw_dof_error}")
    computed_root_xyzw = wxyz_to_xyzw(angle_axis_to_quaternion(torch.from_numpy(robot_pose[:, 0]))).numpy()
    root_error = np.minimum(
        np.max(np.abs(computed_root_xyzw - robot_root_xyzw), axis=-1),
        np.max(np.abs(computed_root_xyzw + robot_root_xyzw), axis=-1),
    )
    raw_root_quat_error = float(np.max(root_error))
    if raw_root_quat_error > 1e-5:
        raise ValueError(f"{pair.name} 的 pose_aa 根旋转与 root_rot 不一致，最大 q/-q 误差 {raw_root_quat_error}")

    smpl_pose = require_array(smpl["pose_aa"], field=f"{pair.name}.smpl.pose_aa", ndim=2, tail_shape=(72,))
    smpl_transl = require_array(smpl["transl"], field=f"{pair.name}.smpl.transl", ndim=2, tail_shape=(3,))
    smpl_joints = require_array(
        smpl["smpl_joints"],
        field=f"{pair.name}.smpl.smpl_joints",
        ndim=3,
        tail_shape=(24, 3),
    )
    smpl_frames = smpl_pose.shape[0]
    if smpl_transl.shape[0] != smpl_frames or smpl_joints.shape[0] != smpl_frames:
        raise ValueError(f"{pair.name} 的 SMPL 字段帧数不一致")
    smpl_fps = float(smpl["fps"])
    if abs(smpl_fps - target_fps) > 1e-6:
        raise ValueError(
            f"{pair.name} 的 SMPL fps={smpl_fps}，当前工具要求它已是目标 {target_fps} Hz；"
            "拒绝使用与训练端不一致的隐式重采样"
        )

    fk = humanoid.fk_batch(
        torch.from_numpy(robot_pose).unsqueeze(0),
        torch.from_numpy(robot_trans).unsqueeze(0),
        return_full=True,
        fps=robot_fps,
        target_fps=target_fps,
        interpolate_data=True,
        use_parallel_fk=True,
    )
    output_frames = int(fk.dof_pos.shape[1])
    if output_frames != smpl_frames:
        raise ValueError(f"{pair.name} 配对帧数不一致：G1 插值后 {output_frames}，SMPL {smpl_frames}")

    joint_pos = fk.dof_pos[0].numpy()[:, G1_MUJOCO_TO_ISAACLAB_DOF]
    joint_vel = fk.dof_vels[0].numpy()[:, G1_MUJOCO_TO_ISAACLAB_DOF]
    body_pos_full = fk.global_translation[0].numpy()[:, G1_MUJOCO_TO_ISAACLAB_BODY]
    body_quat_full_xyzw = fk.global_rotation[0].numpy()[:, G1_MUJOCO_TO_ISAACLAB_BODY]
    body_lin_vel_full = fk.global_velocity[0].numpy()[:, G1_MUJOCO_TO_ISAACLAB_BODY]
    body_ang_vel_full = fk.global_angular_velocity[0].numpy()[:, G1_MUJOCO_TO_ISAACLAB_BODY]

    body_pos = body_pos_full[:, TRACKED_BODY_INDEXES]
    body_quat = body_quat_full_xyzw[:, TRACKED_BODY_INDEXES][..., [3, 0, 1, 2]]
    body_lin_vel = body_lin_vel_full[:, TRACKED_BODY_INDEXES]
    body_ang_vel = body_ang_vel_full[:, TRACKED_BODY_INDEXES]
    smpl_root_quat, smpl_joints_local = convert_smpl_observation(smpl_pose, smpl_joints)
    smpl_root_xyzw = torch.from_numpy(smpl_root_quat[:, [1, 2, 3, 0]])
    smpl_root_ang_vel = humanoid._compute_angular_velocity(
        smpl_root_xyzw.unsqueeze(0).unsqueeze(2), 1.0 / target_fps
    )[0].numpy()

    arrays_to_check = [
        joint_pos,
        joint_vel,
        body_pos,
        body_quat,
        body_lin_vel,
        body_ang_vel,
        smpl_root_quat,
        smpl_root_ang_vel,
        smpl_joints_local,
    ]
    if not all(np.isfinite(array).all() for array in arrays_to_check):
        raise ValueError(f"{pair.name} 转换结果含 NaN 或 Inf")
    robot_quat_norm_error = float(np.max(np.abs(np.linalg.norm(body_quat, axis=-1) - 1.0)))
    smpl_quat_norm_error = float(np.max(np.abs(np.linalg.norm(smpl_root_quat, axis=-1) - 1.0)))
    if max(robot_quat_norm_error, smpl_quat_norm_error) > 1e-4:
        raise ValueError(f"{pair.name} 四元数未归一化：G1={robot_quat_norm_error}，SMPL={smpl_quat_norm_error}")

    # SMPL mode 只保留一个根刚体；根位置借用配对 G1 pelvis，根朝向则必须来自 SMPL。
    # 该组合既满足 C++ anchor 接口，也不会把其余 G1 刚体误称为 SMPL 刚体。
    g1_arrays = {
        "joint_pos": np.ascontiguousarray(joint_pos, dtype=np.float32),
        "joint_vel": np.ascontiguousarray(joint_vel, dtype=np.float32),
        "body_pos": np.ascontiguousarray(body_pos, dtype=np.float32),
        "body_quat": np.ascontiguousarray(body_quat, dtype=np.float32),
        "body_lin_vel": np.ascontiguousarray(body_lin_vel, dtype=np.float32),
        "body_ang_vel": np.ascontiguousarray(body_ang_vel, dtype=np.float32),
    }
    smpl_arrays = {
        "joint_pos": g1_arrays["joint_pos"],
        "joint_vel": g1_arrays["joint_vel"],
        "body_pos": np.ascontiguousarray(body_pos[:, :1], dtype=np.float32),
        "body_quat": np.ascontiguousarray(smpl_root_quat[:, None, :], dtype=np.float32),
        "body_lin_vel": np.ascontiguousarray(body_lin_vel[:, :1], dtype=np.float32),
        "body_ang_vel": np.ascontiguousarray(smpl_root_ang_vel, dtype=np.float32),
        "smpl_joint": np.ascontiguousarray(smpl_joints_local, dtype=np.float32),
        # 当前 release mode 2 不消费 pose，但 C++ 静态/流式协议以 21x3 为标准字段；
        # 因此保留根之后的前 21 个 body pose，排除末尾两个手部姿态。
        "smpl_pose": np.ascontiguousarray(smpl_pose[:, 3:66].reshape(-1, 21, 3)),
    }

    audit = {
        "robot_source_frames": robot_frames,
        "robot_source_fps": robot_fps,
        "robot_source_last_timestamp_s": (robot_frames - 1) / robot_fps,
        "smpl_source_frames": smpl_frames,
        "smpl_source_fps": smpl_fps,
        "smpl_source_last_timestamp_s": (smpl_frames - 1) / smpl_fps,
        "deploy_frames": output_frames,
        "deploy_fps": target_fps,
        "deploy_last_timestamp_s": (output_frames - 1) / target_fps,
        "source_duration_difference_s": abs((robot_frames - 1) / robot_fps - (smpl_frames - 1) / smpl_fps),
        "raw_pose_dof_max_abs_error": raw_dof_error,
        "raw_root_quaternion_sign_invariant_max_abs_error": raw_root_quat_error,
        "g1_quaternion_norm_max_abs_error": robot_quat_norm_error,
        "smpl_quaternion_norm_max_abs_error": smpl_quat_norm_error,
        "paired_root_orientation_error": quaternion_alignment_degrees(body_quat[:, 0], smpl_root_quat),
    }
    return ConvertedPair(pair, g1_arrays, smpl_arrays, audit)


def csv_headers(name: str, array: np.ndarray) -> list[str]:
    """按 C++ reader 的字段命名习惯生成二维或三维数组 CSV 表头。"""

    if name == "joint_pos":
        return [f"joint_{index}" for index in range(array.shape[1])]
    if name == "joint_vel":
        return [f"joint_vel_{index}" for index in range(array.shape[1])]
    if name in {"body_pos", "body_lin_vel", "body_ang_vel"}:
        suffix = {
            "body_pos": "",
            "body_lin_vel": "vel_",
            "body_ang_vel": "angvel_",
        }[name]
        return [f"body_{body}_{suffix}{axis}" for body in range(array.shape[1]) for axis in "xyz"]
    if name == "body_quat":
        return [f"body_{body}_{axis}" for body in range(array.shape[1]) for axis in "wxyz"]
    if name in {"smpl_joint", "smpl_pose"}:
        return [f"{name}_{index}_{axis}" for index in range(array.shape[1]) for axis in "xyz"]
    raise KeyError(f"没有为 {name} 定义 CSV 表头")


def write_csv(path: Path, name: str, array: np.ndarray) -> None:
    """使用九位小数写入带表头 CSV，保留 float32 数据所需精度。"""

    flat = array.reshape(array.shape[0], -1)
    np.savetxt(
        path,
        flat,
        delimiter=",",
        header=",".join(csv_headers(name, array)),
        comments="",
        fmt="%.9f",
    )


def write_metadata(path: Path, motion_name: str, body_indexes: list[int], frames: int) -> None:
    """写入 MotionDataReader 可解析的刚体索引与帧数元数据。"""

    indexes = " ".join(str(index) for index in body_indexes)
    path.write_text(
        f"Metadata for: {motion_name}\n"
        "==============================\n\n"
        "Body part indexes:\n"
        f"[{indexes}]\n\n"
        f"Total timesteps: {frames}\n",
        encoding="utf-8",
    )


def write_info(
    path: Path,
    *,
    motion_name: str,
    encoder_mode: int,
    arrays: dict[str, np.ndarray],
) -> None:
    """写入便于人工检查、但不参与 C++ 加载的中文说明。"""

    lines = [
        f"轨迹名称：{motion_name}",
        f"目标编码器模式：{encoder_mode}",
        "目标频率：50 Hz",
        "",
        "数组摘要：",
    ]
    for name, array in arrays.items():
        lines.append(
            f"- {name}: shape={list(array.shape)}, dtype={array.dtype}, "
            f"range=[{float(array.min()):.6f}, {float(array.max()):.6f}]"
        )
    if encoder_mode == 2:
        lines.extend(
            [
                "",
                "SMPL 说明：smpl_joint 是与训练 mode 2 相同的逐帧根坐标表示；",
                "body_quat 的唯一刚体为处理后的 SMPL 根朝向；body_pos 使用配对 G1 pelvis；",
                "joint_pos/joint_vel 仅为当前 SMPL 编码器所需的配对 G1 手腕条件提供来源。",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_motion_directory(
    output_dir: Path,
    converted: ConvertedPair,
    *,
    mode_name: str,
) -> dict[str, dict[str, Any]]:
    """写一条 mode 0 或 mode 2 轨迹，并返回逐文件哈希与形状。"""

    encoder_mode = 0 if mode_name == "g1" else 2
    arrays = converted.g1_arrays if mode_name == "g1" else converted.smpl_arrays
    body_indexes = TRACKED_BODY_INDEXES if mode_name == "g1" else [0]
    motion_dir = output_dir / mode_name / converted.source.name
    motion_dir.mkdir(parents=True, exist_ok=False)

    records: dict[str, dict[str, Any]] = {}
    for name, array in arrays.items():
        csv_path = motion_dir / f"{name}.csv"
        write_csv(csv_path, name, array)
        records[csv_path.name] = {
            "shape": list(array.shape),
            "sha256": sha256_file(csv_path),
        }
    metadata_path = motion_dir / "metadata.txt"
    write_metadata(metadata_path, converted.source.name, body_indexes, converted.audit["deploy_frames"])
    records[metadata_path.name] = {"sha256": sha256_file(metadata_path)}
    info_path = motion_dir / "info.txt"
    write_info(
        info_path,
        motion_name=converted.source.name,
        encoder_mode=encoder_mode,
        arrays=arrays,
    )
    records[info_path.name] = {"sha256": sha256_file(info_path)}
    return records


def relative_path(path: Path, root: Path) -> str:
    """优先记录相对输入根的路径；异常布局下退回绝对路径。"""

    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def write_dataset(
    output_root: Path,
    converted_pairs: list[ConvertedPair],
    *,
    robot_root: Path,
    smpl_root: Path,
    mjcf_path: Path,
    source_host: str,
    remote_robot_root: str,
    remote_smpl_root: str,
) -> Path:
    """写完整部署数据集与可追溯 manifest；拒绝覆盖任何既有目录。"""

    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，出于数据保护不覆盖：{output_root}")
    output_root.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": "sonic_paired_g1_smpl_deploy_v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": str(Path(__file__).resolve()),
        "source_host": source_host,
        "source_roots": {
            "remote_robot": remote_robot_root,
            "remote_smpl": remote_smpl_root,
            "local_robot": str(robot_root.resolve()),
            "local_smpl": str(smpl_root.resolve()),
        },
        "mjcf": {
            "path": str(mjcf_path.resolve()),
            "sha256": sha256_file(mjcf_path),
        },
        "contracts": {
            "g1": "50 Hz；29 自由度 IsaacLab 顺序；14 刚体世界坐标；四元数 wxyz；encoder mode 0",
            "smpl": (
                "50 Hz；24x3 训练同构逐帧根坐标关节；处理后 SMPL 根四元数 wxyz；"
                "配对 G1 pelvis 平移与手腕关节条件；SMPL 根角速度；encoder mode 2"
            ),
            "smpl_preprocess": (
                "raw Y-up root axis-angle -> quaternion(wxyz) -> +90deg X Y-up-to-Z-up -> "
                "remove [0.5,0.5,0.5,0.5] base rotation -> inverse-root applied to raw smpl_joints"
            ),
        },
        "pair_count": len(converted_pairs),
        "pairs": [],
    }

    for converted in converted_pairs:
        g1_files = write_motion_directory(output_root, converted, mode_name="g1")
        smpl_files = write_motion_directory(output_root, converted, mode_name="smpl")
        manifest["pairs"].append(
            {
                "name": converted.source.name,
                "sources": {
                    "robot": {
                        "path": relative_path(converted.source.robot_path, robot_root),
                        "sha256": sha256_file(converted.source.robot_path),
                    },
                    "smpl": {
                        "path": relative_path(converted.source.smpl_path, smpl_root),
                        "sha256": sha256_file(converted.source.smpl_path),
                    },
                },
                "audit": converted.audit,
                "outputs": {"g1": g1_files, "smpl": smpl_files},
            }
        )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    """解析只读输入、全新输出和来源审计参数。"""

    parser = argparse.ArgumentParser(description="将成对 G1 Robot/SMPL PKL 转换为 SONIC mode 0/mode 2 部署 CSV。")
    parser.add_argument("--robot-root", type=Path, required=True, help="Robot PKL 根目录，可递归")
    parser.add_argument("--smpl-root", type=Path, required=True, help="SMPL PKL 根目录，可递归")
    parser.add_argument("--output-root", type=Path, required=True, help="必须尚不存在的输出目录")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="G1 29-DoF MJCF")
    parser.add_argument("--target-fps", type=int, default=50, help="部署频率，当前模型应为 50")
    parser.add_argument("--source-host", default="noetix-12", help="manifest 中记录的源服务器")
    parser.add_argument(
        "--remote-robot-root",
        default="/data/datasets/bones-seed/sonic/motion_lib_bones_seed/robot_filtered",
        help="manifest 中记录的远端 Robot 根目录",
    )
    parser.add_argument(
        "--remote-smpl-root",
        default="/data/datasets/bones-seed/sonic/training_assets/data/smpl_filtered",
        help="manifest 中记录的远端 SMPL 根目录",
    )
    return parser.parse_args()


def main() -> None:
    """执行完整批次：先全部转换与校验，再创建唯一输出目录。"""

    args = parse_args()
    if args.target_fps != 50:
        raise ValueError(f"当前 SONIC 部署与输出说明固定为 50 Hz，不接受 target-fps={args.target_fps}")
    if args.output_root.exists():
        raise FileExistsError(f"输出目录已存在，出于数据保护不覆盖：{args.output_root}")
    validate_mappings()
    pairs = pair_sources(args.robot_root, args.smpl_root)
    print(f"发现 {len(pairs)} 对同名 Robot/SMPL 轨迹，开始内存转换与严格校验。")
    humanoid = build_humanoid(args.mjcf)
    converted_pairs: list[ConvertedPair] = []
    for index, pair in enumerate(pairs, start=1):
        converted = convert_pair(pair, humanoid, args.target_fps)
        converted_pairs.append(converted)
        root_error = converted.audit["paired_root_orientation_error"]
        print(
            f"[{index:02d}/{len(pairs):02d}] {pair.name}: "
            f"{converted.audit['robot_source_frames']}@{converted.audit['robot_source_fps']:g}Hz -> "
            f"{converted.audit['deploy_frames']}@{args.target_fps}Hz；"
            f"SMPL/G1 根朝向 mean={root_error['mean_deg']:.3f}°，max={root_error['max_deg']:.3f}°"
        )

    manifest_path = write_dataset(
        args.output_root,
        converted_pairs,
        robot_root=args.robot_root,
        smpl_root=args.smpl_root,
        mjcf_path=args.mjcf,
        source_host=args.source_host,
        remote_robot_root=args.remote_robot_root,
        remote_smpl_root=args.remote_smpl_root,
    )
    print(f"转换完成：{len(converted_pairs)} 对，G1 与 SMPL 共 {len(converted_pairs) * 2} 条部署轨迹。")
    print(f"审计清单：{manifest_path.resolve()}")


if __name__ == "__main__":
    main()
