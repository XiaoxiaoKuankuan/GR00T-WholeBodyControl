#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""为 SONIC 构建 BUMI3 质量报告白名单限定的严格 50 Hz 数据集。

本工具只接受质量报告中 ``status=PASS`` 且 ``quality_accepted=true`` 的样本，默认
要求白名单恰好为 2790 条。输入是 ``robot_retargeter.bumi3_mimic_npz_30hz.v1``
契约的 Z-up BUMI3 NPZ：21 个关节位置使用文件内 ``joint_names``，22 个 body 世界
位置/四元数使用 ``body_names``，四元数必须是 wxyz。工具不会把 REVIEW/REJECT、
缺文件、哈希不匹配或名称重复的样本带入训练目录。

30 Hz 到 50 Hz 使用独占末帧的严格时间网格 ``arange(0, (T-1)/30, 1/50)``：根位置、
body 位置和关节位置做线性插值；全部 body 四元数做最短弧 SLERP。随后从 50 Hz 位置
重新计算关节速度、加速度、jerk、body 线速度/角速度，并依据左右 ankle-roll body 的
相对低位、水平/垂直速度和滞回阈值重新检测接触。导出的 SONIC joblib 只保留训练实际
读取的 root/pose/DoF/quaternion 字段；重新计算的动态量与接触写入独立 audit NPZ，避免
把部署无关字段混入 MotionLib 条目。每个 audit NPZ 和 JSONL 报告都可追溯到源哈希。

为避免重复原始数据，PASS NPZ 在同一文件系统内通过硬链接进入 ``source_npz_pass``；
若跨文件系统无法硬链接则直接失败，绝不静默复制。构建先写 sibling staging 目录，
全量验证通过后才原子发布。``pair-smpl`` 子命令在 SONIC 服务器上按 key 和严格 50 Hz
帧数和训练时 Robot-waist/SMPL-root 坐标语义建立 SMPL 硬链接；不存在、帧数不匹配或
整段中位相对姿态差超过 45 度的条目保留为 robot-only 并写明原因。

典型命令：

    python gear_sonic/tools/prepare_bumi3_pass50_dataset.py build \
      --source-root /data0/user/liwei/robot_retargeter_bumi3_hq4_zup_v1 \
      --quality-report /data0/user/liwei/datasets/bumi_quality_robot_retargeter_30hz_v1/quality_report.jsonl \
      --source-mjcf /home/user/liwei/robot_retargeter/asset/robot/bumi3/mjcf/bumi3_retarget.xml \
      --target-mjcf gear_sonic/data/assets/robot_description/mjcf/bumi3.xml \
      --output-root /data0/user/liwei/datasets/sonic_bumi3_hq4_pass50_v1

    python gear_sonic/tools/prepare_bumi3_pass50_dataset.py validate \
      --output-root /data0/user/liwei/datasets/sonic_bumi3_hq4_pass50_v1
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import joblib
import numpy as np
from scipy.spatial.transform import Rotation


SOURCE_FPS = 30.0
TARGET_FPS = 50.0
DEFAULT_EXPECTED_PASS_COUNT = 2790
SOURCE_CONTRACT = "robot_retargeter.bumi3_mimic_npz_30hz.v1"
OUTPUT_CONTRACT = "sonic.bumi3.pass_only_50hz.v1"
FOOT_BODY_NAMES = ("l_ankle_roll_link", "r_ankle_roll_link")


@dataclass(frozen=True)
class KinematicContract:
    """从 MJCF 提取的 SONIC 22-body/21-joint 核心运动学契约。"""

    body_names: tuple[str, ...]
    parent_names: tuple[str | None, ...]
    body_positions: tuple[tuple[float, float, float], ...]
    body_quaternions: tuple[tuple[float, float, float, float], ...]
    body_joint_names: tuple[str | None, ...]
    joint_axes: tuple[tuple[float, float, float] | None, ...]
    joint_positions: tuple[tuple[float, float, float] | None, ...]
    joint_ranges: tuple[tuple[float, float] | None, ...]
    actuator_joint_names: tuple[str, ...]
    source_sha256: str
    core_sha256: str


@dataclass(frozen=True)
class PassRecord:
    """质量报告中一条唯一 PASS 白名单记录。"""

    key: str
    dataset: str
    sample_id: str
    source_path: str
    source_relative_path: str
    source_sha256: str
    source_frames: int
    valid_intervals: list[list[int]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_vector(text: str | None, width: int, default: tuple[float, ...]) -> tuple[float, ...]:
    values = default if text is None else tuple(float(value) for value in text.split())
    if len(values) != width:
        raise ValueError(f"向量宽度应为 {width}，实际为 {values}")
    return tuple(values)


def _extract_contract(path: Path, core_body_names: tuple[str, ...] | None = None) -> KinematicContract:
    """提取 MJCF body 树；可按目标 22 body 名称过滤 retargeter 辅助 marker body。"""

    root = ET.parse(path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF 缺少 worldbody: {path}")
    rows: list[dict[str, Any]] = []

    def walk(parent: ET.Element, parent_name: str | None) -> None:
        for body in parent.findall("body"):
            body_name = body.get("name")
            if not body_name:
                raise ValueError(f"MJCF body 缺少 name: {path}")
            joint = body.find("joint")
            rows.append(
                {
                    "body_name": body_name,
                    "parent_name": parent_name,
                    "body_position": _parse_vector(body.get("pos"), 3, (0.0, 0.0, 0.0)),
                    "body_quaternion": _parse_vector(
                        body.get("quat"), 4, (1.0, 0.0, 0.0, 0.0)
                    ),
                    "joint_name": None if joint is None else joint.get("name"),
                    "joint_axis": None
                    if joint is None
                    else _parse_vector(joint.get("axis"), 3, (0.0, 0.0, 1.0)),
                    "joint_position": None
                    if joint is None
                    else _parse_vector(joint.get("pos"), 3, (0.0, 0.0, 0.0)),
                    "joint_range": None
                    if joint is None or joint.get("range") is None
                    else _parse_vector(joint.get("range"), 2, (0.0, 0.0)),
                }
            )
            walk(body, body_name)

    walk(worldbody, None)
    if core_body_names is not None:
        core_set = set(core_body_names)
        rows = [row for row in rows if row["body_name"] in core_set]
        by_name = {row["body_name"]: row for row in rows}
        if set(by_name) != core_set:
            raise ValueError(
                f"{path} 核心 body 集合不一致: missing={sorted(core_set-set(by_name))}, "
                f"extra={sorted(set(by_name)-core_set)}"
            )
        rows = [by_name[name] for name in core_body_names]

    actuator_root = root.find("actuator")
    if actuator_root is None:
        raise ValueError(f"MJCF 缺少 actuator: {path}")
    actuator_joint_names = tuple(
        element.get("joint") or ""
        for element in actuator_root
        if element.get("joint") is not None
    )
    payload = {
        "rows": rows,
        "actuator_joint_names": actuator_joint_names,
    }
    core_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    contract = KinematicContract(
        body_names=tuple(row["body_name"] for row in rows),
        parent_names=tuple(row["parent_name"] for row in rows),
        body_positions=tuple(row["body_position"] for row in rows),
        body_quaternions=tuple(row["body_quaternion"] for row in rows),
        body_joint_names=tuple(row["joint_name"] for row in rows),
        joint_axes=tuple(row["joint_axis"] for row in rows),
        joint_positions=tuple(row["joint_position"] for row in rows),
        joint_ranges=tuple(row["joint_range"] for row in rows),
        actuator_joint_names=actuator_joint_names,
        source_sha256=_sha256(path),
        core_sha256=hashlib.sha256(core_json.encode("utf-8")).hexdigest(),
    )
    if len(contract.body_names) != 22 or len(contract.actuator_joint_names) != 21:
        raise ValueError(
            f"BUMI3 核心必须为 22 body/21 actuator，实际为 "
            f"{len(contract.body_names)}/{len(contract.actuator_joint_names)}"
        )
    if contract.body_names[0] != "base_link" or contract.body_joint_names[0] is not None:
        raise ValueError("BUMI3 核心首 body 必须是无关节 base_link")
    joint_names = tuple(name for name in contract.body_joint_names if name is not None)
    if len(joint_names) != 21 or set(joint_names) != set(contract.actuator_joint_names):
        raise ValueError("BUMI3 body joint 与 actuator joint 集合不一致")
    return contract


def _assert_source_target_kinematics(source: KinematicContract, target: KinematicContract) -> None:
    """阻止同名但轴、origin、range 不同的 retargeter 数据进入 SONIC。"""

    fields = (
        "body_names",
        "parent_names",
        "body_positions",
        "body_quaternions",
        "body_joint_names",
        "joint_axes",
        "joint_positions",
        "joint_ranges",
        "actuator_joint_names",
    )
    mismatches = [field for field in fields if getattr(source, field) != getattr(target, field)]
    if mismatches:
        raise ValueError(f"retargeter 与 SONIC MJCF 核心运动学不一致: {mismatches}")


def _load_pass_records(
    source_root: Path,
    quality_report: Path,
    expected_count: int,
) -> tuple[list[PassRecord], str]:
    """只从报告读取 PASS 白名单，并验证唯一性、源路径和报告哈希。"""

    rows = [json.loads(line) for line in quality_report.open(encoding="utf-8") if line.strip()]
    passed = [
        row
        for row in rows
        if row.get("status") == "PASS" and row.get("quality_accepted") is True
    ]
    if len(passed) != expected_count:
        raise ValueError(f"PASS 白名单应为 {expected_count}，实际为 {len(passed)}")
    records: list[PassRecord] = []
    seen_keys: set[str] = set()
    source_root_resolved = source_root.resolve()
    for row in passed:
        dataset = str(row["dataset"])
        sample_id = str(row["sample_id"])
        if sample_id.startswith(f"{dataset}/"):
            sample_stem = sample_id[len(dataset) + 1 :]
        else:
            raise ValueError(f"sample_id 与 dataset 不匹配: {sample_id} / {dataset}")
        key = f"{dataset}__{sample_stem}"
        if key in seen_keys:
            raise ValueError(f"PASS key 重复: {key}")
        seen_keys.add(key)
        relative = Path(str(row["source_relative_path"]))
        source_path = (source_root_resolved / relative).resolve()
        if source_root_resolved not in source_path.parents or not source_path.is_file():
            raise FileNotFoundError(f"PASS 源文件越界或不存在: {source_path}")
        records.append(
            PassRecord(
                key=key,
                dataset=dataset,
                sample_id=sample_id,
                source_path=str(source_path),
                source_relative_path=relative.as_posix(),
                source_sha256=str(row["source_sha256"]),
                source_frames=int(row["metrics"]["num_frames"]),
                valid_intervals=[list(map(int, interval)) for interval in row.get("valid_intervals", [])],
            )
        )
    records.sort(key=lambda record: record.key)
    return records, _sha256(quality_report)


def _target_grid(source_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成独占 30 Hz 末帧的严格 50 Hz 网格和相邻源帧插值参数。"""

    if source_frames < 4:
        raise ValueError(f"动作至少需要 4 帧以重算 jerk，实际为 {source_frames}")
    duration = (source_frames - 1) / SOURCE_FPS
    target_frames = int(math.ceil(duration * TARGET_FPS - 1e-12))
    target_times = np.arange(target_frames, dtype=np.float64) / TARGET_FPS
    if target_times[-1] >= duration + 1e-12:
        raise AssertionError("50 Hz 时间网格错误地包含或超过源末帧")
    phase = target_times * SOURCE_FPS
    index_0 = np.floor(phase).astype(np.int64)
    index_1 = np.minimum(index_0 + 1, source_frames - 1)
    blend = phase - index_0
    return target_times, index_0, index_1, blend


def _linear_resample(
    value: np.ndarray,
    index_0: np.ndarray,
    index_1: np.ndarray,
    blend: np.ndarray,
) -> np.ndarray:
    weight_shape = (len(blend),) + (1,) * (value.ndim - 1)
    weight = blend.reshape(weight_shape)
    return value[index_0] * (1.0 - weight) + value[index_1] * weight


def _normalize_quaternions_wxyz(value: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("检测到零范数 quaternion")
    return value / norms


def _slerp_wxyz(
    value: np.ndarray,
    index_0: np.ndarray,
    index_1: np.ndarray,
    blend: np.ndarray,
) -> np.ndarray:
    """对 [T,...,4] wxyz 四元数执行向量化最短弧 SLERP。"""

    source = _normalize_quaternions_wxyz(np.asarray(value, dtype=np.float64))
    q0 = source[index_0]
    q1 = source[index_1].copy()
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    alpha = blend.reshape((len(blend),) + (1,) * (value.ndim - 1))
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-7
    safe = np.where(near, 1.0, sin_theta)
    result = np.sin((1.0 - alpha) * theta) / safe * q0
    result += np.sin(alpha * theta) / safe * q1
    linear = (1.0 - alpha) * q0 + alpha * q1
    result = np.where(near, linear, result)
    return _normalize_quaternions_wxyz(result)


def _derivative(value: np.ndarray, fps: float) -> np.ndarray:
    edge_order = 2 if len(value) >= 3 else 1
    return np.gradient(value, 1.0 / fps, axis=0, edge_order=edge_order)


def _angular_velocity_world_wxyz(quaternion: np.ndarray, fps: float) -> np.ndarray:
    """由 body 世界四元数重算世界系角速度，内部使用中心相对旋转。"""

    quat = _normalize_quaternions_wxyz(np.asarray(quaternion, dtype=np.float64))
    shape = quat.shape[:-1]
    if len(quat) < 3:
        raise ValueError("角速度重算至少需要 3 帧")
    output = np.empty(shape + (3,), dtype=np.float64)

    def relative_rotvec(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_xyzw = left[..., [1, 2, 3, 0]].reshape(-1, 4)
        right_xyzw = right[..., [1, 2, 3, 0]].reshape(-1, 4)
        relative = Rotation.from_quat(left_xyzw) * Rotation.from_quat(right_xyzw).inv()
        return relative.as_rotvec().reshape(left.shape[:-1] + (3,))

    output[0] = relative_rotvec(quat[1], quat[0]) * fps
    output[-1] = relative_rotvec(quat[-1], quat[-2]) * fps
    output[1:-1] = relative_rotvec(quat[2:], quat[:-2]) * (0.5 * fps)
    return output


def _remove_short_segments(mask: np.ndarray, minimum_frames: int = 3) -> np.ndarray:
    output = np.asarray(mask, dtype=bool).copy()
    for foot_index in range(output.shape[1]):
        start: int | None = None
        for frame in range(len(output) + 1):
            active = frame < len(output) and bool(output[frame, foot_index])
            if active and start is None:
                start = frame
            elif not active and start is not None:
                if frame - start < minimum_frames:
                    output[start:frame, foot_index] = False
                start = None
    return output


def _detect_contacts(foot_position: np.ndarray, foot_velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用低位+速度滞回重算左右脚接触，返回 mask 与每脚低位参考。"""

    references = np.percentile(foot_position[..., 2], 15.0, axis=0)
    vertical_speed = np.abs(foot_velocity[..., 2])
    horizontal_speed = np.linalg.norm(foot_velocity[..., :2], axis=-1)
    contacts = np.zeros(foot_position.shape[:2], dtype=bool)
    for foot in range(foot_position.shape[1]):
        active = False
        for frame in range(len(foot_position)):
            height = foot_position[frame, foot, 2]
            if not active:
                active = bool(
                    height <= references[foot] + 0.02
                    and vertical_speed[frame, foot] <= 0.10
                    and horizontal_speed[frame, foot] <= 0.20
                )
            elif (
                height > references[foot] + 0.04
                or vertical_speed[frame, foot] > 0.20
                or horizontal_speed[frame, foot] > 0.40
            ):
                active = False
            contacts[frame, foot] = active
    return _remove_short_segments(contacts), references


def _metric(value: np.ndarray) -> dict[str, float]:
    norm = np.linalg.norm(value, axis=-1)
    return {
        "max": float(np.max(norm, initial=0.0)),
        "p95": float(np.percentile(norm, 95.0)),
    }


def _convert_one(
    args: tuple[PassRecord, KinematicContract, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """转换一条 PASS 动作并返回 manifest/report 行。"""

    record, contract, staging_text = args
    staging = Path(staging_text)
    source_path = Path(record.source_path)
    actual_sha = _sha256(source_path)
    if actual_sha != record.source_sha256:
        raise ValueError(
            f"{record.key} 源哈希错误: report={record.source_sha256}, actual={actual_sha}"
        )
    with np.load(source_path, allow_pickle=False) as payload:
        required = {
            "fps",
            "joint_pos",
            "body_pos_w",
            "body_quat_w",
            "joint_names",
            "body_names",
            "quaternion_order",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{record.key} NPZ 缺少字段: {sorted(missing)}")
        fps = float(payload["fps"])
        joint_names = tuple(str(value) for value in payload["joint_names"].tolist())
        body_names = tuple(str(value) for value in payload["body_names"].tolist())
        quaternion_order = str(payload["quaternion_order"].item())
        joint_pos = np.asarray(payload["joint_pos"], dtype=np.float64)
        body_pos = np.asarray(payload["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(payload["body_quat_w"], dtype=np.float64)
    if not math.isclose(fps, SOURCE_FPS, abs_tol=1e-8):
        raise ValueError(f"{record.key} fps 必须为 30，实际为 {fps}")
    if quaternion_order != "wxyz":
        raise ValueError(f"{record.key} quaternion_order 必须为 wxyz")
    if body_names != contract.body_names:
        raise ValueError(f"{record.key} body 顺序与 SONIC MJCF 不一致")
    if set(joint_names) != set(contract.actuator_joint_names) or len(set(joint_names)) != 21:
        raise ValueError(f"{record.key} joint 名称集合与 SONIC MJCF 不一致")
    if joint_pos.shape != (record.source_frames, 21):
        raise ValueError(f"{record.key} joint_pos shape 错误: {joint_pos.shape}")
    if body_pos.shape != (record.source_frames, 22, 3):
        raise ValueError(f"{record.key} body_pos_w shape 错误: {body_pos.shape}")
    if body_quat.shape != (record.source_frames, 22, 4):
        raise ValueError(f"{record.key} body_quat_w shape 错误: {body_quat.shape}")
    for label, value in (("joint_pos", joint_pos), ("body_pos", body_pos), ("body_quat", body_quat)):
        if not np.isfinite(value).all():
            raise ValueError(f"{record.key} {label} 含 NaN/Inf")

    _, index_0, index_1, blend = _target_grid(record.source_frames)
    joint_pos_50_source = _linear_resample(joint_pos, index_0, index_1, blend)
    body_pos_50 = _linear_resample(body_pos, index_0, index_1, blend)
    body_quat_50 = _slerp_wxyz(body_quat, index_0, index_1, blend)
    source_joint_index = {name: index for index, name in enumerate(joint_names)}
    dof_50 = joint_pos_50_source[
        :, [source_joint_index[name] for name in contract.actuator_joint_names]
    ]

    body_index = {name: index for index, name in enumerate(contract.body_names)}
    root_index = body_index["base_link"]
    root_pos_50 = body_pos_50[:, root_index]
    root_quat_wxyz_50 = body_quat_50[:, root_index]
    pose_aa_50 = np.zeros((len(dof_50), 22, 3), dtype=np.float64)
    pose_aa_50[:, 0] = Rotation.from_quat(
        root_quat_wxyz_50[:, [1, 2, 3, 0]]
    ).as_rotvec()
    dof_index = {name: index for index, name in enumerate(contract.actuator_joint_names)}
    for body_id in range(1, 22):
        joint_name = contract.body_joint_names[body_id]
        axis = contract.joint_axes[body_id]
        if joint_name is None or axis is None:
            raise ValueError(f"{record.key} body {contract.body_names[body_id]} 缺少关节")
        pose_aa_50[:, body_id] = dof_50[:, dof_index[joint_name], None] * np.asarray(axis)

    joint_vel_50 = _derivative(dof_50, TARGET_FPS)
    joint_acc_50 = _derivative(joint_vel_50, TARGET_FPS)
    joint_jerk_50 = _derivative(joint_acc_50, TARGET_FPS)
    body_lin_vel_50 = _derivative(body_pos_50, TARGET_FPS)
    body_ang_vel_50 = _angular_velocity_world_wxyz(body_quat_50, TARGET_FPS)
    foot_indices = [body_index[name] for name in FOOT_BODY_NAMES]
    contacts_50, contact_references = _detect_contacts(
        body_pos_50[:, foot_indices], body_lin_vel_50[:, foot_indices]
    )
    arrays = (
        root_pos_50,
        root_quat_wxyz_50,
        pose_aa_50,
        dof_50,
        joint_vel_50,
        joint_acc_50,
        joint_jerk_50,
        body_lin_vel_50,
        body_ang_vel_50,
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError(f"{record.key} 50 Hz 输出或派生量含 NaN/Inf")
    quaternion_norm_error = float(
        np.max(np.abs(np.linalg.norm(root_quat_wxyz_50, axis=1) - 1.0))
    )
    if quaternion_norm_error > 1e-6:
        raise ValueError(f"{record.key} 50 Hz root quaternion 范数错误")

    raw_link = staging / "source_npz_pass" / f"{record.key}.npz"
    try:
        os.link(source_path, raw_link)
    except OSError as error:
        raise RuntimeError(f"{record.key} 原始 NPZ 必须硬链接，禁止复制: {error}") from error
    robot_path = staging / "built" / "robot_all" / f"{record.key}.pkl"
    audit_path = staging / "audit_50hz" / f"{record.key}.npz"
    joblib.dump(
        {
            record.key: {
                "root_trans_offset": root_pos_50.astype(np.float32),
                "pose_aa": pose_aa_50.astype(np.float32),
                "dof": dof_50.astype(np.float32),
                "root_rot": root_quat_wxyz_50[:, [1, 2, 3, 0]].astype(np.float32),
                "fps": 50,
                "source_motion_contract_version": SOURCE_CONTRACT,
                "output_motion_contract_version": OUTPUT_CONTRACT,
                "source_sha256": actual_sha,
            }
        },
        robot_path,
        compress=3,
    )
    np.savez_compressed(
        audit_path,
        fps=np.asarray(TARGET_FPS, dtype=np.float64),
        joint_vel=joint_vel_50.astype(np.float32),
        joint_acc=joint_acc_50.astype(np.float32),
        joint_jerk=joint_jerk_50.astype(np.float32),
        body_lin_vel_w=body_lin_vel_50.astype(np.float32),
        body_ang_vel_w=body_ang_vel_50.astype(np.float32),
        foot_contact=contacts_50.astype(np.uint8),
        foot_contact_body_names=np.asarray(FOOT_BODY_NAMES),
        foot_contact_height_reference=contact_references.astype(np.float32),
    )
    report = {
        "key": record.key,
        "source_frames": record.source_frames,
        "target_frames": len(dof_50),
        "source_fps": SOURCE_FPS,
        "target_fps": TARGET_FPS,
        "duration_seconds_exclusive": (record.source_frames - 1) / SOURCE_FPS,
        "root_quaternion_norm_max_error": quaternion_norm_error,
        "joint_velocity_l2": _metric(joint_vel_50),
        "joint_acceleration_l2": _metric(joint_acc_50),
        "joint_jerk_l2": _metric(joint_jerk_50),
        "root_linear_velocity": _metric(body_lin_vel_50[:, root_index]),
        "root_angular_velocity": _metric(body_ang_vel_50[:, root_index]),
        "contact_frames": contacts_50.sum(axis=0).astype(int).tolist(),
        "contact_ratios": contacts_50.mean(axis=0).tolist(),
        "finite": True,
    }
    manifest = {
        **asdict(record),
        "robot_file": f"built/robot_all/{record.key}.pkl",
        "raw_hardlink": f"source_npz_pass/{record.key}.npz",
        "audit_file": f"audit_50hz/{record.key}.npz",
        "target_frames": len(dof_50),
        "target_fps": TARGET_FPS,
        "output_contract": OUTPUT_CONTRACT,
    }
    return manifest, report


def _prepare_staging(output_root: Path) -> Path:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging 已存在: {staging}")
    for relative in ("source_npz_pass", "built/robot_all", "audit_50hz", "meta"):
        (staging / relative).mkdir(parents=True, exist_ok=False)
    return staging


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _validate_training_outputs(output_root: Path, expected_count: int) -> dict[str, int]:
    manifests = [
        json.loads(line)
        for line in (output_root / "meta" / "manifest.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    if len(manifests) != expected_count or len({row["key"] for row in manifests}) != expected_count:
        raise ValueError("manifest 数量或 key 唯一性错误")
    robot_files = sorted((output_root / "built" / "robot_all").glob("*.pkl"))
    if len(robot_files) != expected_count:
        raise ValueError(f"robot pkl 应为 {expected_count}，实际为 {len(robot_files)}")
    total_frames = 0
    for index, row in enumerate(manifests, start=1):
        path = output_root / row["robot_file"]
        outer = joblib.load(path)
        if list(outer) != [row["key"]]:
            raise ValueError(f"{row['key']} joblib 外层 key 错误")
        item = outer[row["key"]]
        frames = int(row["target_frames"])
        expected_shapes = {
            "root_trans_offset": (frames, 3),
            "pose_aa": (frames, 22, 3),
            "dof": (frames, 21),
            "root_rot": (frames, 4),
        }
        for field, shape in expected_shapes.items():
            value = np.asarray(item[field])
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"{row['key']} {field} 契约错误: {value.shape}/{value.dtype}")
        if int(item["fps"]) != 50 or item.get("output_motion_contract_version") != OUTPUT_CONTRACT:
            raise ValueError(f"{row['key']} 50 Hz/contract 元数据错误")
        total_frames += frames
        if index % 200 == 0 or index == len(manifests):
            print(f"[validate-training] {index}/{len(manifests)}", flush=True)
    return {"robot_count": len(manifests), "total_target_frames": total_frames}


def _validate_full_outputs(output_root: Path, expected_count: int) -> dict[str, int]:
    summary = _validate_training_outputs(output_root, expected_count)
    manifests = [
        json.loads(line)
        for line in (output_root / "meta" / "manifest.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    audits = sorted((output_root / "audit_50hz").glob("*.npz"))
    raws = sorted((output_root / "source_npz_pass").glob("*.npz"))
    reports = [
        json.loads(line)
        for line in (output_root / "meta" / "resample_report.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    if len(audits) != expected_count or len(raws) != expected_count or len(reports) != expected_count:
        raise ValueError("raw/audit/report 数量不等于 PASS 白名单")
    for index, row in enumerate(manifests, start=1):
        raw = output_root / row["raw_hardlink"]
        source = Path(row["source_path"])
        raw_stat = raw.stat()
        source_stat = source.stat()
        if (raw_stat.st_dev, raw_stat.st_ino) != (source_stat.st_dev, source_stat.st_ino):
            raise ValueError(f"{row['key']} raw 不是源 NPZ 硬链接")
        with np.load(output_root / row["audit_file"], allow_pickle=False) as audit:
            frames = int(row["target_frames"])
            shapes = {
                "joint_vel": (frames, 21),
                "joint_acc": (frames, 21),
                "joint_jerk": (frames, 21),
                "body_lin_vel_w": (frames, 22, 3),
                "body_ang_vel_w": (frames, 22, 3),
                "foot_contact": (frames, 2),
            }
            for field, shape in shapes.items():
                value = np.asarray(audit[field])
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"{row['key']} audit {field} 契约错误")
        if index % 200 == 0 or index == len(manifests):
            print(f"[validate-full] {index}/{len(manifests)}", flush=True)
    return summary


def build(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    target_contract = _extract_contract(args.target_mjcf.resolve())
    source_contract = _extract_contract(args.source_mjcf.resolve(), target_contract.body_names)
    _assert_source_target_kinematics(source_contract, target_contract)
    records, quality_report_sha = _load_pass_records(
        args.source_root.resolve(),
        args.quality_report.resolve(),
        args.expected_pass_count,
    )
    staging = _prepare_staging(output_root)
    jobs = [(record, target_contract, str(staging)) for record in records]
    manifests: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, (manifest, report) in enumerate(executor.map(_convert_one, jobs), start=1):
                manifests.append(manifest)
                reports.append(report)
                if index % 50 == 0 or index == len(jobs):
                    print(f"[convert] {index}/{len(jobs)}", flush=True)
        manifests.sort(key=lambda row: row["key"])
        reports.sort(key=lambda row: row["key"])
        _write_jsonl(staging / "meta" / "manifest.jsonl", manifests)
        _write_jsonl(staging / "meta" / "resample_report.jsonl", reports)
        provenance = {
            "contract_version": OUTPUT_CONTRACT,
            "quality_report": str(args.quality_report.resolve()),
            "quality_report_sha256": quality_report_sha,
            "pass_count": len(records),
            "source_fps": SOURCE_FPS,
            "target_fps": TARGET_FPS,
            "time_grid": "arange(0,(T-1)/30,1/50), endpoint_exclusive",
            "position_interpolation": "linear",
            "quaternion_interpolation": "shortest_arc_slerp_wxyz",
            "derivatives": "recomputed_from_50hz_positions_and_quaternions",
            "contact_detection": {
                "bodies": list(FOOT_BODY_NAMES),
                "height_reference_percentile": 15.0,
                "enter_height_margin_m": 0.02,
                "exit_height_margin_m": 0.04,
                "enter_vertical_speed_mps": 0.10,
                "exit_vertical_speed_mps": 0.20,
                "enter_horizontal_speed_mps": 0.20,
                "exit_horizontal_speed_mps": 0.40,
                "minimum_segment_frames": 3,
            },
            "source_mjcf_sha256": source_contract.source_sha256,
            "target_mjcf_sha256": target_contract.source_sha256,
            "kinematic_core_sha256": target_contract.core_sha256,
            "raw_storage": "hardlink_only",
        }
        (staging / "meta" / "provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary = _validate_full_outputs(staging, args.expected_pass_count)
        (staging / "meta" / "validation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
    except Exception:
        print(f"构建失败，保留 staging 供审计: {staging}", flush=True)
        raise
    print(f"BUMI3_PASS50_BUILD=PASS output={output_root} count={len(records)}")


def validate(args: argparse.Namespace) -> None:
    if args.training_only:
        summary = _validate_training_outputs(args.output_root.resolve(), args.expected_pass_count)
    else:
        summary = _validate_full_outputs(args.output_root.resolve(), args.expected_pass_count)
    print("BUMI3_PASS50_VALIDATE=PASS " + json.dumps(summary, sort_keys=True))


def pair_smpl(args: argparse.Namespace) -> None:
    """按 key、50 Hz 帧数和训练坐标语义建立 SMPL 硬链接。"""

    output_root = args.output_root.resolve()
    smpl_source = args.smpl_source.resolve()
    manifests = [
        json.loads(line)
        for line in (output_root / "meta" / "manifest.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    destination = output_root / "built" / "smpl_all"
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    paired = 0
    for row in manifests:
        source = smpl_source / f"{row['key']}.pkl"
        target = destination / source.name
        # destination 由本工具独占；重跑时先移除旧配对，防止契约变化后残留陈旧硬链接。
        if target.exists() or target.is_symlink():
            target.unlink()
        status = "ROBOT_ONLY_MISSING_SMPL"
        reason = "source file missing"
        pair_median_degrees: float | None = None
        if source.is_file():
            payload = joblib.load(source)
            frames = int(row["target_frames"])
            fps = float(payload.get("fps", math.nan)) if isinstance(payload, dict) else math.nan
            shapes_ok = isinstance(payload, dict) and all(
                np.asarray(payload[field]).shape[0] == frames
                for field in ("pose_aa", "transl", "smpl_joints")
                if field in payload
            ) and all(field in payload for field in ("pose_aa", "transl", "smpl_joints"))
            if math.isclose(fps, TARGET_FPS, abs_tol=1e-8) and shapes_ok:
                robot_payload = joblib.load(output_root / row["robot_file"])[row["key"]]
                robot_root = Rotation.from_quat(
                    np.asarray(robot_payload["root_rot"], dtype=np.float64)
                )
                robot_waist_local = Rotation.from_rotvec(
                    np.asarray(robot_payload["pose_aa"], dtype=np.float64)[:, 1]
                )
                robot_waist = robot_root * robot_waist_local
                smpl_pose = np.asarray(payload["pose_aa"], dtype=np.float64)
                smpl_root_y_up = Rotation.from_rotvec(smpl_pose[:, :3])
                y_to_z = Rotation.from_rotvec(np.array([math.pi / 2.0, 0.0, 0.0]))
                smpl_base = Rotation.from_quat(np.array([0.5, 0.5, 0.5, 0.5]))
                processed_smpl_root = y_to_z * smpl_root_y_up * smpl_base.inv()
                pair_median_degrees = float(
                    np.median(
                        np.degrees((robot_waist.inv() * processed_smpl_root).magnitude())
                    )
                )
                if pair_median_degrees <= args.max_pair_median_degrees:
                    os.link(source, target)
                    status = "PAIRED_HARDLINK"
                    reason = None
                    paired += 1
                else:
                    status = "ROBOT_ONLY_SMPL_COORDINATE_MISMATCH"
                    reason = (
                        f"pair_median_degrees={pair_median_degrees:.9f}, "
                        f"threshold={args.max_pair_median_degrees:.9f}"
                    )
            else:
                status = "ROBOT_ONLY_SMPL_CONTRACT_MISMATCH"
                reason = f"fps={fps}, expected_frames={frames}"
        rows.append(
            {
                "key": row["key"],
                "status": status,
                "reason": reason,
                "pair_median_degrees": pair_median_degrees,
            }
        )
    _write_jsonl(output_root / "meta" / "smpl_pairing_report.jsonl", rows)
    status_counts = dict(sorted(Counter(row["status"] for row in rows).items()))
    paired_medians = [
        float(row["pair_median_degrees"])
        for row in rows
        if row["status"] == "PAIRED_HARDLINK"
    ]
    summary = {
        "robot_count": len(manifests),
        "paired_count": paired,
        "robot_only_count": len(manifests) - paired,
        "status_counts": status_counts,
        "paired_median_degrees_median": (
            float(np.median(paired_medians)) if paired_medians else None
        ),
        "paired_median_degrees_max": max(paired_medians, default=None),
        "smpl_source": str(smpl_source),
        "storage": "hardlink",
        "max_pair_median_degrees": args.max_pair_median_degrees,
    }
    (output_root / "meta" / "smpl_pairing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("BUMI3_PASS50_PAIR_SMPL=PASS " + json.dumps(summary, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source-root", type=Path, required=True)
    build_parser.add_argument("--quality-report", type=Path, required=True)
    build_parser.add_argument("--source-mjcf", type=Path, required=True)
    build_parser.add_argument("--target-mjcf", type=Path, required=True)
    build_parser.add_argument("--output-root", type=Path, required=True)
    build_parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    build_parser.add_argument("--expected-pass-count", type=int, default=DEFAULT_EXPECTED_PASS_COUNT)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output-root", type=Path, required=True)
    validate_parser.add_argument("--expected-pass-count", type=int, default=DEFAULT_EXPECTED_PASS_COUNT)
    validate_parser.add_argument("--training-only", action="store_true")

    pair_parser = subparsers.add_parser("pair-smpl")
    pair_parser.add_argument("--output-root", type=Path, required=True)
    pair_parser.add_argument("--smpl-source", type=Path, required=True)
    pair_parser.add_argument("--max-pair-median-degrees", type=float, default=45.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build(args)
    elif args.command == "validate":
        validate(args)
    elif args.command == "pair-smpl":
        pair_smpl(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
