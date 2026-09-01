#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""构建并审计 BUMI3 SONIC 三数据源只读训练索引。

本工具把服务器上的三个既有数据来源组织成一个 MotionLib 可直接读取的
目录视图：
``bumi3_smpl_97660_v1`` 的 train/test 配对、``hq4_pass50_v1`` 的四库 PASS-only
配对，以及 ``hq_all_v2`` 中精确以 ``mine__`` 开头的 99 条 Robot-only 动作。工具
不会重写、重采样或复制源 PKL；发布目录只包含指向源文件的软链接、
逐动作 JSONL 清单、来源哈希和全量审计汇总，因此可以明确区分源数据与
训练索引。

审计按字段而不是笼统的“整份 PKL 坐标系”判断契约。Robot 必须包含
21 DoF、
22 个 ``pose_aa`` 节点、Z-up 根平移和与 ``pose_aa`` 根姿态一致的 xyzw
``root_rot``。SMPL 必须是 50 Hz，``pose_aa``/``transl`` 保持 SMPL Y-up 语义，
而训练实际读取的 ``smpl_joints`` 必须呈 Z-up 人体轴。配对还会复现 SONIC 的
SMPL Y-up 到 Z-up 旋转、base rotation removal 和 BUMI3 ``waist_yaw_link`` FK，
检查两侧根姿态中位差。Robot 合格但 SMPL 缺失、字段错误、尾帧差超过
2 帧或姿态差超过 45 度时，只移除该条 SMPL 软链接并保留为 Robot-only；
Robot 本身不满足契约则整次构建失败。

构建在输出目录旁的唯一 staging 目录完成，全部检查通过后才原子发布。
默认输出为
``/data/sonic_bumi3/datasets/bumi3_sonic_three_source_v1``；若目标已经存在则拒绝
覆盖，避免误删已有训练索引。典型命令：

    python gear_sonic/tools/build_bumi3_three_source_dataset.py build \
      --workers 8

    python gear_sonic/tools/build_bumi3_three_source_dataset.py validate \
      --output-root /data/sonic_bumi3/datasets/bumi3_sonic_three_source_v1 \
      --workers 8

该工具只验证数据静态契约；它不能替代 Isaac Lab reset/step、奖励曲线、
动作回放或最终控制质量验证。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import joblib
import numpy as np
from scipy.spatial.transform import Rotation


CONTRACT_VERSION = "sonic.bumi3.three_source_index.v1"
TARGET_FPS = 50.0
MAX_PAIR_FRAME_DELTA = 2
MAX_PAIR_MEDIAN_DEGREES = 45.0
ROBOT_POSE_NODES = 22
ROBOT_NUM_DOF = 21
SMPL_NUM_JOINTS = 24
MAX_AUDIT_FRAMES_PER_CLIP = 256

REPO_ROOT = Path(__file__).resolve().parents[2]
CURRENT_MJCF_PATH = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/bumi3.xml"

BUMI3_MUJOCO_DOF_NAMES = [
    "waist_yaw_joint",
    "l_arm_pitch_joint",
    "l_arm_roll_joint",
    "l_arm_yaw_joint",
    "l_elbow_pitch_joint",
    "r_arm_pitch_joint",
    "r_arm_roll_joint",
    "r_arm_yaw_joint",
    "r_elbow_pitch_joint",
    "l_leg_pitch_joint",
    "l_leg_roll_joint",
    "l_leg_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_leg_pitch_joint",
    "r_leg_roll_joint",
    "r_leg_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
]

DEFAULT_DATA_ROOT = Path("/data/sonic_bumi3/datasets")
DEFAULT_LARGE_ROOT = DEFAULT_DATA_ROOT / "bumi3_smpl_97660_v1"
DEFAULT_HQ4_ROOT = DEFAULT_DATA_ROOT / "hq4_pass50_v1"
DEFAULT_MINE_ROBOT_DIR = DEFAULT_DATA_ROOT / "hq_all_v2/built/robot_all"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "bumi3_sonic_three_source_v1"

DEFAULT_EXPECTED_LARGE_TRAIN = 92443
DEFAULT_EXPECTED_LARGE_TEST = 5217
DEFAULT_EXPECTED_HQ4_ROBOT = 2790
DEFAULT_EXPECTED_HQ4_SMPL = 2788
DEFAULT_EXPECTED_MINE = 99


@dataclass(frozen=True)
class SourceRecord:
    """一条待审计动作的来源、拆分和预期帧率契约。"""

    key: str
    split: str
    source: str
    robot_path: str
    smpl_path: str | None
    expected_robot_fps: float


def _load_mjcf_joint_contract(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """按 MJCF body 遍历顺序读取 21 个关节的名称、轴和限位。"""

    if not path.is_file():
        raise ValueError(f"MJCF 契约文件不存在: {path}")
    root = ET.parse(path).getroot()
    names: list[str] = []
    axes: list[np.ndarray] = []
    ranges: list[np.ndarray] = []
    for body in root.findall(".//worldbody//body"):
        joints = body.findall("joint")
        if not joints:
            continue
        if len(joints) != 1:
            raise ValueError(f"{path} body={body.get('name')} 含多个关节")
        joint = joints[0]
        name = joint.get("name")
        axis_text = joint.get("axis")
        range_text = joint.get("range")
        if not name or axis_text is None or range_text is None:
            raise ValueError(f"{path} 关节缺少 name/axis/range: {ET.tostring(joint)}")
        axis = np.fromstring(axis_text, sep=" ", dtype=np.float64)
        joint_range = np.fromstring(range_text, sep=" ", dtype=np.float64)
        if axis.shape != (3,) or joint_range.shape != (2,):
            raise ValueError(f"{path} 关节 {name} 的 axis/range 维度错误")
        axis_norm = np.linalg.norm(axis)
        if not np.isclose(axis_norm, 1.0, atol=1e-8):
            raise ValueError(f"{path} 关节 {name} 的 axis 不是单位向量: {axis}")
        names.append(name)
        axes.append(axis / axis_norm)
        ranges.append(joint_range)
    if names != BUMI3_MUJOCO_DOF_NAMES:
        raise ValueError(f"{path} BUMI3 MuJoCo 关节顺序错误: {names}")
    return names, np.stack(axes), np.stack(ranges)


def _sha256(path: Path) -> str:
    """流式计算大文件 SHA256，避免一次性读取完整 PKL。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CURRENT_DOF_NAMES, CURRENT_DOF_AXES, CURRENT_DOF_RANGES = _load_mjcf_joint_contract(
    CURRENT_MJCF_PATH
)
CURRENT_MJCF_SHA256 = _sha256(CURRENT_MJCF_PATH)


def _sample_indices(num_frames: int) -> np.ndarray:
    """为姿态统计选取覆盖整段的确定性帧索引，限制单条计算成本。"""

    if num_frames <= 0:
        raise ValueError("动作帧数必须大于 0")
    count = min(num_frames, MAX_AUDIT_FRAMES_PER_CLIP)
    return np.linspace(0, num_frames - 1, count, dtype=np.int64)


def _index_pkls(root: Path, pattern: str = "*.pkl") -> dict[str, Path]:
    """递归建立 basename key 到 PKL 的唯一索引，并拒绝静默覆盖重名文件。"""

    if not root.is_dir():
        raise ValueError(f"PKL 目录不存在: {root}")
    result: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = defaultdict(list)
    for path in sorted(root.rglob(pattern)):
        key = path.stem
        if key in result:
            duplicates[key].extend([str(result[key]), str(path)])
        else:
            result[key] = path.resolve()
    if duplicates:
        preview = {key: sorted(set(paths)) for key, paths in list(duplicates.items())[:10]}
        raise ValueError(f"目录中存在重复 basename，MotionLib 会静默覆盖: {preview}")
    return result


def _require_count(label: str, actual: int, expected: int) -> None:
    """校验来源计数，阻止传错目录后仍生成看似可用的索引。"""

    if actual != expected:
        raise ValueError(f"{label} 数量 {actual} != 预期 {expected}")


def _read_json_object(path: Path) -> dict[str, Any]:
    """读取来源 provenance，并要求顶层为 JSON 字典。"""

    if not path.is_file():
        raise ValueError(f"来源 provenance 不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"来源 provenance 顶层不是字典: {path}")
    return payload


def _validate_source_asset_contracts(
    large_root: Path, hq4_root: Path, mine_robot_dir: Path
) -> dict[str, Any]:
    """验证三来源的 MJCF 指纹及大集归档的关节顺序、轴和限位。"""

    large_source_mjcf = large_root / "meta/bumi3.source.xml"
    large_names, large_axes, large_ranges = _load_mjcf_joint_contract(large_source_mjcf)
    if large_names != CURRENT_DOF_NAMES:
        raise ValueError("大集归档 MJCF 的关节顺序与当前 SONIC BUMI3 不一致")
    if not np.allclose(large_axes, CURRENT_DOF_AXES, atol=1e-8):
        raise ValueError("大集归档 MJCF 的关节轴与当前 SONIC BUMI3 不一致")
    if not np.allclose(large_ranges, CURRENT_DOF_RANGES, atol=1e-8):
        raise ValueError("大集归档 MJCF 的关节限位与当前 SONIC BUMI3 不一致")

    hq4_provenance_path = hq4_root / "meta/provenance.json"
    hq4_provenance = _read_json_object(hq4_provenance_path)
    if hq4_provenance.get("target_mjcf_sha256") != CURRENT_MJCF_SHA256:
        raise ValueError(
            "hq4 PASS50 的目标 MJCF 与当前 SONIC BUMI3 不一致: "
            f"{hq4_provenance.get('target_mjcf_sha256')} != {CURRENT_MJCF_SHA256}"
        )

    hq_all_root = mine_robot_dir.parents[1]
    mine_provenance_path = hq_all_root / "meta/provenance.json"
    mine_provenance = _read_json_object(mine_provenance_path)
    if mine_provenance.get("mjcf_sha256") != CURRENT_MJCF_SHA256:
        raise ValueError(
            "Mine Robot 来源 MJCF 与当前 SONIC BUMI3 不一致: "
            f"{mine_provenance.get('mjcf_sha256')} != {CURRENT_MJCF_SHA256}"
        )

    return {
        "current_mjcf_path": str(CURRENT_MJCF_PATH),
        "current_mjcf_sha256": CURRENT_MJCF_SHA256,
        "large_source_mjcf_path": str(large_source_mjcf),
        "large_source_mjcf_sha256": _sha256(large_source_mjcf),
        "hq4_provenance_path": str(hq4_provenance_path),
        "hq4_provenance_sha256": _sha256(hq4_provenance_path),
        "mine_provenance_path": str(mine_provenance_path),
        "mine_provenance_sha256": _sha256(mine_provenance_path),
        "mujoco_dof_names": CURRENT_DOF_NAMES,
        "mujoco_dof_axes": CURRENT_DOF_AXES.tolist(),
        "mujoco_dof_ranges": CURRENT_DOF_RANGES.tolist(),
    }


def discover_records(args: argparse.Namespace) -> tuple[list[SourceRecord], dict[str, Any]]:
    """发现三来源文件，并验证配对集合、拆分和跨来源 key 唯一性。"""

    large_root = args.large_root.resolve()
    hq4_root = args.hq4_root.resolve()
    mine_robot_dir = args.mine_robot_dir.resolve()
    asset_contracts = _validate_source_asset_contracts(
        large_root=large_root,
        hq4_root=hq4_root,
        mine_robot_dir=mine_robot_dir,
    )

    large_train_robot = _index_pkls(large_root / "train/robot_filtered")
    large_train_smpl = _index_pkls(large_root / "train/smpl_filtered")
    large_test_robot = _index_pkls(large_root / "test/robot_filtered")
    large_test_smpl = _index_pkls(large_root / "test/smpl_filtered")
    hq4_robot = _index_pkls(hq4_root / "built/robot_all")
    hq4_smpl = _index_pkls(hq4_root / "built/smpl_all")
    mine_all = _index_pkls(mine_robot_dir)
    mine_robot = {key: path for key, path in mine_all.items() if key.startswith("mine__")}

    _require_count("大集 train Robot", len(large_train_robot), args.expected_large_train)
    _require_count("大集 train SMPL", len(large_train_smpl), args.expected_large_train)
    _require_count("大集 test Robot", len(large_test_robot), args.expected_large_test)
    _require_count("大集 test SMPL", len(large_test_smpl), args.expected_large_test)
    _require_count("hq4 Robot", len(hq4_robot), args.expected_hq4_robot)
    _require_count("hq4 SMPL", len(hq4_smpl), args.expected_hq4_smpl)
    _require_count("Mine Robot-only", len(mine_robot), args.expected_mine)

    if set(large_train_robot) != set(large_train_smpl):
        raise ValueError(
            "大集 train Robot/SMPL key 不完全一致: "
            f"robot_only={sorted(set(large_train_robot)-set(large_train_smpl))[:10]}, "
            f"smpl_only={sorted(set(large_train_smpl)-set(large_train_robot))[:10]}"
        )
    if set(large_test_robot) != set(large_test_smpl):
        raise ValueError("大集 test Robot/SMPL key 不完全一致")
    if not set(hq4_smpl).issubset(hq4_robot):
        raise ValueError(
            f"hq4 存在无 Robot 的 SMPL: {sorted(set(hq4_smpl)-set(hq4_robot))[:10]}"
        )

    train_sets = {
        "large_train": set(large_train_robot),
        "hq4": set(hq4_robot),
        "mine": set(mine_robot),
    }
    labels = list(train_sets)
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            overlap = train_sets[left] & train_sets[right]
            if overlap:
                raise ValueError(f"训练来源 key 冲突 {left}/{right}: {sorted(overlap)[:10]}")
    train_keys = set().union(*train_sets.values())
    test_overlap = train_keys & set(large_test_robot)
    if test_overlap:
        raise ValueError(f"train/test 存在动作 key 交叉: {sorted(test_overlap)[:10]}")

    records: list[SourceRecord] = []
    for key, robot_path in large_train_robot.items():
        records.append(
            SourceRecord(
                key=key,
                split="train",
                source="large_train",
                robot_path=str(robot_path),
                smpl_path=str(large_train_smpl[key]),
                expected_robot_fps=TARGET_FPS,
            )
        )
    for key, robot_path in hq4_robot.items():
        records.append(
            SourceRecord(
                key=key,
                split="train",
                source="hq4_pass50",
                robot_path=str(robot_path),
                smpl_path=None if key not in hq4_smpl else str(hq4_smpl[key]),
                expected_robot_fps=TARGET_FPS,
            )
        )
    for key, robot_path in mine_robot.items():
        records.append(
            SourceRecord(
                key=key,
                split="train",
                source="mine_robot_only",
                robot_path=str(robot_path),
                smpl_path=None,
                expected_robot_fps=30.0,
            )
        )
    for key, robot_path in large_test_robot.items():
        records.append(
            SourceRecord(
                key=key,
                split="test",
                source="large_test",
                robot_path=str(robot_path),
                smpl_path=str(large_test_smpl[key]),
                expected_robot_fps=TARGET_FPS,
            )
        )
    records.sort(key=lambda row: (row.split, row.source, row.key))

    discovery = {
        "large_train_robot": len(large_train_robot),
        "large_train_smpl": len(large_train_smpl),
        "large_test_robot": len(large_test_robot),
        "large_test_smpl": len(large_test_smpl),
        "hq4_robot": len(hq4_robot),
        "hq4_smpl": len(hq4_smpl),
        "mine_robot_only": len(mine_robot),
        "train_unique_keys": len(train_keys),
        "test_unique_keys": len(large_test_robot),
        "asset_contracts": asset_contracts,
    }
    return records, discovery


def _load_robot(path: Path, expected_key: str, expected_fps: float) -> tuple[dict, dict]:
    """读取并严格验证单条 BUMI3 Robot PKL，返回负载和坐标统计。"""

    payload = joblib.load(path)
    if not isinstance(payload, dict) or list(payload) != [expected_key]:
        actual = list(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"Robot 外层 key 错误: {actual}")
    robot = payload[expected_key]
    required = ("root_trans_offset", "pose_aa", "dof", "root_rot", "fps")
    missing = [field for field in required if field not in robot]
    if missing:
        raise ValueError(f"Robot 缺少字段: {missing}")

    root_trans = np.asarray(robot["root_trans_offset"], dtype=np.float64)
    pose_aa = np.asarray(robot["pose_aa"], dtype=np.float64)
    dof = np.asarray(robot["dof"], dtype=np.float64)
    root_xyzw = np.asarray(robot["root_rot"], dtype=np.float64)
    frames = len(root_trans)
    expected_shapes = {
        "root_trans_offset": (frames, 3),
        "pose_aa": (frames, ROBOT_POSE_NODES, 3),
        "dof": (frames, ROBOT_NUM_DOF),
        "root_rot": (frames, 4),
    }
    actual_shapes = {
        "root_trans_offset": root_trans.shape,
        "pose_aa": pose_aa.shape,
        "dof": dof.shape,
        "root_rot": root_xyzw.shape,
    }
    if actual_shapes != expected_shapes:
        raise ValueError(f"Robot shape 错误: actual={actual_shapes}, expected={expected_shapes}")
    if frames <= 1:
        raise ValueError(f"Robot 帧数过短: {frames}")
    if not all(np.isfinite(value).all() for value in (root_trans, pose_aa, dof, root_xyzw)):
        raise ValueError("Robot 存在 NaN/Inf")
    fps = float(robot["fps"])
    if not math.isclose(fps, expected_fps, abs_tol=1e-8):
        raise ValueError(f"Robot fps={fps}，预期 {expected_fps}")

    # BUMI3 的 pose_aa 第 1 至 21 节点应逐帧等于当前 MJCF 对应轴乘以同序 dof。
    # 该恒等式同时锁住 MuJoCo 关节顺序、轴符号和 waist_yaw 取反结果，避免只检查
    # 21 维 shape 却把另一套 BUMI3 顺序或 waist -Z 数据送入训练。
    expected_local_pose = dof[:, :, None] * CURRENT_DOF_AXES[None, :, :]
    pose_dof_axis_error = np.linalg.norm(pose_aa[:, 1:, :] - expected_local_pose, axis=-1)
    pose_dof_axis_error_max = float(np.max(pose_dof_axis_error))
    if pose_dof_axis_error_max > 1e-6:
        max_frame, max_joint = np.unravel_index(
            int(np.argmax(pose_dof_axis_error)), pose_dof_axis_error.shape
        )
        raise ValueError(
            "Robot pose_aa/dof/MJCF 关节顺序或轴符号不一致: "
            f"max_error={pose_dof_axis_error_max}, frame={max_frame}, "
            f"joint={CURRENT_DOF_NAMES[max_joint]}"
        )
    time_origin, time_origin_mode = _declared_time_origin(robot)

    indices = _sample_indices(frames)
    sampled_root = root_xyzw[indices]
    norm_error = float(np.max(np.abs(np.linalg.norm(sampled_root, axis=1) - 1.0)))
    if norm_error > 1e-5:
        raise ValueError(f"Robot root_rot 单位范数误差 {norm_error}")
    pose_root_xyzw = Rotation.from_rotvec(pose_aa[indices, 0]).as_quat()
    root_pose_dot = np.abs(np.sum(pose_root_xyzw * sampled_root, axis=1))
    if float(np.min(root_pose_dot)) < 0.9999:
        raise ValueError(
            "Robot root_rot 不是与 pose_aa 根姿态一致的 xyzw 四元数: "
            f"min_abs_dot={float(np.min(root_pose_dot))}"
        )
    root_up = Rotation.from_quat(sampled_root).apply(np.array([0.0, 0.0, 1.0]))
    root_tilt = np.degrees(np.arccos(np.clip(root_up[:, 2], -1.0, 1.0)))
    stats = {
        "robot_frames": frames,
        "robot_fps": fps,
        "robot_sha256": _sha256(path),
        "root_quat_norm_error_max": norm_error,
        "root_pose_abs_dot_min": float(np.min(root_pose_dot)),
        "pose_dof_axis_error_max": pose_dof_axis_error_max,
        "robot_time_origin_seconds": time_origin,
        "robot_time_origin_mode": time_origin_mode,
        "root_tilt_median_degrees": float(np.median(root_tilt)),
        "root_tilt_gt45_ratio": float(np.mean(root_tilt > 45.0)),
        "root_height_median_m": float(np.median(root_trans[indices, 2])),
    }
    return robot, stats


def _load_smpl(path: Path) -> tuple[dict, dict]:
    """读取并验证单条 50 Hz SMPL PKL，记录关节人体轴供全局坐标检查。"""

    smpl = joblib.load(path)
    if not isinstance(smpl, dict):
        raise ValueError(f"SMPL 顶层必须是字典，实际为 {type(smpl).__name__}")
    required = ("pose_aa", "transl", "smpl_joints", "fps")
    missing = [field for field in required if field not in smpl]
    if missing:
        raise ValueError(f"SMPL 缺少字段: {missing}")
    pose_aa = np.asarray(smpl["pose_aa"], dtype=np.float64)
    transl = np.asarray(smpl["transl"], dtype=np.float64)
    joints = np.asarray(smpl["smpl_joints"], dtype=np.float64)
    frames = len(pose_aa)
    expected_shapes = {
        "pose_aa": (frames, 72),
        "transl": (frames, 3),
        "smpl_joints": (frames, SMPL_NUM_JOINTS, 3),
    }
    actual_shapes = {
        "pose_aa": pose_aa.shape,
        "transl": transl.shape,
        "smpl_joints": joints.shape,
    }
    if actual_shapes != expected_shapes:
        raise ValueError(f"SMPL shape 错误: actual={actual_shapes}, expected={expected_shapes}")
    if frames <= 1:
        raise ValueError(f"SMPL 帧数过短: {frames}")
    if not all(np.isfinite(value).all() for value in (pose_aa, transl, joints)):
        raise ValueError("SMPL 存在 NaN/Inf")
    fps = float(smpl["fps"])
    if not math.isclose(fps, TARGET_FPS, abs_tol=1e-8):
        raise ValueError(f"SMPL fps={fps}，预期 {TARGET_FPS}")
    time_origin, time_origin_mode = _declared_time_origin(smpl)

    indices = _sample_indices(frames)
    sampled_joints = joints[indices]
    extents = np.median(
        np.max(sampled_joints, axis=1) - np.min(sampled_joints, axis=1), axis=0
    )
    stats = {
        "smpl_frames": frames,
        "smpl_fps": fps,
        "smpl_sha256": _sha256(path),
        "smpl_joint_extent_xyz_median": extents.tolist(),
        "smpl_joint_dominant_axis": int(np.argmax(extents)),
        "smpl_time_origin_seconds": time_origin,
        "smpl_time_origin_mode": time_origin_mode,
    }
    return smpl, stats


def _declared_time_origin(data: dict) -> tuple[float, str]:
    """读取可选时间起点；没有时间字段时按数组第 0 帧即 0 秒的目录契约处理。"""

    scalar_fields = ("start_time", "start_time_sec", "time_offset", "time_offset_sec")
    for field in scalar_fields:
        if field in data:
            value = np.asarray(data[field], dtype=np.float64)
            if value.size != 1 or not np.isfinite(value).all():
                raise ValueError(f"时间起点字段 {field} 必须是有限标量")
            return float(value.reshape(-1)[0]), f"explicit:{field}"
    for field in ("timestamps", "times", "time"):
        if field in data:
            value = np.asarray(data[field], dtype=np.float64).reshape(-1)
            if value.size == 0 or not np.isfinite(value).all():
                raise ValueError(f"时间数组字段 {field} 必须是非空有限数组")
            if value.size > 1 and np.any(np.diff(value) <= 0.0):
                raise ValueError(f"时间数组字段 {field} 必须严格递增")
            return float(value[0]), f"explicit:{field}[0]"
    return 0.0, "implicit:index0=0s"


def _pair_median_degrees(robot: dict, smpl: dict) -> float:
    """复现训练端 waist 锚点与 SMPL 根姿态处理，计算整段逐帧中位角差。"""

    robot_frames = len(robot["root_rot"])
    smpl_frames = len(smpl["pose_aa"])
    common_frames = min(robot_frames, smpl_frames)
    # 配对中位角决定是否启用 SMPL Encoder，因此使用全部共同帧；
    # 仅描述数据集总体方向的根倾角和人体轴统计才允许确定性抽样。
    indices = np.arange(common_frames, dtype=np.int64)
    robot_root = Rotation.from_quat(np.asarray(robot["root_rot"], dtype=np.float64)[indices])
    robot_waist_local = Rotation.from_rotvec(
        np.asarray(robot["pose_aa"], dtype=np.float64)[indices, 1]
    )
    robot_waist = robot_root * robot_waist_local
    smpl_root_y_up = Rotation.from_rotvec(
        np.asarray(smpl["pose_aa"], dtype=np.float64)[indices, :3]
    )
    y_to_z = Rotation.from_rotvec(np.array([np.pi / 2.0, 0.0, 0.0]))
    smpl_base = Rotation.from_quat(np.array([0.5, 0.5, 0.5, 0.5]))
    processed_smpl_root = y_to_z * smpl_root_y_up * smpl_base.inv()
    differences = np.degrees((robot_waist.inv() * processed_smpl_root).magnitude())
    return float(np.median(differences))


def audit_record(record: SourceRecord) -> dict[str, Any]:
    """审计单条动作；Robot 错误标为致命，SMPL 错误降级为 Robot-only。"""

    row: dict[str, Any] = asdict(record)
    robot_path = Path(record.robot_path)
    try:
        robot, robot_stats = _load_robot(
            robot_path, expected_key=record.key, expected_fps=record.expected_robot_fps
        )
        row.update(robot_stats)
    except Exception as exc:  # noqa: BLE001
        row.update({"status": "ROBOT_INVALID", "reason": f"{type(exc).__name__}: {exc}"})
        return row

    if record.smpl_path is None:
        row.update({"status": "ROBOT_ONLY_NO_SMPL", "reason": "来源没有同名 SMPL"})
        return row

    smpl_path = Path(record.smpl_path)
    try:
        smpl, smpl_stats = _load_smpl(smpl_path)
        row.update(smpl_stats)
    except Exception as exc:  # noqa: BLE001
        row.update(
            {
                "status": "ROBOT_ONLY_SMPL_CONTRACT_MISMATCH",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return row

    frame_delta = int(row["smpl_frames"]) - int(row["robot_frames"])
    row["frame_delta_smpl_minus_robot"] = frame_delta
    time_origin_delta = float(row["smpl_time_origin_seconds"]) - float(
        row["robot_time_origin_seconds"]
    )
    row["time_origin_delta_smpl_minus_robot_seconds"] = time_origin_delta
    if not math.isclose(time_origin_delta, 0.0, abs_tol=1e-8):
        row.update(
            {
                "status": "ROBOT_ONLY_PAIR_TIME_ORIGIN_MISMATCH",
                "reason": f"Robot/SMPL 时间起点相差 {time_origin_delta} 秒",
            }
        )
        return row
    if not math.isclose(float(row["robot_fps"]), TARGET_FPS, abs_tol=1e-8):
        row.update(
            {
                "status": "ROBOT_ONLY_PAIR_FPS_MISMATCH",
                "reason": f"配对 Robot fps={row['robot_fps']} 不是 {TARGET_FPS}",
            }
        )
        return row
    if abs(frame_delta) > MAX_PAIR_FRAME_DELTA:
        row.update(
            {
                "status": "ROBOT_ONLY_PAIR_FRAME_MISMATCH",
                "reason": f"尾帧差 {frame_delta} 超过 {MAX_PAIR_FRAME_DELTA}",
            }
        )
        return row

    pair_median = _pair_median_degrees(robot, smpl)
    row["pair_median_degrees"] = pair_median
    if pair_median > MAX_PAIR_MEDIAN_DEGREES:
        row.update(
            {
                "status": "ROBOT_ONLY_SMPL_COORDINATE_MISMATCH",
                "reason": f"waist/SMPL 根姿态中位差 {pair_median:.6f} 度",
            }
        )
        return row
    row.update({"status": "PAIRED", "reason": None})
    return row


def _audit_records(records: list[SourceRecord], workers: int) -> list[dict[str, Any]]:
    """并行执行全量审计，按固定顺序返回并周期性打印进度。"""

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, row in enumerate(executor.map(audit_record, records, chunksize=16), start=1):
            rows.append(row)
            if index % 1000 == 0 or index == len(records):
                print(f"[三源审计] {index}/{len(records)}", flush=True)
    rows.sort(key=lambda row: (row["split"], row["source"], row["key"]))
    return rows


def _summarize(rows: list[dict[str, Any]], discovery: dict[str, Any]) -> dict[str, Any]:
    """汇总状态、自然采样比例和全局坐标统计，并执行系统性方向门禁。"""

    robot_invalid = [row for row in rows if row["status"] == "ROBOT_INVALID"]
    if robot_invalid:
        raise ValueError(
            f"存在 {len(robot_invalid)} 条无效 Robot，示例: {robot_invalid[:3]}"
        )

    split_rows = {
        split: [row for row in rows if row["split"] == split] for split in ("train", "test")
    }
    source_counts = Counter(row["source"] for row in split_rows["train"])
    train_total = len(split_rows["train"])
    natural_sampling = {
        source: count / train_total for source, count in sorted(source_counts.items())
    }
    status_counts = {
        split: dict(sorted(Counter(row["status"] for row in entries).items()))
        for split, entries in split_rows.items()
    }

    coordinate_summary: dict[str, Any] = {}
    for source in sorted({row["source"] for row in rows}):
        source_rows = [row for row in rows if row["source"] == source]
        tilt_medians = np.asarray(
            [row["root_tilt_median_degrees"] for row in source_rows], dtype=np.float64
        )
        tilt_ratios = np.asarray(
            [row["root_tilt_gt45_ratio"] for row in source_rows], dtype=np.float64
        )
        smpl_extents = [
            row["smpl_joint_extent_xyz_median"]
            for row in source_rows
            if "smpl_joint_extent_xyz_median" in row
        ]
        source_summary: dict[str, Any] = {
            "robot_root_tilt_clip_median_degrees": float(np.median(tilt_medians)),
            "robot_root_tilt_gt45_frame_ratio_mean": float(np.mean(tilt_ratios)),
        }
        if float(np.median(tilt_medians)) > 30.0 or float(np.mean(tilt_ratios)) > 0.2:
            raise ValueError(f"{source} Robot 出现系统性横躺: {source_summary}")
        if smpl_extents:
            extent_median = np.median(np.asarray(smpl_extents, dtype=np.float64), axis=0)
            source_summary["smpl_joint_extent_xyz_dataset_median"] = extent_median.tolist()
            source_summary["smpl_joint_dataset_dominant_axis"] = int(np.argmax(extent_median))
            if int(np.argmax(extent_median)) != 2:
                raise ValueError(f"{source} smpl_joints 不呈 Z-up 人体轴: {extent_median}")
        pair_medians = [
            row["pair_median_degrees"]
            for row in source_rows
            if row["status"] == "PAIRED"
        ]
        if pair_medians:
            source_summary["paired_root_median_degrees_median"] = float(
                np.median(pair_medians)
            )
            source_summary["paired_root_median_degrees_max"] = float(np.max(pair_medians))
        coordinate_summary[source] = source_summary

    return {
        "contract_version": CONTRACT_VERSION,
        "discovery": discovery,
        "train_robot_count": len(split_rows["train"]),
        "train_smpl_count": sum(row["status"] == "PAIRED" for row in split_rows["train"]),
        "train_robot_only_count": sum(
            row["status"] != "PAIRED" for row in split_rows["train"]
        ),
        "test_robot_count": len(split_rows["test"]),
        "test_smpl_count": sum(row["status"] == "PAIRED" for row in split_rows["test"]),
        "status_counts": status_counts,
        "source_counts": dict(sorted(source_counts.items())),
        "natural_sampling_probabilities": natural_sampling,
        "coordinate_summary": coordinate_summary,
    }


def _write_json(path: Path, payload: Any) -> None:
    """以稳定 UTF-8 格式写 JSON，便于哈希、diff 和人工审阅。"""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """按已排序顺序写逐动作 JSONL 清单。"""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _publish_index(
    output_root: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """在唯一 staging 目录建立软链接和报告，验证后原子发布到新路径。"""

    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output_root}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        for split in ("train", "test"):
            (staging / split / "robot_all").mkdir(parents=True)
            (staging / split / "smpl_all").mkdir(parents=True)
        (staging / "meta").mkdir()

        published_rows: list[dict[str, Any]] = []
        for row in rows:
            split = row["split"]
            robot_link = staging / split / "robot_all" / f"{row['key']}.pkl"
            robot_link.symlink_to(Path(row["robot_path"]).resolve())
            published = dict(row)
            published["robot_link"] = str(robot_link.relative_to(staging))
            published["smpl_link"] = None
            if row["status"] == "PAIRED":
                smpl_link = staging / split / "smpl_all" / f"{row['key']}.pkl"
                smpl_link.symlink_to(Path(row["smpl_path"]).resolve())
                published["smpl_link"] = str(smpl_link.relative_to(staging))
            published_rows.append(published)

        for split in ("train", "test"):
            _write_jsonl(
                staging / "meta" / f"{split}_manifest.jsonl",
                [row for row in published_rows if row["split"] == split],
            )
            # Adaptive sampling 初始化需要所有动作的帧数和帧率。若目录缺少 metadata.pkl，
            # 每个训练进程都会逐个打开近十万条源 PKL；这里从已审计结果生成最小索引，
            # 只缓存 length/fps，不复制或改写任何动作数组。
            motionlib_metadata = {
                row["key"]: {
                    "length": int(row["robot_frames"]),
                    "fps": float(row["robot_fps"]),
                }
                for row in published_rows
                if row["split"] == split
            }
            joblib.dump(
                motionlib_metadata,
                staging / split / "robot_all/metadata.pkl",
                compress=0,
            )
        _write_json(staging / "meta/summary.json", summary)
        provenance = {
            "contract_version": CONTRACT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_roots": {
                "large_root": str(args.large_root.resolve()),
                "hq4_root": str(args.hq4_root.resolve()),
                "mine_robot_dir": str(args.mine_robot_dir.resolve()),
            },
            "storage": "absolute_symlink_index_no_source_rewrite",
            "target_fps": TARGET_FPS,
            "paired_frame_alignment": {
                "mode": "trim_trailing",
                "max_frame_delta": MAX_PAIR_FRAME_DELTA,
            },
            "time_origin": (
                "同名数组均从第0帧开始；若PKL声明时间字段则要求两侧起点完全一致"
            ),
            "asset_contracts": summary["discovery"]["asset_contracts"],
            "smpl_coordinate_contract": {
                "pose_aa_and_transl": "Y-up",
                "smpl_joints": "Z-up",
                "runtime_root_conversion": "Y-up_to_Z-up_once_then_remove_smpl_base_rotation",
            },
            "sampling": "uniform_per_motion_natural_source_ratio",
            "motionlib_metadata": "generated_length_and_fps_only",
        }
        _write_json(staging / "meta/provenance.json", provenance)
        validate_index(staging, workers=args.workers, verify_hashes=False)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 清单并拒绝空行之外的非字典记录。"""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON 字典")
            rows.append(row)
    return rows


def _verify_manifest_hash(row: dict[str, Any]) -> tuple[str, str | None]:
    """重新计算一条清单的源文件哈希，供 validate 全量复核。"""

    robot_actual = _sha256(Path(row["robot_path"]))
    if robot_actual != row["robot_sha256"]:
        raise ValueError(f"{row['key']} Robot SHA256 漂移")
    smpl_actual = None
    if row["status"] == "PAIRED":
        smpl_actual = _sha256(Path(row["smpl_path"]))
        if smpl_actual != row["smpl_sha256"]:
            raise ValueError(f"{row['key']} SMPL SHA256 漂移")
    return robot_actual, smpl_actual


def validate_index(output_root: Path, workers: int, verify_hashes: bool) -> dict[str, Any]:
    """全量验证已发布索引的链接、清单计数、拆分隔离及可选源哈希。"""

    output_root = output_root.resolve()
    summary = json.loads((output_root / "meta/summary.json").read_text(encoding="utf-8"))
    train_rows = _load_manifest(output_root / "meta/train_manifest.jsonl")
    test_rows = _load_manifest(output_root / "meta/test_manifest.jsonl")
    all_rows = train_rows + test_rows
    if len(train_rows) != summary["train_robot_count"]:
        raise ValueError("train manifest 数量与 summary 不一致")
    if len(test_rows) != summary["test_robot_count"]:
        raise ValueError("test manifest 数量与 summary 不一致")
    train_keys = [row["key"] for row in train_rows]
    test_keys = [row["key"] for row in test_rows]
    if len(train_keys) != len(set(train_keys)) or len(test_keys) != len(set(test_keys)):
        raise ValueError("manifest 内存在重复 key")
    if set(train_keys) & set(test_keys):
        raise ValueError("manifest train/test 存在 key 交叉")

    for split, rows in (("train", train_rows), ("test", test_rows)):
        metadata_path = output_root / split / "robot_all/metadata.pkl"
        metadata = joblib.load(metadata_path)
        if not isinstance(metadata, dict) or set(metadata) != {row["key"] for row in rows}:
            raise ValueError(f"{split} MotionLib metadata key 与 manifest 不一致")
        for row in rows:
            entry = metadata[row["key"]]
            expected = {
                "length": int(row["robot_frames"]),
                "fps": float(row["robot_fps"]),
            }
            if entry != expected:
                raise ValueError(
                    f"{split}/{row['key']} MotionLib metadata 错误: {entry} != {expected}"
                )

    for row in all_rows:
        robot_link = output_root / row["robot_link"]
        if not robot_link.is_symlink() or robot_link.resolve() != Path(row["robot_path"]).resolve():
            raise ValueError(f"{row['key']} Robot 软链接目标错误")
        if row["status"] == "PAIRED":
            smpl_link = output_root / row["smpl_link"]
            if not smpl_link.is_symlink() or smpl_link.resolve() != Path(
                row["smpl_path"]
            ).resolve():
                raise ValueError(f"{row['key']} SMPL 软链接目标错误")
        elif row.get("smpl_link") is not None:
            raise ValueError(f"{row['key']} Robot-only 条目不应发布 SMPL 链接")

    actual_train_robot = len(
        [
            path
            for path in (output_root / "train/robot_all").glob("*.pkl")
            if path.name != "metadata.pkl"
        ]
    )
    actual_train_smpl = len(list((output_root / "train/smpl_all").glob("*.pkl")))
    actual_test_robot = len(
        [
            path
            for path in (output_root / "test/robot_all").glob("*.pkl")
            if path.name != "metadata.pkl"
        ]
    )
    actual_test_smpl = len(list((output_root / "test/smpl_all").glob("*.pkl")))
    actual_counts = {
        "train_robot_count": actual_train_robot,
        "train_smpl_count": actual_train_smpl,
        "test_robot_count": actual_test_robot,
        "test_smpl_count": actual_test_smpl,
    }
    for key, actual in actual_counts.items():
        if actual != summary[key]:
            raise ValueError(f"{key}={actual} 与 summary={summary[key]} 不一致")

    if verify_hashes:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, _ in enumerate(
                executor.map(_verify_manifest_hash, all_rows, chunksize=32), start=1
            ):
                if index % 1000 == 0 or index == len(all_rows):
                    print(f"[索引哈希复核] {index}/{len(all_rows)}", flush=True)
    result = {"contract_version": CONTRACT_VERSION, **actual_counts, "validate": "PASS"}
    print("BUMI3_THREE_SOURCE_VALIDATE=PASS " + json.dumps(result, sort_keys=True))
    return result


def build(args: argparse.Namespace) -> None:
    """发现来源、全量审计、建立软链接索引并原子发布。"""

    records, discovery = discover_records(args)
    rows = _audit_records(records, workers=args.workers)
    summary = _summarize(rows, discovery)
    _publish_index(args.output_root, rows, summary, args)
    print("BUMI3_THREE_SOURCE_BUILD=PASS " + json.dumps(summary, sort_keys=True))


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    """添加真实服务器默认路径和可供测试覆盖的来源计数参数。"""

    parser.add_argument("--large-root", type=Path, default=DEFAULT_LARGE_ROOT)
    parser.add_argument("--hq4-root", type=Path, default=DEFAULT_HQ4_ROOT)
    parser.add_argument("--mine-robot-dir", type=Path, default=DEFAULT_MINE_ROBOT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--expected-large-train", type=int, default=DEFAULT_EXPECTED_LARGE_TRAIN)
    parser.add_argument("--expected-large-test", type=int, default=DEFAULT_EXPECTED_LARGE_TEST)
    parser.add_argument("--expected-hq4-robot", type=int, default=DEFAULT_EXPECTED_HQ4_ROBOT)
    parser.add_argument("--expected-hq4-smpl", type=int, default=DEFAULT_EXPECTED_HQ4_SMPL)
    parser.add_argument("--expected-mine", type=int, default=DEFAULT_EXPECTED_MINE)


def _parse_args() -> argparse.Namespace:
    """解析 build/validate 子命令；validate 默认重新核对全部源哈希。"""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser(
        "build", help="审计三来源并原子发布软链接索引"
    )
    _add_source_arguments(build_parser)
    validate_parser = subparsers.add_parser("validate", help="全量复核已发布索引")
    validate_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate_parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    validate_parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="仅调试链接结构时跳过昂贵的源哈希复核；正式门禁不得使用",
    )
    return parser.parse_args()


def main() -> None:
    """执行命令并保持失败时非零退出码，供 tmux 和训练前门禁直接使用。"""

    args = _parse_args()
    if args.workers <= 0:
        raise ValueError("workers 必须大于 0")
    if args.command == "build":
        build(args)
    else:
        validate_index(
            args.output_root,
            workers=args.workers,
            verify_hashes=not args.skip_hash_verification,
        )


if __name__ == "__main__":
    main()
