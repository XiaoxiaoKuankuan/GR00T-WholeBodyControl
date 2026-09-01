#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""将经过人工筛选的 GENMO SMPL/BUMI3 动作整理为 SONIC 训练数据。

本工具专门实现 BUMI3 Robot+SMPL 双编码器的离线数据边界：机器人源文件保持
30Hz，依照每个源文件携带的 ``joint_names`` 重排到当前 BUMI3 MJCF 的 actuator
顺序，再生成 SONIC motion-lib 所需的 root、axis-angle、DoF 和四元数字段。公开
四库的 ``genmo.bumi_legacy_motion.v1`` 根姿态仍在 Y-up legacy frame 中，本工具
显式对根姿态做世界系 ``Rx(+90°)`` 左乘，并在当前 SONIC BUMI3 MJCF 下重新执行
足底高度优化；Mine 的 ``genmo.bumi_csv_qpos_xyzw.v1`` 已是 Z-up，根姿态保持
identity 修正，但其 ``legacy_body_origin_min_zero`` Root-Z 也按当前 MJCF 足底重新
对地，绝不套用公开库的姿态旋转。人体源文件则把 ``pose_aa``、``transl`` 和由
SONIC FK 计算的 ``smpl_joints`` 一起预重采样到 50Hz。SMPL 必须离线完成三项同步重采样，
因为当前 motion-lib 加载器只会自动重采样 SMPL pose，却会直接读取 joints/transl
并断言它们与机器人运行时 50Hz 帧数相等。

输入包含四个名称配对的数据集（AIST++、AIOZ-GDANCE、FineDance、CoMPAS3D）
以及仅有机器人动作的 Mine 数据集。输出使用 ``dataset__sample_id`` 扁平命名，
机器人文件是单 key 的 joblib 字典，SMPL 文件是 SONIC 直接读取的字段字典。构建
先写隐藏 staging 目录，完成全量有限值、维度、配对和帧数校验后才原子切换为
``built/robot_all`` 与 ``built/smpl_all``，同时保存 manifest、来源指纹和 SHA256。

典型用法：

    python gear_sonic/tools/prepare_bumi3_sonic_dataset.py build \
      --smpl-root /data/sonic_bumi3/datasets/hq_all_v1/source_smpl \
      --bumi-root /data/sonic_bumi3/datasets/hq_all_v1/source_bumi \
      --mine-root /data/sonic_bumi3/datasets/hq_all_v1/source_mine \
      --output-root /data/sonic_bumi3/datasets/hq_all_v2

    python gear_sonic/tools/prepare_bumi3_sonic_dataset.py validate \
      --output-root /data/sonic_bumi3/datasets/hq_all_v2
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import torch


SOURCE_FPS = 30.0
TARGET_FPS = 50.0
EXPECTED_ROBOT = 3261
EXPECTED_SMPL = 3162
EXPECTED_MINE = 99
EXPECTED_SOURCE_MJCF_SHA256 = "482138b437dbdabd6171fa8d44b55db5d7125a228c95b69ce3d1159cafe8537c"
DATASET_DIRS = {
    "aistpp": "AIST++",
    "aioz_gdance": "AIOZ-GDANCE",
    "finedance": "FineDance",
    "compas3d": "CoMPAS3D",
}

LEGACY_PUBLIC_ROOT_CONTRACT = "genmo.bumi_legacy_motion.v1"
MINE_ROOT_CONTRACT = "genmo.bumi_csv_qpos_xyzw.v1"
SONIC_ROOT_FRAME_CONTRACT = "sonic.bumi3.root_frame.z_up.v1"
# 世界系左乘 Rx(+90°)：把 legacy world +Y 上轴映射到 SONIC/MuJoCo world +Z。
LEGACY_PUBLIC_ROOT_CORRECTION_WXYZ = (
    math.sqrt(0.5),
    math.sqrt(0.5),
    0.0,
    0.0,
)
IDENTITY_ROOT_CORRECTION_WXYZ = (1.0, 0.0, 0.0, 0.0)
FOOT_BODY_NAMES = ("l_ankle_roll_link", "r_ankle_roll_link")
ROOT_TILT_FRAME_THRESHOLD_DEGREES = 45.0
ROOT_TILT_DATASET_MEDIAN_MAX_DEGREES = 30.0
ROOT_TILT_DATASET_EXCEED_RATIO_MAX = 0.20
SOLE_PENETRATION_TOLERANCE_M = 0.002


_MUJOCO_FOOT_CACHE: dict[
    str,
    tuple[Any, Any, tuple[int, ...], tuple[tuple[int, ...], ...], int],
] = {}


@dataclass(frozen=True)
class SampleRecord:
    """一段训练动作的来源、配对关系及目标文件契约。"""

    key: str
    dataset: str
    sample_id: str
    split: str
    robot_source: str
    smpl_source: str | None
    smpl_motion_key: str | None
    source_frames: int
    source_fps: float
    paired: bool


@dataclass(frozen=True)
class MjcfContract:
    """当前 SONIC MJCF 中与 motion-lib 顺序有关的最小稳定契约。"""

    actuator_joint_names: tuple[str, ...]
    body_joint_names: tuple[str, ...]
    body_joint_axes: tuple[tuple[float, float, float], ...]
    sha256: str
    mjcf_path: str | None = None


_AIST_CACHE: dict[str, dict[str, Any]] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalise_dataset(value: str) -> str:
    name = value.strip().lower().replace("-", "_").replace("++", "pp")
    if name.endswith("_bumi"):
        name = name[: -len("_bumi")]
    aliases = {
        "aist": "aistpp",
        "aistpp": "aistpp",
        "aioz_gdance": "aioz_gdance",
        "finedance": "finedance",
        "compas3d": "compas3d",
        "mine": "mine",
    }
    if name not in aliases:
        raise ValueError(f"不支持的数据集名称: {value!r}")
    return aliases[name]


def _key(dataset: str, sample_id: str) -> str:
    if "/" in sample_id or "\\" in sample_id or sample_id in {"", ".", ".."}:
        raise ValueError(f"非法 sample_id: {sample_id!r}")
    return f"{dataset}__{sample_id}"


def _parse_mjcf(path: Path) -> MjcfContract:
    tree = ET.parse(path)
    root = tree.getroot()
    world_root = root.find("worldbody/body")
    if world_root is None:
        raise ValueError(f"MJCF 缺少 worldbody 根 body: {path}")

    body_joint_names: list[str] = []
    body_joint_axes: list[tuple[float, float, float]] = []

    def visit(body: ET.Element) -> None:
        joints = [joint for joint in body.findall("joint") if joint.get("type") != "free"]
        if len(joints) > 1:
            raise ValueError(f"BUMI3 body {body.get('name')} 含多个非 free joint")
        if joints:
            joint = joints[0]
            name = joint.get("name")
            axis_text = joint.get("axis")
            if name is None or axis_text is None:
                raise ValueError(f"BUMI3 joint 缺少 name/axis: {ET.tostring(joint)}")
            axis = tuple(float(value) for value in axis_text.split())
            if len(axis) != 3 or not math.isclose(sum(v * v for v in axis), 1.0, abs_tol=1e-6):
                raise ValueError(f"BUMI3 joint {name} axis 非单位三维向量: {axis}")
            body_joint_names.append(name)
            body_joint_axes.append(axis)
        for child in body.findall("body"):
            visit(child)

    visit(world_root)
    actuator_names = tuple(
        motor.get("joint") for motor in root.findall("actuator/motor") if motor.get("joint")
    )
    if len(actuator_names) != 21 or len(body_joint_names) != 21:
        raise ValueError(
            f"BUMI3 MJCF 必须是 21 DoF，得到 actuator={len(actuator_names)}, "
            f"body_joint={len(body_joint_names)}"
        )
    if set(actuator_names) != set(body_joint_names) or len(set(actuator_names)) != 21:
        raise ValueError("BUMI3 actuator 与 body joint 名称集合不一致或有重复")
    return MjcfContract(
        actuator_joint_names=actuator_names,
        body_joint_names=tuple(body_joint_names),
        body_joint_axes=tuple(body_joint_axes),
        sha256=_sha256(path),
        mjcf_path=str(path.resolve()),
    )


def _quat_mul_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """批量计算 scalar-first ``wxyz`` 四元数乘法。"""

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _root_tilt_degrees_wxyz(root_quat_wxyz: np.ndarray) -> np.ndarray:
    """计算根局部 +Z 与世界 +Z 的夹角；yaw 不影响该指标。"""

    quat = np.asarray(root_quat_wxyz, dtype=np.float64)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"root quaternion 必须为 [T,4]，实际 {quat.shape}")
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if not np.isfinite(quat).all() or np.any(norms < 1e-8):
        raise ValueError("root quaternion 含 NaN/Inf 或零范数")
    quat = quat / norms
    x = quat[:, 1]
    y = quat[:, 2]
    local_up_world_z = 1.0 - 2.0 * (x * x + y * y)
    return np.rad2deg(np.arccos(np.clip(local_up_world_z, -1.0, 1.0)))


def _resolve_root_frame(
    payload: dict[str, Any],
    record: SampleRecord,
    root_quat_wxyz: torch.Tensor,
) -> tuple[torch.Tensor, tuple[float, float, float, float], str]:
    """按源契约把根姿态解析到 SONIC Z-up；禁止按统计结果猜测。"""

    source_contract = str(payload.get("source_motion_contract_version", ""))
    if record.dataset == "mine":
        expected_contract = MINE_ROOT_CONTRACT
        correction_values = IDENTITY_ROOT_CORRECTION_WXYZ
        height_policy = "mine_csv_current_mjcf_sole_optimized"
    else:
        expected_contract = LEGACY_PUBLIC_ROOT_CONTRACT
        correction_values = LEGACY_PUBLIC_ROOT_CORRECTION_WXYZ
        height_policy = "legacy_public_current_mjcf_sole_optimized"
    if source_contract != expected_contract:
        raise ValueError(
            f"{record.key} root frame 契约错误: expected={expected_contract!r}, "
            f"actual={source_contract!r}"
        )

    correction = torch.tensor(
        correction_values,
        dtype=root_quat_wxyz.dtype,
        device=root_quat_wxyz.device,
    ).expand_as(root_quat_wxyz)
    corrected = _quat_mul_wxyz(correction, root_quat_wxyz)
    corrected = corrected / torch.linalg.vector_norm(corrected, dim=1, keepdim=True)
    return corrected.contiguous(), correction_values, height_policy


def _get_mujoco_foot_resources(
    contract: MjcfContract,
) -> tuple[Any, Any, tuple[int, ...], tuple[tuple[int, ...], ...], int]:
    """加载当前 BUMI3 MJCF 和足部/地面 geom id；每个 worker 只加载一次。"""

    if contract.mjcf_path is None:
        raise ValueError("MjcfContract 缺少 mjcf_path，无法执行足底 FK")
    cache_key = f"{contract.mjcf_path}:{contract.sha256}"
    cached = _MUJOCO_FOOT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import mujoco
    model = mujoco.MjModel.from_xml_path(contract.mjcf_path)
    data = mujoco.MjData(model)
    joint_qpos_addresses: list[int] = []
    for joint_name in contract.actuator_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"BUMI3 MJCF 缺少 hinge joint: {joint_name}")
        joint_qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

    foot_geom_ids: list[tuple[int, ...]] = []
    for body_name in FOOT_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"BUMI3 MJCF 缺少 foot body: {body_name}")
        geom_ids = tuple(
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
        )
        if not geom_ids:
            raise ValueError(f"{body_name} 没有可用于足底审计的 geom")
        foot_geom_ids.append(geom_ids)

    ground_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if ground_geom_id < 0 or model.geom_type[ground_geom_id] != mujoco.mjtGeom.mjGEOM_PLANE:
        raise ValueError("BUMI3 MJCF 缺少名为 ground 的 plane geom")

    resources = (
        model,
        data,
        tuple(joint_qpos_addresses),
        tuple(foot_geom_ids),
        ground_geom_id,
    )
    _MUJOCO_FOOT_CACHE[cache_key] = resources
    return resources


def _compute_foot_fk_metrics(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    dof: np.ndarray,
    contract: MjcfContract,
    *,
    frame_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """在当前 SONIC BUMI3 MJCF 下计算足底最低点和脚 body 原点高度。"""

    import mujoco

    root = np.asarray(root_pos, dtype=np.float64)
    quat = np.asarray(root_quat_wxyz, dtype=np.float64)
    joints = np.asarray(dof, dtype=np.float64)
    if root.shape != (len(quat), 3) or joints.shape != (len(quat), 21):
        raise ValueError(
            f"foot FK 输入 shape 错误: root={root.shape}, quat={quat.shape}, dof={joints.shape}"
        )
    if frame_indices is None:
        indices = np.arange(len(root), dtype=np.int64)
    else:
        indices = np.asarray(frame_indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(root)):
            raise ValueError("foot FK frame_indices 越界或维度错误")

    model, data, joint_qpos_addresses, foot_geom_ids, ground_geom_id = (
        _get_mujoco_foot_resources(contract)
    )
    body_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in FOOT_BODY_NAMES
    )
    sole_z = np.empty((len(indices), len(FOOT_BODY_NAMES)), dtype=np.float64)
    foot_origin_z = np.empty_like(sole_z)
    nearest_points = np.empty(6, dtype=np.float64)
    for output_index, frame_index in enumerate(indices):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = root[frame_index]
        data.qpos[3:7] = quat[frame_index]
        for joint_index, qpos_address in enumerate(joint_qpos_addresses):
            data.qpos[qpos_address] = joints[frame_index, joint_index]
        mujoco.mj_forward(model, data)
        for foot_index, (body_id, geom_ids) in enumerate(
            zip(body_ids, foot_geom_ids, strict=True)
        ):
            distances = [
                mujoco.mj_geomDistance(
                    model,
                    data,
                    ground_geom_id,
                    geom_id,
                    10.0,
                    nearest_points,
                )
                for geom_id in geom_ids
            ]
            sole_z[output_index, foot_index] = float(min(distances))
            foot_origin_z[output_index, foot_index] = float(data.xpos[body_id, 2])
    return sole_z, foot_origin_z


def _remove_short_contact_segments(mask: np.ndarray, min_frames: int = 3) -> np.ndarray:
    """删除短于 ``min_frames`` 的接触脉冲，避免 Root-Z 追逐单帧噪声。"""

    output = np.asarray(mask, dtype=bool).copy()
    start: int | None = None
    for index in range(len(output) + 1):
        active = index < len(output) and bool(output[index])
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start < min_frames:
                output[start:index] = False
            start = None
    return output


def _detect_foot_contacts(sole_z: np.ndarray, fps: float) -> np.ndarray:
    """以每只脚的稳健低位和垂直速度检测接触，仅用于 Root-Z 软目标。"""

    references = np.percentile(sole_z, 15.0, axis=0)
    vertical_speed = np.abs(np.gradient(sole_z, axis=0) * fps)
    contacts = np.zeros_like(sole_z, dtype=bool)
    for foot_index in range(sole_z.shape[1]):
        active = False
        for frame_index in range(len(sole_z)):
            height = sole_z[frame_index, foot_index]
            speed = vertical_speed[frame_index, foot_index]
            if not active:
                active = bool(
                    height <= references[foot_index] + 0.02 and speed <= 0.10
                )
            elif height > references[foot_index] + 0.04 or speed > 0.20:
                active = False
            contacts[frame_index, foot_index] = active
        contacts[:, foot_index] = _remove_short_contact_segments(
            contacts[:, foot_index]
        )
    return contacts


def _optimize_root_height_for_current_mjcf(
    root_pos: torch.Tensor,
    root_quat_wxyz: torch.Tensor,
    dof: torch.Tensor,
    contract: MjcfContract,
    fps: float,
    policy: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """在当前 BUMI3 足底 FK 下重建 Root-Z，同时保留腾空帧。"""

    from scipy.optimize import minimize

    root = root_pos.detach().cpu().numpy().astype(np.float64, copy=True)
    quat = root_quat_wxyz.detach().cpu().numpy().astype(np.float64, copy=False)
    joints = dof.detach().cpu().numpy().astype(np.float64, copy=False)
    raw_sole_z, foot_origin_z = _compute_foot_fk_metrics(root, quat, joints, contract)
    robust_reference = float(np.percentile(np.min(raw_sole_z, axis=1), 15.0))
    constant_shift = -robust_reference
    aligned_root_z = root[:, 2] + constant_shift
    aligned_sole_z = raw_sole_z + constant_shift
    contacts = _detect_foot_contacts(aligned_sole_z, fps)

    lower_bound = np.maximum(
        -0.75,
        -SOLE_PENETRATION_TOLERANCE_M - np.min(aligned_sole_z, axis=1),
    )
    upper_bound = np.full(len(root), 0.75, dtype=np.float64)
    if np.any(lower_bound > upper_bound):
        worst = float(np.max(lower_bound - upper_bound))
        raise RuntimeError(f"{contract.mjcf_path} Root-Z 无穿地约束不可行，缺口 {worst:.6f} m")

    contact_count = contacts.sum(axis=1).astype(np.float64)
    contact_target_sum = np.sum(contacts * aligned_sole_z, axis=1)

    def objective(correction: np.ndarray) -> tuple[float, np.ndarray]:
        value = 0.5 * float(np.dot(correction, correction))
        gradient = correction.copy()
        if len(correction) > 1:
            first = np.diff(correction)
            value += 10.0 * float(np.dot(first, first))
            gradient[:-1] -= 20.0 * first
            gradient[1:] += 20.0 * first
        if len(correction) > 2:
            # 只平滑新增修正，不抹平源动作本身的起跳/落地 Root-Z 加速度。
            second = np.diff(correction, n=2)
            value += 500.0 * float(np.dot(second, second))
            gradient[:-2] += 1000.0 * second
            gradient[1:-1] -= 2000.0 * second
            gradient[2:] += 1000.0 * second
        if contacts.any():
            value += 40.0 * float(
                np.sum(np.square((aligned_sole_z + correction[:, None])[contacts]))
            )
            gradient += 80.0 * (
                contact_count * correction + contact_target_sum
            )
        return value, gradient

    initial = np.clip(np.zeros(len(root), dtype=np.float64), lower_bound, upper_bound)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=list(zip(lower_bound, upper_bound, strict=True)),
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Root-Z 优化失败: {result.message}")
    dynamic_correction = np.clip(result.x, lower_bound, upper_bound)
    root[:, 2] = aligned_root_z + dynamic_correction
    final_sole_z = aligned_sole_z + dynamic_correction[:, None]
    max_penetration = float(np.maximum(-final_sole_z, 0.0).max(initial=0.0))
    if max_penetration > SOLE_PENETRATION_TOLERANCE_M + 1e-6:
        raise RuntimeError(
            f"Root-Z 优化后足底穿透 {max_penetration:.6f} m 超过 "
            f"{SOLE_PENETRATION_TOLERANCE_M:.6f} m"
        )
    contact_error = np.abs(final_sole_z[contacts])
    correction_speed = np.diff(dynamic_correction) * fps
    correction_acceleration = np.diff(dynamic_correction, n=2) * fps**2
    diagnostics = {
        "policy": policy,
        "constant_shift_m": constant_shift,
        "max_dynamic_correction_m": float(np.max(np.abs(dynamic_correction), initial=0.0)),
        "dynamic_correction_rms_m": float(np.sqrt(np.mean(np.square(dynamic_correction)))),
        "max_dynamic_correction_speed_mps": float(
            np.max(np.abs(correction_speed), initial=0.0)
        ),
        "max_dynamic_correction_acceleration_mps2": float(
            np.max(np.abs(correction_acceleration), initial=0.0)
        ),
        "minimum_sole_height_m": float(np.min(final_sole_z)),
        "max_sole_penetration_m": max_penetration,
        "contact_frames_per_foot": contacts.sum(axis=0).astype(int).tolist(),
        "contact_height_error_p95_m": float(
            np.percentile(contact_error, 95) if contact_error.size else 0.0
        ),
        "root_to_foot_origin_z_median_m": float(
            np.median(foot_origin_z - root_pos.detach().cpu().numpy()[:, 2:3])
        ),
        "optimizer_success": True,
    }
    return torch.from_numpy(root).to(dtype=root_pos.dtype), diagnostics


def _collect_robot_records(bumi_root: Path, mine_root: Path) -> dict[str, SampleRecord]:
    records: dict[str, SampleRecord] = {}
    for dataset, dirname in DATASET_DIRS.items():
        dataset_root = bumi_root / dirname
        for split in ("train", "val", "test"):
            manifest = dataset_root / "manifests" / f"{split}.jsonl"
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            for row in _jsonl(manifest):
                if not bool(row.get("quality_accepted", True)):
                    raise ValueError(f"manifest 含未通过质量筛选的动作: {manifest} {row}")
                sample_id = str(row["sample_id"])
                source = (dataset_root / row["motion_path"]).resolve()
                key = _key(dataset, sample_id)
                if key in records:
                    raise ValueError(f"重复机器人 key: {key}")
                records[key] = SampleRecord(
                    key=key,
                    dataset=dataset,
                    sample_id=sample_id,
                    split=split,
                    robot_source=str(source),
                    smpl_source=None,
                    smpl_motion_key=None,
                    source_frames=int(row["num_frames"]),
                    source_fps=float(row["fps"]),
                    paired=False,
                )

    mine_manifest = mine_root / "manifests" / "train.jsonl"
    if not mine_manifest.is_file():
        raise FileNotFoundError(mine_manifest)
    for row in _jsonl(mine_manifest):
        if not bool(row.get("quality_accepted", True)):
            raise ValueError(f"Mine manifest 含未通过质量筛选的动作: {row}")
        sample_id = str(row["sample_id"])
        key = _key("mine", sample_id)
        if key in records:
            raise ValueError(f"重复 Mine key: {key}")
        records[key] = SampleRecord(
            key=key,
            dataset="mine",
            sample_id=sample_id,
            split=str(row.get("split", "train")),
            robot_source=str((mine_root / row["motion_path"]).resolve()),
            smpl_source=None,
            smpl_motion_key=None,
            source_frames=int(row["num_frames"]),
            source_fps=float(row["fps"]),
            paired=False,
        )
    return records


def _attach_smpl(records: dict[str, SampleRecord], smpl_root: Path) -> dict[str, SampleRecord]:
    accepted_path = smpl_root / "reports" / "accepted_master.jsonl"
    if not accepted_path.is_file():
        raise FileNotFoundError(accepted_path)
    seen: set[str] = set()
    for row in _jsonl(accepted_path):
        if row.get("decision") != "keep":
            continue
        dataset = _normalise_dataset(str(row["dataset"]))
        sample_id = str(row["sample_id"])
        key = _key(dataset, sample_id)
        if key in seen:
            raise ValueError(f"重复 SMPL key: {key}")
        seen.add(key)
        if key not in records:
            raise ValueError(f"SMPL 动作没有对应 BUMI3 动作: {key}")
        dataset_root = smpl_root / DATASET_DIRS[dataset]
        source = (dataset_root / str(row["motion_path"])).resolve()
        old = records[key]
        if old.source_frames != int(row["num_frames"]):
            raise ValueError(
                f"配对源帧数不一致 {key}: robot={old.source_frames}, smpl={row['num_frames']}"
            )
        if old.split != str(row["split"]):
            raise ValueError(f"配对 split 不一致 {key}: robot={old.split}, smpl={row['split']}")
        records[key] = SampleRecord(
            **{
                **asdict(old),
                "smpl_source": str(source),
                "smpl_motion_key": str(row.get("motion_key") or sample_id),
                "paired": True,
            }
        )
    return records


def _validate_record_counts(records: dict[str, SampleRecord]) -> None:
    paired = sum(record.paired for record in records.values())
    mine = sum(record.dataset == "mine" for record in records.values())
    if len(records) != EXPECTED_ROBOT or paired != EXPECTED_SMPL or mine != EXPECTED_MINE:
        raise ValueError(
            "训练集合计数不符合已审计契约: "
            f"robot={len(records)} (expected {EXPECTED_ROBOT}), "
            f"paired={paired} (expected {EXPECTED_SMPL}), "
            f"mine={mine} (expected {EXPECTED_MINE})"
        )
    unmatched = [record.key for record in records.values() if not record.paired]
    if any(not key.startswith("mine__") for key in unmatched):
        raise ValueError(f"公共数据集出现未配对动作: {unmatched[:10]}")


def _load_pt(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _convert_robot_job(args: tuple[SampleRecord, MjcfContract, str]) -> str:
    record, contract, output_dir_text = args
    torch.set_num_threads(1)
    source = Path(record.robot_source)
    payload = _load_pt(source)
    required = {
        "contract_version",
        "source_motion_contract_version",
        "qpos",
        "fps",
        "robot_name",
        "joint_names",
        "quaternion_convention",
        "qpos_order",
        "source_dataset",
        "source_sample_id",
        "quality_accepted",
        "source_mjcf_sha256",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"{source} 缺少字段: {missing}")
    if payload["contract_version"] != "genmo.bumi_music.v1":
        raise ValueError(f"{source} contract_version 错误: {payload['contract_version']}")
    if payload["quaternion_convention"] != "wxyz" or payload["qpos_order"] != "mujoco_native":
        raise ValueError(f"{source} 四元数或 qpos 顺序契约错误")
    if payload["robot_name"] != "bumi":
        raise ValueError(f"{source} robot_name 应为 'bumi'，实际 {payload['robot_name']!r}")
    if payload["source_mjcf_sha256"] != EXPECTED_SOURCE_MJCF_SHA256:
        raise ValueError(
            f"{source} source_mjcf_sha256 漂移: {payload['source_mjcf_sha256']}"
        )
    if not bool(payload["quality_accepted"]):
        raise ValueError(f"{source} 未通过质量筛选")
    if _normalise_dataset(str(payload["source_dataset"])) != record.dataset:
        raise ValueError(f"{source} source_dataset 与 manifest 不一致")
    if str(payload["source_sample_id"]) != record.sample_id:
        raise ValueError(f"{source} source_sample_id 与 manifest 不一致")

    qpos = torch.as_tensor(payload["qpos"], dtype=torch.float32).contiguous()
    if qpos.ndim != 2 or qpos.shape[1] != 28 or qpos.shape[0] != record.source_frames:
        raise ValueError(f"{source} qpos 应为 ({record.source_frames},28)，实际 {tuple(qpos.shape)}")
    if not torch.isfinite(qpos).all():
        raise ValueError(f"{source} qpos 含 NaN/Inf")
    fps = float(payload["fps"])
    if not math.isclose(fps, SOURCE_FPS, abs_tol=1e-6):
        raise ValueError(f"{source} fps 必须为 30，实际 {fps}")

    source_names = tuple(str(name) for name in payload["joint_names"])
    if len(source_names) != 21 or len(set(source_names)) != 21:
        raise ValueError(f"{source} joint_names 必须含 21 个唯一名称")
    if set(source_names) != set(contract.actuator_joint_names):
        missing_names = sorted(set(contract.actuator_joint_names) - set(source_names))
        extra_names = sorted(set(source_names) - set(contract.actuator_joint_names))
        raise ValueError(
            f"{source} joint_names 与 SONIC MJCF 不一致: "
            f"missing={missing_names}, extra={extra_names}"
        )
    source_index = {name: index for index, name in enumerate(source_names)}
    dof = torch.stack(
        [qpos[:, 7 + source_index[name]] for name in contract.actuator_joint_names], dim=1
    )

    root_quat_wxyz = qpos[:, 3:7]
    norms = torch.linalg.vector_norm(root_quat_wxyz, dim=1, keepdim=True)
    if bool((norms < 1e-8).any()):
        raise ValueError(f"{source} 含零范数 root quaternion")
    root_quat_wxyz = root_quat_wxyz / norms
    root_quat_wxyz, root_frame_correction, root_height_policy = _resolve_root_frame(
        payload,
        record,
        root_quat_wxyz,
    )
    root_trans_offset = qpos[:, :3].clone()
    root_trans_offset, root_height_diagnostics = _optimize_root_height_for_current_mjcf(
        root_trans_offset,
        root_quat_wxyz,
        dof,
        contract,
        fps,
        root_height_policy,
    )
    from gear_sonic.trl.utils.torch_transform import quaternion_to_angle_axis

    root_axis_angle = quaternion_to_angle_axis(root_quat_wxyz)
    pose_aa = torch.zeros((qpos.shape[0], 22, 3), dtype=torch.float32)
    pose_aa[:, 0] = root_axis_angle
    actuator_index = {name: index for index, name in enumerate(contract.actuator_joint_names)}
    for body_index, (joint_name, axis) in enumerate(
        zip(contract.body_joint_names, contract.body_joint_axes, strict=True), start=1
    ):
        pose_aa[:, body_index] = (
            torch.tensor(axis, dtype=torch.float32) * dof[:, actuator_index[joint_name], None]
        )

    entry = {
        "root_trans_offset": root_trans_offset.numpy().astype(np.float32, copy=False),
        "pose_aa": pose_aa.numpy().astype(np.float32, copy=False),
        "dof": dof.numpy().astype(np.float32, copy=False),
        "root_rot": root_quat_wxyz[:, [1, 2, 3, 0]].numpy().astype(np.float32, copy=False),
        "fps": 30,
        "source_motion_contract_version": str(payload["source_motion_contract_version"]),
        "root_frame_contract_version": SONIC_ROOT_FRAME_CONTRACT,
        "root_frame_correction_wxyz": list(root_frame_correction),
        "root_height_policy": root_height_policy,
        "root_height_diagnostics": root_height_diagnostics,
    }
    output = Path(output_dir_text) / f"{record.key}.pkl"
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    joblib.dump({record.key: entry}, temporary, compress=3)
    os.replace(temporary, output)
    return record.key


def _target_time_grid(num_frames: int, source_fps: float, target_fps: float) -> torch.Tensor:
    if num_frames < 2:
        raise ValueError(f"动作至少需要 2 帧，实际 {num_frames}")
    duration = (num_frames - 1) * 1.0 / source_fps
    return torch.arange(0, duration, 1.0 / target_fps, dtype=torch.float32)


def _resample_pose_and_translation(
    pose_aa: torch.Tensor,
    transl: torch.Tensor,
    source_fps: float,
    target_fps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pose_aa.ndim != 2 or pose_aa.shape[1] != 72:
        raise ValueError(f"SMPL pose_aa 必须是 [T,72]，实际 {tuple(pose_aa.shape)}")
    if transl.shape != (pose_aa.shape[0], 3):
        raise ValueError(f"SMPL transl 与 pose 帧数不一致: {tuple(transl.shape)}")
    if math.isclose(source_fps, target_fps, abs_tol=1e-6):
        return pose_aa.contiguous(), transl.contiguous()

    from gear_sonic.isaac_utils.rotations import slerp
    from gear_sonic.trl.utils.torch_transform import (
        angle_axis_to_quaternion,
        quaternion_to_angle_axis,
    )

    times = _target_time_grid(len(pose_aa), source_fps, target_fps)
    duration = (len(pose_aa) - 1) * 1.0 / source_fps
    phase = times / duration
    index_0 = torch.floor(phase * (len(pose_aa) - 1)).long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(len(pose_aa) - 1))
    blend = phase * (len(pose_aa) - 1) - index_0

    pose_quat = angle_axis_to_quaternion(pose_aa.reshape(-1, 3)).reshape(len(pose_aa), 24, 4)
    pose_quat = slerp(
        pose_quat[index_0],
        pose_quat[index_1],
        blend[:, None, None],
    )
    pose_out = quaternion_to_angle_axis(pose_quat.reshape(-1, 4)).reshape(-1, 72)
    transl_out = transl[index_0] * (1.0 - blend[:, None]) + transl[index_1] * blend[:, None]
    return pose_out.contiguous(), transl_out.contiguous()


def _load_smpl_source(record: SampleRecord) -> tuple[torch.Tensor, torch.Tensor, float]:
    source = Path(record.smpl_source or "")
    if record.dataset == "aistpp":
        cache_key = str(source)
        if cache_key not in _AIST_CACHE:
            aggregate = _load_pt(source)
            if not isinstance(aggregate, dict):
                raise ValueError(f"AIST++ 聚合文件不是字典: {source}")
            _AIST_CACHE[cache_key] = aggregate
        aggregate = _AIST_CACHE[cache_key]
        motion_key = record.smpl_motion_key or record.sample_id
        if motion_key not in aggregate:
            raise KeyError(f"AIST++ 聚合文件缺少 {motion_key}")
        item = aggregate[motion_key]
        pose = torch.as_tensor(item["smpl_pose_global"], dtype=torch.float32)
        transl = torch.as_tensor(item["smpl_trans_global"], dtype=torch.float32)
        fps = SOURCE_FPS
    else:
        item = _load_pt(source)
        global_orient = torch.as_tensor(item["global_orient"], dtype=torch.float32).reshape(-1, 3)
        body_pose = torch.as_tensor(item["body_pose"], dtype=torch.float32).reshape(-1, 63)
        zeros = torch.zeros((len(body_pose), 6), dtype=torch.float32)
        pose = torch.cat([global_orient, body_pose, zeros], dim=1)
        transl = torch.as_tensor(item["transl"], dtype=torch.float32).reshape(-1, 3)
        fps = float(item["fps"])
    pose = pose.reshape(-1, 72).contiguous()
    transl = transl.reshape(-1, 3).contiguous()
    if len(pose) != record.source_frames or len(transl) != record.source_frames:
        raise ValueError(
            f"{record.key} SMPL 源帧数错误: pose={len(pose)}, transl={len(transl)}, "
            f"expected={record.source_frames}"
        )
    if not torch.isfinite(pose).all() or not torch.isfinite(transl).all():
        raise ValueError(f"{record.key} SMPL 源数据含 NaN/Inf")
    return pose, transl, fps


def _compute_smpl_joints(pose_aa: torch.Tensor, human_info_path: str) -> torch.Tensor:
    from gear_sonic.isaac_utils.rotations import smpl_root_ytoz_up
    from gear_sonic.trl.utils.torch_transform import (
        angle_axis_to_quaternion,
        compute_human_joints,
        quaternion_to_angle_axis,
    )

    outputs: list[torch.Tensor] = []
    for start in range(0, len(pose_aa), 8192):
        chunk = pose_aa[start : start + 8192]
        root_quat_y = angle_axis_to_quaternion(chunk[:, :3])
        root_quat_z = smpl_root_ytoz_up(root_quat_y)
        root_axis_angle_z = quaternion_to_angle_axis(root_quat_z)
        joints = compute_human_joints(
            body_pose=chunk[:, 3:66],
            global_orient=root_axis_angle_z,
            human_joints_info_path=human_info_path,
        )
        outputs.append(joints.cpu())
    return torch.cat(outputs, dim=0).contiguous()


def _convert_smpl_job(args: tuple[SampleRecord, str, str]) -> str:
    record, human_info_path, output_dir_text = args
    torch.set_num_threads(1)
    pose, transl, fps = _load_smpl_source(record)
    if not math.isclose(fps, SOURCE_FPS, abs_tol=1e-6):
        raise ValueError(f"{record.key} SMPL fps 必须为 30，实际 {fps}")
    pose_50, transl_50 = _resample_pose_and_translation(pose, transl, fps, TARGET_FPS)
    joints_50 = _compute_smpl_joints(pose_50, human_info_path)
    if not torch.isfinite(pose_50).all() or not torch.isfinite(transl_50).all():
        raise ValueError(f"{record.key} SMPL 50Hz pose/transl 含 NaN/Inf")
    if not torch.isfinite(joints_50).all() or joints_50.shape != (len(pose_50), 24, 3):
        raise ValueError(f"{record.key} SMPL joints 非有限或形状错误: {tuple(joints_50.shape)}")
    output = Path(output_dir_text) / f"{record.key}.pkl"
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    joblib.dump(
        {
            "pose_aa": pose_50.numpy().astype(np.float32, copy=False),
            "transl": transl_50.numpy().astype(np.float32, copy=False),
            "smpl_joints": joints_50.numpy().astype(np.float32, copy=False),
            "fps": 50.0,
        },
        temporary,
        compress=3,
    )
    os.replace(temporary, output)
    return record.key


def _run_jobs(label: str, worker: Any, jobs: list[Any], workers: int) -> None:
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for _ in executor.map(worker, jobs, chunksize=1):
            completed += 1
            if completed == len(jobs) or completed % 50 == 0:
                print(f"[{label}] {completed}/{len(jobs)}", flush=True)


def _runtime_frames(source_frames: int, source_fps: float = SOURCE_FPS) -> int:
    return len(_target_time_grid(source_frames, source_fps, TARGET_FPS))


def _validate_outputs(
    records: list[SampleRecord],
    robot_dir: Path,
    smpl_dir: Path,
    contract: MjcfContract,
    *,
    verbose: bool = True,
) -> None:
    expected_robot_names = {f"{record.key}.pkl" for record in records}
    expected_smpl_names = {f"{record.key}.pkl" for record in records if record.paired}
    actual_robot_names = {path.name for path in robot_dir.glob("*.pkl")}
    actual_smpl_names = {path.name for path in smpl_dir.glob("*.pkl")}
    if actual_robot_names != expected_robot_names:
        raise ValueError(
            f"robot 输出文件集合错误: missing={sorted(expected_robot_names-actual_robot_names)[:10]}, "
            f"extra={sorted(actual_robot_names-expected_robot_names)[:10]}"
        )
    if actual_smpl_names != expected_smpl_names:
        raise ValueError(
            f"SMPL 输出文件集合错误: missing={sorted(expected_smpl_names-actual_smpl_names)[:10]}, "
            f"extra={sorted(actual_smpl_names-expected_smpl_names)[:10]}"
        )

    root_tilts_by_dataset: dict[str, list[np.ndarray]] = {
        dataset: [] for dataset in (*DATASET_DIRS, "mine")
    }
    for index, record in enumerate(records, start=1):
        robot_outer = joblib.load(robot_dir / f"{record.key}.pkl")
        if list(robot_outer) != [record.key]:
            raise ValueError(f"{record.key} robot 外层 key 错误: {list(robot_outer)}")
        robot = robot_outer[record.key]
        expected_robot_shapes = {
            "root_trans_offset": (record.source_frames, 3),
            "pose_aa": (record.source_frames, 22, 3),
            "dof": (record.source_frames, 21),
            "root_rot": (record.source_frames, 4),
        }
        for field, shape in expected_robot_shapes.items():
            value = np.asarray(robot[field])
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"{record.key} robot {field} 契约错误: {value.shape} {value.dtype}")
        if int(robot["fps"]) != 30:
            raise ValueError(f"{record.key} robot fps 不是 30")
        quat_norm_error = np.max(np.abs(np.linalg.norm(robot["root_rot"], axis=1) - 1.0))
        if quat_norm_error > 1e-5:
            raise ValueError(f"{record.key} robot root_rot 范数误差 {quat_norm_error}")

        expected_source_contract = (
            MINE_ROOT_CONTRACT if record.dataset == "mine" else LEGACY_PUBLIC_ROOT_CONTRACT
        )
        expected_correction = (
            IDENTITY_ROOT_CORRECTION_WXYZ
            if record.dataset == "mine"
            else LEGACY_PUBLIC_ROOT_CORRECTION_WXYZ
        )
        if robot.get("source_motion_contract_version") != expected_source_contract:
            raise ValueError(
                f"{record.key} source_motion_contract_version 错误: "
                f"{robot.get('source_motion_contract_version')!r}"
            )
        if robot.get("root_frame_contract_version") != SONIC_ROOT_FRAME_CONTRACT:
            raise ValueError(f"{record.key} 缺少当前 SONIC Z-up root frame 契约")
        actual_correction = np.asarray(
            robot.get("root_frame_correction_wxyz"), dtype=np.float64
        )
        if actual_correction.shape != (4,) or not np.allclose(
            actual_correction,
            np.asarray(expected_correction, dtype=np.float64),
            atol=1e-7,
            rtol=0.0,
        ):
            raise ValueError(
                f"{record.key} root frame correction 错误: "
                f"expected={expected_correction}, actual={actual_correction.tolist()}"
            )
        diagnostics = robot.get("root_height_diagnostics")
        if not isinstance(diagnostics, dict) or diagnostics.get("optimizer_success") is not True:
            raise ValueError(f"{record.key} 缺少成功的 Root-Z 审计")
        expected_height_policy = (
            "mine_csv_current_mjcf_sole_optimized"
            if record.dataset == "mine"
            else "legacy_public_current_mjcf_sole_optimized"
        )
        if robot.get("root_height_policy") != expected_height_policy:
            raise ValueError(
                f"{record.key} Root-Z policy 错误: expected={expected_height_policy!r}, "
                f"actual={robot.get('root_height_policy')!r}"
            )
        max_penetration = float(diagnostics.get("max_sole_penetration_m", math.inf))
        if max_penetration > SOLE_PENETRATION_TOLERANCE_M + 1e-6:
            raise ValueError(
                f"{record.key} 完整序列足底穿透 {max_penetration:.6f} m 超限"
            )

        root_quat_wxyz = np.asarray(robot["root_rot"], dtype=np.float64)[:, [3, 0, 1, 2]]
        root_tilts_by_dataset[record.dataset].append(
            _root_tilt_degrees_wxyz(root_quat_wxyz)
        )
        from gear_sonic.trl.utils.torch_transform import angle_axis_to_quaternion

        pose_root_quat = (
            angle_axis_to_quaternion(torch.from_numpy(robot["pose_aa"][:, 0]))
            .numpy()
            .astype(np.float64, copy=False)
        )
        rotation_dot = np.abs(np.sum(pose_root_quat * root_quat_wxyz, axis=1))
        if float(np.min(rotation_dot)) < 1.0 - 1e-5:
            raise ValueError(
                f"{record.key} pose_aa root 与 root_rot 不一致: min_abs_dot={np.min(rotation_dot)}"
            )

        sample_count = min(record.source_frames, 17)
        sample_indices = np.unique(
            np.linspace(0, record.source_frames - 1, sample_count, dtype=np.int64)
        )
        sampled_sole_z, sampled_foot_origin_z = _compute_foot_fk_metrics(
            np.asarray(robot["root_trans_offset"]),
            root_quat_wxyz,
            np.asarray(robot["dof"]),
            contract,
            frame_indices=sample_indices,
        )
        sampled_penetration = float(np.maximum(-sampled_sole_z, 0.0).max(initial=0.0))
        penetration_limit = SOLE_PENETRATION_TOLERANCE_M + 1e-3
        if sampled_penetration > penetration_limit:
            raise ValueError(
                f"{record.key} sampled foot FK 穿透 {sampled_penetration:.6f} m 超过 "
                f"{penetration_limit:.6f} m"
            )
        sampled_root_z = np.asarray(robot["root_trans_offset"])[sample_indices, 2:3]
        root_to_foot_origin = sampled_foot_origin_z - sampled_root_z
        if not np.isfinite(root_to_foot_origin).all() or float(np.min(root_to_foot_origin)) < -1.0:
            raise ValueError(f"{record.key} foot FK 相对 root 高度异常")

        if record.paired:
            smpl = joblib.load(smpl_dir / f"{record.key}.pkl")
            target_frames = _runtime_frames(record.source_frames, record.source_fps)
            expected_smpl_shapes = {
                "pose_aa": (target_frames, 72),
                "transl": (target_frames, 3),
                "smpl_joints": (target_frames, 24, 3),
            }
            for field, shape in expected_smpl_shapes.items():
                value = np.asarray(smpl[field])
                if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f"{record.key} SMPL {field} 契约错误: {value.shape} {value.dtype}")
            if not math.isclose(float(smpl["fps"]), TARGET_FPS, abs_tol=1e-6):
                raise ValueError(f"{record.key} SMPL fps 不是 50")
        elif (smpl_dir / f"{record.key}.pkl").exists():
            raise ValueError(f"Mine-only 动作意外含 SMPL 文件: {record.key}")
        if verbose and (index == len(records) or index % 100 == 0):
            print(f"[validate] {index}/{len(records)}", flush=True)

    for dataset, chunks in root_tilts_by_dataset.items():
        if not chunks:
            raise ValueError(f"root tilt 审计缺少数据集: {dataset}")
        tilts = np.concatenate(chunks)
        median_tilt = float(np.median(tilts))
        exceed_ratio = float(np.mean(tilts > ROOT_TILT_FRAME_THRESHOLD_DEGREES))
        if median_tilt > ROOT_TILT_DATASET_MEDIAN_MAX_DEGREES:
            raise ValueError(
                f"{dataset} root tilt 中位数 {median_tilt:.3f}° 超过 "
                f"{ROOT_TILT_DATASET_MEDIAN_MAX_DEGREES:.3f}°，疑似坐标系错误"
            )
        if exceed_ratio > ROOT_TILT_DATASET_EXCEED_RATIO_MAX:
            raise ValueError(
                f"{dataset} root tilt >45° 比例 {exceed_ratio:.3%} 超过 "
                f"{ROOT_TILT_DATASET_EXCEED_RATIO_MAX:.3%}"
            )
        if verbose:
            print(
                f"[root-frame] {dataset}: median_tilt={median_tilt:.3f}deg, "
                f"gt45_ratio={exceed_ratio:.3%}",
                flush=True,
            )


def _git_commit(repo_root: Path) -> str:
    """读取构建代码提交；兼容服务器 root 读取普通用户仓库的所有权检查。"""

    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={repo_root}",
                "-C",
                str(repo_root),
                "rev-parse",
                "HEAD",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _write_metadata(
    output_root: Path,
    records: list[SampleRecord],
    contract: MjcfContract,
    human_info_path: Path,
) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = meta_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            row = asdict(record)
            row["robot_file"] = f"built/robot_all/{record.key}.pkl"
            row["smpl_file"] = f"built/smpl_all/{record.key}.pkl" if record.paired else None
            row["robot_stored_fps"] = SOURCE_FPS
            row["runtime_target_fps"] = TARGET_FPS
            row["runtime_frames"] = _runtime_frames(record.source_frames, record.source_fps)
            row["root_frame_contract_version"] = SONIC_ROOT_FRAME_CONTRACT
            row["root_frame_correction_wxyz"] = list(
                IDENTITY_ROOT_CORRECTION_WXYZ
                if record.dataset == "mine"
                else LEGACY_PUBLIC_ROOT_CORRECTION_WXYZ
            )
            row["root_height_policy"] = (
                "mine_csv_current_mjcf_sole_optimized"
                if record.dataset == "mine"
                else "legacy_public_current_mjcf_sole_optimized"
            )
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    provenance = {
        "contract_version": "sonic.bumi3_hq_all.v2",
        "repo_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "mjcf_sha256": contract.sha256,
        "source_bumi_mjcf_sha256": EXPECTED_SOURCE_MJCF_SHA256,
        "human_joints_info_sha256": _sha256(human_info_path),
        "robot_count": len(records),
        "smpl_count": sum(record.paired for record in records),
        "mine_only_count": sum(not record.paired for record in records),
        "robot_stored_fps": SOURCE_FPS,
        "smpl_stored_fps": TARGET_FPS,
        "runtime_target_fps": TARGET_FPS,
        "root_frame_contract_version": SONIC_ROOT_FRAME_CONTRACT,
        "legacy_public_root_correction_wxyz": list(LEGACY_PUBLIC_ROOT_CORRECTION_WXYZ),
        "legacy_public_root_height_policy": "legacy_public_current_mjcf_sole_optimized",
        "mine_root_height_policy": "mine_csv_current_mjcf_sole_optimized",
        "root_tilt_dataset_median_max_degrees": ROOT_TILT_DATASET_MEDIAN_MAX_DEGREES,
        "root_tilt_gt45_ratio_max": ROOT_TILT_DATASET_EXCEED_RATIO_MAX,
        "sole_penetration_tolerance_m": SOLE_PENETRATION_TOLERANCE_M,
        "all_original_splits_used_for_training": True,
    }
    (meta_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checksum_paths = sorted((output_root / "built" / "robot_all").glob("*.pkl"))
    checksum_paths += sorted((output_root / "built" / "smpl_all").glob("*.pkl"))
    checksum_paths += [manifest_path, meta_dir / "provenance.json"]
    with (meta_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in checksum_paths:
            handle.write(f"{_sha256(path)}  {path.relative_to(output_root)}\n")


def build(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    built_root = output_root / "built"
    robot_final = built_root / "robot_all"
    smpl_final = built_root / "smpl_all"
    robot_staging = built_root / ".robot_all.staging"
    smpl_staging = built_root / ".smpl_all.staging"
    if robot_final.exists() or smpl_final.exists():
        raise FileExistsError("最终 robot_all/smpl_all 已存在；为防止覆盖，本工具拒绝重建")
    built_root.mkdir(parents=True, exist_ok=True)
    robot_staging.mkdir(exist_ok=True)
    smpl_staging.mkdir(exist_ok=True)

    mjcf_path = args.mjcf.resolve()
    human_info_path = args.human_joints_info.resolve()
    if not mjcf_path.is_file() or not human_info_path.is_file():
        raise FileNotFoundError(f"缺少 MJCF 或 human_joints_info: {mjcf_path}, {human_info_path}")
    contract = _parse_mjcf(mjcf_path)
    records_map = _collect_robot_records(args.bumi_root.resolve(), args.mine_root.resolve())
    records_map = _attach_smpl(records_map, args.smpl_root.resolve())
    _validate_record_counts(records_map)
    records = [records_map[key] for key in sorted(records_map)]
    print(
        f"DATASET_CONTRACT robot={len(records)} paired={sum(r.paired for r in records)} "
        f"mine_only={sum(not r.paired for r in records)}"
    )

    robot_jobs = [
        (record, contract, str(robot_staging))
        for record in records
        if not (robot_staging / f"{record.key}.pkl").exists()
    ]
    _run_jobs("robot", _convert_robot_job, robot_jobs, args.workers)
    smpl_jobs = [
        (record, str(human_info_path), str(smpl_staging))
        for record in records
        if record.paired and not (smpl_staging / f"{record.key}.pkl").exists()
    ]
    _run_jobs("smpl", _convert_smpl_job, smpl_jobs, args.workers)
    _validate_outputs(records, robot_staging, smpl_staging, contract)

    os.replace(robot_staging, robot_final)
    os.replace(smpl_staging, smpl_final)
    _write_metadata(output_root, records, contract, human_info_path)
    print("BUMI3_SONIC_DATASET_BUILD=PASS")


def validate(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    manifest = output_root / "meta" / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    record_fields = SampleRecord.__dataclass_fields__.keys()
    records = [
        SampleRecord(**{field: row[field] for field in record_fields}) for row in _jsonl(manifest)
    ]
    _validate_record_counts({record.key: record for record in records})
    contract = _parse_mjcf(args.mjcf.resolve())
    _validate_outputs(
        records,
        output_root / "built" / "robot_all",
        output_root / "built" / "smpl_all",
        contract,
    )
    checksum_file = output_root / "meta" / "SHA256SUMS"
    if not checksum_file.is_file():
        raise FileNotFoundError(checksum_file)
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        actual = _sha256(output_root / relative)
        if actual != expected:
            raise ValueError(f"SHA256 不一致: {relative}: expected={expected}, actual={actual}")
    print("BUMI3_SONIC_DATASET_VALIDATE=PASS")


def _parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="转换、全量校验并原子发布数据集")
    build_parser.add_argument("--smpl-root", type=Path, required=True)
    build_parser.add_argument("--bumi-root", type=Path, required=True)
    build_parser.add_argument("--mine-root", type=Path, required=True)
    build_parser.add_argument("--output-root", type=Path, required=True)
    build_parser.add_argument(
        "--mjcf",
        type=Path,
        default=repo_root / "gear_sonic/data/assets/robot_description/mjcf/bumi3.xml",
    )
    build_parser.add_argument(
        "--human-joints-info",
        type=Path,
        default=repo_root / "gear_sonic/data/human/human_joints_info.pkl",
    )
    build_parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    build_parser.set_defaults(func=build)

    validate_parser = subparsers.add_parser("validate", help="重新校验已发布数据和 SHA256")
    validate_parser.add_argument("--output-root", type=Path, required=True)
    validate_parser.add_argument(
        "--mjcf",
        type=Path,
        default=repo_root / "gear_sonic/data/assets/robot_description/mjcf/bumi3.xml",
    )
    validate_parser.set_defaults(func=validate)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if getattr(args, "workers", 1) < 1:
        raise ValueError("--workers 必须大于 0")
    args.func(args)


if __name__ == "__main__":
    main()
