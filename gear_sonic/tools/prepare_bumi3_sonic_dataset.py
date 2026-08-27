#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""将经过人工筛选的 GENMO SMPL/BUMI3 动作整理为 SONIC 训练数据。

本工具专门实现 BUMI3 Robot+SMPL 双编码器的离线数据边界：机器人源文件保持
30Hz，依照每个源文件携带的 ``joint_names`` 重排到当前 BUMI3 MJCF 的 actuator
顺序，再生成 SONIC motion-lib 所需的 root、axis-angle、DoF 和四元数字段；人体
源文件则把 ``pose_aa``、``transl`` 和由 SONIC FK 计算的 ``smpl_joints`` 一起
预重采样到 50Hz。SMPL 必须离线完成三项同步重采样，因为当前 motion-lib 加载器
只会自动重采样 SMPL pose，却会直接读取 joints/transl 并断言它们与机器人运行时
50Hz 帧数相等。

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
      --output-root /data/sonic_bumi3/datasets/hq_all_v1

    python gear_sonic/tools/prepare_bumi3_sonic_dataset.py validate \
      --output-root /data/sonic_bumi3/datasets/hq_all_v1
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
    )


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
        raise ValueError(f"{source} joint_names 与 SONIC MJCF 不一致: missing={missing_names}, extra={extra_names}")
    source_index = {name: index for index, name in enumerate(source_names)}
    dof = torch.stack(
        [qpos[:, 7 + source_index[name]] for name in contract.actuator_joint_names], dim=1
    )

    root_quat_wxyz = qpos[:, 3:7]
    norms = torch.linalg.vector_norm(root_quat_wxyz, dim=1, keepdim=True)
    if bool((norms < 1e-8).any()):
        raise ValueError(f"{source} 含零范数 root quaternion")
    root_quat_wxyz = root_quat_wxyz / norms
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
        "root_trans_offset": qpos[:, :3].numpy().astype(np.float32, copy=False),
        "pose_aa": pose_aa.numpy().astype(np.float32, copy=False),
        "dof": dof.numpy().astype(np.float32, copy=False),
        "root_rot": root_quat_wxyz[:, [1, 2, 3, 0]].numpy().astype(np.float32, copy=False),
        "fps": 30,
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
    records: list[SampleRecord], robot_dir: Path, smpl_dir: Path, *, verbose: bool = True
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


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    provenance = {
        "contract_version": "sonic.bumi3_hq_all.v1",
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
    _validate_outputs(records, robot_staging, smpl_staging)

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
    _validate_outputs(
        records,
        output_root / "built" / "robot_all",
        output_root / "built" / "smpl_all",
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
    validate_parser.set_defaults(func=validate)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if getattr(args, "workers", 1) < 1:
        raise ValueError("--workers 必须大于 0")
    args.func(args)


if __name__ == "__main__":
    main()
