#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""按 base_link 锚点审计旧 hq_all_v2 BUMI3 Robot/SMPL 坐标契约。

本脚本只读已经构建的 ``robot_all`` 与 ``smpl_all``，不会生成、转换、重采样或
删除训练数据。Robot 侧严格复现 MotionLib 的 30Hz→50Hz 独占末帧时间网格，
对浮动根四元数做 Slerp，并直接把 ``base_link`` 世界姿态作为 Robot Encoder
锚点。SMPL 侧严格复现训练命令项的 ``pose_aa`` Y-up→Z-up 左乘旋转和 SMPL
base rotation removal。两侧统一到 Z-up 后，按整段相对旋转角中位数 45 度
审计。若配置声明隔离 key，则要求检测结果与配置精确一致；也可用命令行显式
指定预期异常数。旧腰部锚点下得到的 55 条历史结果不能再当作 base_link
契约的固定真值。

通过条件还包括：3261 条 Robot、3162 个同名 Robot/SMPL 配对和 99 条 Mine-only。
新训练不会直接使用这个旧目录，而由三源构建工具按同一 base_link 契约逐条
决定配对状态。该检查不能替代 Isaac Lab reset/step、奖励曲线或最终动作质量验证。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bumi3.yaml"
)
EXPECTED_ROBOT_COUNT = 3261
EXPECTED_SMPL_COUNT = 3162
EXPECTED_MINE_ONLY_COUNT = 99
SOURCE_FPS = 30.0
TARGET_FPS = 50.0
PAIR_MEDIAN_THRESHOLD_DEGREES = 45.0


def _unwrap_robot(path: Path, key: str) -> dict:
    """读取 SONIC robot PKL，并验证外层动作 key。"""
    payload = joblib.load(path)
    if not isinstance(payload, dict) or list(payload) != [key]:
        actual = list(payload) if isinstance(payload, dict) else type(payload)
        raise ValueError(f"{path} 外层 key 错误: {actual}")
    return payload[key]


def _runtime_times(num_source_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """复现 MotionLib 的源/目标时间轴以及末帧排除规则。"""
    duration = (num_source_frames - 1) / SOURCE_FPS
    source_times = np.arange(num_source_frames, dtype=np.float64) / SOURCE_FPS
    target_times = np.arange(0.0, duration, 1.0 / TARGET_FPS, dtype=np.float64)
    return source_times, target_times


def _robot_base_rotation(robot: dict) -> Rotation:
    """从 30Hz Robot 数据恢复 MotionLib 50Hz 的 base_link 世界姿态。"""
    root_xyzw = np.asarray(robot["root_rot"], dtype=np.float64)
    pose_aa = np.asarray(robot["pose_aa"], dtype=np.float64)
    if root_xyzw.ndim != 2 or root_xyzw.shape[1] != 4:
        raise ValueError(f"Robot root_rot shape 错误: {root_xyzw.shape}")
    if pose_aa.shape != (len(root_xyzw), 22, 3):
        raise ValueError(f"Robot pose_aa shape 错误: {pose_aa.shape}")
    source_times, target_times = _runtime_times(len(root_xyzw))
    return Slerp(source_times, Rotation.from_quat(root_xyzw))(target_times)


def _processed_smpl_root_rotation(smpl: dict) -> Rotation:
    """复现训练端 SMPL 根姿态的 Y-up→Z-up 与 base rotation removal。"""
    pose_aa = np.asarray(smpl["pose_aa"], dtype=np.float64)
    if pose_aa.ndim != 2 or pose_aa.shape[1] != 72:
        raise ValueError(f"SMPL pose_aa shape 错误: {pose_aa.shape}")
    root_y_up = Rotation.from_rotvec(pose_aa[:, :3])
    y_to_z = Rotation.from_rotvec(np.array([np.pi / 2.0, 0.0, 0.0]))
    smpl_base = Rotation.from_quat(np.array([0.5, 0.5, 0.5, 0.5]))
    return y_to_z * root_y_up * smpl_base.inv()


def _load_excluded_keys(config_path: Path) -> list[str]:
    """读取配置中的可选精确隔离 key；新三源配置允许为空。"""
    cfg = OmegaConf.load(config_path)
    keys = OmegaConf.to_container(
        cfg.manager_env.commands.motion.motion_lib_cfg.exclude_motion_keys,
        resolve=True,
    )
    if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
        raise ValueError("exclude_motion_keys 必须是字符串列表")
    if len(keys) != len(set(keys)):
        raise ValueError("exclude_motion_keys 含重复 key")
    return keys


def validate(args: argparse.Namespace) -> None:
    """执行全量同名配对姿态审计，并打印可直接写入训练记录的计数。"""
    robot_files = {path.stem: path for path in args.robot_dir.glob("*.pkl")}
    smpl_files = {path.stem: path for path in args.smpl_dir.glob("*.pkl")}
    excluded_keys = _load_excluded_keys(args.config)

    if len(robot_files) != EXPECTED_ROBOT_COUNT:
        raise ValueError(f"Robot 数量 {len(robot_files)} != {EXPECTED_ROBOT_COUNT}")
    if len(smpl_files) != EXPECTED_SMPL_COUNT:
        raise ValueError(f"SMPL 数量 {len(smpl_files)} != {EXPECTED_SMPL_COUNT}")
    if not set(smpl_files).issubset(robot_files):
        raise ValueError(
            "SMPL 存在无 Robot 配对: "
            f"{sorted(set(smpl_files) - set(robot_files))[:10]}"
        )
    mine_only = sorted(set(robot_files) - set(smpl_files))
    if len(mine_only) != EXPECTED_MINE_ONLY_COUNT or any(
        not key.startswith("mine__") for key in mine_only
    ):
        raise ValueError(f"Mine-only 契约错误: count={len(mine_only)}")
    if excluded_keys:
        absent_robot = sorted(set(excluded_keys) - set(robot_files))
        absent_smpl = sorted(set(excluded_keys) - set(smpl_files))
        if absent_robot or absent_smpl:
            raise ValueError(f"隔离 key 缺失: robot={absent_robot}, smpl={absent_smpl}")

    detected_bad: list[str] = []
    pair_medians: dict[str, float] = {}
    for index, key in enumerate(sorted(smpl_files), start=1):
        robot = _unwrap_robot(robot_files[key], key)
        smpl = joblib.load(smpl_files[key])
        robot_base = _robot_base_rotation(robot)
        smpl_root = _processed_smpl_root_rotation(smpl)
        if len(robot_base) != len(smpl_root):
            raise ValueError(
                f"{key} 50Hz 长度不一致: robot={len(robot_base)}, smpl={len(smpl_root)}"
            )
        relative_degrees = np.degrees((robot_base.inv() * smpl_root).magnitude())
        median_degrees = float(np.median(relative_degrees))
        pair_medians[key] = median_degrees
        if median_degrees > PAIR_MEDIAN_THRESHOLD_DEGREES:
            detected_bad.append(key)
        if args.verbose and (index % 250 == 0 or index == len(smpl_files)):
            print(f"[coordinate-audit] {index}/{len(smpl_files)}", flush=True)

    actual_bad = set(detected_bad)
    if args.expected_bad_count is not None and len(actual_bad) != args.expected_bad_count:
        raise ValueError(
            f"base_link 契约检出异常配对 {len(actual_bad)} != {args.expected_bad_count}"
        )
    configured_bad = set(excluded_keys)
    if configured_bad and actual_bad != configured_bad:
        raise ValueError(
            "45 度审计结果与隔离清单不一致: "
            f"新增={sorted(actual_bad - configured_bad)}, "
            f"清单但未检出={sorted(configured_bad - actual_bad)}"
        )

    retained_pair_medians = [
        value for key, value in pair_medians.items() if key not in actual_bad
    ]
    retained_robot_count = len(robot_files) - len(actual_bad)
    retained_smpl_count = len(smpl_files) - len(actual_bad)
    print(
        "BUMI3_TRAINING_COORDINATES=PASS "
        f"robot_total={len(robot_files)} smpl_paired={len(smpl_files)} "
        f"mine_only={len(mine_only)} base_anchor_detected={len(actual_bad)} "
        f"active_exclusions={len(configured_bad)} "
        f"training_robot={retained_robot_count} training_smpl={retained_smpl_count}"
    )
    print(
        "COORDINATE_CONTRACT robot=Z-up; "
        "smpl_pose_aa=Y-up->runtime_Z-up_once; smpl_joints=offline_Z-up"
    )
    print(
        "PAIR_MEDIAN_DEGREES "
        f"retained_median={np.median(retained_pair_medians):.6f} "
        f"retained_max={np.max(retained_pair_medians):.6f} "
        "base_anchor_bad_min="
        f"{min((pair_medians[key] for key in actual_bad), default=float('nan')):.6f}"
    )


def _parse_args() -> argparse.Namespace:
    """解析只读数据目录与实验配置路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-dir", type=Path, required=True)
    parser.add_argument("--smpl-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--expected-bad-count",
        type=int,
        help="可选的 base_link 锚点异常配对预期数；默认只报告实际检测结果",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(_parse_args())
