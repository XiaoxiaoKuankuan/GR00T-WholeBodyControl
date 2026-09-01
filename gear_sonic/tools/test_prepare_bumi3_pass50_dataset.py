# SPDX-License-Identifier: Apache-2.0
"""验证 BUMI3 SONIC PASS-only 30 Hz 到 50 Hz 数据构建契约。

本测试使用一个可追溯的合成动作走完整的 ``build``、全量 ``validate`` 与
``pair-smpl`` 路径，重点防止四类容易静默污染训练的数据错误：非 PASS 样本进入白名单、
原始 NPZ 被复制而不是硬链接、四元数被线性插值或顺序误用、机器人与 SMPL 在帧数不一致
时仍被强行配对。合成动作沿 X 轴匀速移动，根节点绕 Z 轴匀速旋转，因此还能精确检查
50 Hz 位置插值、SLERP、速度重算及四元数单位范数。

测试只读取项目自带的 BUMI3 MJCF，不启动 Isaac Lab，也不修改任何真实训练数据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np

from gear_sonic.tools import prepare_bumi3_pass50_dataset as prepare


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, int]:
    mjcf = (
        Path(__file__).parents[1]
        / "data"
        / "assets"
        / "robot_description"
        / "mjcf"
        / "bumi3.xml"
    )
    contract = prepare._extract_contract(mjcf)
    source_root = tmp_path / "source"
    source_path = source_root / "aioz_gdance" / "mimic_npz" / "bumi3" / "clip.npz"
    source_path.parent.mkdir(parents=True)
    source_frames = 7
    source_time = np.arange(source_frames, dtype=np.float64) / prepare.SOURCE_FPS
    joint_names = tuple(reversed(contract.actuator_joint_names))
    joint_pos = np.stack(
        [source_time * float(index + 1) for index in range(len(joint_names))], axis=1
    )
    body_pos = np.zeros((source_frames, 22, 3), dtype=np.float64)
    body_pos[..., 2] = 0.5
    body_pos[..., 0] = source_time[:, None]
    body_index = {name: index for index, name in enumerate(contract.body_names)}
    body_pos[:, body_index["l_ankle_roll_link"], 2] = 0.02
    body_pos[:, body_index["r_ankle_roll_link"], 2] = 0.02
    body_quat = np.zeros((source_frames, 22, 4), dtype=np.float64)
    body_quat[..., 0] = 1.0
    yaw = source_time * (math.pi / 2.0)
    body_quat[..., 0] = np.cos(yaw[:, None] / 2.0)
    body_quat[..., 3] = np.sin(yaw[:, None] / 2.0)
    np.savez_compressed(
        source_path,
        fps=np.asarray(30.0),
        joint_pos=joint_pos.astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        joint_names=np.asarray(joint_names),
        body_names=np.asarray(contract.body_names),
        quaternion_order=np.asarray("wxyz"),
    )
    report = tmp_path / "quality_report.jsonl"
    report.write_text(
        json.dumps(
            {
                "dataset": "aioz_gdance",
                "sample_id": "aioz_gdance/clip",
                "source_relative_path": "aioz_gdance/mimic_npz/bumi3/clip.npz",
                "source_sha256": _sha256(source_path),
                "status": "PASS",
                "quality_accepted": True,
                "valid_intervals": [[0, source_frames]],
                "metrics": {"num_frames": source_frames},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return source_root, source_path, report, source_frames


def test_build_resamples_and_preserves_hardlink_contract(tmp_path: Path) -> None:
    source_root, source_path, report, source_frames = _write_fixture(tmp_path)
    mjcf = (
        Path(__file__).parents[1]
        / "data"
        / "assets"
        / "robot_description"
        / "mjcf"
        / "bumi3.xml"
    )
    output = tmp_path / "pass50"
    prepare.build(
        argparse.Namespace(
            source_root=source_root,
            quality_report=report,
            source_mjcf=mjcf,
            target_mjcf=mjcf,
            output_root=output,
            workers=1,
            expected_pass_count=1,
        )
    )
    summary = prepare._validate_full_outputs(output, 1)
    assert summary["robot_count"] == 1
    manifest = json.loads((output / "meta" / "manifest.jsonl").read_text())
    expected_frames = len(prepare._target_grid(source_frames)[0])
    assert manifest["target_frames"] == expected_frames
    raw = output / manifest["raw_hardlink"]
    assert (raw.stat().st_dev, raw.stat().st_ino) == (
        source_path.stat().st_dev,
        source_path.stat().st_ino,
    )

    motion = joblib.load(output / manifest["robot_file"])[manifest["key"]]
    target_time = np.arange(expected_frames, dtype=np.float64) / prepare.TARGET_FPS
    np.testing.assert_allclose(motion["root_trans_offset"][:, 0], target_time, atol=1e-6)
    expected_root_xyzw = np.stack(
        [
            np.zeros(expected_frames),
            np.zeros(expected_frames),
            np.sin(target_time * math.pi / 4.0),
            np.cos(target_time * math.pi / 4.0),
        ],
        axis=1,
    )
    np.testing.assert_allclose(motion["root_rot"], expected_root_xyzw, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(motion["root_rot"], axis=1), 1.0, atol=1e-6)
    with np.load(output / manifest["audit_file"], allow_pickle=False) as audit:
        np.testing.assert_allclose(audit["body_lin_vel_w"][:, 0, 0], 1.0, atol=1e-5)
        assert audit["joint_vel"].shape == (expected_frames, 21)
        assert audit["joint_acc"].shape == (expected_frames, 21)
        assert audit["joint_jerk"].shape == (expected_frames, 21)


def test_pair_smpl_only_links_exact_50hz_frame_match(tmp_path: Path) -> None:
    source_root, _, report, source_frames = _write_fixture(tmp_path)
    mjcf = (
        Path(__file__).parents[1]
        / "data"
        / "assets"
        / "robot_description"
        / "mjcf"
        / "bumi3.xml"
    )
    output = tmp_path / "pass50"
    prepare.build(
        argparse.Namespace(
            source_root=source_root,
            quality_report=report,
            source_mjcf=mjcf,
            target_mjcf=mjcf,
            output_root=output,
            workers=1,
            expected_pass_count=1,
        )
    )
    frames = len(prepare._target_grid(source_frames)[0])
    smpl_source = tmp_path / "smpl"
    smpl_source.mkdir()
    smpl_path = smpl_source / "aioz_gdance__clip.pkl"
    robot = joblib.load(output / "built" / "robot_all" / "aioz_gdance__clip.pkl")[
        "aioz_gdance__clip"
    ]
    robot_root = prepare.Rotation.from_quat(robot["root_rot"])
    robot_waist = robot_root * prepare.Rotation.from_rotvec(robot["pose_aa"][:, 1])
    y_to_z = prepare.Rotation.from_rotvec(np.array([math.pi / 2.0, 0.0, 0.0]))
    smpl_base = prepare.Rotation.from_quat(np.array([0.5, 0.5, 0.5, 0.5]))
    smpl_root_y_up = y_to_z.inv() * robot_waist * smpl_base
    smpl_pose = np.zeros((frames, 72), dtype=np.float32)
    smpl_pose[:, :3] = smpl_root_y_up.as_rotvec().astype(np.float32)
    joblib.dump(
        {
            "fps": 50,
            "pose_aa": smpl_pose,
            "transl": np.zeros((frames, 3), dtype=np.float32),
            "smpl_joints": np.zeros((frames, 24, 3), dtype=np.float32),
        },
        smpl_path,
    )
    prepare.pair_smpl(
        argparse.Namespace(
            output_root=output,
            smpl_source=smpl_source,
            max_pair_median_degrees=45.0,
        )
    )
    linked = output / "built" / "smpl_all" / smpl_path.name
    assert (linked.stat().st_dev, linked.stat().st_ino) == (
        smpl_path.stat().st_dev,
        smpl_path.stat().st_ino,
    )
    pairing = json.loads((output / "meta" / "smpl_pairing_summary.json").read_text())
    assert pairing["paired_count"] == 1
    assert pairing["robot_only_count"] == 0

    # 把 processed SMPL root 额外旋转 90 度；重跑必须移除旧硬链接并降为 robot-only。
    mismatched_root = robot_waist * prepare.Rotation.from_rotvec(
        np.tile(np.array([[math.pi / 2.0, 0.0, 0.0]]), (frames, 1))
    )
    smpl_pose[:, :3] = (y_to_z.inv() * mismatched_root * smpl_base).as_rotvec()
    joblib.dump(
        {
            "fps": 50,
            "pose_aa": smpl_pose,
            "transl": np.zeros((frames, 3), dtype=np.float32),
            "smpl_joints": np.zeros((frames, 24, 3), dtype=np.float32),
        },
        smpl_path,
    )
    prepare.pair_smpl(
        argparse.Namespace(
            output_root=output,
            smpl_source=smpl_source,
            max_pair_median_degrees=45.0,
        )
    )
    assert not linked.exists()
    pairing = json.loads((output / "meta" / "smpl_pairing_summary.json").read_text())
    assert pairing["paired_count"] == 0
    assert pairing["status_counts"] == {"ROBOT_ONLY_SMPL_COORDINATE_MISMATCH": 1}


def test_pass_loader_rejects_review_even_when_quality_accepted(tmp_path: Path) -> None:
    source_root, _, report, _ = _write_fixture(tmp_path)
    row = json.loads(report.read_text())
    row["status"] = "REVIEW"
    report.write_text(json.dumps(row) + "\n", encoding="utf-8")
    try:
        prepare._load_pass_records(source_root, report, expected_count=1)
    except ValueError as error:
        assert "实际为 0" in str(error)
    else:
        raise AssertionError("REVIEW 不得进入 PASS 白名单")
