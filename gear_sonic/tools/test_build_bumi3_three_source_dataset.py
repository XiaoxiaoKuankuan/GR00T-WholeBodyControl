# SPDX-License-Identifier: Apache-2.0
"""验证 BUMI3 SONIC 三数据源软链接索引和审计报告。

测试构造一套缩小版的大集、hq4 和 Mine 来源：大集 train 的 SMPL 比
Robot 少两帧，hq4 含一条配对与一条 Robot-only，Mine 只含 Robot，另保留
一个不应进入索引的旧公开动作。完整调用 ``build`` 后检查原子输出、
软链接目标、train/test 隔离、配对降级、自然按动作条数采样比例以及
SHA256 复核。测试不会启动 Isaac Lab，也不会接触服务器真实数据目录。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.tools import build_bumi3_three_source_dataset as build_tool


def _write_robot(path: Path, frames: int, fps: float) -> None:
    """写入满足 22-node/21-DoF/xyzw 契约的直立 BUMI3 合成动作。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    root_rot = np.zeros((frames, 4), dtype=np.float32)
    root_rot[:, 3] = 1.0
    root_trans = np.zeros((frames, 3), dtype=np.float32)
    root_trans[:, 2] = 0.45
    payload = {
        path.stem: {
            "root_trans_offset": root_trans,
            "pose_aa": np.zeros((frames, 22, 3), dtype=np.float32),
            "dof": np.zeros((frames, 21), dtype=np.float32),
            "root_rot": root_rot,
            "fps": fps,
        }
    }
    joblib.dump(payload, path)


def _write_smpl(path: Path, frames: int) -> None:
    """写入 pose 为 Y-up、关节几何为 Z-up 且与直立 base 对齐的合成 SMPL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    y_to_z = Rotation.from_rotvec(np.array([np.pi / 2.0, 0.0, 0.0]))
    smpl_base = Rotation.from_quat(np.array([0.5, 0.5, 0.5, 0.5]))
    root_y_up = y_to_z.inv() * smpl_base
    pose_aa = np.zeros((frames, 72), dtype=np.float32)
    pose_aa[:, :3] = root_y_up.as_rotvec().astype(np.float32)
    joints = np.zeros((frames, 24, 3), dtype=np.float32)
    joints[:, :, 0] = np.linspace(-0.2, 0.2, 24, dtype=np.float32)
    joints[:, :, 2] = np.linspace(0.0, 1.7, 24, dtype=np.float32)
    joblib.dump(
        {
            "pose_aa": pose_aa,
            "transl": np.zeros((frames, 3), dtype=np.float32),
            "smpl_joints": joints,
            "fps": 50.0,
        },
        path,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_build_and_validate_three_source_index(tmp_path: Path) -> None:
    large = tmp_path / "large"
    hq4 = tmp_path / "hq4"
    mine_dir = tmp_path / "hq_all_v2/built/robot_all"
    output = tmp_path / "three_source"

    _write_robot(large / "train/robot_filtered/nested/large_train.pkl", frames=10, fps=50.0)
    _write_smpl(large / "train/smpl_filtered/large_train.pkl", frames=8)
    _write_robot(large / "test/robot_filtered/nested/large_test.pkl", frames=7, fps=50.0)
    _write_smpl(large / "test/smpl_filtered/large_test.pkl", frames=7)

    _write_robot(hq4 / "built/robot_all/hq4__paired.pkl", frames=6, fps=50.0)
    _write_smpl(hq4 / "built/smpl_all/hq4__paired.pkl", frames=6)
    _write_robot(hq4 / "built/robot_all/hq4__robot_only.pkl", frames=6, fps=50.0)

    _write_robot(mine_dir / "mine__self.pkl", frames=9, fps=30.0)
    _write_robot(mine_dir / "aistpp__must_not_reenter.pkl", frames=9, fps=30.0)

    large_source_mjcf = large / "meta/bumi3.source.xml"
    large_source_mjcf.parent.mkdir(parents=True)
    shutil.copy2(build_tool.CURRENT_MJCF_PATH, large_source_mjcf)
    hq4_provenance = hq4 / "meta/provenance.json"
    hq4_provenance.parent.mkdir(parents=True)
    hq4_provenance.write_text(
        json.dumps({"target_mjcf_sha256": build_tool.CURRENT_MJCF_SHA256}),
        encoding="utf-8",
    )
    mine_provenance = tmp_path / "hq_all_v2/meta/provenance.json"
    mine_provenance.parent.mkdir(parents=True)
    mine_provenance.write_text(
        json.dumps({"mjcf_sha256": build_tool.CURRENT_MJCF_SHA256}),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        large_root=large,
        hq4_root=hq4,
        mine_robot_dir=mine_dir,
        output_root=output,
        workers=1,
        expected_large_train=1,
        expected_large_test=1,
        expected_hq4_robot=2,
        expected_hq4_smpl=1,
        expected_mine=1,
    )
    build_tool.build(args)
    result = build_tool.validate_index(output, workers=1, verify_hashes=True)
    assert result == {
        "contract_version": build_tool.CONTRACT_VERSION,
        "train_robot_count": 4,
        "train_smpl_count": 2,
        "test_robot_count": 1,
        "test_smpl_count": 1,
        "validate": "PASS",
    }

    summary = json.loads((output / "meta/summary.json").read_text(encoding="utf-8"))
    assert summary["train_robot_count"] == 4
    assert summary["train_smpl_count"] == 2
    assert summary["train_robot_only_count"] == 2
    assert summary["natural_sampling_probabilities"] == {
        "hq4_pass50": 0.5,
        "large_train": 0.25,
        "mine_robot_only": 0.25,
    }
    train_metadata = joblib.load(output / "train/robot_all/metadata.pkl")
    assert train_metadata == {
        "hq4__paired": {"length": 6, "fps": 50.0},
        "hq4__robot_only": {"length": 6, "fps": 50.0},
        "large_train": {"length": 8, "fps": 50.0},
        "mine__self": {"length": 9, "fps": 30.0},
    }

    train_rows = {row["key"]: row for row in _read_jsonl(output / "meta/train_manifest.jsonl")}
    assert train_rows["large_train"]["status"] == "PAIRED"
    assert train_rows["large_train"]["frame_delta_smpl_minus_robot"] == -2
    assert train_rows["large_train"]["aligned_source_frames"] == 8
    assert train_rows["hq4__robot_only"]["status"] == "ROBOT_ONLY_NO_SMPL"
    assert train_rows["mine__self"]["status"] == "ROBOT_ONLY_NO_SMPL"
    assert "aistpp__must_not_reenter" not in train_rows

    robot_link = output / train_rows["large_train"]["robot_link"]
    smpl_link = output / train_rows["large_train"]["smpl_link"]
    assert robot_link.is_symlink()
    assert smpl_link.is_symlink()
    assert robot_link.resolve() == (
        large / "train/robot_filtered/nested/large_train.pkl"
    ).resolve()
    assert smpl_link.resolve() == (large / "train/smpl_filtered/large_train.pkl").resolve()


def test_robot_pose_dof_axis_gate_rejects_waist_sign_mismatch(tmp_path: Path) -> None:
    """关节轴门禁必须拒绝 dof 为正但 waist pose 使用负 Z 轴的数据。"""

    robot_path = tmp_path / "bad_waist_sign.pkl"
    _write_robot(robot_path, frames=6, fps=50.0)
    payload = joblib.load(robot_path)
    payload[robot_path.stem]["dof"][:, 0] = 0.2
    payload[robot_path.stem]["pose_aa"][:, 1, 2] = -0.2
    joblib.dump(payload, robot_path)
    row = build_tool.audit_record(
        build_tool.SourceRecord(
            key=robot_path.stem,
            split="train",
            source="unit",
            robot_path=str(robot_path),
            smpl_path=None,
            expected_robot_fps=50.0,
        )
    )
    assert row["status"] == "ROBOT_INVALID"
    assert "关节顺序或轴符号不一致" in row["reason"]


def test_pair_time_origin_gate_downgrades_only_smpl(tmp_path: Path) -> None:
    """同名两侧声明不同时间起点时保留 Robot，并把该 SMPL 降级。"""

    robot_path = tmp_path / "time_mismatch.pkl"
    smpl_path = tmp_path / "smpl/time_mismatch.pkl"
    _write_robot(robot_path, frames=6, fps=50.0)
    _write_smpl(smpl_path, frames=6)
    robot_payload = joblib.load(robot_path)
    robot_payload[robot_path.stem]["start_time"] = 0.0
    joblib.dump(robot_payload, robot_path)
    smpl_payload = joblib.load(smpl_path)
    smpl_payload["start_time"] = 0.02
    joblib.dump(smpl_payload, smpl_path)
    row = build_tool.audit_record(
        build_tool.SourceRecord(
            key=robot_path.stem,
            split="train",
            source="unit",
            robot_path=str(robot_path),
            smpl_path=str(smpl_path),
            expected_robot_fps=50.0,
        )
    )
    assert row["status"] == "ROBOT_ONLY_PAIR_TIME_ORIGIN_MISMATCH"
    assert row["time_origin_delta_smpl_minus_robot_seconds"] == pytest.approx(0.02)


def test_pair_orientation_gate_uses_base_root_not_waist_joint(tmp_path: Path) -> None:
    """腰关节转动不得改变 base 锚点下 Robot/SMPL 配对是否合格。"""

    robot_path = tmp_path / "waist_rotated.pkl"
    smpl_path = tmp_path / "smpl/waist_rotated.pkl"
    _write_robot(robot_path, frames=6, fps=50.0)
    _write_smpl(smpl_path, frames=6)
    payload = joblib.load(robot_path)
    payload[robot_path.stem]["dof"][:, 0] = 1.0
    payload[robot_path.stem]["pose_aa"][:, 1, 2] = 1.0
    joblib.dump(payload, robot_path)

    row = build_tool.audit_record(
        build_tool.SourceRecord(
            key=robot_path.stem,
            split="train",
            source="unit",
            robot_path=str(robot_path),
            smpl_path=str(smpl_path),
            expected_robot_fps=50.0,
        )
    )
    assert row["status"] == "PAIRED"
    assert row["pair_median_degrees"] == pytest.approx(0.0, abs=1e-5)
