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

import joblib
import numpy as np
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
    """写入 pose 为 Y-up、关节几何为 Z-up 且与直立 waist 对齐的合成 SMPL。"""

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

    train_rows = {row["key"]: row for row in _read_jsonl(output / "meta/train_manifest.jsonl")}
    assert train_rows["large_train"]["status"] == "PAIRED"
    assert train_rows["large_train"]["frame_delta_smpl_minus_robot"] == -2
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
