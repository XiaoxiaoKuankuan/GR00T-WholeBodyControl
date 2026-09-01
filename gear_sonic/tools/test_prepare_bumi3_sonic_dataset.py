# SPDX-License-Identifier: Apache-2.0
"""验证 BUMI3 SONIC 数据转换中最容易产生静默错位的纯函数契约。

这些测试不启动 Isaac Sim，也不把静态检查冒充训练验证。覆盖重点是：当前 BUMI3
MJCF 的 21 个 actuator/body joint 集合、源 ``joint_names`` 任意顺序到 SONIC
actuator 顺序的重排、Mine Z-up identity 契约、公开 legacy Y-up→Z-up 根姿态修正、
当前 MJCF 足底 Root-Z 优化、wxyz→xyzw 序列化、关节轴角写入 body traversal 顺序，
以及 30Hz→50Hz 必须复现 motion-lib ``arange`` 末帧排除规则。全量文件计数、有限
值、根倾角分布、FK 足底、配对和 SHA256 由准备工具的 ``validate`` 子命令另行执行。
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
import torch

from gear_sonic.tools.prepare_bumi3_sonic_dataset import (
    LEGACY_PUBLIC_ROOT_CONTRACT,
    MINE_ROOT_CONTRACT,
    SONIC_ROOT_FRAME_CONTRACT,
    MjcfContract,
    SampleRecord,
    _convert_robot_job,
    _parse_mjcf,
    _resample_pose_and_translation,
    _runtime_frames,
)
from gear_sonic.utils.motion_lib.motion_lib_base import exclude_motion_data_by_exact_keys


REPO_ROOT = Path(__file__).resolve().parents[2]
MJCF = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/bumi3.xml"


def test_exact_motion_exclusion_does_not_use_prefix_semantics() -> None:
    source = {
        "aistpp__bad": {"value": 1},
        "aistpp__bad_variant": {"value": 2},
        "mine__good": {"value": 3},
    }
    filtered, matched, missing = exclude_motion_data_by_exact_keys(
        source, ["aistpp__bad", "absent__sample"]
    )
    assert list(filtered) == ["aistpp__bad_variant", "mine__good"]
    assert matched == ["aistpp__bad"]
    assert missing == ["absent__sample"]
    # helper 不得原地修改 MotionLib 的完整索引。
    assert list(source) == ["aistpp__bad", "aistpp__bad_variant", "mine__good"]


def test_exact_motion_exclusion_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="重复动作 key"):
        exclude_motion_data_by_exact_keys({"bad": {}}, ["bad", "bad"])


def test_parse_bumi3_mjcf_contract() -> None:
    contract = _parse_mjcf(MJCF)
    assert len(contract.actuator_joint_names) == 21
    assert len(contract.body_joint_names) == 21
    assert set(contract.actuator_joint_names) == set(contract.body_joint_names)
    assert contract.actuator_joint_names[0] == "waist_yaw_joint"
    assert contract.body_joint_names[0] == "waist_yaw_joint"


def test_robot_conversion_reorders_source_names(tmp_path: Path) -> None:
    contract = _parse_mjcf(MJCF)
    source_names = tuple(reversed(contract.actuator_joint_names))
    frames = 3
    qpos = torch.zeros((frames, 28), dtype=torch.float32)
    qpos[:, 3] = 1.0
    for source_index in range(21):
        qpos[:, 7 + source_index] = float(source_index + 1) / 100.0
    source = tmp_path / "source.pt"
    torch.save(
        {
            "contract_version": "genmo.bumi_music.v1",
            "source_motion_contract_version": MINE_ROOT_CONTRACT,
            "qpos": qpos,
            "fps": 30,
            "robot_name": "bumi",
            "joint_names": list(source_names),
            "quaternion_convention": "wxyz",
            "qpos_order": "mujoco_native",
            "source_dataset": "mine",
            "source_sample_id": "sample",
            "quality_accepted": True,
            "source_mjcf_sha256": (
                "482138b437dbdabd6171fa8d44b55db5d7125a228c95b69ce3d1159cafe8537c"
            ),
        },
        source,
    )
    record = SampleRecord(
        key="mine__sample",
        dataset="mine",
        sample_id="sample",
        split="train",
        robot_source=str(source),
        smpl_source=None,
        smpl_motion_key=None,
        source_frames=frames,
        source_fps=30.0,
        paired=False,
    )
    _convert_robot_job((record, contract, str(tmp_path)))
    converted = joblib.load(tmp_path / "mine__sample.pkl")["mine__sample"]
    source_index = {name: index for index, name in enumerate(source_names)}
    expected = np.array(
        [(source_index[name] + 1) / 100.0 for name in contract.actuator_joint_names],
        dtype=np.float32,
    )
    np.testing.assert_allclose(converted["dof"][0], expected)
    np.testing.assert_allclose(converted["root_rot"][0], [0.0, 0.0, 0.0, 1.0])
    assert converted["root_frame_contract_version"] == SONIC_ROOT_FRAME_CONTRACT
    assert converted["root_frame_correction_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert converted["root_height_policy"] == "mine_csv_current_mjcf_sole_optimized"
    assert converted["root_height_diagnostics"]["max_sole_penetration_m"] <= 0.002001
    for body_index, (name, axis) in enumerate(
        zip(contract.body_joint_names, contract.body_joint_axes, strict=True), start=1
    ):
        actuator_index = contract.actuator_joint_names.index(name)
        np.testing.assert_allclose(
            converted["pose_aa"][0, body_index],
            np.asarray(axis, dtype=np.float32) * expected[actuator_index],
        )


def test_public_legacy_root_frame_is_rotated_to_z_up_and_regrounded(tmp_path: Path) -> None:
    contract = _parse_mjcf(MJCF)
    frames = 8
    qpos = torch.zeros((frames, 28), dtype=torch.float32)
    qpos[:, 2] = 0.5
    # Legacy local +Z 指向世界 +Y；世界左乘 Rx(+90°) 后应回到 identity Z-up。
    qpos[:, 3:7] = torch.tensor([2**-0.5, -(2**-0.5), 0.0, 0.0])
    source = tmp_path / "legacy.pt"
    torch.save(
        {
            "contract_version": "genmo.bumi_music.v1",
            "source_motion_contract_version": LEGACY_PUBLIC_ROOT_CONTRACT,
            "qpos": qpos,
            "fps": 30,
            "robot_name": "bumi",
            "joint_names": list(contract.actuator_joint_names),
            "quaternion_convention": "wxyz",
            "qpos_order": "mujoco_native",
            "source_dataset": "finedance",
            "source_sample_id": "sample",
            "quality_accepted": True,
            "source_mjcf_sha256": (
                "482138b437dbdabd6171fa8d44b55db5d7125a228c95b69ce3d1159cafe8537c"
            ),
        },
        source,
    )
    record = SampleRecord(
        key="finedance__sample",
        dataset="finedance",
        sample_id="sample",
        split="train",
        robot_source=str(source),
        smpl_source=None,
        smpl_motion_key=None,
        source_frames=frames,
        source_fps=30.0,
        paired=False,
    )
    _convert_robot_job((record, contract, str(tmp_path)))
    converted = joblib.load(tmp_path / "finedance__sample.pkl")["finedance__sample"]
    np.testing.assert_allclose(
        converted["root_rot"],
        np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1)),
        atol=1e-6,
    )
    assert converted["root_frame_contract_version"] == SONIC_ROOT_FRAME_CONTRACT
    np.testing.assert_allclose(
        converted["root_frame_correction_wxyz"],
        [2**-0.5, 2**-0.5, 0.0, 0.0],
        atol=1e-7,
    )
    diagnostics = converted["root_height_diagnostics"]
    assert diagnostics["optimizer_success"] is True
    assert diagnostics["max_sole_penetration_m"] <= 0.002001


def test_root_frame_contract_mismatch_fails_closed(tmp_path: Path) -> None:
    contract = _parse_mjcf(MJCF)
    qpos = torch.zeros((2, 28), dtype=torch.float32)
    qpos[:, 3] = 1.0
    source = tmp_path / "wrong_contract.pt"
    torch.save(
        {
            "contract_version": "genmo.bumi_music.v1",
            "source_motion_contract_version": MINE_ROOT_CONTRACT,
            "qpos": qpos,
            "fps": 30,
            "robot_name": "bumi",
            "joint_names": list(contract.actuator_joint_names),
            "quaternion_convention": "wxyz",
            "qpos_order": "mujoco_native",
            "source_dataset": "finedance",
            "source_sample_id": "sample",
            "quality_accepted": True,
            "source_mjcf_sha256": (
                "482138b437dbdabd6171fa8d44b55db5d7125a228c95b69ce3d1159cafe8537c"
            ),
        },
        source,
    )
    record = SampleRecord(
        key="finedance__sample",
        dataset="finedance",
        sample_id="sample",
        split="train",
        robot_source=str(source),
        smpl_source=None,
        smpl_motion_key=None,
        source_frames=2,
        source_fps=30.0,
        paired=False,
    )
    with pytest.raises(ValueError, match="root frame 契约错误"):
        _convert_robot_job((record, contract, str(tmp_path)))


def test_smpl_resampling_matches_motion_lib_frame_grid() -> None:
    frames = 4
    pose = torch.zeros((frames, 72), dtype=torch.float32)
    transl = torch.stack(
        [torch.arange(frames, dtype=torch.float32), torch.zeros(frames), torch.ones(frames)], dim=1
    )
    pose_50, transl_50 = _resample_pose_and_translation(pose, transl, 30.0, 50.0)
    assert len(pose_50) == _runtime_frames(frames) == 5
    assert transl_50.shape == (5, 3)
    np.testing.assert_allclose(transl_50[0].numpy(), transl[0].numpy())
    assert float(transl_50[-1, 0]) < float(transl[-1, 0])


def test_contract_type_is_serializable() -> None:
    contract = MjcfContract(("a",), ("a",), ((0.0, 0.0, 1.0),), "sha")
    assert contract.actuator_joint_names == ("a",)
