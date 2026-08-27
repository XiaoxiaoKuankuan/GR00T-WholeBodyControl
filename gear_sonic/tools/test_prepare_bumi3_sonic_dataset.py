# SPDX-License-Identifier: Apache-2.0
"""验证 BUMI3 SONIC 数据转换中最容易产生静默错位的纯函数契约。

这些测试不启动 Isaac Sim，也不把静态检查冒充训练验证。覆盖重点是：当前 BUMI3
MJCF 的 21 个 actuator/body joint 集合、源 ``joint_names`` 任意顺序到 SONIC
actuator 顺序的重排、wxyz→xyzw 根四元数转换、关节轴角写入 body traversal 顺序，
以及 30Hz→50Hz 必须复现 motion-lib ``arange`` 末帧排除规则。全量文件计数、有限
值、配对和 SHA256 由准备工具的 ``validate`` 子命令另行执行。
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from gear_sonic.tools.prepare_bumi3_sonic_dataset import (
    MjcfContract,
    SampleRecord,
    _convert_robot_job,
    _parse_mjcf,
    _resample_pose_and_translation,
    _runtime_frames,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MJCF = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/bumi3.xml"


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
            "qpos": qpos,
            "fps": 30,
            "robot_name": "bumi",
            "joint_names": list(source_names),
            "quaternion_convention": "wxyz",
            "qpos_order": "mujoco_native",
            "source_dataset": "mine",
            "source_sample_id": "sample",
            "quality_accepted": True,
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
    for body_index, (name, axis) in enumerate(
        zip(contract.body_joint_names, contract.body_joint_axes, strict=True), start=1
    ):
        actuator_index = contract.actuator_joint_names.index(name)
        np.testing.assert_allclose(
            converted["pose_aa"][0, body_index],
            np.asarray(axis, dtype=np.float32) * expected[actuator_index],
        )


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
