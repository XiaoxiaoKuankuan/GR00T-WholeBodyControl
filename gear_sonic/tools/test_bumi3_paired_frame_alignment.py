# SPDX-License-Identifier: Apache-2.0
"""验证 MotionLib 的 BUMI3 Robot/SMPL 可选尾帧对齐契约。

这些测试只覆盖纯配置与长度决策，不加载 Isaac Lab。目标是保证默认
``strict`` 模式继续保留历史行为，而 ``trim_trailing`` 只接受两侧都已是目标
帧率、SMPL 三个时间字段内部一致且尾差不超过上限的配对。任何可能掩盖
错帧率、缺字段或较大时间偏移的问题都必须在 FK 前失败，不能通过重复帧
或静默裁剪进入训练。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gear_sonic.utils.motion_lib import motion_lib_base
from gear_sonic.utils.motion_lib.motion_lib_base import (
    FixHeightMode,
    MotionLibBase,
    resolve_paired_frame_alignment,
)


def _smpl(frames: int, fps: float = 50.0) -> dict:
    """创建字段长度一致的最小 SMPL 时间负载。"""

    return {
        "fps": fps,
        "pose_aa": np.zeros((frames, 72), dtype=np.float32),
        "smpl_joints": np.zeros((frames, 24, 3), dtype=np.float32),
        "transl": np.zeros((frames, 3), dtype=np.float32),
    }


class _FakeMeshParser:
    """只保留本测试需要的 FK 时间维，避免引入 Isaac Lab 或真实 MJCF。"""

    def fk_batch(self, pose_aa: torch.Tensor, trans: torch.Tensor, **_kwargs) -> dict:
        frames = pose_aa.shape[1]
        return {
            "global_translation": trans.unsqueeze(2).repeat(1, 1, 22, 1),
            "dof_pos": torch.zeros((1, frames, 21), dtype=trans.dtype),
        }


def _minimal_motion_lib(alignment_cfg: dict) -> MotionLibBase:
    """绕过重型初始化，构造可执行单条加载路径的最小 MotionLib。"""

    library = MotionLibBase.__new__(MotionLibBase)
    library.target_fps = 50.0
    library.paired_frame_alignment_cfg = alignment_cfg
    library.mesh_parsers = _FakeMeshParser()
    library.use_parallel_fk = False
    library.m_cfg = {}
    library.randomize_upper_body_poses = False
    library.randomize_wrist_poses = False
    library.smpl_data = [object()]
    library.soma_data = None
    library.object_data = None
    library.has_action = False
    library.vid_smpl_pose = None
    library.curr_motion_keys = ["unit__clip"]
    library.foot_detect = lambda positions, _threshold, _height: (
        torch.zeros(positions.shape[0]),
        torch.zeros(positions.shape[0]),
    )
    return library


def _robot(frames: int) -> dict:
    """创建可辨认时间索引的最小 50 Hz Robot 动作。"""

    root_trans = np.zeros((frames, 3), dtype=np.float32)
    root_trans[:, 0] = np.arange(frames, dtype=np.float32)
    return {
        "root_trans_offset": root_trans,
        "pose_aa": np.zeros((frames, 22, 3), dtype=np.float32),
        "fps": 50.0,
    }


def _indexed_smpl(frames: int) -> dict:
    """创建三个时间字段都携带帧号的最小 50 Hz SMPL 动作。"""

    data = _smpl(frames)
    frame_values = np.arange(frames, dtype=np.float32)
    data["pose_aa"][:, 0] = frame_values
    data["smpl_joints"][:, 0, 0] = frame_values
    data["transl"][:, 0] = frame_values
    return data


def test_strict_mode_preserves_robot_length() -> None:
    assert resolve_paired_frame_alignment(10, 30.0, {}, 50.0, {"mode": "strict"}) == 10


def test_strict_mode_rejects_target_fps_length_mismatch() -> None:
    with pytest.raises(ValueError, match="长度完全一致"):
        resolve_paired_frame_alignment(10, 50.0, _smpl(11), 50.0, {"mode": "strict"})


@pytest.mark.parametrize(
    ("robot_frames", "smpl_frames", "expected"),
    [(10, 10, 10), (10, 9, 9), (10, 8, 8), (8, 10, 8)],
)
def test_trim_trailing_accepts_zero_to_two_frame_delta(
    robot_frames: int, smpl_frames: int, expected: int
) -> None:
    actual = resolve_paired_frame_alignment(
        robot_frames,
        50.0,
        _smpl(smpl_frames),
        50.0,
        {"mode": "trim_trailing", "max_frame_delta": 2},
    )
    assert actual == expected


def test_trim_trailing_rejects_large_delta_and_fps_mismatch() -> None:
    with pytest.raises(ValueError, match="尾帧差超过"):
        resolve_paired_frame_alignment(
            10,
            50.0,
            _smpl(7),
            50.0,
            {"mode": "trim_trailing", "max_frame_delta": 2},
        )
    with pytest.raises(ValueError, match="已经处于目标帧率"):
        resolve_paired_frame_alignment(
            10,
            30.0,
            _smpl(10),
            50.0,
            {"mode": "trim_trailing", "max_frame_delta": 2},
        )


def test_trim_trailing_rejects_missing_or_inconsistent_smpl_fields() -> None:
    missing = _smpl(10)
    del missing["transl"]
    with pytest.raises(ValueError, match="缺少时间字段"):
        resolve_paired_frame_alignment(
            10,
            50.0,
            missing,
            50.0,
            {"mode": "trim_trailing", "max_frame_delta": 2},
        )

    inconsistent = _smpl(10)
    inconsistent["smpl_joints"] = np.zeros((9, 24, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="长度不一致"):
        resolve_paired_frame_alignment(
            10,
            50.0,
            inconsistent,
            50.0,
            {"mode": "trim_trailing", "max_frame_delta": 2},
        )


def test_trim_trailing_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="只支持"):
        resolve_paired_frame_alignment(10, 50.0, _smpl(10), 50.0, {"mode": "unknown"})
    with pytest.raises(ValueError, match="非负整数"):
        resolve_paired_frame_alignment(
            10,
            50.0,
            _smpl(10),
            50.0,
            {"mode": "trim_trailing", "max_frame_delta": True},
        )


def test_random_segment_reuses_same_robot_and_smpl_window(monkeypatch) -> None:
    """随机截段必须让 Robot、SMPL pose/joints/transl 共用同一时间窗。"""

    library = _minimal_motion_lib({"mode": "strict", "max_frame_delta": 0})
    monkeypatch.setattr(motion_lib_base.random, "randint", lambda _left, _right: 3)
    result = library.load_motion_with_skeleton(
        ids=[0],
        motion_data_list=[_robot(10)],
        smpl_data_list=[_indexed_smpl(10)],
        object_data_list=None,
        soma_data_list=None,
        fix_height=FixHeightMode.no_fix,
        target_heading=None,
        max_len=4,
        is_evaluation=True,
        queue=None,
        pid=1,
    )[0][1]
    expected = torch.arange(3, 7, dtype=torch.float32)
    assert torch.equal(result.global_translation[:, 0, 0], expected)
    assert torch.equal(result.smpl_pose[:, 0], expected)
    assert torch.equal(result.smpl_joints[:, 0, 0], expected)
    assert torch.equal(result.smpl_transl[:, 0], expected)


def test_trimmed_pair_freezes_robot_and_smpl_at_same_frame(monkeypatch) -> None:
    """尾帧裁剪后，freeze-frame augmentation 仍须同步冻结两侧数据。"""

    library = _minimal_motion_lib({"mode": "trim_trailing", "max_frame_delta": 2})
    library.m_cfg = {"freeze_frame_aug": True, "freeze_frame_prob": 1.0}
    monkeypatch.setattr(motion_lib_base.np.random, "random", lambda: 0.0)
    monkeypatch.setattr(motion_lib_base.np.random, "randint", lambda _left, _right=None: 2)
    result = library.load_motion_with_skeleton(
        ids=[0],
        motion_data_list=[_robot(10)],
        smpl_data_list=[_indexed_smpl(8)],
        object_data_list=None,
        soma_data_list=None,
        fix_height=FixHeightMode.no_fix,
        target_heading=None,
        max_len=-1,
        is_evaluation=False,
        queue=None,
        pid=1,
    )[0][1]
    expected = torch.tensor([0, 1, 2, 2, 2, 2, 2, 2], dtype=torch.float32)
    assert torch.equal(result.global_translation[:, 0, 0], expected)
    assert torch.equal(result.smpl_pose[:, 0], expected)
    assert torch.equal(result.smpl_joints[:, 0, 0], expected)
    assert torch.equal(result.smpl_transl[:, 0], expected)
