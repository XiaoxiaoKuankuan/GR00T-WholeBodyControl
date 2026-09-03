"""验证 SONIC 通用 MotionLib 世界系角速度的有限差分契约。

本文件覆盖 BUMI3、G1 等机器人共同使用的 ``Humanoid_Batch`` 角速度实现，重点
检查四元数双覆盖 ``q == -q`` 不应改变物理角速度，以及内部帧必须采用中心差分并
与整数帧时刻对齐。测试使用解析可求导的单轴旋转，不依赖机器人资产、训练数据或
Isaac Lab 仿真，因此能够精确区分算法错误、数据问题和运行环境问题。
"""

import numpy as np
import pytest
import torch

from gear_sonic.isaac_utils.rotations import quat_sequence_angular_velocity
from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch


def _z_axis_quaternion(angles: torch.Tensor, w_last: bool = True) -> torch.Tensor:
    """把绕世界 Z 轴的角度序列转换为 ``[1, T, 1, 4]`` 四元数。"""
    half_angles = angles * 0.5
    xyz = torch.stack(
        [torch.zeros_like(angles), torch.zeros_like(angles), torch.sin(half_angles)],
        dim=-1,
    )
    scalar = torch.cos(half_angles).unsqueeze(-1)
    quaternions = torch.cat([xyz, scalar], dim=-1)
    if not w_last:
        quaternions = torch.cat([scalar, xyz], dim=-1)
    return quaternions[None, :, None, :]


@pytest.mark.parametrize("w_last", [True, False])
def test_common_quaternion_delta_is_invariant_to_sign(w_last: bool) -> None:
    """xyzw/wxyz 相邻帧任意切换 q/-q 时，物理角速度必须保持不变。"""
    dt = 0.02
    omega = 1.75
    times = torch.arange(41, dtype=torch.float64) * dt
    continuous = _z_axis_quaternion(times * omega, w_last=w_last)
    sign_flipped = continuous.clone()
    sign_flipped[:, 7:19] *= -1.0
    sign_flipped[:, 28::2] *= -1.0

    expected = torch.zeros(1, len(times), 1, 3, dtype=torch.float64)
    expected[..., 2] = omega
    continuous_velocity = quat_sequence_angular_velocity(continuous, dt, w_last=w_last)
    sign_flipped_velocity = quat_sequence_angular_velocity(
        sign_flipped, dt, w_last=w_last
    )

    torch.testing.assert_close(continuous_velocity, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(sign_flipped_velocity, expected, atol=1e-10, rtol=1e-10)


def test_angular_velocity_uses_centered_integer_frame_timestamps() -> None:
    """二次角度轨迹的内部角速度应精确等于对应整数帧的一阶导数。"""
    dt = 0.02
    acceleration = 3.0
    times = torch.arange(31, dtype=torch.float64) * dt
    angles = 0.5 * acceleration * times.square()
    quaternions = _z_axis_quaternion(angles)

    velocity = Humanoid_Batch._compute_angular_velocity(
        quaternions, dt, guassian_filter=False
    )[0, :, 0]
    expected_z = acceleration * times

    # 内部中心差分对二次函数的一阶导数是精确的；端点按单边区间的中点速度定义。
    torch.testing.assert_close(
        velocity[1:-1, 2], expected_z[1:-1], atol=1e-10, rtol=1e-10
    )
    assert velocity[0, 2].item() == pytest.approx(acceleration * dt * 0.5)
    assert velocity[-1, 2].item() == pytest.approx(
        acceleration * (times[-1].item() - dt * 0.5)
    )
    torch.testing.assert_close(
        velocity[:, :2], torch.zeros_like(velocity[:, :2]), atol=1e-12, rtol=0.0
    )


@pytest.mark.parametrize("num_frames", [1, 2])
def test_angular_velocity_handles_short_sequences(num_frames: int) -> None:
    """一帧返回零，两帧在两个端点复用同一个有限差分且保持滤波后有限。"""
    dt = 0.02
    omega = 0.8
    times = torch.arange(num_frames, dtype=torch.float32) * dt
    quaternions = _z_axis_quaternion(times * omega)

    velocity = Humanoid_Batch._compute_angular_velocity(
        quaternions, dt, guassian_filter=True
    )

    assert velocity.shape == (1, num_frames, 1, 3)
    assert torch.isfinite(velocity).all()
    if num_frames == 1:
        assert torch.count_nonzero(velocity) == 0
    else:
        np.testing.assert_allclose(
            velocity[0, :, 0, 2].numpy(), np.full(2, omega), atol=2e-5, rtol=2e-5
        )
