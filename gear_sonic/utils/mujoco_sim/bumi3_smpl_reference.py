"""读取 BUMI SONIC 的 SMPL 参考，并复刻训练侧人体编码器的观测。

本模块只依赖 NumPy、SciPy 和 joblib，不需要启动 Isaac Lab 或加载 SMPL 身体模型。
输入是已经用于 SONIC 训练的 50 FPS PKL/NPZ，必须提供 pose_aa[T,72] 和
smpl_joints[T,24,3]。pose_aa 的根旋转仍为 Y-up，关键点已经由离线数据生成器
转为 Z-up；这里仅转换根朝向，并右乘 SMPL 基准旋转的逆，随后用每帧根朝向的逆
旋转关键点，严格对应训练的 smpl_joints_multi_future_local。不能再次转关键点轴、
减去 pelvis 或叠加 transl，否则会改变 checkpoint 实际见到的输入。

人体编码器读取当前起连续十帧：先展开 720 维局部关键点，再展开 60 维相对机器人
完整根朝向的六维旋转表示。等待播放时十帧全部固定为当前帧，末尾截断或循环由调用方
提供的索引决定。该模块不从人体姿态推算机器人关节，不改源数据、不参与动力学。
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation


SMPL_FUTURE_FRAMES = 10
SMPL_FUTURE_STRIDE = 1
SMPL_TOKENIZER_DIM = 780


@dataclass(frozen=True)
class SmplReference:
    """保留每帧训练兼容的局部关键点与 Z-up 根朝向；旋转四元数采用 wxyz。"""

    local_joints: np.ndarray
    root_quat_wxyz: np.ndarray
    fps: float
    name: str

    @property
    def num_frames(self) -> int:
        return int(self.local_joints.shape[0])


def load_smpl_reference(
    path: str | Path, *, motion_key: str | None = None, target_fps: float = 50.0,
) -> SmplReference:
    """读取直接字典或具名动作容器；缺字段、非有限值、错误帧率或形状均显式拒绝。"""

    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".pkl":
        content = joblib.load(path)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            content = dict(archive)
    else:
        raise ValueError("SMPL 入口仅支持训练格式的 PKL/NPZ，必须包含 pose_aa 和 smpl_joints")
    if not isinstance(content, dict):
        raise ValueError("SMPL 文件顶层必须为字典")
    name = motion_key or path.stem
    if "pose_aa" not in content:
        keys = [key for key, value in content.items() if isinstance(value, dict)]
        if motion_key is None:
            if len(keys) != 1:
                raise ValueError(f"SMPL 容器必须指定 motion_key，可选值: {keys}")
            name = keys[0]
        if name not in content or not isinstance(content[name], dict):
            raise ValueError(f"SMPL 容器不存在动作 {name!r}")
        content = content[name]
    missing = {"pose_aa", "smpl_joints", "fps"} - content.keys()
    if missing:
        raise ValueError(f"SMPL 缺少训练字段: {sorted(missing)}")
    pose = np.asarray(content["pose_aa"], dtype=np.float64)
    joints = np.asarray(content["smpl_joints"], dtype=np.float64)
    fps_value = np.asarray(content["fps"], dtype=np.float64)
    if fps_value.size != 1:
        raise ValueError("SMPL fps 必须为单个数值")
    fps = float(fps_value.reshape(-1)[0])
    if not np.isfinite(fps) or not np.isclose(fps, target_fps, rtol=0.0, atol=1e-6):
        raise ValueError(f"SMPL fps 必须为 {target_fps}，实际为 {fps}；入口不自动重采样")
    if pose.ndim != 2 or pose.shape[1] != 72 or pose.shape[0] < 1:
        raise ValueError(f"SMPL pose_aa 应为非空 [T,72]，实际为 {pose.shape}")
    if joints.shape != (pose.shape[0], 24, 3):
        raise ValueError(f"SMPL smpl_joints 应为 [T,24,3] 并与 pose_aa 同帧数，实际为 {joints.shape}")
    if not np.isfinite(pose).all() or not np.isfinite(joints).all():
        raise ValueError("SMPL pose_aa/smpl_joints 含 NaN/Inf")

    # 对齐 commands.py 的 smpl_root_quat_w_multi_future：左乘转轴，右乘基准旋转逆。
    root = (
        Rotation.from_rotvec([np.pi / 2.0, 0.0, 0.0])
        * Rotation.from_rotvec(pose[:, :3])
        * Rotation.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
    )
    local_joints = np.einsum("tji,tkj->tki", root.as_matrix(), joints)
    return SmplReference(
        local_joints=local_joints.astype(np.float32),
        root_quat_wxyz=root.as_quat()[:, [3, 0, 1, 2]],
        fps=fps, name=name,
    )


def build_smpl_tokenizer(
    reference: SmplReference, indices: np.ndarray,
    robot_anchor_wxyz: np.ndarray, heading_delta_wxyz: np.ndarray,
) -> np.ndarray:
    """按导出模型的字段顺序拼接 720+60 维；参考水平对齐不改变其局部关键点。"""

    if np.asarray(indices).shape != (SMPL_FUTURE_FRAMES,):
        raise ValueError("SMPL tokenizer 必须恰好读取十帧参考")
    root = Rotation.from_quat(reference.root_quat_wxyz[indices][:, [1, 2, 3, 0]])
    robot = Rotation.from_quat(np.asarray(robot_anchor_wxyz)[[1, 2, 3, 0]])
    delta = Rotation.from_quat(np.asarray(heading_delta_wxyz)[[1, 2, 3, 0]])
    orientation = (robot.inv() * delta * root).as_matrix()[..., :2]
    return np.concatenate((reference.local_joints[indices].reshape(-1), orientation.reshape(-1))).astype(
        np.float32
    )
