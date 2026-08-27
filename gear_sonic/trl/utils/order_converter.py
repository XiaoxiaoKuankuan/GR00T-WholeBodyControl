"""
IsaacLab <-> MuJoCo ordering conversion utilities.

This module provides utilities for converting both DOF and body ordering
between IsaacLab and MuJoCo conventions for humanoid robots.

qpos format: [root_trans(3), root_quat(4), dof_angles(N)]
"""

from abc import ABC

import numpy as np
import torch


class IsaacLabMuJoCoConverter(ABC):
    """Abstract base class for DOF/body order conversion between IsaacLab and MuJoCo.

    Subclasses must define DOF_MAPPINGS for their specific robot.
    """

    ROOT_QPOS_OFFSET = 7  # root_trans(3) + root_quat(4)
    DOF_MAPPINGS: dict = {}  # Must be overridden by subclasses
    VALID_DOF_ORDERS = ("mujoco", "isaaclab")

    def convert(self, data: torch.Tensor, from_order: str, to_order: str) -> torch.Tensor:
        """Convert DOF order between conventions.

        Auto-detects input format:
            - qpos:           [..., 7 + num_dof]           -> reorder DOF portion, keep root
            - dof angles:     [..., num_dof]                -> reorder directly
            - body transforms [..., num_dof + 1, D]         -> keep root body, reorder DOF bodies
            - body transforms [..., num_dof, D]             -> reorder directly
            - rotation mats   [..., num_dof (+ 1), 3, 3]   -> same as above on dim -3

        Note: Body reordering for per-body transforms uses DOF_MAPPINGS (not
        BODY_MAPPINGS) because each DOF link has exactly one parent body in G1,
        so DOF order and DOF-body order are consistent by construction.  The
        separate BODY_MAPPINGS include the root (pelvis) at index 0 and are used
        only when the full 30-body ordering is needed.
        """
        if from_order == to_order:
            return data

        mapping = self.DOF_MAPPINGS[(from_order, to_order)]
        last_dim = data.shape[-1]

        if last_dim == self.ROOT_QPOS_OFFSET + self.num_dof:
            # qpos: [..., 7 + num_dof]
            return torch.cat(
                [
                    data[..., : self.ROOT_QPOS_OFFSET],
                    data[..., self.ROOT_QPOS_OFFSET :][..., mapping],
                ],
                dim=-1,
            )
        elif last_dim == self.num_dof:
            # Raw DOF angles: [..., num_dof]
            return data[..., mapping]
        else:
            # Per-body transforms: body dim is -3 for [..., J, 3, 3], else -2 for [..., J, D]
            body_dim = (
                -3 if (data.ndim >= 3 and data.shape[-1] == 3 and data.shape[-2] == 3) else -2
            )
            num_bodies = data.shape[body_dim]

            if num_bodies == self.num_dof + 1:
                # Includes root at index 0 — keep root, reorder DOF bodies
                body_order = [0] + [m + 1 for m in mapping]
            elif num_bodies == self.num_dof:
                body_order = mapping
            else:
                raise ValueError(
                    f"Cannot detect format: last_dim={last_dim}, body_dim size={num_bodies}, "
                    f"expected num_dof+1={self.num_dof + 1} or num_dof={self.num_dof}"
                )

            if body_dim == -2:
                return data[..., body_order, :]
            else:  # body_dim == -3
                return data[..., body_order, :, :]

    def to_mujoco(self, data: torch.Tensor) -> torch.Tensor:
        """Convert from IsaacLab to MuJoCo DOF order."""
        return self.convert(data, from_order="isaaclab", to_order="mujoco")

    def to_isaaclab(self, data: torch.Tensor) -> torch.Tensor:
        """Convert from MuJoCo to IsaacLab DOF order."""
        return self.convert(data, from_order="mujoco", to_order="isaaclab")

    @property
    def num_dof(self) -> int:
        """Number of actuated DOFs (excluding root)."""
        return len(self.DOF_MAPPINGS[(self.VALID_DOF_ORDERS[0], self.VALID_DOF_ORDERS[1])])

    @property
    def isaaclab_to_mujoco_dof(self):
        """DOF reorder indices: IsaacLab -> MuJoCo."""
        return self.DOF_MAPPINGS[("isaaclab", "mujoco")]

    @property
    def mujoco_to_isaaclab_dof(self):
        """DOF reorder indices: MuJoCo -> IsaacLab."""
        return self.DOF_MAPPINGS[("mujoco", "isaaclab")]

    @property
    def isaaclab_to_mujoco_body(self):
        """Body reorder indices: IsaacLab -> MuJoCo."""
        return self.BODY_MAPPINGS[("isaaclab", "mujoco")]

    @property
    def mujoco_to_isaaclab_body(self):
        """Body reorder indices: MuJoCo -> IsaacLab."""
        return self.BODY_MAPPINGS[("mujoco", "isaaclab")]

    def get_isaaclab_to_mujoco_mapping(self):
        """Return the full mapping dict for body/DOF reordering."""
        mapping = {
            "isaaclab_joints": self.JOINT_NAMES,
            "isaaclab_to_mujoco_dof": self.isaaclab_to_mujoco_dof,
            "mujoco_to_isaaclab_dof": self.mujoco_to_isaaclab_dof,
            "isaaclab_to_mujoco_body": self.isaaclab_to_mujoco_body,
            "mujoco_to_isaaclab_body": self.mujoco_to_isaaclab_body,
        }
        if hasattr(self, "LOWER_JOINT_INDICES_MUJOCO"):
            mapping["lower_joint_indices_mujoco"] = self.LOWER_JOINT_INDICES_MUJOCO
        return mapping


class G1Converter(IsaacLabMuJoCoConverter):
    """G1 ordering converter.

    Imports G1 body/DOF/joint mappings from gear_sonic.envs.manager_env.robots.g1.
    """

    def __init__(self):
        # Lazy import to avoid circular dependency:
        # order_converter -> g1 -> mdp/__init__ -> commands -> order_converter
        from gear_sonic.envs.manager_env.robots.g1 import (
            G1_ISAACLAB_JOINTS,
            G1_ISAACLAB_TO_MUJOCO_BODY,
            G1_ISAACLAB_TO_MUJOCO_DOF,
            G1_MUJOCO_TO_ISAACLAB_BODY,
            G1_MUJOCO_TO_ISAACLAB_DOF,
        )

        self.JOINT_NAMES = G1_ISAACLAB_JOINTS
        self.DOF_MAPPINGS = {
            ("isaaclab", "mujoco"): G1_ISAACLAB_TO_MUJOCO_DOF,
            ("mujoco", "isaaclab"): G1_MUJOCO_TO_ISAACLAB_DOF,
        }
        self.BODY_MAPPINGS = {
            ("isaaclab", "mujoco"): G1_ISAACLAB_TO_MUJOCO_BODY,
            ("mujoco", "isaaclab"): G1_MUJOCO_TO_ISAACLAB_BODY,
        }

    # Body subset names for MPJPE metrics (used by reconstruction_trainer)
    VR_3POINTS_BODY_NAMES = ["torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

    @property
    def vr_3points_mujoco_indices(self):
        """VR 3-point body indices in full (30-body) MuJoCo body order.

        These index into the full body array after isaaclab_to_mujoco_body
        reordering, NOT the 14-body motion.yaml body_names subset.
        """
        mj_names = [self.JOINT_NAMES[i] for i in self.isaaclab_to_mujoco_body]
        return [mj_names.index(n) for n in self.VR_3POINTS_BODY_NAMES]

    @property
    def foot_mujoco_indices(self):
        """Foot body indices in full (30-body) MuJoCo body order.

        These index into the full body array after isaaclab_to_mujoco_body
        reordering, NOT the 14-body motion.yaml body_names subset.
        """
        mj_names = [self.JOINT_NAMES[i] for i in self.isaaclab_to_mujoco_body]
        return [mj_names.index(n) for n in self.FOOT_BODY_NAMES]

    @property
    def isaaclab_to_mujoco_dof(self):
        """DOF reorder indices: IsaacLab -> MuJoCo."""
        return self.DOF_MAPPINGS[("isaaclab", "mujoco")]

    @property
    def mujoco_to_isaaclab_dof(self):
        """DOF reorder indices: MuJoCo -> IsaacLab."""
        return self.DOF_MAPPINGS[("mujoco", "isaaclab")]

    @property
    def isaaclab_to_mujoco_body(self):
        """Body reorder indices: IsaacLab -> MuJoCo."""
        return self.BODY_MAPPINGS[("isaaclab", "mujoco")]

    @property
    def mujoco_to_isaaclab_body(self):
        """Body reorder indices: MuJoCo -> IsaacLab."""
        return self.BODY_MAPPINGS[("mujoco", "isaaclab")]

    def get_isaaclab_to_mujoco_mapping(self):
        """Return the full mapping dict for body/DOF reordering."""
        return {
            "isaaclab_joints": self.JOINT_NAMES,
            "isaaclab_to_mujoco_dof": self.isaaclab_to_mujoco_dof,
            "mujoco_to_isaaclab_dof": self.mujoco_to_isaaclab_dof,
            "isaaclab_to_mujoco_body": self.isaaclab_to_mujoco_body,
            "mujoco_to_isaaclab_body": self.mujoco_to_isaaclab_body,
        }


class H2Converter(IsaacLabMuJoCoConverter):
    """H2 robot joint/body order converter between IsaacLab and MuJoCo conventions."""

    def __init__(self):
        from gear_sonic.envs.manager_env.robots.h2 import (
            H2_ISAACLAB_JOINTS,
            H2_ISAACLAB_TO_MUJOCO_BODY,
            H2_ISAACLAB_TO_MUJOCO_DOF,
            H2_MUJOCO_TO_ISAACLAB_BODY,
            H2_MUJOCO_TO_ISAACLAB_DOF,
        )

        self.JOINT_NAMES = H2_ISAACLAB_JOINTS
        self.DOF_MAPPINGS = {
            ("isaaclab", "mujoco"): H2_ISAACLAB_TO_MUJOCO_DOF,
            ("mujoco", "isaaclab"): H2_MUJOCO_TO_ISAACLAB_DOF,
        }
        self.BODY_MAPPINGS = {
            ("isaaclab", "mujoco"): H2_ISAACLAB_TO_MUJOCO_BODY,
            ("mujoco", "isaaclab"): H2_MUJOCO_TO_ISAACLAB_BODY,
        }

    VR_3POINTS_BODY_NAMES = ["torso_link", "left_wrist_pitch_link", "right_wrist_pitch_link"]
    FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


class Bumi3Converter(IsaacLabMuJoCoConverter):
    """BUMI3 的 Isaac Lab/MuJoCo 关节与刚体顺序转换器。

    BUMI3 机器人模块采用延迟导入，避免 ``order_converter -> robots -> mdp ->
    commands -> order_converter`` 形成循环依赖。转换本身沿用 SONIC 的通用张量
    逻辑，不增加网络或训练算法分支。
    """

    VR_3POINTS_BODY_NAMES = [
        "l_elbow_pitch_link",
        "r_elbow_pitch_link",
        "waist_yaw_link",
    ]
    FOOT_BODY_NAMES = ["l_ankle_roll_link", "r_ankle_roll_link"]

    def __init__(self):
        from gear_sonic.envs.manager_env.robots.bumi3 import (
            BUMI3_ISAACLAB_BODY_NAMES,
            BUMI3_ISAACLAB_TO_MUJOCO_BODY,
            BUMI3_ISAACLAB_TO_MUJOCO_DOF,
            BUMI3_LOWER_JOINT_INDICES_MUJOCO,
            BUMI3_MUJOCO_TO_ISAACLAB_BODY,
            BUMI3_MUJOCO_TO_ISAACLAB_DOF,
        )

        self.JOINT_NAMES = BUMI3_ISAACLAB_BODY_NAMES
        self.DOF_MAPPINGS = {
            ("isaaclab", "mujoco"): BUMI3_ISAACLAB_TO_MUJOCO_DOF,
            ("mujoco", "isaaclab"): BUMI3_MUJOCO_TO_ISAACLAB_DOF,
        }
        self.BODY_MAPPINGS = {
            ("isaaclab", "mujoco"): BUMI3_ISAACLAB_TO_MUJOCO_BODY,
            ("mujoco", "isaaclab"): BUMI3_MUJOCO_TO_ISAACLAB_BODY,
        }
        self.LOWER_JOINT_INDICES_MUJOCO = BUMI3_LOWER_JOINT_INDICES_MUJOCO

    @property
    def vr_3points_mujoco_indices(self):
        """返回 BUMI3 三个跟踪点在完整 MuJoCo body 顺序中的索引。"""
        mujoco_names = [self.JOINT_NAMES[i] for i in self.isaaclab_to_mujoco_body]
        return [mujoco_names.index(name) for name in self.VR_3POINTS_BODY_NAMES]

    @property
    def foot_mujoco_indices(self):
        """返回 BUMI3 双脚在完整 MuJoCo body 顺序中的索引。"""
        mujoco_names = [self.JOINT_NAMES[i] for i in self.isaaclab_to_mujoco_body]
        return [mujoco_names.index(name) for name in self.FOOT_BODY_NAMES]


def get_order_converter(robot_type: str | None = None) -> IsaacLabMuJoCoConverter:
    """按机器人类型创建顺序转换器，缺省保持原 G1 行为。"""

    normalized_type = (robot_type or "g1").lower()
    converter_types = {
        "g1": G1Converter,
        "g1_model_12_dex": G1Converter,
        "h2": H2Converter,
        "bumi3": Bumi3Converter,
    }
    if normalized_type not in converter_types:
        supported = ", ".join(sorted(converter_types))
        raise ValueError(f"Unsupported robot_type={robot_type!r}; supported: {supported}")
    return converter_types[normalized_type]()


def load_qpos_from_csv(csv_path: str) -> torch.Tensor:
    """Load qpos [T, D] from CSV."""
    import pandas as pd

    return torch.from_numpy(pd.read_csv(csv_path).values.astype(np.float32))


def save_qpos_to_csv(qpos: torch.Tensor, csv_path: str):
    """Save qpos to CSV."""
    import pandas as pd

    data = qpos[0].cpu().numpy() if qpos.dim() == 3 else qpos.cpu().numpy()
    pd.DataFrame(data).to_csv(csv_path, index=False)


if __name__ == "__main__":
    from isaacsim import SimulationApp

    _sim_app = SimulationApp({"headless": True})

    # Test DOF conversion round-trip
    converter = G1Converter()

    # Create random qpos (G1 has 29 DOFs)
    T, num_dof = 30, 29
    qpos = torch.cat(
        [
            torch.tensor([[0.0, 0.0, 1.0]]).expand(T, 3),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(T, 4),
            torch.randn(T, num_dof) * 0.3,
        ],
        dim=-1,
    )

    # Test round-trip conversion
    qpos_mujoco = converter.to_mujoco(qpos)
    qpos_back = converter.to_isaaclab(qpos_mujoco)
    print(f"Round-trip error: {(qpos - qpos_back).abs().max():.2e}")

    # Test explicit convert API
    qpos_mujoco2 = converter.convert(qpos, from_order="isaaclab", to_order="mujoco")
    print(f"to_mujoco vs convert match: {torch.allclose(qpos_mujoco, qpos_mujoco2)}")

    _sim_app.close()
