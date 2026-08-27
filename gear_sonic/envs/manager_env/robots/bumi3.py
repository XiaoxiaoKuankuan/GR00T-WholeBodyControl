# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""BUMI3 在 Isaac Lab 与 SONIC 中使用的原生机器人配置。

本模块只复刻参考 ``Bumi_CFG`` 当前实际生效的 URDF、初始姿态、刚体属性和
延迟 PD 执行器参数，不依赖 ``NoetixRobot`` Python 包。它同时以关节/刚体名称
自动构造 Isaac Lab 与 MuJoCo 的双向顺序映射，并在导入时核对 BUMI3 参考配置
给出的 21 自由度排列，防止训练数据、运动学模型和仿真关节顺序静默错位。内部仍沿用
SONIC 的 ``g1`` 编码器/解码器键名；这里的 BUMI3 仅代表机器人载体与顺序契约。
"""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from gear_sonic.envs.manager_env.mdp.actuators import DelayedImplicitActuatorCfg


ASSET_DIR = "gear_sonic/data/assets"


BUMI3_MUJOCO_DOF_NAMES = [
    "waist_yaw_joint",
    "l_arm_pitch_joint",
    "l_arm_roll_joint",
    "l_arm_yaw_joint",
    "l_elbow_pitch_joint",
    "r_arm_pitch_joint",
    "r_arm_roll_joint",
    "r_arm_yaw_joint",
    "r_elbow_pitch_joint",
    "l_leg_pitch_joint",
    "l_leg_roll_joint",
    "l_leg_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_leg_pitch_joint",
    "r_leg_roll_joint",
    "r_leg_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
]

BUMI3_ISAACLAB_DOF_NAMES = [
    "l_leg_pitch_joint",
    "r_leg_pitch_joint",
    "waist_yaw_joint",
    "l_leg_roll_joint",
    "r_leg_roll_joint",
    "l_arm_pitch_joint",
    "r_arm_pitch_joint",
    "l_leg_yaw_joint",
    "r_leg_yaw_joint",
    "l_arm_roll_joint",
    "r_arm_roll_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_arm_yaw_joint",
    "r_arm_yaw_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_elbow_pitch_joint",
    "r_elbow_pitch_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
]

BUMI3_MUJOCO_BODY_NAMES = [
    "base_link",
    "waist_yaw_link",
    "l_arm_pitch_link",
    "l_arm_roll_link",
    "l_arm_yaw_link",
    "l_elbow_pitch_link",
    "r_arm_pitch_link",
    "r_arm_roll_link",
    "r_arm_yaw_link",
    "r_elbow_pitch_link",
    "l_leg_pitch_link",
    "l_leg_roll_link",
    "l_leg_yaw_link",
    "l_knee_pitch_link",
    "l_ankle_pitch_link",
    "l_ankle_roll_link",
    "r_leg_pitch_link",
    "r_leg_roll_link",
    "r_leg_yaw_link",
    "r_knee_pitch_link",
    "r_ankle_pitch_link",
    "r_ankle_roll_link",
]

# Isaac Lab 的刚体顺序等于根刚体加 URDF 导入后的 21 个关节子刚体顺序。
BUMI3_ISAACLAB_BODY_NAMES = [
    "base_link",
    "l_leg_pitch_link",
    "r_leg_pitch_link",
    "waist_yaw_link",
    "l_leg_roll_link",
    "r_leg_roll_link",
    "l_arm_pitch_link",
    "r_arm_pitch_link",
    "l_leg_yaw_link",
    "r_leg_yaw_link",
    "l_arm_roll_link",
    "r_arm_roll_link",
    "l_knee_pitch_link",
    "r_knee_pitch_link",
    "l_arm_yaw_link",
    "r_arm_yaw_link",
    "l_ankle_pitch_link",
    "r_ankle_pitch_link",
    "l_elbow_pitch_link",
    "r_elbow_pitch_link",
    "l_ankle_roll_link",
    "r_ankle_roll_link",
]


def _target_order_indices(source_names: list[str], target_names: list[str]) -> list[int]:
    """按名称生成从 source 排列到 target 排列所需的索引。"""

    assert len(source_names) == len(set(source_names)), "source 名称存在重复"
    assert len(target_names) == len(set(target_names)), "target 名称存在重复"
    assert set(source_names) == set(target_names), "source/target 名称集合不一致"
    return [source_names.index(name) for name in target_names]


BUMI3_ISAACLAB_TO_MUJOCO_DOF = _target_order_indices(
    BUMI3_ISAACLAB_DOF_NAMES, BUMI3_MUJOCO_DOF_NAMES
)
BUMI3_MUJOCO_TO_ISAACLAB_DOF = _target_order_indices(
    BUMI3_MUJOCO_DOF_NAMES, BUMI3_ISAACLAB_DOF_NAMES
)
BUMI3_ISAACLAB_TO_MUJOCO_BODY = _target_order_indices(
    BUMI3_ISAACLAB_BODY_NAMES, BUMI3_MUJOCO_BODY_NAMES
)
BUMI3_MUJOCO_TO_ISAACLAB_BODY = _target_order_indices(
    BUMI3_MUJOCO_BODY_NAMES, BUMI3_ISAACLAB_BODY_NAMES
)

assert BUMI3_ISAACLAB_TO_MUJOCO_DOF == [
    2,
    5,
    9,
    13,
    17,
    6,
    10,
    14,
    18,
    0,
    3,
    7,
    11,
    15,
    19,
    1,
    4,
    8,
    12,
    16,
    20,
]
assert BUMI3_MUJOCO_TO_ISAACLAB_DOF == [
    9,
    15,
    0,
    10,
    16,
    1,
    5,
    11,
    17,
    2,
    6,
    12,
    18,
    3,
    7,
    13,
    19,
    4,
    8,
    14,
    20,
]

BUMI3_LOWER_JOINT_INDICES_MUJOCO = list(range(9, 21))

BUMI3_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": BUMI3_ISAACLAB_BODY_NAMES,
    "isaaclab_dof_names": BUMI3_ISAACLAB_DOF_NAMES,
    "mujoco_dof_names": BUMI3_MUJOCO_DOF_NAMES,
    "isaaclab_to_mujoco_dof": BUMI3_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": BUMI3_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": BUMI3_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": BUMI3_MUJOCO_TO_ISAACLAB_BODY,
    "lower_joint_indices_mujoco": BUMI3_LOWER_JOINT_INDICES_MUJOCO,
}


BUMI3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=f"{ASSET_DIR}/robot_description/urdf/bumi3/bumi.urdf",
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            "l_leg_yaw_joint": 0.0,
            "l_leg_roll_joint": 0.0,
            "l_leg_pitch_joint": -0.1495,
            "l_knee_pitch_joint": 0.3215,
            "l_ankle_pitch_joint": -0.1720,
            "l_ankle_roll_joint": 0.0,
            "r_leg_yaw_joint": 0.0,
            "r_leg_roll_joint": 0.0,
            "r_leg_pitch_joint": -0.1495,
            "r_knee_pitch_joint": 0.3215,
            "r_ankle_pitch_joint": -0.1720,
            "r_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "l_arm_pitch_joint": 0.0,
            "l_arm_roll_joint": 0.3,
            "l_arm_yaw_joint": 0.0,
            "l_elbow_pitch_joint": 0.0,
            "r_arm_pitch_joint": 0.0,
            "r_arm_roll_joint": -0.3,
            "r_arm_yaw_joint": 0.0,
            "r_elbow_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_leg_yaw_joint",
                ".*_leg_roll_joint",
                ".*_leg_pitch_joint",
                ".*_knee_pitch_joint",
            ],
            effort_limit_sim={
                ".*_leg_yaw_joint": 12,
                ".*_leg_roll_joint": 50.0,
                ".*_leg_pitch_joint": 50.0,
                ".*_knee_pitch_joint": 50.0,
            },
            velocity_limit_sim={
                ".*_leg_yaw_joint": 12.0,
                ".*_leg_roll_joint": 12.0,
                ".*_leg_pitch_joint": 12.0,
                ".*_knee_pitch_joint": 12.0,
            },
            stiffness={
                ".*_leg_yaw_joint": 20,
                ".*_leg_roll_joint": 45,
                ".*_leg_pitch_joint": 45,
                ".*_knee_pitch_joint": 45,
            },
            damping={
                ".*_leg_yaw_joint": 1.0,
                ".*_leg_roll_joint": 3.0,
                ".*_leg_pitch_joint": 3.0,
                ".*_knee_pitch_joint": 2.0,
            },
            min_delay=0,
            max_delay=4,
        ),
        "waist": DelayedImplicitActuatorCfg(
            effort_limit_sim=27.0,
            velocity_limit_sim=9.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=53,
            damping=3.4,
            min_delay=0,
            max_delay=4,
        ),
        "feet": DelayedImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=9.0,
            velocity_limit_sim=12.0,
            stiffness={".*_ankle_pitch_joint": 8, ".*_ankle_roll_joint": 8},
            damping={".*_ankle_pitch_joint": 0.5, ".*_ankle_roll_joint": 0.5},
            armature={
                ".*_ankle_pitch_joint": 0.012574,
                ".*_ankle_roll_joint": 0.009608,
            },
            min_delay=0,
            max_delay=4,
        ),
        "arms": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_arm_pitch_joint",
                ".*_arm_roll_joint",
                ".*_arm_yaw_joint",
                ".*_elbow_pitch_joint",
            ],
            effort_limit_sim=4.0,
            velocity_limit_sim=12,
            stiffness=8,
            damping=0.4,
            min_delay=0,
            max_delay=4,
        ),
    },
)


# 与原 SONIC 相同，动作缩放由执行器仿真力矩上限和刚度实时推导，
# 避免复制硬编码值。
BUMI3_ACTION_SCALE = {}
for actuator in BUMI3_CFG.actuators.values():
    effort = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    names = actuator.joint_names_expr
    if not isinstance(effort, dict):
        effort = dict.fromkeys(names, effort)
    if not isinstance(stiffness, dict):
        stiffness = dict.fromkeys(names, stiffness)
    for name in names:
        if name in effort and name in stiffness and stiffness[name]:
            BUMI3_ACTION_SCALE[name] = 0.25 * effort[name] / stiffness[name]
