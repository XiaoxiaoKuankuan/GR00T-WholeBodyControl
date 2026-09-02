# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""验证 BUMI3 SONIC sim2sim 的资产、顺序、网络输入和闭环有限值。

脚本只解析并锁定 SONIC 仓库内 BUMI3 MJCF，不读取任何外部机器人仓库；随后验证
所有 mesh、21 DoF、22 robot bodies、22 个可视网格、14 个碰撞体、静态 reset
无自碰撞/陷地、双向排列、动作缩放、PD/armature、SONIC 1170 维输入和 21 维输出，
并检查红色参考影子只复制不参与物理的 22 个可视 geom、根高不跟随真实机器人。
默认还会使用静态参考与零动作策略执行 100 个 50 Hz 控制周期（共 400 个 MuJoCo
step），检查 observation、action、torque、qpos 和 qvel 无 NaN/Inf。

若同时提供 ``--policy`` 和 ``--motion``，smoke 会改用真实 ``*_g1.onnx`` 与真实
BUMI3 动作；不提供时的零策略结果只能证明接口和数值稳定，不能证明动作质量。
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import (
    Bumi3Contract,
    Bumi3SonicSim2Sim,
    DEFAULT_BUMI3_SIM2SIM_CONFIG,
    OnnxRobotPolicy,
    ZeroPolicy,
    load_reference_motion,
    make_static_reference_motion,
)


EXPECTED_LOCAL_MJCF_SHA256 = (
    "c4521504388c6eba296b8070fd80d73bb85c506b7346722031cefa3bcea11c04"
)

EXPECTED_COLLISION_BODIES = {
    "base_link",
    "waist_yaw_link",
    "l_arm_roll_link",
    "l_elbow_pitch_link",
    "r_arm_roll_link",
    "r_elbow_pitch_link",
    "l_leg_roll_link",
    "l_knee_pitch_link",
    "l_ankle_pitch_link",
    "l_ankle_roll_link",
    "r_leg_roll_link",
    "r_knee_pitch_link",
    "r_ankle_pitch_link",
    "r_ankle_roll_link",
}
EXPECTED_CAPSULES = {
    "base_link": ((-0.0013853, 0.0, 0.065525), (0.052, 0.06)),
    "l_leg_roll_link": ((0.0, 0.0, -0.02), (0.03, 0.04)),
    "r_leg_roll_link": ((0.0, 0.0, -0.02), (0.03, 0.04)),
    "l_knee_pitch_link": ((0.008475, 0.0, -0.0894694), (0.025, 0.065)),
    "r_knee_pitch_link": ((0.008475, 0.0, -0.0894694), (0.025, 0.065)),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_meshes(model_path: Path) -> int:
    root = ET.parse(model_path).getroot()
    compiler = root.find("compiler")
    mesh_dir = model_path.parent / (compiler.get("meshdir") if compiler is not None else ".")
    mesh_nodes = root.findall("./asset/mesh")
    missing = [node.get("file") for node in mesh_nodes if not (mesh_dir / node.get("file")).is_file()]
    if missing:
        raise FileNotFoundError(f"MJCF 引用 mesh 不存在: {missing}")
    if len(mesh_nodes) != 22:
        raise AssertionError(f"BUMI3 应引用 22 个 link mesh，实际为 {len(mesh_nodes)}")
    return len(mesh_nodes)


def _validate_model(contract: Bumi3Contract) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(contract.model_path))
    assert (model.nq, model.nv, model.nu, model.nbody - 1) == (28, 27, 21, 22)
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
    ]
    assert joint_names[0] == "root"
    assert joint_names[1:] == list(contract.mujoco_joint_names)
    assert len(joint_names) == len(set(joint_names))
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(1, model.nbody)
    ]
    assert len(body_names) == len(set(body_names)) == 22
    assert contract.anchor_body_name in body_names
    assert contract.reference_root_body_name in body_names

    actuator_joint_names = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuator_joint_names.append(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        )
    assert actuator_joint_names == list(contract.mujoco_joint_names)
    return model


def _validate_collision_contract(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """验证 XML 编译后的可视/碰撞分离、简化 capsule 数值和地面高度。"""

    robot_geom_ids = np.flatnonzero(model.geom_bodyid != 0)
    visual_geom_ids = robot_geom_ids[model.geom_group[robot_geom_ids] == 1]
    collision_geom_ids = robot_geom_ids[model.geom_group[robot_geom_ids] == 3]
    assert robot_geom_ids.size == 36
    assert visual_geom_ids.size == 22
    assert collision_geom_ids.size == 14
    assert np.all(model.geom_type[visual_geom_ids] == mujoco.mjtGeom.mjGEOM_MESH)
    assert np.all(model.geom_contype[visual_geom_ids] == 0)
    assert np.all(model.geom_conaffinity[visual_geom_ids] == 0)
    assert np.all(model.geom_contype[collision_geom_ids] == 1)
    assert np.all(model.geom_conaffinity[collision_geom_ids] == 0)
    assert np.all(model.geom_condim[collision_geom_ids] == 3)

    visual_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        for geom_id in visual_geom_ids
    }
    body_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(1, model.nbody)
    }
    assert visual_names == {f"{body_name}_visual" for body_name in body_names}
    collision_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        for geom_id in collision_geom_ids
    }
    assert collision_names == {
        f"{body_name}_collision" for body_name in EXPECTED_COLLISION_BODIES
    }

    for body_name, (expected_pos, expected_size) in EXPECTED_CAPSULES.items():
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{body_name}_collision"
        )
        assert geom_id >= 0
        assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE
        np.testing.assert_allclose(model.geom_pos[geom_id], expected_pos)
        np.testing.assert_allclose(model.geom_size[geom_id, :2], expected_size)
    mesh_collision_count = np.count_nonzero(
        model.geom_type[collision_geom_ids] == mujoco.mjtGeom.mjGEOM_MESH
    )
    assert mesh_collision_count == 9

    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    assert ground_id >= 0
    assert np.isclose(model.geom_pos[ground_id, 2], -0.02)
    assert model.geom_contype[ground_id] == 0
    assert model.geom_conaffinity[ground_id] == 1
    assert model.geom_condim[ground_id] == 3
    return visual_geom_ids, collision_geom_ids


def _validate_static_reset_contacts(contract: Bumi3Contract) -> tuple[int, float]:
    """用部署回退初始姿态检查自碰撞和地面穿透，返回接触数与最小距离。"""

    runner = Bumi3SonicSim2Sim(
        contract,
        make_static_reference_motion(contract),
        ZeroPolicy(contract),
    )
    ground_id = mujoco.mj_name2id(
        runner.model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
    )
    self_contacts = []
    ground_penetrations = []
    distances = []
    for contact_index in range(runner.data.ncon):
        contact = runner.data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        distance = float(contact.dist)
        distances.append(distance)
        if runner.model.geom_bodyid[geom1] != 0 and runner.model.geom_bodyid[geom2] != 0:
            self_contacts.append((geom1, geom2, distance))
        if ground_id in (geom1, geom2) and distance < -1e-6:
            ground_penetrations.append((geom1, geom2, distance))
    assert not self_contacts, f"静态 reset 存在自碰撞: {self_contacts}"
    assert not ground_penetrations, f"静态 reset 存在地面穿透: {ground_penetrations}"
    initial_contact_count = runner.data.ncon

    # 下移根节点后必须只产生地面接触，证明碰撞 bitmask 没有误关触地能力。
    runner.data.qpos[runner.root_qpos_address + 2] -= 0.04
    mujoco.mj_forward(runner.model, runner.data)
    assert runner.data.ncon > 0, "根节点下移后未产生地面接触"
    for contact_index in range(runner.data.ncon):
        contact = runner.data.contact[contact_index]
        assert ground_id in (int(contact.geom1), int(contact.geom2)), (
            "碰撞 bitmask 未完全隔离机器人自碰撞"
        )
    return initial_contact_count, min(distances, default=0.0)


def _validate_contract_values(contract: Bumi3Contract) -> None:
    np.testing.assert_allclose(contract.initial_root_position, [0.0, 0.0, 0.4744])
    assert contract.anchor_body_name == "base_link", (
        "BUMI3 sim2sim Robot Encoder 锚点必须与训练统一为 base_link"
    )
    assert contract.policy_to_mujoco.tolist() == [
        2, 5, 9, 13, 17, 6, 10, 14, 18, 0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20
    ]
    assert contract.mujoco_to_policy.tolist() == [
        9, 15, 0, 10, 16, 1, 5, 11, 17, 2, 6, 12, 18, 3, 7, 13, 19, 4, 8, 14, 20
    ]
    source = np.arange(21)
    np.testing.assert_array_equal(
        source[contract.policy_to_mujoco][contract.mujoco_to_policy], source
    )

    scale_by_name = dict(zip(contract.mujoco_joint_names, contract.action_scale_mujoco, strict=True))
    for side in ("l", "r"):
        assert np.isclose(scale_by_name[f"{side}_leg_yaw_joint"], 0.15)
        assert np.isclose(scale_by_name[f"{side}_leg_roll_joint"], 0.25 * 50.0 / 45.0)
        assert np.isclose(scale_by_name[f"{side}_leg_pitch_joint"], 0.25 * 50.0 / 45.0)
        assert np.isclose(scale_by_name[f"{side}_knee_pitch_joint"], 0.25 * 50.0 / 45.0)
        assert np.isclose(scale_by_name[f"{side}_ankle_pitch_joint"], 0.28125)
        assert np.isclose(scale_by_name[f"{side}_ankle_roll_joint"], 0.28125)
        for suffix in ("arm_pitch", "arm_roll", "arm_yaw", "elbow_pitch"):
            assert np.isclose(scale_by_name[f"{side}_{suffix}_joint"], 0.125)
    assert np.isclose(scale_by_name["waist_yaw_joint"], 0.25 * 27.0 / 53.0)
    assert np.count_nonzero(contract.armature_mujoco) == 4
    assert set(contract.velocity_mujoco.tolist()) == {9.0, 12.0}


def validate(args: argparse.Namespace) -> None:
    contract = Bumi3Contract.from_yaml(args.config)
    assert _sha256(contract.model_path) == EXPECTED_LOCAL_MJCF_SHA256, (
        "SONIC 仓库内 BUMI3 MJCF 指纹变化，必须同步审计 sim2sim 契约"
    )
    mesh_count = _validate_meshes(contract.model_path)
    model = _validate_model(contract)
    visual_geom_ids, collision_geom_ids = _validate_collision_contract(model)
    initial_contact_count, initial_minimum_contact_distance = _validate_static_reset_contacts(
        contract
    )
    _validate_contract_values(contract)

    if (args.policy is None) != (args.motion is None):
        raise ValueError("--policy 与 --motion 必须同时提供或同时省略")
    if args.policy is None:
        motion = make_static_reference_motion(contract)
        policy = ZeroPolicy(contract)
        smoke_kind = "zero_policy_static_reference"
    else:
        motion = load_reference_motion(
            args.motion,
            contract,
            motion_key=args.motion_key,
            joint_order=args.joint_order,
            quaternion_order=args.quaternion_order,
        )
        policy = OnnxRobotPolicy(args.policy, contract, provider=args.provider)
        smoke_kind = "real_onnx_real_reference"

    runner = Bumi3SonicSim2Sim(contract, motion, policy, loop_motion=True)
    actual_armature = runner.model.dof_armature[runner.dof_addresses]
    np.testing.assert_allclose(actual_armature, contract.armature_mujoco, atol=1e-12)
    for field in (
        "geom_type",
        "geom_bodyid",
        "geom_contype",
        "geom_conaffinity",
        "geom_condim",
        "geom_dataid",
        "geom_group",
        "geom_priority",
    ):
        np.testing.assert_array_equal(getattr(runner.model, field), getattr(model, field))
    for field in (
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_margin",
        "geom_gap",
        "geom_rgba",
    ):
        np.testing.assert_allclose(getattr(runner.model, field), getattr(model, field))
    robot_geom_ids = np.flatnonzero(runner.model.geom_bodyid != 0)
    observation = runner.build_observation()
    assert observation.shape == (1170,) and np.isfinite(observation).all()
    assert runner._build_robot_tokenizer().shape == (480,)
    assert runner._build_proprioception().shape == (690,)
    assert runner.reference_anchor_quat_wxyz.shape == (motion.num_frames, 4)
    assert np.isfinite(runner.reference_anchor_quat_wxyz).all()
    reference_diagnostics = runner.reference_pose_diagnostics(0)
    assert all(np.isfinite(value) for value in reference_diagnostics.values())
    reference_markers = runner.reference_marker_specs(0, alpha=0.32)
    assert len(reference_markers) == 22
    assert runner.reference_visual_model is not runner.model
    base_geom_id = mujoco.mj_name2id(
        runner.model, mujoco.mjtObj.mjOBJ_GEOM, "base_link_visual"
    )
    assert base_geom_id >= 0
    assert runner.model.geom_type[base_geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    assert (
        runner.reference_visual_model.geom_type[base_geom_id]
        == mujoco.mjtGeom.mjGEOM_MESH
    )
    assert all("dataid" in marker and "matid" in marker for marker in reference_markers)
    assert all(
        np.isfinite(marker["pos"]).all()
        and np.isfinite(marker["mat"]).all()
        and np.isfinite(marker["size"]).all()
        for marker in reference_markers
    )
    reference_scene = mujoco.MjvScene(runner.model, maxgeom=1000)
    assert runner._write_reference_markers_to_scene(
        reference_scene, 0, alpha=0.32
    ) == len(reference_markers)
    assert all(
        reference_scene.geoms[index].category == mujoco.mjtCatBit.mjCAT_DECOR
        for index in range(reference_scene.ngeom)
    )
    stats = None
    if not args.skip_smoke:
        stats = runner.run(args.steps, headless=True, real_time=False)
        assert np.isfinite(np.asarray(list(stats.values()), dtype=np.float64)).all()
        assert np.isclose(stats["simulation_time"], args.steps * contract.control_dt)

    print("BUMI3_SIM2SIM_VALIDATION=PASS")
    print(f"LOCAL_MJCF_SHA256={_sha256(contract.model_path)}")
    print(f"MJCF_MODEL=nq:{model.nq},nv:{model.nv},nu:{model.nu},robot_bodies:{model.nbody - 1}")
    print(f"MESH_COUNT={mesh_count}")
    print(
        "COLLISION_CONTRACT="
        f"visual_meshes:{visual_geom_ids.size},collision_geoms:{collision_geom_ids.size},"
        "capsules:5,collision_meshes:9,self_collision:false,ground_z:-0.02"
    )
    print(
        "INITIAL_CONTACT_GATE="
        f"contacts:{initial_contact_count},self_contacts:0,ground_penetrations:0,"
        f"minimum_distance:{initial_minimum_contact_distance:.9f}"
    )
    print(
        "RESOLVED="
        f"sim_dt:{contract.sim_dt},decimation:{contract.decimation},"
        f"control_hz:{1.0 / contract.control_dt},target_fps:{contract.target_fps},"
        f"align_heading:{contract.align_reference_heading},"
        f"tokenizer:{contract.robot_tokenizer_dim},proprio:{contract.actor_proprioception_dim},"
        f"policy_input:{contract.combined_policy_input_dim},action:{contract.action_dim}"
    )
    print(f"SMOKE_KIND={smoke_kind}")
    print(
        "REFERENCE_RESET="
        f"root_position:{'motion' if motion.root_position_world is not None else 'fallback'},"
        f"anchor_body:{contract.anchor_body_name},anchor_source:mjcf_fk,"
        "collision_source:bumi3_xml"
    )
    print(
        "POLICY_VISUAL="
        f"visual_geoms:{visual_geom_ids.size},collision_geoms:{collision_geom_ids.size},"
        "physics:true,state_source:dynamics_qpos,"
        "collision_source:bumi3_xml,render_source:dynamics_bumi3_xml"
    )
    print(
        "REFERENCE_SHADOW="
        f"geoms:{len(reference_markers)},alpha:0.32,physics:false,"
        "render_source:independent_ref_model_mjv_updateScene,"
        f"base_tilt_deg:{reference_diagnostics['base_tilt_degrees']:.6f},"
        f"anchor_tilt_deg:{reference_diagnostics['anchor_tilt_degrees']:.6f}"
    )
    if stats is not None:
        print(f"SMOKE_STEPS={args.steps},SIMULATION_TIME={stats['simulation_time']:.6f}")
        print(f"SMOKE_ROOT_HEIGHT={stats['root_height']:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_BUMI3_SIM2SIM_CONFIG)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--motion-key")
    parser.add_argument(
        "--joint-order", choices=("auto", "policy", "isaaclab", "mujoco"), default="auto"
    )
    parser.add_argument(
        "--quaternion-order", choices=("auto", "wxyz", "xyzw"), default="auto"
    )
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--skip-smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
