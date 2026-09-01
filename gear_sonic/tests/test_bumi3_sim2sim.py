# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BUMI3 SONIC sim2sim 的回归测试。

测试覆盖名称生成的 IsaacLab/MuJoCo 双向排列、训练 PKL 与部署 CSV 的顺序/浮动根
状态约定、参考状态 reset、``waist_yaw_link`` FK 锚点语义、1170 维联合 policy 输入、
不会跟随真实机器人高度的半透明参考影子、白色 policy 完整使用原始 XML
geom 同时执行碰撞与渲染的契约、直立/横躺倾角诊断，以及零动作下的无界面 MuJoCo
闭环。测试不依赖 Isaac Lab、GPU 或训练数据，也不会生成持久数据；临时动作
仅用于验证加载器契约。
"""

from pathlib import Path

import joblib
import mujoco
import numpy as np

from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import (
    Bumi3Contract,
    Bumi3SonicSim2Sim,
    DEFAULT_BUMI3_SIM2SIM_CONFIG,
    ZeroPolicy,
    load_reference_motion,
    make_static_reference_motion,
    quaternion_to_rotation_6d,
)


def _contract() -> Bumi3Contract:
    return Bumi3Contract.from_yaml(DEFAULT_BUMI3_SIM2SIM_CONFIG)


def test_contract_mapping_dimensions_and_action_scale() -> None:
    contract = _contract()
    assert contract.policy_to_mujoco.tolist() == [
        2, 5, 9, 13, 17, 6, 10, 14, 18, 0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20
    ]
    assert contract.mujoco_to_policy.tolist() == [
        9, 15, 0, 10, 16, 1, 5, 11, 17, 2, 6, 12, 18, 3, 7, 13, 19, 4, 8, 14, 20
    ]
    policy_values = np.arange(21)
    np.testing.assert_array_equal(
        policy_values[contract.policy_to_mujoco][contract.mujoco_to_policy], policy_values
    )
    assert contract.actor_proprioception_dim == 690
    assert contract.robot_tokenizer_dim == 480
    assert contract.combined_policy_input_dim == 1170
    assert contract.action_dim == 21
    np.testing.assert_allclose(
        contract.action_scale_mujoco,
        0.25 * contract.effort_mujoco / contract.stiffness_mujoco,
    )


def test_load_training_pkl_converts_mujoco_and_xyzw(tmp_path: Path) -> None:
    contract = _contract()
    frames = 4
    dof_mujoco = np.repeat(np.arange(21, dtype=np.float32)[None], frames, axis=0)
    root_xyzw = np.repeat(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), frames, axis=0)
    root_position = np.repeat(np.asarray([[0.1, -0.2, 0.53]], dtype=np.float32), frames, axis=0)
    path = tmp_path / "motion.pkl"
    joblib.dump(
        {
            "sample": {
                "dof": dof_mujoco,
                "root_rot": root_xyzw,
                "root_trans_offset": root_position,
                "fps": 50,
            }
        },
        path,
    )

    motion = load_reference_motion(path, contract)

    np.testing.assert_allclose(
        motion.joint_pos_policy[0], dof_mujoco[0, contract.mujoco_to_policy]
    )
    np.testing.assert_allclose(motion.root_quat_wxyz[0], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(motion.root_position_world, root_position)
    assert motion.joint_vel_policy.shape == (frames, 21)
    assert np.isfinite(motion.joint_vel_policy).all()


def test_load_g1_style_csv_uses_policy_order_and_wxyz(tmp_path: Path) -> None:
    contract = _contract()
    frames = 3
    joint_pos = np.repeat(contract.default_policy[None], frames, axis=0)
    joint_vel = np.zeros_like(joint_pos)
    root_quat = np.repeat(np.asarray([[1.0, 0.0, 0.0, 0.0]]), frames, axis=0)
    np.savetxt(
        tmp_path / "joint_pos.csv",
        joint_pos,
        delimiter=",",
        header=",".join(f"joint_{index}" for index in range(21)),
        comments="",
    )
    np.savetxt(
        tmp_path / "joint_vel.csv",
        joint_vel,
        delimiter=",",
        header=",".join(f"joint_vel_{index}" for index in range(21)),
        comments="",
    )
    np.savetxt(
        tmp_path / "body_quat.csv",
        root_quat,
        delimiter=",",
        header="body_0_w,body_0_x,body_0_y,body_0_z",
        comments="",
    )

    motion = load_reference_motion(tmp_path, contract)

    np.testing.assert_allclose(motion.joint_pos_policy, joint_pos)
    np.testing.assert_allclose(motion.joint_vel_policy, joint_vel)
    np.testing.assert_allclose(motion.root_quat_wxyz, root_quat)
    assert motion.root_position_world is None


def test_headless_zero_policy_builds_finite_observation_and_steps() -> None:
    contract = _contract()
    runner = Bumi3SonicSim2Sim(
        contract,
        make_static_reference_motion(contract),
        ZeroPolicy(contract),
        loop_motion=True,
    )

    observation = runner.build_observation()
    assert observation.shape == (1170,)
    assert np.isfinite(observation).all()
    assert runner._build_robot_tokenizer().shape == (480,)
    assert runner._build_proprioception().shape == (690,)

    stats = runner.run(10, headless=True, real_time=False)
    assert np.isclose(stats["simulation_time"], 10 * 0.02)
    assert np.isfinite(np.asarray(list(stats.values()))).all()
    assert np.isfinite(runner.data.qpos).all()
    assert np.isfinite(runner.data.qvel).all()


def test_reset_fills_history_like_isaaclab_first_append() -> None:
    contract = _contract()
    runner = Bumi3SonicSim2Sim(
        contract,
        make_static_reference_motion(contract),
        ZeroPolicy(contract),
    )

    for history in runner.histories.values():
        stacked = np.stack(history, axis=0)
        np.testing.assert_allclose(stacked, np.repeat(stacked[-1:], 10, axis=0))


def test_runtime_visual_and_collision_geoms_match_original_xml() -> None:
    contract = _contract()
    runner = Bumi3SonicSim2Sim(
        contract,
        make_static_reference_motion(contract),
        ZeroPolicy(contract),
    )
    xml_model = mujoco.MjModel.from_xml_path(str(contract.model_path))

    # sim2sim 是 MuJoCo→MuJoCo，碰撞与可视 geom 都必须保留 XML 编译结果；
    # 不允许用 Isaac Lab URDF importer 的 capsule/禁用碰撞规则覆盖。
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
        np.testing.assert_array_equal(getattr(runner.model, field), getattr(xml_model, field))
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
        np.testing.assert_allclose(getattr(runner.model, field), getattr(xml_model, field))

    robot_geom_ids = np.flatnonzero(runner.model.geom_bodyid != 0)
    assert robot_geom_ids.size == 22
    assert np.all(runner.model.geom_type[robot_geom_ids] == mujoco.mjtGeom.mjGEOM_MESH)


def test_reference_start_heading_is_aligned_to_robot() -> None:
    contract = _contract()
    motion = make_static_reference_motion(contract)
    yaw = np.pi / 2.0
    motion.root_quat_wxyz[:] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    runner = Bumi3SonicSim2Sim(contract, motion, ZeroPolicy(contract))

    tokenizer = runner._build_robot_tokenizer()
    orientation = tokenizer[-60:].reshape(10, 6)
    identity_6d = quaternion_to_rotation_6d(np.asarray([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(orientation, np.repeat(identity_6d[None], 10, axis=0), atol=1e-6)


def test_reset_uses_reference_root_and_joint_state() -> None:
    contract = _contract()
    motion = make_static_reference_motion(contract)
    frame = 1
    motion.root_position_world[frame] = [0.12, -0.08, 0.54]
    motion.joint_pos_policy[frame] += np.linspace(-0.02, 0.02, 21)
    motion.joint_vel_policy[frame] = np.linspace(-0.1, 0.1, 21)
    runner = Bumi3SonicSim2Sim(
        contract,
        motion,
        ZeroPolicy(contract),
        start_frame=frame,
    )

    root = runner.root_qpos_address
    np.testing.assert_allclose(runner.data.qpos[root : root + 3], [0.12, -0.08, 0.54])
    np.testing.assert_allclose(
        runner.data.qpos[runner.qpos_addresses],
        motion.joint_pos_policy[frame][contract.policy_to_mujoco],
    )
    np.testing.assert_allclose(
        runner.data.qvel[runner.dof_addresses],
        motion.joint_vel_policy[frame][contract.policy_to_mujoco],
    )


def test_robot_encoder_anchor_uses_waist_fk_not_root_quaternion() -> None:
    contract = _contract()
    motion = make_static_reference_motion(contract)
    waist_policy_index = contract.policy_joint_names.index("waist_yaw_joint")
    waist_yaw = 0.4
    motion.joint_pos_policy[:, waist_policy_index] = waist_yaw
    runner = Bumi3SonicSim2Sim(contract, motion, ZeroPolicy(contract))

    root_inverse = np.asarray([1.0, 0.0, 0.0, 0.0])
    waist_relative = runner.reference_anchor_quat_wxyz[0]
    assert not np.allclose(waist_relative, root_inverse, atol=1e-3)
    np.testing.assert_allclose(
        waist_relative,
        [np.cos(waist_yaw / 2.0), 0.0, 0.0, np.sin(waist_yaw / 2.0)],
        atol=1e-6,
    )
    tokenizer = runner._build_robot_tokenizer()
    orientation = tokenizer[-60:].reshape(10, 6)
    identity_6d = quaternion_to_rotation_6d(root_inverse)
    np.testing.assert_allclose(
        orientation,
        np.repeat(identity_6d[None], 10, axis=0),
        atol=1e-6,
    )


def test_reference_pose_diagnostics_distinguishes_upright_and_sideways() -> None:
    contract = _contract()
    upright = Bumi3SonicSim2Sim(
        contract,
        make_static_reference_motion(contract),
        ZeroPolicy(contract),
    )
    upright_diagnostics = upright.reference_pose_diagnostics(0)
    assert upright_diagnostics["base_tilt_degrees"] < 1e-5
    assert upright_diagnostics["anchor_tilt_degrees"] < 1e-5

    sideways_motion = make_static_reference_motion(contract)
    sideways_motion.root_quat_wxyz[:] = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]
    sideways = Bumi3SonicSim2Sim(
        contract,
        sideways_motion,
        ZeroPolicy(contract),
    )
    sideways_diagnostics = sideways.reference_pose_diagnostics(0)
    assert np.isclose(sideways_diagnostics["base_tilt_degrees"], 90.0, atol=1e-5)
    assert np.isclose(sideways_diagnostics["anchor_tilt_degrees"], 90.0, atol=1e-5)


def test_reference_shadow_copies_resolved_mjv_scene_as_decorative_geoms() -> None:
    contract = _contract()
    motion = make_static_reference_motion(contract)
    motion.root_position_world[:, 2] = 0.56
    runner = Bumi3SonicSim2Sim(contract, motion, ZeroPolicy(contract))

    # 即使真实机器人已经降低，参考影子的根高仍必须来自训练参考，不能跟随真实 robot。
    runner.data.qpos[runner.root_qpos_address + 2] = 0.08
    mujoco.mj_forward(runner.model, runner.data)
    reference_qpos = runner._aligned_reference_qpos(0)
    assert np.isclose(reference_qpos[runner.root_qpos_address + 2], 0.56)

    markers = runner.reference_marker_specs(0, alpha=0.32)

    # 回归参考脚本的关键语义：先让 MuJoCo 解析完整 MjvScene，再复制最终 MjvGeom。
    # 不能退回从 model.geom_size + data.geom_xpos/geom_xmat 自行重建 marker；mesh
    # 的局部变换/缩放以及 capsule 的 MjvGeom size 约定都可能因此丢失。
    expected_scene = mujoco.MjvScene(runner.reference_visual_model, maxgeom=1000)
    expected_camera = mujoco.MjvCamera()
    expected_option = mujoco.MjvOption()
    expected_perturb = mujoco.MjvPerturb()
    mujoco.mjv_updateScene(
        runner.reference_visual_model,
        runner.reference_visual_data,
        expected_option,
        expected_perturb,
        expected_camera,
        mujoco.mjtCatBit.mjCAT_ALL.value,
        expected_scene,
    )
    expected_geoms = []
    for index in range(expected_scene.ngeom):
        geom = expected_scene.geoms[index]
        if int(geom.objtype) != int(mujoco.mjtObj.mjOBJ_GEOM):
            continue
        geom_id = int(geom.objid)
        if not 0 <= geom_id < runner.reference_visual_model.ngeom:
            continue
        if int(runner.reference_visual_model.geom_bodyid[geom_id]) == 0:
            continue
        expected_geoms.append(geom)

    assert len(markers) == len(expected_geoms)
    assert markers
    # policy 动力学模型和参考模型都从同一 XML 加载，base 均必须
    # 保留原始 mesh；两者区别只是 qpos 状态和是否参与物理。
    dynamics_base_geom_id = int(runner._body_geom_ids("base_link")[0])
    # BUMI3 MJCF 的 geom 没有 name；两个模型从同一 XML 加载并已在
    # runner 中校验相同拓扑，因此 geom id 对应一致。
    reference_base_geom_id = dynamics_base_geom_id
    assert runner.model.geom_type[dynamics_base_geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    assert (
        runner.reference_visual_model.geom_type[reference_base_geom_id]
        == mujoco.mjtGeom.mjGEOM_MESH
    )
    assert next(
        marker for marker in markers if marker["objid"] == reference_base_geom_id
    )["type"] == mujoco.mjtGeom.mjGEOM_MESH
    for marker, expected in zip(markers, expected_geoms, strict=True):
        assert marker["objtype"] == mujoco.mjtObj.mjOBJ_GEOM
        assert marker["objid"] == expected.objid
        assert marker["type"] == expected.type
        assert marker["dataid"] == expected.dataid
        assert marker["matid"] == expected.matid
        assert marker["label"] == expected.label
        assert np.isclose(marker["rgba"][3], 0.32)
        np.testing.assert_allclose(marker["pos"], expected.pos)
        np.testing.assert_allclose(marker["mat"], np.asarray(expected.mat).reshape(9))
        np.testing.assert_allclose(marker["size"], expected.size)

    scene = mujoco.MjvScene(runner.model, maxgeom=1000)
    written = runner._write_reference_markers_to_scene(scene, 0, alpha=0.32)
    assert written == len(markers) == scene.ngeom
    for index, marker in enumerate(markers):
        written_geom = scene.geoms[index]
        assert written_geom.category == mujoco.mjtCatBit.mjCAT_DECOR
        assert written_geom.dataid == marker["dataid"]
        assert written_geom.matid == marker["matid"]
        assert written_geom.label == marker["label"]
        np.testing.assert_allclose(written_geom.pos, marker["pos"])
        np.testing.assert_allclose(
            np.asarray(written_geom.mat).reshape(9), marker["mat"]
        )
        np.testing.assert_allclose(written_geom.size, marker["size"])
