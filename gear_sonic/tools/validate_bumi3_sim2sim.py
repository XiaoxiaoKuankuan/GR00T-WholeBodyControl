# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""验证 BUMI3 SONIC sim2sim 的资产、顺序、网络输入和闭环有限值。

脚本会解析本仓库与 ``legged_lab`` 当前 BUMI3 MJCF，忽略项目布局必需的 meshdir
差异后比较完整 XML 语义；随后验证所有 mesh、21 DoF、22 robot bodies、双向排列、
动作缩放、PD/armature、SONIC 1170 维输入和 21 维输出，并检查红色参考影子确实由
独立 FK 状态生成、全部为不参与物理的 decorative geom、根高不跟随真实机器人。
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


DEFAULT_REFERENCE_MJCF = Path(
    "/home/weili/legged_lab/source/NoetixRobot/NoetixRobot/"
    "assets/robots/bumi3/mjcf/bumi3.xml"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xml_signature(element: ET.Element) -> tuple:
    """构造忽略格式空白且归一化 compiler.meshdir 的 XML 语义签名。"""

    attributes = dict(element.attrib)
    if element.tag == "compiler" and "meshdir" in attributes:
        attributes["meshdir"] = "<BUMI3_MESH_DIR>"
    text = (element.text or "").strip()
    return (
        element.tag,
        tuple(sorted(attributes.items())),
        text,
        tuple(_xml_signature(child) for child in element),
    )


def _validate_mjcf_source(local_path: Path, reference_path: Path) -> None:
    if not reference_path.is_file():
        raise FileNotFoundError(f"legged_lab BUMI3 MJCF 不存在: {reference_path}")
    local_root = ET.parse(local_path).getroot()
    reference_root = ET.parse(reference_path).getroot()
    if _xml_signature(local_root) != _xml_signature(reference_root):
        raise AssertionError("本地 BUMI3 MJCF 除 meshdir 外已偏离当前 legged_lab 参考")


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


def _validate_contract_values(contract: Bumi3Contract) -> None:
    np.testing.assert_allclose(contract.initial_root_position, [0.0, 0.0, 0.4744])
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
    _validate_mjcf_source(contract.model_path, args.reference_mjcf)
    mesh_count = _validate_meshes(contract.model_path)
    model = _validate_model(contract)
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
    assert robot_geom_ids.size == 22
    assert np.all(runner.model.geom_type[robot_geom_ids] == mujoco.mjtGeom.mjGEOM_MESH)
    observation = runner.build_observation()
    assert observation.shape == (1170,) and np.isfinite(observation).all()
    assert runner._build_robot_tokenizer().shape == (480,)
    assert runner._build_proprioception().shape == (690,)
    assert runner.reference_anchor_quat_wxyz.shape == (motion.num_frames, 4)
    assert np.isfinite(runner.reference_anchor_quat_wxyz).all()
    reference_diagnostics = runner.reference_pose_diagnostics(0)
    assert all(np.isfinite(value) for value in reference_diagnostics.values())
    reference_markers = runner.reference_marker_specs(0, alpha=0.32)
    assert len(reference_markers) == int(
        np.count_nonzero(runner.reference_visual_model.geom_bodyid != 0)
    )
    assert runner.reference_visual_model is not runner.model
    base_geom_id = int(runner._body_geom_ids("base_link")[0])
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
    print(f"REFERENCE_MJCF_SHA256={_sha256(args.reference_mjcf)}")
    print(f"MJCF_MODEL=nq:{model.nq},nv:{model.nv},nu:{model.nu},robot_bodies:{model.nbody - 1}")
    print(f"MESH_COUNT={mesh_count}")
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
        f"geoms:{robot_geom_ids.size},physics:true,state_source:dynamics_qpos,"
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
    parser.add_argument("--reference-mjcf", type=Path, default=DEFAULT_REFERENCE_MJCF)
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
