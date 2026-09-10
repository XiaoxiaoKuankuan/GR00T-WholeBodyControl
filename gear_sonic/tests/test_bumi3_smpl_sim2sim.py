"""验证 BUMI SMPL sim2sim 的训练观测一致性、输入选择和播放边界。

数值测试读取仓库当前训练函数的语法树，直接执行真实的 SMPL 根朝向和关键点观测函数，
通过最小命令对象替代 Isaac Lab 场景，再与 NumPy/SciPy 部署实现比较。这样不需要
启动 Isaac Sim，也不会以部署函数自身生成期望值掩盖转轴、根相对坐标或字段顺序错误。
其余测试使用临时人体/机器人数据和真实 MuJoCo，检查配对错误、首帧保持、连续十帧、
末尾截断/循环、T/P 切换和纯人体入口；这些测试验证接口契约，不代表训练策略质量。
测试不连接服务器、不读取正式 checkpoint，不修改正式数据或机器人参数。
"""

import ast
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils.mujoco_sim.bumi3_motion_dataset import (
    load_motion_dataset,
    load_smpl_reference_motion,
)
from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import (
    Bumi3Contract,
    Bumi3SonicSim2Sim,
    DEFAULT_BUMI3_SIM2SIM_CONFIG,
    ZeroPolicy,
)
from gear_sonic.utils.mujoco_sim.bumi3_smpl_reference import (
    build_smpl_tokenizer,
    load_smpl_reference,
)


@pytest.fixture
def contract():
    """沿用当前 BUMI 资产与 PD，不为 SMPL 测试更换动力学参数。"""
    return Bumi3Contract.from_yaml(DEFAULT_BUMI3_SIM2SIM_CONFIG)


def _write_pair(tmp_path, contract, *, name="clip", count=16):
    """构造随帧变化且含非零平移的关键点，便于发现误减根坐标与冻结窗口问题。"""
    rng = np.random.default_rng(437)
    smpl = {
        "pose_aa": rng.normal(0, 0.2, (count, 72)),
        "smpl_joints": rng.normal(0, 0.1, (count, 24, 3)) + [0.3, -0.2, 0.7],
        "transl": np.repeat([[4, 1.2, 7]], count, axis=0),
        "fps": 50.0,
    }
    robot = {
        "dof": np.repeat(contract.default_mujoco[None], count, axis=0),
        "root_rot": np.repeat([[0.0, 0.0, 0.0, 1.0]], count, axis=0),
        "root_trans_offset": np.repeat([[0.0, 0.0, 0.4744]], count, axis=0),
        "fps": 50,
    }
    sp = tmp_path / f"{name}_smpl.pkl"
    rp = tmp_path / f"{name}_robot.pkl"
    joblib.dump(smpl, sp)
    joblib.dump({name: robot}, rp)
    return sp, rp, smpl


def _training_namespace(torch):
    """执行当前训练源码中所需函数体；保留真实旋转实现，仅替换场景访问对象。"""
    from gear_sonic.isaac_utils import rotations
    from gear_sonic.trl.utils import torch_transform

    namespace = {
        "torch": torch, "np": np, "rotations": rotations, "torch_transform": torch_transform,
        "ManagerBasedEnv": object, "commands": SimpleNamespace(TrackingCommand=object),
        "quat_inv": lambda q: rotations.quat_conjugate(q, w_last=False),
        "quat_mul": lambda q, r: rotations.quat_mul(q, r, w_last=False),
        "quat_apply": lambda q, v: rotations.quat_apply(q, v, w_last=False),
        "matrix_from_quat": rotations.quaternion_to_matrix,
    }
    root = Path(__file__).resolve().parents[1] / "envs/manager_env/mdp"
    wanted = {
        "commands.py": {
            "smpl_root_ytoz_up", "smpl_root_quat_w_multi_future",
            "smpl_root_quat_w_dif_l_multi_future",
        },
        "observations.py": {"smpl_joints_multi_future_local", "smpl_root_ori_b_mf"},
    }
    for filename, names in wanted.items():
        tree = ast.parse((root / filename).read_text())
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in names:
                node.decorator_list = []
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(root / filename), "exec"), namespace)
                found.add(node.name)
        assert found == names
    return namespace


def test_smpl_matches_actual_training_observations(tmp_path, contract):
    """用训练函数逐项验证基准旋转、局部关键点、完整根朝向及导出字段布局。"""
    torch = pytest.importorskip("torch")
    sp, _, content = _write_pair(tmp_path, contract)
    before = sp.read_bytes()
    reference = load_smpl_reference(sp)
    ns = _training_namespace(torch)
    indices = np.arange(3, 13)
    pose = torch.tensor(content["pose_aa"][indices][None], dtype=torch.float64)
    anchor_xyzw = Rotation.from_euler("xyz", [0.3, -0.2, 0.7]).as_quat()
    anchor_wxyz = anchor_xyzw[[3, 0, 1, 2]]
    command = SimpleNamespace(
        num_envs=1, smpl_num_future_frames=10,
        smpl_future_motion_ids=None, smpl_future_time_steps=None,
        motion_lib=SimpleNamespace(get_smpl_pose=lambda *_: pose, smpl_y_up=True),
        robot_anchor_quat_w=torch.tensor(anchor_wxyz[None]),
        smpl_joints_multi_future=torch.tensor(content["smpl_joints"][indices][None]),
    )
    command.smpl_root_ytoz_up = lambda q: ns["smpl_root_ytoz_up"](command, q)
    command.smpl_root_quat_w_multi_future = ns["smpl_root_quat_w_multi_future"](command)
    command.smpl_root_quat_w_dif_l_multi_future = ns["smpl_root_quat_w_dif_l_multi_future"](command)
    env = SimpleNamespace(num_envs=1, command_manager=SimpleNamespace(get_term=lambda _: command))
    local = ns["smpl_joints_multi_future_local"](env, "motion", non_flatten=True)
    ori = ns["smpl_root_ori_b_mf"](env, "motion", non_flatten=True)
    expected = torch.cat((local.flatten(), ori.flatten())).numpy()
    actual = build_smpl_tokenizer(reference, indices, anchor_wxyz, np.array([1, 0, 0, 0]))
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    assert sp.read_bytes() == before
    # transl 和机器人关节参考都不是 SMPL tokenizer 的输入，改变前者不能影响人体编码。
    content["transl"] *= 100
    joblib.dump(content, sp)
    np.testing.assert_array_equal(load_smpl_reference(sp).local_joints, reference.local_joints)


@pytest.mark.parametrize("problem", ["fps", "missing", "pose_shape", "joints_shape", "nan"])
def test_smpl_rejects_invalid_training_data(tmp_path, contract, problem):
    sp, _, content = _write_pair(tmp_path, contract)
    if problem == "fps":
        content["fps"] = 30
    elif problem == "missing":
        del content["smpl_joints"]
    elif problem == "pose_shape":
        content["pose_aa"] = content["pose_aa"].reshape(-1, 24, 3)
    elif problem == "joints_shape":
        content["smpl_joints"] = content["smpl_joints"][:-1]
    else:
        content["smpl_joints"][0, 0, 0] = np.nan
    joblib.dump(content, sp)
    with pytest.raises(ValueError):
        load_smpl_reference(sp)


def test_single_smpl_npz_container_and_default_initialization(tmp_path, contract):
    sp, _, content = _write_pair(tmp_path, contract)
    npz = tmp_path / "clip.npz"
    np.savez(npz, **content)
    np.testing.assert_array_equal(load_smpl_reference(npz).local_joints, load_smpl_reference(sp).local_joints)
    joblib.dump({"first": content, "second": content}, sp)
    with pytest.raises(ValueError, match="motion_key"):
        load_smpl_reference(sp)
    motion = load_smpl_reference_motion(sp, contract, motion_key="second")
    runner = Bumi3SonicSim2Sim(
        contract, motion, ZeroPolicy(contract, encoder="smpl"), encoder="smpl", start_paused=True,
    )
    np.testing.assert_allclose(runner.data.qpos[:3], contract.initial_root_position)
    np.testing.assert_allclose(runner.data.qpos[runner.qpos_addresses], contract.default_mujoco)
    assert runner.reference_marker_specs(0) == []
    assert runner.build_observation().shape == (1470,)


def test_dataset_pair_validation_and_smpl_only_entry(tmp_path, contract):
    sp, rp, content = _write_pair(tmp_path, contract)
    entry = {"name": "chosen", "robot": rp.name, "smpl": sp.name, "motion_key": "clip"}
    path = tmp_path / "dataset.json"
    manifest = {"version": 1, "robot_type": "bumi3", "motions": [entry]}
    path.write_text(json.dumps(manifest))
    motion = load_motion_dataset(path, contract, encoder="smpl")[0]
    assert motion.name == "chosen" and motion.has_robot_reference
    content["pose_aa"] = content["pose_aa"][:-1]
    content["smpl_joints"] = content["smpl_joints"][:-1]
    joblib.dump(content, sp)
    with pytest.raises(ValueError, match="配对帧数"):
        load_motion_dataset(path, contract, encoder="smpl")
    del entry["robot"]
    path.write_text(json.dumps(manifest))
    assert not load_motion_dataset(path, contract, encoder="smpl")[0].has_robot_reference
    with pytest.raises(ValueError, match="robot"):
        load_motion_dataset(path, contract)
    del entry["smpl"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="smpl"):
        load_motion_dataset(path, contract, encoder="smpl")


def test_smpl_hold_future_stride_end_loop_and_t_p(tmp_path, contract):
    sp, rp, _ = _write_pair(tmp_path, contract)
    first = load_smpl_reference_motion(sp, contract, robot_path=rp)
    second = replace(first, name="second", smpl_reference=replace(
        first.smpl_reference, local_joints=first.smpl_reference.local_joints + 0.01,
    ))
    runner = Bumi3SonicSim2Sim(
        contract, first, ZeroPolicy(contract, encoder="smpl"), encoder="smpl",
        start_paused=True, motions=[first, second],
    )
    def observed_joints():
        return runner.build_observation()[:720].reshape(10, 24, 3)

    np.testing.assert_array_equal(observed_joints(), np.repeat(first.smpl_reference.local_joints[:1], 10, axis=0))
    runner.enqueue_key(ord("T"))
    runner._process_key_events()
    np.testing.assert_array_equal(observed_joints(), first.smpl_reference.local_joints[:10])
    runner.motion_frame = first.num_frames - 2
    expected = first.smpl_reference.local_joints[[14] + [15] * 9]
    np.testing.assert_array_equal(observed_joints(), expected)
    runner.loop_motion = True
    np.testing.assert_array_equal(observed_joints(), first.smpl_reference.local_joints[[14, 15, *range(8)]])
    runner.loop_motion = False
    runner.last_action_policy.fill(3)
    runner.enqueue_key(ord("P"))
    runner._process_key_events()
    assert runner.motion.name == "second" and runner.motion_frame == 0 and not runner.playing
    assert not runner.last_action_policy.any()
    np.testing.assert_array_equal(observed_joints(), np.repeat(second.smpl_reference.local_joints[:1], 10, axis=0))
    runner.enqueue_key(ord("T"))
    runner._process_key_events()
    runner.motion_frame = first.num_frames - 2
    runner.step_control()
    assert runner.motion_frame == first.num_frames - 1 and not runner.playing
    np.testing.assert_array_equal(observed_joints(), np.repeat(second.smpl_reference.local_joints[-1:], 10, axis=0))
    # 改配对机器人的关节参考不改变人体 tokenizer；实际状态及人体参考保持不变。
    before = runner.build_observation()[:780].copy()
    runner.motion = replace(runner.motion, joint_pos_policy=runner.motion.joint_pos_policy + 0.3)
    np.testing.assert_array_equal(runner.build_observation()[:780], before)


def test_reject_encoder_mismatch_and_cli_motion_selection(tmp_path, contract, monkeypatch, capsys):
    from gear_sonic.scripts import run_bumi3_sim2sim as cli

    sp, rp, _ = _write_pair(tmp_path, contract)
    motion = load_smpl_reference_motion(sp, contract, robot_path=rp)
    with pytest.raises(ValueError, match="input_dim"):
        Bumi3SonicSim2Sim(contract, motion, ZeroPolicy(contract), encoder="smpl")
    with pytest.raises(ValueError, match="encoder"):
        contract.policy_input_dim("wrong")
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"version": 1, "robot_type": "bumi3", "motions": [
        {"name": n, "robot": rp.name, "smpl": sp.name} for n in ("one", "two", "three")
    ]}))
    monkeypatch.setattr(cli, "OnnxRobotPolicy", lambda _, c, provider, encoder: ZeroPolicy(c, encoder=encoder))
    cli.main(cli.Args(
        policy=tmp_path / "unused.onnx", encoder="smpl", dataset=path,
        motion_name="two", validate_only=True,
    ))
    output = capsys.readouterr().out
    resolved = json.JSONDecoder().raw_decode(output.split("BUMI3_SIM2SIM_RESOLVED=", 1)[1])[0]
    assert resolved["motion_names"] == ["two", "three", "one"]
    assert resolved["policy_input_dim"] == 1470 and resolved["future_frame_stride"] == 1
    assert "BUMI3_SIM2SIM_VALIDATE_ONLY=PASS" in output
