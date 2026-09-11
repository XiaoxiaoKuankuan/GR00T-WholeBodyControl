"""验证 BUMI3 多轨迹清单与 MuJoCo 播放交互。

本文件用临时 PKL/JSON/YAML 检查清单路径、配对文件存在性与轨迹顺序，并通过
真实 MuJoCo 动力学配合零策略验证首帧保持、T 开始、P 切换、末帧停止和历史重置。
窗口集成测试替换可视窗口对象，直接触发传给 launch_passive 的键盘回调，确认
无时长 GUI 能持续等待且关闭窗口后退出；它不等价于人工图形界面或策略质量验收。
测试不连接服务器、不启动训练，不修改正式动作、模型或机器人资产。
"""

from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import yaml

from gear_sonic.utils.mujoco_sim.bumi3_motion_dataset import load_motion_dataset
from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import (
    Bumi3Contract,
    Bumi3SonicSim2Sim,
    DEFAULT_BUMI3_SIM2SIM_CONFIG,
    ZeroPolicy,
    make_static_reference_motion,
)


@pytest.fixture
def contract():
    """使用仓库现有资产契约，使交互检查仍经过真实关节映射和 MuJoCo reset。"""
    return Bumi3Contract.from_yaml(DEFAULT_BUMI3_SIM2SIM_CONFIG)


@pytest.fixture
def runner(contract):
    """构造位置、速度均有变化的两条短参考，避免静态数据掩盖暂停窗口错误。"""
    motions = []
    for index in range(2):
        motion = make_static_reference_motion(contract, num_frames=8)
        positions = motion.joint_pos_policy.copy()
        positions[:, 0] += index * 0.05 + np.arange(8) * 0.002
        motions.append(replace(
            motion, name=f"clip_{index}", joint_pos_policy=positions,
            joint_vel_policy=np.full_like(positions, 0.1),
        ))
    return Bumi3SonicSim2Sim(
        contract, motions[0], ZeroPolicy(contract), motions=motions, start_paused=True,
    )


def _write_dataset(tmp_path: Path, contract) -> dict:
    """生成仅供加载测试使用的配对路径；SMPL 内容不属于 MuJoCo Robot 分支输入。"""
    (tmp_path / "robot").mkdir()
    (tmp_path / "smpl").mkdir()
    for index in range(2):
        motion = {
            "dof": np.repeat(contract.default_mujoco[None], 4, axis=0),
            "root_rot": np.repeat([[0.0, 0.0, 0.0, 1.0]], 4, axis=0),
            "root_trans_offset": np.repeat([[0.0, 0.0, 0.5]], 4, axis=0),
            "fps": 50,
        }
        joblib.dump({"inside": motion}, tmp_path / "robot" / f"{index}.pkl")
        joblib.dump({}, tmp_path / "smpl" / f"{index}.pkl")
    return {"version": 1, "robot_type": "bumi3", "motions": [
        {"name": f"name_{index}", "robot": f"robot/{index}.pkl",
         "smpl": f"smpl/{index}.pkl", "motion_key": "inside"}
        for index in (1, 0)
    ]}


@pytest.mark.parametrize("suffix", ["json", "yaml"])
def test_dataset_relative_paths_order_and_names(tmp_path, monkeypatch, contract, suffix):
    content = _write_dataset(tmp_path, contract)
    path = tmp_path / f"dataset.{suffix}"
    path.write_text(json.dumps(content) if suffix == "json" else yaml.safe_dump(content))
    monkeypatch.chdir(tmp_path.parent)
    motions = load_motion_dataset(path, contract)
    assert [motion.name for motion in motions] == ["name_1", "name_0"]
    assert [motion.num_frames for motion in motions] == [4, 4]


@pytest.mark.parametrize("problem", ["empty", "duplicate", "robot_type", "robot", "smpl", "fps"])
def test_dataset_rejects_unusable_entries(tmp_path, contract, problem):
    content = _write_dataset(tmp_path, contract)
    if problem == "empty":
        content["motions"] = []
    elif problem == "duplicate":
        content["motions"][1]["name"] = content["motions"][0]["name"]
    elif problem == "robot_type":
        content["robot_type"] = "g1"
    elif problem in ("robot", "smpl"):
        content["motions"][0][problem] = "missing.pkl"
    else:
        robot_path = tmp_path / content["motions"][0]["robot"]
        robot = joblib.load(robot_path)
        robot["inside"]["fps"] = 30
        joblib.dump(robot, robot_path)
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(content))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_motion_dataset(path, contract)


def _assert_held_tokenizer(runner):
    """保持阶段的整个未来窗口都应等于当前帧，且所有参考关节速度为零。"""
    count = runner.contract.num_future_frames
    width = count * runner.contract.action_dim
    tokenizer = runner._build_robot_tokenizer()
    positions = np.repeat(runner.motion.joint_pos_policy[runner.motion_frame][None], count, axis=0)
    np.testing.assert_allclose(tokenizer[:width], positions.reshape(-1))
    np.testing.assert_array_equal(tokenizer[width:2 * width], 0.0)


def test_hold_advances_physics_but_not_reference(runner):
    np.testing.assert_array_equal(runner.data.qvel, 0.0)
    initial_qpos = runner.data.qpos.copy()
    for _ in range(3):
        runner.step_control()
    assert not runner.playing and runner.motion_frame == 0
    assert runner.data.time == pytest.approx(3 * runner.contract.control_dt)
    assert not np.array_equal(runner.data.qpos, initial_qpos)
    _assert_held_tokenizer(runner)


def test_start_switch_reset_and_wrap(runner):
    runner.enqueue_key(ord("T"))
    assert not runner.playing
    runner.step_control()
    assert runner.playing and runner.motion_frame == 1
    runner.enqueue_key(ord("t"))
    runner.step_control()
    assert runner.motion_frame == 2
    runner.enqueue_key(ord("P"))
    runner._process_key_events()
    assert runner.motion_index == 1 and runner.motion_frame == 0 and not runner.playing
    assert runner.data.time == 0
    # 换动作后仅允许根 Z 增加已记录的校正量，其余参考姿态必须完整保留。
    expected_qpos = runner._reference_qpos(0)
    expected_qpos[2] += runner.reset_ground_alignment["root_z_offset_m"]
    np.testing.assert_allclose(runner.data.qpos, expected_qpos)
    np.testing.assert_array_equal(runner.data.qvel, 0.0)
    np.testing.assert_array_equal(runner.last_action_policy, 0.0)
    for history in runner.histories.values():
        values = np.stack(history)
        np.testing.assert_allclose(values, np.repeat(values[-1:], len(history), axis=0))
    _assert_held_tokenizer(runner)
    runner.enqueue_key(ord("p"))
    runner.step_control()
    assert runner.motion_index == 0 and runner.motion_frame == 0 and not runner.playing


def test_switch_corrects_each_motion_once_and_start_preserves_state(contract, capsys):
    """P 切换按每条原始高度重新校正；等待时按 T 不能再次移动机器人或重置速度。"""
    motions = []
    for index, height in enumerate((0.44, 0.42)):
        motion = make_static_reference_motion(contract, num_frames=4)
        roots = motion.root_position_world.copy()
        roots[:, 2] = height
        motions.append(replace(motion, name=f"penetrating_{index}", root_position_world=roots))
    runner = Bumi3SonicSim2Sim(
        contract, motions[0], ZeroPolicy(contract), motions=motions, start_paused=True,
    )
    first_z = runner.data.qpos[2]
    first_offset = runner.reset_ground_alignment["root_z_offset_m"]
    runner.enqueue_key(ord("P"))
    runner._process_key_events()
    assert runner.reset_ground_alignment["motion_name"] == "penetrating_1"
    assert runner.reset_ground_alignment["root_z_offset_m"] == pytest.approx(first_offset + 0.02)
    assert runner.data.qpos[2] == pytest.approx(first_z)
    runner.data.qpos[2] += 0.01
    runner.data.qvel[0] = 0.05
    before_qpos, before_qvel = runner.data.qpos.copy(), runner.data.qvel.copy()
    runner.enqueue_key(ord("T"))
    runner._process_key_events()
    np.testing.assert_array_equal(runner.data.qpos, before_qpos)
    np.testing.assert_array_equal(runner.data.qvel, before_qvel)
    assert runner.playing
    assert capsys.readouterr().out.count("BUMI3_RESET_GROUND_ALIGNMENT=") == 2


def test_end_hold_restart_and_ignored_key(runner):
    runner.enqueue_key(ord("N"))
    runner.step_control()
    assert not runner.playing
    runner.enqueue_key(ord("T"))
    for _ in range(runner.motion.num_frames + 2):
        runner.step_control()
    assert runner.motion_frame == runner.motion.num_frames - 1
    assert not runner.playing
    _assert_held_tokenizer(runner)
    runner.enqueue_key(ord("T"))
    runner.step_control()
    assert runner.motion_frame == 1 and runner.playing


def test_loop_keeps_playing_and_single_motion_remains_supported(contract):
    motion = make_static_reference_motion(contract, num_frames=2)
    runner = Bumi3SonicSim2Sim(contract, motion, ZeroPolicy(contract), loop_motion=True)
    stats = runner.run(3, headless=True)
    assert runner.playing and runner.motion_frame == 1
    assert stats["completed_control_steps"] == 3
    with pytest.raises(ValueError, match="有限"):
        runner.run(None, headless=True)


def test_gui_callback_reaches_controller_and_close_exits(monkeypatch, runner):
    """经过真实 run 路径投递按键，验证窗口接线而不依赖图形服务器。"""
    from mujoco import viewer as mujoco_viewer

    statuses = []

    class FakeViewer:
        """记录每次同步时的状态，并依次模拟 T、P 和窗口关闭。"""
        def __init__(self, callback):
            self.callback = callback
            self.closed = False

        def is_running(self):
            return len(statuses) < 5

        def sync(self):
            statuses.append(runner.playback_status())
            if len(statuses) == 1:
                self.callback(ord("T"))
            elif len(statuses) == 3:
                self.callback(ord("P"))

        def lock(self):
            return nullcontext()

        def close(self):
            self.closed = True

    windows = []

    def launch(model, data, *, key_callback):
        window = FakeViewer(key_callback)
        windows.append(window)
        return window

    monkeypatch.setattr(mujoco_viewer, "launch_passive", launch)
    stats = runner.run(None, headless=False, real_time=False, show_reference=False)
    assert stats["completed_control_steps"] == 5
    assert statuses[0]["frame"] == 0 and not statuses[0]["playing"]
    assert statuses[1]["frame"] == 1 and statuses[1]["playing"]
    assert statuses[3]["motion_index"] == 1 and statuses[3]["frame"] == 0
    assert not statuses[3]["playing"] and windows[0].closed
