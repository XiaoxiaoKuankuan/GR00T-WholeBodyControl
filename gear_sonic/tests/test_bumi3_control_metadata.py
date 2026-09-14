"""验证 BUMI3 ONNX 控制契约的完整生产与消费链路。

测试使用可计算的最小联合网络执行真实 ONNX 导出与推理；以故意区别于 YAML 的
PD、零位、动作缩放和力矩上限验证 MuJoCo 实际力矩，防止读取元数据却没有用于控制。
同时检查名称重排、缺失或损坏元数据拒绝、名义配置与随机化状态隔离、全部驱动关节
armature 以及未传入元数据时的通用导出兼容性。模型仅写入 pytest 临时目录。
"""

import json
from types import SimpleNamespace as NS

import numpy as np
import onnx
import pytest
import torch

from gear_sonic.utils.bumi3_control_metadata import (
    ARRAY_FIELDS, METADATA_KEY, build_bumi3_control_metadata, read_bumi3_control_metadata,
)
from gear_sonic.utils.inference_helpers import export_universal_token_module_as_onnx
from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import (
    Bumi3Contract, Bumi3SonicSim2Sim, DEFAULT_BUMI3_SIM2SIM_CONFIG,
    OnnxRobotPolicy, make_static_reference_motion,
)


def _fixture():
    """用与 YAML 不同的参数模拟实际环境；随机化后的状态故意填入错误值。"""
    c = Bumi3Contract.from_yaml(DEFAULT_BUMI3_SIM2SIM_CONFIG)
    names = list(c.policy_joint_names)
    gains = {n: 10.0 + i for i, n in enumerate(names)}
    cfg = NS(init_state=NS(joint_pos={n: 0.001 * i for i, n in enumerate(names)}),
             actuators={"all": NS(joint_names_expr=[".*"], stiffness=gains,
                                  damping=0.7, effort_limit_sim=2.0)})
    robot = NS(joint_names=names, cfg=cfg, data=NS(default_joint_pos=np.full((1, 21), 999)))
    action = NS(_joint_names=names, cfg=NS(use_default_offset=True, scale=0.2),
                _offset=np.full((1, 21), 999))
    env = NS(scene={"robot": robot}, step_dt=0.02,
             action_manager=NS(get_term=lambda name: action))
    return c, build_bumi3_control_metadata(env, action_clip=3.0)


class _ToyModule(torch.nn.Module):
    """最小联合网络保留可核对的动作输出，覆盖通用导出器的真实代码路径。"""

    def __init__(self):
        super().__init__()
        self.encoder_input_features = {"g1": ["ref"]}
        self.decoder_input_features = {"g1_dyn": ["proprioception"]}
        self.tokenizer_obs_names = ["ref"]
        self.tokenizer_obs_dims = {"ref": (480,)}
        self.obs_dim_dict = {"actor_obs": 690}

    def encode(self, name, obs):
        return obs["ref"].squeeze(1), None

    def decode(self, name, obs):
        return {"action": obs["proprioception"][..., :21] * 0.1}


@pytest.mark.parametrize("with_metadata", [False, True])
def test_real_export_and_actual_pd(tmp_path, with_metadata):
    """检查通用导出兼容性，并验证元数据最终作用于实际 MuJoCo motor 力矩。"""
    c, metadata = _fixture()
    path = tmp_path / "policy.onnx"
    export_universal_token_module_as_onnx(
        _ToyModule(), "g1", "g1_dyn", str(tmp_path), path.name,
        control_metadata=metadata if with_metadata else None,
    )
    props = {x.key: x.value for x in onnx.load(path).metadata_props}
    if not with_metadata:
        assert props == {}
        with pytest.raises(ValueError, match="缺少"):
            OnnxRobotPolicy(path, c)
        return
    policy = OnnxRobotPolicy(path, c)
    np.testing.assert_allclose(policy(np.ones(1170)), 0.1)
    # 故意传入原 YAML 契约，证明非 CLI 调用也使用策略绑定的 ONNX 参数。
    runner = Bumi3SonicSim2Sim(c.with_physics_substeps(5), make_static_reference_motion(c), policy)
    actual = runner.contract
    assert actual.sim_dt == 0.001
    assert actual.decimation == 20
    np.testing.assert_allclose(actual.default_policy, np.arange(21) * 0.001)
    np.testing.assert_allclose(actual.action_scale_policy, 0.2)
    np.testing.assert_allclose(actual.stiffness_mujoco, (10 + np.arange(21))[c.policy_to_mujoco])
    np.testing.assert_allclose(runner.model.dof_armature[runner.dof_addresses], 0.01)
    np.testing.assert_allclose(runner.model.dof_armature[:6], 0.0)
    runner.data.qpos[runner.qpos_addresses] = actual.default_mujoco + 0.1
    runner.data.qvel[runner.dof_addresses] = 0.5
    runner.last_action_policy[:] = 0.0
    runner._apply_pd_control()
    expected = np.clip(-actual.stiffness_mujoco * 0.1 - 0.7 * 0.5, -2, 2)
    np.testing.assert_allclose(runner.data.ctrl[runner.actuator_ids], expected)


def test_metadata_named_parameter_permutation():
    """参数表可按名字重排，但网络输出本身的策略关节顺序必须独立核验。"""
    c, metadata = _fixture()
    before = read_bumi3_control_metadata(metadata, c.policy_joint_names, c.control_dt)
    p = json.loads(metadata[METADATA_KEY])
    p["joint_names"].reverse()
    for key in ARRAY_FIELDS:
        p[key].reverse()
    after = read_bumi3_control_metadata({METADATA_KEY: json.dumps(p)}, c.policy_joint_names, c.control_dt)
    for key in before:
        np.testing.assert_equal(before[key], after[key])


@pytest.mark.parametrize("fault", ["missing", "duplicate", "policy_order", "nan", "negative",
                                  "short", "period", "wrong_robot", "bad_json"])
def test_reject_invalid_metadata(fault):
    """不完整模型和不一致契约必须在进入仿真前失败，避免静默调用旧 YAML 参数。"""
    c, metadata = _fixture()
    p = json.loads(metadata[METADATA_KEY])
    if fault == "missing":
        del p["joint_damping"]
    elif fault == "duplicate":
        p["joint_names"][1] = p["joint_names"][0]
    elif fault == "policy_order":
        p["policy_joint_names"].reverse()
    elif fault == "nan":
        p["joint_stiffness"][0] = float("nan")
    elif fault == "negative":
        p["joint_damping"][0] = -1
    elif fault == "short":
        p["default_joint_pos"].pop()
    elif fault == "period":
        p["control_dt"] = 0.01
    elif fault == "wrong_robot":
        p["robot_type"] = "g1"
    with pytest.raises(ValueError):
        read_bumi3_control_metadata(
            {METADATA_KEY: "{broken" if fault == "bad_json" else json.dumps(p)},
            c.policy_joint_names, c.control_dt,
        )
