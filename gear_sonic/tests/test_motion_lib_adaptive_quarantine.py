"""验证 MotionLib 动力学门禁与坏动作 quarantine 的双条件行为。

测试覆盖原始 DOF 的速度/加速度门禁、连续高失败计数、动力学通过动作不得被
隔离，以及 quarantine 动作只能获得 uniform 采样分量。测试使用 CPU 合成张量，
不读取用户数据、不创建 Isaac Lab 场景；因此它证明算法边界正确，但不宣称当前
三源数据中具体有多少动作会触发门禁。
"""

from __future__ import annotations

import torch

from gear_sonic.utils.motion_lib.motion_lib_base import (
    ADAPTIVE_SAMPLING_STATE_VERSION,
    MotionLibBase,
    evaluate_reference_dynamics,
)


def _dynamics_gate_cfg() -> dict:
    """返回两关节合成轨迹使用的严格门禁配置。"""

    return {
        "dof_names": ["joint_a", "joint_b"],
        "dof_velocity_limits": {"joint_a": 1.0, "joint_b": 2.0},
        "max_velocity_exceedance_fraction": 0.0,
        "max_velocity_ratio": 2.0,
        "max_acceleration_ratio": 1.0,
    }


def _minimal_quarantine_lib() -> MotionLibBase:
    """绕过数据加载，构造只含两个单分箱动作的最小 MotionLib 状态。"""

    motion_lib = MotionLibBase.__new__(MotionLibBase)
    motion_lib._device = torch.device("cpu")
    motion_lib._num_unique_motions = 2
    motion_lib.adp_samp_bins = torch.tensor([[0, 0, 50], [1, 0, 50]], dtype=torch.long)
    motion_lib.adp_samp_num_episodes = torch.tensor([0.0, 0.0])
    motion_lib.adp_samp_num_failures = torch.tensor([0.0, 0.0])
    motion_lib.adp_samp_dynamics_gate_failed = torch.tensor([True, False])
    motion_lib.adp_samp_motion_quarantined = torch.zeros(2, dtype=torch.bool)
    motion_lib.adp_samp_quarantine_consecutive = torch.zeros(2, dtype=torch.long)
    motion_lib.adp_samp_quarantine_eval_episodes = torch.zeros(2)
    motion_lib.adp_samp_motion_num_evaluations = torch.zeros(2)
    motion_lib.adp_samp_motion_num_failures = torch.zeros(2)
    motion_lib.adp_samp_motion_failure_rate = torch.zeros(2)
    motion_lib.adp_samp_init_num_failures = 0.0
    motion_lib.use_adaptive_sampling = True
    motion_lib.use_adaptive_quarantine = True
    motion_lib.quarantine_cfg = {
        "high_failure_rate": 0.9,
        "min_motion_episodes": 5.0,
        "min_new_motion_episodes": 3.0,
        "consecutive_evaluations": 3,
    }
    return motion_lib


def test_reference_dynamics_accepts_smooth_motion_and_rejects_spike() -> None:
    """平滑静态轨迹通过，单帧严重跳变触发速度和加速度门禁。"""

    smooth = torch.zeros(20, 2)
    smooth_result = evaluate_reference_dynamics(smooth, 50.0, _dynamics_gate_cfg())
    assert smooth_result["passed"] is True

    spike = smooth.clone()
    spike[10, 0] = 1.0
    spike_result = evaluate_reference_dynamics(spike, 50.0, _dynamics_gate_cfg())
    assert spike_result["passed"] is False
    assert spike_result["max_velocity_ratio"] > 2.0
    assert spike_result["max_acceleration_ratio"] > 1.0


def test_reference_dynamics_rejects_joint_name_contract_mismatch() -> None:
    """名称集合与 DOF 维度不一致时必须报错，禁止按错误顺序套速度上限。"""

    cfg = _dynamics_gate_cfg()
    cfg["dof_names"] = ["joint_a"]
    try:
        evaluate_reference_dynamics(torch.zeros(5, 2), 50.0, cfg)
    except ValueError as exc:
        assert "关节数" in str(exc)
    else:
        raise AssertionError("错误关节契约没有触发 ValueError")


def test_quarantine_requires_three_new_high_failure_evaluations_and_failed_gate() -> None:
    """只有门禁失败动作在三次新增高失败证据后进入 quarantine。"""

    motion_lib = _minimal_quarantine_lib()
    for total_episodes in (6.0, 9.0, 12.0):
        motion_lib.adp_samp_motion_num_evaluations[:] = total_episodes
        motion_lib.adp_samp_motion_num_failures[:] = total_episodes * 0.95
        motion_lib._update_adaptive_sampling_quarantine()

    assert motion_lib.adp_samp_quarantine_consecutive.tolist() == [3, 0]
    assert motion_lib.adp_samp_motion_quarantined.tolist() == [True, False]


def test_low_failure_evidence_resets_consecutive_counter() -> None:
    """一次有足够新增回合的低失败评估必须打断连续高失败计数。"""

    motion_lib = _minimal_quarantine_lib()
    motion_lib.adp_samp_motion_num_evaluations[:] = 6.0
    motion_lib.adp_samp_motion_num_failures[:] = 5.7
    motion_lib._update_adaptive_sampling_quarantine()
    assert motion_lib.adp_samp_quarantine_consecutive[0].item() == 1

    motion_lib.adp_samp_motion_num_evaluations[:] = 9.0
    motion_lib.adp_samp_motion_num_failures[:] = 4.5
    motion_lib._update_adaptive_sampling_quarantine()
    assert motion_lib.adp_samp_quarantine_consecutive[0].item() == 0


def test_motion_outcomes_count_early_failure_and_natural_completion() -> None:
    """动作级失败率必须由提前失败/自然跑完计数，不受分箱长度稀释。"""

    motion_lib = _minimal_quarantine_lib()
    motion_lib._curr_motion_ids = torch.tensor([0, 1], dtype=torch.long)
    motion_lib.adp_samp_num_frames = torch.tensor([100, 50], dtype=torch.long)
    motion_lib.adp_samp_length_starts = torch.tensor([0, 100], dtype=torch.long)
    motion_lib.adp_samp_frame_to_bin = torch.cat(
        [torch.zeros(100, dtype=torch.long), torch.ones(50, dtype=torch.long)]
    )
    motion_lib.adp_samp_num_bins = 2
    motion_lib.adp_samp_bin_motion_length = torch.tensor([100.0, 50.0])
    motion_lib.adaptive_sampling_cfg = {"failure_counts_multiplier": 1}

    motion_lib.update_adaptive_sampling(
        failure=torch.tensor([True, False]),
        motion_ids=torch.tensor([0, 1], dtype=torch.long),
        motion_time_steps=torch.tensor([20, 49], dtype=torch.long),
    )
    motion_lib.record_adaptive_sampling_outcomes(
        failure=torch.tensor([True, False]),
        success=torch.tensor([False, True]),
        motion_ids=torch.tensor([0, 1], dtype=torch.long),
    )

    assert motion_lib.adp_samp_motion_num_evaluations.tolist() == [1.0, 1.0]
    assert motion_lib.adp_samp_motion_num_failures.tolist() == [1.0, 0.0]
    _, failure_rate = motion_lib._motion_level_adaptive_statistics()
    assert failure_rate.tolist() == [1.0, 0.0]


def test_frame_statistics_do_not_implicitly_create_motion_outcomes() -> None:
    """帧曝光接口不得再依据时间游标猜测动作成功，防止 reset 后身份错配。"""

    motion_lib = _minimal_quarantine_lib()
    motion_lib._curr_motion_ids = torch.tensor([0, 1], dtype=torch.long)
    motion_lib.adp_samp_length_starts = torch.tensor([0, 100], dtype=torch.long)
    motion_lib.adp_samp_frame_to_bin = torch.cat(
        [torch.zeros(100, dtype=torch.long), torch.ones(50, dtype=torch.long)]
    )
    motion_lib.adp_samp_num_bins = 2
    motion_lib.adp_samp_bin_motion_length = torch.tensor([100.0, 50.0])
    motion_lib.adaptive_sampling_cfg = {"failure_counts_multiplier": 1}

    motion_lib.update_adaptive_sampling(
        failure=torch.tensor([False]),
        motion_ids=torch.tensor([1], dtype=torch.long),
        motion_time_steps=torch.tensor([49], dtype=torch.long),
    )

    assert motion_lib.adp_samp_motion_num_evaluations.sum().item() == 0.0
    assert motion_lib.adp_samp_motion_num_failures.sum().item() == 0.0


def test_timeout_success_is_counted_without_requiring_last_motion_frame() -> None:
    """达到环境 timeout 即形成成功证据，不要求随机起点 episode 覆盖整条片段。"""

    motion_lib = _minimal_quarantine_lib()
    motion_lib._curr_motion_ids = torch.tensor([0, 1], dtype=torch.long)

    motion_lib.record_adaptive_sampling_outcomes(
        failure=torch.tensor([False, True]),
        success=torch.tensor([True, False]),
        motion_ids=torch.tensor([0, 1], dtype=torch.long),
    )

    assert motion_lib.adp_samp_motion_num_evaluations.tolist() == [1.0, 1.0]
    assert motion_lib.adp_samp_motion_num_failures.tolist() == [0.0, 1.0]


def test_duplicate_motion_ids_are_aggregated_without_losing_successes() -> None:
    """多个环境并行执行同一动作时，成功和失败都必须按真实 episode 数累加。"""

    motion_lib = _minimal_quarantine_lib()
    motion_lib._curr_motion_ids = torch.tensor([0, 1], dtype=torch.long)

    motion_lib.record_adaptive_sampling_outcomes(
        failure=torch.tensor([True, False, False]),
        success=torch.tensor([False, True, True]),
        motion_ids=torch.tensor([0, 0, 1], dtype=torch.long),
    )

    assert motion_lib.adp_samp_motion_num_evaluations.tolist() == [2.0, 1.0]
    assert motion_lib.adp_samp_motion_num_failures.tolist() == [1.0, 0.0]


def test_motion_outcome_rejects_conflicting_success_and_failure() -> None:
    """同一 episode 同时成功和失败属于调用方错误，必须立即失败而非污染统计。"""

    motion_lib = _minimal_quarantine_lib()
    motion_lib._curr_motion_ids = torch.tensor([0, 1], dtype=torch.long)

    try:
        motion_lib.record_adaptive_sampling_outcomes(
            failure=torch.tensor([True]),
            success=torch.tensor([True]),
            motion_ids=torch.tensor([0], dtype=torch.long),
        )
    except ValueError as exc:
        assert "不能同时" in str(exc)
    else:
        raise AssertionError("冲突的动作结果没有触发 ValueError")


def test_checkpoint_marks_new_outcome_semantics_and_rejects_legacy_statistics() -> None:
    """新状态必须带版本；旧 checkpoint 的错位分箱和 quarantine 统计不得恢复。"""

    motion_lib = _minimal_quarantine_lib()
    new_state = motion_lib.get_state_dict()
    assert new_state["adaptive_sampling_state_version"] == ADAPTIVE_SAMPLING_STATE_VERSION

    legacy_state = {
        "adp_samp_num_episodes": torch.tensor([100.0, 200.0]),
        "adp_samp_num_failures": torch.tensor([90.0, 180.0]),
        "adp_samp_motion_num_evaluations": torch.tensor([10.0, 10.0]),
        "adp_samp_motion_num_failures": torch.tensor([10.0, 10.0]),
        "adp_samp_motion_quarantined": torch.tensor([True, True]),
    }
    motion_lib.load_state_dict(legacy_state)

    assert motion_lib.adp_samp_num_episodes.tolist() == [0.0, 0.0]
    assert motion_lib.adp_samp_num_failures.tolist() == [0.0, 0.0]
    assert motion_lib.adp_samp_motion_num_evaluations.tolist() == [0.0, 0.0]
    assert motion_lib.adp_samp_motion_num_failures.tolist() == [0.0, 0.0]
    assert motion_lib.adp_samp_motion_quarantined.tolist() == [False, False]


def test_quarantined_motion_keeps_only_uniform_sampling_component() -> None:
    """quarantine 动作不得再因高失败率获得困难采样加权。"""

    motion_lib = _minimal_quarantine_lib()
    motion_lib.adp_samp_failure_rate = torch.tensor([1.0, 0.1])
    motion_lib.adp_samp_active_motion_bins = torch.tensor([0, 1], dtype=torch.long)
    motion_lib.adp_samp_bin_weights = torch.ones(2)
    motion_lib.adp_samp_failure_rate_max_over_mean = 200.0
    motion_lib.uniform_sampling_rate = 0.2
    motion_lib.max_prob_per_bin_cfg = None
    motion_lib.max_prob_per_motion_cfg = None
    motion_lib.adp_samp_motion_quarantined[:] = torch.tensor([True, False])

    motion_lib.update_adaptive_sampling_probabilities()

    assert torch.allclose(
        motion_lib.adp_sampling_active_prob,
        torch.tensor([0.1, 0.9]),
        atol=1.0e-6,
    )
