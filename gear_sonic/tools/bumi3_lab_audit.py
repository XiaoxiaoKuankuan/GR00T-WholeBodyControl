"""采集 BUMI3 在 Isaac Lab 中的部署对照证据。

本模块作为 eval_agent_trl.py 的可选 Hydra 回调加载，只支持单环境、有限步数评估。
它在每次环境 step 前记录真正送入策略的观测、确定性动作、机器人状态和参考时间，
并保存实际观测排列、关节顺序、PD 参数、动作缩放与刚体质量等运行时信息。
输出的 NPZ/JSON 可用于在相同输入下比较 PyTorch 与 ONNX，以及重建 MuJoCo 观测，
避免把两个仿真器自由演化后的状态差异误判为观测代码错误。

回调不修改策略、奖励、动作、物理参数或终止条件，不执行训练或 ONNX 导出。
输出目录必须是本次诊断专用目录；既有采集文件存在时拒绝覆盖。达到 max_steps
后通知评估循环退出。模块顶层不导入 Isaac Sim，确保由正式评估入口先启动应用。
"""

import json
from pathlib import Path

import numpy as np


def _array(tensor):
    """复制张量到 CPU，防止下一次仿真更新覆盖已采集状态。"""
    return tensor.detach().cpu().numpy().copy()


class Bumi3LabAuditCallback:
    """通过既有评估回调接口采样，保留真实 reset、推理和终止行为。"""

    def __init__(self, output_dir: str, max_steps: int = 219):
        if max_steps <= 0:
            raise ValueError("max_steps 必须为正数")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("lab_trace.npz", "lab_contract.json"):
            if (self.output_dir / filename).exists():
                raise FileExistsError(f"拒绝覆盖既有诊断文件: {self.output_dir / filename}")
        self.max_steps = max_steps
        self.rows = []

    def on_step_end(self, args, state, control, *, env, model, accelerator):
        """模型加载后挂接一次采集器；只观察原始 step 的输入输出。"""
        if env.num_envs != 1:
            raise ValueError("部署对照采集要求 num_envs=1")
        self.env = env
        self.model = model
        robot = env.env.scene["robot"]
        observations = env.env.observation_manager
        action_term = env.env.action_manager.get_term("joint_pos")
        metadata = {
            "checkpoint_step": int(state.global_step),
            "joint_names": robot.joint_names,
            "body_names": robot.body_names,
            "observation_names": observations._group_obs_term_names,
            "observation_dims": {
                group: [[int(dim) for dim in shape] for shape in shapes]
                for group, shapes in observations._group_obs_term_dim.items()
            },
            "action_scale": _array(action_term._scale).tolist(),
            "action_offset": _array(action_term._offset).tolist(),
            "root_body": robot.body_names[0],
            "motion_anchor_body": env.motion_command.cfg.anchor_body,
            "sim_dt": float(env.env.physics_dt),
            "control_dt": float(env.env.step_dt),
        }
        for field in ("default_joint_pos", "joint_stiffness", "joint_damping", "joint_armature",
                      "joint_friction_coeff", "joint_effort_limits", "joint_vel_limits", "default_mass"):
            value = getattr(robot.data, field, None)
            if value is not None:
                metadata[field] = _array(value).tolist()
        for field, getter in (("actual_mass", "get_masses"), ("actual_inertia", "get_inertias"),
                              ("actual_com", "get_coms"), ("material", "get_material_properties")):
            metadata[field] = _array(getattr(robot.root_physx_view, getter)()).tolist()
        (self.output_dir / "lab_contract.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        original_step = env.step

        def traced_step(actor_state):
            """保存当前观测对应的状态及动作，然后调用未修改的环境 step。"""
            obs = actor_state.get("obs_dict")
            if obs is None:
                # 常规评估不返回 obs_dict；读取 Actor 已填充的推理缓冲，保持其开关不变。
                obs = self.model.policy.obs_dict_buffer
            command = env.motion_command
            row = {
                "actor_obs": _array(obs["actor_obs"]).reshape(-1),
                "tokenizer": _array(obs["tokenizer"]).reshape(-1),
                "action": _array(actor_state["actions"]).reshape(-1),
                "root_pose": _array(robot.data.root_link_pose_w)[0],
                "root_velocity": _array(robot.data.root_link_vel_w)[0],
                "joint_pos": _array(robot.data.joint_pos)[0],
                "joint_vel": _array(robot.data.joint_vel)[0],
                "body_pos": _array(robot.data.body_pos_w)[0],
                "body_quat": _array(robot.data.body_quat_w)[0],
                "reference_frame": _array(command.motion_start_time_steps + command.time_steps)[0],
                "future_frames": _array(command.future_time_steps).reshape(-1),
                "reference_joint_pos": _array(command.joint_pos_multi_future).reshape(-1),
                "reference_joint_vel": _array(command.joint_vel_multi_future).reshape(-1),
                "reference_anchor_quat": _array(command.anchor_quat_w_multi_future).reshape(-1),
                "encoder_index": _array(command.episode_encoder_index).reshape(-1),
            }
            result = original_step(actor_state)
            row["done"] = _array(result[2]).reshape(-1)
            row["terminated"] = _array(env.env.reset_terminated).reshape(-1)
            row["time_out"] = _array(env.env.reset_time_outs).reshape(-1)
            row["reward"] = _array(result[1]).reshape(-1)
            row["joint_torque"] = _array(robot.data.applied_torque)[0]
            self.rows.append(row)
            return result

        env.step = traced_step

    def eval_step(self, env, results):
        """步数达到上限后落盘并结束评估；保留 done，避免 reset 掩盖失败。"""
        if len(self.rows) < self.max_steps:
            return False
        arrays = {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
        np.savez_compressed(self.output_dir / "lab_trace.npz", **arrays)
        heights = arrays["root_pose"][:, 2]
        print("BUMI3_LAB_AUDIT=" + json.dumps({
            "steps": len(self.rows), "done_steps": np.flatnonzero(arrays["done"].reshape(-1)).tolist(),
            "root_height_min": float(heights.min()), "root_height_max": float(heights.max()),
            "peak_joint_velocity": float(np.abs(arrays["joint_vel"]).max()),
            "terminated_steps": np.flatnonzero(arrays["terminated"].reshape(-1)).tolist(),
            "timeout_steps": np.flatnonzero(arrays["time_out"].reshape(-1)).tolist(),
            "output_dir": str(self.output_dir),
        }, ensure_ascii=False), flush=True)
        return True
