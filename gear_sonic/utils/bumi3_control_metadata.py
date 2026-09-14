"""BUMI3 联合 ONNX 的名义控制参数读写与校验。

本模块在导出时从实际 Isaac Lab 环境的机器人、动作配置中解析逐关节参数，
写入 ONNX 自带的 JSON 元数据。读取端按名称恢复 PD、默认角、动作缩放和力矩上限，
避免模型更换后继续使用部署 YAML 中另一组控制参数。导出只读取名义配置，
不读取被关节零位、增益等域随机化改变的第零个环境状态。
模式版本、策略顺序、时间单位和全部数值都必须完整且有效；旧模型缺少元数据时
明确报错并要求重新导出，不静默回退。模块不导入 Isaac Lab，便于纯 MuJoCo 部署。
"""

import json
from pathlib import Path
import re

import numpy as np


METADATA_KEY = "sonic_bumi3_control"
ARRAY_FIELDS = (
    "joint_stiffness", "joint_damping", "default_joint_pos", "action_scale",
    "joint_effort_limit",
)


def _resolve(value, name):
    """按完整正则匹配解析名义参数，拒绝重复覆盖或缺失关节。"""
    if isinstance(value, dict):
        matches = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError(f"名义参数必须唯一覆盖关节 {name}: {matches}")
        value = matches[0]
    if value is None:
        raise ValueError(f"关节 {name} 的名义参数未显式配置")
    return float(value)


def build_bumi3_control_metadata(env, *, action_clip):
    """从实际管理器环境的名义配置构造参数，不依赖随机化后的运行时张量。"""
    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    names = list(robot.joint_names)
    if list(action._joint_names) != names or not action.cfg.use_default_offset:
        raise ValueError("BUMI3 导出要求动作覆盖全部关节、顺序一致且使用默认角偏置")
    payload = {
        "version": 1, "robot_type": "bumi3", "joint_names": names,
        "policy_joint_names": names, "control_dt": float(env.step_dt),
        "action_clip": float(action_clip),
        "source": "IsaacLab robot.cfg.actuators/init_state and joint_pos.cfg.scale",
        "default_joint_pos": [_resolve(robot.cfg.init_state.joint_pos, n) for n in names],
        "action_scale": [_resolve(action.cfg.scale, n) for n in names],
    }
    for field, attr in (("joint_stiffness", "stiffness"),
                        ("joint_damping", "damping"),
                        ("joint_effort_limit", "effort_limit_sim")):
        values = []
        for name in names:
            groups = [a for a in robot.cfg.actuators.values()
                      if any(re.fullmatch(p, name) for p in a.joint_names_expr)]
            if len(groups) != 1:
                raise ValueError(f"执行器配置必须唯一覆盖 {name}")
            values.append(_resolve(getattr(groups[0], attr), name))
        payload[field] = values
    encoded = {METADATA_KEY: json.dumps(payload, ensure_ascii=False, allow_nan=False)}
    read_bumi3_control_metadata(encoded, names, env.step_dt)
    return encoded


def read_bumi3_control_metadata(metadata, policy_names, control_dt):
    """校验元数据并返回按策略关节顺序排列的数组，绝不根据数组长度猜顺序。"""
    if METADATA_KEY not in metadata:
        raise ValueError("ONNX 缺少 sonic_bumi3_control 控制元数据；请用新版 "
                         "eval_agent_trl.py 从原 checkpoint 重新导出，不能回退到 YAML PD")
    try:
        payload = json.loads(metadata[METADATA_KEY])
        if payload["version"] != 1 or payload["robot_type"] != "bumi3":
            raise ValueError("ONNX 控制元数据版本或机器人类型不匹配")
        names = payload["joint_names"]
        if (not isinstance(names, list) or not all(isinstance(n, str) for n in names)
                or len(names) != len(set(names)) or set(names) != set(policy_names)):
            raise ValueError("ONNX 控制参数关节名缺失、重复或不匹配")
        if payload["policy_joint_names"] != list(policy_names):
            raise ValueError("ONNX 策略输入输出关节顺序与 BUMI3 观测契约不一致")
        if not np.isclose(float(payload["control_dt"]), control_dt, rtol=0, atol=1e-9):
            raise ValueError("ONNX 策略周期与 sim2sim 不一致")
        indices = [names.index(n) for n in policy_names]
        result = {}
        for field in ARRAY_FIELDS:
            values = np.asarray(payload[field], dtype=np.float64)
            if values.shape != (len(names),) or not np.isfinite(values).all():
                raise ValueError(f"ONNX {field} 长度不正确或包含 NaN/Inf")
            if field != "default_joint_pos":
                invalid = values < 0 if field == "joint_damping" else values <= 0
                if np.any(invalid):
                    raise ValueError(f"ONNX {field} 存在不合法的非正数")
            result[field] = values[indices]
        clip = float(payload["action_clip"])
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("ONNX action_clip 必须为有限正数")
        result["action_clip"] = clip
        return result
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"ONNX 控制元数据格式错误或缺字段: {error}") from error


def attach_control_metadata(path, metadata):
    """只更新 ONNX 元数据；保留计算图、权重和其他已有元数据。"""
    import onnx

    path = Path(path)
    model = onnx.load(path)
    existing = {entry.key: entry.value for entry in model.metadata_props}
    existing.update(metadata)
    onnx.helper.set_model_props(model, existing)
    onnx.checker.check_model(model)
    onnx.save(model, path)
