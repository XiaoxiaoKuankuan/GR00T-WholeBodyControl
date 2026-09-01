# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""验证 TrackingCommand 的参考锚点姿态始终使用配置指定的刚体。

BUMI3 的浮动根是 ``base_link``，而 SONIC 跟踪锚点是 ``waist_yaw_link``。两者在
``waist_yaw_joint`` 非零时姿态不同，因此 Robot Encoder 不能用 MotionLib root
quaternion 代替 waist 的 FK quaternion。本测试不创建 Isaac Lab 场景，因为普通
``pytest`` 收集阶段尚未启动 SimulationApp，直接导入 manager-env 模块会缺少 ``pxr``。
测试改为解析实际生产源码的 AST，锁定单帧和多未来帧属性只能读取命名 anchor，且不得
重新调用 ``get_root_quat_w``。sim2sim 的数值 FK 行为由相邻的
``test_bumi3_sim2sim.py`` 独立覆盖。
"""

import ast
from pathlib import Path


COMMANDS_PATH = Path(__file__).resolve().parents[1] / "envs/manager_env/mdp/commands.py"


def _method_attribute_names(method_name: str) -> set[str]:
    """从 TrackingCommand 指定方法中收集全部属性访问名称。"""

    tree = ast.parse(COMMANDS_PATH.read_text(encoding="utf-8"))
    tracking_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TrackingCommand"
    )
    method = next(
        node
        for node in tracking_class.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return {node.attr for node in ast.walk(method) if isinstance(node, ast.Attribute)}


def test_single_frame_relative_orientation_uses_named_anchor() -> None:
    """单帧相对姿态必须读取 reference anchor，而不是 MotionLib root。"""

    attributes = _method_attribute_names("root_rot_dif_l")
    assert "anchor_quat_w" in attributes
    assert "robot_anchor_quat_w" in attributes
    assert "get_root_quat_w" not in attributes


def test_multi_future_relative_orientation_uses_named_anchor() -> None:
    """多未来帧 tokenizer 必须读取 waist FK quaternion 序列。"""

    attributes = _method_attribute_names("root_rot_dif_l_multi_future")
    assert "anchor_quat_w_multi_future" in attributes
    assert "robot_anchor_quat_w" in attributes
    assert "get_root_quat_w" not in attributes
