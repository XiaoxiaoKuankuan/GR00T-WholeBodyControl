"""验证 TrackingCommand 自适应采样结算使用旧动作身份且严格早于重采样写入。

本回归测试针对 Isaac Lab 的关键生命周期：环境先计算 termination，再在同一步内
reset 并重采样命令，最后才调用 ``TrackingCommand._update_command``。生产代码不能把
上一条 episode 的 ``reset_terminated`` 与 reset 后的新 ``motion_ids`` 拼在一起。

普通 pytest 尚未启动 Isaac Sim ``SimulationApp``，直接导入 manager-env 模块会依赖
``pxr``。因此这里解析真实生产源码的 AST，锁定以下结构契约：旧动作必须在
``_resample_command`` 覆盖 ID/时间游标前结算；``_update_command`` 只能统计未 reset
环境；训练器主动 ``reset_all`` 必须先声明为外部中断。MotionLib 的计数数值与错误输入
边界由 ``test_motion_lib_adaptive_quarantine.py`` 使用真实 Torch 张量独立验证。
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMANDS_PATH = REPO_ROOT / "gear_sonic/envs/manager_env/mdp/commands.py"
WRAPPER_PATH = REPO_ROOT / "gear_sonic/envs/wrapper/manager_env_wrapper.py"


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    """从生产源码中取得指定类方法节点，缺失时让测试立即失败。"""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _called_attribute_lines(method: ast.FunctionDef, attribute_name: str) -> list[int]:
    """返回方法中调用指定属性方法的源码行号。"""

    return [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attribute_name
    ]


def _assigned_self_subscript_lines(method: ast.FunctionDef, attribute_name: str) -> list[int]:
    """返回写入 ``self.<attribute>[...]`` 的源码行号。"""

    lines = []
    for node in ast.walk(method):
        if not isinstance(node, (ast.Assign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "self"
                and target.value.attr == attribute_name
            ):
                lines.append(node.lineno)
    return lines


def test_old_outcome_is_recorded_before_motion_identity_is_overwritten() -> None:
    """旧 episode 结算调用必须严格早于 motion ID 和时间游标的第一次写入。"""

    method = _class_method(COMMANDS_PATH, "TrackingCommand", "_resample_command")
    record_lines = _called_attribute_lines(method, "_record_adaptive_sampling_before_resample")
    overwrite_lines = [
        *_assigned_self_subscript_lines(method, "time_steps"),
        *_assigned_self_subscript_lines(method, "motion_ids"),
        *_assigned_self_subscript_lines(method, "motion_start_time_steps"),
    ]

    assert record_lines
    assert overwrite_lines
    assert min(record_lines) < min(overwrite_lines)


def test_post_reset_update_never_combines_new_motion_with_old_termination() -> None:
    """常规逐步统计必须排除 reset 环境，且不得读取旧 reset_terminated。"""

    method = _class_method(COMMANDS_PATH, "TrackingCommand", "_update_command")
    attributes = {node.attr for node in ast.walk(method) if isinstance(node, ast.Attribute)}
    string_literals = {
        node.value for node in ast.walk(method) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "reset_buf" in attributes | string_literals
    assert "reset_terminated" not in attributes
    assert _called_attribute_lines(method, "update_adaptive_sampling")


def test_external_reset_is_declared_before_environment_reset() -> None:
    """初始化和 motion 换批必须先标记人工中断，再让环境覆盖旧动作。"""

    method = _class_method(WRAPPER_PATH, "ManagerEnvWrapper", "reset")
    prepare_lines = _called_attribute_lines(method, "prepare_adaptive_sampling_external_reset")
    reset_lines = [
        node.lineno
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reset"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "env"
    ]

    assert prepare_lines
    assert reset_lines
    assert min(prepare_lines) < min(reset_lines)


def test_reset_settlement_uses_explicit_failure_timeout_and_old_motion_fields() -> None:
    """重采样前结算必须同时读取旧身份、旧时间、termination 与 timeout。"""

    method = _class_method(
        COMMANDS_PATH,
        "TrackingCommand",
        "_record_adaptive_sampling_before_resample",
    )
    attributes = {node.attr for node in ast.walk(method) if isinstance(node, ast.Attribute)}

    assert {
        "motion_ids",
        "motion_start_time_steps",
        "time_steps",
        "reset_terminated",
        "reset_time_outs",
        "update_adaptive_sampling",
        "record_adaptive_sampling_outcomes",
    } <= attributes
