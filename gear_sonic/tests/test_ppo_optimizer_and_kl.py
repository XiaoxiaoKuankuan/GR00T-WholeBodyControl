"""验证 SONIC PPO 的 Actor/Critic 独立学习率与整轮 KL 控制契约。

本测试使用极小的纯 PyTorch Policy/Value 模型，不创建 Isaac Lab 环境。它锁定
optimizer 参数必须按角色及 weight decay 具名分组、KL 只能修改 Actor、adaptive
模式不得创建 HuggingFace scheduler、旧版两组 checkpoint 可以确定性迁移动量，
并通过源码结构检查保证学习率调整不再发生在 micro-batch loss 内。这里验证的是
训练器控制流与 checkpoint 兼容性，不代表仿真 reset/step 或训练质量已经通过。
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
from torch import nn

from gear_sonic.trl.trainer.ppo_trainer import (
    OPTIMIZER_GROUP_NAME_KEY,
    OPTIMIZER_ROLE_KEY,
    PolicyAndValueWrapper,
    TRLPPOTrainer,
    build_role_aware_optimizer_groups,
    get_optimizer_learning_rates_by_role,
)


class _TinyActor(nn.Module):
    """提供一层权重和偏置，模拟 Actor 的 decay/no-decay 参数。"""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)


class _TinyCritic(nn.Module):
    """提供一层权重和偏置，模拟 Critic 的 decay/no-decay 参数。"""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)


class _UnwrapOnlyAccelerator:
    """只实现 legacy optimizer 迁移所需的模型解包接口。"""

    @staticmethod
    def unwrap_model(model):
        return model


def _build_model_and_groups():
    """构造固定名称的小模型及其角色参数组。"""

    model = PolicyAndValueWrapper(_TinyActor(), _TinyCritic())
    decay_names = {"policy.linear.weight", "value_model.linear.weight"}
    groups = build_role_aware_optimizer_groups(
        model,
        decay_parameter_names=decay_names,
        actor_learning_rate=2.0e-5,
        critic_learning_rate=1.0e-3,
        weight_decay=0.01,
    )
    return model, decay_names, groups


def test_optimizer_groups_have_independent_actor_and_critic_learning_rates() -> None:
    """Actor/Critic 各自的 decay/no-decay 组必须使用独立 LR。"""

    _, _, groups = _build_model_and_groups()
    optimizer = torch.optim.AdamW(groups, lr=2.0e-5)

    assert [group[OPTIMIZER_GROUP_NAME_KEY] for group in optimizer.param_groups] == [
        "actor_decay",
        "actor_no_decay",
        "critic_decay",
        "critic_no_decay",
    ]
    assert get_optimizer_learning_rates_by_role(optimizer) == {
        "actor": 2.0e-5,
        "critic": 1.0e-3,
    }
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_kl_adjustment_changes_only_actor_groups() -> None:
    """高 KL 必须降低 Actor LR，同时保持 Critic LR 完全不变。"""

    model, _, groups = _build_model_and_groups()
    trainer = TRLPPOTrainer.__new__(TRLPPOTrainer)
    trainer.optimizer = torch.optim.AdamW(groups, lr=2.0e-5)
    trainer.model = model
    trainer.value_model = model.value_model
    trainer.args = SimpleNamespace(learning_rate=9.9e-4)
    trainer.desired_kl = 0.01
    trainer.adaptive_lr_min = 1.0e-5
    trainer.adaptive_lr_max = 2.0e-4

    new_actor_lr = trainer._adjust_learning_rate_based_on_kl(0.03)

    assert new_actor_lr == 2.0e-5 / 1.5
    assert trainer.args.learning_rate == new_actor_lr
    assert get_optimizer_learning_rates_by_role(trainer.optimizer) == {
        "actor": new_actor_lr,
        "critic": 1.0e-3,
    }


def test_adaptive_schedule_does_not_create_huggingface_scheduler() -> None:
    """adaptive 模式必须把已有 scheduler 引用也清空。"""

    trainer = TRLPPOTrainer.__new__(TRLPPOTrainer)
    trainer.config = {"schedule": "adaptive"}
    trainer.lr_scheduler = object()

    assert trainer.create_scheduler(num_training_steps=100, optimizer=None) is None
    assert trainer.lr_scheduler is None
    assert trainer._created_lr_scheduler is False


def test_legacy_optimizer_checkpoint_migrates_moments_and_actual_actor_lr() -> None:
    """旧 checkpoint 恢复真实 Actor LR/Adam 状态，并给 Critic 启用新 LR。"""

    model, decay_names, groups = _build_model_and_groups()
    legacy_groups = [
        {
            "params": [parameter for name, parameter in model.named_parameters() if name in decay_names],
            "weight_decay": 0.0,
        },
        {
            "params": [parameter for name, parameter in model.named_parameters() if name not in decay_names],
            "weight_decay": 0.0,
        },
    ]
    legacy_optimizer = torch.optim.AdamW(legacy_groups, lr=2.0e-5)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    legacy_optimizer.step()
    for group in legacy_optimizer.param_groups:
        group["lr"] = 1.0e-5

    trainer = TRLPPOTrainer.__new__(TRLPPOTrainer)
    trainer.optimizer = torch.optim.AdamW(groups, lr=2.0e-5)
    trainer.model = model
    trainer.value_model = model.value_model
    trainer.accelerator = _UnwrapOnlyAccelerator()
    trainer.args = SimpleNamespace(learning_rate=2.0e-5)
    trainer.get_decay_parameter_names = lambda _: list(decay_names)

    restore_mode = trainer._load_optimizer_state_dict_compat(legacy_optimizer.state_dict())
    actual_lrs = trainer._validate_optimizer_learning_rate_contract("测试恢复")

    assert restore_mode == "legacy-migrated"
    assert actual_lrs == {"actor": 1.0e-5, "critic": 1.0e-3}
    assert trainer.args.learning_rate == 1.0e-5
    assert len(trainer.optimizer.state) == len(list(model.parameters()))


def test_role_aware_checkpoint_restores_each_actual_learning_rate() -> None:
    """新 schema checkpoint 必须分别恢复 Actor/Critic 实际 LR。"""

    model, decay_names, groups = _build_model_and_groups()
    saved_optimizer = torch.optim.AdamW(groups, lr=2.0e-5)
    for group in saved_optimizer.param_groups:
        if group[OPTIMIZER_ROLE_KEY] == "actor":
            group["lr"] = 1.0e-5
        elif group[OPTIMIZER_ROLE_KEY] == "critic":
            group["lr"] = 2.5e-4

    fresh_groups = build_role_aware_optimizer_groups(
        model,
        decay_parameter_names=decay_names,
        actor_learning_rate=2.0e-5,
        critic_learning_rate=1.0e-3,
        weight_decay=0.01,
    )
    trainer = TRLPPOTrainer.__new__(TRLPPOTrainer)
    trainer.optimizer = torch.optim.AdamW(fresh_groups, lr=2.0e-5)
    trainer.model = model
    trainer.value_model = model.value_model
    trainer.args = SimpleNamespace(learning_rate=2.0e-5)

    restore_mode = trainer._load_optimizer_state_dict_compat(saved_optimizer.state_dict())
    actual_lrs = trainer._validate_optimizer_learning_rate_contract("测试新 schema 恢复")

    assert restore_mode == "role-aware"
    assert actual_lrs == {"actor": 1.0e-5, "critic": 2.5e-4}
    assert trainer.args.learning_rate == 1.0e-5


def test_kl_adjustment_exists_once_in_iteration_and_not_in_loss() -> None:
    """源码控制流必须保证每轮一次 KL 调整，而不是每个 micro-batch 调整。"""

    loss_source = inspect.getsource(TRLPPOTrainer._compute_ppo_loss)
    train_source = inspect.getsource(TRLPPOTrainer.train)

    assert "_adjust_learning_rate_based_on_kl" not in loss_source
    assert train_source.count("self._adjust_learning_rate_based_on_kl(") == 1
    assert 'metrics["lr/actor_actual"]' in train_source
    assert 'metrics["lr/critic_actual"]' in train_source


def test_all_optimizer_groups_have_explicit_roles() -> None:
    """禁止生成缺少角色元数据、会被 KL 控制器误处理的参数组。"""

    _, _, groups = _build_model_and_groups()
    assert all(group[OPTIMIZER_ROLE_KEY] in {"actor", "critic"} for group in groups)
