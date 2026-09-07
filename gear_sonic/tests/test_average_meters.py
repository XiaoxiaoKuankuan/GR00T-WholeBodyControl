"""验证训练日志平均器对浮点、整数和布尔指标的统一数值契约。

SONIC 会把每个 rollout 的环境指标先积累，再在 iteration 结束时求均值。连续奖励通常是
浮点张量，而动作数、隔离数等计数可能由 ``long`` 或 ``bool`` 张量产生。本文件用纯 CPU
合成张量验证三类输入都能稳定输出可记录的均值，防止日志辅助指标在真实多卡训练首轮结束
时中断 PPO。测试不创建 Isaac Lab 环境，也不读取训练数据或写入临时产物。
"""

import torch

from gear_sonic.utils.average_meters import TensorAverageMeter


def test_tensor_average_meter_preserves_floating_mean() -> None:
    """已有浮点指标应保持原 dtype，并按所有样本求均值。"""

    meter = TensorAverageMeter()
    meter.add(torch.tensor([1.0, 3.0], dtype=torch.float64))
    result = meter.mean()

    assert result.dtype == torch.float64
    assert result.item() == 2.0


def test_tensor_average_meter_accepts_long_counts() -> None:
    """整数计数应转换为浮点后求均值，不能触发 PyTorch dtype 异常。"""

    meter = TensorAverageMeter()
    meter.add(torch.tensor(2, dtype=torch.long))
    meter.add(torch.tensor(4, dtype=torch.long))

    assert meter.mean().item() == 3.0


def test_tensor_average_meter_accepts_boolean_flags() -> None:
    """布尔标志也应得到等价于真值比例的浮点均值。"""

    meter = TensorAverageMeter()
    meter.add(torch.tensor([True, False, True]))

    torch.testing.assert_close(meter.mean(), torch.tensor(2.0 / 3.0))
