"""策略引擎模块: 策略注册表与工厂。"""

from qtcore.strategy.base import StrategyBase
from qtcore.strategy.ma_cross import MACrossStrategy

# 策略注册表: 新策略只需在此登记即可被命令行/配置中心按名称加载
STRATEGY_REGISTRY: dict[str, type[StrategyBase]] = {
    "ma_cross": MACrossStrategy,
}


def create_strategy(name: str, params: dict | None = None) -> StrategyBase:
    """策略工厂: 按注册名创建策略实例。"""
    if name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"未注册的策略: {name!r}, 可用策略: {sorted(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[name](params)


__all__ = ["StrategyBase", "MACrossStrategy", "STRATEGY_REGISTRY", "create_strategy"]
