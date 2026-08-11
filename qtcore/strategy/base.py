"""
StrategyBase 策略基类
====================

核心约束(架构红线):
1. 策略只负责"行情 -> 因子 -> 信号", 禁止直接操作资金或下单;
2. 信号只表达目标仓位状态(LONG / SHORT / FLAT), 由回测/执行引擎撮合;
3. 所有策略必须实现 compute_indicators() 与 target_positions();
4. 同一策略对象可同时用于回测、实盘模拟与 Docker 微服务, 无需修改策略代码。

接口说明:
- compute_indicators(bars) -> DataFrame : 在行情上追加因子列(可缓存)
- target_positions(bars)   -> Series   : 每根K线收盘的目标仓位 1/0/-1
- generate_signal_events() -> list[SignalEvent] : 由仓位变化生成事件流
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from qtcore.events import SignalAction, SignalEvent


class StrategyBase(ABC):
    """策略抽象基类。"""

    name: str = "base"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})
        self.validate_params()

    def validate_params(self) -> None:
        """参数校验钩子, 子类按需覆写。"""

    def get_param(self, key: str, default: Any = None) -> Any:
        """安全读取策略参数。"""
        return self.params.get(key, default)

    # ------------------------------------------------------------------
    # 子类必须实现
    # ------------------------------------------------------------------
    @abstractmethod
    def compute_indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """在行情 DataFrame 上计算因子, 返回追加因子列后的副本。"""

    @abstractmethod
    def target_positions(self, bars: pd.DataFrame) -> pd.Series:
        """
        计算每根K线收盘后的目标仓位。

        返回:
            pd.Series, index 与 bars 一致, 取值 1(多) / 0(空仓) / -1(空)
        """

    # ------------------------------------------------------------------
    # 通用信号事件生成(不需要子类覆写)
    # ------------------------------------------------------------------
    def generate_signal_events(self, bars: pd.DataFrame) -> list[SignalEvent]:
        """
        由目标仓位序列的变化生成 SignalEvent 列表。
        事件时间戳 = 收盘时间, 实际成交由回测引擎延迟到下一根K线开盘。
        """
        positions = self.target_positions(bars).astype(int)
        code = bars.attrs.get("code", "")
        events: list[SignalEvent] = []
        prev = 0
        for ts, target in positions.items():
            if target != prev:
                if target == 1:
                    action = SignalAction.LONG
                elif target == -1:
                    action = SignalAction.SHORT
                else:
                    action = SignalAction.FLAT
                events.append(
                    SignalEvent(
                        ts=ts,
                        code=code,
                        action=action,
                        metadata={"prev_position": int(prev), "target_position": int(target)},
                    )
                )
                prev = target
        return events
