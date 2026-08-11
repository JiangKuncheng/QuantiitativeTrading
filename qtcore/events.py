"""
事件驱动核心
============

系统内部通过事件对象解耦模块:

    BarEvent      DataCenter 发布, 携带单根K线行情
    SignalEvent   StrategyEngine 发布, 携带买卖信号(不涉及资金)
    OrderEvent    BacktestEngine 生成, 表示一笔拟下单
    FillEvent     BacktestEngine 撮合后产生, 表示一笔成交

事件流:
    BarEvent -> SignalEvent -> OrderEvent -> FillEvent -> Account 记账

这样设计的好处:
1. 策略与资金/撮合完全解耦, 同一策略可无缝用于回测与实盘;
2. 后续微服务化时, 事件对象可直接序列化为消息队列(如 Kafka/RabbitMQ)中的 payload。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

import pandas as pd


class EventType(Enum):
    """事件类型枚举。"""

    BAR = auto()       # K线行情事件
    SIGNAL = auto()    # 策略信号事件
    ORDER = auto()     # 订单事件
    FILL = auto()      # 成交事件


class SignalAction(Enum):
    """策略信号动作: 只描述"目标状态", 不关心资金与下单细节。"""

    LONG = "LONG"    # 目标: 持有多头
    SHORT = "SHORT"  # 目标: 持有空头(需开启 allow_short)
    FLAT = "FLAT"    # 目标: 空仓


@dataclass(kw_only=True)
class BaseEvent:
    """所有事件的基类。"""

    event_type: EventType
    ts: datetime
    code: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict, 便于落盘/日志/消息队列传输。"""
        return {
            "event_type": self.event_type.name,
            "ts": self.ts.isoformat(),
            "code": self.code,
        }


@dataclass(kw_only=True)
class BarEvent(BaseEvent):
    """单根K线行情事件。"""

    event_type: EventType = EventType.BAR
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    amount: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = EventType.BAR

    @classmethod
    def from_row(cls, code: str, ts: Any, row: pd.Series) -> "BarEvent":
        """从统一行情 DataFrame 的一行构造 BarEvent。"""
        return cls(
            event_type=EventType.BAR,
            ts=ts,
            code=code,
            open=float(row.get("open", 0.0)),
            high=float(row.get("high", 0.0)),
            low=float(row.get("low", 0.0)),
            close=float(row.get("close", 0.0)),
            volume=float(row.get("volume", 0.0) or 0.0),
            amount=float(row.get("amount", 0.0) or 0.0),
        )


@dataclass(kw_only=True)
class SignalEvent(BaseEvent):
    """策略信号事件: 只表达目标仓位状态, 不直接下单。"""

    event_type: EventType = EventType.SIGNAL
    action: SignalAction = SignalAction.FLAT
    strength: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_type = EventType.SIGNAL


@dataclass(kw_only=True)
class OrderEvent(BaseEvent):
    """订单事件: 回测引擎根据信号生成。"""

    event_type: EventType = EventType.ORDER
    side: str = "BUY"        # BUY / SELL / SELL_SHORT / COVER
    units: int = 0

    def __post_init__(self) -> None:
        self.event_type = EventType.ORDER


@dataclass(kw_only=True)
class FillEvent(BaseEvent):
    """成交事件: 撮合后产生, 由 Account 记账。"""

    event_type: EventType = EventType.FILL
    side: str = "BUY"
    units: int = 0
    price: float = 0.0
    commission: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = EventType.FILL
