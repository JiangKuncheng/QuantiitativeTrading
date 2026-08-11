"""
模拟账户模块 (Account)
======================

职责:
1. 管理现金与持仓(支持多头, 可选空头);
2. 记录每笔成交与已实现盈亏;
3. 每日收盘按最新价估值, 生成权益曲线;
4. 为回测引擎提供绩效统计所需的原始数据。

注意:
- 本模块只"记账", 不决定买卖什么、买卖多少(那是策略与引擎的职责);
- A 股 T+1 约束: 由于本骨架采用"收盘出信号、次日开盘成交",
  当日买入的仓位最早也在下一交易日卖出, 天然满足 T+1。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Position:
    """持仓对象: 单标的单方向持仓。"""

    code: str
    side: str          # "long" 多头 / "short" 空头
    units: int         # 持仓数量(股)
    avg_cost: float    # 平均成本价
    opened_at: pd.Timestamp = field(default_factory=pd.Timestamp.now)

    @property
    def notional(self) -> float:
        """开仓名义市值。"""
        return self.units * self.avg_cost


class Account:
    """
    模拟账户: 现金 + 持仓 + 成交记录 + 权益曲线。
    """

    def __init__(
        self,
        initial_capital: float,
        commission_rate: float,
        lot_size: int = 100,
        leverage: float = 1.0,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("初始资金必须为正数")
        self.initial_capital = float(initial_capital)
        self.commission_rate = float(commission_rate)
        self.lot_size = int(lot_size)
        self.leverage = float(leverage) if leverage >= 1.0 else 1.0

        self.cash: float = self.initial_capital
        self.positions: dict[str, Position] = {}
        self.equity_curve: list[dict[str, Any]] = []  # 每个交易日收盘的估值快照
        self.trades: list[dict[str, Any]] = []        # 全部成交记录(含未实现开仓)

    # ------------------------------------------------------------------
    # 交易操作
    # ------------------------------------------------------------------
    def open_long(self, code: str, units: int, price: float, ts: pd.Timestamp) -> dict:
        """开多: 扣现金, 建仓/加仓。"""
        self._validate(units, price)
        commission = units * price * self.commission_rate
        total_cost = units * price + commission
        if total_cost > self.cash * self.leverage + 1e-9:
            raise ValueError(
                f"资金/杠杆不足: 需 {total_cost:.2f}, "
                f"可用 {self.cash * self.leverage:.2f} (现金 {self.cash:.2f} x 杠杆 {self.leverage})"
            )
        self.cash -= total_cost

        pos = self.positions.get(code)
        if pos is None:
            self.positions[code] = Position(code, "long", units, price, ts)
        else:
            # 已有同向持仓: 加权摊平成本(本骨架为全进全出, 此分支供加仓策略扩展)
            new_units = pos.units + units
            pos.avg_cost = (pos.avg_cost * pos.units + price * units) / new_units
            pos.units = new_units

        return self._record(code, ts, "BUY", units, price, commission, pnl=None, reason="open_long")

    def close_long(self, code: str, units: int, price: float, ts: pd.Timestamp) -> dict:
        """平多: 回笼现金, 结算已实现盈亏。"""
        pos = self.positions.get(code)
        if pos is None or pos.side != "long":
            raise ValueError(f"{code} 无多头持仓可平")
        units = min(units, pos.units)
        proceeds = units * price
        commission = proceeds * self.commission_rate
        self.cash += proceeds - commission
        pnl = (price - pos.avg_cost) * units - commission

        pos.units -= units
        if pos.units <= 0:
            del self.positions[code]
        return self._record(code, ts, "SELL", units, price, commission, pnl=pnl, reason="close_long")

    def open_short(self, code: str, units: int, price: float, ts: pd.Timestamp) -> dict:
        """开空(可选): 卖出所得计入现金, 负债以市价跟踪。"""
        self._validate(units, price)
        proceeds = units * price
        commission = proceeds * self.commission_rate
        self.cash += proceeds - commission

        pos = self.positions.get(code)
        if pos is None:
            self.positions[code] = Position(code, "short", units, price, ts)
        else:
            new_units = pos.units + units
            pos.avg_cost = (pos.avg_cost * pos.units + price * units) / new_units
            pos.units = new_units
        return self._record(code, ts, "SELL_SHORT", units, price, commission, pnl=None, reason="open_short")

    def close_short(self, code: str, units: int, price: float, ts: pd.Timestamp) -> dict:
        """平空(买回): 扣现金, 结算已实现盈亏。"""
        pos = self.positions.get(code)
        if pos is None or pos.side != "short":
            raise ValueError(f"{code} 无空头持仓可平")
        units = min(units, pos.units)
        cost = units * price
        commission = cost * self.commission_rate
        self.cash -= cost + commission
        pnl = (pos.avg_cost - price) * units - commission

        pos.units -= units
        if pos.units <= 0:
            del self.positions[code]
        return self._record(code, ts, "COVER", units, price, commission, pnl=pnl, reason="close_short")

    # ------------------------------------------------------------------
    # 估值与查询
    # ------------------------------------------------------------------
    def live_equity(self, prices: dict[str, float]) -> float:
        """按给定价格即时估值(用于下单前计算可用资金)。"""
        value = self.cash
        for code, pos in self.positions.items():
            price = prices.get(code)
            if price is None or price <= 0:
                continue
            if pos.side == "long":
                value += pos.units * price
            else:
                # 开空所得已在 cash 中, 此处减去当前回购市值即可
                value -= pos.units * price
        return value

    def mark_to_market(self, ts: pd.Timestamp, prices: dict[str, float]) -> None:
        """收盘估值: 追加一条权益曲线记录。"""
        value = self.live_equity(prices)
        prev = self.equity_curve[-1]["equity"] if self.equity_curve else self.initial_capital
        daily_return = (value / prev - 1.0) if prev > 0 else 0.0
        self.equity_curve.append(
            {
                "datetime": ts,
                "equity": value,
                "cash": self.cash,
                "position_value": value - self.cash,
                "daily_return": daily_return,
            }
        )

    def current_equity(self) -> float:
        """最近一次收盘估值(无估值时返回初始资金)。"""
        if self.equity_curve:
            return self.equity_curve[-1]["equity"]
        return self.initial_capital

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _validate(units: int, price: float) -> None:
        if units <= 0 or price <= 0:
            raise ValueError(f"非法交易参数: units={units}, price={price}")

    def _record(
        self,
        code: str,
        ts: pd.Timestamp,
        side: str,
        units: int,
        price: float,
        commission: float,
        pnl: float | None,
        reason: str,
    ) -> dict:
        """记录一笔成交, 返回记录 dict 便于上层转为 FillEvent。"""
        record = {
            "datetime": ts,
            "code": code,
            "side": side,
            "units": units,
            "price": round(price, 4),
            "commission": round(commission, 4),
            "pnl": None if pnl is None else round(pnl, 4),
            "reason": reason,
            "cash_after": round(self.cash, 4),
        }
        self.trades.append(record)
        return record
