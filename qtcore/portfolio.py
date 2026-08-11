"""
PortfolioAccount 组合账户
=========================
管理现金与多标持仓, 支持买卖撮合(佣金/滑点)、收盘估值、状态序列化。
与回测引擎的 Account 不同: 这里面向"从今天开始的持续运行",
每个交易日增量更新, 状态可持久化到 SQLite。
"""

from __future__ import annotations

from typing import Any


class PortfolioAccount:
    """多标组合账户。"""

    def __init__(
        self,
        initial_capital: float,
        commission_rate: float = 0.0003,
        slippage_rate: float = 0.0002,
        lot_size: int = 100,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.commission_rate = float(commission_rate)
        self.slippage_rate = float(slippage_rate)
        self.lot_size = int(lot_size)
        self.cash = float(initial_capital)
        self.positions: dict[str, dict[str, float]] = {}  # code -> {units, avg_cost}

    # ------------------------------------------------------------------
    # 价格
    # ------------------------------------------------------------------
    def buy_price(self, price: float) -> float:
        return price * (1.0 + self.slippage_rate)

    def sell_price(self, price: float) -> float:
        return price * (1.0 - self.slippage_rate)

    # ------------------------------------------------------------------
    # 交易
    # ------------------------------------------------------------------
    def buy(self, code: str, units: int, price: float) -> dict[str, Any] | None:
        """买入: 资金不足时自动缩量到整手; 完全买不起返回 None。"""
        px = self.buy_price(float(price))
        units = int(units)
        if units <= 0:
            return None
        cost = units * px
        commission = cost * self.commission_rate
        total = cost + commission
        if total > self.cash + 1e-9:
            units = int(self.cash / (px * self.lot_size * (1 + self.commission_rate))) * self.lot_size
            if units <= 0:
                return None
            cost = units * px
            commission = cost * self.commission_rate
            total = cost + commission
        self.cash -= total
        pos = self.positions.get(code)
        if pos:
            new_units = pos["units"] + units
            pos["avg_cost"] = (pos["avg_cost"] * pos["units"] + px * units) / new_units
            pos["units"] = float(new_units)
        else:
            self.positions[code] = {"units": float(units), "avg_cost": px}
        return {
            "code": code,
            "side": "BUY",
            "units": units,
            "price": round(px, 4),
            "commission": round(commission, 4),
            "pnl": None,
            "cash_after": round(self.cash, 4),
        }

    def sell(self, code: str, price: float) -> dict[str, Any] | None:
        """全部卖出并结算已实现盈亏。"""
        pos = self.positions.get(code)
        if not pos:
            return None
        px = self.sell_price(float(price))
        units = int(pos["units"])
        proceeds = units * px
        commission = proceeds * self.commission_rate
        self.cash += proceeds - commission
        pnl = (px - pos["avg_cost"]) * units - commission
        record = {
            "code": code,
            "side": "SELL",
            "units": units,
            "price": round(px, 4),
            "commission": round(commission, 4),
            "pnl": round(pnl, 4),
            "cash_after": round(self.cash, 4),
        }
        del self.positions[code]
        return record

    # ------------------------------------------------------------------
    # 估值与状态
    # ------------------------------------------------------------------
    def equity(self, close_prices: dict[str, float]) -> float:
        value = self.cash
        for code, pos in self.positions.items():
            px = close_prices.get(code)
            value += pos["units"] * (px if px is not None else pos["avg_cost"])
        return value

    def state_dict(self) -> dict[str, Any]:
        return {"cash": round(self.cash, 4), "positions": self.positions}

    def load_state(self, cash: float, positions: dict[str, dict[str, float]]) -> None:
        self.cash = float(cash)
        self.positions = {k: dict(v) for k, v in positions.items()}
