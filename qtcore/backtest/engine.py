"""
BacktestEngine 回测/执行模块
============================

职责:
1. 读取统一行情, 调用策略生成目标仓位;
2. 按"收盘出信号 -> 次日开盘成交"的规则模拟撮合(规避未来函数);
3. 处理滑点与佣金, 交由 Account 记账;
4. 计算总收益、年化收益、夏普、最大回撤、胜率、盈亏比等绩效指标;
5. 输出权益曲线、成交明细与统计 JSON。

撮合规则(与 A 股交易习惯对齐):
- 信号在 T 日收盘产生, 在 T+1 日开盘价成交;
- 买入价 = open * (1 + slippage), 卖出价 = open * (1 - slippage);
- 单次建仓金额 = 最近权益 * position_ratio, 向下取整到整手(lot_size);
- 默认仅做多; 开启 allow_short 后支持多空双向;
- T+1: 当日买入最早次日卖出(本骨架天然满足)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qtcore.backtest.account import Account
from qtcore.config import BacktestConfig
from qtcore.events import SignalEvent
from qtcore.strategy.base import StrategyBase


@dataclass
class BacktestResult:
    """回测结果: 权益曲线 + 成交明细 + 绩效统计 + 信号事件流。"""

    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    stats: dict[str, float]
    signal_events: list[SignalEvent] = field(default_factory=list)

    def save(self, output_dir: Path) -> dict[str, Path]:
        """将结果落盘到指定目录, 返回文件路径映射。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        equity_path = output_dir / "equity_curve.csv"
        trades_path = output_dir / "trades.csv"
        stats_path = output_dir / "stats.json"

        self.equity_curve.to_csv(equity_path, encoding="utf-8-sig")
        self.trades.to_csv(trades_path, encoding="utf-8-sig", index=False)
        with stats_path.open("w", encoding="utf-8") as fp:
            json.dump(self.stats, fp, ensure_ascii=False, indent=2)
        return {"equity_curve": equity_path, "trades": trades_path, "stats": stats_path}


class BacktestEngine:
    """轻量级事件驱动回测引擎。"""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self, bars: pd.DataFrame, strategy: StrategyBase) -> BacktestResult:
        """
        执行回测。

        参数:
            bars    : UnifiedBar 标准行情 DataFrame
            strategy: 已实例化的策略对象
        """
        if bars is None or bars.empty:
            raise ValueError("输入行情为空, 无法回测")

        code = bars.attrs.get("code", "DEMO")
        account = Account(
            initial_capital=self.config.initial_capital,
            commission_rate=self.config.commission_rate,
            lot_size=self.config.lot_size,
            leverage=self.config.leverage,
        )

        # 策略输出: 目标仓位序列 + 信号事件流
        targets = strategy.target_positions(bars).astype(int)
        signal_events = strategy.generate_signal_events(bars)

        # 逐日事件循环
        pending_target: int | None = None  # 上一根K线收盘产生的目标仓位
        halted = False                     # 回撤熔断: 触发后停止开新仓并清仓
        halt_since_i: int | None = None    # 熔断触发时的K线序号(用于冷却期计算)
        prev_close: float | None = None    # 上一根K线收盘价(限价单基准)
        for i, ts in enumerate(bars.index):
            bar = bars.iloc[i]
            open_price = float(bar["open"])
            close_price = float(bar["close"])
            high_price = float(bar["high"])
            low_price = float(bar["low"])

            # 0) 止损/止盈检查(先于新信号, 用当日最高/最低近似触发)
            self._check_stops(account, code, ts, high_price, low_price)

            # 0.5) 回撤熔断: 权益回撤超限后清仓并停止开新仓
            if halted:
                self._close_all(account, code, open_price, ts, reason="halt")
                # 冷却期结束且回撤恢复到阈值以内 -> 解除熔断, 允许重新建仓
                cooldown = self.config.halt_cooldown_days
                if halt_since_i is not None and (i - halt_since_i) >= cooldown:
                    eq = pd.Series([r["equity"] for r in account.equity_curve])
                    dd = float(eq.iloc[-1] / eq.cummax().iloc[-1] - 1.0)
                    resume_limit = self.config.halt_resume_drawdown
                    if resume_limit <= 0 or dd > -abs(resume_limit):
                        halted = False
                        halt_since_i = None
                        print(
                            f"[Backtest] {ts.date()} 熔断恢复"
                            f"(冷却 {cooldown} 日, 当前回撤 {dd:.2%}), 允许重新建仓"
                        )

            # 1) 撮合昨日收盘信号: 以今日开盘价成交(延迟 1 根K线)
            if pending_target is not None and self._is_rebalance_day(ts, i, bars.index):
                if not halted:
                    self._execute_target(account, code, pending_target, open_price, ts, prev_close)
                pending_target = None

            # 2) 今日收盘更新目标仓位(次日开盘执行)
            pending_target = int(targets.iloc[i])

            # 3) 收盘估值, 记录权益曲线
            account.mark_to_market(ts, {code: close_price})

            # 4) 回撤熔断检测
            if not halted and self.config.max_drawdown_halt > 0:
                eq = pd.Series([r["equity"] for r in account.equity_curve])
                dd = float(eq.iloc[-1] / eq.cummax().iloc[-1] - 1.0)
                if dd <= -abs(self.config.max_drawdown_halt):
                    halted = True
                    halt_since_i = i
                    print(
                        f"[Backtest] {ts.date()} 触发回撤熔断: {dd:.2%}, "
                        f"清仓并暂停开仓(冷却 {self.config.halt_cooldown_days} 日)"
                    )

            prev_close = close_price

        equity_curve = pd.DataFrame(account.equity_curve).set_index("datetime")
        trades = pd.DataFrame(account.trades)
        stats = self.compute_stats(equity_curve, trades)

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            stats=stats,
            signal_events=signal_events,
        )

    # ------------------------------------------------------------------
    # 撮合逻辑
    # ------------------------------------------------------------------
    def _execute_target(
        self,
        account: Account,
        code: str,
        target: int,
        price: float,
        ts: pd.Timestamp,
        prev_close: float | None = None,
    ) -> None:
        """
        将当前持仓调整到目标仓位。
        target: 1 持多 / 0 空仓 / -1 持空(需允许做空)
        """
        pos = account.positions.get(code)
        current = 0
        if pos is not None:
            current = 1 if pos.side == "long" else -1

        if target == current:
            return  # 无需调仓

        # 订单类型: limit 限价单 -> 次日开盘不满足限价条件则放弃本次交易
        if self.config.order_type == "limit" and prev_close is not None and prev_close > 0:
            if target == 1 and price > prev_close * 1.001:
                return  # 买入限价未触及(高开跳过)
            if target == -1 and price < prev_close * 0.999:
                return

        # 滑点容忍度: 实际滑点超过设定容忍则放弃交易
        if self.config.slippage_tolerance_pct > 0:
            if self.config.slippage_rate > self.config.slippage_tolerance_pct:
                return

        if target == 0:
            self._close_all(account, code, price, ts)
            return

        if target == 1:
            if pos is not None and pos.side == "short":  # 空头 -> 多头
                self._close_all(account, code, price, ts)
            units = self._sizing(account, price)
            if units > 0:
                account.open_long(code, units, self._exec_price(price, is_buy=True), ts)
            return

        # target == -1
        if not self.config.allow_short:
            self._close_all(account, code, price, ts)  # 不允许做空时视为清仓
            return
        if pos is not None and pos.side == "long":  # 多头 -> 空头
            self._close_all(account, code, price, ts)
        units = self._sizing(account, price)
        if units > 0:
            account.open_short(code, units, self._exec_price(price, is_buy=True), ts)

    def _check_stops(
        self,
        account: Account,
        code: str,
        ts: pd.Timestamp,
        high: float,
        low: float,
    ) -> None:
        """止损/止盈: 用当日 high/low 与持仓成本价比较, 触发即按触发价平仓。"""
        pos = account.positions.get(code)
        if pos is None:
            return
        sl = self.config.stop_loss_pct
        tp = self.config.take_profit_pct
        if sl <= 0 and tp <= 0:
            return
        if pos.side == "long":
            stop_price = pos.avg_cost * (1 - sl) if sl > 0 else None
            take_price = pos.avg_cost * (1 + tp) if tp > 0 else None
            if take_price is not None and high >= take_price:
                account.close_long(code, pos.units, take_price, ts)
                return
            if stop_price is not None and low <= stop_price:
                account.close_long(code, pos.units, stop_price, ts)
        elif pos.side == "short":
            stop_price = pos.avg_cost * (1 + sl) if sl > 0 else None
            take_price = pos.avg_cost * (1 - tp) if tp > 0 else None
            if take_price is not None and low <= take_price:
                account.close_short(code, pos.units, take_price, ts)
                return
            if stop_price is not None and high >= stop_price:
                account.close_short(code, pos.units, stop_price, ts)

    def _is_rebalance_day(self, ts: pd.Timestamp, i: int, index) -> bool:
        """调仓周期: daily 每天 / weekly 每周首个交易日 / monthly 每月首个交易日。"""
        freq = self.config.rebalance
        if freq == "weekly":
            return i == 0 or ts.isocalendar().week != index[i - 1].isocalendar().week
        if freq == "monthly":
            return i == 0 or ts.month != index[i - 1].month
        return True  # daily

    def _close_all(
        self, account: Account, code: str, price: float, ts: pd.Timestamp, reason: str = "signal"
    ) -> None:
        """清仓该标的所有持仓。"""
        pos = account.positions.get(code)
        if pos is None:
            return
        sell_price = self._exec_price(price, is_buy=False)
        if pos.side == "long":
            account.close_long(code, pos.units, sell_price, ts)
        else:
            account.close_short(code, pos.units, sell_price, ts)

    def _sizing(self, account: Account, price: float) -> int:
        """
        仓位计算: 预算 = 即时权益 * position_ratio * leverage,
        再受单标的持仓上限 max_position_ratio 约束, 向下取整到整手。
        以即时估值(现金 + 持仓市值)为基准, 平仓后再开仓也能正确扩仓。
        """
        equity = account.live_equity({pos.code: price for pos in account.positions.values()})
        budget = equity * self.config.position_ratio * self.config.leverage
        lot_cost = price * self.config.lot_size * (1 + self.config.commission_rate)
        if lot_cost <= 0:
            return 0
        units = int(budget / lot_cost) * self.config.lot_size
        cap_units = (
            int(equity * self.config.max_position_ratio / (price * self.config.lot_size))
            * self.config.lot_size
        )
        return min(units, cap_units)

    def _exec_price(self, price: float, is_buy: bool) -> float:
        """按滑点调整成交价: 买入更贵, 卖出更便宜。"""
        slip = self.config.slippage_rate
        return price * (1 + slip) if is_buy else price * (1 - slip)

    # ------------------------------------------------------------------
    # 绩效统计
    # ------------------------------------------------------------------
    def compute_stats(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
    ) -> dict[str, float]:
        """由权益曲线与成交明细计算绩效指标。"""
        if equity_curve.empty:
            return {}

        equity = equity_curve["equity"]
        final_equity = float(equity.iloc[-1])
        n_days = len(equity)
        total_return = final_equity / self.config.initial_capital - 1.0
        annual_return = (
            (1.0 + total_return) ** (self.config.annual_trading_days / n_days) - 1.0
            if n_days > 0 and total_return > -1.0
            else -1.0
        )

        # 夏普比率(日频收益年化)
        daily_returns = equity.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0:
            sharpe = float(
                daily_returns.mean() / daily_returns.std(ddof=1)
                * np.sqrt(self.config.annual_trading_days)
            )
        else:
            sharpe = 0.0

        # 最大回撤(负数表示)
        drawdown = equity / equity.cummax() - 1.0
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        # 胜率 / 盈亏比(仅统计已平仓交易, 即 pnl 非空)
        closed = trades[trades["pnl"].notna()] if not trades.empty else trades
        n_trades = int(len(closed))
        wins = int((closed["pnl"] > 0).sum()) if n_trades else 0
        win_rate = wins / n_trades if n_trades else 0.0

        gross_profit = float(closed.loc[closed["pnl"] > 0, "pnl"].sum()) if n_trades else 0.0
        gross_loss = float(closed.loc[closed["pnl"] < 0, "pnl"].sum()) if n_trades else 0.0
        if gross_loss < 0:
            profit_factor = gross_profit / abs(gross_loss)
        else:
            profit_factor = float("inf") if gross_profit > 0 else 0.0

        return {
            "initial_capital": self.config.initial_capital,
            "final_equity": round(final_equity, 2),
            "total_return": round(total_return, 6),
            "annual_return": round(annual_return, 6),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_drawdown, 6),
            "n_trades": n_trades,
            "win_rate": round(win_rate, 6),
            "profit_factor": round(profit_factor, 4) if np.isfinite(profit_factor) else profit_factor,
            "avg_pnl": round(float(closed["pnl"].mean()), 4) if n_trades else 0.0,
            # 防御: 空成交表(整个区间零交易)时无 commission 列
            "total_commission": round(float(trades["commission"].sum()), 2)
            if not trades.empty and "commission" in trades.columns
            else 0.0,
        }
