"""
DailyTrader 每日自动交易与报告
==============================

流程(每个交易日收盘后运行):
    1. 交易日判断(交易日历, 失败回退工作日判断; 周末/节假日不交易不发日报)
    2. 加载股票池(配置指定或沪深300前 N, 本地缓存)
    3. 逐只拉取历史数据(多数据源+统一重试) -> 按策略回测到"今天"
    4. 汇总: 今日交易、今日组合收益、当前持仓、与沪深300基准对比
    5. 写入 SQLite(data/trading.db): 每日权益/交易/信号/运行日志/报告
    6. DeepSeek 写日报 -> 发邮件
    7. 每周最后一个交易日发周报; 每月最后一个交易日发月报+复盘

任何一步异常 -> DeepSeek 写突发说明 -> 立即发邮件 -> 写日志 -> 非零退出。
"""

from __future__ import annotations

import json
import traceback
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qtcore.backtest.engine import BacktestEngine
from qtcore.config import AppConfig
from qtcore.datacenter.data_center import DataCenter
from qtcore.emailer import load_email_config, send_email
from qtcore.report_writer import (
    write_daily_report,
    write_incident_report,
    write_monthly_report,
    write_weekly_report,
)
from qtcore.store import Store
from qtcore.strategy import create_strategy


ROOT = Path(__file__).resolve().parent.parent


class DailyTrader:
    """每日自动交易 + 报告 + 告警编排。"""

    def __init__(
        self,
        trading_config: dict[str, Any] | None = None,
        db_path: Path | str = "data/trading.db",
    ) -> None:
        cfg_path = ROOT / "config" / "trading_config.json"
        self.cfg = trading_config or json.loads(cfg_path.read_text(encoding="utf-8"))
        self.store = Store(db_path)
        self.email_cfg = load_email_config()
        self.app = AppConfig()
        self.app.data.fetch_retries = 3
        self.dc = DataCenter(self.app.data, self.app.paths)
        self._names = self._load_names()
        self._calendar = self._load_trade_calendar()

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------
    def _load_names(self) -> dict[str, str]:
        names: dict[str, str] = {}
        p = ROOT / "data" / "cache" / "universe_all.csv"
        if p.exists():
            try:
                df = pd.read_csv(p, dtype=str)
                names = dict(zip(df["code"].astype(str).str.zfill(6), df["name"]))
            except Exception:
                pass
        return names

    def _load_trade_calendar(self) -> set[str]:
        path = ROOT / "data" / "trade_calendar.csv"
        if path.exists():
            try:
                df = pd.read_csv(path, dtype=str)
                return set(df["trade_date"].astype(str).str.replace("-", ""))
            except Exception:
                pass
        try:
            import akshare as ak

            df = ak.tool_trade_date_hist_sina()
            df.to_csv(path, index=False, encoding="utf-8")
            return set(pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d"))
        except Exception:
            return set()

    def is_trading_day(self, d: date) -> bool:
        d_str = d.strftime("%Y%m%d")
        if self._calendar:
            return d_str in self._calendar
        return d.weekday() < 5  # 无日历时回退: 周一~周五

    def next_trading_day(self, d: date) -> date | None:
        if not self._calendar:
            nxt = d + timedelta(days=1)
            while nxt.weekday() >= 5:
                nxt += timedelta(days=1)
            return nxt
        dates = sorted(self._calendar)
        d_str = d.strftime("%Y%m%d")
        for x in dates:
            if x > d_str:
                return datetime.strptime(x, "%Y%m%d").date()
        return None

    def _load_pool(self) -> list[str]:
        pool_cfg = self.cfg.get("pool", {})
        symbols = pool_cfg.get("symbols")
        if symbols:
            return [str(s) for s in symbols]
        cache_path = ROOT / "data" / "pool_csi300.csv"
        if cache_path.exists():
            df = pd.read_csv(cache_path, dtype=str)
            return [str(c) for c in df["code"]]
        from qtcore.screener import StockScreener

        screener = StockScreener(self.app, synthetic=False, universe="csi300", use_snapshot=False)
        cands = screener.filter_universe()
        codes = [str(c) for c in cands.head(int(pool_cfg.get("size", 15)))["code"]]
        pd.DataFrame({"code": codes}).to_csv(cache_path, index=False, encoding="utf-8")
        return codes

    def _select_pool(
        self,
        pool_all: list[str],
        params: dict[str, Any],
        timeframe: str,
        today: date,
    ) -> list[str]:
        """
        按训练逻辑选股: 用近一年数据逐只跑策略, 按 select_metric 取 Top-K。
        每月重新选一次(月度缓存), 期间保持持仓不变。
        """
        top_k = int(params.get("top_k", 5))
        metric = str(params.get("select_metric", "sharpe"))
        month = today.strftime("%Y-%m")
        cache_path = ROOT / "data" / f"selected_pool_{timeframe}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("month") == month and cached.get("timeframe") == timeframe:
                    print(f"[Daily] 复用本月选股: {cached['symbols']}")
                    return [str(s) for s in cached["symbols"]]
            except Exception:
                pass

        sel_start = (today - timedelta(days=400)).strftime("%Y%m%d")
        sel_end = today.strftime("%Y%m%d")
        rows: list[dict[str, Any]] = []
        for code in pool_all:
            try:
                bars = self.dc.get_bars(code, sel_start, sel_end, timeframe)
                if bars is None or len(bars) < 60:
                    continue
                strategy = create_strategy("ma_cross", params)
                result = BacktestEngine(self._bt_config(params)).run(bars, strategy)
                rows.append({"code": code, "sharpe": float(result.stats.get("sharpe", 0.0))})
            except Exception as exc:
                print(f"[Daily] 选股评分失败 {code}: {exc!r}")

        if not rows:
            print("[Daily] 选股评分全部失败, 退化为取前 top_k")
            return pool_all[:top_k]
        scores = pd.DataFrame(rows)
        selected = scores.nlargest(top_k, metric)["code"].tolist()
        cache_path.write_text(
            json.dumps(
                {"month": month, "timeframe": timeframe, "symbols": selected},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[Daily] 本月选股完成: Top{top_k} {selected}")
        return selected

    def _bt_config(self, params: dict[str, Any]):
        bt = replace(self.app.backtest)
        bt.initial_capital = float(self.cfg.get("initial_capital", 1_000_000))
        bt.position_ratio = float(params.get("position_ratio", bt.position_ratio))
        bt.max_position_ratio = float(params.get("max_position_ratio", bt.max_position_ratio))
        bt.rebalance = str(params.get("rebalance", bt.rebalance))
        bt.order_type = str(params.get("order_type", bt.order_type))
        bt.slippage_tolerance_pct = float(params.get("slippage_tolerance_pct", bt.slippage_tolerance_pct))
        bt.leverage = float(params.get("leverage", bt.leverage))
        bt.stop_loss_pct = float(params.get("stop_loss_pct", bt.stop_loss_pct))
        bt.take_profit_pct = float(params.get("take_profit_pct", bt.take_profit_pct))
        bt.max_drawdown_halt = float(params.get("max_drawdown_halt", bt.max_drawdown_halt))
        bt.halt_cooldown_days = int(params.get("halt_cooldown_days", bt.halt_cooldown_days))
        bt.halt_resume_drawdown = float(params.get("halt_resume_drawdown", bt.halt_resume_drawdown))
        return bt

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self, run_date: str | None = None, force_weekly: bool = False, force_monthly: bool = False) -> dict[str, Any]:
        today = datetime.strptime(run_date, "%Y%m%d").date() if run_date else date.today()
        d_str = today.strftime("%Y%m%d")
        d_iso = today.strftime("%Y-%m-%d")
        try:
            return self._run_inner(today, d_str, d_iso, force_weekly, force_monthly)
        except Exception as exc:
            self._handle_incident(today, "daily_run", exc)
            raise

    def _run_inner(self, today: date, d_str: str, d_iso: str, force_weekly: bool, force_monthly: bool) -> dict[str, Any]:
        if not self.is_trading_day(today):
            self.store.log_run(d_iso, "calendar", "skip", "非交易日")
            print(f"[Daily] {d_iso} 非交易日, 跳过")
            return {"trading_day": False, "date": d_iso}

        print(f"[Daily] {d_iso} 交易日, 开始运行")
        params = dict(self.cfg["strategy"]["params"])
        timeframe = str(params.get("timeframe", "daily"))
        pool = self._load_pool()
        pool = self._select_pool(pool, params, timeframe, today)
        print(f"[Daily] 持仓池 {len(pool)} 只 | 周期 {timeframe}")

        # 1) 逐只回测到今日
        returns: dict[str, pd.Series] = {}
        last_dates: dict[str, str] = {}
        today_trades: list[dict[str, Any]] = []
        holdings: dict[str, dict[str, Any]] = {}
        targets: dict[str, int] = {}
        failed: list[str] = []

        for code in pool:
            try:
                bars = self.dc.get_bars(
                    symbol=code,
                    start_date=self.cfg["data_start"],
                    end_date=d_str,
                    timeframe=timeframe,
                )
                strategy = create_strategy("ma_cross", params)
                result = BacktestEngine(self._bt_config(params)).run(bars, strategy)

                eq = result.equity_curve
                returns[code] = eq["daily_return"]
                last_dates[code] = bars.index[-1].strftime("%Y-%m-%d")
                trades = result.trades
                if not trades.empty:
                    mask = trades["datetime"].dt.strftime("%Y-%m-%d") == d_iso
                    for _, t in trades[mask].iterrows():
                        today_trades.append(
                            {
                                "date": d_iso,
                                "code": code,
                                "name": self._names.get(code, code),
                                "side": t["side"],
                                "units": int(t["units"]),
                                "price": float(t["price"]),
                                "commission": float(t["commission"]),
                                "pnl": None if pd.isna(t["pnl"]) else float(t["pnl"]),
                                "reason": t["reason"],
                            }
                        )
                # 当前持仓(按成交累计)
                units = 0
                buy_cost = 0.0
                for _, t in trades.iterrows():
                    if t["side"] in ("BUY", "SELL_SHORT"):
                        units += int(t["units"])
                        buy_cost += int(t["units"]) * float(t["price"])
                    elif t["side"] in ("SELL", "COVER"):
                        units -= int(t["units"])
                if units > 0:
                    last_close = float(bars["close"].iloc[-1])
                    holdings[code] = {
                        "name": self._names.get(code, code),
                        "units": units,
                        "avg_cost": round(buy_cost / units, 3) if units else None,
                        "last_close": round(last_close, 3),
                        "value": round(units * last_close, 2),
                    }
                targets[code] = int(strategy.target_positions(bars).iloc[-1])
            except Exception as exc:
                failed.append(f"{code}: {type(exc).__name__} {str(exc)[:80]}")
                print(f"[Daily] {code} 回测失败: {exc!r}")

        if not returns:
            raise RuntimeError("全部标的数据获取/回测失败: " + "; ".join(failed))

        # 数据新鲜度校验: 若大量标的的最新K线不是今天, 说明当日行情未发布, 不发假日报
        stale = [c for c, d in last_dates.items() if d < d_iso]
        if len(stale) > len(pool) // 2:
            raise RuntimeError(
                f"今日行情数据未发布/延迟: 过期标的 {len(stale)}/{len(pool)}, 示例 {stale[:5]}"
            )

        # 2) 组合与基准
        ret_df = pd.DataFrame(returns).fillna(0.0)
        port_ret = ret_df.mean(axis=1)
        # 记账链: 昨日账户权益(数据库) × (1 + 今日组合收益)
        today_bt_return = float(port_ret.iloc[-1])
        prev_equity = self.store.prev_equity_before(d_iso) or float(self.cfg["initial_capital"])
        today_equity = prev_equity * (1.0 + today_bt_return)
        daily_return = today_bt_return
        strategy_total = today_equity / float(self.cfg["initial_capital"]) - 1.0

        bench_df = self.dc.get_index_daily(
            str(self.cfg.get("benchmark", "000300")), self.cfg["data_start"], d_str
        )
        if bench_df.index[-1].strftime("%Y-%m-%d") < d_iso:
            raise RuntimeError(
                f"基准指数数据未更新到今日: 最新 {bench_df.index[-1].strftime('%Y-%m-%d')}"
            )
        bench_close = bench_df["close"].reindex(port_ret.index).ffill()
        bench_ret = bench_close.pct_change()
        benchmark_return = float(bench_ret.iloc[-1]) if not np.isnan(bench_ret.iloc[-1]) else 0.0
        benchmark_total = float(bench_close.iloc[-1] / bench_close.iloc[0] - 1.0) if len(bench_close) > 1 else 0.0

        # 建仓日: 账户从今天开始, 按今日收盘信号建仓, 当日盈亏记 0
        first_day = d_str == str(self.cfg.get("account_start", d_str))
        if first_day:
            # 建仓日: 等权建仓(预算 = 本金 × 仓位比例 ÷ 建仓只数), 当日盈亏记 0
            build_trades: list[dict[str, Any]] = []
            build_codes = [code for code, h in holdings.items() if targets.get(code) == 1]
            bt_cfg = self._bt_config(params)
            if build_codes:
                budget_each = float(self.cfg["initial_capital"]) * bt_cfg.position_ratio / len(build_codes)
                for code in build_codes:
                    h = holdings[code]
                    price = h["last_close"]
                    lot = self.app.backtest.lot_size
                    units = int(budget_each / (price * lot)) * lot
                    if units <= 0:
                        continue
                    build_trades.append(
                        {
                            "date": d_iso,
                            "code": code,
                            "name": h["name"],
                            "side": "BUY",
                            "units": units,
                            "price": round(price, 3),
                            "commission": round(units * price * bt_cfg.commission_rate, 4),
                            "pnl": None,
                            "reason": "initial_build",
                        }
                    )
            today_trades = build_trades
            today_equity = float(self.cfg["initial_capital"])
            prev_equity = today_equity
            daily_return = 0.0
            strategy_total = 0.0

        # 3) 落库
        self.store.save_equity(
            {
                "date": d_iso,
                "equity": round(today_equity, 2),
                "cash": None,
                "position_value": None,
                "daily_return": round(daily_return, 6),
                "benchmark_return": round(benchmark_return, 6),
                "strategy_total": round(strategy_total, 6),
                "benchmark_total": round(benchmark_total, 6),
            }
        )
        self.store.save_trades(today_trades)
        self.store.save_signals(d_iso, targets, params)
        self.store.log_run(d_iso, "daily", "ok", f"equity={today_equity:.2f}")

        # 4) 日报
        daily_data = {
            "date": d_iso,
            "timeframe": timeframe,
            "account_start": str(self.cfg.get("account_start", d_iso)),
            "initial_capital": self.cfg["initial_capital"],
            "today_equity": round(today_equity, 2),
            "today_profit": round(today_equity - prev_equity, 2),
            "today_return": f"{daily_return:.2%}",
            "benchmark_return_today": f"{benchmark_return:.2%}",
            "cumulative_return_since_start": f"{strategy_total:.2%}",
            "benchmark_since_start": f"{benchmark_total:.2%}",
            "trades_today": today_trades,
            "holdings": holdings,
            "signals": targets,
            "data_failed": failed,
            "note": "账户从 account_start 起新建仓; 建仓日按收盘信号建仓、当日盈亏记0, 次日开始计收益",
        }
        body = write_daily_report(daily_data)
        subject = f"[QuantTrader] 日报 {d_iso}"
        send_email(subject, body, config=self.email_cfg)
        self.store.save_report(d_iso, "daily", subject, body)
        print(f"[Daily] 日报已发送: {subject}")

        # 5) 周报 / 月报
        if force_weekly or self._is_last_trading_day_of_week(today):
            self._send_weekly_report(today, d_iso)
        if force_monthly or self._is_last_trading_day_of_month(today):
            self._send_monthly_report(today, d_iso)

        return {
            "trading_day": True,
            "date": d_iso,
            "equity": today_equity,
            "daily_return": daily_return,
            "benchmark_return": benchmark_return,
            "trades": len(today_trades),
            "failed_symbols": failed,
        }

    # ------------------------------------------------------------------
    # 周报 / 月报
    # ------------------------------------------------------------------
    def _is_last_trading_day_of_week(self, d: date) -> bool:
        nxt = self.next_trading_day(d)
        if nxt is None:
            return d.weekday() == 4
        return nxt.isocalendar().week != d.isocalendar().week

    def _is_last_trading_day_of_month(self, d: date) -> bool:
        nxt = self.next_trading_day(d)
        if nxt is None:
            return d.month != (d + timedelta(days=1)).month or d.day >= 28
        return nxt.month != d.month

    def _week_month_boundary(self, d: date) -> str:
        return (d - timedelta(days=d.weekday())).strftime("%Y-%m-%d")

    def _send_weekly_report(self, today: date, d_iso: str) -> None:
        start = self._week_month_boundary(today)
        eq = self.store.equity_between(start, d_iso)
        trades = self.store.trades_between(start, d_iso)
        week_return = (eq[-1]["equity"] / eq[0]["equity"] - 1.0) if len(eq) >= 2 else 0.0
        bench_week = 1.0
        for r in eq:
            if r.get("benchmark_return") is not None:
                bench_week *= 1.0 + float(r["benchmark_return"])
        data = {
            "week": f"{start} ~ {d_iso}",
            "week_return": f"{week_return:.2%}",
            "benchmark_week_return": f"{bench_week - 1.0:.2%}",
            "start_equity": eq[0]["equity"] if eq else None,
            "end_equity": eq[-1]["equity"] if eq else None,
            "trades": trades,
            "n_trades": len(trades),
        }
        body = write_weekly_report(data)
        subject = f"[QuantTrader] 周报 {d_iso}"
        send_email(subject, body, config=self.email_cfg)
        self.store.save_report(d_iso, "weekly", subject, body)
        print(f"[Daily] 周报已发送: {subject}")

    def _send_monthly_report(self, today: date, d_iso: str) -> None:
        start = today.strftime("%Y-%m-01")
        eq = self.store.equity_between(start, d_iso)
        trades = self.store.trades_between(start, d_iso)
        month_return = (eq[-1]["equity"] / eq[0]["equity"] - 1.0) if len(eq) >= 2 else 0.0
        eqs = [r["equity"] for r in eq]
        mdd = 0.0
        if eqs:
            peak = eqs[0]
            for v in eqs:
                peak = max(peak, v)
                mdd = min(mdd, v / peak - 1.0)
        closed = [t for t in trades if t.get("pnl") is not None]
        biggest_win = max(closed, key=lambda t: t["pnl"]) if closed else None
        biggest_loss = min(closed, key=lambda t: t["pnl"]) if closed else None
        bench_month = 1.0
        for r in eq:
            if r.get("benchmark_return") is not None:
                bench_month *= 1.0 + float(r["benchmark_return"])
        data = {
            "month": today.strftime("%Y-%m"),
            "month_return": f"{month_return:.2%}",
            "benchmark_month_return": f"{bench_month - 1.0:.2%}",
            "month_max_drawdown": f"{mdd:.2%}",
            "start_equity": eq[0]["equity"] if eq else None,
            "end_equity": eq[-1]["equity"] if eq else None,
            "n_trades": len(trades),
            "biggest_win": biggest_win,
            "biggest_loss": biggest_loss,
            "trades": trades,
        }
        body = write_monthly_report(data)
        subject = f"[QuantTrader] 月报+复盘 {d_iso}"
        send_email(subject, body, config=self.email_cfg)
        self.store.save_report(d_iso, "monthly", subject, body)
        print(f"[Daily] 月报已发送: {subject}")

    # ------------------------------------------------------------------
    # 突发情况
    # ------------------------------------------------------------------
    def _handle_incident(self, today: date, stage: str, exc: Exception) -> None:
        d_iso = today.strftime("%Y-%m-%d")
        context = {
            "date": d_iso,
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-8:]),
            "suggestion": "请检查行情数据源/网络/配置后重试; 若连续失败请人工介入。",
        }
        try:
            body = write_incident_report(context)
        except Exception:
            body = json.dumps(context, ensure_ascii=False, indent=2)
        try:
            subject = f"[QuantTrader] 突发告警 {d_iso}"
            send_email(subject, body, config=self.email_cfg)
            print(f"[Daily] 突发告警已发送: {subject}")
        except Exception as mail_exc:
            print(f"[Daily] 突发邮件发送失败: {mail_exc!r}")
        self.store.save_report(d_iso, "incident", f"突发告警 {stage}", body)
        self.store.log_run(d_iso, stage, "error", str(exc)[:300])
