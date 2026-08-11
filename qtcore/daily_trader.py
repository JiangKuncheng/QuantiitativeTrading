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
import time
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
from qtcore.holdings_chart import build_daily_chart
from qtcore.report_writer import (
    write_daily_report,
    write_incident_report,
    write_monthly_report,
    write_weekly_report,
)
from qtcore.realtime import get_open_price
from qtcore.store import Store
from qtcore.strategy import create_strategy


ROOT = Path(__file__).resolve().parent.parent
HALT_STATE_PATH = ROOT / "data" / "halt_state.json"


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

    def _signal_bt_config(self, params: dict[str, Any]):
        """信号/持仓回放配置: 关闭历史熔断(账户层熔断单独判断, 不套历史回撤)。"""
        bt = self._bt_config(params)
        bt.max_drawdown_halt = 0.0
        bt.halt_cooldown_days = 0
        bt.halt_resume_drawdown = 0.0
        return bt

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(
        self,
        run_date: str | None = None,
        force_weekly: bool = False,
        force_monthly: bool = False,
        settle_retries: int = 20,
        settle_retry_interval: int = 15,
    ) -> dict[str, Any]:
        today = datetime.strptime(run_date, "%Y%m%d").date() if run_date else date.today()
        d_str = today.strftime("%Y%m%d")
        d_iso = today.strftime("%Y-%m-%d")
        for attempt in range(settle_retries + 1):
            try:
                return self._run_inner(today, d_str, d_iso, force_weekly, force_monthly)
            except RuntimeError as exc:
                # 当日行情未发布是常见情况(新浪/腾讯收盘后一两个小时才出当日日线),
                # 自动等待重试, 数据一到即出日报; 超过重试上限才发突发告警
                stale_msgs = ("行情数据未发布", "基准指数数据未更新到今日")
                if any(k in str(exc) for k in stale_msgs) and attempt < settle_retries:
                    print(
                        f"[Daily] 当日行情未发布, 第 {attempt + 1}/{settle_retries} 次等待, "
                        f"{settle_retry_interval} 分钟后重试..."
                    )
                    time.sleep(settle_retry_interval * 60)
                    continue
                self._handle_incident(today, "daily_run", exc)
                raise
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
                result = BacktestEngine(self._signal_bt_config(params)).run(bars, strategy)

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
        # 基准指数仅用于日报对比, 若数据源延迟则降级为最近可用数据并注明, 不阻塞结算
        benchmark_note = ""
        if bench_df.index[-1].strftime("%Y-%m-%d") < d_iso:
            benchmark_note = (
                f"基准指数最新数据为 {bench_df.index[-1].strftime('%Y-%m-%d')}"
                f"(未更新到今日), 大盘对比使用最近可用数据"
            )
            print(f"[Daily] 警告: {benchmark_note}")
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
            # 持仓口径与账户一致: 用实际建仓股数覆盖回测口径
            for t in build_trades:
                code = t["code"]
                if code in holdings:
                    holdings[code]["units"] = t["units"]
                    holdings[code]["last_close"] = t["price"]
                    holdings[code]["value"] = round(t["units"] * t["price"], 2)
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
        # 账户层熔断状态机(基于建仓以来的权益链, 不套历史回撤)
        halted = self._account_halt_check(today, d_iso, params)
        self.store.save_trades(today_trades)
        self.store.save_signals(d_iso, targets, params)
        # 生成明日交易计划(供次日 9:20 执行 / 模拟模式 16:00 回放)
        tomorrow_plan = self._save_tomorrow_plan(today, params, targets, holdings, halted)
        self.store.log_run(d_iso, "daily", "ok", f"equity={today_equity:.2f}")

        # 生成日报附件图: 左持仓饼图 + 右收益折线
        chart_path: Path | None = None
        try:
            chart_path = build_daily_chart(
                db_path=self.store.path,
                cache_dir=self.app.paths.cache_dir,
                out_path=self.app.paths.output_dir / f"daily_chart_{d_str}.png",
            )
            print(f"[Daily] 日报图表已生成: {chart_path}")
        except Exception as exc:
            print(f"[Daily] 日报图表生成失败(不影响日报): {exc!r}")

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
            "benchmark_note": benchmark_note,
            "trades_today": today_trades,
            "holdings": holdings,
            "signals": targets,
            "tomorrow_plan": tomorrow_plan,
            "data_failed": failed,
            "note": "账户从 account_start 起新建仓; 建仓日按收盘信号建仓、当日盈亏记0, 次日开始计收益",
        }
        body = write_daily_report(daily_data)
        subject = f"[QuantTrader] 日报 {d_iso}"
        send_email(
            subject,
            body,
            config=self.email_cfg,
            attachments=[chart_path] if chart_path else None,
        )
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

    def _save_tomorrow_plan(
        self,
        today: date,
        params: dict[str, Any],
        targets: dict[str, int],
        holdings: dict[str, dict[str, Any]],
        halted: bool = False,
    ) -> list[dict[str, Any]]:
        """根据今日收盘信号生成明日交易计划并入库; 熔断期间只生成清仓计划。"""
        nxt = self.next_trading_day(today)
        if nxt is None:
            return []
        nxt_iso = nxt.strftime("%Y-%m-%d")
        plans: list[dict[str, Any]] = []
        if halted:
            # 熔断: 次日全部清仓, 不再开新仓
            for code, held in holdings.items():
                plans.append(
                    {
                        "date": nxt_iso,
                        "code": code,
                        "action": "SELL",
                        "units": held["units"],
                        "params": json.dumps(params, ensure_ascii=False),
                    }
                )
        else:
            for code, tgt in targets.items():
                held = holdings.get(code)
                if tgt == 1:
                    action = "HOLD" if held else "BUY"
                    plans.append(
                        {
                            "date": nxt_iso,
                            "code": code,
                            "action": action,
                            "units": held["units"] if held else None,
                            "params": json.dumps(params, ensure_ascii=False),
                        }
                    )
                elif tgt == 0 and held:
                    plans.append(
                        {
                            "date": nxt_iso,
                            "code": code,
                            "action": "SELL",
                            "units": held["units"],
                            "params": json.dumps(params, ensure_ascii=False),
                        }
                    )
        self.store.save_plans(nxt_iso, plans)
        print(f"[Daily] 明日计划已生成({nxt_iso}): {[(p['code'], p['action']) for p in plans]}")
        return plans

    def _account_halt_check(
        self,
        today: date,
        d_iso: str,
        params: dict[str, Any],
    ) -> bool:
        """
        账户层熔断状态机: 用"建仓以来"的账户权益链计算回撤,
        触发 -> 冷却 -> 恢复, 状态存 data/halt_state.json。
        """
        halt_limit = float(params.get("max_drawdown_halt", 0.0))
        resume_limit = float(params.get("halt_resume_drawdown", 0.0))
        cooldown = int(params.get("halt_cooldown_days", 0))
        if halt_limit <= 0:
            return False

        acc_start = str(self.cfg.get("account_start", d_iso))
        acc_start_iso = f"{acc_start[:4]}-{acc_start[4:6]}-{acc_start[6:]}"
        eq_rows = self.store.equity_between(acc_start_iso, d_iso)
        if len(eq_rows) < 2:
            return False
        eqs = [r["equity"] for r in eq_rows]
        peak = max(eqs)
        dd = eqs[-1] / peak - 1.0 if peak > 0 else 0.0

        state: dict[str, Any] = {}
        if HALT_STATE_PATH.exists():
            try:
                state = json.loads(HALT_STATE_PATH.read_text(encoding="utf-8"))
            except Exception:
                state = {}

        if state.get("halted"):
            cooldown_until = str(state.get("cooldown_until", ""))
            recovered = resume_limit <= 0 or dd > -abs(resume_limit)
            if d_iso >= cooldown_until and recovered:
                HALT_STATE_PATH.unlink(missing_ok=True)
                print(f"[Daily] {d_iso} 账户熔断解除(回撤 {dd:.2%} 回到阈值内)")
                return False
            print(f"[Daily] {d_iso} 账户熔断中(回撤 {dd:.2%}, 冷却至 {cooldown_until}), 次日计划=清仓")
            return True

        if dd <= -abs(halt_limit):
            cooldown_until = d_iso
            for _ in range(max(cooldown, 1)):
                nxt = self.next_trading_day(
                    datetime.strptime(cooldown_until, "%Y-%m-%d").date()
                )
                cooldown_until = nxt.strftime("%Y-%m-%d") if nxt else cooldown_until
            HALT_STATE_PATH.write_text(
                json.dumps(
                    {
                        "halted": True,
                        "halted_since": d_iso,
                        "cooldown_until": cooldown_until,
                        "drawdown": round(dd, 6),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[Daily] {d_iso} 触发账户熔断: 回撤 {dd:.2%}, 次日计划=清仓")
            return True
        return False

    def execute_plan(
        self,
        run_date: str | None = None,
        retries: int = 5,
        retry_interval: int = 10,
    ) -> dict[str, Any]:
        """
        9:20 执行任务(两步):
          Step 1: 读取昨日收盘生成的当日计划(plans 表);
          Step 2: 9:25 集合竞价结束后取"今日开盘价"成交; 拿不到则每 retry_interval 分钟
                  重试, 最多 retries 次; 重试成功后仍按开盘价(今开)成交, 不按重试时最新价;
                  全部重试后仍拿不到则发突发告警, 不瞎成交。
        - simulation 模式: 计划由 16:00 结算回放执行, 此处只展示不成交;
        - realtime 模式: 拉实时价成交, 拿不到实时价则发突发告警。
        """
        today = datetime.strptime(run_date, "%Y%m%d").date() if run_date else date.today()
        d_iso = today.strftime("%Y-%m-%d")
        if not self.is_trading_day(today):
            print(f"[Execute] {d_iso} 非交易日, 跳过")
            return {"executed": False, "reason": "non_trading"}

        plans = self.store.load_plans(d_iso)
        if not plans:
            print(f"[Execute] {d_iso} 无交易计划(可能 16:00 结算尚未生成)")
            return {"executed": False, "reason": "no_plan"}

        mode = str(self.cfg.get("execution_mode", "simulation"))
        if mode == "simulation":
            print(f"[Execute] {d_iso} 模拟模式: 计划由 16:00 结算回放执行, 此处不成交:")
            for p in plans:
                print(f"  {p['code']} {p['action']} units={p['units']}")
            return {"executed": False, "reason": "simulation_mode", "plans": plans}

        # realtime 模式: 两步执行
        needed = [p for p in plans if p["action"] != "HOLD"]
        if not needed:
            print(f"[Execute] {d_iso} 计划全部为 HOLD, 今日无需成交")
            return {"executed": False, "reason": "no_action", "plans": plans}

        # Step 2: 等到开盘价可用后成交(9:25 集合竞价结束, 轮询最长 wait_minutes 分钟)
        print(
            f"[Execute] {d_iso} 获取开盘价: 9:25 首次, 失败每 {retry_interval} 分钟重试, "
            f"最多 {retries} 次..."
        )
        got, missing = self._wait_open_prices(
            [p["code"] for p in needed],
            retries=retries,
            interval_minutes=retry_interval,
        )
        if missing:
            self._handle_incident(
                today,
                "execute_plan",
                RuntimeError(f"开盘价不可用, 以下标的无法成交: {missing}"),
            )
            return {"executed": False, "reason": "realtime_unavailable", "missing": missing}

        params = dict(self.cfg["strategy"]["params"])
        bt_cfg = self._bt_config(params)
        prev_equity = self.store.prev_equity_before(d_iso) or float(self.cfg["initial_capital"])
        buys = [p for p in needed if p["action"] == "BUY"]
        budget_each = prev_equity * bt_cfg.position_ratio / max(len(buys), 1)
        fills: list[dict[str, Any]] = []

        for p in needed:
            price = got[p["code"]]
            if p["action"] == "BUY":
                units = int(budget_each / (price * self.app.backtest.lot_size)) * self.app.backtest.lot_size
            else:
                units = int(p.get("units") or 0)
            if units <= 0:
                continue
            fills.append(
                {
                    "date": d_iso,
                    "code": p["code"],
                    "name": self._names.get(p["code"], p["code"]),
                    "side": "BUY" if p["action"] == "BUY" else "SELL",
                    "units": units,
                    "price": round(price, 3),
                    "commission": round(units * price * bt_cfg.commission_rate, 4),
                    "pnl": None,
                    "reason": "plan_execute",
                }
            )

        if fills:
            self.store.save_trades(fills)
        print(f"[Execute] {d_iso} 成交 {len(fills)} 笔(实时行情)")
        for f in fills:
            print(f"  {f['code']} {f['side']} {f['units']}股 @ {f['price']}")
        return {"executed": True, "fills": len(fills), "prices": {k: round(v, 3) for k, v in got.items()}}

    def _wait_open_prices(
        self,
        codes: list[str],
        retries: int = 5,
        interval_minutes: int = 10,
    ) -> tuple[dict[str, float], list[str]]:
        """
        获取开盘价: 9:25 集合竞价结束后首次尝试(取"今开"), 未取齐则每 interval_minutes
        分钟重试一次, 最多 retries 次; 全部拿到返回 (prices, []), 否则返回 (已有, 缺失)。
        """
        got: dict[str, float] = {}
        attempt = 0
        while True:
            now = datetime.now()
            if (now.hour, now.minute) >= (9, 25):  # 9:25 集合竞价结束, 开盘价确定
                for code in codes:
                    if code in got:
                        continue
                    quote = get_open_price(code)
                    if quote is not None:
                        got[code] = float(quote["price"])
                if len(got) == len(codes):
                    print(f"[Execute] 开盘价就绪: {got}")
                    return got, []
            if attempt >= retries:
                break
            attempt += 1
            print(
                f"[Execute] 第 {attempt}/{retries} 次未取齐, "
                f"{interval_minutes} 分钟后重试..."
            )
            time.sleep(interval_minutes * 60)
        missing = [c for c in codes if c not in got]
        return got, missing

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
