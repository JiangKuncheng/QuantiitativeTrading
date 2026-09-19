"""
冒烟测试: 验证"数据 -> 策略 -> 回测 -> 结果"全链路在离线环境下可跑通。

运行:
    python -m unittest discover -s tests -v
"""

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import qtcore.daily_trader as dt_module
from qtcore.backtest.engine import BacktestEngine
from qtcore.config import AppConfig
from qtcore.datacenter.data_center import DataCenter
from qtcore.daily_trader import DailyTrader
from qtcore.main_manager import MainManager
from qtcore.screener import StockScreener
from qtcore.store import Store
from qtcore.strategy import create_strategy
from qtcore.trainer import (
    TrainingConfig,
    WalkForwardTrainer,
    balanced_years_split,
    classify_yearly_regimes,
    compare_with_benchmark,
)


class PipelineSmokeTest(unittest.TestCase):
    """全链路冒烟测试。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = AppConfig()
        self.config.paths.root = Path(self.tmp.name)
        self.config.data.offline_fallback = True
        self.config.data.use_cache = False

    def test_offline_pipeline_via_main_manager(self) -> None:
        """MainManager 完整工作流。"""
        manager = MainManager(self.config)
        result = manager.run()
        self.assertGreater(len(result.equity_curve), 0)
        self.assertIn("max_drawdown", result.stats)
        self.assertIn("sharpe", result.stats)
        saved = result.save(self.config.paths.output_dir)
        self.assertTrue(saved["equity_curve"].exists())
        self.assertTrue(saved["trades"].exists())
        self.assertTrue(saved["stats"].exists())

    def test_strategy_ma_cross_signal_count(self) -> None:
        """双均线策略在合成数据上应产生至少 1 个信号事件。"""
        dc = DataCenter(self.config.data, self.config.paths)
        bars = dc.generate_synthetic_bars(days=300)
        strategy = create_strategy("ma_cross", {"fast": 5, "slow": 20})
        events = strategy.generate_signal_events(bars)
        self.assertGreater(len(events), 0)

    def test_backtest_engine_direct(self) -> None:
        """绕过 MainManager 直接驱动引擎。"""
        dc = DataCenter(self.config.data, self.config.paths)
        bars = dc.generate_synthetic_bars(days=250)
        strategy = create_strategy("ma_cross")
        engine = BacktestEngine(self.config.backtest)
        result = engine.run(bars, strategy)
        self.assertEqual(len(result.equity_curve), len(bars))

    def test_invalid_strategy_name(self) -> None:
        """未注册策略应抛出明确异常。"""
        with self.assertRaises(KeyError):
            create_strategy("not_exist")

    def test_screener_offline_rank(self) -> None:
        """选股器离线模式: 合成股票池 -> 回测 -> 排名。"""
        screener = StockScreener(self.config, fast=5, slow=20, synthetic=True)
        candidates = screener.filter_universe()
        self.assertEqual(len(candidates), 20)
        result = screener.rank(candidates, limit=5, metric="sharpe")
        self.assertGreaterEqual(len(result), 1)
        self.assertEqual(
            list(result.columns[:2]),
            ["code", "name"],
        )

    def test_walk_forward_trainer_offline(self) -> None:
        """训练器离线模式: 股票池 -> 选股 -> 验证/测试组合评估。"""
        training = TrainingConfig(pool_size=6, top_k=3)
        trainer = WalkForwardTrainer(self.config, training, synthetic=True)
        trainer.pool = trainer.load_pool()
        result = trainer.run_proposal({"fast": 5, "slow": 20})
        self.assertNotIn("error", result)
        self.assertIn("val", result)
        self.assertIn("test", result)
        self.assertEqual(len(result["top_symbols"].split(",")), 3)

    def test_rolling_walk_forward_offline(self) -> None:
        """滚动 Walk-Forward: 多折聚合, 输出平均验证/测试指标。"""
        training = TrainingConfig(pool_size=6, top_k=3)
        trainer = WalkForwardTrainer(self.config, training, synthetic=True)
        trainer.pool = trainer.load_pool()
        folds = WalkForwardTrainer.rolling_folds(2)
        result = trainer.run_proposal_rolling({"fast": 5, "slow": 20}, folds)
        self.assertNotIn("error", result)
        self.assertEqual(result["n_folds"], 2)
        self.assertIn("avg_val_sharpe", result)
        self.assertIn("avg_test_total_return", result)
        self.assertIn("fold_results", result)

    def test_engine_risk_params(self) -> None:
        """回测引擎支持 止损/止盈/杠杆/调仓周期/限价单/滑点容忍。"""
        dc = DataCenter(self.config.data, self.config.paths)
        bars = dc.generate_synthetic_bars(days=300)
        bt = replace(
            self.config.backtest,
            stop_loss_pct=0.08,
            take_profit_pct=0.20,
            leverage=1.5,
            rebalance="weekly",
            order_type="limit",
            slippage_tolerance_pct=0.0005,
        )
        result = BacktestEngine(bt).run(bars, create_strategy("ma_cross"))
        self.assertEqual(len(result.equity_curve), len(bars))
        self.assertIn("final_equity", result.stats)

    def test_regime_split(self) -> None:
        """牛/熊/横盘分类与均衡切分: 训练/测试均覆盖多类行情且不重叠。"""
        plan = {
            2020: (1000, 1200), 2021: (1200, 1224), 2022: (1224, 1040),
            2023: (1040, 1040), 2024: (1040, 1300), 2025: (1300, 1495),
            2026: (1495, 1495),
        }
        idx = pd.bdate_range("2020-01-01", "2026-08-10")
        close = []
        for ts in idx:
            start, end = plan[ts.year]
            days = 366 if ts.is_leap_year else 365
            close.append(start + (end - start) * (ts.dayofyear / days))
        bench = pd.DataFrame({"close": close}, index=idx)

        regimes = classify_yearly_regimes(bench)
        self.assertEqual(regimes[2020], "bull")
        self.assertEqual(regimes[2022], "bear")
        self.assertIn(regimes[2023], ("sideways",))

        train_years, test_years, info = balanced_years_split(regimes)
        self.assertTrue(train_years and test_years)
        self.assertEqual(set(train_years) & set(test_years), set())
        self.assertGreaterEqual(info["regime_score"], 4)

    def test_benchmark_compare(self) -> None:
        """策略 vs 基准对比: 输出超额/贝塔/捕获率等指标。"""
        rng = np.random.default_rng(7)
        strat = pd.Series(
            rng.normal(0.001, 0.01, 200),
            index=pd.bdate_range("2025-01-01", periods=200),
        )
        bench_close = 100.0 * np.exp(
            np.cumsum(np.random.default_rng(8).normal(0.0005, 0.008, 250))
        )
        bench_df = pd.DataFrame(
            {"close": bench_close},
            index=pd.bdate_range("2024-12-01", periods=250),
        )
        cmp = compare_with_benchmark(strat, bench_df)
        self.assertIn("excess_total_return", cmp)
        self.assertIn("beta", cmp)
        self.assertIn("up_capture", cmp)


class AccountReconcileTest(unittest.TestCase):
    """账户口径对账: 现金 + 持仓市值 落库, 并与策略权益链比对告警。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "trading.db"
        # 只覆盖对账逻辑: DailyTrader.__init__ 需要邮件配置(.env), 这里手工装配
        # store 与 cfg, 避免测试依赖线上密钥。
        self.trader = DailyTrader.__new__(DailyTrader)
        self.trader.cfg = {"initial_capital": 1_000_000}
        self.trader.store = Store(self.db)
        # Windows 上必须显式关闭 SQLite 连接, 否则临时目录删不掉
        self.addCleanup(self.trader.store.close)
        self.trader.store.save_equity(
            {
                "date": "2026-09-18",
                "equity": 1_000_000.0,
                "cash": None,
                "position_value": None,
                "daily_return": 0.0,
                "benchmark_return": 0.0,
                "strategy_total": 0.0,
                "benchmark_total": 0.0,
            }
        )
        # 买入 1,000 股 @200, 佣金 60 -> 现金 799,940
        self.trader.store.save_trades(
            [
                {
                    "date": "2026-09-18",
                    "code": "000001",
                    "name": "测试标的",
                    "side": "BUY",
                    "units": 1000,
                    "price": 200.0,
                    "commission": 60.0,
                    "pnl": None,
                    "reason": "open_long",
                }
            ]
        )

    def _last_status(self) -> tuple[str, str]:
        row = self.trader.store.conn.execute(
            "SELECT status, detail FROM run_log WHERE stage = 'reconcile'"
            " ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["status"], row["detail"]

    def test_reconcile_ok_persists_cash_and_position(self) -> None:
        holdings = {"000001": {"units": 1000, "value": 200_000.0}}
        # 一本账: 落库权益必须等于 现金 + 持仓市值
        out = self.trader._reconcile_account(
            "2026-09-18", 999_940.0, holdings, {"max_position_ratio": 1.0},
            strategy_equity=1_000_000.0,
        )
        self.assertAlmostEqual(out["cash"], 799_940.0, places=2)
        self.assertAlmostEqual(out["position_value"], 200_000.0, places=2)
        self.assertAlmostEqual(out["account_equity"], 999_940.0, places=2)
        self.assertAlmostEqual(out["book_gap"], 0.0, places=2)

        row = self.trader.store.conn.execute(
            "SELECT cash, position_value FROM equity_daily WHERE date = '2026-09-18'"
        ).fetchone()
        self.assertAlmostEqual(row["cash"], 799_940.0, places=2)
        self.assertAlmostEqual(row["position_value"], 200_000.0, places=2)
        self.assertEqual(self._last_status()[0], "ok")

    def test_reconcile_flags_position_over_capital(self) -> None:
        """持仓市值突破本金必须报 error —— 这正是历史 bug 的形态。"""
        holdings = {"000001": {"units": 1000, "value": 1_500_000.0}}
        self.trader._reconcile_account(
            "2026-09-18", 2_299_940.0, holdings, {"max_position_ratio": 1.0}
        )
        status, detail = self._last_status()
        self.assertEqual(status, "error")
        self.assertIn("超过上限", detail)

    def test_reconcile_flags_book_inconsistency(self) -> None:
        """落库权益与"现金+持仓"对不上必须报 error。"""
        holdings = {"000001": {"units": 1000, "value": 200_000.0}}
        self.trader._reconcile_account(
            "2026-09-18", 1_000_000.0, holdings, {"max_position_ratio": 1.0}
        )
        status, detail = self._last_status()
        self.assertEqual(status, "error")
        self.assertIn("不一致", detail)


class BuyBudgetTest(unittest.TestCase):
    """买入预算: 组合总仓位 / 单票集中度 / 现金 三重约束。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.trader = DailyTrader.__new__(DailyTrader)
        self.trader.cfg = {"initial_capital": 1_000_000}
        self.trader.store = Store(Path(self.tmp.name) / "trading.db")
        self.addCleanup(self.trader.store.close)

        bt = AppConfig().backtest
        bt.position_ratio = 0.95
        bt.max_position_ratio = 0.4
        self.bt = bt

    def test_single_name_cap_binds(self) -> None:
        """单票上限 40%: 预算被压到 40 万, 而不是 95 万。"""
        budgets = self.trader._buy_budgets(
            1_000_000.0, self.bt, {}, set(), [], ["000001"]
        )
        self.assertAlmostEqual(budgets["000001"], 400_000.0, places=2)

    def test_total_cap_is_shared_across_names(self) -> None:
        """6 个标的时是总仓位 95% 在起作用(95/6 = 15.8% < 单票上限 40%)。"""
        budgets = self.trader._buy_budgets(
            1_000_000.0,
            self.bt,
            {},
            set(),
            [],
            ["000001", "000002", "000003", "000004", "000005", "000006"],
        )
        self.assertAlmostEqual(budgets["000001"], 950_000.0 / 6, places=2)
        self.assertAlmostEqual(sum(budgets.values()), 950_000.0, places=2)

    def test_single_cap_binds_on_every_name(self) -> None:
        """2 个标的时单票上限 40% 各自生效(2 x 40% < 95%)。"""
        budgets = self.trader._buy_budgets(
            1_000_000.0, self.bt, {}, set(), [], ["000001", "000002"]
        )
        self.assertAlmostEqual(budgets["000001"], 400_000.0, places=2)
        self.assertAlmostEqual(budgets["000002"], 400_000.0, places=2)

    def test_retained_position_consumes_headroom(self) -> None:
        """已有仓位占用预算: 单票已用满 40% 则不再加仓。"""
        holdings = {"000001": {"units": 4000, "avg_cost": 100.0}}  # 成本 40 万
        budgets = self.trader._buy_budgets(
            1_000_000.0, self.bt, holdings, set(), [], ["000001"]
        )
        self.assertAlmostEqual(budgets["000001"], 0.0, places=2)

    def test_cash_binds_when_below_headroom(self) -> None:
        """现金不足时以现金为准。"""
        self.trader.store.save_trades(
            [
                {
                    "date": "2026-09-18",
                    "code": "000009",
                    "name": "测试",
                    "side": "BUY",
                    "units": 1000,
                    "price": 800.0,
                    "commission": 0.0,
                    "pnl": None,
                    "reason": "open_long",
                }
            ]
        )
        budgets = self.trader._buy_budgets(
            1_000_000.0, self.bt, {}, set(), [], ["000001"]
        )
        # 现金仅剩 20 万, 低于单票上限 40 万
        self.assertAlmostEqual(budgets["000001"], 200_000.0 / 1.0003, places=1)


class HaltRecoveryTest(unittest.TestCase):
    """
    熔断必须能被解除。

    旧逻辑用"账户回撤恢复到阈值内"当解除条件, 但熔断后账户天天清仓、100% 现金,
    权益被冻结 => 回撤永远停在触发时的水平, 条件恒不可满足 => 永久熔断。
    """

    @staticmethod
    def _crash_bars() -> pd.DataFrame:
        closes: list[float] = []
        price = 100.0
        for _ in range(30):      # 上涨段: 建仓并创出权益峰值
            price *= 1.01
            closes.append(price)
        price *= 0.75            # 单日暴跌 25%: 95% 仓位 -> 权益回撤约 23.8%
        closes.append(price)
        for _ in range(60):      # 反弹段: 策略信号应重新转多
            price *= 1.01
            closes.append(price)
        index = pd.bdate_range("2024-01-01", periods=len(closes))
        bars = pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [1_000_000.0] * len(closes),
            },
            index=index,
        )
        bars.attrs["code"] = "TEST"
        return bars

    def test_halt_releases_and_account_reenters(self) -> None:
        bt = replace(
            AppConfig().backtest,
            initial_capital=1_000_000.0,
            position_ratio=0.95,
            max_position_ratio=1.0,
            max_drawdown_halt=0.2,
            halt_cooldown_days=5,
            halt_resume_drawdown=0.1,
        )
        result = BacktestEngine(bt).run(
            self._crash_bars(), create_strategy("ma_cross", {"fast": 2, "slow": 5})
        )
        equity = result.equity_curve["equity"]
        drawdown = equity / equity.cummax() - 1.0
        self.assertLessEqual(float(drawdown.min()), -0.2, "构造的数据应触发回撤熔断")

        halt_day = drawdown[drawdown <= -0.2].index[0]
        after = result.trades[
            pd.to_datetime(result.trades["datetime"]) > halt_day
        ]
        self.assertGreater(
            len(after[after["side"] == "BUY"]),
            0,
            "熔断冷却期过后应能重新建仓(旧逻辑会永久锁死)",
        )


class StopExitReentryTest(unittest.TestCase):
    """
    止损/止盈平仓后必须能重新建仓。

    旧逻辑: _check_stops 平仓后没有同步 last_target_fraction, 引擎仍认为"已满仓多头",
    策略继续给多头信号也不再建仓 —— 实测 688072 在止盈后空仓近 3 个月。
    """

    def test_take_profit_exit_reenters_while_signal_stays_long(self) -> None:
        closes: list[float] = []
        price = 100.0
        for _ in range(60):      # 单边上涨: 策略始终多头, 且会反复触发止盈
            price *= 1.015
            closes.append(price)
        index = pd.bdate_range("2024-01-01", periods=len(closes))
        bars = pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [1_000_000.0] * len(closes),
            },
            index=index,
        )
        bars.attrs["code"] = "TEST"

        bt = replace(
            AppConfig().backtest,
            initial_capital=1_000_000.0,
            position_ratio=0.95,
            max_position_ratio=1.0,
            stop_loss_pct=0.0,
            take_profit_pct=0.25,
            max_drawdown_halt=0.0,
        )
        result = BacktestEngine(bt).run(
            bars, create_strategy("ma_cross", {"fast": 2, "slow": 5})
        )
        trades = result.trades
        sells = trades[trades["side"] == "SELL"]
        buys = trades[trades["side"] == "BUY"]
        self.assertGreaterEqual(len(sells), 1, "单边上涨应触发止盈平仓")

        first_sell = pd.to_datetime(sells["datetime"]).min()
        reentries = buys[pd.to_datetime(buys["datetime"]) > first_sell]
        self.assertGreater(
            len(reentries), 0, "止盈平仓后信号仍为多头时必须重新建仓"
        )


class AccountHaltTest(unittest.TestCase):
    """实盘账户层熔断: 冷却期满必须解除, 不能因为空仓而永久锁死。"""

    PARAMS = {
        "max_drawdown_halt": 0.2,
        "halt_cooldown_days": 5,
        "halt_resume_drawdown": 0.1,
    }

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.halt_path = Path(self.tmp.name) / "halt_state.json"
        self._original_halt_path = dt_module.HALT_STATE_PATH
        dt_module.HALT_STATE_PATH = self.halt_path
        self.addCleanup(
            lambda: setattr(dt_module, "HALT_STATE_PATH", self._original_halt_path)
        )

        self.trader = DailyTrader.__new__(DailyTrader)
        self.trader.cfg = {"initial_capital": 1_000_000, "account_start": "20260810"}
        self.trader.store = Store(Path(self.tmp.name) / "trading.db")
        self.addCleanup(self.trader.store.close)
        self.trader._calendar = set()  # 无交易日历 -> 回退到"周一至周五"

    def _equity(self, day: str, value: float) -> None:
        self.trader.store.save_equity(
            {
                "date": day,
                "equity": value,
                "cash": value,
                "position_value": 0.0,
                "daily_return": 0.0,
                "benchmark_return": 0.0,
                "strategy_total": value / 1_000_000 - 1.0,
                "benchmark_total": 0.0,
            }
        )

    def test_halt_triggers_then_releases_after_cooldown(self) -> None:
        self._equity("2026-08-10", 1_000_000.0)
        self._equity("2026-08-11", 780_000.0)  # -22% -> 触发熔断
        self.assertTrue(
            self.trader._account_halt_check(
                date(2026, 8, 11), "2026-08-11", self.PARAMS
            )
        )
        self.assertTrue(self.halt_path.exists(), "应写入熔断状态文件")

        # 冷却期内: 仍然熔断(次日计划清仓)
        self.assertTrue(
            self.trader._account_halt_check(
                date(2026, 8, 12), "2026-08-12", self.PARAMS
            )
        )

        # 空仓期间权益不变, 冷却期满(5 个交易日后 = 08-18) 必须解除
        self._equity("2026-08-18", 780_000.0)
        self.assertFalse(
            self.trader._account_halt_check(
                date(2026, 8, 18), "2026-08-18", self.PARAMS
            ),
            "冷却期满应解除熔断(旧逻辑会因空仓永远锁死)",
        )
        self.assertTrue(self.halt_path.exists(), "解除后保留状态文件以携带新的回撤基准")
        self.assertFalse(
            json.loads(self.halt_path.read_text(encoding="utf-8"))["halted"]
        )

        # 关键回归: 解除后的第二天不能因为旧峰值而立刻再次熔断(否则就是死循环)
        self._equity("2026-08-19", 780_000.0)
        self.assertFalse(
            self.trader._account_halt_check(
                date(2026, 8, 19), "2026-08-19", self.PARAMS
            ),
            "解除后不应立刻再次触发熔断",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
