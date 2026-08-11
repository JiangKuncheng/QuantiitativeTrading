"""
冒烟测试: 验证"数据 -> 策略 -> 回测 -> 结果"全链路在离线环境下可跑通。

运行:
    python -m unittest discover -s tests -v
"""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from qtcore.backtest.engine import BacktestEngine
from qtcore.config import AppConfig
from qtcore.datacenter.data_center import DataCenter
from qtcore.main_manager import MainManager
from qtcore.screener import StockScreener
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
