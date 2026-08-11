"""
Trainer 训练模块
================

把"选股 + 交易 + 时间窗口"变成可联合训练的对象:

1. 数据集: 默认使用 2020-01-01 ~ 2026-08-10 全窗口,
   自动切分为 train / val / test 三段(可配置);
2. 组合级评估: 先按训练集对股票池逐只打分并选出 Top-K(选股),
   再用同一套交易参数对 Top-K 做等权组合回测(交易);
3. Walk-Forward: 训练集选股调参 -> 验证集挑选方案 -> 测试集给出最终报告,
   严格避免"用未来数据选参数";
4. 大模型调参循环: 每轮由 LLM/人工提出 proposal(参数提案),
   系统评估并返回指标, 下一轮提案参考上一轮结果迭代。

提案(proposal)可训练的维度:
- 选股: top_k(持仓数量), select_metric(选股排序指标), 股票池(pool)
- 交易: fast/slow(均线窗口), long_short(多空), position_ratio(仓位比例)
- 时间: train/val/test 窗口本身可配置(数据集即训练对象)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qtcore.backtest.engine import BacktestEngine
from qtcore.config import AppConfig
from qtcore.datacenter.data_center import DataCenter
from qtcore.screener import StockScreener
from qtcore.strategy import create_strategy


SAMPLING_OFFSETS: dict[str, str | None] = {
    "daily": None,
    "weekly": "W",
    "monthly": "ME",
}


def resample_bars(bars: pd.DataFrame, sampling: str = "daily") -> pd.DataFrame:
    """
    数据采样频率: daily 原样 / weekly 周线 / monthly 月线。
    OHLCV 聚合规则: open 取首, high 取高, low 取低, close 取尾, volume/amount 求和。
    """
    offset = SAMPLING_OFFSETS.get(str(sampling).lower(), None)
    if offset is None:
        return bars
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "amount": "sum",
    }
    df = bars.resample(offset).agg(agg).dropna(subset=["close"])
    df.attrs = dict(bars.attrs)
    return df


@dataclass
class TrainingConfig:
    """2020-2026 训练数据集切分与训练超参数。"""

    train_start: str = "20200101"
    train_end: str = "20221231"
    val_start: str = "20230101"
    val_end: str = "20231231"
    test_start: str = "20240101"
    test_end: str = "20260810"
    pool_size: int = 12       # 参与训练的股票池大小
    top_k: int = 5            # 选出的持仓数量
    select_metric: str = "sharpe"   # 选股排序指标
    initial_capital: float = 1_000_000.0

    @property
    def windows(self) -> dict[str, tuple[str, str]]:
        """三段数据集窗口。"""
        return {
            "train": (self.train_start, self.train_end),
            "val": (self.val_start, self.val_end),
            "test": (self.test_start, self.test_end),
        }


class PortfolioEvaluator:
    """组合级评估: 选股结果 + 交易参数 -> 等权组合绩效。"""

    def __init__(self, app: AppConfig, training: TrainingConfig, synthetic: bool = False) -> None:
        self.app = app
        self.training = training
        self.synthetic = synthetic
        self.dc = DataCenter(app.data, app.paths)

    # ------------------------------------------------------------------
    # 单只评分(用于训练集选股)
    # ------------------------------------------------------------------
    def score_symbols(
        self,
        symbols: list[str],
        strategy_params: dict[str, Any],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        在指定时间窗口对每只股票跑完整回测, 返回逐只绩效表。
        单只失败自动跳过, 保证训练流程不被个别坏数据打断。
        """
        rows: list[dict[str, Any]] = []
        for code in symbols:
            try:
                timeframe = str(strategy_params.get("timeframe", "daily")).lower()
                bars = self._load_bars(code, start, end, timeframe)
                min_bars = 240 if timeframe != "daily" else 120
                if bars is None or len(bars) < min_bars:
                    continue
                strategy = create_strategy("ma_cross", strategy_params)
                stats = BacktestEngine(self._bt_config(strategy_params)).run(bars, strategy).stats
                rows.append({"code": code, **stats})
            except Exception as exc:
                print(f"[Trainer] {code} 训练集回测失败, 跳过: {exc!r}")
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 组合级评估(选出的 Top-K 等权组合)
    # ------------------------------------------------------------------
    def portfolio_metrics(
        self,
        symbols: list[str],
        strategy_params: dict[str, Any],
        start: str,
        end: str,
        window_label: str = "",
    ) -> dict[str, Any]:
        """
        对 Top-K 股票做等权组合回测:
        - 每只股票独立跑策略得到日收益序列;
        - 组合日收益 = 当日各股收益均值(停牌日按 0 处理, 每日近似再平衡);
        - 组合权益 = 初始资金 x cumprod(1 + 组合日收益)。
        """
        n_requested = len(symbols)
        returns, trades = self._collect_returns(symbols, strategy_params, start, end)

        if not returns:
            return {"window": window_label, "n_symbols_requested": n_requested, "n_symbols": 0, "coverage": 0.0, "error": "no data"}

        ret_df = pd.DataFrame(returns).fillna(0.0)
        port_ret = ret_df.mean(axis=1)  # 等权组合日收益
        equity = self.training.initial_capital * (1.0 + port_ret).cumprod()

        total_return = float(equity.iloc[-1] / self.training.initial_capital - 1.0)
        n_days = len(port_ret)
        annual_return = (1.0 + total_return) ** (252 / n_days) - 1.0 if n_days > 0 and total_return > -1.0 else -1.0
        sharpe = (
            float(port_ret.mean() / port_ret.std(ddof=1) * np.sqrt(252))
            if len(port_ret) > 1 and port_ret.std(ddof=1) > 0
            else 0.0
        )
        drawdown = equity / equity.cummax() - 1.0
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        # 聚合成交: 胜率 / 盈亏比 / 交易次数(防御: 全部零交易时 concat 出无列空表)
        if trades and not all(t.empty for t in trades):
            all_trades = pd.concat(trades, ignore_index=True)
            if all_trades.empty or "pnl" not in all_trades.columns:
                n_trades = wins = 0
                profit_factor = win_rate = 0.0
            else:
                closed = all_trades[all_trades["pnl"].notna()]
                n_trades = int(len(closed))
                wins = int((closed["pnl"] > 0).sum())
                gross_profit = float(closed.loc[closed["pnl"] > 0, "pnl"].sum())
                gross_loss = float(closed.loc[closed["pnl"] < 0, "pnl"].sum())
                profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else (float("inf") if gross_profit > 0 else 0.0)
                win_rate = wins / n_trades if n_trades else 0.0
        else:
            n_trades = wins = 0
            profit_factor = win_rate = 0.0

        return {
            "window": window_label,
            "n_symbols_requested": n_requested,
            "n_symbols": len(ret_df.columns),
            "coverage": round(len(ret_df.columns) / n_requested, 4) if n_requested else 0.0,
            "total_return": round(total_return, 6),
            "annual_return": round(annual_return, 6),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_drawdown, 6),
            "n_trades": n_trades,
            "win_rate": round(win_rate, 6),
            "profit_factor": round(profit_factor, 4),
        }

    def portfolio_returns(
        self,
        symbols: list[str],
        strategy_params: dict[str, Any],
        start: str,
        end: str,
    ) -> pd.Series:
        """组合日收益序列(用于与大盘基准对比)。"""
        returns: dict[str, pd.Series] = {}
        returns, _ = self._collect_returns(symbols, strategy_params, start, end)
        if not returns:
            return pd.Series(dtype=float)
        return pd.DataFrame(returns).fillna(0.0).mean(axis=1)

    def _collect_returns(
        self,
        symbols: list[str],
        strategy_params: dict[str, Any],
        start: str,
        end: str,
    ) -> tuple[dict[str, pd.Series], list[pd.DataFrame]]:
        """逐只回测并收集日收益序列与成交明细(数据走缓存, 多次调用开销小)。"""
        returns: dict[str, pd.Series] = {}
        trades: list[pd.DataFrame] = []

        for code in symbols:
            try:
                timeframe = str(strategy_params.get("timeframe", "daily")).lower()
                bars = self._load_bars(code, start, end, timeframe)
                min_bars = 240 if timeframe != "daily" else 60
                if bars is None or len(bars) < min_bars:
                    continue
                strategy = create_strategy("ma_cross", strategy_params)
                result = BacktestEngine(self._bt_config(strategy_params)).run(bars, strategy)
                returns[code] = result.equity_curve["daily_return"]
                trades.append(result.trades)
            except Exception as exc:
                print(f"[Trainer] {code} 回测失败, 跳过: {exc!r}")
        return returns, trades

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _bt_config(self, strategy_params: dict[str, Any]):
        bt = replace(self.app.backtest)
        bt.initial_capital = self.training.initial_capital
        bt.position_ratio = float(strategy_params.get("position_ratio", bt.position_ratio))
        bt.allow_short = bool(strategy_params.get("allow_short", bt.allow_short))
        bt.rebalance = str(strategy_params.get("rebalance", bt.rebalance))
        bt.order_type = str(strategy_params.get("order_type", bt.order_type))
        bt.slippage_tolerance_pct = float(strategy_params.get("slippage_tolerance_pct", bt.slippage_tolerance_pct))
        bt.leverage = float(strategy_params.get("leverage", bt.leverage))
        bt.max_position_ratio = float(strategy_params.get("max_position_ratio", bt.max_position_ratio))
        bt.stop_loss_pct = float(strategy_params.get("stop_loss_pct", bt.stop_loss_pct))
        bt.take_profit_pct = float(strategy_params.get("take_profit_pct", bt.take_profit_pct))
        bt.max_drawdown_halt = float(strategy_params.get("max_drawdown_halt", bt.max_drawdown_halt))
        bt.halt_cooldown_days = int(strategy_params.get("halt_cooldown_days", bt.halt_cooldown_days))
        bt.halt_resume_drawdown = float(strategy_params.get("halt_resume_drawdown", bt.halt_resume_drawdown))
        return bt

    def _load_bars(self, code: str, start: str, end: str, timeframe: str = "daily") -> pd.DataFrame:
        if self.synthetic:
            seed = sum(ord(ch) for ch in code) % 10000 or 1
            return self.dc.generate_synthetic_bars(days=600, symbol=code, seed=seed)
        # 网络重试已统一在 DataCenter 层(指数退避), 这里单次调用即可
        data_cfg = replace(self.app.data, offline_fallback=False, start_date=start, end_date=end)
        return DataCenter(data_cfg, self.app.paths).get_bars(
            symbol=code,
            start_date=start,
            end_date=end,
            timeframe=timeframe,
        )


class WalkForwardTrainer:
    """
    Walk-Forward 训练器:
        训练集选股(Top-K) + 调参 -> 验证集挑选方案 -> 测试集最终报告。
    支持两种模式:
        run_proposal           单窗口(简单模式)
        run_proposal_rolling   滚动多段交叉验证(推荐, 验证信号更可靠)
    """

    def __init__(
        self,
        app: AppConfig,
        training: TrainingConfig,
        synthetic: bool = False,
        universe: str = "all",
    ) -> None:
        self.app = app
        self.training = training
        self.synthetic = synthetic
        self.universe = universe
        self.evaluator = PortfolioEvaluator(app, training, synthetic=synthetic)

    def load_pool(self, symbols: list[str] | None = None, universe: str | None = None) -> list[str]:
        """加载股票池: 显式列表优先, 否则用选股器全市场初筛取前 pool_size。"""
        if symbols:
            return symbols
        screener = StockScreener(
            self.app,
            synthetic=self.synthetic,
            universe=universe or self.universe,
        )
        candidates = screener.filter_universe()
        return [str(c) for c in candidates.head(self.training.pool_size)["code"]]

    @staticmethod
    def rolling_folds(folds: int = 3) -> list[dict[str, tuple[str, str]]]:
        """
        默认滚动折(扩展式训练集, 覆盖 2020-2026):
            折1: train 2020-2022 / val 2023 / test 2024
            折2: train 2020-2023 / val 2024 / test 2025
            折3: train 2020-2024 / val 2025 / test 2026(截至当前)
        """
        all_folds = [
            {
                "train": ("20200101", "20221231"),
                "val": ("20230101", "20231231"),
                "test": ("20240101", "20241231"),
            },
            {
                "train": ("20200101", "20231231"),
                "val": ("20240101", "20241231"),
                "test": ("20250101", "20251231"),
            },
            {
                "train": ("20200101", "20241231"),
                "val": ("20250101", "20251231"),
                "test": ("20260101", "20260810"),
            },
        ]
        return all_folds[:folds]

    def run_proposal(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """
        执行一个提案(一轮训练):
        1. 训练集逐只回测 -> 按 select_metric 选出 Top-K(选股训练);
        2. 验证集组合回测(交易验证);
        3. 测试集组合回测(最终报告)。
        """
        strategy_params = {
            "fast": int(proposal.get("fast", 5)),
            "slow": int(proposal.get("slow", 20)),
            "long_short": bool(proposal.get("long_short", False)),
            "position_ratio": float(proposal.get("position_ratio", self.app.backtest.position_ratio)),
            "allow_short": bool(proposal.get("allow_short", False)),
        }
        top_k = int(proposal.get("top_k", self.training.top_k))
        select_metric = proposal.get("select_metric", self.training.select_metric)

        # 1) 选股: 训练集打分排序
        train_start, train_end = self.training.windows["train"]
        scores = self.evaluator.score_symbols(self.pool, strategy_params, train_start, train_end)
        if scores.empty or len(scores) < top_k:
            return {**proposal, "error": f"训练集可用标的不足({len(scores)}/{top_k})"}
        top_symbols = scores.nlargest(top_k, select_metric)["code"].tolist()

        # 2) 验证 3) 测试
        val_start, val_end = self.training.windows["val"]
        test_start, test_end = self.training.windows["test"]
        val = self.evaluator.portfolio_metrics(top_symbols, strategy_params, val_start, val_end, "val")
        test = self.evaluator.portfolio_metrics(top_symbols, strategy_params, test_start, test_end, "test")

        return {
            **proposal,
            "top_symbols": ",".join(top_symbols),
            "train_best_score": float(scores.iloc[0][select_metric]),
            "val": val,
            "test": test,
        }

    def run_proposal_rolling(
        self,
        proposal: dict[str, Any],
        folds: list[dict[str, tuple[str, str]]],
    ) -> dict[str, Any]:
        """
        滚动多段交叉验证:
        每个 fold 都在自己的训练段选股、验证段/测试段评估,
        最后聚合成"平均验证/平均测试"指标, 消除单一窗口偶然性。
        """
        strategy_params = {
            "fast": int(proposal.get("fast", 5)),
            "slow": int(proposal.get("slow", 20)),
            "long_short": bool(proposal.get("long_short", False)),
            "position_ratio": float(proposal.get("position_ratio", self.app.backtest.position_ratio)),
            "allow_short": bool(proposal.get("allow_short", False)),
        }
        top_k = int(proposal.get("top_k", self.training.top_k))
        select_metric = proposal.get("select_metric", self.training.select_metric)

        fold_results: list[dict[str, Any]] = []
        for i, fold in enumerate(folds, 1):
            train_start, train_end = fold["train"]
            val_start, val_end = fold["val"]
            test_start, test_end = fold["test"]

            scores = self.evaluator.score_symbols(self.pool, strategy_params, train_start, train_end)
            if scores.empty or len(scores) < top_k:
                fold_results.append({"fold": i, "error": f"训练段可用标的不足({len(scores)}/{top_k})"})
                continue
            top_symbols = scores.nlargest(top_k, select_metric)["code"].tolist()

            val = self.evaluator.portfolio_metrics(
                top_symbols, strategy_params, val_start, val_end, f"val_fold{i}"
            )
            test = self.evaluator.portfolio_metrics(
                top_symbols, strategy_params, test_start, test_end, f"test_fold{i}"
            )
            fold_results.append({"fold": i, "symbols": ",".join(top_symbols), "val": val, "test": test})

        valid = [f for f in fold_results if "val" in f]
        if not valid:
            return {**proposal, "error": "所有折的训练段均无足够数据"}

        avg_val_sharpe = round(float(np.mean([f["val"]["sharpe"] for f in valid])), 6)
        avg_test_sharpe = round(float(np.mean([f["test"]["sharpe"] for f in valid])), 6)
        avg_test_return = round(float(np.mean([f["test"]["total_return"] for f in valid])), 6)
        avg_test_mdd = round(float(np.mean([f["test"]["max_drawdown"] for f in valid])), 6)
        avg_coverage = round(float(np.mean([f["val"]["coverage"] for f in valid])), 4)
        positive_test_folds = sum(1 for f in valid if f["test"]["total_return"] > 0)

        return {
            **proposal,
            "n_folds": len(valid),
            "avg_val_sharpe": avg_val_sharpe,
            "avg_test_sharpe": avg_test_sharpe,
            "avg_test_total_return": avg_test_return,
            "avg_test_max_drawdown": avg_test_mdd,
            "avg_coverage": avg_coverage,
            "test_positive_folds": f"{positive_test_folds}/{len(valid)}",
            "fold_results": fold_results,
        }


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    """把一轮结果拍平为一行, 便于写 CSV 日志。"""
    flat: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, dict) and key in ("val", "test"):
            for sub_key, sub_value in value.items():
                flat[f"{key}_{sub_key}"] = sub_value
        elif key != "error":
            flat[key] = value
    return flat


def flatten_rolling_result(result: dict[str, Any]) -> dict[str, Any]:
    """
    把滚动训练的一轮结果拍平为一行:
    聚合指标 + 每个 fold 的 验证/测试 指标(带 fold 编号列)。
    """
    flat: dict[str, Any] = {}
    for key, value in result.items():
        if key == "fold_results":
            for fold in value:
                if "val" not in fold:
                    continue
                i = fold["fold"]
                for sub_key, sub_value in fold["val"].items():
                    flat[f"fold{i}_val_{sub_key}"] = sub_value
                for sub_key, sub_value in fold["test"].items():
                    flat[f"fold{i}_test_{sub_key}"] = sub_value
                flat[f"fold{i}_symbols"] = fold.get("symbols", "")
        elif key != "error":
            flat[key] = value
    return flat


def random_proposals(rounds: int, seed: int = 42) -> list[dict[str, Any]]:
    """随机基线提案(用于测试训练框架; 正式调参由大模型提案驱动)。"""
    rng = np.random.default_rng(seed)
    proposals = []
    for _ in range(rounds):
        fast = int(rng.choice([3, 5, 8, 10, 15]))
        slow = int(rng.choice([20, 30, 40, 60]))
        if fast >= slow:
            fast, slow = 5, 30
        proposals.append(
            {
                "fast": fast,
                "slow": slow,
                "long_short": bool(rng.integers(0, 2)),
                "top_k": int(rng.choice([3, 5, 8])),
            }
        )
    return proposals


def load_proposals(path: Path) -> list[dict[str, Any]]:
    """从 JSON 文件加载提案列表(文件格式: [{"fast":.., "slow":.., ...}, ...])。"""
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if isinstance(data, dict):
        data = [data]
    return data


def save_proposal_template(path: Path) -> None:
    """生成提案模板, 供大模型/人工填写。"""
    template = [
        {"fast": 5, "slow": 20, "long_short": False, "top_k": 5, "select_metric": "sharpe"},
        {"fast": 8, "slow": 30, "long_short": False, "top_k": 5, "select_metric": "sharpe"},
    ]
    with path.open("w", encoding="utf-8") as fp:
        json.dump(template, fp, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------
# 数据集切分与基准对比(供 DeepSeek 训练循环使用)
# ----------------------------------------------------------------------
def classify_yearly_regimes(
    index_df: pd.DataFrame,
    bull_threshold: float = 0.12,
    bear_threshold: float = -0.08,
) -> dict[int, str]:
    """
    按指数年度收益率把年份分为 bull / bear / sideways。
    注: 当前年度为年内至今(YTD)收益。
    """
    regimes: dict[int, str] = {}
    for year, group in index_df.groupby(index_df.index.year)["close"]:
        ret = float(group.iloc[-1] / group.iloc[0] - 1.0)
        if ret >= bull_threshold:
            regimes[int(year)] = "bull"
        elif ret <= bear_threshold:
            regimes[int(year)] = "bear"
        else:
            regimes[int(year)] = "sideways"
    return regimes


def balanced_years_split(
    regimes: dict[int, str],
) -> tuple[list[int], list[int], dict[str, object]]:
    """
    在连续年份中寻找切分点: 训练集与测试集各自尽量同时包含 牛/熊/横盘。
    返回 (train_years, test_years, coverage_info)。
    """
    years = sorted(regimes)
    if len(years) < 2:
        raise ValueError("年份不足, 无法切分")

    def coverage(ys: list[int]) -> dict[str, bool]:
        labels = {regimes[y] for y in ys}
        return {
            "bull": "bull" in labels,
            "bear": "bear" in labels,
            "sideways": "sideways" in labels,
        }

    best: tuple[int, int, dict[str, bool], dict[str, bool]] | None = None
    for b in range(1, len(years)):
        ct, cx = coverage(years[:b]), coverage(years[b:])
        score = sum(ct.values()) + sum(cx.values())
        if best is None or score > best[0]:
            best = (score, b, ct, cx)

    score, b, ct, cx = best  # type: ignore[misc]
    return years[:b], years[b:], {
        "train_coverage": ct,
        "test_coverage": cx,
        "regime_score": score,
        "max_possible": 6,
        "note": "牛/熊/横盘齐全=满分; 若未满分, 说明该切分下某侧缺少某类行情",
    }


def years_to_range(years: list[int]) -> tuple[str, str]:
    """年份列表 -> (start_date, end_date)。2026 为当前年度, 截止到最新交易日。"""
    years = sorted(years)
    start = f"{years[0]}0101"
    end = "20260810" if years[-1] >= 2026 else f"{years[-1]}1231"
    return start, end


def compare_with_benchmark(
    strategy_returns: pd.Series,
    bench_df: pd.DataFrame,
) -> dict[str, float | None]:
    """
    策略 vs 大盘基准(等权组合日收益 vs 指数日收益):
    总收益/年化/夏普/回撤/超额/贝塔/相关性/上涨下跌捕获率。
    用于判断跑赢大盘是"选股+操作"的功劳, 还是单纯吃了市场红利。
    """
    # 日内周期按日聚合, 与日线基准对齐(60min 一天4根 -> 当日收益求和)
    s_idx = pd.to_datetime(strategy_returns.index.date)
    s_daily = strategy_returns.groupby(s_idx).sum()
    bench = bench_df["close"].reindex(s_daily.index).pct_change()
    df = pd.concat([s_daily.rename("strategy"), bench.rename("bench")], axis=1).dropna()
    if len(df) < 20:
        return {"error": f"策略与基准重叠样本不足({len(df)})"}

    s, b = df["strategy"], df["bench"]
    n = len(df)
    strat_total = float((1.0 + s).prod() - 1.0)
    bench_total = float((1.0 + b).prod() - 1.0)

    def annualize(total: float, days: int) -> float:
        if days <= 0 or total <= -1.0:
            return -1.0
        return float((1.0 + total) ** (252 / days) - 1.0)

    def sharpe(r: pd.Series) -> float:
        std = r.std(ddof=1)
        return float(r.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    def mdd(r: pd.Series) -> float:
        eq = (1.0 + r).cumprod()
        return float((eq / eq.cummax() - 1.0).min())

    var_b = float(np.var(b, ddof=1))
    beta = float(np.cov(s, b, ddof=1)[0, 1] / var_b) if var_b > 0 else 0.0
    corr = float(np.corrcoef(s, b)[0, 1]) if n > 2 else 0.0

    up_idx = b > 0
    down_idx = b < 0
    up_capture = (
        float(s[up_idx].mean() / b[up_idx].mean())
        if up_idx.any() and b[up_idx].mean() != 0
        else None
    )
    down_capture = (
        float(s[down_idx].mean() / b[down_idx].mean())
        if down_idx.any() and b[down_idx].mean() != 0
        else None
    )

    return {
        "n_days": n,
        "strategy_total_return": round(strat_total, 6),
        "benchmark_total_return": round(bench_total, 6),
        "excess_total_return": round(strat_total - bench_total, 6),
        "strategy_annual_return": round(annualize(strat_total, n), 6),
        "benchmark_annual_return": round(annualize(bench_total, n), 6),
        "strategy_sharpe": round(sharpe(s), 4),
        "benchmark_sharpe": round(sharpe(b), 4),
        "strategy_max_drawdown": round(mdd(s), 6),
        "benchmark_max_drawdown": round(mdd(b), 6),
        "beta": round(beta, 4),
        "correlation": round(corr, 4),
        "up_capture": round(up_capture, 4) if up_capture is not None else None,
        "down_capture": round(down_capture, 4) if down_capture is not None else None,
        "positive_days_ratio": round(float((s > 0).mean()), 4),
    }
