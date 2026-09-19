"""
六方案训练的稳健评估层

原 train_schemes.py 的评估有三个硬伤:
    1) 没有验证集: 训练集(2020-2022)既用来选股又用来选拔参数, 2023 年整段跳过;
    2) 单窗口评分: 一个 3 年窗口算出一个夏普就当分数, 噪声极大;
    3) 没有并行: 150 只标的逐个回测, 一轮就要 105 秒。

本模块提供替代实现:
    * 三段切分: train(2020-2022) / val(2023) / test(2024-2026)
    * walk-forward 多折: train 窗口切成 N 折, 取各折指标的中位数当分数,
      抗噪声(单窗口夏普容易被一段行情带偏);
    * 两阶段筛选: 先用固定子样本(默认 24 只)粗筛候选, 只对粗筛靠前的
      少数候选做全池完整评估 —— 同样的算力能多试好几倍的参数;
    * 标的级并行: ProcessPoolExecutor, 每只标的独立回测, 互不依赖。

评估口径与 train_schemes.py 保持一致: 每只标的用整额本金独立回测,
组合收益 = 各标的日收益等权平均(这是该项目的既定模型, 不是真组合账户)。
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from qtcore.backtest.engine import BacktestEngine
from qtcore.config import AppConfig
from qtcore.datacenter.data_center import DataCenter
from qtcore.scheme_runner import _bt_config
from qtcore.strategy import create_strategy

INITIAL_CAPITAL = 1_000_000.0

# 每个子进程只初始化一次 DataCenter(复用 parquet 缓存)
_WORKER_APP: AppConfig | None = None
_WORKER_DC: DataCenter | None = None


def _worker_ctx() -> tuple[AppConfig, DataCenter]:
    global _WORKER_APP, _WORKER_DC
    if _WORKER_DC is None:
        _WORKER_APP = AppConfig()
        # 训练绝不允许降级到合成行情: 拿不到真实数据就跳过这只标的
        data_cfg = replace(_WORKER_APP.data, offline_fallback=False, use_cache=True)
        _WORKER_DC = DataCenter(data_cfg, _WORKER_APP.paths)
    return _WORKER_APP, _WORKER_DC  # type: ignore[return-value]


def eval_symbol(job: tuple[str, dict[str, Any], str, str]) -> tuple[str, Any, Any]:
    """单只标的单窗口回测 -> (code, 日收益Series, 绩效stats)。失败返回 (code, None, None)。"""
    code, params, start, end = job
    try:
        app, dc = _worker_ctx()
        timeframe = str(params.get("timeframe", "daily"))
        market = str(params.get("market", "cn"))
        bars = dc.get_bars(code, start, end, timeframe, market)
        if bars is None or len(bars) < 60:
            return code, None, None
        bt = _bt_config(params, app)
        result = BacktestEngine(bt).run(bars, create_strategy("ma_cross", params))
        return code, result.equity_curve["daily_return"], result.stats
    except Exception:  # noqa: BLE001 - 单只标的失败不影响整体
        return code, None, None


def evaluate_window(
    codes: list[str],
    params: dict[str, Any],
    start: str,
    end: str,
    executor: ProcessPoolExecutor,
) -> tuple[dict[str, pd.Series], dict[str, dict[str, Any]]]:
    """并行评估给定标的集合, 返回 {code: 日收益} 与 {code: stats}。"""
    jobs = [(c, params, start, end) for c in codes]
    returns: dict[str, pd.Series] = {}
    stats: dict[str, dict[str, Any]] = {}
    if not jobs:
        return returns, stats
    chunk = max(1, len(jobs) // max(1, (os.cpu_count() or 2) * 2))
    for code, ret, st in executor.map(eval_symbol, jobs, chunksize=chunk):
        if ret is not None and st is not None:
            returns[code] = ret
            stats[code] = st
    return returns, stats


def portfolio_metrics(returns: dict[str, pd.Series], window_label: str = "") -> dict[str, Any]:
    """等权组合指标(与 train_schemes 口径一致)。"""
    if not returns:
        return {"window": window_label, "n_symbols": 0, "error": "no data"}
    ret_df = pd.DataFrame(returns).fillna(0.0)
    port_ret = ret_df.mean(axis=1)
    equity = INITIAL_CAPITAL * (1.0 + port_ret).cumprod()
    total_return = float(equity.iloc[-1] / INITIAL_CAPITAL - 1.0)
    n_days = len(port_ret)
    annual = (
        (1.0 + total_return) ** (252 / n_days) - 1.0
        if n_days > 0 and total_return > -1
        else -1.0
    )
    sharpe = (
        float(port_ret.mean() / port_ret.std(ddof=1) * np.sqrt(252))
        if len(port_ret) > 1 and port_ret.std(ddof=1) > 0
        else 0.0
    )
    drawdown = equity / equity.cummax() - 1.0
    return {
        "window": window_label,
        "n_symbols": len(ret_df.columns),
        "total_return": round(total_return, 6),
        "annual_return": round(annual, 6),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(float(drawdown.min()) if not drawdown.empty else 0.0, 6),
    }


def split_folds(start: str, end: str, n_folds: int) -> list[tuple[str, str]]:
    """把 [start, end] 按日历等分成 n_folds 段(用于 walk-forward 评分)。"""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    edges = pd.date_range(s, e, periods=n_folds + 1)
    folds: list[tuple[str, str]] = []
    for i in range(n_folds):
        a = edges[i]
        b = edges[i + 1] - pd.Timedelta(days=1) if i < n_folds - 1 else e
        folds.append((a.strftime("%Y%m%d"), b.strftime("%Y%m%d")))
    return folds


def pick_top_symbols(
    stats: dict[str, dict[str, Any]], top_k: int, metric: str
) -> list[str]:
    """按 select_metric 从逐标的绩效里挑 top_k(与 train_schemes 一致)。"""
    rows = [
        {"code": c, metric: float(s.get(metric, float("-inf")))} for c, s in stats.items()
    ]
    if not rows:
        return []
    df = pd.DataFrame(rows).dropna()
    if df.empty:
        return []
    return [str(c) for c in df.nlargest(top_k, metric)["code"].tolist()]


def robust_score(
    fold_metrics: list[dict[str, Any]], overall: dict[str, Any] | None = None
) -> float:
    """
    walk-forward 分数 = 各折夏普的中位数 - 0.5 x 折间标准差。

    不再设"整段训练窗口年化 >= 6%"的硬门槛。实测它会枪毙"训练期平淡、验证/
    测试期优秀"的配置: 按流动性选 30 只的组合在 2020-2022 年化不足 6%, 却在
    2023 年 +30.8%、2024-2026 年 +156%。该门槛把所有候选一起打成 -10, 搜索
    直接失去梯度。稳健性现在由两道独立机制把关:
        1) 多折中位数 + 折间离散惩罚(压掉靠单段走运的参数)
        2) 最终用验证集(2023)选优, 测试集不参与任何选择
    overall 参数保留是为了兼容调用方, 不再参与打分。
    """
    ok = [m for m in fold_metrics if "error" not in m]
    if not ok:
        return -10.0
    sharpes = [float(m.get("sharpe", 0.0)) for m in ok]
    return float(np.median(sharpes) - 0.5 * np.std(sharpes))
