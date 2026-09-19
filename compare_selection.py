"""
对比两种选股方式(其余参数完全相同, 只换选股)

    A) 静态选股(现状): 用整段训练窗口(2020-2022)在池内排序选出 top_k, 之后焊死不变
    B) 滚动选股: 每隔 REBAL 个交易日, 用"当时可得的过去 LOOKBACK 日"重新排序选 top_k

两边的策略参数、数据窗口、组合口径完全一致, 差异只来自"什么时候用哪些数据选股"。
重点看 val(2023) 与 test(2024-2026) —— 这两段都不是选股用的数据。

用法:
    python compare_selection.py --scheme cn_full
    python compare_selection.py --all
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from qtcore.config import AppConfig
from qtcore.scheme_eval import eval_symbol
from train_schemes import load_pool

ROOT = Path(__file__).resolve().parent
SCHEMES_DIR = ROOT / "output" / "schemes"

FULL_START = "20200101"
FULL_END = "20260810"
TRAIN_END = "2022-12-31"
VAL_RANGE = ("2023-01-01", "2023-12-31")
TEST_START = "2024-01-01"

LOOKBACK = 250      # 滚动选股的回看窗口(交易日, 约 1 年)
REBAL = 21          # 滚动选股的调仓间隔(交易日, 约 1 个月)
MIN_HISTORY = 60    # 至少要有这么多天数据才参与排序


def sharpe(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < MIN_HISTORY or s.std(ddof=1) == 0:
        return float("-inf")
    return float(s.mean() / s.std(ddof=1) * np.sqrt(252))


def stats_of(series: pd.Series) -> dict[str, float]:
    s = series.dropna()
    if s.empty:
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    equity = (1.0 + s.fillna(0.0)).cumprod()
    dd = equity / equity.cummax() - 1.0
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "sharpe": sharpe(s),
        "max_drawdown": float(dd.min()),
    }


def build_returns(scheme: dict, pool: list[str], workers: int) -> pd.DataFrame:
    params = dict(scheme)
    jobs = [(c, params, FULL_START, FULL_END) for c in pool]
    data: dict[str, pd.Series] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        chunk = max(1, len(jobs) // max(1, workers * 2))
        for code, ret, _ in ex.map(eval_symbol, jobs, chunksize=chunk):
            if ret is not None:
                data[code] = ret
    return pd.DataFrame(data).sort_index()


def static_top(df: pd.DataFrame, top_k: int) -> list[str]:
    """现状做法: 用整段训练窗口(2020-2022)的策略夏普排序。"""
    train = df.loc[:TRAIN_END]
    scores = {c: sharpe(train[c]) for c in df.columns}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [c for c, s in ranked if s != float("-inf")][:top_k]


def rolling_portfolio(df: pd.DataFrame, top_k: int, start: str) -> tuple[pd.Series, list]:
    """滚动选股: 每隔 REBAL 日用截止当日的过去 LOOKBACK 日重新排序。"""
    dates = [d for d in df.index if d.strftime("%Y-%m-%d") >= start]
    selected: list[tuple[str, list[str]]] = []
    current: list[str] = []
    out: list[float] = []
    for i, d in enumerate(dates):
        if i % REBAL == 0 or not current:
            hist = df.loc[:d].iloc[-LOOKBACK:]
            scores = {c: sharpe(hist[c]) for c in df.columns}
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            current = [c for c, s in ranked if s != float("-inf")][:top_k]
            selected.append((d.strftime("%Y-%m-%d"), list(current)))
        row = df.loc[d, current]
        out.append(float(row.mean()) if len(row) else 0.0)
    return pd.Series(out, index=dates), selected


def compare(
    name: str, scheme: dict, pool: list[str], workers: int, n_draws: int = 40
) -> dict:
    top_k = int(scheme.get("top_k", 5))
    df = build_returns(scheme, pool, workers)
    if df.empty:
        print(f"[{name}] 无可用标的")
        return {}

    val_a, val_b = VAL_RANGE
    static_sel = static_top(df, top_k)
    static_ret = df[static_sel].mean(axis=1)

    roll_ret, roll_sel = rolling_portfolio(df, top_k, val_a)

    def seg(series: pd.Series, lo: str, hi: str | None) -> dict[str, float]:
        s = series.loc[(series.index.strftime("%Y-%m-%d") >= lo)]
        if hi:
            s = s.loc[s.index.strftime("%Y-%m-%d") <= hi]
        return stats_of(s)

    print(f"\n===== {name}  (池 {len(df.columns)} 只, top_k={top_k}) =====")
    print(f"  静态选股标的: {static_sel}")
    print(f"  滚动换手: {len(set(tuple(s) for _, s in roll_sel))} 个不同组合, 共调仓 {len(roll_sel)} 次")
    print(f"  滚动最后一次选中: {roll_sel[-1][1]}")

    # 第三个对照: 完全不选股 —— 整个池子等权持有
    pool_ret = df.mean(axis=1)

    # 第四个对照: 随机选 top_k(多次抽样, 看静态选股在随机分布里的位置)
    rng = random.Random(0)
    cols = list(df.columns)
    rand_series = [
        df[rng.sample(cols, min(top_k, len(cols)))].mean(axis=1)
        for _ in range(n_draws)
    ]

    rows = []
    for label, lo, hi in (("验证集 2023", val_a, val_b), ("测试集 2024-2026", TEST_START, None)):
        a = seg(static_ret, lo, hi)
        b = seg(roll_ret, lo, hi)
        c = seg(pool_ret, lo, hi)
        draws = [seg(s, lo, hi) for s in rand_series]
        dr = np.array([d["total_return"] for d in draws])
        ds = np.array([d["sharpe"] for d in draws])
        dd = np.array([d["max_drawdown"] for d in draws])
        pct = float((dr < a["total_return"]).mean() * 100)
        rows.append((label, a, b))
        print(
            f"  {label:<18} 静态: 收益 {a['total_return']:>8.2%} 夏普 {a['sharpe']:>6.2f} 回撤 {a['max_drawdown']:>7.2%}"
            f"  |  滚动: 收益 {b['total_return']:>8.2%} 夏普 {b['sharpe']:>6.2f} 回撤 {b['max_drawdown']:>7.2%}"
        )
        print(
            f"  {'':<18} 不选股(全池等权): 收益 {c['total_return']:>8.2%} 夏普 {c['sharpe']:>6.2f} 回撤 {c['max_drawdown']:>7.2%}"
        )
        print(
            f"  {'':<18} 随机选{top_k}({n_draws}次): 收益中位 {np.median(dr):>8.2%}"
            f"  均值 {dr.mean():>8.2%}  最差 {dr.min():>8.2%}  最好 {dr.max():>8.2%}"
            f"  夏普均值 {ds.mean():>5.2f}  回撤均值 {dd.mean():>7.2%}"
        )
        print(
            f"  {'':<18} >>> 静态选股在随机分布里排第 {pct:>5.1f} 个百分点"
            + ("  (低于中位数 = 选股在帮倒忙)" if pct < 50 else "  (高于中位数 = 选股有用)")
        )
    return {"name": name, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="cn_full")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--draws", type=int, default=40, help="随机选股的抽样次数")
    args = parser.parse_args()

    app = AppConfig()
    names = (
        [p.stem for p in sorted(SCHEMES_DIR.glob("*.json"))]
        if args.all
        else [args.scheme]
    )
    for name in names:
        path = SCHEMES_DIR / f"{name}.json"
        if not path.exists():
            print(f"[{name}] 配置不存在, 跳过")
            continue
        scheme = json.loads(path.read_text(encoding="utf-8"))
        market = str(scheme.get("market", "cn"))
        pool = load_pool(market, app)
        compare(name, scheme, pool, args.workers, args.draws)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
