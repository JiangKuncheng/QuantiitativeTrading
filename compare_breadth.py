"""
选股宽度 x 选股依据 对照

问的问题: "不选股"之所以赢, 是因为"分散"(宽度) 还是因为"不按历史收益选"(依据)?

对照组合(策略参数完全相同, 只换选股规则):
    按历史夏普选 5 / 10 / 30 / 60 只   <- 只加宽, 依据不变
    按流动性选 30 / 60 只              <- 换依据(与收益无关)
    全池 150 只等权                    <- 宽度拉满

用法: python compare_breadth.py --scheme cn_full
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from qtcore.config import AppConfig
from qtcore.scheme_eval import eval_symbol

ROOT = Path(__file__).resolve().parent
SCHEMES_DIR = ROOT / "output" / "schemes"
FULL_START, FULL_END = "20200101", "20260810"
TRAIN_END = "2022-12-31"
VAL = ("2023-01-01", "2023-12-31")
TEST_FROM = "2024-01-01"
MIN_HISTORY = 60


def shp(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < MIN_HISTORY or s.std(ddof=1) == 0:
        return float("-inf")
    return float(s.mean() / s.std(ddof=1) * np.sqrt(252))


def seg_stats(series: pd.Series, lo: str, hi: str | None) -> tuple[float, float, float]:
    s = series.loc[series.index.strftime("%Y-%m-%d") >= lo]
    if hi:
        s = s.loc[s.index.strftime("%Y-%m-%d") <= hi]
    s = s.dropna()
    if s.empty:
        return 0.0, 0.0, 0.0
    eq = (1.0 + s.fillna(0.0)).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    return float(eq.iloc[-1] - 1.0), shp(s), float(dd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="cn_full")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    scheme = json.loads((SCHEMES_DIR / f"{args.scheme}.json").read_text(encoding="utf-8"))
    market = str(scheme.get("market", "cn"))
    pool_file = ROOT / "data" / f"pool_{market}_real.csv"
    pool_df = pd.read_csv(pool_file, dtype=str)
    pool = [str(c) for c in pool_df["code"]]

    # 逐标的回测一次(整段窗口), 后面所有变体都复用这份收益矩阵
    jobs = [(c, dict(scheme), FULL_START, FULL_END) for c in pool]
    data: dict[str, pd.Series] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for code, ret, _ in ex.map(eval_symbol, jobs, chunksize=4):
            if ret is not None:
                data[code] = ret
    df = pd.DataFrame(data).sort_index()
    print(f"{args.scheme}: 有效标的 {len(df.columns)} / 池 {len(pool)}")

    # 依据一: 历史策略夏普(训练窗口)
    train = df.loc[:TRAIN_END]
    perf_rank = sorted(df.columns, key=lambda c: shp(train[c]), reverse=True)
    # 依据二: 流动性(pool 文件里的 avg_amount, 与收益无关)
    liq = pool_df.assign(a=pool_df["avg_amount"].astype(float)).sort_values("a", ascending=False)
    liq_rank = [str(c) for c in liq["code"] if str(c) in set(df.columns)]

    variants: list[tuple[str, list[str]]] = [
        ("按历史夏普选 5 (现状)", perf_rank[:5]),
        ("按历史夏普选 10", perf_rank[:10]),
        ("按历史夏普选 30", perf_rank[:30]),
        ("按历史夏普选 60", perf_rank[:60]),
        ("按流动性选 30", liq_rank[:30]),
        ("按流动性选 60", liq_rank[:60]),
        ("全池等权", list(df.columns)),
    ]

    print(f"\n{'选股规则':<24}{'val2023':>10}{'夏普':>7}{'回撤':>9}   {'test2024-26':>12}{'夏普':>7}{'回撤':>9}")
    print("-" * 82)
    for label, codes in variants:
        if not codes:
            continue
        port = df[codes].mean(axis=1)
        vr, vs, vd = seg_stats(port, *VAL)
        tr, ts, td = seg_stats(port, TEST_FROM, None)
        print(
            f"{label:<24}{vr:>10.2%}{vs:>7.2f}{vd:>9.2%}   {tr:>12.2%}{ts:>7.2f}{td:>9.2%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
