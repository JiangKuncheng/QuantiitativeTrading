"""
全市场选股样本外验证(无选择偏差)
=================================

方法:
    选择窗口: 2024-01-01 ~ 2024-12-31(全市场 5/20 双均线按夏普排名)
    测试窗口: 2025-01-01 ~ 2025-12-31(与选股窗口完全无重叠)

目的:
    验证"全市场选股"是否真的有样本外区分度, 而不是靠选择偏差刷数字。
    若 Top-K 在 2025 的表现显著好于全市场平均, 说明选股有信号;
    否则说明高排名只是 2024 的过拟合。

数据: 全部来自本地缓存(data/cache, 全市场扫描时落盘), 无需联网。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from qtcore.backtest.engine import BacktestEngine
from qtcore.config import AppConfig
from qtcore.strategy import create_strategy


CACHE = Path("data/cache")
SELECT_START, SELECT_END = "2024-01-01", "2024-12-31"
TEST_START, TEST_END = "2025-01-01", "2025-12-31"


def run_one(path: Path) -> dict | None:
    code = path.name.split("_")[1]
    df = pd.read_parquet(path)
    # 防御: 缓存索引统一还原为日期索引
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    sel = df[(df.index >= SELECT_START) & (df.index <= SELECT_END)]
    test = df[(df.index >= TEST_START) & (df.index <= TEST_END)]
    if len(sel) < 120 or len(test) < 120:
        return None
    strategy = create_strategy("ma_cross", {"fast": 5, "slow": 20})
    engine = BacktestEngine(AppConfig().backtest)
    s = engine.run(sel, strategy).stats
    t = engine.run(test, strategy).stats
    return {
        "code": code,
        "sel_sharpe": s["sharpe"],
        "sel_return": s["total_return"],
        "test_return": t["total_return"],
        "test_sharpe": t["sharpe"],
        "test_max_drawdown": t["max_drawdown"],
    }


def main() -> int:
    files = sorted(CACHE.glob("daily_*_20240101_20251231.parquet"))
    print(f"[Validation] 全市场缓存标的: {len(files)}")

    rows = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(run_one, f) for f in files]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if row is not None:
                rows.append(row)
            if i % 1000 == 0:
                print(f"[Validation] 进度 {i}/{len(files)}")

    df = pd.DataFrame(rows)
    df = df.sort_values("sel_sharpe", ascending=False).reset_index(drop=True)
    out_path = Path("output") / "fullmarket_walkforward.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    n = len(df)
    market_avg_2025 = df["test_return"].mean()
    market_med_2025 = df["test_return"].median()
    print(f"[Validation] 有效标的 {n} 只")
    print(f"全市场 2025 平均收益: {market_avg_2025:.2%} | 中位数: {market_med_2025:.2%}")

    for top_k in (5, 10, 20, 50):
        top = df.head(top_k)
        positive = int((top["test_return"] > 0).sum())
        print(
            f"Top{top_k:<3} 2025 平均收益 {top['test_return'].mean():.2%} | "
            f"中位 {top['test_return'].median():.2%} | "
            f"平均夏普 {top['test_sharpe'].mean():.3f} | "
            f"正收益 {positive}/{top_k}"
        )

    print(f"\nTop-20 明细已保存: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
