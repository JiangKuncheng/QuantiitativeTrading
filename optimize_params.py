"""
参数优化器: 网格搜索 + Walk-Forward 样本外验证
================================================

用于回答"双均线策略能不能通过调参调优":
- 网格搜索 (fast, slow) 参数组合, 每组参数走完整回测(含佣金/滑点);
- 默认按夏普比率排序输出全部组合到 output/param_search.csv;
- --walk-forward 模式下: 前 60% 数据选最优参数, 后 40% 数据做样本外验证,
  用于评估"调出来的最优参数"是否真的稳健, 还是只是拟合了历史行情。

用法:
    python optimize_params.py
    python optimize_params.py --symbol 600519 --fasts "3 5 8 10 15" --slows "20 30 40 60"
    python optimize_params.py --metric total_return --top 10
    python optimize_params.py --walk-forward --train-ratio 0.6
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from qtcore.backtest.engine import BacktestEngine
from qtcore.config import AppConfig
from qtcore.datacenter.data_center import DataCenter
from qtcore.strategy import create_strategy


PERCENT_KEYS = {"total_return", "annual_return", "max_drawdown", "win_rate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="optimize_params",
        description="双均线策略参数网格搜索 + Walk-Forward 验证",
    )
    parser.add_argument("--symbol", default="000001", help="标的代码")
    parser.add_argument("--start", default="20230101", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default="20251231", help="结束日期 YYYYMMDD")
    parser.add_argument("--fasts", default="3 5 8 10 15", help="快线窗口列表(空格分隔)")
    parser.add_argument("--slows", default="20 30 40 60", help="慢线窗口列表(空格分隔)")
    parser.add_argument("--metric", default="sharpe", help="排序指标: sharpe/total_return/profit_factor")
    parser.add_argument("--top", type=int, default=10, help="终端展示前 N 组")
    parser.add_argument("--walk-forward", action="store_true", help="启用样本外验证")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="训练集占比(0~1)")
    return parser.parse_args()


def run_one(bars: pd.DataFrame, cfg: AppConfig, fast: int, slow: int) -> dict[str, Any]:
    """用一组参数跑完整回测, 返回绩效指标 + 参数。"""
    strategy = create_strategy("ma_cross", {"fast": fast, "slow": slow})
    engine = BacktestEngine(cfg.backtest)
    result = engine.run(bars, strategy)
    return {
        **result.stats,
        "fast": fast,
        "slow": slow,
        "n_bars": len(bars),
    }


def grid_search(bars: pd.DataFrame, cfg: AppConfig, fasts: list[int], slows: list[int]) -> pd.DataFrame:
    """遍历 (fast, slow) 组合, 要求 fast < slow。"""
    rows = [
        run_one(bars, cfg, fast, slow)
        for fast in fasts
        for slow in slows
        if fast < slow
    ]
    return pd.DataFrame(rows)


def format_row(row: pd.Series) -> str:
    """格式化一行指标用于终端展示。"""
    parts = [f"fast={int(row['fast']):>2} slow={int(row['slow']):>2}"]
    for key in ("total_return", "annual_return", "sharpe", "max_drawdown", "win_rate", "profit_factor", "n_trades"):
        value = row[key]
        if key in PERCENT_KEYS:
            parts.append(f"{key}={value:.2%}")
        elif key == "sharpe":
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def main() -> int:
    args = parse_args()
    fasts = [int(v) for v in args.fasts.split()]
    slows = [int(v) for v in args.slows.split()]

    cfg = AppConfig()
    cfg.data.symbol = args.symbol
    cfg.data.start_date = args.start
    cfg.data.end_date = args.end

    # 数据走 DataCenter(命中本地 Parquet 缓存, 无需联网)
    dc = DataCenter(cfg.data, cfg.paths)
    bars = dc.get_daily_bars()
    if bars is None or bars.empty:
        raise RuntimeError("行情为空, 无法优化")

    print(f"\n标的 {args.symbol} | {len(bars)} 根K线 | 网格 {len(fasts)}x{len(slows)}"
          f" | 指标 {args.metric} | 参数组合 {len([(f, s) for f in fasts for s in slows if f < s])} 组")

    result_df = grid_search(bars, cfg, fasts, slows)
    result_df = result_df.sort_values(args.metric, ascending=False).reset_index(drop=True)

    out_path = cfg.paths.output_dir / "param_search.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"全部组合已保存: {out_path}\n")

    print(f"===== 全样本(含样本内) Top {args.top} =====")
    for i, row in result_df.head(args.top).iterrows():
        print(f"#{i + 1:>2} {format_row(row)}")

    # Walk-Forward: 前段选参, 后段验证(检验过拟合)
    if args.walk_forward:
        split = int(len(bars) * args.train_ratio)
        train_bars = bars.iloc[:split]
        test_bars = bars.iloc[split:]
        if len(test_bars) < 60:
            raise RuntimeError("测试集样本太少, 请减小 train_ratio 或拉长数据区间")

        train_df = grid_search(train_bars, cfg, fasts, slows).sort_values(args.metric, ascending=False)
        best = train_df.iloc[0]
        test_stats = run_one(test_bars, cfg, int(best["fast"]), int(best["slow"]))

        print(f"\n===== Walk-Forward (训练集 {args.train_ratio:.0%} / 测试集 {1 - args.train_ratio:.0%}) =====")
        print(f"训练集最优: {format_row(best)}")
        print(f"同一参数在样本外: {format_row(pd.Series(test_stats))}")

        # 简单过拟合度量: 样本外收益是否仍为正/是否显著衰减
        train_ret = best["total_return"]
        test_ret = test_stats["total_return"]
        print(f"样本内收益 {train_ret:.2%} -> 样本外收益 {test_ret:.2%} "
              f"({'保留' if test_ret > 0 else '失效'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
