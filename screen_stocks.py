"""
选股器命令行入口
================

两种选股模式:
1. 全市场: 初筛(排除ST/价格/成交额) -> 按成交额取前 N 只 -> 双均线回测排序
2. 指定池: --symbols "000001 600519 ..." 只在这几只里挑

用法:
    python screen_stocks.py --limit 20 --start 20240101 --end 20251231
    python screen_stocks.py --symbols "000001 600519 300750" --top 5
    python screen_stocks.py --offline                      # 合成数据演示
"""

from __future__ import annotations

import argparse

from qtcore.config import AppConfig
from qtcore.screener import StockScreener, format_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="screen_stocks", description="A 股选股器")
    parser.add_argument("--symbols", default=None, help="指定股票池, 空格分隔, 如 '000001 600519'")
    parser.add_argument("--start", default="20240101", help="回测起始日期 YYYYMMDD")
    parser.add_argument("--end", default="20251231", help="回测结束日期 YYYYMMDD")
    parser.add_argument("--fast", type=int, default=5, help="双均线快线")
    parser.add_argument("--slow", type=int, default=20, help="双均线慢线")
    parser.add_argument("--min-price", type=float, default=2.0, help="初筛最低价")
    parser.add_argument("--max-price", type=float, default=200.0, help="初筛最高价")
    parser.add_argument("--min-amount", type=float, default=2e8, help="初筛最低成交额(元)")
    parser.add_argument("--limit", type=int, default=30, help="参与精选的候选数量")
    parser.add_argument("--metric", default="sharpe", help="排序指标")
    parser.add_argument("--top", type=int, default=10, help="终端展示前 N 名")
    parser.add_argument("--offline", action="store_true", help="合成数据演示模式")
    parser.add_argument("--no-cache", action="store_true", help="禁用行情缓存")
    parser.add_argument("--universe", choices=["all", "csi300"], default="all",
                        help="股票池来源: all 全市场 / csi300 沪深300成分股")
    parser.add_argument("--workers", type=int, default=4, help="并发拉取线程数(全市场建议 8~12)")
    parser.add_argument("--retries", type=int, default=3, help="网络拉取统一重试次数")
    parser.add_argument("--no-snapshot", action="store_true",
                        help="跳过东财快照接口, 直接用全市场代码列表初筛(东财被拦/崩溃时的安全选项)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg = AppConfig()
    cfg.data.start_date = args.start
    cfg.data.end_date = args.end
    cfg.data.use_cache = not args.no_cache
    cfg.data.fetch_retries = args.retries

    screener = StockScreener(
        cfg,
        fast=args.fast,
        slow=args.slow,
        synthetic=args.offline,
        universe=args.universe,
        use_snapshot=not args.no_snapshot,
    )

    # 模式一: 指定股票池(只在这几只里挑)
    if args.symbols:
        symbols = args.symbols.split()
        candidates = screener.from_symbols(symbols)
        print(f"[Screener] 指定股票池模式: {len(symbols)} 只 -> {symbols}")
    # 模式二: 全市场初筛
    else:
        candidates = screener.filter_universe(
            min_price=args.min_price,
            max_price=args.max_price,
            min_amount=args.min_amount,
        )

    result = screener.rank(
        candidates,
        limit=args.limit,
        metric=args.metric,
        workers=args.workers,
    )
    if result.empty:
        return 1

    out_path = cfg.paths.output_dir / "screen_result.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n全部候选已保存: {out_path}\n")

    print(f"===== 选股结果 Top {args.top} (指标: {args.metric}) =====")
    for i, row in result.head(args.top).iterrows():
        print(f"#{i + 1:>2} {format_row(row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
