"""
MainManager 主控/工作流模块
===========================

职责:
1. 环境初始化: 创建目录、加载统一配置;
2. 串联工作流: DataCenter -> StrategyEngine -> BacktestEngine -> 结果落盘;
3. 结果汇总展示, 为后续微服务 API / 定时任务 / 消息驱动提供统一入口。

运行方式:
    python -m qtcore                    # 等价于 python -m qtcore.main_manager
    python run_demo.py --offline        # 离线演示
    python run_demo.py --symbol 600519 --fast 10 --slow 30
"""

from __future__ import annotations

import argparse

from qtcore.backtest.engine import BacktestEngine, BacktestResult
from qtcore.config import AppConfig
from qtcore.datacenter.data_center import DataCenter
from qtcore.strategy import create_strategy


class MainManager:
    """主控: 编排数据 -> 策略 -> 回测 -> 输出 的完整工作流。"""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.config.ensure_paths()

    def run(self) -> BacktestResult:
        """
        执行完整工作流, 返回回测结果。

        工作流(每个步骤都可以被替换成独立微服务):
            1. DataCenter    提供统一行情
            2. StrategyEngine提供目标仓位/信号
            3. BacktestEngine模拟撮合并统计绩效
            4. 结果落盘到 output/ 目录
        """
        # 1) 数据层
        data_center = DataCenter(self.config.data, self.config.paths)
        bars = data_center.get_daily_bars()
        if bars is None or bars.empty:
            raise RuntimeError("行情数据为空, 请检查数据源或改用 --offline 离线演示")

        # 2) 策略层
        strategy = create_strategy(self.config.strategy.name, self.config.strategy.params)

        # 3) 回测/执行层
        engine = BacktestEngine(self.config.backtest)
        result = engine.run(bars, strategy)

        # 4) 结果落盘与展示
        saved = result.save(self.config.paths.output_dir)
        if self.config.verbose:
            self._print_summary(bars, strategy, result, saved)
        return result

    # ------------------------------------------------------------------
    # 展示
    # ------------------------------------------------------------------
    @staticmethod
    def _print_summary(
        bars: object,
        strategy: object,
        result: BacktestResult,
        saved: dict[str, object],
    ) -> None:
        """终端打印回测摘要。"""
        stats = result.stats
        sep = "=" * 68
        print(f"\n{sep}")
        print("回测完成")
        print(f"{sep}")
        print(f"标的     : {bars.attrs.get('code', 'DEMO')}   K线数: {len(bars)}")
        print(f"策略     : {strategy}")
        print(f"初始资金 : {stats['initial_capital']:,.2f}")
        print(f"期末权益 : {stats['final_equity']:,.2f}")
        print(f"总收益率 : {stats['total_return']:.2%}")
        print(f"年化收益 : {stats['annual_return']:.2%}")
        print(f"夏普比率 : {stats['sharpe']:.3f}")
        print(f"最大回撤 : {stats['max_drawdown']:.2%}")
        print(f"交易次数 : {stats['n_trades']}  (胜率 {stats['win_rate']:.2%})")
        print(f"盈亏比   : {stats['profit_factor']:.2f}")
        print(f"信号事件 : {len(result.signal_events)} 个")
        print(f"\n输出文件:")
        for label, path in saved.items():
            print(f"  {label:12s}: {path}")
        print(sep)


# ----------------------------------------------------------------------
# 命令行入口
# ----------------------------------------------------------------------
def build_cli_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="QuantitativeTrading",
        description="量化交易模型核心骨架: DataCenter -> Strategy -> Backtest",
    )
    parser.add_argument("--symbol", default=None, help="标的代码, 如 000001 / 600519")
    parser.add_argument("--start", default=None, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default=None, help="结束日期 YYYYMMDD")
    parser.add_argument("--adjust", default=None, help="复权方式: qfq/hfq/none")
    parser.add_argument("--strategy", default=None, help="策略注册名(默认 ma_cross)")
    parser.add_argument("--fast", type=int, default=None, help="快线窗口")
    parser.add_argument("--slow", type=int, default=None, help="慢线窗口")
    parser.add_argument("--long-short", action="store_true", help="开启多空双向")
    parser.add_argument("--capital", type=float, default=None, help="初始资金")
    parser.add_argument("--offline", action="store_true", help="强制离线合成数据")
    parser.add_argument("--no-cache", action="store_true", help="禁用行情缓存")
    parser.add_argument("--quiet", action="store_true", help="不打印摘要")
    return parser


def main(argv: list[str] | None = None) -> int:
    """程序入口。"""
    args = build_cli_parser().parse_args(argv)

    config = AppConfig()
    if args.symbol:
        config.data.symbol = args.symbol
    if args.start:
        config.data.start_date = args.start
    if args.end:
        config.data.end_date = args.end
    if args.adjust:
        config.data.adjust = args.adjust
    if args.offline:
        config.data.offline_fallback = True
    if args.no_cache:
        config.data.use_cache = False

    if args.strategy:
        config.strategy.name = args.strategy
    if args.fast is not None:
        config.strategy.params["fast"] = args.fast
    if args.slow is not None:
        config.strategy.params["slow"] = args.slow
    if args.long_short:
        config.strategy.params["long_short"] = True

    if args.capital is not None:
        config.backtest.initial_capital = args.capital
    config.verbose = not args.quiet

    manager = MainManager(config)
    manager.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
