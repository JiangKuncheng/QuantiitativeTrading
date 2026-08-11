"""回测/执行模块: 模拟账户、撮合引擎与绩效统计。"""

from qtcore.backtest.account import Account, Position
from qtcore.backtest.engine import BacktestEngine, BacktestResult

__all__ = ["Account", "Position", "BacktestEngine", "BacktestResult"]
