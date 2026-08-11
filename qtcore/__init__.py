"""
QuantitativeTrading 核心包
==========================

模块化架构:
    DataCenter    -> 数据获取/清洗/标准化
    StrategyEngine-> 因子计算与信号生成
    BacktestEngine-> 账户管理与模拟撮合
    MainManager   -> 工作流串联与入口
"""

__version__ = "0.1.0"
