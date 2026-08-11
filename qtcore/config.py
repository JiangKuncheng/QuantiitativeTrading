"""
全局配置模块
============

设计原则:
1. 所有路径统一通过 pathlib.Path 管理, 杜绝裸字符串拼接;
2. 所有运行参数集中到 dataclass, 便于后续对接配置中心/环境变量/命令行;
3. 配置对象可以整体序列化, 为 Docker 微服务化部署预留接口。

配置层次:
    AppConfig
    ├── ProjectPaths   项目/数据/缓存/输出目录
    ├── DataConfig     数据模块参数
    ├── StrategyConfig 策略模块参数
    └── BacktestConfig 回测/撮合/账户参数
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProjectPaths:
    """项目路径管理: 所有读写目录的唯一来源。"""

    root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def cache_dir(self) -> Path:
        """行情缓存目录(Parquet), 避免重复请求 akshare。"""
        return self.data_dir / "cache"

    @property
    def output_dir(self) -> Path:
        """回测结果输出目录。"""
        return self.root / "output"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> "ProjectPaths":
        """确保所有目录存在, 幂等操作。"""
        for directory in (self.data_dir, self.cache_dir, self.output_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class DataConfig:
    """DataCenter 数据模块配置。"""

    symbol: str = "000001"          # 标的代码(平安银行)
    start_date: str = "20200101"    # 起始日期 YYYYMMDD
    end_date: str = "20251231"      # 结束日期 YYYYMMDD
    adjust: str = "qfq"             # 复权方式: qfq 前复权 / hfq 后复权 / "" 不复权
    period: str = "daily"           # K线周期: daily / weekly / monthly
    use_cache: bool = True          # 是否启用本地 Parquet 缓存
    offline_fallback: bool = True   # akshare 失败时是否降级为合成数据(演示/联调用)
    fetch_retries: int = 3          # 网络拉取失败时的统一重试次数(代理断连等)
    fetch_backoff: float = 2.0      # 重试退避基数(秒), 第 n 次重试等待 n*backoff
    synthetic_days: int = 600       # 合成数据长度
    synthetic_seed: int = 42        # 合成数据随机种子, 保证可复现


@dataclass
class StrategyConfig:
    """StrategyEngine 策略模块配置。"""

    name: str = "ma_cross"                                  # 策略注册名
    params: dict[str, Any] = field(default_factory=dict)    # 策略参数, 如 {"fast": 5, "slow": 20}


@dataclass
class BacktestConfig:
    """BacktestEngine 回测/执行模块配置。"""

    initial_capital: float = 1_000_000.0    # 初始资金
    commission_rate: float = 0.0003         # 佣金费率(万3)
    slippage_rate: float = 0.0002           # 滑点(按成交价比例)
    position_ratio: float = 0.95            # 单次建仓占用可用资金比例
    lot_size: int = 100                     # A股最小交易单位: 1手=100股
    allow_short: bool = False               # 是否允许做空(A股默认不允许)
    signal_delay_bars: int = 1              # 信号延迟K线数: 收盘出信号, 次日开盘成交, 规避未来函数
    annual_trading_days: int = 252          # 年化交易日数(用于夏普比率)
    # ---- 交易执行参数(可训练) ----
    rebalance: str = "daily"                # 调仓周期: daily / weekly / monthly
    order_type: str = "market"              # 订单类型: market 市价 / limit 限价(次日开盘满足条件才成交)
    slippage_tolerance_pct: float = 0.0     # 滑点容忍度: 实际滑点超过该比例则放弃交易(0=不限制)
    # ---- 仓位与资金管理(可训练) ----
    leverage: float = 1.0                   # 杠杆倍数(1.0=无杠杆)
    max_position_ratio: float = 1.0         # 单标的持仓上限: 名义市值/权益 (0~1), 用于分散风险
    # ---- 风险控制(可训练) ----
    stop_loss_pct: float = 0.0              # 单笔止损比例(0=关闭), 如 0.08 表示亏损 8% 平仓
    take_profit_pct: float = 0.0            # 单笔止盈比例(0=关闭)
    max_drawdown_halt: float = 0.0          # 账户回撤熔断(0=关闭), 如 0.15 表示回撤 15% 停止开新仓
    halt_cooldown_days: int = 0             # 熔断后强制空仓的交易日数(0=无冷却, 恢复条件满足即恢复)
    halt_resume_drawdown: float = 0.0       # 熔断恢复阈值: 回撤恢复到该比例以内才允许重新建仓(0=需创新高)


@dataclass
class AppConfig:
    """总配置: 串联路径、数据、策略、回测四组配置。"""

    paths: ProjectPaths = field(default_factory=ProjectPaths)
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    verbose: bool = True

    def ensure_paths(self) -> "AppConfig":
        """初始化所有目录, 链式调用。"""
        self.paths.ensure()
        return self
