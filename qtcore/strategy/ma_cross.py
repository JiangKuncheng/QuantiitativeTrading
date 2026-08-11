"""
经典双均线交叉动量策略 (Dual Moving Average Crossover)
=======================================================

逻辑:
    fast_ma  = SMA(close, fast_window)
    slow_ma  = SMA(close, slow_window)

    long_only (默认):
        fast_ma >  slow_ma  -> 目标仓位 1(持多)
        fast_ma <= slow_ma  -> 目标仓位 0(空仓)

    long_short (可选):
        fast_ma >  slow_ma  -> 目标仓位  1
        fast_ma <= slow_ma  -> 目标仓位 -1

技术指标适配层:
    优先使用 pandas-ta; 若未安装则回退到 pandas.rolling, 保证环境极简可运行。
    若团队选用 ta-lib, 只需把 _sma() 替换为 talib.SMA(close.values, timeperiod=window),
    策略其余代码零改动 —— 这是"指标库可插拔"的设计意图。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qtcore.strategy.base import StrategyBase


class MACrossStrategy(StrategyBase):
    """双均线交叉策略。"""

    name = "ma_cross"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        raw = dict(params or {})
        # 先解析参数, 再调用基类: 基类 __init__ 会触发 validate_params(),
        # 因此均线窗口属性必须先于 super().__init__() 就绪
        self.fast_window = int(raw.get("fast", 5))
        self.slow_window = int(raw.get("slow", 20))
        self.long_short = bool(raw.get("long_short", False))
        # RSI 过滤(可训练): 金叉且 RSI<买入阈值 才做多; 死叉或 RSI>卖出阈值 离场
        self.rsi_window = int(raw.get("rsi_window", 14))
        self.rsi_buy = float(raw.get("rsi_buy", 30.0))
        self.rsi_sell = float(raw.get("rsi_sell", 70.0))
        self.use_rsi = bool(raw.get("use_rsi", False))
        super().__init__(raw)

    def validate_params(self) -> None:
        if self.fast_window >= self.slow_window:
            raise ValueError(
                f"fast({self.fast_window}) 必须小于 slow({self.slow_window})"
            )
        if self.fast_window <= 0 or self.slow_window <= 0:
            raise ValueError("均线窗口必须为正整数")

    # ------------------------------------------------------------------
    # 指标适配层
    # ------------------------------------------------------------------
    @staticmethod
    def _sma(close: pd.Series, window: int) -> pd.Series:
        """
        简单移动平均: 优先 pandas-ta, 回退 pandas.rolling。
        指标库更换点: 如需 ta-lib, 在此处改为 talib.SMA(...) 即可。
        """
        try:
            import pandas_ta as ta

            result = ta.sma(close, length=window)
            if result is not None:
                return result
        except ImportError:
            pass  # pandas-ta 未安装, 走回退
        except Exception:
            pass  # 指标计算异常时也回退, 保证骨架稳定
        return close.rolling(window=window).mean()

    @staticmethod
    def _rsi(close: pd.Series, window: int) -> pd.Series:
        """RSI: 优先 pandas-ta, 回退手写 Wilder 平滑。"""
        try:
            import pandas_ta as ta

            result = ta.rsi(close, length=window)
            if result is not None:
                return result
        except Exception:
            pass
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return rsi.fillna(50.0)

    # ------------------------------------------------------------------
    # StrategyBase 接口实现
    # ------------------------------------------------------------------
    def compute_indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        df["fast_ma"] = self._sma(df["close"], self.fast_window)
        df["slow_ma"] = self._sma(df["close"], self.slow_window)
        df["ma_gap"] = df["fast_ma"] - df["slow_ma"]
        df["rsi"] = self._rsi(df["close"], self.rsi_window)
        return df

    def target_positions(self, bars: pd.DataFrame) -> pd.Series:
        ind = self.compute_indicators(bars)
        fast = ind["fast_ma"]
        slow = ind["slow_ma"]

        valid = fast.notna() & slow.notna()
        if self.use_rsi:
            valid = valid & ind["rsi"].notna()
        position = pd.Series(0, index=bars.index, dtype=int)
        long_cond = valid & (fast > slow)
        if self.use_rsi:
            long_cond = long_cond & (ind["rsi"] < self.rsi_buy)
        position[long_cond] = 1
        if self.long_short:
            short_cond = valid & (fast <= slow)
            if self.use_rsi:
                short_cond = short_cond & (ind["rsi"] > self.rsi_sell)
            position[short_cond] = -1
        elif self.use_rsi:
            # 多头模式下 RSI 超买视为离场信号(目标仓位置 0)
            position[valid & (ind["rsi"] > self.rsi_sell)] = 0
        return position

    def __repr__(self) -> str:  # 便于日志打印策略参数
        return (
            f"MACrossStrategy(fast={self.fast_window}, slow={self.slow_window}, "
            f"long_short={self.long_short})"
        )
