"""
实时行情适配器
==============
用于 9:20 开盘执行阶段获取实时价, 多源探测:
    东财盘口(stock_bid_ask_em) -> 新浪全市场快照(stock_zh_a_spot) -> 新浪60分钟(当日最新)
全部失败返回 None, 由调用方决定发告警还是跳过。
本地免费源在开盘前/盘中可能不可用; 阿里云上东财接口大概率可用。
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

try:
    import akshare as ak

    _HAS_AKSHARE = True
except ImportError:  # pragma: no cover
    ak = None
    _HAS_AKSHARE = False


def _to_exchange_symbol(code: str) -> str:
    code = str(code).zfill(6)
    if code[0] in ("6", "9"):
        return f"sh{code}"
    if code[0] in ("4", "8"):
        return f"bj{code}"
    return f"sz{code}"


def get_realtime_price(symbol: str) -> dict[str, Any] | None:
    """
    多源获取实时价, 返回 {"price": float, "source": str} 或 None。
    任一源成功即返回, 顺序: 东财盘口 -> 新浪快照 -> 新浪60分钟(当日)。
    """
    if not _HAS_AKSHARE:
        return None
    code = str(symbol).zfill(6)

    # 1) 东财盘口(阿里云可用; 本机被网关拦)
    try:
        df = ak.stock_bid_ask_em(symbol=code)
        item = {}
        for _, row in df.iterrows():
            item[str(row.iloc[0])] = row.iloc[1]
        price = item.get("最新") or item.get("卖一") or item.get("买一")
        if price is not None and float(price) > 0:
            return {"price": float(price), "source": "eastmoney"}
    except Exception:
        pass

    # 2) 新浪全市场快照(慢, 但一次可拿全市场)
    try:
        df = ak.stock_zh_a_spot()
        row = df[df["代码"].astype(str).str.zfill(6) == code]
        if not row.empty:
            price = float(row.iloc[0]["最新价"])
            if price > 0:
                return {"price": price, "source": "sina_spot"}
    except Exception:
        pass

    # 3) 新浪60分钟: 当日最新一根(开盘后第一根小时线生成前可能没有当日bar)
    try:
        df = ak.stock_zh_a_minute(symbol=_to_exchange_symbol(code), period="60", adjust="qfq")
        if isinstance(df, pd.DataFrame) and len(df):
            last_day = pd.to_datetime(df["day"].iloc[-1]).date()
            if last_day == date.today():
                close = float(df["close"].iloc[-1])
                if close > 0:
                    return {"price": close, "source": "sina_60min"}
    except Exception:
        pass

    return None


def get_open_price(symbol: str) -> dict[str, Any] | None:
    """
    获取"今日开盘价"(今开): 东财盘口"今开" -> 新浪快照"今开" -> 新浪60分钟当日首根bar的open。
    9:25 集合竞价结束后可用; 之后任意时刻取到都是同一个开盘价, 用于重试后仍按开盘价成交。
    """
    if not _HAS_AKSHARE:
        return None
    code = str(symbol).zfill(6)

    # 1) 东财盘口: 今开
    try:
        df = ak.stock_bid_ask_em(symbol=code)
        item = {str(row.iloc[0]): row.iloc[1] for _, row in df.iterrows()}
        price = item.get("今开")
        if price is not None and float(price) > 0:
            return {"price": float(price), "source": "eastmoney_open"}
    except Exception:
        pass

    # 2) 新浪全市场快照: 今开
    try:
        df = ak.stock_zh_a_spot()
        row = df[df["代码"].astype(str).str.zfill(6) == code]
        if not row.empty:
            price = float(row.iloc[0]["今开"])
            if price > 0:
                return {"price": price, "source": "sina_open"}
    except Exception:
        pass

    # 3) 新浪60分钟: 当日第一根bar的 open(即今日开盘价)
    try:
        df = ak.stock_zh_a_minute(symbol=_to_exchange_symbol(code), period="60", adjust="qfq")
        if isinstance(df, pd.DataFrame) and len(df):
            today_mask = pd.to_datetime(df["day"]).dt.date == date.today()
            today_bars = df[today_mask]
            opens = pd.to_numeric(today_bars["open"], errors="coerce")
            valid = today_bars[opens.notna() & (opens > 0)]
            if len(valid):
                return {"price": float(valid.iloc[0]["open"]), "source": "sina_60min_open"}
    except Exception:
        pass

    return None
