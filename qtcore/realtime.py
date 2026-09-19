"""
实时行情适配器
==============
用于 9:20 开盘执行阶段获取实时价 / 今日开盘价, 多源探测。

数据源优先级(按实测有效性排序):
    1) 腾讯快照 qt.gtimg.cn      —— 9:25 集合竞价一结束就有"今开", 且带行情时间戳可校验;
    2) 东财盘口 stock_bid_ask_em —— 字段最全, 但部分机房 IP 会被拒(RemoteDisconnected);
    3) 新浪全市场快照 stock_zh_a_spot —— 慢, 新版 akshare 已不稳定;
    4) 新浪60分钟 stock_zh_a_minute —— 兜底, 但当日第一根小时线 **10:30 才生成**,
       所以 9:30~10:20 之间拿不到当日 bar, 重试窗口必须覆盖到 10:30 之后。

历史坑: 早期每个数据源都是 `except Exception: pass`, 失败时不留任何日志,
导致"9:20 执行从未成交"只能靠猜。现在失败原因会逐条打印。
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import pandas as pd
import requests

try:
    import akshare as ak

    _HAS_AKSHARE = True
except ImportError:  # pragma: no cover
    ak = None
    _HAS_AKSHARE = False


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _fetch_tencent_snapshot(code: str) -> dict[str, Any]:
    """
    腾讯行情快照。失败抛异常, 由调用方记录原因。

    响应格式: v_sz000333="51~美的集团~000333~84.40~85.04~84.90~...~20260918161439~..."
    字段: [1]名称 [3]最新价 [4]昨收 [5]今开 [30]行情时间戳 YYYYMMDDHHMMSS
    """
    symbol = _to_exchange_symbol(code)
    response = requests.get(
        f"https://qt.gtimg.cn/q={symbol}",
        timeout=8,
        headers={"User-Agent": USER_AGENT, "Referer": "https://gu.qq.com/"},
    )
    response.raise_for_status()
    response.encoding = "gbk"
    match = re.search(r'"([^"]*)"', response.text)
    if not match:
        raise ValueError("响应中没有引号包裹的行情数据")
    fields = match.group(1).split("~")
    if len(fields) < 31:
        raise ValueError(f"行情字段数异常({len(fields)})")
    return {
        "name": fields[1],
        "last": float(fields[3] or 0),
        "open": float(fields[5] or 0),
        "stamp": fields[30],
    }


def _tencent_quote(code: str, field: str) -> tuple[dict[str, Any] | None, str]:
    """取腾讯快照的某个价格字段, 返回 (结果, 失败原因)。"""
    try:
        snap = _fetch_tencent_snapshot(code)
    except Exception as exc:  # noqa: BLE001 - 记录原因后交给下一个数据源
        return None, f"tencent: {type(exc).__name__} {str(exc)[:120]}"
    today = date.today().strftime("%Y%m%d")
    if not str(snap["stamp"]).startswith(today):
        return None, f"tencent: 行情时间为 {snap['stamp'][:8]}, 不是今日({today})"
    price = float(snap["open"] if field == "open" else snap["last"])
    if price <= 0:
        return None, f"tencent: {field} 为 0"
    return {"price": price, "source": f"tencent_{field}"}, ""


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
    任一源成功即返回, 顺序: 腾讯快照 -> 东财盘口 -> 新浪快照 -> 新浪60分钟(当日)。
    """
    code = str(symbol).zfill(6)
    errors: list[str] = []

    quote, error = _tencent_quote(code, "last")
    if quote:
        return quote
    errors.append(error)

    if not _HAS_AKSHARE:
        print(f"[Realtime] {code} 实时价获取失败: " + "; ".join(errors))
        return None

    # 东财盘口(阿里云等机房 IP 常被拒)
    try:
        df = ak.stock_bid_ask_em(symbol=code)
        item = {}
        for _, row in df.iterrows():
            item[str(row.iloc[0])] = row.iloc[1]
        price = item.get("最新") or item.get("卖一") or item.get("买一")
        if price is not None and float(price) > 0:
            return {"price": float(price), "source": "eastmoney"}
        errors.append("eastmoney: 无有效价格字段")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"eastmoney: {type(exc).__name__} {str(exc)[:120]}")

    # 新浪60分钟: 当日最新一根(开盘后第一根小时线 10:30 生成前没有当日 bar)
    try:
        df = ak.stock_zh_a_minute(symbol=_to_exchange_symbol(code), period="60", adjust="qfq")
        if isinstance(df, pd.DataFrame) and len(df):
            last_day = pd.to_datetime(df["day"].iloc[-1]).date()
            if last_day == date.today():
                close = float(df["close"].iloc[-1])
                if close > 0:
                    return {"price": close, "source": "sina_60min"}
            errors.append(f"sina_60min: 最新 bar 为 {last_day}, 非今日")
        else:
            errors.append("sina_60min: 返回空数据")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"sina_60min: {type(exc).__name__} {str(exc)[:120]}")

    # 新浪全市场快照(慢, 且新版 akshare 常取不到, 故放最后)
    try:
        df = ak.stock_zh_a_spot()
        row = df[df["代码"].astype(str).str.zfill(6) == code]
        if not row.empty:
            price = float(row.iloc[0]["最新价"])
            if price > 0:
                return {"price": price, "source": "sina_spot"}
        errors.append("sina_spot: 未命中该代码")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"sina_spot: {type(exc).__name__} {str(exc)[:120]}")

    print(f"[Realtime] {code} 实时价获取失败: " + "; ".join(errors))
    return None


def get_open_price(symbol: str) -> dict[str, Any] | None:
    """
    获取"今日开盘价"(今开): 腾讯快照 -> 东财盘口 -> 新浪快照 -> 新浪60分钟当日首根 bar。
    9:25 集合竞价结束后可用; 之后任意时刻取到都是同一个开盘价, 用于重试后仍按开盘价成交。
    """
    code = str(symbol).zfill(6)
    errors: list[str] = []

    # 1) 腾讯快照: 9:25 后立即有"今开", 且带行情时间戳可校验新鲜度
    quote, error = _tencent_quote(code, "open")
    if quote:
        return quote
    errors.append(error)

    if not _HAS_AKSHARE:
        print(f"[Realtime] {code} 开盘价获取失败: " + "; ".join(errors))
        return None

    # 2) 东财盘口: 今开
    try:
        df = ak.stock_bid_ask_em(symbol=code)
        item = {str(row.iloc[0]): row.iloc[1] for _, row in df.iterrows()}
        price = item.get("今开")
        if price is not None and float(price) > 0:
            return {"price": float(price), "source": "eastmoney_open"}
        errors.append("eastmoney_open: 无有效今开")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"eastmoney_open: {type(exc).__name__} {str(exc)[:120]}")

    # 3) 新浪60分钟: 当日第一根 bar 的 open(注意 10:30 之后才会有当日 bar)
    try:
        df = ak.stock_zh_a_minute(symbol=_to_exchange_symbol(code), period="60", adjust="qfq")
        if isinstance(df, pd.DataFrame) and len(df):
            today_mask = pd.to_datetime(df["day"]).dt.date == date.today()
            today_bars = df[today_mask]
            opens = pd.to_numeric(today_bars["open"], errors="coerce")
            valid = today_bars[opens.notna() & (opens > 0)]
            if len(valid):
                return {"price": float(valid.iloc[0]["open"]), "source": "sina_60min_open"}
            errors.append("sina_60min_open: 尚无当日分钟线(10:30 前不会生成)")
        else:
            errors.append("sina_60min_open: 返回空数据")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"sina_60min_open: {type(exc).__name__} {str(exc)[:120]}")

    # 4) 新浪全市场快照: 今开。
    #    放最后: 它要拉全市场 70 页(实测约 27 秒)且新版 akshare 常取不到,
    #    放中间会白白吃掉重试窗口, 也会拖慢唯一可用的 10:30 兜底源。
    try:
        df = ak.stock_zh_a_spot()
        row = df[df["代码"].astype(str).str.zfill(6) == code]
        if not row.empty:
            price = float(row.iloc[0]["今开"])
            if price > 0:
                return {"price": price, "source": "sina_open"}
        errors.append("sina_open: 未命中该代码")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"sina_open: {type(exc).__name__} {str(exc)[:120]}")

    print(f"[Realtime] {code} 开盘价获取失败: " + "; ".join(errors))
    return None
