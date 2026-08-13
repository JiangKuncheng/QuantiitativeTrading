"""
真实全市场股票池(数据驱动, 非人工名单)
======================================
股票池来源必须是真实官方/权威列表:
    cn: akshare 全A列表(已缓存 universe_all.csv, 5539只)
    us: SEC company_tickers.json(全美股) 过滤普通股
    hk: 港交所 ListOfSecurities.xlsx 过滤普通股
列表缓存到 data/cache/universe_<market>.csv。

选股流程(全部有数据支撑):
    1. 真实全市场列表
    2. 流动性初筛: 用 2024-2025 窗口逐只拉日线, 统计K线数与日均成交额,
       保留数据充足且流动性前 N 名(证明: pool_<market>_real.csv 里有每只的统计)
    3. 训练集 Top-K 选股(按策略回测夏普等指标排序, 有逐只指标)
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from qtcore.config import AppConfig
from qtcore.datacenter.data_center import DataCenter


ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"


def _us_universe() -> pd.DataFrame:
    """
    美股官方交易所上市列表(Nasdaq + NYSE/AMEX):
    nasdaqlisted.txt / otherlisted.txt, 过滤 ETF 与非正股。
    """
    frames = []
    for fname, url in (
        ("nasdaqlisted.txt", "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"),
        ("otherlisted.txt", "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"),
    ):
        path = CACHE / fname
        if not path.exists():
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=90)
            r.raise_for_status()
            path.write_bytes(r.content)
        df = pd.read_csv(path, sep="|", dtype=str)
        df = df[df["ETF"].fillna("N").str.upper() == "N"]
        test_col = "Test Issue" if "Test Issue" in df.columns else "Test Issue"
        if test_col in df.columns:
            df = df[df[test_col].fillna("N").str.upper() == "N"]
        sym_col = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
        name_col = "Security Name"
        frames.append(df[[sym_col, name_col]].rename(columns={sym_col: "code", name_col: "name"}))
    out = pd.concat(frames, ignore_index=True).drop_duplicates("code")
    out = out[out["code"].str.match(r"^[A-Z][A-Z0-9.\-]{0,5}$", na=False)]
    return out.reset_index(drop=True)


def _hkex_universe() -> pd.DataFrame:
    """港交所官方证券列表, 表头在第2行, 过滤 Category=Equity 的普通股。"""
    url = "https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=90)
    r.raise_for_status()
    tmp = CACHE / "hkex_list.xlsx"
    tmp.write_bytes(r.content)
    # 结构: 第1行标题, 第2行更新日期, 第3行表头, 第4行起数据
    df = pd.read_excel(tmp, sheet_name=0, header=None, dtype=str)
    df.columns = [str(c).strip() for c in df.iloc[2]]
    df = df.iloc[3:].reset_index(drop=True)
    out = df[["Stock Code", "Name of Securities", "Category"]].rename(
        columns={"Stock Code": "code", "Name of Securities": "name", "Category": "cat"}
    )
    out["code"] = out["code"].astype(str).str.extract(r"(\d{5})")[0]
    out = out.dropna(subset=["code"]).drop_duplicates("code")
    out = out[out["cat"].astype(str).str.strip().str.upper() == "EQUITY"]
    return out[["code", "name"]].reset_index(drop=True)


def load_real_universe(market: str) -> pd.DataFrame:
    """加载真实全市场列表(带本地缓存)。"""
    cache_file = CACHE / f"universe_{market}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, dtype=str)
    if market == "cn":
        p = CACHE / "universe_all.csv"
        if not p.exists():
            raise RuntimeError("A股全市场列表缓存缺失, 请先运行全市场扫描生成 universe_all.csv")
        df = pd.read_csv(p, dtype=str)
    elif market == "us":
        df = _us_universe()
    elif market == "hk":
        df = _hkex_universe()
    else:
        raise ValueError(market)
    df.to_csv(cache_file, index=False, encoding="utf-8")
    source = "Nasdaq/NYSE 官方上市列表" if market == "us" else "HKEX 港交所官方" if market == "hk" else "akshare 全A"
    print(f"[Universe] {market.upper()} 真实全市场列表: {len(df)} 只 (来源: {source})")
    return df


def build_data_pool(
    market: str,
    lookback_start: str = "20240101",
    lookback_end: str = "20260810",
    top_n: int = 150,
    min_bars: int = 200,
    workers: int = 12,
) -> pd.DataFrame:
    """
    数据驱动的流动性初筛:
    对真实全市场列表逐只拉 lookback 窗口日线, 统计K线数/日均成交额,
    保留数据充足且流动性前 top_n 名。结果存 pool_<market>_real.csv 作为选股依据。
    """
    app = AppConfig()
    dc = DataCenter(app.data, app.paths)
    universe = load_real_universe(market)
    symbols = universe["code"].tolist()
    total = len(symbols)
    rows: list[dict[str, Any]] = []
    done = 0

    def one(code: str) -> dict[str, Any] | None:
        try:
            bars = dc.get_bars(code, lookback_start, lookback_end, "daily", market)
            if bars is None or len(bars) < min_bars:
                return None
            amount = bars["amount"] if "amount" in bars.columns else bars["close"] * bars["volume"]
            avg_amount = float(amount.mean())
            return {
                "code": code,
                "name": universe.loc[universe["code"] == code, "name"].iloc[0] if "name" in universe.columns else "",
                "bars": len(bars),
                "avg_amount": round(avg_amount, 2),
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, c) for c in symbols]
        for f in as_completed(futures):
            done += 1
            r = f.result()
            if r is not None:
                rows.append(r)
            if done % 500 == 0 or done == total:
                print(f"[Pool] {market.upper()} 进度 {done}/{total} (有效 {len(rows)})")

    if not rows:
        raise RuntimeError(f"{market} 无有效数据")
    df = pd.DataFrame(rows).sort_values("avg_amount", ascending=False).reset_index(drop=True)
    top = df.head(top_n)
    out = ROOT / "data" / f"pool_{market}_real.csv"
    top.to_csv(out, index=False, encoding="utf-8")
    print(f"[Pool] {market.upper()} 数据初筛: {total} -> 有效 {len(df)} -> Top{top_n} 已保存 {out}")
    return top
