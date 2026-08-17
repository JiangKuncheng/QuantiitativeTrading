"""
DataCenter 数据模块
===================

职责:
1. 通过 akshare 获取 A 股历史日线/实时行情;
2. 清洗与标准化: 统一输出为 UnifiedBar 标准 DataFrame;
3. 本地 Parquet 缓存, 减少重复网络请求;
4. 离线降级: 网络不可用时生成可复现的合成行情, 保证全流程可跑通。

统一数据格式 (UnifiedBar):
    index   : DatetimeIndex (name='datetime'), 升序、无重复
    columns : open, high, low, close, volume, amount
    attrs   : {"code": "000001"}

设计说明:
- 所有数据源(akshare、tushare、数据库等)都收敛到 normalize() 这一道口,
  上层策略/回测只依赖统一格式, 与数据源完全解耦;
- 实时行情在 akshare 中为轮询接口, 生产环境建议替换为 WebSocket 推送,
  本模块已把实时数据也归一化到统一结构。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import time
import os

try:  # akshare 为可选依赖: 未安装时自动进入合成数据/离线模式
    import akshare as ak

    _HAS_AKSHARE = True
except ImportError:  # pragma: no cover - 环境依赖
    ak = None
    _HAS_AKSHARE = False

from qtcore.config import DataConfig, ProjectPaths
from qtcore.events import BarEvent


@dataclass(frozen=True)
class UnifiedBarSpec:
    """统一行情格式规范: 全系统唯一的数据契约。"""

    index_name: str = "datetime"
    columns: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")
    code_attr: str = "code"


UNIFIED_BAR = UnifiedBarSpec()

# 多数据源字段映射 -> 统一英文字段:
#   东财(stock_zh_a_hist) 为中文列名; 新浪/腾讯为英文小写列名
_AK_COLUMN_MAP: dict[str, str] = {
    "日期": "datetime",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "date": "datetime",
    "day": "datetime",
    "datetime": "datetime",
    "open": "open",
    "close": "close",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "amount": "amount",
}


class DataCenter:
    """
    数据中心: 数据获取 + 清洗 + 标准化 + 缓存。

    使用示例:
        dc = DataCenter(DataConfig(), ProjectPaths())
        bars = dc.get_daily_bars(symbol="000001")
    """

    def __init__(self, config: DataConfig, paths: ProjectPaths) -> None:
        self.config = config
        self.paths = paths
        self.paths.ensure()

    # ------------------------------------------------------------------
    # 对外主接口
    # ------------------------------------------------------------------
    def get_daily_bars(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        获取标准化日线数据(带缓存与离线降级)。

        参数:
            symbol    : 标的代码, 默认取配置
            start_date: 起始日期 YYYYMMDD
            end_date  : 结束日期 YYYYMMDD
            use_cache : 是否使用本地缓存

        返回:
            符合 UnifiedBar 格式的 DataFrame
        """
        symbol = symbol or self.config.symbol
        start = start_date or self.config.start_date
        end = end_date or self.config.end_date
        use_cache = self.config.use_cache if use_cache is None else use_cache
        start_ts = self._parse_date(start)
        end_ts = self._parse_date(end)

        # 1) 尝试读取缓存
        cache_path = self._cache_path(symbol, start, end)
        if use_cache and cache_path.exists():
            cached = self._read_cache(cache_path, symbol)
            if cached is not None and not cached.empty:
                cached_last = cached.index[-1].strftime("%Y%m%d")
                end_key = end_ts.strftime("%Y%m%d") if end_ts is not None else ""
                if end_key and cached_last < end_key:
                    # 缓存最新日早于请求日(例如当天数据刚发布, 缓存是收盘前拉的过期数据):
                    # 视为缓存过期, 强制重新拉取覆盖
                    print(
                        f"[DataCenter] 缓存最新日 {cached_last} < 请求日 {end_key}, "
                        f"重新拉取覆盖"
                    )
                else:
                    print(f"[DataCenter] 命中缓存: {cache_path.name} ({len(cached)} 根K线)")
                    return self._slice(cached, start_ts, end_ts)

        # 2) 实时获取 akshare, 失败则降级合成数据
        try:
            # 统一重试: 网络代理偶发断连/远端超时是家常便饭,
            # 在数据层做指数退避重试, 上层(选股器/训练器)无需各自实现
            df = None
            last_exc: Exception | None = None
            for attempt in range(1, self.config.fetch_retries + 1):
                try:
                    df = self.fetch_historical(
                        symbol=symbol,
                        start_date=start,
                        end_date=end,
                        adjust=self.config.adjust,
                        period=self.config.period,
                    )
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < self.config.fetch_retries:
                        wait = self.config.fetch_backoff * attempt
                        print(
                            f"[DataCenter] akshare 获取失败(第 {attempt}/{self.config.fetch_retries} 次), "
                            f"{wait:.0f}s 后重试: {exc!r}"
                        )
                        time.sleep(wait)
            if df is None:
                raise last_exc  # type: ignore[misc]
            source = df.attrs.get("source", "eastmoney")
        except Exception as exc:  # 网络异常 / 数据源异常
            if not self.config.offline_fallback:
                raise RuntimeError(f"数据获取失败且未开启离线降级: {exc}") from exc
            print(f"[DataCenter] akshare 获取失败({exc!r}), 降级为合成行情")
            df = self.generate_synthetic_bars(symbol=symbol)
            source = "synthetic"

        df = self._slice(df, start_ts, end_ts)

        # 3) 写入缓存: 仅真实数据落缓存, 合成降级数据绝不入缓存,
        #    避免污染真实标的的缓存(否则下次运行会命中"假数据")
        if use_cache and source != "synthetic" and not df.empty:
            try:
                self._write_cache(df, cache_path)
            except Exception as exc:
                print(f"[DataCenter] 缓存写入失败, 跳过缓存: {exc!r}")

        print(f"[DataCenter] 行情就绪: {symbol} {len(df)} 根K线, 来源={source}")
        return df

    # ------------------------------------------------------------------
    # 多时间框架(日内线)
    # ------------------------------------------------------------------
    INTRADAY_TIMEFRAMES = ("60min", "2h", "4h", "6h")

    def get_bars(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = "daily",
        market: str = "cn",
    ) -> pd.DataFrame:
        """
        按时间框架获取统一K线: daily / 60min / 2h / 4h / 6h。
        market: cn(A股, 日线+新浪60分钟) / us(美股, 新浪日线) / hk(港股, 新浪日线)。
        """
        if market in ("us", "hk"):
            if timeframe != "daily":
                return self._get_us_hk_intraday(symbol, start_date, end_date, market, timeframe)
            return self._get_us_hk_bars(symbol, start_date, end_date, market)
        if timeframe == "daily" or timeframe is None:
            return self.get_daily_bars(symbol, start_date, end_date)
        if timeframe not in self.INTRADAY_TIMEFRAMES:
            raise ValueError(f"不支持的时间框架: {timeframe}, 可选 {self.INTRADAY_TIMEFRAMES}")
        return self._get_intraday_bars(symbol, start_date, end_date, timeframe)

    def _get_us_hk_bars(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        market: str,
    ) -> pd.DataFrame:
        """美股/港股日线(新浪): us -> stock_us_daily, hk -> stock_hk_daily。"""
        self._require_akshare()
        start = start_date or self.config.start_date
        end = end_date or self.config.end_date
        cache_path = self.paths.cache_dir / f"{market}_daily_{symbol}_{start}_{end}.parquet"
        if self.config.use_cache and cache_path.exists():
            cached = self._read_cache(cache_path, symbol)
            if cached is not None and not cached.empty:
                cached_last = cached.index[-1].strftime("%Y%m%d")
                end_key = self._parse_date(end).strftime("%Y%m%d") if self._parse_date(end) else ""
                if not (end_key and cached_last < end_key):
                    print(f"[DataCenter] 命中缓存: {cache_path.name} ({len(cached)} 根K线)")
                    return self._slice(cached, self._parse_date(start), self._parse_date(end))
        raw = (
            ak.stock_us_daily(symbol=str(symbol), adjust="qfq")
            if market == "us"
            else ak.stock_hk_daily(symbol=str(symbol), adjust="qfq")
        )
        df = self.normalize(raw, code=str(symbol))
        df.attrs["source"] = f"sina_{market}"
        df = self._slice(df, self._parse_date(start), self._parse_date(end))
        if self.config.use_cache and not df.empty:
            try:
                df.to_parquet(cache_path)
            except Exception:
                pass
        print(f"[DataCenter] {market.upper()} 行情就绪: {symbol} {len(df)} 根K线(来源 sina)")
        return df

    def _get_us_hk_intraday(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        market: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        美股/港股分钟线(多源自动回退):
            1) TwelveData(1小时线, 免费key, 阿里云/本地均可达, 深度约2年) -> 重采样 2h/4h/6h;
            2) 东财 K线接口(secid: us=105/106/107, hk=116, 仅网络可达时可用);
            3) akshare 港股分钟接口(仅 hk, 最后兜底)。
        统一返回 UnifiedBar(60min 或重采样后的 timeframe)。
        """
        start = start_date or self.config.start_date
        end = end_date or self.config.end_date
        cache_path = self.paths.cache_dir / f"{market}_{timeframe}_{symbol}_{start}_{end}.parquet"
        if self.config.use_cache and cache_path.exists():
            cached = self._read_cache(cache_path, symbol)
            if cached is not None and not cached.empty:
                print(f"[DataCenter] 命中缓存: {cache_path.name}")
                return cached

        errors: list[str] = []
        # 1) TwelveData: 服务器/本地都可达, 免费key, 1小时线深度约2年
        if self._twelvedata_key():
            try:
                raw = self._fetch_twelvedata_60min(symbol, start, end, market)
                df = self.normalize(raw, code=str(symbol))
                df.attrs["source"] = "twelvedata"
                df = self._resample_intraday(df, symbol, timeframe)
                df = self._slice(df, self._parse_date(start), self._parse_date(end))
                self._save_intraday_cache(df, cache_path)
                print(f"[DataCenter] {market.upper()} {timeframe} 行情就绪: {symbol} {len(df)} 根(来源 twelvedata)")
                return df
            except Exception as exc:
                errors.append(f"twelvedata: {exc!r}")

        # 2) 东财 K线接口直连(secid: us=105/106/107 逐一尝试, hk=116)
        try:
            if market == "us":
                raw = None
                last_err: Exception | None = None
                for prefix in ("105", "106", "107"):
                    try:
                        probe = self._fetch_em_kline_secid(f"{prefix}.{symbol}", start, end)
                        if probe is not None and not probe.empty:
                            raw = probe
                            break
                    except Exception as exc:  # noqa: PERF203
                        last_err = exc
                if raw is None or raw.empty:
                    raise RuntimeError(f"eastmoney kline us {symbol}: 无数据 ({last_err!r})")
            else:
                raw = self._fetch_em_kline_secid(f"116.{symbol}", start, end)
            df = self.normalize(raw, code=str(symbol))
            df.attrs["source"] = "eastmoney_kline"
            df = self._resample_intraday(df, symbol, timeframe)
            df = self._slice(df, self._parse_date(start), self._parse_date(end))
            self._save_intraday_cache(df, cache_path)
            print(f"[DataCenter] {market.upper()} {timeframe} 行情就绪: {symbol} {len(df)} 根(来源 eastmoney_kline)")
            return df
        except Exception as exc:
            errors.append(f"eastmoney_kline: {exc!r}")

        # 3) akshare 港股分钟接口(仅 hk 兜底)
        if market == "hk":
            try:
                self._require_akshare()
                raw = ak.stock_hk_hist_min_em(
                    symbol=str(symbol), period="60", adjust="qfq",
                    start_date=start, end_date=end,
                )
                df = self.normalize(raw, code=str(symbol))
                df.attrs["source"] = "eastmoney_min_ak"
                df = self._resample_intraday(df, symbol, timeframe)
                df = self._slice(df, self._parse_date(start), self._parse_date(end))
                self._save_intraday_cache(df, cache_path)
                print(f"[DataCenter] HK {timeframe} 行情就绪: {symbol} {len(df)} 根(来源 eastmoney_min_ak)")
                return df
            except Exception as exc:
                errors.append(f"hk_min_ak: {exc!r}")

        raise RuntimeError(f"{market.upper()} {timeframe} 所有数据源均失败: {symbol}: " + "; ".join(errors))

    # ------------------------------------------------------------------
    # 美股/港股日内数据源实现
    # ------------------------------------------------------------------
    def _twelvedata_key(self) -> str:
        """TwelveData key: 优先 DataConfig, 其次环境变量 TWELVEDATA_API_KEY。"""
        if not self.config.twelvedata_api_key and "TWELVEDATA_API_KEY" not in os.environ:
            try:  # Docker/cron 场景: 代码从 /app/.env 兜底加载
                from qtcore.dotenv import load_dotenv

                load_dotenv()
            except Exception:
                pass
        return self.config.twelvedata_api_key or os.environ.get("TWELVEDATA_API_KEY", "")

    def _fetch_twelvedata_60min(
        self, symbol: str, start: str, end: str, market: str
    ) -> pd.DataFrame:
        """TwelveData 1小时线: us=AAPL, hk=0700.HK; 返回原始 DataFrame(datetime/open/high/low/close/volume)。"""
        import requests

        key = self._twelvedata_key()
        td_symbol = f"{symbol}.HK" if market == "hk" else str(symbol).upper()
        params = {
            "symbol": td_symbol,
            "interval": "1h",
            "outputsize": "5000",
            "apikey": key,
            "start_date": self._to_iso(start),
            "end_date": self._to_iso(end),
            "timezone": "Exchange",
        }
        last_exc: Exception | None = None
        for attempt in range(max(1, self.config.fetch_retries)):
            try:
                r = requests.get(
                    "https://api.twelvedata.com/time_series",
                    params=params, timeout=30,
                    headers={"User-Agent": "Mozilla/5.0 (QuantitativeTrading)"},
                )
                r.raise_for_status()
                data = r.json()
                if data.get("status") != "ok":
                    msg = data.get("message") or data.get("error") or str(data)[:200]
                    raise RuntimeError(f"twelvedata status={data.get('status')}: {msg}")
                values = data.get("values") or []
                if not values:
                    raise RuntimeError(f"twelvedata 无数据: {td_symbol}")
                df = pd.DataFrame(values)
                return df
            except Exception as exc:
                last_exc = exc
                time.sleep(1.0 + attempt * self.config.fetch_backoff)
        raise RuntimeError(f"TwelveData 拉取失败 {td_symbol}: {last_exc!r}")

    def _fetch_em_kline_secid(
        self, secid: str, start: str, end: str, retries: int = 3
    ) -> pd.DataFrame:
        """东财 K线接口直连(push2his): klt=60(60分钟), 返回原始 DataFrame。"""
        import requests

        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "klt": "60",
            "fqt": "1",
            "secid": secid,
            "beg": start.replace("-", ""),
            "end": end.replace("-", ""),
            "lmt": "1000000",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://quote.eastmoney.com/",
        }
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                r = requests.get(url, params=params, timeout=25, headers=headers)
                r.raise_for_status()
                data = r.json()
                klines = (data.get("data") or {}).get("klines") or []
                if not klines:
                    return pd.DataFrame()
                cols = ("datetime", "open", "close", "high", "low",
                        "volume", "amount", "amplitude", "pct_chg", "chg", "turnover")
                rows = [dict(zip(cols, k.split(","))) for k in klines]
                return pd.DataFrame(rows)
            except Exception as exc:
                last_exc = exc
                time.sleep(1.0 + attempt)
        raise RuntimeError(f"EM kline 拉取失败 {secid}: {last_exc!r}")

    @staticmethod
    def _to_iso(date_str: str) -> str:
        """20240101 -> 2024-01-01 (TwelveData 需要 ISO 格式)。"""
        s = str(date_str).strip().replace("-", "").replace("/", "")
        if len(s) >= 8 and s[:8].isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return date_str

    def _resample_intraday(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> pd.DataFrame:
        """60分钟基础线 -> 目标 timeframe(2h/4h/6h); 60min 则直接返回。"""
        if timeframe != "60min":
            agg: dict[str, str] = {"open": "first", "high": "max", "low": "min", "close": "last"}
            if "volume" in df.columns:
                agg["volume"] = "sum"
            if "amount" in df.columns:
                agg["amount"] = "sum"
            df = df.resample(timeframe).agg(agg).dropna(subset=["close"])
            df.attrs[UNIFIED_BAR.code_attr] = str(symbol)
        return df

    def _save_intraday_cache(self, df: pd.DataFrame, cache_path: Path) -> None:
        """写入 Parquet 缓存(失败静默, 不影响主流程)。"""
        if not self.config.use_cache or df is None or df.empty:
            return
        try:
            df.to_parquet(cache_path)
        except Exception:
            pass

    def _get_intraday_bars(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        timeframe: str,
    ) -> pd.DataFrame:
        start = start_date or self.config.start_date
        end = end_date or self.config.end_date
        cache_path = self.paths.cache_dir / f"{timeframe}_{symbol}_{start}_{end}.parquet"

        if self.config.use_cache and cache_path.exists():
            cached = self._read_cache(cache_path, symbol)
            if cached is not None and not cached.empty:
                cached_last = cached.index[-1].strftime("%Y%m%d")
                end_key = self._parse_date(end).strftime("%Y%m%d") if self._parse_date(end) else ""
                if end_key and cached_last < end_key:
                    print(
                        f"[DataCenter] 日内缓存最新日 {cached_last} < 请求日 {end_key}, "
                        f"重新拉取覆盖"
                    )
                else:
                    print(f"[DataCenter] 命中缓存: {cache_path.name} ({len(cached)} 根K线)")
                    return self._slice(cached, self._parse_date(start), self._parse_date(end))

        raw = self._fetch_60min(symbol)
        df = self.normalize(raw, code=symbol)
        if timeframe != "60min":
            agg = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "amount": "sum",
            }
            df = df.resample(timeframe).agg(agg).dropna(subset=["close"])
            df.attrs[UNIFIED_BAR.code_attr] = symbol
            df.attrs["source"] = "sina_intraday"
        df = self._slice(df, self._parse_date(start), self._parse_date(end))
        if self.config.use_cache:
            try:
                df.to_parquet(cache_path)
            except Exception:
                pass
        print(f"[DataCenter] 日内行情就绪: {symbol} {timeframe} {len(df)} 根K线")
        return df

    def _fetch_60min(self, symbol: str) -> pd.DataFrame:
        """60分钟基础线: 新浪(约2年) -> 东财(全历史, 服务器可用)。"""
        self._require_akshare()
        errors: list[str] = []
        sx = self._to_exchange_symbol(symbol)
        try:
            return ak.stock_zh_a_minute(symbol=sx, period="60", adjust="qfq")
        except Exception as exc:
            errors.append(f"sina60: {exc!r}")
        try:
            return ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period="60",
                adjust="qfq",
                start_date="19900101",
                end_date="21000101",
            )
        except Exception as exc:
            errors.append(f"eastmoney60: {exc!r}")
        raise RuntimeError("60分钟数据源均失败: " + "; ".join(errors))

    def get_realtime_quote(self, symbol: Optional[str] = None) -> dict[str, float]:
        """
        获取实时报价(演示用轮询, 生产建议替换为 WebSocket 推送)。

        返回:
            {"last": 最新价, "bid1": 买一价, "ask1": 卖一价}
        """
        symbol = symbol or self.config.symbol
        self._require_akshare()
        raw = ak.stock_bid_ask_em(symbol=symbol)
        quote: dict[str, float] = {}
        for _, row in raw.iterrows():
            item = str(row.iloc[0])
            try:
                value = float(row.iloc[1])
            except (TypeError, ValueError):
                continue
            if "最新" in item:
                quote["last"] = value
            elif "买一" in item:
                quote["bid1"] = value
            elif "卖一" in item:
                quote["ask1"] = value
        return quote

    def get_index_daily(
        self,
        symbol: str = "000300",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取指数日线(基准对比用, 如沪深300), 走新浪指数接口。
        返回统一 UnifiedBar 格式, attrs["code"] = 指数代码。
        """
        self._require_akshare()
        prefix = "sh" if symbol.startswith(("000", "88", "99")) else "sz"
        raw = ak.stock_zh_index_daily(symbol=f"{prefix}{symbol}")
        df = self.normalize(raw, code=symbol)
        df.attrs["source"] = "sina_index"
        start_ts = self._parse_date(start_date)
        end_ts = self._parse_date(end_date)
        return self._slice(df, start_ts, end_ts)

    def to_bar_events(self, df: pd.DataFrame) -> list[BarEvent]:
        """将统一行情 DataFrame 转为 BarEvent 列表(供事件驱动/流式计算使用)。"""
        code = df.attrs.get(UNIFIED_BAR.code_attr, self.config.symbol)
        return [
            BarEvent.from_row(code=code, ts=ts, row=row)
            for ts, row in df.iterrows()
        ]

    # ------------------------------------------------------------------
    # 数据源适配层: 新数据源只需在这里加一个方法
    # ------------------------------------------------------------------
    def fetch_historical(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
        period: str = "daily",
    ) -> pd.DataFrame:
        """
        多数据源自动回退拉取历史行情:
            东财 stock_zh_a_hist -> 新浪 stock_zh_a_daily -> 腾讯 stock_zh_a_hist_tx
        任一数据源成功即返回标准化数据; 全部失败时抛出汇总异常。
        数据源标记写入 df.attrs["source"], 供日志/缓存使用。
        """
        self._require_akshare()
        errors: list[str] = []
        sources = [
            ("eastmoney", lambda: ak.stock_zh_a_hist(
                symbol=symbol, period=period, start_date=start_date,
                end_date=end_date, adjust=adjust,
            )),
            ("sina", lambda: ak.stock_zh_a_daily(
                symbol=self._to_exchange_symbol(symbol),
                start_date=start_date, end_date=end_date, adjust=adjust,
            )),
            ("tencent", lambda: ak.stock_zh_a_hist_tx(
                symbol=self._to_exchange_symbol(symbol),
                start_date=start_date, end_date=end_date, adjust=adjust,
            )),
        ]
        for name, fetcher in sources:
            try:
                raw = fetcher()
                df = self.normalize(raw, code=symbol)
                if df is None or df.empty:
                    errors.append(f"{name}: empty data, try next source")
                    continue
                df.attrs["source"] = name
                return df
            except Exception as exc:
                errors.append(f"{name}: {exc!r}")
        raise RuntimeError("所有数据源均失败: " + "; ".join(errors))

    @staticmethod
    def _to_exchange_symbol(code: str) -> str:
        """6 位代码 -> 带交易所前缀(新浪/腾讯接口需要): sh/sz/bj。"""
        code = str(code).zfill(6)
        if code[0] in ("6", "9"):
            return f"sh{code}"
        if code[0] in ("4", "8"):
            return f"bj{code}"
        return f"sz{code}"

    def normalize(self, raw: pd.DataFrame, code: str) -> pd.DataFrame:
        """
        清洗标准化: 任意来源的原始 DataFrame -> UnifiedBar。

        - 中文字段映射为统一英文字段;
        - 日期解析为 DatetimeIndex 并升序去重;
        - 数值列统一转为 float, 剔除无效行。
        """
        if raw is None or raw.empty:
            return pd.DataFrame(columns=list(UNIFIED_BAR.columns))

        df = raw.rename(columns=_AK_COLUMN_MAP)
        if UNIFIED_BAR.index_name not in df.columns:
            raise ValueError("原始数据缺少日期列, 请检查数据源字段")

        df = df[df[UNIFIED_BAR.index_name].notna()].copy()
        df[UNIFIED_BAR.index_name] = pd.to_datetime(df[UNIFIED_BAR.index_name])
        df = df.set_index(UNIFIED_BAR.index_name)
        df = df[~df.index.duplicated(keep="first")].sort_index()

        keep = [col for col in UNIFIED_BAR.columns if col in df.columns]
        for col in keep:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0.0)
        if "amount" in df.columns:
            df["amount"] = df["amount"].fillna(0.0)

        df.index.name = UNIFIED_BAR.index_name
        df.attrs[UNIFIED_BAR.code_attr] = code
        return df

    def generate_synthetic_bars(
        self, days: Optional[int] = None, symbol: str = "DEMO", seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        生成合成日线行情(几何布朗运动), 用于离线演示/单元测试/CI。
        保证 open/high/low/close 满足 OHLC 约束。
        """
        days = days or self.config.synthetic_days
        rng = np.random.default_rng(seed if seed is not None else self.config.synthetic_seed)

        end = pd.Timestamp.today().normalize()
        index = pd.bdate_range(end=end, periods=days)

        daily_return = rng.normal(0.0004, 0.018, size=days)
        close = 10.0 * np.exp(np.cumsum(daily_return))
        open_ = close * (1.0 + rng.normal(0.0, 0.004, size=days))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.006, size=days)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.006, size=days)))
        volume = rng.integers(1_000_000, 20_000_000, size=days).astype(float)
        amount = volume * close

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
            },
            index=index,
        )
        df.index.name = UNIFIED_BAR.index_name
        df.attrs[UNIFIED_BAR.code_attr] = symbol
        return df

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[pd.Timestamp]:
        """兼容 '20200101' 与 '2020-01-01' 两种格式。"""
        if not value:
            return None
        return pd.to_datetime(str(value))

    @staticmethod
    def _slice(
        df: pd.DataFrame,
        start_ts: Optional[pd.Timestamp],
        end_ts: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        """按时间区间裁剪, 保证返回空 DataFrame 时结构一致。"""
        if df.empty:
            return df
        if start_ts is not None:
            df = df[df.index >= start_ts]
        if end_ts is not None:
            df = df[df.index <= end_ts]
        return df

    def _cache_path(self, symbol: str, start: str, end: str) -> Path:
        """缓存文件命名: 标的 + 区间, 区间变化自动生成新缓存。"""
        return self.paths.cache_dir / f"daily_{symbol}_{start}_{end}.parquet"

    @staticmethod
    def _write_cache(df: pd.DataFrame, path: Path) -> None:
        """
        写缓存: 优先 Parquet(体积小、类型保留); 缺少 pyarrow/fastparquet
        时自动回退 CSV, 保证缓存功能在任何环境都可用, 无需额外依赖。
        """
        try:
            df.to_parquet(path)
            fmt = "parquet"
        except Exception:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, encoding="utf-8")
            fmt = "csv"
        print(f"[DataCenter] 行情已缓存({fmt}): {path.name}")

    def _read_cache(self, path: Path, symbol: str) -> Optional[pd.DataFrame]:
        """
        读缓存: Parquet 优先, CSV 兜底; 文件缺失/损坏时返回 None,
        由调用方触发重新拉取。
        """
        candidates = [path, path.with_suffix(".csv")]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                if candidate.suffix.lower() == ".csv":
                    df = pd.read_csv(candidate, index_col=0, parse_dates=True)
                else:
                    df = pd.read_parquet(candidate)
                if df.empty or UNIFIED_BAR.columns[0] not in df.columns:
                    return None
                # 防御: 部分 parquet 文件索引可能被存为整型列, 统一还原为日期索引
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df.index.name = UNIFIED_BAR.index_name
                df.attrs[UNIFIED_BAR.code_attr] = symbol
                return df
            except Exception as exc:
                print(f"[DataCenter] 缓存读取失败({candidate.name}), 将重新获取: {exc!r}")
        return None

    @staticmethod
    def _require_akshare() -> None:
        if not _HAS_AKSHARE:
            raise RuntimeError(
                "akshare 未安装, 请先执行: pip install akshare"
                " (或使用 generate_synthetic_bars 离线演示)"
            )


def validate_unified(df: pd.DataFrame) -> bool:
    """校验 DataFrame 是否符合 UnifiedBar 契约, 供上层防御式使用。"""
    return (
        isinstance(df.index, pd.DatetimeIndex)
        and df.index.name == UNIFIED_BAR.index_name
        and all(col in df.columns for col in UNIFIED_BAR.columns)
    )
