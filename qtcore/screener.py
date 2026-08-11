"""
StockScreener 选股模块
======================

两级选股流水线:
    第一级 初筛(全市场):
        akshare 全 A 股票列表 + 实时快照, 过滤 ST / 价格区间 / 成交额流动性
    第二级 精选(因子回测):
        对候选股逐个跑双均线策略完整回测(含佣金/滑点), 按绩效指标排序

也支持"指定股票池": 给定代码列表, 直接跳过全市场初筛, 只在这几只里挑。

输出:
    output/screen_result.csv —— 全部候选的绩效排名
    终端打印 Top N

注意:
- 全市场逐只拉历史数据较慢, 建议先用 --limit 限制候选数量(按成交额取前 N);
- 个股数据走 DataCenter 缓存, 第二次运行同一股票池会命中 Parquet, 速度大幅提升;
- 网络不可用时 --offline 模式用合成数据跑通全流程(仅演示)。
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

try:
    import akshare as ak

    _HAS_AKSHARE = True
except ImportError:  # pragma: no cover
    ak = None
    _HAS_AKSHARE = False

from qtcore.backtest.engine import BacktestEngine
from qtcore.config import AppConfig
from qtcore.datacenter.data_center import DataCenter
from qtcore.strategy import create_strategy


PERCENT_KEYS = {"total_return", "annual_return", "max_drawdown", "win_rate"}


class StockScreener:
    """全市场/股票池选股器。"""

    def __init__(
        self,
        config: AppConfig,
        fast: int = 5,
        slow: int = 20,
        synthetic: bool = False,
        universe: str = "all",
        use_snapshot: bool = True,
    ) -> None:
        self.config = config
        self.fast = int(fast)
        self.slow = int(slow)
        self.synthetic = synthetic  # True 时全程用合成数据(离线演示/CI)
        self.universe = universe    # "all" 全市场 / "csi300" 沪深300成分股
        self.use_snapshot = use_snapshot  # 是否尝试东财快照(该接口在本机可能原生崩溃)
        self.dc = DataCenter(config.data, config.paths)

    # ------------------------------------------------------------------
    # 第一级: 全市场初筛
    # ------------------------------------------------------------------
    def filter_universe(
        self,
        min_price: float = 2.0,
        max_price: float = 200.0,
        min_amount: float = 2e8,
    ) -> pd.DataFrame:
        """
        初筛: 全 A 股票 -> 排除 ST/退市 -> 价格区间 -> 成交额流动性。
        返回 DataFrame: code, name, price, amount, market_cap, 按成交额降序。
        """
        if self.synthetic or not _HAS_AKSHARE:
            return self._synthetic_universe()
        if self.universe == "csi300":
            return self._fetch_csi300()
        if not self.use_snapshot:
            # 快照接口不可用(东财被拦/原生崩溃)时, 直接走全市场代码列表初筛
            return self._universe_code_list()

        try:
            snapshot = self.fetch_snapshot()
            if snapshot is None or snapshot.empty:
                raise RuntimeError("快照为空")
            df = snapshot[
                ~snapshot["name"].str.contains("ST|退", na=False)
                & snapshot["price"].between(min_price, max_price)
                & (snapshot["amount"] >= min_amount)
            ]
            df = df.sort_values("amount", ascending=False).reset_index(drop=True)
            print(
                f"[Screener] 全市场 {len(snapshot)} 只 -> 初筛后 {len(df)} 只"
                f" (排除ST, 价格 {min_price}~{max_price}, 成交额 >= {min_amount / 1e8:.1f} 亿)"
            )
            return df
        except Exception as exc:
            # 快照接口(东财 clist)在某些网络不可用: 降级为全市场代码列表,
            # 仍按名称排除 ST/退, 但跳过价格/成交额过滤
            print(f"[Screener] 快照接口不可用({exc!r}), 降级为全市场代码列表初筛")
            return self._universe_code_list()

    def _universe_code_list(self) -> pd.DataFrame:
        """
        全市场代码列表初筛: 排除 ST/退市/北交所, 保留沪深主板/创业板/科创板。
        获取链路: 本地缓存 -> akshare 全市场列表 -> 沪深交易所官方列表。
        """
        universe_cache = self.config.paths.cache_dir / "universe_all.csv"
        universe: pd.DataFrame | None = None

        # 1) 本地缓存(网络不可用时保证可复现)
        if universe_cache.exists():
            try:
                universe = pd.read_csv(universe_cache, dtype={"code": str})
                print(f"[Screener] 全市场列表使用本地缓存: {len(universe)} 只")
            except Exception:
                universe = None

        # 2) akshare 全市场列表(含北交所, 需 bse.cn 可达)
        if universe is None:
            try:
                universe = self._retry(lambda: ak.stock_info_a_code_name())
            except Exception as exc1:
                print(f"[Screener] 全市场列表接口失败({exc1!r}), 尝试沪深交易所官方列表")
                # 3) 沪深交易所官方列表(上交所 + 深交所)
                try:
                    sh = self._retry(lambda: self._name_code_from(ak.stock_info_sh_name_code()))
                    sz = self._retry(lambda: self._name_code_from(ak.stock_info_sz_name_code()))
                    universe = pd.concat([sh, sz], ignore_index=True)
                    print(f"[Screener] 沪深交易所官方列表: {len(universe)} 只")
                except Exception as exc2:
                    # 4) 最后兜底: 沪深300+中证500+中证1000 成分并集(约 1800+ 只)
                    try:
                        universe = self._fetch_csi_union()
                        print(
                            f"[Screener] 指数成分并集兜底: {len(universe)} 只"
                            f" (akshare: {str(exc1)[:50]}; 交易所: {str(exc2)[:50]})"
                        )
                    except Exception as exc3:
                        raise RuntimeError(
                            f"无法获取股票列表(akshare: {exc1!r}; 交易所: {exc2!r}; 指数: {exc3!r})"
                        ) from exc3

        # 缓存成功获取的列表, 后续运行离线可用
        try:
            universe.to_csv(universe_cache, index=False, encoding="utf-8")
        except Exception:
            pass

        df = universe[~universe["name"].str.contains("ST|退", na=False)].copy()
        # 排除北交所(4/8 开头), 保留沪市(6: 主板/科创板)与深市(0/3: 主板/创业板)
        df = df[df["code"].astype(str).str.zfill(6).str.startswith(("6", "0", "3"))].copy()
        df = df.assign(
            price=np.nan,
            amount=np.nan,
            market_cap=np.nan,
        )
        df = df.sort_values("code").reset_index(drop=True)
        print(
            f"[Screener] 全市场 {len(universe)} 只 -> 初筛后 {len(df)} 只"
            f" (代码列表模式: 排除 ST/退/北交所, 未过滤价格与成交额)"
        )
        return df

    @staticmethod
    def _retry(fn, tries: int = 3, backoff: float = 3.0):
        """网络接口重试: 指数退避, 用于股票列表等一次性请求。"""
        last: Exception | None = None
        for i in range(tries):
            try:
                return fn()
            except Exception as exc:
                last = exc
                if i < tries - 1:
                    time.sleep(backoff * (i + 1))
        raise last  # type: ignore[misc]

    def _fetch_csi_union(self) -> pd.DataFrame:
        """沪深300 + 中证500 + 中证1000 成分并集(宽基兜底股票池)。"""
        frames = []
        for index_code in ("000300", "000905", "000852"):
            raw = self._retry(lambda c=index_code: ak.index_stock_cons_csindex(symbol=c))
            frames.append(
                raw.rename(columns={"成分券代码": "code", "成分券名称": "name"})[["code", "name"]]
            )
        union = pd.concat(frames, ignore_index=True).drop_duplicates(subset="code")
        union["code"] = union["code"].astype(str).str.zfill(6)
        return union.reset_index(drop=True)

    @staticmethod
    def _name_code_from(df: pd.DataFrame) -> pd.DataFrame:
        """从交易所列表 DataFrame 中按列名模糊匹配 代码/名称 两列。"""
        code_col = next((c for c in df.columns if "代码" in str(c)), None)
        name_col = next(
            (c for c in df.columns if ("简称" in str(c) or "名称" in str(c))),
            None,
        )
        if code_col is None or name_col is None:
            raise ValueError(f"无法识别列表列名: {list(df.columns)}")
        out = df[[code_col, name_col]].rename(
            columns={code_col: "code", name_col: "name"}
        )
        out["code"] = out["code"].astype(str).str.zfill(6)
        return out

    def _fetch_csi300(self) -> pd.DataFrame:
        """
        获取沪深300成分股(代表性大盘池):
        优先 csindex 官方列表, 失败回退新浪列表, 再失败降级合成池。
        """
        df: pd.DataFrame | None = None
        try:
            raw = self._retry(lambda: ak.index_stock_cons_csindex(symbol="000300"))
            df = raw.rename(columns={"成分券代码": "code", "成分券名称": "name"})[["code", "name"]]
            source = "csindex"
        except Exception as exc:
            print(f"[Screener] csindex 成分股获取失败({exc!r}), 尝试新浪接口")
            try:
                raw = self._retry(lambda: ak.index_stock_cons(symbol="000300"))
                df = raw.rename(columns={"品种代码": "code", "品种名称": "name"})[["code", "name"]]
                source = "sina"
            except Exception as exc2:
                print(f"[Screener] 成分股接口均失败({exc2!r}), 降级为合成池")
                return self._synthetic_universe()

        df = df.copy()
        df["code"] = df["code"].astype(str).str.zfill(6)
        df = df.assign(price=np.nan, amount=np.nan, market_cap=np.nan)
        df = df.reset_index(drop=True)
        print(f"[Screener] CSI300 成分股 {len(df)} 只 (来源: {source})")
        return df

    def fetch_snapshot(self) -> pd.DataFrame:
        """全 A 实时快照(最新价/成交额/总市值), 一次接口调用。"""
        raw = ak.stock_zh_a_spot_em()
        column_map = {
            "代码": "code",
            "名称": "name",
            "最新价": "price",
            "成交额": "amount",
            "总市值": "market_cap",
        }
        df = raw.rename(columns=column_map)
        for col in ("price", "amount", "market_cap"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["code", "name", "price", "amount", "market_cap"]]

    def from_symbols(self, symbols: list[str]) -> pd.DataFrame:
        """指定股票池模式: 只在这些代码里挑, 名称以代码代替。"""
        return pd.DataFrame({"code": symbols, "name": symbols})

    # ------------------------------------------------------------------
    # 第二级: 精选(逐只回测排序)
    # ------------------------------------------------------------------
    def rank(
        self,
        candidates: pd.DataFrame,
        limit: int = 30,
        metric: str = "sharpe",
        workers: int = 4,
    ) -> pd.DataFrame:
        """
        对候选并发逐只跑双均线回测(IO 密集, 多线程加速), 按 metric 降序排序。
        单只失败自动跳过(记录原因), 不影响整体流程。
        """
        rows: list[dict[str, Any]] = []
        skipped = 0
        fail_printed = 0
        data_cfg = (
            replace(self.config.data, offline_fallback=False)
            if not self.synthetic
            else self.config.data
        )

        def _process(row: Any) -> tuple[str, str, str, Any]:
            code = str(row["code"])
            name = str(row.get("name", code))
            try:
                bars = self._load_bars(code, data_cfg)
                if bars is None or len(bars) < 120:
                    return ("skip", code, name, None)
                strategy = create_strategy("ma_cross", {"fast": self.fast, "slow": self.slow})
                stats = BacktestEngine(self.config.backtest).run(bars, strategy).stats
                return ("ok", code, name, stats)
            except Exception as exc:
                return ("fail", code, name, exc)

        total = min(limit, len(candidates))
        lock = threading.Lock()
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process, row) for _, row in candidates.head(limit).iterrows()]
            for future in as_completed(futures):
                status, code, name, payload = future.result()
                with lock:
                    done += 1
                    if status == "ok":
                        rows.append({"code": code, "name": name, **payload})
                    elif status == "fail":
                        skipped += 1
                        if fail_printed < 20:
                            print(f"[Screener] {code} {name} 回测失败, 跳过: {payload!r}")
                            fail_printed += 1
                    if done % 200 == 0 or done == total:
                        print(f"[Screener] 进度 {done}/{total} (成功 {len(rows)}, 失败/跳过 {skipped})")

        if not rows:
            print("[Screener] 没有任何候选完成回测, 请放宽筛选条件或检查数据")
            return pd.DataFrame()

        result = pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)
        print(f"[Screener] 完成 {len(rows)} 只, 失败/跳过 {skipped} 只, 排序指标: {metric}")
        return result

    def _load_bars(self, code: str, data_cfg) -> pd.DataFrame:
        """加载单只行情: 合成模式直接生成; 真实模式走 DataCenter(含缓存)。"""
        if self.synthetic:
            # 按代码派生种子, 让合成标的之间有不同的行情走势
            seed = sum(ord(ch) for ch in code) % 10000 or 1
            return self.dc.generate_synthetic_bars(symbol=code, seed=seed)
        return DataCenter(data_cfg, self.config.paths).get_daily_bars(symbol=code)

    @staticmethod
    def _synthetic_universe(size: int = 20) -> pd.DataFrame:
        """离线演示用合成股票池。"""
        return pd.DataFrame(
            {
                "code": [f"SYN{i:06d}" for i in range(1, size + 1)],
                "name": [f"模拟标的{i}" for i in range(1, size + 1)],
            }
        )


def format_row(row: pd.Series) -> str:
    """格式化一行选股结果用于终端展示。"""
    parts = [f"{row['code']} {str(row['name']):<10}"]
    for key in ("total_return", "annual_return", "sharpe", "max_drawdown", "win_rate", "profit_factor", "n_trades"):
        value = row[key]
        if key in PERCENT_KEYS:
            parts.append(f"{key}={value:.2%}")
        elif key == "sharpe":
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)
