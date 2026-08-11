"""
每日持仓与收益图表
==================
生成一张组合图:
    左侧: 当前持仓分布饼图(按最新市值)
    右侧: 账户权益 vs 沪深300 收益折线
供日报邮件附件使用, 也可单独运行 make_holdings_chart.py 查看。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


_CJK_FONTS = ("Noto Sans CJK SC", "Microsoft YaHei", "SimHei")


def _cjk_available() -> bool:
    """检测系统是否有可用的中文字体(容器内通常没有, 自动切英文标签)。"""
    for name in _CJK_FONTS:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return True
        except Exception:
            continue
    return False


_USE_CN = _cjk_available()
plt.rcParams["font.sans-serif"] = list(_CJK_FONTS) + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _load_names(cache_dir: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    p = cache_dir / "universe_all.csv"
    if p.exists():
        try:
            df = pd.read_csv(p, dtype=str)
            names = dict(zip(df["code"].astype(str).str.zfill(6), df["name"]))
        except Exception:
            pass
    return names


def _holding_price(code: str, cache_dir: Path, conn: sqlite3.Connection) -> float | None:
    """优先取缓存最新收盘价, 否则取该标的最新一笔成交价。"""
    for f in sorted(cache_dir.glob(f"daily_{code}_*.parquet")):
        try:
            df = pd.read_parquet(f)
            if len(df):
                return float(df["close"].iloc[-1])
        except Exception:
            continue
    row = conn.execute(
        "SELECT price FROM trades WHERE code = ? ORDER BY datetime DESC, id DESC LIMIT 1",
        (code,),
    ).fetchone()
    return float(row[0]) if row else None


def build_daily_chart(
    db_path: Path | str,
    cache_dir: Path | str,
    out_path: Path | str,
) -> Path:
    """生成持仓饼图 + 收益折线图, 返回图片路径。"""
    db_path = Path(db_path)
    cache_dir = Path(cache_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    names = _load_names(cache_dir)

    # 当前持仓: 按成交净额计算股数
    net: dict[str, int] = {}
    for code, side, units in conn.execute("SELECT code, side, units FROM trades"):
        sign = 1 if side in ("BUY", "SELL_SHORT") else -1
        net[code] = net.get(code, 0) + sign * int(units)

    holdings = []
    for code, units in net.items():
        if units <= 0:
            continue
        price = _holding_price(str(code), cache_dir, conn)
        if price is None:
            continue
        holdings.append(
            {
                "code": str(code),
                "name": names.get(str(code), str(code)),
                "units": units,
                "value": units * price,
            }
        )
    holdings_df = pd.DataFrame(holdings).sort_values("value", ascending=False)

    # 权益曲线(与基准对比)
    eq = pd.read_sql_query(
        "SELECT date, equity, benchmark_return FROM equity_daily ORDER BY date",
        conn,
    )
    conn.close()
    if not len(eq):
        raise ValueError("equity_daily 无数据, 无法生成图表")
    eq["date"] = pd.to_datetime(eq["date"])
    base = float(eq["equity"].iloc[0])
    eq["bench_equity"] = base * (1.0 + eq["benchmark_return"].fillna(0.0)).cumprod()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左侧: 持仓饼图
    if len(holdings_df):
        labels = []
        for r in holdings_df.to_dict("records"):
            if _USE_CN:
                labels.append(f"{r['name']} {r['code']}\n{r['value']:,.0f}元")
            else:
                labels.append(f"{r['code']}\n{r['value']:,.0f} CNY")
        ax1.pie(
            holdings_df["value"],
            labels=labels,
            autopct="%.1f%%",
            startangle=90,
            textprops={"fontsize": 9},
        )
        ax1.set_title("当前持仓分布" if _USE_CN else "Holdings Distribution", fontsize=13)
    else:
        ax1.text(0.5, 0.5, "当前无持仓", ha="center", va="center", fontsize=13)
        ax1.set_title("当前持仓分布" if _USE_CN else "Holdings Distribution", fontsize=13)

    # 右侧: 收益折线
    ax2.plot(
        eq["date"],
        eq["equity"],
        label="策略账户" if _USE_CN else "Strategy",
        color="#2f6fbf",
        linewidth=2,
    )
    ax2.plot(
        eq["date"],
        eq["bench_equity"],
        label="沪深300" if _USE_CN else "CSI300",
        color="#d9822b",
        linewidth=1.6,
        linestyle="--",
    )
    ax2.set_title(
        "账户收益 vs 沪深300" if _USE_CN else "Account Equity vs CSI300",
        fontsize=13,
    )
    ax2.set_ylabel("权益(元)" if _USE_CN else "Equity (CNY)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    last = eq.iloc[-1]
    fig.suptitle(
        (
            f"QuantTrader 日报图表 · 最新交易日 {last['date'].strftime('%Y-%m-%d')} · "
            f"权益 {last['equity']:,.0f}元"
            if _USE_CN
            else f"QuantTrader Daily Chart · {last['date'].strftime('%Y-%m-%d')} · "
            f"Equity {last['equity']:,.0f} CNY"
        ),
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
