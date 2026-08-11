"""
生成当前持仓图
==============
从 SQLite 读取建仓记录, 结合本地最新收盘价(行情缓存)计算市值与占比,
输出: output/holdings_YYYYMMDD.png (饼图 + 条形图)。

用法: python make_holdings_chart.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "trading.db"
CACHE = ROOT / "data" / "cache"
OUT_DIR = ROOT / "output"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def load_names() -> dict[str, str]:
    names: dict[str, str] = {}
    p = CACHE / "universe_all.csv"
    if p.exists():
        try:
            df = pd.read_csv(p, dtype=str)
            names = dict(zip(df["code"].astype(str).str.zfill(6), df["name"]))
        except Exception:
            pass
    return names


def latest_close(code: str) -> float | None:
    files = sorted(CACHE.glob(f"daily_{code}_*.parquet"))
    for f in reversed(files):
        try:
            df = pd.read_parquet(f)
            if len(df):
                return float(df["close"].iloc[-1])
        except Exception:
            continue
    return None


def main() -> int:
    conn = sqlite3.connect(str(DB))
    rows = conn.execute(
        "SELECT code, units, price FROM trades WHERE reason='initial_build' ORDER BY code"
    ).fetchall()
    conn.close()
    if not rows:
        print("未找到建仓记录(initial_build)")
        return 1

    names = load_names()
    data = []
    for code, units, cost in rows:
        close = latest_close(str(code)) or float(cost)
        value = units * close
        data.append(
            {
                "code": str(code),
                "name": names.get(str(code), str(code)),
                "units": int(units),
                "cost": round(float(cost), 3),
                "last_close": round(close, 3),
                "value": round(value, 2),
            }
        )
    df = pd.DataFrame(data)
    df["weight"] = df["value"] / df["value"].sum()
    df = df.sort_values("value", ascending=False).reset_index(drop=True)

    labels = [f"{r['name']}\n{r['code']}" for r in df.to_dict("records")]
    colors = plt.cm.tab20.colors[: len(df)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.pie(
        df["weight"],
        labels=labels,
        autopct="%.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax1.set_title("持仓市值占比", fontsize=12)

    ax2.barh(df["code"], df["value"], color=colors)
    ax2.set_title("持仓市值（元）", fontsize=12)
    ax2.invert_yaxis()
    for i, r in df.iterrows():
        ax2.text(
            r["value"] * 1.01,
            i,
            f"{r['value']:,.0f} 元 · {r['units']:,}股",
            va="center",
            fontsize=9,
        )

    total = df["value"].sum()
    fig.suptitle(
        f"当前持仓（建仓日 2026-08-10 · 共 {len(df)} 只 · 总市值 {total:,.0f} 元）",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "holdings_20260810.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"已生成: {out}")
    print(df[["code", "name", "units", "cost", "last_close", "value", "weight"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
