"""
六方案每日评估与日报
====================
每天 16:00 结算后运行:
    1. 加载 6 个训练好的方案(3市场 x 全仓/补仓)配置
    2. 对每个方案持有的 Top-K 标的逐只回测到当天, 得到当日组合收益
    3. 权益按"昨日权益 x (1+当日收益)"记账到 scheme_equity_daily
    4. 生成 6 张方案权益曲线图 + 1 张总对比图
    5. DeepSeek 写六方案日报并发送邮件(附 7 张图)
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from qtcore.backtest.engine import BacktestEngine
from qtcore.config import AppConfig
from qtcore.datacenter.data_center import DataCenter
from qtcore.emailer import send_email
from qtcore.report_writer import _ask
from qtcore.store import Store
from qtcore.strategy import create_strategy


ROOT = Path(__file__).resolve().parent.parent
SCHEMES_DIR = ROOT / "output" / "schemes"
OUT_DIR = ROOT / "output" / "schemes" / "daily"
DATA_START = "20200101"
INTRADAY_DATA_START = "20240801"   # 免费源(如 TwelveData)美股/港股小时线深度约 2 年
INITIAL = 1_000_000.0


def load_schemes() -> list[dict[str, Any]]:
    """读取训练好的方案配置(日线 6 方案 + 美股/港股日内 4 方案)。"""
    schemes = []
    patterns = ("*_full.json", "*_staged.json", "*_full_intraday.json", "*_staged_intraday.json")
    seen: set[str] = set()
    files: list[Path] = []
    for pat in patterns:
        for p in SCHEMES_DIR.glob(pat):
            if p not in files:
                files.append(p)
    for f in sorted(files, key=lambda p: p.name):
        data = json.loads(f.read_text(encoding="utf-8"))
        scheme_id = Path(f).stem  # 如 us_full / us_full_intraday, 保证日线与日内分账
        if scheme_id in seen:
            continue
        seen.add(scheme_id)
        data["scheme_id"] = scheme_id
        schemes.append(data)
    return schemes


def _bt_config(params: dict[str, Any], app: AppConfig):
    bt = replace(app.backtest)
    bt.initial_capital = INITIAL
    for key in (
        "position_ratio", "max_position_ratio", "rebalance", "order_type",
        "slippage_tolerance_pct", "leverage", "stop_loss_pct", "take_profit_pct",
        "max_drawdown_halt", "halt_cooldown_days", "halt_resume_drawdown",
    ):
        if key in params:
            setattr(bt, key, params[key])
    return bt


def _eval_one(
    scheme: dict[str, Any],
    dc: DataCenter,
    d_str: str,
) -> dict[str, Any]:
    """单个方案当日评估: Top-K 等权组合当日收益。"""
    top = [str(c) for c in str(scheme.get("top_symbols", "")).split(",") if c]
    market = str(scheme.get("market", "cn"))
    timeframe = str(scheme.get("timeframe", "daily"))
    data_start = INTRADAY_DATA_START if timeframe != "daily" else DATA_START
    app = AppConfig()
    bt = _bt_config(scheme, app)
    rets = []
    for code in top:
        try:
            bars = dc.get_bars(code, data_start, d_str, timeframe, market)
            if bars is None or len(bars) < 60:
                continue
            result = BacktestEngine(bt).run(bars, create_strategy("ma_cross", scheme))
            rets.append(float(result.equity_curve["daily_return"].iloc[-1]))
        except Exception as exc:
            print(f"[Scheme] {market}/{code} 评估失败: {exc!r}")
    port_ret = float(pd.Series(rets).mean()) if rets else 0.0
    return {
        "scheme": str(scheme.get("scheme_id") or f"{market}_{scheme.get('position_mode', 'full')}"),
        "market": market,
        "mode": str(scheme.get("position_mode", "full")),
        "top_symbols": ",".join(top),
        "daily_return": port_ret,
        "n_symbols": len(rets),
    }


def run_schemes_daily(
    store: Store,
    dc: DataCenter,
    today: date,
    d_str: str,
    email_cfg: dict[str, Any] | None = None,
    send_email: bool = True,
) -> dict[str, Any]:
    """评估 6 方案、记账、出图; send_email=True 时单独发送六方案日报邮件。"""
    schemes = load_schemes()
    if not schemes:
        raise RuntimeError("未找到训练好的方案配置(output/schemes)")
    d_iso = today.strftime("%Y-%m-%d")
    rows = []
    for s in schemes:
        r = _eval_one(s, dc, d_str)
        prev = store.prev_scheme_equity(r["scheme"], d_iso) or INITIAL
        equity = prev * (1.0 + r["daily_return"])
        r["equity"] = equity
        r["cum_return"] = equity / INITIAL - 1.0
        store.save_scheme_equity(r["scheme"], d_iso, equity, r["daily_return"], r["top_symbols"])
        rows.append(r)
        print(f"[Scheme] {r['scheme']}: 当日 {r['daily_return']:+.2%} 累计 {r['cum_return']:+.2%}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart_paths = _charts(rows, store)
    report = _report_text(rows)
    subject = f"[QuantTrader] 六方案模拟日报 {d_iso}"
    if email_cfg and send_email:
        send_email(subject, report, config=email_cfg, attachments=chart_paths)
        store.save_report(d_iso, "schemes_daily", subject, report)
        print(f"[Scheme] 六方案日报已发送: {subject}")
    return {"rows": rows, "charts": chart_paths, "report": report}


def _charts(rows: list[dict[str, Any]], store: Store) -> list[Path]:
    """6 张方案权益曲线 + 1 张总对比图。"""
    paths: list[Path] = []
    for r in rows:
        series = store.scheme_equity_series(r["scheme"])
        fig, ax = plt.subplots(figsize=(9, 4))
        if series:
            df = pd.DataFrame(series)
            df["date"] = pd.to_datetime(df["date"])
            ax.plot(df["date"], df["equity"], color="#2f6fbf", linewidth=2)
        ax.text(0.02, 0.95, f"Today {r['daily_return']:+.2%} | Cum {r['cum_return']:+.2%}",
                transform=ax.transAxes, fontsize=10, va="top")
        ax.set_title(f"{r['market'].upper()}-{r['mode']} Daily Equity")
        ax.set_ylabel("Equity (CNY)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = OUT_DIR / f"daily_{r['scheme']}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    # 总对比图
    fig, ax = plt.subplots(figsize=(11, 5))
    for r in rows:
        series = store.scheme_equity_series(r["scheme"])
        if not series:
            continue
        df = pd.DataFrame(series)
        df["date"] = pd.to_datetime(df["date"])
        ax.plot(df["date"], df["equity"] / INITIAL,
                label=f"{r['market'].upper()}-{r['mode']}", linewidth=1.6)
    ax.set_title("6 Schemes Equity (normalized)")
    ax.set_ylabel("Net Value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUT_DIR / "daily_overall.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


def _report_text(rows: list[dict[str, Any]]) -> str:
    table = "\n".join(
        f"- {r['market'].upper()}-{r['mode']}: 当日 {r['daily_return']:+.2%}, "
        f"累计 {r['cum_return']:+.2%}, 持仓 {r['top_symbols']}"
        for r in rows
    )
    try:
        return _ask(
            "你是量化交易助手, 用简洁中文写六方案日报: 今日各方案表现、哪个最好/最差、"
            "与昨日相比有无异常、风险提示。约300字, 不构成投资建议。",
            "六方案今日数据:\n" + table,
        )
    except Exception as exc:
        return "六方案日报(DeepSeek 生成失败):\n" + table + f"\n({exc!r})"
