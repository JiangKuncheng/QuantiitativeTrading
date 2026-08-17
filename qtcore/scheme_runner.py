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
SCHEME_DATA_START = "20240101"   # 六方案评估起点: 与训练测试窗口一致(2024起重新起跑), 避免2021-2023熔断状态永久带入
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
    prev_equity: float | None = None,
) -> dict[str, Any]:
    """
    单个方案当日评估:
        - 每个标的按"方案昨日权益 / K"等权分配模拟资金;
        - 运行 BacktestEngine(收盘出信号 -> 次日开盘价成交, 含滑点/佣金/止损/熔断);
        - 返回当日收益、今日成交明细(开盘价成交)与收盘持仓快照。
    """
    top = [str(c) for c in str(scheme.get("top_symbols", "")).split(",") if c]
    market = str(scheme.get("market", "cn"))
    timeframe = str(scheme.get("timeframe", "daily"))
    data_start = INTRADAY_DATA_START if timeframe != "daily" else SCHEME_DATA_START
    app = AppConfig()
    per_symbol_capital = (prev_equity or INITIAL) / max(1, len(top))
    trades_today: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    rets = []
    for code in top:
        try:
            bars = dc.get_bars(code, data_start, d_str, timeframe, market)
            if bars is None or len(bars) < 60:
                continue
            bt = _bt_config(scheme, app)
            bt.initial_capital = per_symbol_capital
            result = BacktestEngine(bt).run(bars, create_strategy("ma_cross", scheme))
            rets.append(float(result.equity_curve["daily_return"].iloc[-1]))

            # 今日成交: 昨日收盘信号 -> 今日开盘价成交
            if result.trades is not None and not result.trades.empty:
                tr = result.trades.copy()
                tr["date"] = tr["datetime"].dt.date.astype(str)
                today_tr = tr[tr["date"] == d_str]
                for _, row in today_tr.iterrows():
                    trades_today.append(
                        {
                            "code": str(row["code"]),
                            "side": str(row["side"]),
                            "units": int(row["units"]),
                            "price": float(row["price"]),
                            "commission": float(row.get("commission") or 0.0),
                            "pnl": None if row.get("pnl") is None else float(row["pnl"]),
                            "reason": str(row.get("reason") or ""),
                        }
                    )

            # 收盘持仓快照: 由全历史成交净额还原 + 最新收盘价
            net: dict[str, dict[str, Any]] = {}
            for _, row in result.trades.iterrows():
                c = str(row["code"])
                side = str(row["side"])
                units = int(row["units"])
                price = float(row["price"])
                e = net.setdefault(c, {"units": 0, "buy_cost": 0.0, "buy_units": 0})
                if side == "BUY":
                    e["units"] += units
                    e["buy_cost"] += units * price
                    e["buy_units"] += units
                elif side == "SELL":
                    e["units"] -= units
            last_close = float(bars["close"].iloc[-1])
            for c, e in net.items():
                if e["units"] <= 0:
                    continue
                avg_price = e["buy_cost"] / e["buy_units"] if e["buy_units"] else 0.0
                positions.append(
                    {
                        "code": c,
                        "units": e["units"],
                        "avg_price": round(avg_price, 4),
                        "last_price": round(last_close, 4),
                        "market_value": round(e["units"] * last_close, 2),
                    }
                )
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
        "trades": trades_today,
        "positions": positions,
    }


def run_schemes_daily(
    store: Store,
    dc: DataCenter,
    today: date,
    d_str: str,
    email_cfg: dict[str, Any] | None = None,
    send_mail: bool = True,
) -> dict[str, Any]:
    """评估 6 方案、记账、出图; send_mail=True 时单独发送六方案日报邮件。"""
    schemes = load_schemes()
    if not schemes:
        raise RuntimeError("未找到训练好的方案配置(output/schemes)")
    d_iso = today.strftime("%Y-%m-%d")
    rows = []
    for s in schemes:
        scheme_id = str(s.get("scheme_id") or f"{s.get('market')}_{s.get('position_mode', 'full')}")
        prev = store.prev_scheme_equity(scheme_id, d_iso) or INITIAL
        r = _eval_one(s, dc, d_str, prev_equity=prev)
        equity = prev * (1.0 + r["daily_return"])
        r["equity"] = equity
        r["cum_return"] = equity / INITIAL - 1.0
        store.save_scheme_equity(r["scheme"], d_iso, equity, r["daily_return"], r["top_symbols"])
        store.save_scheme_trades(r["scheme"], d_iso, r.get("trades", []))
        store.save_scheme_positions(r["scheme"], d_iso, r.get("positions", []))
        rows.append(r)
        op = f", 成交 {len(r.get('trades', []))} 笔, 持仓 {len(r.get('positions', []))} 只"
        print(f"[Scheme] {r['scheme']}: 当日 {r['daily_return']:+.2%} 累计 {r['cum_return']:+.2%}{op}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart_paths = _charts(rows, store)
    report = _report_text(rows)
    subject = f"[QuantTrader] 六方案模拟日报 {d_iso}"
    if email_cfg and send_mail:
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
    def _clean(x: Any) -> Any:
        """把 NaN 转成 None, 保证 JSON 合法。"""
        try:
            if x is not None and x != x:  # NaN
                return None
        except Exception:
            pass
        return x

    def _deep_clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _deep_clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_deep_clean(v) for v in obj]
        return _clean(obj)

    def _ops(r: dict[str, Any]) -> str:
        t = r.get("trades", [])
        if not t:
            return "无成交"
        return "; ".join(
            f"{x['code']} {'买入' if x['side'] == 'BUY' else '卖出'} {x['units']}股@{x['price']:.2f}"
            for x in t
        )

    def _pos(r: dict[str, Any]) -> str:
        p = r.get("positions", [])
        if not p:
            return "空仓"
        return ", ".join(f"{x['code']} {x['units']}股(成本{x['avg_price']:.2f})" for x in p)

    table = "\n".join(
        f"- {r['scheme']}: 当日 {r['daily_return']:+.2%}, 累计 {r['cum_return']:+.2%}, "
        f"权益 {r['equity']:,.0f}, 今日操作: {_ops(r)}, 收盘持仓: {_pos(r)}"
        for r in rows
    )
    payload = [
        {
            "scheme": r["scheme"],
            "daily_return": r["daily_return"],
            "cum_return": r["cum_return"],
            "equity": r["equity"],
            "trades_today": r.get("trades", []),
            "positions": r.get("positions", []),
        }
        for r in rows
    ]
    detail = json.dumps(_deep_clean(payload), ensure_ascii=False)
    try:
        return _ask(
            "你是量化交易助手, 用简洁中文写六方案日报: 今日各方案表现、哪个最好/最差、"
            "今日各方案模拟成交了什么(按昨日信号今日开盘价成交)、当前持仓、风险提示。"
            "约400字, 不构成投资建议。",
            "六方案今日数据:\n" + table + "\n详细成交与持仓:\n" + detail,
        )
    except Exception as exc:
        return "六方案日报(DeepSeek 生成失败, 结构化摘要):\n" + table + f"\n({exc!r})"
