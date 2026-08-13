"""
6 方案训练: 3 市场(A股/美股/港股) x 2 玩法(全仓/补仓)
====================================================
每个方案: DeepSeek 调参 -> 训练集选股(Top-K)+评估 -> 测试集样本外评估
输出:
    output/schemes/<market>_<mode>.json   每个方案最优配置与绩效
    output/schemes/chart_<market>_<mode>.png  每方案权益曲线图(6张)
    output/schemes/overall_chart.png          6方案对比图
    output/schemes/report.md                  总体报告(含DeepSeek分析)

用法: python train_schemes.py [--rounds 4] [--markets cn us hk] [--modes full staged]
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from qtcore.config import AppConfig
from qtcore.deepseek_tuner import DeepSeekTuner, load_api_key
from qtcore.report_writer import _ask
from qtcore.trainer import PortfolioEvaluator, TrainingConfig


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "schemes"

TRAIN_WINDOW = ("20200101", "20221231")
TEST_WINDOW = ("20240101", "20260810")

MARKET_NAMES = {"cn": "A股", "us": "美股", "hk": "港股"}
MODE_NAMES = {"full": "全仓", "staged": "补仓"}


def load_pool(market: str, app: AppConfig) -> list[str]:
    pools = json.loads((ROOT / "config" / "pools.json").read_text(encoding="utf-8"))
    if market == "cn":
        cache = ROOT / "data" / "pool_csi300.csv"
        if cache.exists():
            df = pd.read_csv(cache, dtype=str)
            return [str(c) for c in df["code"]]
        from qtcore.screener import StockScreener

        screener = StockScreener(app, synthetic=False, universe="csi300", use_snapshot=False)
        cands = screener.filter_universe()
        codes = [str(c) for c in cands.head(int(pools["cn"]["size"]))["code"]]
        pd.DataFrame({"code": codes}).to_csv(cache, index=False, encoding="utf-8")
        return codes
    symbols = pools[market]["symbols"]
    return list(dict.fromkeys(symbols))  # 去重保序


def sanitize(p: dict[str, Any], market: str, mode: str) -> dict[str, Any]:
    lo, hi = 0.2, 1.0

    def cl(v, a, b):
        return max(a, min(b, v))

    fast = int(cl(p.get("fast", 5), 3, 60))
    slow = int(cl(p.get("slow", 25), 10, 120))
    if slow <= fast:
        slow = fast + 10
    allowed_tf = ("daily", "60min", "2h", "4h") if market == "cn" else ("daily",)
    tf = p.get("timeframe", "daily")
    if tf not in allowed_tf:
        tf = "daily"
    out = {
        "market": market,
        "position_mode": mode,
        "fast": fast,
        "slow": slow,
        "timeframe": tf,
        "use_rsi": bool(p.get("use_rsi", False)),
        "top_k": int(cl(p.get("top_k", 5), 2, 10)),
        "select_metric": p.get("select_metric", "sharpe")
        if p.get("select_metric", "sharpe") in ("sharpe", "total_return", "profit_factor")
        else "sharpe",
        "position_ratio": float(cl(p.get("position_ratio", 0.95), 0.1, 1.0)),
        "max_position_ratio": float(cl(p.get("max_position_ratio", 1.0), 0.2, 1.0)),
        "rebalance": p.get("rebalance", "daily")
        if p.get("rebalance", "daily") in ("daily", "weekly", "monthly")
        else "daily",
        "order_type": p.get("order_type", "market")
        if p.get("order_type", "market") in ("market", "limit")
        else "market",
        "slippage_tolerance_pct": float(cl(p.get("slippage_tolerance_pct", 0.0), 0.0, 0.01)),
        "leverage": float(cl(p.get("leverage", 1.0), 1.0, 2.0)),
        "stop_loss_pct": float(cl(p.get("stop_loss_pct", 0.0), 0.0, 0.2)),
        "take_profit_pct": float(cl(p.get("take_profit_pct", 0.0), 0.0, 0.5)),
        "max_drawdown_halt": float(cl(p.get("max_drawdown_halt", 0.0), 0.0, 0.3)),
        "halt_cooldown_days": int(cl(p.get("halt_cooldown_days", 0), 0, 20)),
        "halt_resume_drawdown": float(cl(p.get("halt_resume_drawdown", 0.0), 0.0, 0.2)),
        "entry_ratio": float(cl(p.get("entry_ratio", 0.5), 0.2, 0.8)),
        "add_trigger_pct": float(cl(p.get("add_trigger_pct", 0.05), 0.02, 0.15)),
        "add_ratio": float(cl(p.get("add_ratio", 0.5), 0.2, 0.8)),
        "max_adds": int(cl(p.get("max_adds", 1), 0, 3)),
    }
    return out


def proposal_score(pf: dict[str, Any]) -> float:
    annual = float(pf.get("annual_return", -1.0))
    return float(pf["sharpe"]) if annual >= 0.06 else -10.0 + annual


def run_scheme(
    market: str,
    mode: str,
    pool: list[str],
    rounds: int,
    tuner: DeepSeekTuner | None,
) -> dict[str, Any]:
    app = AppConfig()
    training = TrainingConfig()
    evaluator = PortfolioEvaluator(app, training, synthetic=False)
    history: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []

    for r in range(1, rounds + 1):
        if tuner is None:
            proposal = sanitize({"fast": 5, "slow": 25, "timeframe": "daily"}, market, mode)
        else:
            proposal = sanitize(tuner.propose(history), market, mode)
        proposal["market"] = market
        proposal["position_mode"] = mode
        print(f"[{market}/{mode}] 第 {r}/{rounds} 轮: {json.dumps({k: proposal[k] for k in ('fast','slow','timeframe','top_k','position_ratio','entry_ratio','add_trigger_pct','add_ratio','max_adds') if k in proposal}, ensure_ascii=False)}")

        scores = evaluator.score_symbols(pool, proposal, *TRAIN_WINDOW)
        if scores.empty or len(scores) < proposal["top_k"]:
            history.append({"role": "user", "content": f"第 {r} 轮标的不足"})
            continue
        top = scores.nlargest(proposal["top_k"], proposal["select_metric"])["code"].tolist()
        train_pf = evaluator.portfolio_metrics(top, proposal, *TRAIN_WINDOW, "train")
        test_pf = evaluator.portfolio_metrics(top, proposal, *TEST_WINDOW, "test")
        test_returns = evaluator.portfolio_returns(top, proposal, *TEST_WINDOW)
        score = proposal_score(train_pf)
        results.append(
            {
                **proposal,
                "top_symbols": ",".join(top),
                "train": train_pf,
                "test": test_pf,
                "test_returns": test_returns,
                "score": score,
            }
        )
        print(
            f"  训练: 年化 {train_pf['annual_return']:.2%} 夏普 {train_pf['sharpe']:.3f} | "
            f"测试: 收益 {test_pf['total_return']:.2%} 夏普 {test_pf['sharpe']:.3f} | 评分 {score:.3f}"
        )
        history.extend(
            [
                {"role": "user", "content": f"第 {r} 轮提案: {json.dumps(proposal, ensure_ascii=False)}"},
                {"role": "assistant", "content": json.dumps(proposal, ensure_ascii=False)},
                {
                    "role": "user",
                    "content": "第 " + str(r) + " 轮评估(训练集): "
                    + json.dumps(
                        {k: train_pf.get(k) for k in ("total_return", "annual_return", "sharpe", "max_drawdown", "win_rate", "profit_factor", "n_trades", "coverage")},
                        ensure_ascii=False,
                    ),
                },
            ]
        )

    if not results:
        raise RuntimeError(f"{market}/{mode} 所有轮次失败")
    best = max(results, key=lambda x: x["score"])
    return {"market": market, "mode": mode, "best": best, "history": history}


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    return initial * (1.0 + returns).cumprod()


def chart_scheme(scheme: dict[str, Any]) -> Path:
    best = scheme["best"]
    ret = best["test_returns"]
    eq = equity_curve(ret)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq.index, eq.values, label="策略", color="#2f6fbf", linewidth=2)
    if isinstance(best["test"].get("benchmark_total_return"), float):
        pass
    ax.set_title(f"{MARKET_NAMES[scheme['market']]} · {MODE_NAMES[scheme['mode']]} 方案(测试集)")
    ax.set_ylabel("净值")
    ax.grid(alpha=0.3)
    ax.legend()
    t = best["test"]
    ax.text(
        0.02, 0.95,
        f"收益 {t['total_return']:.2%} | 夏普 {t['sharpe']:.2f} | 回撤 {t['max_drawdown']:.2%}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
    )
    fig.tight_layout()
    p = OUT_DIR / f"chart_{scheme['market']}_{scheme['mode']}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_overall(schemes: list[dict[str, Any]]) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    for s in schemes:
        eq = equity_curve(s["best"]["test_returns"])
        ax.plot(eq.index, eq.values, label=f"{MARKET_NAMES[s['market']]}-{MODE_NAMES[s['mode']]}", linewidth=1.6)
    ax.set_title("6 方案测试集净值对比")
    ax.set_ylabel("净值(1.0 起)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUT_DIR / "overall_chart.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--markets", default="cn us hk")
    parser.add_argument("--modes", default="full staged")
    parser.add_argument("--no-llm", action="store_true", help="不使用 DeepSeek, 固定基线参数(测试用)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app = AppConfig()
    tuner = None
    if not args.no_llm:
        tuner = DeepSeekTuner(api_key=load_api_key())

    markets = args.markets.split()
    modes = args.modes.split()
    schemes: list[dict[str, Any]] = []
    for market in markets:
        pool = load_pool(market, app)
        print(f"\n===== 股票池 {MARKET_NAMES[market]}: {len(pool)} 只 =====")
        for mode in modes:
            print(f"\n===== 训练方案: {MARKET_NAMES[market]} {MODE_NAMES[mode]} =====")
            scheme = run_scheme(market, mode, pool, args.rounds, tuner)
            schemes.append(scheme)

    # 保存与绘图
    summary_rows = []
    for s in schemes:
        b = s["best"]
        out_json = OUT_DIR / f"{s['market']}_{s['mode']}.json"
        payload = {k: v for k, v in b.items() if k != "test_returns"}
        payload["test_returns"] = b["test_returns"].to_dict()
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        chart_scheme(s)
        summary_rows.append(
            {
                "方案": f"{MARKET_NAMES[s['market']]}-{MODE_NAMES[s['mode']]}",
                "参数": f"{b['fast']}/{b['slow']} {b['timeframe']} top{b['top_k']}",
                "训练年化": f"{b['train']['annual_return']:.2%}",
                "测试收益": f"{b['test']['total_return']:.2%}",
                "测试夏普": f"{b['test']['sharpe']:.2f}",
                "测试回撤": f"{b['test']['max_drawdown']:.2%}",
            }
        )
    chart_overall(schemes)

    table = pd.DataFrame(summary_rows)
    md_table = table.to_markdown(index=False)
    try:
        analysis = _ask(
            "你是量化研究员, 用中文对六个方案做横向对比分析: 谁最优、全仓vs补仓各市场结论、日内周期是否入选、风险提示。约500字。",
            "方案汇总:\n" + md_table + "\n各方案最优参数见同目录json。",
        )
    except Exception as exc:
        analysis = f"(DeepSeek 分析生成失败: {exc!r})"
    report = (
        "# 6 方案训练总报告\n\n"
        f"生成日期: {date.today()}\n\n"
        "## 方案汇总\n\n" + md_table + "\n\n"
        "## DeepSeek 分析\n\n" + analysis + "\n"
    )
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print("\n" + md_table)
    print(f"\n报告: {OUT_DIR / 'report.md'}")
    print(f"图表: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
