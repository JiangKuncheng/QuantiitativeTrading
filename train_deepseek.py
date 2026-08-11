"""
DeepSeek 驱动的模型训练
=======================

流程:
    1. 数据: 2020-2026 全窗口; 用沪深300指数自动把年份分为 牛/熊/横盘;
       选择连续切分点, 使训练集与测试集各自包含三类行情且互不重叠;
    2. 股票池: CSI300 前 N 只(或 --symbols 指定), 训练集选股(Top-K)+调参;
    3. DeepSeek 调参循环: 每轮提案 -> 训练集评估 -> 反馈历史 -> 下一轮;
    4. 最优提案(按训练集夏普)在测试集做最终评估,
       并与沪深300基准对比(超额收益/贝塔/上涨下跌捕获), 判断是否跑赢大盘。

用法:
    python train_deepseek.py --pool-size 15 --rounds 8
    python train_deepseek.py --symbols "000001 600519 300750" --rounds 6
    python train_deepseek.py --offline --rounds 3            # 测试框架(随机提案, 合成数据)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from qtcore.config import AppConfig
from qtcore.deepseek_tuner import DeepSeekTuner, load_api_key
from qtcore.screener import StockScreener
from qtcore.trainer import (
    PortfolioEvaluator,
    TrainingConfig,
    balanced_years_split,
    classify_yearly_regimes,
    compare_with_benchmark,
    random_proposals,
    years_to_range,
)


ALLOWED_SELECT_METRICS = {"sharpe", "total_return", "profit_factor"}
ALLOWED_REBALANCE = {"daily", "weekly", "monthly"}
ALLOWED_ORDER = {"market", "limit"}
ALLOWED_TIMEFRAMES = {"daily", "60min", "2h", "4h", "6h"}

# 日内周期数据只有约 2 年(新浪60分钟): 训练 2024-08~2025-07 / 测试 2026-01~2026-08
INTRADAY_WINDOWS = {
    "train": ("20240801", "20250731"),
    "test": ("20260101", "20260810"),
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sanitize_proposal(p: dict[str, Any]) -> dict[str, Any]:
    """把 DeepSeek 的原始提案收敛到合法范围, 防止越界参数破坏回测。"""
    fast = int(clamp(p.get("fast", 5), 3, 60))
    slow = int(clamp(p.get("slow", 20), 10, 120))
    if slow <= fast:
        slow = fast + 10
    rsi_buy = float(clamp(p.get("rsi_buy", 30.0), 20.0, 45.0))
    rsi_sell = float(clamp(p.get("rsi_sell", 70.0), 60.0, 85.0))
    if rsi_sell <= rsi_buy:
        rsi_sell = min(85.0, rsi_buy + 15.0)
    return {
        "fast": fast,
        "slow": slow,
        "timeframe": p.get("timeframe", "daily")
        if p.get("timeframe", "daily") in ALLOWED_TIMEFRAMES
        else "daily",
        "use_rsi": bool(p.get("use_rsi", False)),
        "rsi_window": int(clamp(p.get("rsi_window", 14), 5, 30)),
        "rsi_buy": rsi_buy,
        "rsi_sell": rsi_sell,
        "top_k": int(clamp(p.get("top_k", 5), 2, 10)),
        "select_metric": p.get("select_metric", "sharpe")
        if p.get("select_metric", "sharpe") in ALLOWED_SELECT_METRICS
        else "sharpe",
        "position_ratio": float(clamp(p.get("position_ratio", 0.95), 0.1, 1.0)),
        "max_position_ratio": float(clamp(p.get("max_position_ratio", 1.0), 0.2, 1.0)),
        "rebalance": p.get("rebalance", "daily")
        if p.get("rebalance", "daily") in ALLOWED_REBALANCE
        else "daily",
        "order_type": p.get("order_type", "market")
        if p.get("order_type", "market") in ALLOWED_ORDER
        else "market",
        "slippage_tolerance_pct": float(clamp(p.get("slippage_tolerance_pct", 0.0), 0.0, 0.01)),
        "leverage": float(clamp(p.get("leverage", 1.0), 1.0, 2.0)),
        "stop_loss_pct": float(clamp(p.get("stop_loss_pct", 0.0), 0.0, 0.2)),
        "take_profit_pct": float(clamp(p.get("take_profit_pct", 0.0), 0.0, 0.5)),
        "max_drawdown_halt": float(clamp(p.get("max_drawdown_halt", 0.0), 0.0, 0.3)),
        "halt_cooldown_days": int(clamp(p.get("halt_cooldown_days", 0), 0, 20)),
        "halt_resume_drawdown": float(clamp(p.get("halt_resume_drawdown", 0.0), 0.0, 0.2)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="train_deepseek", description="DeepSeek 驱动训练")
    parser.add_argument("--pool-size", type=int, default=15, help="CSI300 股票池大小")
    parser.add_argument("--symbols", default=None, help="显式股票池, 空格分隔")
    parser.add_argument("--rounds", type=int, default=8, help="DeepSeek 调参轮数")
    parser.add_argument("--model", default=None, help="DeepSeek 模型(默认取配置文件)")
    parser.add_argument("--benchmark", default="000300", help="基准指数代码")
    parser.add_argument("--retries", type=int, default=2, help="数据拉取重试次数")
    parser.add_argument("--offline", action="store_true", help="合成数据+随机提案(测试框架)")
    parser.add_argument("--min-annual", type=float, default=0.06,
                        help="训练集年化收益硬性门槛(默认6%), 不达标提案判无效")
    parser.add_argument("--timeframe", choices=ALLOWED_TIMEFRAMES, default=None,
                        help="强制所有提案使用该K线周期(用于分别探索日线/日内线)")
    return parser.parse_args()


def proposal_score(train_pf: dict[str, Any], min_annual: float) -> float:
    """训练评分: 年化收益不达标直接淘汰, 达标后按夏普排序。"""
    annual = float(train_pf.get("annual_return", -1.0))
    if annual >= min_annual:
        return float(train_pf["sharpe"])
    return -10.0 + annual  # 硬性淘汰(分数远低于任何达标提案)


def compact_metrics(m: dict[str, Any]) -> dict[str, Any]:
    keys = ("total_return", "annual_return", "sharpe", "max_drawdown", "win_rate", "profit_factor", "n_trades", "coverage")
    return {k: m.get(k) for k in keys if k in m}


def main() -> int:
    args = parse_args()
    app = AppConfig()
    app.data.fetch_retries = args.retries
    training = TrainingConfig(pool_size=args.pool_size, top_k=5)
    evaluator = PortfolioEvaluator(app, training, synthetic=args.offline)

    # 1) 股票池
    if args.symbols:
        pool = args.symbols.split()
        print(f"[Train] 股票池 {len(pool)} 只(显式): {pool[:15]}")
    else:
        screener = StockScreener(app, synthetic=args.offline, universe="csi300")
        candidates = screener.filter_universe()
        pool = [str(c) for c in candidates.head(args.pool_size)["code"]]
        print(f"[Train] 股票池 {len(pool)} 只(CSI300): {pool[:15]}")

    # 2) 基准指数 + 牛熊横盘分类 + 均衡切分
    if args.offline:
        bench = evaluator.dc.generate_synthetic_bars(days=1600, symbol=args.benchmark, seed=7)
    else:
        bench = evaluator.dc.get_index_daily(args.benchmark, "20200101", "20260810")
    regimes = classify_yearly_regimes(bench)
    train_years, test_years, split_info = balanced_years_split(regimes)
    train_range = years_to_range(train_years)
    test_range = years_to_range(test_years)
    print("\n[Split] 年度行情分类:", regimes)
    print(f"[Split] 训练集 {train_years} -> 区间 {train_range}")
    print(f"[Split] 测试集 {test_years} -> 区间 {test_range}")
    print(f"[Split] 覆盖情况: 训练 {split_info['train_coverage']} | 测试 {split_info['test_coverage']} "
          f"(分数 {split_info['regime_score']}/{split_info['max_possible']})\n")

    # 3) 第 0 轮: 注入活跃基线, 给 DeepSeek 一个"有交易、仓位足"的参考点
    history: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []
    base_tf = args.timeframe or "daily"
    baseline = sanitize_proposal(
        {
            "fast": 8, "slow": 30, "timeframe": base_tf, "use_rsi": False,
            "top_k": 5, "select_metric": "sharpe", "position_ratio": 0.95,
            "max_position_ratio": 1.0, "rebalance": "daily", "order_type": "market",
            "slippage_tolerance_pct": 0.0, "leverage": 1.0,
            "stop_loss_pct": 0.0, "take_profit_pct": 0.0, "max_drawdown_halt": 0.0,
        }
    )
    print(f"===== 第 0 轮(活跃基线, 供 DeepSeek 参考): {json.dumps(baseline, ensure_ascii=False)}")
    base_scores = evaluator.score_symbols(pool, baseline, *train_range)
    if len(base_scores) >= baseline["top_k"]:
        base_top = base_scores.nlargest(baseline["top_k"], baseline["select_metric"])["code"].tolist()
        base_pf = evaluator.portfolio_metrics(base_top, baseline, *train_range, "train")
        results.append(
            {
                **baseline,
                "top_symbols": ",".join(base_top),
                "train": base_pf,
                "train_score": proposal_score(base_pf, args.min_annual),
            }
        )
        history.extend(
            [
                {"role": "user", "content": "第 0 轮(活跃基线提案): " + json.dumps(baseline, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(baseline, ensure_ascii=False)},
                {
                    "role": "user",
                    "content": "第 0 轮评估(训练集): "
                    + json.dumps(compact_metrics(base_pf), ensure_ascii=False),
                },
            ]
        )
        print(f"  Top-K {base_top}")
        print(f"  训练集: 收益 {base_pf['total_return']:.2%} | 年化 {base_pf['annual_return']:.2%} | "
              f"夏普 {base_pf['sharpe']:.3f} | 交易 {base_pf['n_trades']} | "
              f"评分 {results[-1]['train_score']:.3f}\n")
    else:
        print("基线训练集标的不足, 跳过基线\n")

    # 4) DeepSeek 调参循环
    if args.offline:
        proposals = random_proposals(args.rounds, seed=args.rounds)
        tuner = None
    else:
        key = load_api_key()
        tuner = DeepSeekTuner(api_key=key, model=args.model or "deepseek-chat")
        proposals = []

    seen = {json.dumps(baseline, sort_keys=True, ensure_ascii=False)}
    for r in range(1, args.rounds + 1):
        proposal = None
        for attempt in range(3):  # 去重: 与历史提案完全相同则重新请求
            if tuner is not None:
                proposal = sanitize_proposal(tuner.propose(history))
            else:
                proposal = sanitize_proposal(proposals[r - 1])
            if args.timeframe:
                proposal["timeframe"] = args.timeframe
            key = json.dumps(proposal, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                break
            print(f"  (提案与历史重复, 第 {attempt + 1} 次重新请求)")
            if tuner is not None:
                history.append(
                    {
                        "role": "user",
                        "content": f"第 {r} 轮你给出的提案与历史完全重复, 请换一组不同的参数: "
                        + json.dumps(proposal, ensure_ascii=False),
                    }
                )
        seen.add(json.dumps(proposal, sort_keys=True, ensure_ascii=False))
        print(f"===== 第 {r}/{args.rounds} 轮提案: {json.dumps(proposal, ensure_ascii=False)}")

        # 按时间框架选择训练/测试窗口: 日内线只有 2024-08 之后的数据
        tf = proposal.get("timeframe", "daily")
        if tf == "daily":
            tr, te = train_range, test_range
        else:
            tr, te = INTRADAY_WINDOWS["train"], INTRADAY_WINDOWS["test"]
        scores = evaluator.score_symbols(pool, proposal, *tr)
        if scores.empty or len(scores) < proposal["top_k"]:
            msg = f"第 {r} 轮: 训练集可用标的不足({len(scores)}/{proposal['top_k']})"
            history.extend([
                {"role": "user", "content": f"第 {r} 轮提案: {json.dumps(proposal, ensure_ascii=False)}"},
                {"role": "assistant", "content": json.dumps(proposal, ensure_ascii=False)},
                {"role": "user", "content": msg},
            ])
            continue

        top_symbols = scores.nlargest(proposal["top_k"], proposal["select_metric"])["code"].tolist()
        train_pf = evaluator.portfolio_metrics(top_symbols, proposal, *tr, "train")
        record = {
            **proposal,
            "top_symbols": ",".join(top_symbols),
            "train": train_pf,
            "train_score": proposal_score(train_pf, args.min_annual),
        }
        results.append(record)

        summary = compact_metrics(train_pf)
        print(f"  Top-K {top_symbols}")
        print(f"  训练集: 收益 {train_pf['total_return']:.2%} | 年化 {train_pf['annual_return']:.2%} | "
              f"夏普 {train_pf['sharpe']:.3f} | 交易 {train_pf['n_trades']} | 评分 {record['train_score']:.3f} | "
              f"回撤 {train_pf['max_drawdown']:.2%} | 胜率 {train_pf['win_rate']:.1%} | "
              f"盈亏比 {train_pf['profit_factor']:.2f}\n")

        if train_pf.get("annual_return", -1.0) < args.min_annual:
            feedback_msg = (
                f"第 {r} 轮评估(训练集): {json.dumps(summary, ensure_ascii=False)} | "
                f"结论: 无效, 年化 {train_pf['annual_return']:.2%} < {args.min_annual:.0%} 门槛, "
                f"交易次数 {train_pf['n_trades']}. 若交易过少或仓位过低, 请切换方向: "
                f"use_rsi=false, sampling=daily, rebalance=daily, position_ratio>=0.8, "
                f"fast 3~10, slow 20~40, 止损止盈设置勿过紧。"
            )
        else:
            feedback_msg = f"第 {r} 轮评估(训练集): {json.dumps(summary, ensure_ascii=False)}"

        history.extend([
            {"role": "user", "content": f"第 {r} 轮提案: {json.dumps(proposal, ensure_ascii=False)}"},
            {"role": "assistant", "content": json.dumps(proposal, ensure_ascii=False)},
            {"role": "user", "content": feedback_msg},
        ])

    if not results:
        print("所有轮次均失败")
        return 1

    # 4) 最优提案 -> 测试集 + 大盘对比
    best = max(results, key=lambda x: x["train_score"])
    print("=" * 68)
    print(f"最优提案(评分 {best['train_score']:.3f}): {json.dumps({k: best[k] for k in ('fast','slow','timeframe','use_rsi','top_k','position_ratio','max_position_ratio','rebalance','order_type','stop_loss_pct','take_profit_pct','leverage','max_drawdown_halt')}, ensure_ascii=False)}")
    print(f"训练集: 收益 {best['train']['total_return']:.2%} | 年化 {best['train']['annual_return']:.2%} | "
          f"夏普 {best['train']['sharpe']:.3f} | 回撤 {best['train']['max_drawdown']:.2%}")
    print(f"选股: {best['top_symbols']}")

    best_tf = best.get("timeframe", "daily")
    if best_tf == "daily":
        tr, te = train_range, test_range
    else:
        tr, te = INTRADAY_WINDOWS["train"], INTRADAY_WINDOWS["test"]
    test_pf = evaluator.portfolio_metrics(
        best["top_symbols"].split(","), best, *te, "test"
    )
    test_returns = evaluator.portfolio_returns(
        best["top_symbols"].split(","), best, *te
    )
    bench_cmp = compare_with_benchmark(test_returns, bench)

    print(f"测试集组合: 收益 {test_pf['total_return']:.2%} | 夏普 {test_pf['sharpe']:.3f} | "
          f"回撤 {test_pf['max_drawdown']:.2%} | 覆盖率 {test_pf['coverage']:.0%}\n")
    print("===== 测试集: 策略 vs 沪深300 基准 =====")
    if "error" in bench_cmp:
        print(bench_cmp)
    else:
        for k, v in bench_cmp.items():
            if isinstance(v, float):
                if k in ("strategy_total_return", "benchmark_total_return", "excess_total_return",
                         "strategy_annual_return", "benchmark_annual_return",
                         "strategy_max_drawdown", "benchmark_max_drawdown"):
                    print(f"  {k:26s}: {v:.2%}")
                else:
                    print(f"  {k:26s}: {v:.4f}")
            else:
                print(f"  {k:26s}: {v}")
        print("\n解读: excess_total_return>0 且 beta 不高、up_capture>1 说明选股/操盘有贡献;")
        print("若 excess≈0 而策略收益高, 说明主要是市场 beta 贡献(吃了市场红利)。")

    # 5) 落盘
    out_dir = app.paths.output_dir
    log_path = out_dir / "deepseek_tuning_log.jsonl"
    with log_path.open("w", encoding="utf-8") as fp:
        for rec in results:
            fp.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    report = {
        "split": {"regimes": regimes, "train_years": train_years, "test_years": test_years, **split_info},
        "best_proposal": {k: v for k, v in best.items() if k != "train"},
        "test_portfolio": test_pf,
        "benchmark_comparison": bench_cmp,
    }
    report_path = out_dir / "deepseek_final_report.json"
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2, default=str)
    print(f"\n日志: {log_path}")
    print(f"报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
