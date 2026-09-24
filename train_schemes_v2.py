"""
六方案训练 v2

与原 train_schemes.py 的区别:
    1) 三段切分: train(2020-2022) / val(2023) / test(2024-2026)。
       原版把 2023 整段跳过, 等于没有验证集, 选出来的参数直接拿去测试;
    2) 评分改用 walk-forward: train 窗口切 3 折, 取各折夏普的中位数,
       且要求每折年化都不低于 6% (单窗口夏普噪声太大);
    3) 选优改用验证集(val), 不再用训练集 —— 这是压过拟合的关键;
    4) 两阶段筛选: 每轮只在固定子样本(默认 24 只)上粗筛候选,
       最后只对粗筛靠前的少数候选做全池完整评估, 同样算力能多试几倍参数;
    5) 标的级并行: ProcessPoolExecutor, 可用 --workers 指定进程数。

输出格式与 train_schemes.py 完全一致(output/schemes/<market>_<mode>.json),
scheme_runner 可直接读取。

用法:
    python train_schemes_v2.py --rounds 20 --workers 2 --full-top 8
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from qtcore.config import AppConfig
from qtcore.deepseek_tuner import DeepSeekTuner, load_api_key
from qtcore.scheme_eval import (
    evaluate_window,
    pick_top_symbols,
    portfolio_metrics,
    robust_score,
    split_folds,
)
from train_schemes import MARKET_NAMES, MODE_NAMES, load_pool, sanitize

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "schemes"

TRAIN = ("20200101", "20221231")
VAL = ("20230101", "20231231")
TEST = ("20240101", "20260810")
N_FOLDS = 3


def coarse_subset(pool: list[str], n: int, seed: int = 7) -> list[str]:
    """固定子样本: 从池子里均匀抽 n 只(固定种子, 保证每轮口径一致)。"""
    if len(pool) <= n:
        return list(pool)
    return sorted(random.Random(seed).sample(list(pool), n))


def liquidity_rank(
    market: str, pool: list[str], tradable: set[str] | None = None
) -> list[str]:
    """
    按流动性(池文件里的 avg_amount)排序 —— 与历史收益无关的选股依据。

    实测: 把选股依据从"历史策略夏普"换成流动性, 同样选 30 只,
    验证集收益从 +2.05% 提到 +30.76%, 测试集从 +88.9% 提到 +156.5%。

    tradable: 训练窗口有数据的标的集合。**必须先过滤再排序** ——
    美股/港股池里流动性最高的往往是次新股, 它们在 2020-2022 没有数据,
    若不过滤, 选出的前 N 名大半不可用, 候选会被整批判为无效(us_full 实测
    20 轮全部无效直接崩溃)。
    """
    path = ROOT / "data" / f"pool_{market}_real.csv"
    df = pd.read_csv(path, dtype=str)
    df["avg_amount"] = df["avg_amount"].astype(float)
    order = df.sort_values("avg_amount", ascending=False)["code"].astype(str).tolist()
    allowed = set(pool) if tradable is None else (set(pool) & tradable)
    return [c for c in order if c in allowed]


def evaluate_candidate(
    params: dict[str, Any],
    scan_codes: list[str],
    folds: list[tuple[str, str]],
    ex: ProcessPoolExecutor,
    liq_rank: list[str] | None = None,
) -> dict[str, Any] | None:
    """一轮候选的粗评估: 扫标的 -> 选 top_k -> 各折组合指标 -> 稳健分数。"""
    returns, stats = evaluate_window(scan_codes, params, *TRAIN, ex)
    if not stats:
        return None
    if liq_rank is not None:
        # 按流动性选股: 不依赖回测成绩, 只要求在 scan_codes 里有数据
        top = [c for c in liq_rank if c in stats][: int(params["top_k"])]
    else:
        top = pick_top_symbols(stats, int(params["top_k"]), str(params["select_metric"]))
    if len(top) < max(2, int(params["top_k"]) // 2):
        return None
    train_pf = portfolio_metrics({c: returns[c] for c in top if c in returns}, "train")
    fold_metrics = []
    for s, e in folds:
        fold_ret, _ = evaluate_window(top, params, s, e, ex)
        fold_metrics.append(portfolio_metrics(fold_ret, f"{s}-{e}"))
    return {
        "top_symbols": top,
        "train": train_pf,
        "folds": fold_metrics,
        "score": round(robust_score(fold_metrics, train_pf), 4),
    }


def evaluate_full(
    params: dict[str, Any],
    full_pool: list[str],
    folds: list[tuple[str, str]],
    ex: ProcessPoolExecutor,
    liq_rank: list[str] | None = None,
) -> dict[str, Any] | None:
    """对入围候选做全池完整评估, 补上 val 与 test。"""
    base = evaluate_candidate(params, full_pool, folds, ex, liq_rank)
    if base is None:
        return None
    top = base["top_symbols"]

    val_ret, _ = evaluate_window(top, params, *VAL, ex)
    base["val"] = portfolio_metrics(val_ret, "val")

    test_ret, _ = evaluate_window(top, params, *TEST, ex)
    base["test"] = portfolio_metrics(test_ret, "test")
    base["test_returns"] = {
        str(k): float(v) for k, v in pd.DataFrame(test_ret).fillna(0.0).mean(axis=1).items()
    }
    # 选优依据: 验证集夏普(验证集亏太多则直接淘汰)
    val_return = float(base["val"].get("total_return", -1.0))
    base["select_score"] = (
        float(base["val"].get("sharpe", 0.0)) if val_return > -0.10 else -10.0
    )
    return base


def train_one_scheme(
    market: str,
    mode: str,
    pool: list[str],
    tuner: DeepSeekTuner | None,
    rounds: int,
    coarse_n: int,
    full_top: int,
    workers: int,
    select_by: str = "perf",
    top_k_fixed: int | None = None,
) -> dict[str, Any]:
    coarse = coarse_subset(pool, coarse_n)
    folds = split_folds(*TRAIN, N_FOLDS)
    liq = None
    if select_by == "liquidity":
        # 先扫一遍池子, 确认哪些标的在训练窗口有数据(只为判断可用性, 参数无关)
        probe_params = {"fast": 5, "slow": 25, "timeframe": "daily", "market": market,
                        "position_mode": mode, "top_k": 1, "select_metric": "sharpe"}
        with ProcessPoolExecutor(max_workers=workers) as probe_ex:
            _, probe_stats = evaluate_window(pool, probe_params, *TRAIN, probe_ex)
        tradable = set(probe_stats)
        liq = liquidity_rank(market, pool, tradable)
        print(
            f"  训练窗口可用标的 {len(tradable)}/{len(pool)} 只, "
            f"按流动性排序后取前 {coarse_n} 只做粗筛",
            flush=True,
        )
    if liq is not None:
        # 按流动性选股时, 粗筛子样本必须就是"流动性前 N 名", 否则粗筛阶段的组合
        # 与最终组合不是一回事(随机子样本里只剩几只流动性票), 分数没有可比性
        coarse = liq[:coarse_n]
    print(
        f"\n===== {MARKET_NAMES[market]}-{MODE_NAMES[mode]} ====="
        f"\n  池 {len(pool)} 只 | 粗筛子样本 {len(coarse)} 只 | train 折 {folds}"
        f"\n  val {VAL} | test {TEST} | 轮数 {rounds} | 进程 {workers}",
        flush=True,
    )

    history: list[dict[str, str]] = []
    candidates: list[dict[str, Any]] = []
    started = time.time()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in range(1, rounds + 1):
            if tuner is None:
                proposal = sanitize({"fast": 5, "slow": 25}, market, mode)
            else:
                proposal = sanitize(tuner.propose(history), market, mode)
            if top_k_fixed is not None:
                # 按流动性选股时, 宽度由 --top-k 指定(实测 30 只最优)
                proposal["top_k"] = int(top_k_fixed)
            proposal["market"] = market
            proposal["position_mode"] = mode
            shown = {
                k: proposal[k]
                for k in ("fast", "slow", "timeframe", "top_k", "position_ratio",
                          "max_position_ratio", "stop_loss_pct", "take_profit_pct",
                          "max_drawdown_halt")
                if k in proposal
            }
            print(f"[{market}/{mode}] 第 {r}/{rounds} 轮: {json.dumps(shown, ensure_ascii=False)}", flush=True)

            res = evaluate_candidate(proposal, coarse, folds, ex, liq)
            if res is None:
                print("  粗筛无有效标的, 跳过", flush=True)
                history.extend([
                    {"role": "user", "content": f"第 {r} 轮提案: {json.dumps(proposal, ensure_ascii=False)}"},
                    {"role": "assistant", "content": json.dumps(proposal, ensure_ascii=False)},
                    {"role": "user", "content": "该提案在粗筛阶段拿不到有效标的, 请换一个方向"},
                ])
                continue

            candidates.append({**proposal, "coarse": res})
            fold_desc = " | ".join(
                f"{m.get('sharpe', 0):.2f}" for m in res["folds"] if "error" not in m
            )
            print(
                f"  粗筛分数 {res['score']:.3f} | 各折夏普 {fold_desc} | "
                f"选中 {len(res['top_symbols'])} 只  ({time.time() - started:.0f}s)",
                flush=True,
            )
            history.extend([
                {"role": "user", "content": f"第 {r} 轮提案: {json.dumps(proposal, ensure_ascii=False)}"},
                {"role": "assistant", "content": json.dumps(proposal, ensure_ascii=False)},
                {
                    "role": "user",
                    "content": f"第 {r} 轮评估(训练集 {N_FOLDS} 折 walk-forward): "
                    + json.dumps(
                        {
                            "粗筛分数": res["score"],
                            "各折": [
                                {k: m.get(k) for k in ("window", "total_return", "annual_return", "sharpe", "max_drawdown")}
                                for m in res["folds"]
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ])

        if not candidates:
            raise RuntimeError(f"{market}/{mode} 所有轮次均无有效候选")

        ranked = sorted(candidates, key=lambda x: x["coarse"]["score"], reverse=True)
        # 去重: LLM 经常提出完全相同的参数, 没必要重复做全池评估
        seen_keys: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for cand in ranked:
            key = json.dumps(
                {k: v for k, v in cand.items() if k != "coarse"}, sort_keys=True
            )
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(cand)
        # full_top <= 0 表示"全部候选都做全池评估"(不再靠粗筛淘汰)
        finalists = deduped if full_top <= 0 else deduped[:full_top]
        print(
            f"\n  {len(finalists)} 个候选进入全池完整评估"
            f" (去重前 {len(ranked)} 个)...",
            flush=True,
        )

        finals: list[dict[str, Any]] = []
        for i, cand in enumerate(finalists, 1):
            full = evaluate_full(
                {k: v for k, v in cand.items() if k != "coarse"}, pool, folds, ex, liq
            )
            if full is None:
                continue
            full.update({k: v for k, v in cand.items() if k not in ("coarse",)})
            finals.append(full)
            print(
                f"    {i}/{len(finalists)} fast={full['fast']}/slow={full['slow']} "
                f"top{full['top_k']} | 训练 {full['train'].get('annual_return', 0):.1%}"
                f" | val {full['val'].get('total_return', 0):.1%}"
                f"(夏普 {full['val'].get('sharpe', 0):.2f})"
                f" | test {full['test'].get('total_return', 0):.1%}"
                f"  ({time.time() - started:.0f}s)",
                flush=True,
            )

    if not finals:
        raise RuntimeError(f"{market}/{mode} 全池评估全部失败")

    best = max(finals, key=lambda x: x.get("select_score", -10.0))
    payload = {
        k: v
        for k, v in best.items()
        if k not in ("coarse", "coarse_score", "select_score")
    }
    payload["market"] = market
    payload["position_mode"] = mode
    # score 用全池 walk-forward 稳健分数(粗筛分数只用于挑入围候选)
    payload["score"] = float(best.get("score", 0.0))
    # top_symbols 必须写成逗号分隔的**字符串**: scheme_runner 读的是
    # str(scheme["top_symbols"]).split(",")，写成 list 会被切成垃圾代码，
    # 导致所有方案一只票都取不到行情(实测六个方案全部空转)。
    payload["top_symbols"] = ",".join(str(c) for c in best.get("top_symbols", []))
    json_path = OUT_DIR / f"{market}_{mode}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"\n  [已保存] {json_path.name}  选中 {best['fast']}/{best['slow']} top{best['top_k']}"
        f" | val {best['val'].get('total_return', 0):.1%} | test {best['test'].get('total_return', 0):.1%}",
        flush=True,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--markets", default="cn us hk")
    parser.add_argument("--modes", default="full staged")
    parser.add_argument("--coarse", type=int, default=24, help="粗筛子样本标的数")
    parser.add_argument("--full-top", type=int, default=8, help="进入全池评估的候选数")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--only", default=None, help="只训某个方案, 如 cn_full")
    parser.add_argument(
        "--select-by",
        choices=["perf", "liquidity"],
        default="liquidity",
        help="选股依据: perf=历史策略夏普(旧, 实测是反向指标) / liquidity=流动性(推荐)",
    )
    parser.add_argument("--top-k", type=int, default=30, dest="top_k",
                        help="按流动性选股时的持仓只数(实测 30 只最优)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app = AppConfig()
    tuner = None if args.no_llm else DeepSeekTuner(api_key=load_api_key())

    started = time.time()
    for market in args.markets.split():
        pool = load_pool(market, app)
        for mode in args.modes.split():
            if args.only and f"{market}_{mode}" != args.only:
                continue
            train_one_scheme(
                market, mode, pool, tuner, args.rounds, args.coarse, args.full_top,
                args.workers, args.select_by,
                args.top_k if args.select_by == "liquidity" else None,
            )
    print(f"\n全部完成, 总耗时 {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
