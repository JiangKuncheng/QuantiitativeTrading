"""
模型训练入口: 2020-2026 数据集 + Walk-Forward + 大模型调参循环
==============================================================

用法:
    # 1) 生成提案模板
    python train_model.py --write-template proposals.json

    # 2) 大模型/人工填写提案后训练(模板里是示例, 可自行修改)
    python train_model.py --proposals proposals.json --pool-size 12 --top-k 5

    # 3) 随机基线(测试框架用, 正式调参请用大模型提案)
    python train_model.py --rounds 3

    # 4) 指定股票池(不跑全市场初筛)
    python train_model.py --proposals proposals.json --symbols "000001 600519 300750"

    # 5) 离线演示(合成数据)
    python train_model.py --rounds 2 --offline

输出:
    output/training_log.csv     每轮提案的 训练/验证/测试 完整指标
    output/training_summary.json 最优提案(按验证集)及其测试集表现
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qtcore.config import AppConfig
from qtcore.trainer import (
    TrainingConfig,
    WalkForwardTrainer,
    flatten_result,
    flatten_rolling_result,
    load_proposals,
    random_proposals,
    save_proposal_template,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="train_model", description="量化模型训练(2020-2026)")
    parser.add_argument("--proposals", default=None, help="提案 JSON 文件(大模型/人工填写)")
    parser.add_argument("--rounds", type=int, default=0, help="无提案文件时生成 N 个随机基线提案")
    parser.add_argument("--write-template", default=None, help="生成提案模板文件后退出")

    parser.add_argument("--symbols", default=None, help="指定股票池, 空格分隔")
    parser.add_argument("--pool-size", type=int, default=12, help="全市场模式股票池大小")
    parser.add_argument("--top-k", type=int, default=5, help="默认持仓数量")
    parser.add_argument("--select-metric", default="sharpe", help="默认选股排序指标")

    parser.add_argument("--train-start", default="20200101")
    parser.add_argument("--train-end", default="20221231")
    parser.add_argument("--val-start", default="20230101")
    parser.add_argument("--val-end", default="20231231")
    parser.add_argument("--test-start", default="20240101")
    parser.add_argument("--test-end", default="20260810")

    parser.add_argument("--offline", action="store_true", help="合成数据演示")
    parser.add_argument("--no-cache", action="store_true", help="禁用行情缓存")
    parser.add_argument("--rolling", action="store_true", help="滚动 Walk-Forward 多段交叉验证(推荐)")
    parser.add_argument("--folds", type=int, default=3, help="滚动折数(1~3)")
    parser.add_argument("--universe", choices=["all", "csi300"], default="all",
                        help="股票池来源: all 全市场 / csi300 沪深300成分股")
    parser.add_argument("--retries", type=int, default=3, help="网络拉取统一重试次数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.write_template:
        save_proposal_template(Path(args.write_template))
        print(f"提案模板已写入: {args.write_template}")
        return 0

    training = TrainingConfig(
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
        pool_size=args.pool_size,
        top_k=args.top_k,
        select_metric=args.select_metric,
    )

    app = AppConfig()
    app.data.use_cache = not args.no_cache
    app.data.fetch_retries = args.retries
    trainer = WalkForwardTrainer(
        app,
        training,
        synthetic=args.offline,
        universe=args.universe,
    )

    symbols = args.symbols.split() if args.symbols else None
    trainer.pool = trainer.load_pool(symbols, universe=args.universe)
    print(f"[Trainer] 股票池 {len(trainer.pool)} 只(来源: {args.universe}): {trainer.pool[:12]}")

    if args.proposals:
        proposals = load_proposals(Path(args.proposals))
    else:
        proposals = random_proposals(args.rounds)
    print(f"[Trainer] 提案 {len(proposals)} 个, 窗口: "
          f"train {training.train_start}~{training.train_end} | "
          f"val {training.val_start}~{training.val_end} | "
          f"test {training.test_start}~{training.test_end}\n")

    results = []
    for i, proposal in enumerate(proposals, 1):
        print(f"===== 第 {i}/{len(proposals)} 个提案: {proposal} =====")
        if args.rolling:
            result = trainer.run_proposal_rolling(
                proposal, WalkForwardTrainer.rolling_folds(args.folds)
            )
        else:
            result = trainer.run_proposal(proposal)
        if "error" in result:
            print(f"  [跳过] {result['error']}")
            continue
        flat = flatten_rolling_result(result) if args.rolling else flatten_result(result)
        results.append(flat)
        if args.rolling:
            print(f"  平均验证夏普 {result['avg_val_sharpe']:.3f} | "
                  f"平均测试: 收益 {result['avg_test_total_return']:.2%} 夏普 {result['avg_test_sharpe']:.3f} "
                  f"回撤 {result['avg_test_max_drawdown']:.2%} | "
                  f"测试为正的折 {result['test_positive_folds']} | 覆盖率 {result['avg_coverage']:.0%}\n")
        else:
            print(f"  选股(Top-K): {result['top_symbols']}")
            v, t = result["val"], result["test"]
            print(f"  验证集: 收益 {v['total_return']:.2%} | 夏普 {v['sharpe']:.3f} | 回撤 {v['max_drawdown']:.2%}")
            print(f"  测试集: 收益 {t['total_return']:.2%} | 夏普 {t['sharpe']:.3f} | 回撤 {t['max_drawdown']:.2%}\n")

    if not results:
        print("所有提案均失败, 请检查数据/股票池")
        return 1

    log_df = pd.DataFrame(results)
    out_dir = app.paths.output_dir
    log_path = out_dir / "training_log.csv"
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")

    if args.rolling:
        best = max(results, key=lambda r: r.get("avg_val_sharpe", -999))
    else:
        best = max(results, key=lambda r: r.get("val_sharpe", -999))
    summary = {
        "best_proposal": best,
        "selection_rule": "avg_val_sharpe" if args.rolling else "val_sharpe",
        "note": "滚动模式下按平均验证夏普挑选, 平均测试集为最终样本外报告",
    }
    summary_path = out_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("=" * 68)
    print("训练完成")
    print(f"全部日志: {log_path}")
    print(f"最优提案(按{'平均验证' if args.rolling else '验证'}夏普): {best['fast']}/{best['slow']} "
          f"top_k={best.get('top_k')} select={best.get('select_metric')}")
    if args.rolling:
        print(f"  平均验证夏普: {best['avg_val_sharpe']:.3f}")
        print(f"  平均测试集: 收益 {best['avg_test_total_return']:.2%} | 夏普 {best['avg_test_sharpe']:.3f} "
              f"| 回撤 {best['avg_test_max_drawdown']:.2%} | 正收益折 {best['test_positive_folds']}")
    else:
        print(f"  验证集: 收益 {best['val_total_return']:.2%} | 夏普 {best['val_sharpe']:.3f}")
        print(f"  测试集: 收益 {best['test_total_return']:.2%} | 夏普 {best['test_sharpe']:.3f} "
              f"| 回撤 {best['test_max_drawdown']:.2%}")
    print(f"汇总: {summary_path}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
