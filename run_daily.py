"""
每日自动交易入口
================

用法:
    python run_daily.py --test-email          # 发送一封测试邮件(验证邮箱配置)
    python run_daily.py --today               # 运行今天完整流程(交易日才交易/发日报)
    python run_daily.py --date 20260810       # 指定日期回补
    python run_daily.py --today --weekly --monthly   # 强制附带周报/月报

Docker/定时任务:
    部署到服务器后, 每天收盘后运行一次 run_daily.py --today
    (docker-entrypoint.sh 内置 cron: 工作日 15:30 执行)
"""

from __future__ import annotations

import argparse
from datetime import datetime

from qtcore.daily_trader import DailyTrader
from qtcore.emailer import load_email_config, send_email


def main() -> int:
    parser = argparse.ArgumentParser(prog="run_daily", description="每日自动交易与报告")
    parser.add_argument("--test-email", action="store_true", help="发送测试邮件")
    parser.add_argument("--today", action="store_true", help="运行今天完整流程")
    parser.add_argument("--settle", action="store_true", help="16:00 结算+日报+生成明日计划")
    parser.add_argument("--execute", action="store_true", help="9:20 执行当日计划(实时成交)")
    parser.add_argument("--retries", type=int, default=5, help="开盘价获取失败后的重试次数(默认5)")
    parser.add_argument("--retry-interval", type=int, default=10, help="重试间隔分钟(默认10)")
    parser.add_argument("--date", default=None, help="指定日期 YYYYMMDD(回补用)")
    parser.add_argument("--weekly", action="store_true", help="强制发送周报")
    parser.add_argument("--monthly", action="store_true", help="强制发送月报")
    parser.add_argument("--db", default="data/trading.db", help="SQLite 数据库路径")
    args = parser.parse_args()

    if args.test_email:
        cfg = load_email_config()
        body = (
            "这是一封来自 QuantTrader 量化交易系统的测试邮件。\n\n"
            "如果收到这封邮件, 说明邮箱配置正常, 后续的日报/周报/月报和突发告警都会发送到这个邮箱。\n"
            f"发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"发送方: {cfg['sender']}\n"
        )
        info = send_email("[QuantTrader] 测试邮件", body, config=cfg)
        print(f"测试邮件已发送: {info['subject']} -> {info['to']}")
        return 0

    if not (args.today or args.settle or args.execute or args.date):
        parser.error("请指定 --test-email 或 --settle/--execute/--today/--date")

    trader = DailyTrader(db_path=args.db)
    if args.execute:
        result = trader.execute_plan(
            run_date=args.date,
            retries=args.retries,
            retry_interval=args.retry_interval,
        )
        print(result)
        return 0
    result = trader.run(
        run_date=args.date,
        force_weekly=args.weekly,
        force_monthly=args.monthly,
    )
    if not result.get("trading_day"):
        print("非交易日, 未执行交易/报告")
    else:
        print(
            f"完成: 日期={result['date']} 权益={result['equity']:.2f} "
            f"今日收益={result['daily_return']:.2%} "
            f"大盘={result['benchmark_return']:.2%} 交易笔数={result['trades']}"
        )
        if result.get("failed_symbols"):
            print("部分标的数据失败:", result["failed_symbols"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
