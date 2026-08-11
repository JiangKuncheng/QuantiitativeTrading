"""
生成当前持仓与收益图(供查看/测试)
==================================
用法: python make_holdings_chart.py
输出: output/holdings_latest.png (左: 持仓饼图; 右: 权益vs沪深300折线)
"""

from __future__ import annotations

from pathlib import Path

from qtcore.holdings_chart import build_daily_chart


ROOT = Path(__file__).resolve().parent


def main() -> int:
    out = build_daily_chart(
        db_path=ROOT / "data" / "trading.db",
        cache_dir=ROOT / "data" / "cache",
        out_path=ROOT / "output" / "holdings_latest.png",
    )
    print(f"已生成: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
