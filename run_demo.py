"""
一键演示入口
============

用法:
    python run_demo.py                 # 优先拉取真实数据(000001), 失败自动降级合成数据
    python run_demo.py --offline       # 强制离线合成数据
    python run_demo.py --symbol 600519 --fast 10 --slow 30 --capital 2000000

注意: 需要先创建 Conda 环境 QuantitativeTrading 并安装依赖(见 README.md)。
"""

from qtcore.main_manager import main


if __name__ == "__main__":
    raise SystemExit(main())
