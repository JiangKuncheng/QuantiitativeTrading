"""支持 `python -m qtcore` 直接启动。"""

from qtcore.main_manager import main


if __name__ == "__main__":
    raise SystemExit(main())
