"""极简 .env 加载器(零第三方依赖)。

规则:
- 每行 ``KEY=VALUE``, 以 ``#`` 开头为注释;
- 已存在的系统环境变量不会被覆盖;
- 值可带单/双引号, 读取时自动去除;
- 空值或含占位符 ``YOUR_`` 的值会被忽略(方便直接复制 .env.example 使用)。
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_dotenv(path: str | Path | None = None) -> Path | None:
    """读取项目根目录的 .env 并写入 os.environ, 返回文件路径(不存在则返回 None)。"""
    env_file = Path(path) if path else DEFAULT_ENV_FILE
    if not env_file.exists():
        return None

    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value or "YOUR_" in value:
            continue
        if key not in os.environ:
            os.environ[key] = value
    return env_file
