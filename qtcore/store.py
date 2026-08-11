"""
SQLite 存储模块
===============
保存每日权益、交易记录、报告与运行状态, 供日/周/月报告聚合。
单文件数据库(data/trading.db), Docker 部署时挂载该目录即可持久化。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS equity_daily (
    date TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    cash REAL,
    position_value REAL,
    daily_return REAL,
    benchmark_return REAL,
    strategy_total REAL,
    benchmark_total REAL,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    code TEXT NOT NULL,
    name TEXT,
    side TEXT NOT NULL,
    units INTEGER,
    price REAL,
    commission REAL,
    pnl REAL,
    reason TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    code TEXT NOT NULL,
    action TEXT,
    params TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    kind TEXT NOT NULL,
    subject TEXT,
    body TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    stage TEXT,
    status TEXT,
    detail TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS account_state (
    date TEXT PRIMARY KEY,
    start_date TEXT NOT NULL,
    start_capital REAL NOT NULL,
    cash REAL NOT NULL,
    positions TEXT,
    pending TEXT,
    halted INTEGER DEFAULT 0,
    peak_equity REAL,
    last_equity REAL,
    created_at TEXT
);
"""


class Store:
    """SQLite 数据访问层(单线程使用)。"""

    def __init__(self, db_path: Path | str = "data/trading.db") -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def save_equity(self, row: dict[str, Any]) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO equity_daily
               (date, equity, cash, position_value, daily_return,
                benchmark_return, strategy_total, benchmark_total, created_at)
               VALUES (:date, :equity, :cash, :position_value, :daily_return,
                       :benchmark_return, :strategy_total, :benchmark_total, :created_at)""",
            {**row, "created_at": datetime.now().isoformat(timespec="seconds")},
        )
        self.conn.commit()

    def save_trades(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        # 建仓记录可重跑: 先清掉当日旧建仓单, 避免重复累积
        if any(r.get("reason") == "initial_build" for r in rows):
            dates = {r["date"] for r in rows if r.get("reason") == "initial_build"}
            for d in dates:
                self.conn.execute(
                    "DELETE FROM trades WHERE date = ? AND reason = 'initial_build'",
                    (d,),
                )
        self.conn.executemany(
            """INSERT INTO trades (date, code, name, side, units, price,
               commission, pnl, reason, created_at)
               VALUES (:date, :code, :name, :side, :units, :price,
                       :commission, :pnl, :reason, :created_at)""",
            [
                {**r, "created_at": datetime.now().isoformat(timespec="seconds")}
                for r in rows
            ],
        )
        self.conn.commit()

    def save_report(self, date: str, kind: str, subject: str, body: str) -> None:
        self.conn.execute(
            "INSERT INTO reports (date, kind, subject, body, created_at) VALUES (?, ?, ?, ?, ?)",
            (date, kind, subject, body, datetime.now().isoformat(timespec="seconds")),
        )
        self.conn.commit()

    def log_run(self, date: str, stage: str, status: str, detail: str = "") -> None:
        self.conn.execute(
            "INSERT INTO run_log (date, stage, status, detail, created_at) VALUES (?, ?, ?, ?, ?)",
            (date, stage, status, detail, datetime.now().isoformat(timespec="seconds")),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def equity_between(self, start: str, end: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM equity_daily WHERE date >= ? AND date <= ? ORDER BY date",
            (start, end),
        ).fetchall()
        return [dict(r) for r in rows]

    def trades_between(self, start: str, end: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE date >= ? AND date <= ? ORDER BY date, id",
            (start, end),
        ).fetchall()
        return [dict(r) for r in rows]

    def last_equity_date(self) -> str | None:
        row = self.conn.execute("SELECT MAX(date) AS d FROM equity_daily").fetchone()
        return row["d"] if row and row["d"] else None

    def prev_equity_before(self, date_iso: str) -> float | None:
        """取指定日期之前最近一个交易日的账户权益(用于记账链)。"""
        row = self.conn.execute(
            "SELECT equity FROM equity_daily WHERE date < ? ORDER BY date DESC LIMIT 1",
            (date_iso,),
        ).fetchone()
        return row["equity"] if row else None

    def latest_reports(self, kind: str, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM reports WHERE kind = ? ORDER BY date DESC LIMIT ?",
            (kind, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_signals(self, date: str, actions: dict[str, int], params: dict[str, Any]) -> None:
        """把当天各标的的目标仓位写入 signals 表。"""
        for code, action in actions.items():
            self.conn.execute(
                "INSERT INTO signals (date, code, action, params, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    date,
                    code,
                    str(action),
                    json.dumps(params, ensure_ascii=False),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
        self.conn.commit()

    def save_state(
        self,
        date: str,
        start_date: str,
        start_capital: float,
        cash: float,
        positions: dict[str, Any],
        pending: dict[str, int] | None,
        halted: bool,
        peak_equity: float,
        last_equity: float,
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO account_state
               (date, start_date, start_capital, cash, positions, pending,
                halted, peak_equity, last_equity, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                date,
                start_date,
                float(start_capital),
                float(cash),
                json.dumps(positions, ensure_ascii=False),
                json.dumps(pending, ensure_ascii=False) if pending is not None else None,
                1 if halted else 0,
                float(peak_equity),
                float(last_equity),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        self.conn.commit()

    def load_state(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM account_state ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["positions"] = json.loads(d["positions"] or "{}")
        d["pending"] = json.loads(d["pending"]) if d.get("pending") else None
        d["halted"] = bool(d.get("halted"))
        return d

    def clear_date(self, date: str) -> None:
        """重跑某一天前清理该日全部记录(权益/交易/信号/报告/账户状态)。"""
        for table in ("equity_daily", "trades", "signals", "reports", "account_state"):
            self.conn.execute(f"DELETE FROM {table} WHERE date = ?", (date,))
        self.conn.commit()
