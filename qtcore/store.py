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
CREATE TABLE IF NOT EXISTS scheme_equity_daily (
    scheme TEXT NOT NULL,
    date TEXT NOT NULL,
    equity REAL,
    daily_return REAL,
    top_symbols TEXT,
    created_at TEXT,
    PRIMARY KEY (scheme, date)
);
CREATE TABLE IF NOT EXISTS scheme_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scheme TEXT NOT NULL,
    date TEXT NOT NULL,
    code TEXT NOT NULL,
    side TEXT NOT NULL,
    units INTEGER,
    price REAL,
    commission REAL,
    pnl REAL,
    reason TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS scheme_positions (
    scheme TEXT NOT NULL,
    date TEXT NOT NULL,
    code TEXT NOT NULL,
    units INTEGER NOT NULL,
    avg_price REAL,
    last_price REAL,
    market_value REAL,
    created_at TEXT,
    PRIMARY KEY (scheme, date, code)
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
CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    code TEXT NOT NULL,
    action TEXT NOT NULL,
    units INTEGER,
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

    def save_scheme_equity(self, scheme: str, date: str, equity: float, daily_return: float, top_symbols: str) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO scheme_equity_daily
               (scheme, date, equity, daily_return, top_symbols, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (scheme, date, equity, daily_return, top_symbols, datetime.now().isoformat(timespec="seconds")),
        )
        self.conn.commit()

    def save_scheme_trades(self, scheme: str, date: str, rows: list[dict[str, Any]]) -> None:
        """保存某方案当日模拟成交(重跑时先清掉当日旧记录, 保证幂等)。"""
        self.conn.execute(
            "DELETE FROM scheme_trades WHERE scheme = ? AND date = ?", (scheme, date)
        )
        now = datetime.now().isoformat(timespec="seconds")
        self.conn.executemany(
            """INSERT INTO scheme_trades
               (scheme, date, code, side, units, price, commission, pnl, reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    scheme, date, r["code"], r.get("side"), r.get("units"),
                    r.get("price"), r.get("commission"), r.get("pnl"), r.get("reason"), now,
                )
                for r in rows
            ],
        )
        self.conn.commit()

    def save_scheme_positions(self, scheme: str, date: str, rows: list[dict[str, Any]]) -> None:
        """保存某方案当日收盘持仓快照(重跑时先清掉当日旧记录)。"""
        self.conn.execute(
            "DELETE FROM scheme_positions WHERE scheme = ? AND date = ?", (scheme, date)
        )
        now = datetime.now().isoformat(timespec="seconds")
        self.conn.executemany(
            """INSERT INTO scheme_positions
               (scheme, date, code, units, avg_price, last_price, market_value, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    scheme, date, r["code"], r.get("units", 0), r.get("avg_price"),
                    r.get("last_price"), r.get("market_value"), now,
                )
                for r in rows
            ],
        )
        self.conn.commit()

    def scheme_trades_on(self, scheme: str, date: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM scheme_trades WHERE scheme = ? AND date = ? ORDER BY id",
            (scheme, date),
        ).fetchall()
        return [dict(r) for r in rows]

    def scheme_positions_on(self, scheme: str, date: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM scheme_positions WHERE scheme = ? AND date = ? ORDER BY code",
            (scheme, date),
        ).fetchall()
        return [dict(r) for r in rows]

    def scheme_equity_series(self, scheme: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM scheme_equity_daily WHERE scheme = ? ORDER BY date",
            (scheme,),
        ).fetchall()
        return [dict(r) for r in rows]

    def prev_scheme_equity(self, scheme: str, date_iso: str) -> float | None:
        row = self.conn.execute(
            "SELECT equity FROM scheme_equity_daily WHERE scheme = ? AND date < ? ORDER BY date DESC LIMIT 1",
            (scheme, date_iso),
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

    def save_plans(self, date: str, plans: list[dict[str, Any]]) -> None:
        """保存某交易日的计划(先清后写, 可重跑)。"""
        self.conn.execute("DELETE FROM plans WHERE date = ?", (date,))
        self.conn.executemany(
            "INSERT INTO plans (date, code, action, units, params, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    date,
                    p["code"],
                    p["action"],
                    p.get("units"),
                    p.get("params", ""),
                    datetime.now().isoformat(timespec="seconds"),
                )
                for p in plans
            ],
        )
        self.conn.commit()

    def load_plans(self, date: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM plans WHERE date = ? ORDER BY code",
            (date,),
        ).fetchall()
        return [dict(r) for r in rows]

    def holdings_net(self) -> dict[str, dict[str, Any]]:
        """按成交记录计算当前持仓(账户口径): code -> {units, avg_cost}。"""
        net: dict[str, int] = {}
        cost: dict[str, float] = {}
        for code, side, units, price in self.conn.execute(
            "SELECT code, side, units, price FROM trades ORDER BY id"
        ):
            u = int(units)
            if side in ("BUY", "SELL_SHORT"):
                net[code] = net.get(code, 0) + u
                cost[code] = cost.get(code, 0.0) + u * float(price)
            else:
                net[code] = net.get(code, 0) - u
        out: dict[str, dict[str, Any]] = {}
        for code, units in net.items():
            if units > 0:
                out[code] = {
                    "units": units,
                    "avg_cost": round(cost.get(code, 0.0) / units, 4),
                }
        return out

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
