"""Minimal persistent run history using SQLite."""
import json
import os
import sqlite3
from typing import Any


def _connect():
    path = os.getenv("DATABASE_PATH", "neuroforge.db")
    connection = sqlite3.connect(path)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal TEXT NOT NULL,
            answer TEXT NOT NULL,
            steps INTEGER NOT NULL,
            trace_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return connection


def save_run(goal: str, answer: str, steps: int, trace: list[dict[str, Any]]) -> int:
    with _connect() as connection:
        cursor = connection.execute(
            "INSERT INTO runs (goal, answer, steps, trace_json) VALUES (?, ?, ?, ?)",
            (goal, answer, steps, json.dumps(trace)),
        )
        return int(cursor.lastrowid)


def recent_runs(limit: int = 10) -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT id, goal, answer, steps, created_at FROM runs ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 50)),),
        ).fetchall()
    return [
        {"id": row[0], "goal": row[1], "answer": row[2], "steps": row[3], "created_at": row[4]}
        for row in rows
    ]
