from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional, Tuple

DB_PATH = os.path.join("data", "app.db")


def ensure_db_dir_exists() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


@contextmanager
def get_conn() -> Iterable[sqlite3.Connection]:
    ensure_db_dir_exists()
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp DATETIME,
                feature TEXT,
                user_input TEXT,
                model_output TEXT,
                model_name TEXT,
                latency_ms REAL,
                success INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                session_id TEXT,
                feature TEXT,
                usability_rating INTEGER,
                accuracy_rating INTEGER,
                would_recommend INTEGER,
                comments TEXT
            );
            """
        )


def insert_interaction(
    session_id: str,
    timestamp: datetime,
    feature: str,
    user_input: str,
    model_output: str,
    model_name: str,
    latency_ms: float,
    success: bool,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO interactions (
                session_id, timestamp, feature, user_input, model_output,
                model_name, latency_ms, success, input_tokens, output_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                timestamp,
                feature,
                user_input,
                model_output,
                model_name,
                latency_ms,
                1 if success else 0,
                input_tokens,
                output_tokens,
            ),
        )


def insert_feedback(
    timestamp: datetime,
    session_id: str,
    feature: str,
    usability_rating: int,
    accuracy_rating: int,
    would_recommend: bool,
    comments: Optional[str],
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO feedback (
                timestamp, session_id, feature, usability_rating, accuracy_rating,
                would_recommend, comments
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                session_id,
                feature,
                usability_rating,
                accuracy_rating,
                1 if would_recommend else 0,
                comments,
            ),
        )


def fetch_dataframe(query: str, params: Tuple[Any, ...] = ()) -> "pandas.DataFrame":
    import pandas as pd
    with get_conn() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    # Coerce timestamp if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


@dataclass
class MetricSummary:
    requests: int
    avg_latency_ms: float
    success_rate: float
    avg_usability: Optional[float]
    avg_accuracy: Optional[float]
    recommend_rate: Optional[float]


def compute_metric_summary() -> MetricSummary:
    interactions = fetch_dataframe("SELECT * FROM interactions")
    feedback_df = fetch_dataframe("SELECT * FROM feedback")
    requests = len(interactions)
    avg_latency = float(interactions["latency_ms"].mean()) if requests > 0 else 0.0
    success_rate = float(interactions["success"].mean()) if requests > 0 else 0.0
    avg_usability = float(feedback_df["usability_rating"].mean()) if len(feedback_df) > 0 else None
    avg_accuracy = float(feedback_df["accuracy_rating"].mean()) if len(feedback_df) > 0 else None
    recommend_rate = float(feedback_df["would_recommend"].mean()) if len(feedback_df) > 0 else None
    return MetricSummary(
        requests=requests,
        avg_latency_ms=avg_latency,
        success_rate=success_rate,
        avg_usability=avg_usability,
        avg_accuracy=avg_accuracy,
        recommend_rate=recommend_rate,
    )