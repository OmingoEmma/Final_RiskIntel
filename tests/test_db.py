from pathlib import Path
import os
import sys
from datetime import datetime


def test_db_flow(tmp_path: Path):
    proj = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(proj))
    import db

    # redirect DB path
    db.DB_PATH = os.path.join(tmp_path, "app.db")

    db.init_db()

    db.insert_interaction(
        session_id="s1",
        timestamp=datetime.utcnow(),
        feature="summarize",
        user_input="hello",
        model_output="hello",
        model_name="RuleBased v1",
        latency_ms=10.0,
        success=True,
        input_tokens=1,
        output_tokens=1,
    )

    db.insert_feedback(
        timestamp=datetime.utcnow(),
        session_id="s1",
        feature="summarize",
        usability_rating=4,
        accuracy_rating=5,
        would_recommend=True,
        comments="good",
    )

    df_i = db.fetch_dataframe("SELECT * FROM interactions")
    df_f = db.fetch_dataframe("SELECT * FROM feedback")

    assert len(df_i) == 1
    assert len(df_f) == 1

    summary = db.compute_metric_summary()
    assert summary.requests == 1
    assert summary.avg_latency_ms >= 0

