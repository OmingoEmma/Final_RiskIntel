import os
from pathlib import Path
import json
import pandas as pd


def test_process_new_raw_files(tmp_path: Path):
    # Create minimal raw payload
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    payload = {
        "timestamp": "2024-01-01T00:00:00Z",
        "countries": ["gb"],
        "keywords": ["finance"],
        "count": 1,
        "articles": [
            {
                "title": "Markets rally on good news",
                "description": "Stocks close higher",
                "content": "Investors cheer gains.",
                "publishedAt": "2024-01-01T12:00:00Z",
                "country": "gb",
            }
        ],
    }
    with open(raw_dir / "news_010124_1200.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    processed_path = tmp_path / "data" / "processed" / "merged_features.csv"

    # Run preprocessing
    import sys
    SRC = Path(__file__).resolve().parents[1] / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from preprocessing.news_preprocess import process_new_raw_files

    process_new_raw_files(raw_dir=str(raw_dir), processed_path=str(processed_path))

    assert processed_path.exists()
    df = pd.read_csv(processed_path)
    assert set(["date", "sentiment_score", "gdp", "unemployment", "cpi"]).issubset(df.columns)
    assert len(df) >= 1

