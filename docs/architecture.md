## Architecture Overview

This system ingests financial news, preprocesses it into features, scores sentiment, trains a simple risk model, and exposes insights via a Streamlit UI with analytics and explainability artifacts.

### Components

- Ingestion (`src/ingestion/simulate_live_news.py`): Fetches headlines from NewsAPI, filters financial content, writes timestamped JSON to `data/raw/`.
- Preprocessing (`src/preprocessing/news_preprocess.py`): Aggregates news into daily signals, merges macro indicators, writes `data/processed/merged_features.csv`.
- Sentiment (`src/sentiment_analysis.py`): FinBERT-based sentiment scoring with AWS Comprehend fallback.
- Modeling (`src/modeling/train_model.py`): Trains RandomForestRegressor on features, outputs predictions and SHAP explainability artifacts under `reports/`.
- Explainability (`src/explainability/shap_utils.py`): SHAP value computation and report generation (summary, feature importance, waterfall, force plot).
- UI (`streamlit_app.py`, `pages/*`): Streamlit app for demo, feedback capture, and analytics.
- Storage (`db.py`): SQLite database for interactions and feedback.

### Data Flow

```text
NewsAPI -> data/raw/news_*.json -> preprocessing -> data/processed/merged_features.csv
                                                   -> modeling/train -> reports/*.png, *.html, *.json
                                                   -> models/risk_model.pkl
```

### Sequence Diagram (Textual)

```text
User -> Streamlit: Request summary / navigate analytics
Streamlit -> SQLite: insert_interaction / fetch analytics
Operator -> Ingestion: run_ingestion_once/loop
Ingestion -> NewsAPI: fetch_top_headlines
Ingestion -> disk: save_batch (data/raw)
Ingestion -> Preprocess: process_new_raw_files -> data/processed/merged_features.csv
Ingestion -> Modeling: train_and_save_model -> models/, reports/
Modeling -> Explainability: generate_and_save_shap_reports -> reports/
```

### Operational Concerns

- Idempotent preprocessing appends and deduplicates by date.
- Ingestion uses exponential backoff and filters on financial keywords.
- SHAP plots use headless Matplotlib backend for CI environments.
- FinBERT runs on GPU if available; otherwise CPU. Fallback to AWS Comprehend on failure.

### Scaling Considerations

- Replace SQLite with Postgres for concurrent writes.
- Move ingestion to a scheduled job or streaming pipeline.
- Containerize and run behind a reverse proxy (e.g., Nginx) for production.
- Use object storage (e.g., S3) for artifacts and logs.

