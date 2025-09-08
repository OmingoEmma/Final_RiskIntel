## RiskIntel: Financial Risk System (Demo + Analytics + Explainability)

This repository contains a production-ready demo of a financial risk analytics system with:

- Streamlit user interface and dashboards
- News ingestion and preprocessing pipeline
- Sentiment analysis (FinBERT with AWS Comprehend fallback)
- Modeling with scikit-learn and SHAP explainability reports
- SQLite persistence, feedback capture, and analytics
- Pytest-based tests with coverage and CI pipeline
- Dockerized environment and environment configuration templates

For a deeper dive, see:

- `docs/architecture.md` – System design and data flows
- `docs/tech_stack.md` – Technology choices and rationale
- `docs/api_documentation.md` – API reference for modules

### Quickstart (Local)

1) Python setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Optional: seed example data

```bash
python seed_data.py
```

3) Run Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open the URL printed by Streamlit.

### Make Targets

```bash
make install-deps      # Install dependencies
make setup-nltk        # Ensure VADER lexicon is available
make preprocess        # Process any raw news into features
make ingest-once       # Fetch news once (requires NEWSAPI_API_KEY)
make ingest-live       # Continuous ingestion loop (nohup)
make stop-ingest       # Stop the live loop
make score             # Train model and generate SHAP reports
```

### Running Tests and Coverage

```bash
pytest -q
pytest --cov=src --cov-report=term-missing --cov-report=xml
```

After running, a coverage XML is produced for CI and a terminal report is shown.

### Docker

Build and run the containerized app:

```bash
docker build -t riskintel:latest .
docker run --rm -p 8501:8501 \
  -e NEWSAPI_API_KEY=$NEWSAPI_API_KEY \
  -v $(pwd)/data:/app/data \
  riskintel:latest
```

Then open `http://localhost:8501`.

You can also use docker compose (see `docker-compose.yml`):

```bash
docker compose up --build
```

### Environment Configuration

Copy `.env.example` to `.env` and set values. The application reads environment variables like:

- `NEWSAPI_API_KEY`: API key for NewsAPI (optional unless running ingestion)
- `FINBERT_MODEL_NAME`: Override default FinBERT model (default `ProsusAI/finbert`)

### Project Structure

- `streamlit_app.py`: Main Streamlit app (summarization demo + logging)
- `pages/1_Feedback.py`: Feedback form page
- `pages/2_Analytics.py`: Analytics dashboard page
- `src/ingestion/simulate_live_news.py`: News ingestion and orchestration
- `src/preprocessing/news_preprocess.py`: Preprocess raw news JSON to features
- `src/sentiment_analysis.py`: FinBERT sentiment with AWS fallback
- `src/modeling/train_model.py`: Train RandomForest and produce SHAP reports
- `src/explainability/shap_utils.py`: SHAP utilities and report generation
- `db.py`: SQLite schema and access
- `tests/`: Pytest suite
- `docs/`: Architecture, tech stack, API docs

### Screenshots

Below are representative screenshots; your paths may vary based on runtime:

```text
reports/
  shap_summary_YYYYMMDD_HHMMSS.png
  shap_feature_importance_YYYYMMDD_HHMMSS.png
  shap_waterfall_YYYYMMDD_HHMMSS.png
```

You can generate them by running:

```bash
make score
```

### Deployment

Recommended deployment options:

- Docker container on a VM or container service
- Streamlit Community Cloud (for demos; ensure no secrets are committed)
- Any orchestrator that can run `streamlit run streamlit_app.py`

CI/CD is configured via GitHub Actions to run tests and upload coverage. You can extend it to build and push Docker images.