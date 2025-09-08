## API Documentation

This document summarizes the public APIs across modules. For detailed docstrings, open the corresponding files under `src/`.

### Module: `src/sentiment_analysis.py`

- `is_gpu_available() -> bool`
  - Detects if CUDA GPU is available for accelerating FinBERT.

- `analyze_sentiment(text: Optional[str]) -> Dict[str, Any>`
  - Analyze a single text with FinBERT. Returns `{label, score, confidence, probabilities}` where `score` is polarity in [-1, 1]. Falls back to AWS Comprehend on failure.

- `analyze_sentiments(texts: List[Optional[str]]) -> List[Dict[str, Any]]`
  - Batch analysis; returns a result per input.

- `score_sentiment(text: Optional[str]) -> float`
  - Backward-compatible helper returning only polarity score.

- `batch_score_sentiments(texts: List[Optional[str]]) -> List[float]`
  - Batch version returning only polarity scores.

Key environment variables:
- `FINBERT_MODEL_NAME` – HuggingFace model id (default: `ProsusAI/finbert`).

### Module: `src/explainability/shap_utils.py`

- `get_shap_explanation(model, X, feature_names=None) -> Dict[str, Any>`
  - Computes SHAP values using TreeExplainer. Returns `shap_values`, `expected_value`, and `feature_names`.

- `generate_and_save_shap_reports(model, X, feature_names=None, output_dir="reports", instance_index=0, max_display=10) -> Dict[str, str]`
  - Produces global and local explanations as PNG/HTML and returns file paths.

Artifacts written:
- `reports/shap_summary_*.png|html`
- `reports/shap_feature_importance_*.png|html`
- `reports/shap_waterfall_*.png`
- `reports/shap_force_*.html`
- `reports/shap_manifest_*.json`, `reports/latest_manifest.json`

### Module: `src/modeling/train_model.py`

- `load_data(path="data/processed/merged_features.csv") -> pd.DataFrame`
- `add_fake_risk_score(df: pd.DataFrame) -> pd.DataFrame`
- `train_and_save_model(df: pd.DataFrame) -> None`
  - Trains `RandomForestRegressor`, saves `models/risk_model.pkl`, predictions CSV/JSON under `data/processed/` and `examples/`, and SHAP artifacts under `reports/`.

### Module: `src/preprocessing/news_preprocess.py`

- `process_new_raw_files(raw_dir="data/raw", processed_path="data/processed/merged_features.csv") -> None`
  - Reads `news_*.json`, computes daily sentiment via VADER, merges macro indicators if available, and writes/updates the processed features CSV.

### Module: `src/ingestion/simulate_live_news.py`

- `fetch_top_headlines(country, api_key, keywords, page_size, logger) -> List[Dict[str, Any]]`
- `save_batch(articles, countries, keywords, logger) -> Optional[str]`
- `run_ingestion_once(countries, keywords, page_size, logger) -> bool`
- `main() -> None`
  - CLI entrypoint supporting one-shot or looped ingestion; writes to `data/raw/`, triggers preprocessing and training.

### Database Utilities: `db.py`

- `init_db() -> None`
- `insert_interaction(...) -> None`
- `insert_feedback(...) -> None`
- `fetch_dataframe(query: str, params: Tuple[Any, ...] = ()) -> pandas.DataFrame`
- `compute_metric_summary() -> MetricSummary`

### Example Usage

```python
from src.sentiment_analysis import analyze_sentiment

res = analyze_sentiment("Strong earnings and good guidance")
print(res["label"], res["score"], res["confidence"])  # positive, >0, 0..1
```

```python
from src.explainability.shap_utils import generate_and_save_shap_reports
artifacts = generate_and_save_shap_reports(model, X, feature_names=features)
print(artifacts["summary_png"])
```

