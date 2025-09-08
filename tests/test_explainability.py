import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor


# Ensure the src directory is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
   sys.path.insert(0, str(SRC_PATH))

from explainability.shap_utils import get_shap_explanation, generate_and_save_shap_reports


def _toy_df(n_rows: int = 50) -> pd.DataFrame:
   rng = np.random.default_rng(42)
   df = pd.DataFrame({
      "sentiment_score": rng.normal(0, 1, n_rows),
      "gdp": rng.normal(100, 10, n_rows),
      "unemployment": rng.uniform(3, 10, n_rows),
      "cpi": rng.normal(2, 0.5, n_rows),
   })
   df["risk_score"] = (
      0.4 * df["sentiment_score"]
      - 0.2 * df["gdp"]
      + 0.3 * df["unemployment"]
      + 0.1 * df["cpi"]
      + rng.normal(0, 1, n_rows)
   )
   return df


def test_get_shap_explanation_shapes():
   df = _toy_df(60)
   features = ["sentiment_score", "gdp", "unemployment", "cpi"]
   X = df[features]
   y = df["risk_score"]

   model = RandomForestRegressor(n_estimators=50, random_state=0)
   model.fit(X, y)

   result = get_shap_explanation(model, X, features)
   shap_values = result["shap_values"]
   assert shap_values.shape == (len(X), len(features))
   assert isinstance(result["expected_value"], float)
   assert result["feature_names"] == features


def test_generate_and_save_shap_reports(tmp_path: Path):
   df = _toy_df(40)
   features = ["sentiment_score", "gdp", "unemployment", "cpi"]
   X = df[features]
   y = df["risk_score"]

   model = RandomForestRegressor(n_estimators=30, random_state=1)
   model.fit(X, y)

   artifacts = generate_and_save_shap_reports(
      model=model,
      X=X,
      feature_names=features,
      output_dir=str(tmp_path),
      instance_index=0,
      max_display=8,
   )

   # Check essential artifacts
   assert "summary_png" in artifacts and os.path.exists(artifacts["summary_png"])
   assert "feature_importance_png" in artifacts and os.path.exists(artifacts["feature_importance_png"])
   assert "waterfall_png" in artifacts and os.path.exists(artifacts["waterfall_png"])
   assert "force_html" in artifacts and os.path.exists(artifacts["force_html"])
   assert "manifest_json" in artifacts and os.path.exists(artifacts["manifest_json"])


def test_get_shap_explanation_errors():
   model = RandomForestRegressor(n_estimators=10, random_state=0)
   # Not fitted model with empty data should raise a clean error from our validator
   with pytest.raises(ValueError):
      get_shap_explanation(model, None, ["a", "b"])  # type: ignore[arg-type]

   with pytest.raises(ValueError):
      get_shap_explanation(model, np.empty((0, 4)), ["f1", "f2", "f3", "f4"])  # empty X

