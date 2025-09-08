import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

from src.explainability.shap_utils import (
   get_shap_explanation,
   generate_and_save_shap_reports,
)


def load_data(path="data/processed/merged_features.csv"):
   return pd.read_csv(path)


def add_fake_risk_score(df):
   # You’ll replace this later with a real scoring model or human labels
   np.random.seed(42)
   df["risk_score"] = (
       0.4 * df["sentiment_score"].fillna(0)
       - 0.2 * df["gdp"].fillna(0)
       + 0.3 * df["unemployment"].fillna(0)
       + 0.1 * df["cpi"].fillna(0)
       + np.random.normal(0, 1, len(df))  # noise
   )
   return df


def _timestamp() -> str:
   return datetime.now().strftime("%Y%m%d_%H%M%S")


def train_and_save_model(df):
   features = ["sentiment_score", "gdp", "unemployment", "cpi"]
   target = "risk_score"

   df = df.dropna(subset=features + [target])
   X = df[features]
   y = df[target]

   # For now, use all data as train+test to keep parity with current behavior
   X_train, X_test, y_train, y_test = X, X, y, y

   model = RandomForestRegressor(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)

   preds = model.predict(X_test)

   df.loc[:, "predicted_risk"] = model.predict(X)
   os.makedirs("data/processed", exist_ok=True)
   df.to_csv("data/processed/predicted_risks.csv", index=False)
   print("Saved: data/processed/predicted_risks.csv")

   mae = mean_absolute_error(y_test, preds)
   r2 = r2_score(y_test, preds)

   print(f"MAE: {mae:.4f}")
   print(f"R2 Score: {r2:.4f}")

   os.makedirs("models", exist_ok=True)
   os.makedirs("examples", exist_ok=True)
   os.makedirs("reports", exist_ok=True)

   joblib.dump(model, "models/risk_model.pkl")

   # Save sample predictions
   df_preds = pd.DataFrame({
       "actual": y_test,
       "predicted": preds
   }).reset_index(drop=True)

   df_preds.head(5).to_json("examples/sample_predictions.json", orient="records", indent=2)

   # SHAP explainability integration (global + local)
   try:
      artifacts = generate_and_save_shap_reports(
         model=model,
         X=X_test,
         feature_names=features,
         output_dir="reports",
         instance_index=0,
         max_display=10,
      )

      # Save raw SHAP values for downstream use (Streamlit, analysis)
      try:
         shap_result = get_shap_explanation(model, X_test, features)
         shap_values = np.asarray(shap_result["shap_values"])  # (n_samples, n_features)
         ts = _timestamp()
         shap_csv = os.path.join("reports", f"shap_values_{ts}.csv")
         shap_npy = os.path.join("reports", f"shap_values_{ts}.npy")
         np.save(shap_npy, shap_values)
         shap_df = pd.DataFrame(shap_values, columns=features)
         shap_df.to_csv(shap_csv, index=False)
         artifacts["shap_values_csv"] = shap_csv
         artifacts["shap_values_npy"] = shap_npy
      except Exception as exc_values:
         print(f"Warning: failed to persist raw SHAP values: {exc_values}")

      # Write a lightweight manifest for Streamlit integration
      manifest = {
         "metrics": {"mae": float(mae), "r2": float(r2)},
         "artifacts": artifacts,
      }
      with open(os.path.join("reports", "latest_training_manifest.json"), "w", encoding="utf-8") as fp:
         json.dump(manifest, fp, indent=2)

      print("SHAP artifacts saved to reports/ (PNG + HTML + manifest)")
   except Exception as exc:
      print(f"Warning: SHAP explainability generation failed: {exc}")


if __name__ == "__main__":
   df = load_data()
   df = add_fake_risk_score(df)
   # After risk score is added
   os.makedirs("data/processed", exist_ok=True)
   df.to_csv("data/processed/merged_features.csv", index=False)

   train_and_save_model(df)

