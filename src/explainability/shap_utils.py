import os
import json
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap


def _ensure_directory(path: str) -> None:
   try:
      os.makedirs(path, exist_ok=True)
   except Exception as exc:
      raise RuntimeError(f"Failed to create directory: {path}") from exc


def _now_stamp() -> str:
   return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_shap_explanation(
   model: object,
   X: Union[pd.DataFrame, np.ndarray],
   feature_names: Optional[List[str]] = None,
) -> Dict[str, Union[np.ndarray, float, List[str]]]:
   """
   Compute SHAP values using TreeExplainer for a tree-based model.

   Returns a dict with keys: shap_values (ndarray), expected_value (float), feature_names (list[str]).
   """
   if X is None:
      raise ValueError("X must not be None")

   if isinstance(X, pd.DataFrame):
      data_matrix = X.values
      resolved_feature_names = list(X.columns) if feature_names is None else list(feature_names)
   else:
      data_matrix = np.asarray(X)
      if feature_names is None:
         resolved_feature_names = [f"feature_{i}" for i in range(data_matrix.shape[1])]
      else:
         resolved_feature_names = list(feature_names)

   if data_matrix.ndim != 2 or data_matrix.shape[0] == 0 or data_matrix.shape[1] == 0:
      raise ValueError("X must be a non-empty 2D array-like with shape (n_samples, n_features)")

   try:
      # Explicitly use TreeExplainer for tree-based models as requested
      explainer = shap.TreeExplainer(model)
      # For tree models, shap_values returns ndarray for regression
      shap_values = explainer.shap_values(data_matrix)
      # expected_value is a scalar for regression
      expected_value = float(np.asarray(explainer.expected_value).ravel()[0])
      return {
         "shap_values": np.asarray(shap_values),
         "expected_value": expected_value,
         "feature_names": resolved_feature_names,
      }
   except Exception as exc:
      # Surface a clean error while preserving original exception details
      raise RuntimeError("Failed to compute SHAP values with TreeExplainer") from exc


def generate_and_save_shap_reports(
   model: object,
   X: Union[pd.DataFrame, np.ndarray],
   feature_names: Optional[List[str]] = None,
   output_dir: str = "reports",
   instance_index: int = 0,
   max_display: int = 10,
) -> Dict[str, str]:
   """
   Generate global and local SHAP reports and save as PNG and HTML.

   Returns a dict of artifact file paths.
   """
   _ensure_directory(output_dir)

   # Normalize X and names
   if isinstance(X, pd.DataFrame):
      X_df = X.copy()
      resolved_feature_names = list(X_df.columns) if feature_names is None else list(feature_names)
   else:
      X_arr = np.asarray(X)
      if feature_names is None:
         resolved_feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]
      else:
         resolved_feature_names = list(feature_names)
      X_df = pd.DataFrame(X_arr, columns=resolved_feature_names)

   if X_df.empty:
      raise ValueError("X must contain at least one row to generate reports")

   # Compute SHAP values
   try:
      result = get_shap_explanation(model, X_df, resolved_feature_names)
   except Exception as exc:
      raise RuntimeError("SHAP computation failed; cannot generate reports") from exc

   shap_values: np.ndarray = np.asarray(result["shap_values"])  # (n_samples, n_features)
   expected_value: float = float(result["expected_value"])  # scalar for regression

   timestamp = _now_stamp()
   artifacts: Dict[str, str] = {}

   # Global summary (beeswarm) PNG
   try:
      plt.figure(figsize=(10, 6))
      shap.summary_plot(shap_values, X_df, show=False, feature_names=resolved_feature_names)
      summary_png = os.path.join(output_dir, f"shap_summary_{timestamp}.png")
      plt.tight_layout()
      plt.savefig(summary_png, dpi=200)
      plt.close()
      artifacts["summary_png"] = summary_png
      # Minimal HTML wrapper for Streamlit embedding
      summary_html = os.path.join(output_dir, f"shap_summary_{timestamp}.html")
      with open(summary_html, "w", encoding="utf-8") as fp:
         fp.write(f"<html><head><title>SHAP Summary</title></head><body><img src=\"{os.path.basename(summary_png)}\" style=\"max-width:100%;\"/></body></html>")
      artifacts["summary_html"] = summary_html
   except Exception as exc:
      warnings.warn(f"Failed to generate SHAP summary plot: {exc}")

   # Global feature importance (bar) PNG
   try:
      plt.figure(figsize=(10, 6))
      shap.summary_plot(shap_values, X_df, plot_type="bar", show=False, feature_names=resolved_feature_names)
      bar_png = os.path.join(output_dir, f"shap_feature_importance_{timestamp}.png")
      plt.tight_layout()
      plt.savefig(bar_png, dpi=200)
      plt.close()
      artifacts["feature_importance_png"] = bar_png
      # Minimal HTML wrapper
      bar_html = os.path.join(output_dir, f"shap_feature_importance_{timestamp}.html")
      with open(bar_html, "w", encoding="utf-8") as fp:
         fp.write(f"<html><head><title>SHAP Feature Importance</title></head><body><img src=\"{os.path.basename(bar_png)}\" style=\"max-width:100%;\"/></body></html>")
      artifacts["feature_importance_html"] = bar_html
   except Exception as exc:
      warnings.warn(f"Failed to generate SHAP feature importance bar plot: {exc}")

   # Local explanation (waterfall PNG) and force plot HTML
   try:
      idx = int(instance_index) if 0 <= int(instance_index) < len(X_df) else 0
      row_values = shap_values[idx]
      row_data = X_df.iloc[idx]

      # Waterfall plot (PNG)
      try:
         explanation = shap.Explanation(
            values=row_values,
            base_values=expected_value,
            data=row_data.values,
            feature_names=resolved_feature_names,
         )
         plt.figure(figsize=(10, 6))
         shap.plots.waterfall(explanation, max_display=max_display, show=False)
         waterfall_png = os.path.join(output_dir, f"shap_waterfall_{timestamp}.png")
         plt.tight_layout()
         plt.savefig(waterfall_png, dpi=200)
         plt.close()
         artifacts["waterfall_png"] = waterfall_png
      except Exception as exc_wf:
         warnings.warn(f"Failed to generate SHAP waterfall plot: {exc_wf}")

      # Force plot (interactive HTML)
      try:
         force_plot = shap.force_plot(
            expected_value,
            row_values,
            row_data,
            feature_names=resolved_feature_names,
            matplotlib=False,
         )
         force_html = os.path.join(output_dir, f"shap_force_{timestamp}.html")
         shap.save_html(force_html, force_plot)
         artifacts["force_html"] = force_html
      except Exception as exc_force:
         warnings.warn(f"Failed to generate SHAP force plot HTML: {exc_force}")
   except Exception as exc:
      warnings.warn(f"Local explanation generation failed: {exc}")

   # Manifest for Streamlit to discover artifacts
   try:
      manifest_path = os.path.join(output_dir, f"shap_manifest_{timestamp}.json")
      with open(manifest_path, "w", encoding="utf-8") as fp:
         json.dump({"artifacts": artifacts, "timestamp": timestamp}, fp, indent=2)
      artifacts["manifest_json"] = manifest_path

      # Update latest pointer for dashboard to poll a stable path
      latest_pointer = os.path.join(output_dir, "latest_manifest.json")
      with open(latest_pointer, "w", encoding="utf-8") as fp:
         json.dump({"manifest": os.path.basename(manifest_path)}, fp, indent=2)
      artifacts["latest_manifest"] = latest_pointer
   except Exception as exc:
      warnings.warn(f"Failed to write SHAP manifest: {exc}")

   return artifacts

