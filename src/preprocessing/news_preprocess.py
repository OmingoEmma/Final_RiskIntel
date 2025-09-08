import os
import json
from glob import glob
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

try:
   from nltk.sentiment import SentimentIntensityAnalyzer
   import nltk
except Exception:  # pragma: no cover - best-effort import
   import nltk  # type: ignore
   nltk.download("vader_lexicon")
   from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore


def _ensure_directories() -> None:
   os.makedirs("data/processed", exist_ok=True)


def _load_json(path: str) -> Optional[Dict[str, Any]]:
   try:
      with open(path, "r", encoding="utf-8") as f:
         return json.load(f)
   except Exception:
      return None


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
   if not date_str:
      return None
   for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S%z"):
      try:
         return datetime.strptime(date_str, fmt)
      except Exception:
         continue
   try:
      return datetime.fromisoformat(date_str)
   except Exception:
      return None


def _compute_sentiment_scores(texts: List[str]) -> List[float]:
   try:
      sia = SentimentIntensityAnalyzer()
   except LookupError:
      import nltk  # lazy download if needed
      nltk.download("vader_lexicon")
      sia = SentimentIntensityAnalyzer()

   scores: List[float] = []
   for t in texts:
      t2 = (t or "").strip()
      if not t2:
         scores.append(0.0)
         continue
      compound = float(sia.polarity_scores(t2).get("compound", 0.0))
      scores.append(compound)
   return scores


def _load_macro_indicators() -> pd.DataFrame:
   macro_path = os.path.join("data", "external", "macro.csv")
   if os.path.exists(macro_path):
      try:
         df = pd.read_csv(macro_path)
         # Expect columns: date,gdp,unemployment,cpi
         if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
         return df[["date", "gdp", "unemployment", "cpi"]]
      except Exception:
         pass
   # Fallback defaults if none available
   return pd.DataFrame({
      "date": [],
      "gdp": [],
      "unemployment": [],
      "cpi": [],
   })


def process_new_raw_files(raw_dir: str = os.path.join("data", "raw"), processed_path: str = os.path.join("data", "processed", "merged_features.csv")) -> None:
   _ensure_directories()

   files = sorted(glob(os.path.join(raw_dir, "news_*.json")))
   if not files:
      return

   records: List[Dict[str, Any]] = []
   for path in files:
      payload = _load_json(path)
      if not payload:
         continue
      articles = payload.get("articles", []) or []
      for a in articles:
         title = a.get("title") or ""
         description = a.get("description") or ""
         content = a.get("content") or ""
         combined = " \n".join([title, description, content]).strip()
         records.append({
            "publishedAt": a.get("publishedAt"),
            "country": a.get("country"),
            "text": combined,
         })

   if not records:
      return

   df = pd.DataFrame.from_records(records)
   df["published_dt"] = df["publishedAt"].apply(_parse_date)
   df["date"] = pd.to_datetime(df["published_dt"]).dt.date
   df["sentiment_score"] = _compute_sentiment_scores(df["text"].astype(str).tolist())

   # Aggregate to daily signal per country, then overall average
   daily_country = (
      df.dropna(subset=["date"])
        .groupby(["date", "country"], as_index=False)[["sentiment_score"]]
        .mean()
   )
   daily = (
      daily_country.groupby("date", as_index=False)[["sentiment_score"]]
                  .mean()
   )

   macro = _load_macro_indicators()
   if not macro.empty:
      merged = pd.merge(daily, macro, on="date", how="left")
   else:
      # Provide neutral defaults if macro not provided
      merged = daily.copy()
      merged["gdp"] = 0.0
      merged["unemployment"] = 0.0
      merged["cpi"] = 0.0

   # If an existing processed file exists, append new rows and drop duplicates on date
   if os.path.exists(processed_path):
      try:
         existing = pd.read_csv(processed_path)
         if "date" in existing.columns:
            existing["date"] = pd.to_datetime(existing["date"]).dt.date
         merged = pd.concat([existing, merged], axis=0, ignore_index=True)
         merged = merged.drop_duplicates(subset=["date"], keep="last")
      except Exception:
         # If existing is unreadable, just overwrite
         pass

   # Ensure column order for downstream model
   columns = ["date", "sentiment_score", "gdp", "unemployment", "cpi"]
   for col in columns:
      if col not in merged.columns:
         merged[col] = 0.0 if col != "date" else pd.NaT
   merged = merged[columns]

   merged.to_csv(processed_path, index=False)


if __name__ == "__main__":
   process_new_raw_files()

