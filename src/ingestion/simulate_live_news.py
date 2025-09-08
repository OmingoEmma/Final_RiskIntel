import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

import requests


# Ensure "src" is importable when running this file directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
   sys.path.append(SRC_DIR)

from preprocessing.news_preprocess import process_new_raw_files  # noqa: E402
from modeling.train_model import load_data, add_fake_risk_score, train_and_save_model  # noqa: E402


def _ensure_directories() -> None:
   os.makedirs("data/raw", exist_ok=True)
   os.makedirs("data/processed", exist_ok=True)
   os.makedirs("logs", exist_ok=True)


def _setup_logger(verbose: bool = False) -> logging.Logger:
   _ensure_directories()
   logger = logging.getLogger("ingestion")
   logger.setLevel(logging.DEBUG if verbose else logging.INFO)

   # Avoid duplicate handlers
   if logger.handlers:
      return logger

   formatter = logging.Formatter(
      fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
   )

   file_handler = logging.FileHandler("logs/ingestion.log")
   file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
   file_handler.setFormatter(formatter)
   logger.addHandler(file_handler)

   console_handler = logging.StreamHandler()
   console_handler.setLevel(logging.INFO)
   console_handler.setFormatter(formatter)
   logger.addHandler(console_handler)

   return logger


def _timestamp_filename() -> str:
   # DDMMYY_HHMM
   return datetime.now().strftime("%d%m%y_%H%M")


def _request_with_backoff(url: str, params: Dict[str, Any], headers: Optional[Dict[str, str]], logger: logging.Logger, max_retries: int = 5) -> Optional[requests.Response]:
   backoff_seconds = 2
   for attempt in range(1, max_retries + 1):
      try:
         resp = requests.get(url, params=params, headers=headers, timeout=20)
         if resp.status_code == 200:
            return resp
         if resp.status_code in (429, 500, 502, 503, 504):
            logger.warning(f"Transient error {resp.status_code} from NewsAPI (attempt {attempt}/{max_retries}). Backing off {backoff_seconds}s.")
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 60)
            continue
         logger.error(f"Non-retryable error {resp.status_code}: {resp.text[:300]}")
         return None
      except requests.RequestException as exc:
         logger.warning(f"Request exception on attempt {attempt}/{max_retries}: {exc}. Backing off {backoff_seconds}s.")
         time.sleep(backoff_seconds)
         backoff_seconds = min(backoff_seconds * 2, 60)
   logger.error("Exceeded maximum retries contacting NewsAPI")
   return None


def _contains_financial_keyword(article: Dict[str, Any], keywords: List[str]) -> bool:
   haystacks: List[str] = []
   for key in ("title", "description", "content"):
      value = article.get(key) or ""
      if isinstance(value, str):
         haystacks.append(value.lower())
   text = " \n".join(haystacks)
   return any(kw.lower() in text for kw in keywords)


def fetch_top_headlines(country: str, api_key: str, keywords: List[str], page_size: int, logger: logging.Logger) -> List[Dict[str, Any]]:
   url = "https://newsapi.org/v2/top-headlines"
   # Best-effort server-side narrowing; still filter client-side for robustness
   query_string = " OR ".join([kw for kw in keywords if kw])
   params = {
      "country": country,
      "pageSize": page_size,
      "category": "business",
   }
   if query_string:
      params["q"] = query_string

   headers = {"X-Api-Key": api_key}
   resp = _request_with_backoff(url, params, headers, logger)
   if resp is None:
      return []

   try:
      data = resp.json()
   except Exception:
      logger.error("Failed to parse NewsAPI response as JSON")
      return []

   if data.get("status") != "ok":
      logger.error(f"NewsAPI returned error: {data}")
      return []

   articles = data.get("articles", []) or []
   filtered = [a for a in articles if _contains_financial_keyword(a, keywords)]
   logger.info(f"Fetched {len(articles)} headlines for {country}; {len(filtered)} matched financial filter")
   return filtered


def save_batch(articles: List[Dict[str, Any]], countries: List[str], keywords: List[str], logger: logging.Logger) -> Optional[str]:
   if not articles:
      logger.info("No articles to save for this batch")
      return None

   ts = _timestamp_filename()
   path = os.path.join("data", "raw", f"news_{ts}.json")
   payload = {
      "timestamp": datetime.now().isoformat(),
      "countries": countries,
      "keywords": keywords,
      "count": len(articles),
      "articles": articles,
   }
   try:
      with open(path, "w", encoding="utf-8") as f:
         json.dump(payload, f, ensure_ascii=False, indent=2)
      logger.info(f"Saved {len(articles)} articles to {path}")
      return path
   except Exception as exc:
      logger.error(f"Failed to write batch JSON: {exc}")
      return None


def run_ingestion_once(countries: List[str], keywords: List[str], page_size: int, logger: logging.Logger) -> bool:
   api_key = os.getenv("NEWSAPI_API_KEY")
   if not api_key:
      logger.error("NEWSAPI_API_KEY not set in environment. Aborting this cycle.")
      return False

   all_articles: List[Dict[str, Any]] = []
   for country in countries:
      batch = fetch_top_headlines(country=country, api_key=api_key, keywords=keywords, page_size=page_size, logger=logger)
      if batch:
         # Normalize fields we care about for downstream processing
         for a in batch:
            normalized = {
               "title": a.get("title"),
               "description": a.get("description"),
               "content": a.get("content"),
               "source": (a.get("source") or {}).get("name"),
               "url": a.get("url"),
               "publishedAt": a.get("publishedAt"),
               "country": country,
            }
            all_articles.append(normalized)

   saved_path = save_batch(all_articles, countries, keywords, logger)

   # Even if no new data, attempt preprocessing to pick up any previous unsynced files
   try:
      process_new_raw_files(raw_dir=os.path.join("data", "raw"), processed_path=os.path.join("data", "processed", "merged_features.csv"))
   except Exception as exc:
      logger.error(f"Preprocessing failed: {exc}")

   # Trigger (re-)scoring/model update
   try:
      df = load_data(path=os.path.join("data", "processed", "merged_features.csv"))
      df = add_fake_risk_score(df)
      # Keep merged_features up-to-date after score augmentation for consistency with current trainer
      df.to_csv(os.path.join("data", "processed", "merged_features.csv"), index=False)
      train_and_save_model(df)
      logger.info("Model training and scoring complete.")
   except Exception as exc:
      logger.error(f"Scoring failed: {exc}")

   return saved_path is not None


def main() -> None:
   parser = argparse.ArgumentParser(description="Simulate live news ingestion using NewsAPI")
   parser.add_argument("--countries", type=str, default="gb,ke", help="Comma-separated country codes, e.g., gb,ke")
   parser.add_argument("--keywords", type=str, default="finance,economy,loan,debt,bank,market,stocks,budget,investment,interest,inflation,currency,forex,credit,mortgage,bond,treasury", help="Comma-separated financial keywords")
   parser.add_argument("--page-size", type=int, default=100, help="NewsAPI pageSize per country (max 100)")
   parser.add_argument("--interval", type=int, default=30, help="Interval in minutes between fetches when looping")
   parser.add_argument("--loop", action="store_true", help="Run continuously, sleeping between fetches")
   parser.add_argument("--verbose", action="store_true", help="Verbose logging")

   args = parser.parse_args()
   logger = _setup_logger(verbose=bool(args.verbose))
   _ensure_directories()

   countries = [c.strip() for c in args.countries.split(",") if c.strip()]
   keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]

   if not args.loop:
      run_ingestion_once(countries=countries, keywords=keywords, page_size=int(args.page_size), logger=logger)
      return

   interval_seconds = max(60, int(args.interval) * 60)  # minimum 1 minute safety
   logger.info(f"Starting live ingestion loop. Countries={countries} Interval={interval_seconds // 60}m")
   while True:
      try:
         run_ingestion_once(countries=countries, keywords=keywords, page_size=int(args.page_size), logger=logger)
      except Exception as exc:
         logger.error(f"Unexpected error during ingestion cycle: {exc}")
      logger.info(f"Sleeping for {interval_seconds // 60} minutes...")
      time.sleep(interval_seconds)


if __name__ == "__main__":
   main()

