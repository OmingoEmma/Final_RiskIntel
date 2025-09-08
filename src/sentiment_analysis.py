import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)


# Lazy-loaded globals for FinBERT
_FINBERT_PIPELINE: Any = None
_FINBERT_TOKENIZER: Any = None


def is_gpu_available() -> bool:
   try:
      import torch  # type: ignore
      return bool(getattr(torch.cuda, "is_available", lambda: False)())
   except Exception:
      return False


def _load_finbert_pipeline() -> Tuple[Any, Any]:
   """Load and cache the FinBERT pipeline and tokenizer.

   Returns a tuple of (pipeline, tokenizer).
   """
   global _FINBERT_PIPELINE, _FINBERT_TOKENIZER
   if _FINBERT_PIPELINE is not None and _FINBERT_TOKENIZER is not None:
      return _FINBERT_PIPELINE, _FINBERT_TOKENIZER

   try:
      from transformers import (  # type: ignore
         AutoModelForSequenceClassification,
         AutoTokenizer,
         pipeline,
      )
   except Exception as exc:  # pragma: no cover - import failure path
      _logger.exception("Failed to import transformers: %s", exc)
      raise

   model_name = os.environ.get("FINBERT_MODEL_NAME", "ProsusAI/finbert")

   try:
      tokenizer = AutoTokenizer.from_pretrained(model_name)
      model = AutoModelForSequenceClassification.from_pretrained(model_name)
      device = 0 if is_gpu_available() else -1
      nlp = pipeline(
         "text-classification",
         model=model,
         tokenizer=tokenizer,
         device=device,
         return_all_scores=True,
         truncation=True,
         top_k=None,
      )
      _FINBERT_PIPELINE, _FINBERT_TOKENIZER = nlp, tokenizer
      return nlp, tokenizer
   except Exception as exc:  # pragma: no cover - network or runtime failure path
      _logger.exception("Failed to load FinBERT model '%s': %s", model_name, exc)
      raise


_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_HTML = re.compile(r"<[^>]+>")
_RE_CASHTAG = re.compile(r"\$([A-Za-z]{1,10})")


def _preprocess_text(text: Optional[str]) -> str:
   if text is None:
      return ""
   if not isinstance(text, str):
      text = str(text)
   t = text.strip()
   if not t:
      return ""
   # Normalize whitespace, remove HTML, replace URLs, normalize cashtags
   t = _RE_HTML.sub(" ", t)
   t = _RE_URL.sub(" URL ", t)
   t = _RE_CASHTAG.sub(r"\1", t)
   t = re.sub(r"\s+", " ", t)
   # FinBERT is uncased; lowercasing is acceptable
   t = t.lower()
   return t


def _chunk_text_by_tokens(text: str, tokenizer: Any, max_tokens: int = 450) -> List[str]:
   """Split long text into chunks by token count to fit BERT limits.

   max_tokens excludes special tokens; 450 is a safe default under the 512 cap.
   """
   if not text:
      return [""]
   try:
      words = text.split()
      chunks: List[str] = []
      current: List[str] = []
      current_len = 0
      for w in words:
         tok_len = len(tokenizer.tokenize(w))
         if current_len + tok_len > max_tokens and current:
            chunks.append(" ".join(current))
            current = [w]
            current_len = tok_len
         else:
            current.append(w)
            current_len += tok_len
      if current:
         chunks.append(" ".join(current))
      return chunks or [text]
   except Exception:
      # Fallback: simple naive chunk by characters
      limit = 1800
      return [text[i : i + limit] for i in range(0, len(text), limit)]


def _scores_to_polarity(prob_by_label: Dict[str, float]) -> float:
   p_pos = float(prob_by_label.get("positive", 0.0))
   p_neg = float(prob_by_label.get("negative", 0.0))
   polarity = p_pos - p_neg
   # Clamp to [-1, 1] for safety
   return max(-1.0, min(1.0, polarity))


def _aggregate_chunk_scores(chunk_outputs: List[List[Dict[str, float]]]) -> Dict[str, float]:
   # Average probabilities across chunks
   sums: Dict[str, float] = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
   count = 0
   for out in chunk_outputs:
      # out is a list like [{label: 'positive', score: 0.7}, ...]
      tmp: Dict[str, float] = {}
      for item in out:
         lbl = str(item.get("label", "")).lower()
         sc = float(item.get("score", 0.0))
         tmp[lbl] = sc
      for k in sums.keys():
         sums[k] += float(tmp.get(k, 0.0))
      count += 1
   if count == 0:
      return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
   return {k: v / count for k, v in sums.items()}


def _analyze_with_finbert(texts: List[str]) -> List[Dict[str, Any]]:
   pipeline, tokenizer = _load_finbert_pipeline()
   results: List[Dict[str, Any]] = []
   # Preprocess and chunk
   preprocessed = [_preprocess_text(t) for t in texts]
   all_chunks: List[List[str]] = [
      _chunk_text_by_tokens(t, tokenizer) if t else [""] for t in preprocessed
   ]

   # Flatten to call pipeline in batches
   flat_inputs: List[str] = [c for chunks in all_chunks for c in chunks]
   if not flat_inputs:
      return [{"label": "neutral", "score": 0.0, "confidence": 0.0, "probabilities": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}} for _ in texts]

   outputs = pipeline(flat_inputs, return_all_scores=True, truncation=True, top_k=None)
   # Reshape back per text
   idx = 0
   for chunks in all_chunks:
      num = len(chunks)
      chunk_outs = outputs[idx : idx + num]
      idx += num
      prob_by_label = _aggregate_chunk_scores(chunk_outs)
      label = max(prob_by_label.items(), key=lambda kv: kv[1])[0]
      confidence = float(prob_by_label[label])
      score = _scores_to_polarity(prob_by_label)
      results.append({
         "label": label,
         "score": score,
         "confidence": confidence,
         "probabilities": prob_by_label,
      })
   return results


def _analyze_with_comprehend(texts: List[str]) -> List[Dict[str, Any]]:
   try:
      import boto3  # type: ignore
   except Exception as exc:  # pragma: no cover - import failure path
      _logger.warning("boto3 unavailable for Comprehend fallback: %s", exc)
      return [{"label": "neutral", "score": 0.0, "confidence": 0.0, "probabilities": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}} for _ in texts]

   try:
      client = boto3.client("comprehend")
   except Exception as exc:  # pragma: no cover - runtime config failure path
      _logger.warning("AWS Comprehend client init failed: %s", exc)
      return [{"label": "neutral", "score": 0.0, "confidence": 0.0, "probabilities": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}} for _ in texts]

   results: List[Dict[str, Any]] = []
   batch_size = 25
   for i in range(0, len(texts), batch_size):
      batch = [(_preprocess_text(t) or " ")[:5000] for t in texts[i : i + batch_size]]
      try:
         resp = client.batch_detect_sentiment(TextList=batch, LanguageCode="en")
      except Exception as exc:  # pragma: no cover - service failure path
         _logger.warning("AWS Comprehend call failed: %s", exc)
         results.extend([{"label": "neutral", "score": 0.0, "confidence": 0.0, "probabilities": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}} for _ in batch])
         continue
      # Map results by index
      items = resp.get("ResultList", [])
      # Initialize defaults; fill successful ones by Index
      batch_results: List[Dict[str, Any]] = [
         {"label": "neutral", "score": 0.0, "confidence": 0.0, "probabilities": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}}
         for _ in batch
      ]
      for it in items:
         idx = int(it.get("Index", 0))
         sent = str(it.get("Sentiment", "NEUTRAL")).lower()
         scores = it.get("SentimentScore", {}) or {}
         p_pos = float(scores.get("Positive", 0.0))
         p_neg = float(scores.get("Negative", 0.0))
         p_neu = float(scores.get("Neutral", 0.0))
         p_mix = float(scores.get("Mixed", 0.0))
         prob_by_label = {"positive": p_pos, "negative": p_neg, "neutral": p_neu, "mixed": p_mix}
         # Compute polarity from Positive - Negative, ignore Mixed in polarity
         score = max(-1.0, min(1.0, p_pos - p_neg))
         # Confidence is top probability over all available labels
         label = max(prob_by_label.items(), key=lambda kv: kv[1])[0]
         confidence = float(prob_by_label[label])
         # Keep only FinBERT-like labels in probabilities for consistency
         probs_norm = {"positive": p_pos, "negative": p_neg, "neutral": p_neu}
         batch_results[idx] = {
            "label": label,
            "score": score,
            "confidence": confidence,
            "probabilities": probs_norm,
         }
      results.extend(batch_results)
   return results


def analyze_sentiment(text: Optional[str]) -> Dict[str, Any]:
   """Analyze a single text with FinBERT and include confidence and probabilities.

   Returns a dict with keys: label, score (polarity in [-1,1]), confidence, probabilities.
   """
   if not text or not isinstance(text, str):
      return {"label": "neutral", "score": 0.0, "confidence": 0.0, "probabilities": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}}
   try:
      res = _analyze_with_finbert([text])[0]
      return res
   except Exception as exc:
      _logger.warning("FinBERT analysis failed, falling back to AWS Comprehend: %s", exc)
      try:
         return _analyze_with_comprehend([text])[0]
      except Exception:
         return {"label": "neutral", "score": 0.0, "confidence": 0.0, "probabilities": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}}


def analyze_sentiments(texts: List[Optional[str]]) -> List[Dict[str, Any]]:
   if not texts:
      return []
   try:
      return _analyze_with_finbert([t or "" for t in texts])
   except Exception as exc:
      _logger.warning("FinBERT batch analysis failed, falling back to AWS Comprehend: %s", exc)
      try:
         return _analyze_with_comprehend([t or "" for t in texts])
      except Exception:
         return [{"label": "neutral", "score": 0.0, "confidence": 0.0, "probabilities": {"positive": 0.0, "negative": 0.0, "neutral": 1.0}} for _ in texts]


def score_sentiment(text: Optional[str]) -> float:
   """Backward-compatible API returning a polarity float in [-1, 1]."""
   res = analyze_sentiment(text)
   return float(res.get("score", 0.0))


def batch_score_sentiments(texts: List[Optional[str]]) -> List[float]:
   """Batch version returning only polarity scores for backward compatibility."""
   results = analyze_sentiments(texts)
   return [float(r.get("score", 0.0)) for r in results]


__all__ = [
   "is_gpu_available",
   "analyze_sentiment",
   "analyze_sentiments",
   "score_sentiment",
   "batch_score_sentiments",
]

