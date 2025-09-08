import types
import pytest


def stub_pipeline_behavior(text):
   text = (text or "").lower()
   if "good" in text or "gain" in text:
      return [
         {"label": "positive", "score": 0.8},
         {"label": "neutral", "score": 0.15},
         {"label": "negative", "score": 0.05},
      ]
   if "bad" in text or "loss" in text:
      return [
         {"label": "negative", "score": 0.7},
         {"label": "neutral", "score": 0.2},
         {"label": "positive", "score": 0.1},
      ]
   return [
      {"label": "neutral", "score": 0.6},
      {"label": "positive", "score": 0.2},
      {"label": "negative", "score": 0.2},
   ]


class StubTokenizer:
   def tokenize(self, w):
      return w.split()


class StubPipeline:
   def __call__(self, inputs, return_all_scores=True, truncation=True, top_k=None):
      if isinstance(inputs, list):
         return [stub_pipeline_behavior(x) for x in inputs]
      return stub_pipeline_behavior(inputs)


@pytest.fixture(autouse=True)
def stub_finbert(monkeypatch):
   import src.sentiment_analysis as sa

   def _stub_loader():
      return StubPipeline(), StubTokenizer()

   monkeypatch.setattr(sa, "_load_finbert_pipeline", _stub_loader)
   # Clear any cached pipeline/tokenizer if present
   monkeypatch.setattr(sa, "_FINBERT_PIPELINE", None, raising=False)
   monkeypatch.setattr(sa, "_FINBERT_TOKENIZER", None, raising=False)
   yield


def test_score_sentiment_float_and_range():
   from src.sentiment_analysis import score_sentiment

   s1 = score_sentiment("Markets show good momentum today")
   s2 = score_sentiment("Earnings were bad and losses widened")
   s3 = score_sentiment("")

   assert isinstance(s1, float)
   assert isinstance(s2, float)
   assert -1.0 <= s1 <= 1.0
   assert -1.0 <= s2 <= 1.0
   assert s1 > 0
   assert s2 < 0
   assert s3 == 0.0


def test_analyze_sentiments_batch_confidence_and_labels():
   from src.sentiment_analysis import analyze_sentiments

   texts = [
      "Strong quarter, very good guidance",
      "Weak results, bad outlook",
   ]
   results = analyze_sentiments(texts)
   assert len(results) == 2
   for res in results:
      assert set(["label", "score", "confidence", "probabilities"]) <= set(res.keys())
      assert 0.0 <= res["confidence"] <= 1.0
      probs = res["probabilities"]
      assert set(["positive", "negative", "neutral"]) <= set(probs.keys())
   assert results[0]["label"] == "positive"
   assert results[0]["score"] > 0
   assert results[1]["label"] == "negative"
   assert results[1]["score"] < 0


def test_fallback_to_comprehend(monkeypatch):
   import src.sentiment_analysis as sa

   # Make FinBERT raise
   def _raise_loader():
      raise RuntimeError("finbert down")

   monkeypatch.setattr(sa, "_load_finbert_pipeline", _raise_loader)

   # Stub comprehend to return positive
   def _stub_comprehend(texts):
      out = []
      for _ in texts:
         out.append({
            "label": "positive",
            "score": 0.8 - 0.1,  # p_pos - p_neg = 0.7
            "confidence": 0.8,
            "probabilities": {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
         })
      return out

   monkeypatch.setattr(sa, "_analyze_with_comprehend", _stub_comprehend)

   s = sa.score_sentiment("Stocks rally on good news")
   assert s > 0.0

