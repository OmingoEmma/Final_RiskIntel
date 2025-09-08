import sys
from pathlib import Path

import types


def test_fetch_top_headlines_filters_keywords(monkeypatch):
    SRC = Path(__file__).resolve().parents[1] / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from ingestion import simulate_live_news as ing

    # Stub network call to return deterministic payload
    class Resp:
        status_code = 200

        def json(self):
            return {
                "status": "ok",
                "articles": [
                    {"title": "Sports news", "description": "Football", "content": "Match"},
                    {"title": "Finance update", "description": "Stocks rally", "content": "Markets gain"},
                ],
            }

    def _stub_request(url, params, headers, logger):
        return Resp()

    monkeypatch.setattr(ing, "_request_with_backoff", _stub_request)

    out = ing.fetch_top_headlines(country="gb", api_key="x", keywords=["finance", "stocks"], page_size=10, logger=types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None))
    assert len(out) == 1
    assert "Finance" in (out[0]["title"] or "")

