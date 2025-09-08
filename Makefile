PYTHON ?= python3
INTERVAL ?= 30
COUNTRIES ?= gb,ke
KEYWORDS ?= finance,economy,loan,debt,bank,market,stocks,budget,investment,interest,inflation,currency,forex,credit,mortgage,bond,treasury

.PHONY: install-deps setup-nltk ingest-once ingest-live stop-ingest preprocess score all

install-deps:
	$(PYTHON) -m pip install -r requirements.txt

setup-nltk:
	$(PYTHON) -c "import nltk; nltk.download('vader_lexicon')" | cat

preprocess:
	$(PYTHON) src/preprocessing/news_preprocess.py

ingest-once:
	@mkdir -p logs
	NEWSAPI_API_KEY=$$NEWSAPI_API_KEY $(PYTHON) src/ingestion/simulate_live_news.py --countries $(COUNTRIES) --keywords "$(KEYWORDS)" --page-size 100 --interval $(INTERVAL)

# Runs the live ingestion loop in the background (nohup) and writes to logs/ingestion.out
ingest-live:
	@mkdir -p logs
	@echo "Starting live ingestion loop (interval=$(INTERVAL)m, countries=$(COUNTRIES))"
	NO_COLOR=1 NEWSAPI_API_KEY=$$NEWSAPI_API_KEY nohup $(PYTHON) src/ingestion/simulate_live_news.py --loop --countries $(COUNTRIES) --keywords "$(KEYWORDS)" --page-size 100 --interval $(INTERVAL) >> logs/ingestion.out 2>&1 & echo $$! > .ingestion.pid
	@echo "PID written to .ingestion.pid; logs in logs/ingestion.out and logs/ingestion.log"

stop-ingest:
	@if [ -f .ingestion.pid ]; then kill `cat .ingestion.pid` || true; rm -f .ingestion.pid; echo "Stopped live ingestion"; else echo "No .ingestion.pid found"; fi

score:
	$(PYTHON) src/modeling/train_model.py

all: ingest-once preprocess score

