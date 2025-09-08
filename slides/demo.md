---
marp: true
theme: default
paginate: true
class: lead
backgroundColor: #0f172a
color: #e2e8f0
style: |
  section {
    background: #0f172a;
    color: #e2e8f0;
  }
  h1, h2, h3 { color: #f8fafc; }
  .accent { color: #38bdf8; }
  .muted { color: #94a3b8; }
  .kpi { font-size: 1.4rem; }
---

# End-to-End Demo

**Summarization · Feedback · Analytics**

<div class="muted">Streamlit + SQLite · Local-first · 5–7 minutes</div>

---

## Problem & Goal

- **Problem**: Hard to demonstrate AI features and collect actionable feedback quickly.
- **Goal**: A turnkey, local-first demo with usage logging, feedback, and analytics.

---

## Architecture

- **App**: Streamlit UI
- **Storage**: SQLite (`data/app.db`)
- **Model**: Rule-based summarizer + baseline echo
- **Analytics**: KPIs + charts (Altair)

---

## Workflow

1. Use feature (summarize text)
2. Log interaction (inputs, outputs, latency, tokens)
3. Collect feedback (usability, accuracy, NPS-style recommend)
4. Analyze metrics & patterns

---

## Demo Highlights

- Simple summarizer with configurable length
- Structured feedback via `st.form`
- Analytics dashboard: requests, latency, success, ratings

---

## Analytics Views

- Usage over time
- Latency distribution
- Success rate trend
- Breakdowns: by feature, by model, tokens vs latency
- Feedback insights: distributions and scatter

---

## Results (Sample)

- <span class="kpi">Requests:</span> 200+
- <span class="kpi">Avg latency:</span> ~180 ms
- <span class="kpi">Success rate:</span> ~95%
- <span class="kpi">Avg usability/accuracy:</span> ~4.1 / 4.0

---

## Extensibility

- Plug in hosted LLMs or internal models
- Add features and dashboards
- Export data for BI tools

---

## Next Steps

- Integrate production model endpoints
- A/B test prompts and variations
- Schedule automated reporting

---

## Thank You

`streamlit run streamlit_app.py`

Questions?