# 5–7 Minute Demo Script: End-to-End Workflow

Duration target: 6 minutes. Keep a steady pace and focus on outcomes.

## 0. Opening (30s)
- Introduce the goal: demonstrate an end-to-end AI feature workflow—usage, feedback collection, and analytics.
- Mention local-only setup: Streamlit + SQLite; no external services.

## 1. App Overview (30s)
- Show the app header and sidebar.
- Explain two "models": RuleBased v1 (simple summarizer) and Echo v1 (baseline).
- Highlight pages: Main app, Feedback, Analytics.

## 2. Live Summarization (90s)
- Paste a small paragraph (3–6 sentences).
- Set Max summary sentences to 2.
- Click Summarize; point out the generated summary and speed.
- Talking points:
  - We log each interaction: inputs, outputs, latency, tokens, model name, success.
  - Data stored in `data/app.db` for analysis.

## 3. Submit Feedback (60s)
- Navigate to Feedback page.
- Select feature "summarize", set Usability=4–5, Accuracy=4, "Would you recommend?" = Yes.
- Add a short comment about clarity or control over length.
- Submit and confirm the success message.
- Talking points:
  - Structured `st.form` enables consistent ratings and optional free text.
  - Feedback links to session for cohort analysis.

## 4. Analytics Dashboard (2 min)
- Navigate to Analytics page.
- Walk through KPIs: Requests, Avg latency, Success rate, Avg usability/accuracy, Recommend rate.
- Show Usage Over Time chart; explain that this reflects demo interactions.
- Show Latency Distribution and Success Rate Over Time; mention rolling metrics.
- Open Breakdowns:
  - By feature and by model: requests, average latency, success rate, token metrics.
  - Scatter plot: input tokens vs latency per model.
- Feedback Insights:
  - Ratings distribution and usability vs accuracy scatter.
  - Feedback table with timestamps for recency.

## 5. Seeded Data (optional, 30s)
- If the dataset is sparse, run `python seed_data.py` prior to the demo.
- Mention synthetic but realistic ranges (latency, ratings, success).

## 6. Wrap-Up (30s)
- Key takeaways:
  - End-to-end loop: Use → Log → Collect Feedback → Analyze → Improve.
  - Easy to extend: swap models, add features, export data.
- Next steps: plug into your preferred model API; build team dashboards.