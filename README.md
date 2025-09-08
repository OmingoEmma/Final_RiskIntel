## Streamlit Demo + Feedback + Analytics

This project provides a turnkey demo app with feedback collection and analytics. It includes:

- Demo Streamlit app with a simple text summarizer and usage logging
- Feedback form (`st.form`) with structured questions
- Analytics dashboard with usage patterns and feedback summaries
- SQLite storage and a data seeding script
- 5–7 minute demo script and a professional slides template (Marp)

### Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Seed the database with synthetic data:

```bash
python seed_data.py
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit in your browser.

### Project Structure

- `streamlit_app.py`: Main demo app with simple summarization and logging
- `pages/1_Feedback.py`: Feedback form page
- `pages/2_Analytics.py`: Analytics dashboard page
- `db.py`: SQLite layer and data access utilities
- `seed_data.py`: Synthetic data generator
- `demo_script.md`: 5–7 minute demo script
- `slides/demo.md`: Slides in Marp markdown
- `data/app.db`: SQLite database (created on first run)

### Notes

- The demo "model" is a lightweight extractive summarizer for portability.
- No external services are required. All data stays local in SQLite.
- You can safely delete the database at any time; a fresh one will be created.