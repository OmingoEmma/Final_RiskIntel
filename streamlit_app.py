import time
import uuid
from datetime import datetime

import streamlit as st

from db import init_db, insert_interaction


def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def summarize_text(text: str, max_sentences: int = 2) -> str:
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    return ". ".join(sentences[:max_sentences]) + ("." if sentences[:max_sentences] else "")


def main():
    st.set_page_config(page_title="Demo App: Summarizer + Feedback + Analytics", layout="wide")
    init_db()

    st.title("End-to-End Demo: Summarization, Feedback, and Analytics")
    st.caption("Record interactions, collect feedback, and analyze usage in one app.")

    session_id = get_session_id()

    with st.sidebar:
        st.subheader("Demo Controls")
        model_name = st.selectbox("Model", ["RuleBased v1", "Echo v1"], index=0)
        max_sentences = st.slider("Max summary sentences", min_value=1, max_value=5, value=2)

    st.header("Summarize Text")
    default_text = (
        "Streamlit makes it easy to build data apps. This demo shows a simple summarizer. "
        "We also collect feedback via a form and visualize usage analytics. "
        "All data is stored locally in SQLite for portability."
    )
    user_text = st.text_area("Enter text to summarize", value=default_text, height=180)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Summarize"):
            start = time.perf_counter()
            if model_name == "RuleBased v1":
                output = summarize_text(user_text, max_sentences=max_sentences)
            else:
                output = user_text[: max(60, len(user_text) // 2)]
            latency_ms = (time.perf_counter() - start) * 1000.0

            st.success("Summary generated")
            st.code(output)

            insert_interaction(
                session_id=session_id,
                timestamp=datetime.utcnow(),
                feature="summarize",
                user_input=user_text,
                model_output=output,
                model_name=model_name,
                latency_ms=latency_ms,
                success=True,
                input_tokens=len(user_text.split()),
                output_tokens=len(output.split()),
            )
    with col2:
        st.info(
            "Tip: Use the Feedback page to rate usability and accuracy of the outputs.\n"
            "Then, visit Analytics to explore usage patterns and metrics."
        )

    st.divider()
    st.markdown(
        "Navigate to the Feedback and Analytics pages from the sidebar or top navigation."
    )


if __name__ == "__main__":
    main()