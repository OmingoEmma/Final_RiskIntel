from datetime import datetime

import streamlit as st

from db import init_db, insert_feedback


def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = "anonymous"
    return st.session_state["session_id"]


def main():
    st.set_page_config(page_title="Feedback", layout="centered")
    init_db()

    st.title("Feedback")
    st.caption("Help us improve by rating usability and accuracy.")

    session_id = get_session_id()

    with st.form("feedback_form", clear_on_submit=True):
        feature = st.selectbox("Feature", ["summarize"], index=0)
        usability = st.slider("Usability (1=poor, 5=excellent)", 1, 5, 4)
        accuracy = st.slider("Accuracy (1=poor, 5=excellent)", 1, 5, 4)
        would_recommend = st.radio("Would you recommend this?", ["Yes", "No"], index=0)
        comments = st.text_area("Comments (optional)")

        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            insert_feedback(
                timestamp=datetime.utcnow(),
                session_id=session_id,
                feature=feature,
                usability_rating=int(usability),
                accuracy_rating=int(accuracy),
                would_recommend=(would_recommend == "Yes"),
                comments=comments.strip() if comments else None,
            )
            st.success("Thank you! Your feedback has been recorded.")

    st.info("All feedback is stored locally in SQLite (`data/app.db`).")


if __name__ == "__main__":
    main()