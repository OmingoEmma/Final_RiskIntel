import os
import io
import glob
import json
from datetime import datetime, date

import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px


st.set_page_config(
    page_title="RiskIntel Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Utilities and cached loaders
# -----------------------------

def _resolve_first_existing_path(candidate_paths: list[str]) -> str | None:
    for candidate in candidate_paths:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


@st.cache_data(ttl=60, show_spinner=False)
def load_dataframe() -> pd.DataFrame:
    """Load core dataset with graceful fallback to a small demo frame.

    Auto-parses likely date columns and ensures a canonical `date` column exists.
    """
    csv_path = _resolve_first_existing_path(
        [
            "/workspace/data/processed/merged_features.csv",
            "data/processed/merged_features.csv",
        ]
    )

    if csv_path:
        try:
            dataframe = pd.read_csv(csv_path)
        except Exception:
            dataframe = pd.DataFrame()
    else:
        dataframe = pd.DataFrame()

    if dataframe.empty:
        # Fallback demo data
        today = pd.Timestamp(datetime.utcnow().date())
        dataframe = pd.DataFrame(
            {
                "country": ["USA", "USA", "Germany", "Germany", "Japan"],
                "title": [
                    "Policy update boosts markets",
                    "Economic report signals growth",
                    "Manufacturing slows slightly",
                    "Consumer confidence rises",
                    "Exports rebound strongly",
                ],
                "sentiment_score": [0.2, 0.35, -0.1, 0.25, 0.15],
                "gdp": [21.0, 21.1, 4.0, 4.0, 5.1],
                "unemployment": [3.6, 3.5, 5.0, 4.9, 2.7],
                "risk_score": [0.42, 0.38, 0.47, 0.41, 0.36],
                "model_version": ["v1", "v1", "v1", "v1", "v1"],
                "date": [
                    today - pd.Timedelta(days=7),
                    today - pd.Timedelta(days=3),
                    today - pd.Timedelta(days=10),
                    today - pd.Timedelta(days=2),
                    today - pd.Timedelta(days=1),
                ],
            }
        )

    # Standardize/ensure a date column exists
    likely_date_cols = [
        col
        for col in dataframe.columns
        if col.lower() in {"date", "timestamp", "published_at"}
    ]
    if likely_date_cols:
        date_col = likely_date_cols[0]
        dataframe[date_col] = pd.to_datetime(dataframe[date_col], errors="coerce")
        dataframe.rename(columns={date_col: "date"}, inplace=True)
    else:
        # Try to detect datetime-like columns
        parsed_any = False
        for col in dataframe.columns:
            if dataframe[col].dtype == object:
                try:
                    dataframe[col] = pd.to_datetime(dataframe[col])
                    dataframe.rename(columns={col: "date"}, inplace=True)
                    parsed_any = True
                    break
                except Exception:
                    continue
        if not parsed_any:
            dataframe["date"] = pd.to_datetime(datetime.utcnow().date())

    # Ensure required columns exist
    if "country" not in dataframe.columns:
        dataframe["country"] = "Unknown"
    if "risk_score" not in dataframe.columns:
        dataframe["risk_score"] = 0.0
    if "model_version" not in dataframe.columns:
        dataframe["model_version"] = "v1"

    # Normalize types
    dataframe["risk_score"] = pd.to_numeric(dataframe["risk_score"], errors="coerce").fillna(0.0)
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")

    return dataframe


@st.cache_data(ttl=60, show_spinner=False)
def load_sample_predictions(df: pd.DataFrame) -> pd.DataFrame:
    json_path = _resolve_first_existing_path(
        [
            "/workspace/examples/sample_predictions.json",
            "examples/sample_predictions.json",
        ]
    )
    if json_path:
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                parsed = json.load(handle)
            return pd.json_normalize(parsed)
        except Exception:
            pass
    # Fallback: construct from dataframe
    cols = [c for c in ["country", "date", "title", "risk_score", "model_version"] if c in df.columns]
    return df[cols].copy().sort_values("date", ascending=False).head(200)


@st.cache_data(ttl=60, show_spinner=False)
def discover_shap_images() -> dict:
    """Discover SHAP images in reports directory.

    Returns a dict with keys: summary (str|None), individuals (list[str]).
    """
    figures_dirs = [
        "/workspace/reports/figures",
        "reports/figures",
        "/workspace/reports",
        "reports",
    ]
    base_dir = None
    for d in figures_dirs:
        if os.path.isdir(d):
            base_dir = d
            break
    if base_dir is None:
        return {"summary": None, "individuals": []}

    # Likely summary image patterns
    summary_candidates = []
    for pattern in ["shap*summary*.png", "*shap*summary*.png", "shap_summary.png"]:
        summary_candidates.extend(glob.glob(os.path.join(base_dir, pattern)))
    summary_path = summary_candidates[0] if summary_candidates else None

    # Individual explanations
    individuals = []
    for pattern in ["shap*force*.png", "shap*waterfall*.png", "shap*individual*.png", "*shap*force*.png"]:
        individuals.extend(glob.glob(os.path.join(base_dir, pattern)))
    # De-duplicate and stable sort
    individuals = sorted(list(dict.fromkeys(individuals)))

    return {"summary": summary_path, "individuals": individuals}


def format_date_for_display(dt_value: pd.Timestamp | datetime | date | None) -> str:
    if dt_value is None or pd.isna(dt_value):
        return ""
    try:
        return pd.to_datetime(dt_value).strftime("%Y-%m-%d")
    except Exception:
        return str(dt_value)


# -----------------------------
# Sidebar: filters and actions
# -----------------------------

df = load_dataframe()
sample_predictions_df = load_sample_predictions(df)
shap_assets = discover_shap_images()

min_date = pd.to_datetime(df["date"]).min()
max_date = pd.to_datetime(df["date"]).max()
all_countries = sorted([c for c in df["country"].dropna().astype(str).unique()])
all_versions = sorted([v for v in df["model_version"].dropna().astype(str).unique()])

with st.sidebar:
    st.markdown("### Filters")
    selected_countries = st.multiselect(
        "Countries",
        options=all_countries,
        default=all_countries[:5] if len(all_countries) > 5 else all_countries,
    )

    date_range = st.date_input(
        "Date range",
        value=(min_date.date() if pd.notna(min_date) else date.today(), max_date.date() if pd.notna(max_date) else date.today()),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = (min_date.date(), max_date.date())

    selected_versions = st.multiselect(
        "Model versions",
        options=all_versions,
        default=all_versions,
    )

    st.markdown("### Data refresh")
    st.caption("Data caches auto-expire every 60s. Use Refresh to force reload now.")
    refresh_clicked = st.button("Refresh now", use_container_width=True)
    if refresh_clicked:
        load_dataframe.clear()
        load_sample_predictions.clear()
        discover_shap_images.clear()
        st.experimental_rerun()

    st.markdown("### Export")
    # Export filtered table CSV will be built after we have filtered_df in main area

    st.markdown("---")
    st.markdown("### Options")
    show_dataset = st.checkbox("Show raw dataset (head)", value=False)


# Apply filters
mask_country = df["country"].astype(str).isin(selected_countries) if selected_countries else pd.Series(True, index=df.index)
mask_version = df["model_version"].astype(str).isin(selected_versions) if selected_versions else pd.Series(True, index=df.index)

df_dates = df.copy()
df_dates["date_only"] = pd.to_datetime(df_dates["date"]).dt.date
mask_date = (df_dates["date_only"] >= start_date) & (df_dates["date_only"] <= end_date)

filtered_df = df_dates[mask_country & mask_version & mask_date].drop(columns=["date_only"]) if not df.empty else df_dates


# -----------------------------
# Main layout with tabs
# -----------------------------

st.title("RiskIntel Dashboard")
st.markdown("**Country-Level Risk Monitoring Using Media Sentiment + Macroeconomics**")

if show_dataset:
    st.dataframe(df.head(10), use_container_width=True)

tab_overview, tab_explain, tab_predictions, tab_feedback = st.tabs(
    ["Overview", "Model Explainability", "Predictions", "Feedback"]
)


with tab_overview:
    st.subheader("Risk over time")
    if filtered_df.empty:
        st.info("No data for the selected filters.")
    else:
        fig = px.line(
            filtered_df.sort_values("date"),
            x="date",
            y="risk_score",
            color="country" if len(selected_countries) != 1 else None,
            markers=True,
            title=None,
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=420,
            legend_title_text="Country",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

        # Export chart HTML
        html_buf = io.StringIO()
        html_buf.write(fig.to_html(include_plotlyjs="cdn", full_html=False))
        st.download_button(
            label="Download chart (HTML)",
            data=html_buf.getvalue().encode("utf-8"),
            file_name="risk_over_time.html",
            mime="text/html",
            use_container_width=True,
        )

    st.subheader("Filtered data")
    st.dataframe(filtered_df, use_container_width=True)

    # Sidebar export now that filtered_df is defined
    with st.sidebar:
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download filtered CSV",
            data=csv_buffer.getvalue(),
            file_name="filtered_predictions.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_filtered_csv",
        )


with tab_explain:
    st.subheader("SHAP Feature Importance")
    if shap_assets.get("summary") and os.path.exists(shap_assets["summary"]):
        try:
            st.image(Image.open(shap_assets["summary"]), caption="SHAP Summary Plot", use_column_width=True)
        except Exception:
            st.warning("Unable to load SHAP summary image.")
    else:
        st.info("SHAP summary plot not found in reports directory.")

    with st.expander("Individual prediction explanations", expanded=False):
        individuals = shap_assets.get("individuals", [])
        if not individuals:
            st.caption("No individual SHAP explanation images were found.")
        else:
            selected_img = st.selectbox(
                "Select explanation image",
                options=individuals,
                format_func=lambda p: os.path.basename(p),
            )
            try:
                st.image(Image.open(selected_img), caption=os.path.basename(selected_img), use_column_width=True)
            except Exception:
                st.warning("Unable to display the selected SHAP image.")

            # Zip and export all SHAP images
            import zipfile
            from io import BytesIO

            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                if shap_assets.get("summary") and os.path.exists(shap_assets["summary"]):
                    zf.write(shap_assets["summary"], arcname=os.path.basename(shap_assets["summary"]))
                for img_path in individuals:
                    if os.path.exists(img_path):
                        zf.write(img_path, arcname=os.path.basename(img_path))
            st.download_button(
                label="Download SHAP report (ZIP)",
                data=zip_buf.getvalue(),
                file_name="shap_report.zip",
                mime="application/zip",
                use_container_width=True,
            )


with tab_predictions:
    st.subheader("Sample Risk Predictions")
    st.dataframe(sample_predictions_df, use_container_width=True)
    pred_csv_buf = io.StringIO()
    sample_predictions_df.to_csv(pred_csv_buf, index=False)
    st.download_button(
        label="Download sample predictions (CSV)",
        data=pred_csv_buf.getvalue(),
        file_name="sample_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


with tab_feedback:
    st.subheader("We value your feedback")
    st.write("Help us improve the dashboard and models.")

    feedback_dir = _resolve_first_existing_path([
        "/workspace/reports/feedback",
        "reports/feedback",
        "/workspace/reports",
        "reports",
        "/workspace/data",
        "data",
    ])
    if feedback_dir is None:
        # Default to workspace reports/feedback
        feedback_dir = "/workspace/reports/feedback"
    os.makedirs(feedback_dir, exist_ok=True)
    feedback_path = os.path.join(feedback_dir, "feedback.csv")

    with st.form("feedback_form", clear_on_submit=True):
        rating = st.slider("Overall rating", min_value=1, max_value=5, value=4, help="1 = Poor, 5 = Excellent")
        comments = st.text_area("Comments", placeholder="What worked well? What can be improved?", height=120)
        email = st.text_input("Contact (optional)", placeholder="you@example.com")
        submitted = st.form_submit_button("Submit feedback")

    if submitted:
        fb_row = {
            "timestamp": datetime.utcnow().isoformat(),
            "rating": rating,
            "comments": comments.strip(),
            "email": email.strip(),
            "filters_countries": ",".join(selected_countries) if selected_countries else "",
            "filters_start_date": format_date_for_display(start_date),
            "filters_end_date": format_date_for_display(end_date),
            "filters_versions": ",".join(selected_versions) if selected_versions else "",
        }
        try:
            if os.path.exists(feedback_path):
                existing = pd.read_csv(feedback_path)
                updated = pd.concat([existing, pd.DataFrame([fb_row])], ignore_index=True)
            else:
                updated = pd.DataFrame([fb_row])
            updated.to_csv(feedback_path, index=False)
            st.success("Thank you! Your feedback has been recorded.")
        except Exception as exc:
            st.error(f"Failed to save feedback: {exc}")

    # Simple aggregate view
    if os.path.exists(feedback_path):
        try:
            feedback_df = pd.read_csv(feedback_path)
            if not feedback_df.empty and "rating" in feedback_df.columns:
                avg_rating = round(float(pd.to_numeric(feedback_df["rating"], errors="coerce").mean()), 2)
                st.metric(label="Average rating", value=avg_rating)
        except Exception:
            pass

