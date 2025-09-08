from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from db import init_db, fetch_dataframe, compute_metric_summary


def _metric(value, label, help_text=None, format_fn=None):
    col = st.container()
    with col:
        if format_fn is not None and value is not None:
            st.metric(label, format_fn(value), help=help_text)
        elif value is None:
            st.metric(label, "—", help=help_text)
        else:
            st.metric(label, value, help=help_text)


def _format_ms(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:,.0f} ms"


def _format_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{100.0 * v:,.1f}%"


def main():
    st.set_page_config(page_title="Analytics", layout="wide")
    init_db()

    st.title("Analytics Dashboard")
    st.caption("Usage patterns, performance metrics, and feedback insights.")

    interactions = fetch_dataframe("SELECT * FROM interactions")
    feedback = fetch_dataframe("SELECT * FROM feedback")

    summary = compute_metric_summary()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        _metric(summary.requests, "Requests")
    with m2:
        _metric(summary.avg_latency_ms, "Avg latency", format_fn=_format_ms)
    with m3:
        _metric(summary.success_rate, "Success rate", format_fn=_format_pct)
    with m4:
        _metric(summary.avg_usability, "Avg usability")
    with m5:
        _metric(summary.avg_accuracy, "Avg accuracy")
    with m6:
        _metric(summary.recommend_rate, "Recommend rate", format_fn=_format_pct)

    if interactions.empty and feedback.empty:
        st.info("No data yet. Use the app and submit feedback, or run `python seed_data.py`.")
        return

    st.subheader("Usage Over Time")
    if not interactions.empty:
        interactions["date"] = interactions["timestamp"].dt.floor("h")
        by_time = (interactions.groupby("date").size().reset_index(name="requests"))
        chart = (
            alt.Chart(by_time)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Time"),
                y=alt.Y("requests:Q", title="Requests"),
                tooltip=["date:T", "requests:Q"],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Latency Distribution")
        if not interactions.empty:
            hist = (
                alt.Chart(interactions)
                .mark_bar()
                .encode(
                    x=alt.X("latency_ms:Q", bin=alt.Bin(maxbins=30), title="Latency (ms)"),
                    y=alt.Y("count()", title="Count"),
                    tooltip=["count()"],
                )
                .properties(height=240)
            )
            st.altair_chart(hist, use_container_width=True)

    with c2:
        st.subheader("Success Rate Over Time")
        if not interactions.empty:
            sr = (
                interactions.assign(success_rate=interactions["success"].rolling(20, min_periods=1).mean())
                .assign(date=lambda d: d["timestamp"].dt.floor("h"))
            )
            sr_chart = (
                alt.Chart(sr)
                .mark_line()
                .encode(
                    x=alt.X("date:T", title="Time"),
                    y=alt.Y("success_rate:Q", title="Success rate"),
                    tooltip=["date:T", alt.Tooltip("success_rate:Q", format=".2f")],
                )
                .properties(height=240)
            )
            st.altair_chart(sr_chart, use_container_width=True)

    st.subheader("Breakdowns")
    b1, b2, b3 = st.columns(3)
    if not interactions.empty:
        with b1:
            st.caption("By feature")
            by_feature = interactions.groupby("feature").agg(
                requests=("id", "count"),
                avg_latency_ms=("latency_ms", "mean"),
                success_rate=("success", "mean"),
            ).reset_index()
            st.dataframe(by_feature)
        with b2:
            st.caption("By model")
            by_model = interactions.groupby("model_name").agg(
                requests=("id", "count"),
                avg_latency_ms=("latency_ms", "mean"),
                success_rate=("success", "mean"),
                avg_input_tokens=("input_tokens", "mean"),
                avg_output_tokens=("output_tokens", "mean"),
            ).reset_index()
            st.dataframe(by_model)
        with b3:
            st.caption("Tokens vs Latency")
            if "input_tokens" in interactions.columns:
                sc = (
                    alt.Chart(interactions)
                    .mark_point(opacity=0.5)
                    .encode(
                        x=alt.X("input_tokens:Q", title="Input tokens"),
                        y=alt.Y("latency_ms:Q", title="Latency (ms)"),
                        color=alt.Color("model_name:N", title="Model"),
                        tooltip=["model_name", "input_tokens", "latency_ms"],
                    )
                    .properties(height=240)
                )
                st.altair_chart(sc, use_container_width=True)

    st.subheader("Feedback Insights")
    if not feedback.empty:
        fc1, fc2 = st.columns(2)
        with fc1:
            st.caption("Ratings distribution")
            rat_df = feedback.melt(
                value_vars=["usability_rating", "accuracy_rating"],
                var_name="metric",
                value_name="rating",
            )
            rat_chart = (
                alt.Chart(rat_df)
                .mark_bar()
                .encode(
                    x=alt.X("rating:Q", bin=alt.Bin(step=1), title="Rating"),
                    y=alt.Y("count()", title="Count"),
                    color=alt.Color("metric:N", title="Metric"),
                    tooltip=["metric", "count()"],
                )
                .properties(height=240)
            )
            st.altair_chart(rat_chart, use_container_width=True)
        with fc2:
            st.caption("Usability vs Accuracy")
            uvac = (
                alt.Chart(feedback)
                .mark_circle(opacity=0.6)
                .encode(
                    x=alt.X("usability_rating:Q", title="Usability"),
                    y=alt.Y("accuracy_rating:Q", title="Accuracy"),
                    color=alt.Color("feature:N", title="Feature"),
                    tooltip=["feature", "usability_rating", "accuracy_rating"],
                )
                .properties(height=240)
            )
            st.altair_chart(uvac, use_container_width=True)

        st.caption("Feedback table")
        st.dataframe(feedback.sort_values("timestamp", ascending=False), use_container_width=True)


if __name__ == "__main__":
    main()