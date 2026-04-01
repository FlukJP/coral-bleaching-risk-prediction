from __future__ import annotations

from io import StringIO

import pandas as pd
import streamlit as st

from model_inference import (
    DATA_PATH,
    MLP_ARTIFACT_PATH,
    STACKING_ARTIFACT_PATH,
    load_mlp_artifact,
    load_reference_features,
    load_stacking_artifact,
    predict_all_models,
)


st.set_page_config(
    page_title="Coral Bleaching AI Studio",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 30%),
            radial-gradient(circle at top right, rgba(8, 145, 178, 0.14), transparent 28%),
            linear-gradient(180deg, #f5fbfa 0%, #eef8fb 45%, #f9fcff 100%);
    }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(8, 145, 178, 0.96), rgba(15, 118, 110, 0.94));
        color: #f8fffe;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2.3rem;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0;
        opacity: 0.96;
        font-size: 1rem;
        max-width: 58rem;
    }
    .panel {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }
    .status-ok, .status-missing {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        font-size: 0.86rem;
        font-weight: 600;
        margin: 0.15rem 0.2rem 0.15rem 0;
    }
    .status-ok {
        background: rgba(16, 185, 129, 0.14);
        color: #047857;
    }
    .status-missing {
        background: rgba(239, 68, 68, 0.12);
        color: #b91c1c;
    }
    .note {
        background: rgba(255, 255, 255, 0.74);
        border-left: 4px solid #0891b2;
        padding: 0.85rem 1rem;
        border-radius: 14px;
        margin: 0.4rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_reference_features() -> pd.DataFrame:
    return load_reference_features()


@st.cache_resource
def get_stacking_artifact():
    return load_stacking_artifact()


@st.cache_resource
def get_mlp_bundle():
    return load_mlp_artifact()


def artifact_status(path) -> str:
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        return f'<span class="status-ok">{path.name} ready | {size_mb:.1f} MB</span>'
    return f'<span class="status-missing">{path.name} missing</span>'


def render_single_prediction(prediction_row: pd.Series):
    metric_cols = st.columns(4)
    metric_cols[0].metric("Hybrid Prediction", f"{prediction_row['hybrid_prediction']:.2f}%")
    metric_cols[1].metric("Stacking Prediction", f"{prediction_row['stacking_prediction']:.2f}%")
    metric_cols[2].metric("Residual MLP", f"{prediction_row['mlp_prediction']:.2f}%")
    metric_cols[3].metric("Risk Probability", f"{prediction_row['bleaching_risk_probability'] * 100:.1f}%")

    flag = "High risk" if int(prediction_row["bleaching_risk_flag"]) == 1 else "Below threshold"
    gap = prediction_row["hybrid_vs_mlp_gap"]
    st.markdown(
        f"""
        <div class="note">
        Hybrid model status: <strong>{flag}</strong><br>
        Two-model gap: <strong>{gap:+.2f}</strong> percentage points between hybrid and MLP predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_template_csv(df: pd.DataFrame) -> str:
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


st.markdown(
    """
    <section class="hero">
        <h1>Coral Bleaching AI Studio</h1>
        <p>
            Compare the hybrid Stacking Ensemble and the Custom Residual MLP in one place.
            This page accepts the processed feature schema from <code>data/bleaching_model_ready.csv</code>,
            runs both models, and returns ready-to-use bleaching predictions for your web workflow.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


left, right = st.columns([1.6, 1.0], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Input")
    st.caption("Use the processed feature schema only. Raw environmental data must be transformed first.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Model Status")
    st.markdown(artifact_status(STACKING_ARTIFACT_PATH), unsafe_allow_html=True)
    st.markdown(artifact_status(MLP_ARTIFACT_PATH), unsafe_allow_html=True)
    st.caption(f"Reference schema: {DATA_PATH}")
    st.markdown("</div>", unsafe_allow_html=True)


if not STACKING_ARTIFACT_PATH.exists() or not MLP_ARTIFACT_PATH.exists():
    st.error("At least one model artifact is missing. Re-run the training notebooks before using this web app.")
    st.stop()


reference_features = get_reference_features()
stacking_artifact = get_stacking_artifact()
mlp_artifact, mlp_model = get_mlp_bundle()

st.markdown(
    f"""
    <div class="note">
    Required input columns: <strong>{reference_features.shape[1]}</strong> features.
    You can edit sample rows directly below or upload a CSV that follows the same schema as the processed training data.
    </div>
    """,
    unsafe_allow_html=True,
)

template_col, schema_col = st.columns([1.0, 1.6], gap="large")
with template_col:
    st.download_button(
        "Download input template",
        data=build_template_csv(reference_features.head(1)),
        file_name="coral_bleaching_input_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
with schema_col:
    st.caption("Template includes one sample row with the exact processed feature schema expected by both models.")


input_tab, upload_tab = st.tabs(["Manual Editor", "CSV Upload"])

with input_tab:
    st.write("Edit one or more rows directly in the table.")
    seed_row_count = st.slider("Rows to preload", min_value=1, max_value=5, value=1, step=1)
    editable_seed = reference_features.head(seed_row_count).copy()
    manual_input_df = st.data_editor(
        editable_seed,
        num_rows="dynamic",
        use_container_width=True,
        height=320,
        key="manual_editor",
    )

with upload_tab:
    st.write("Upload a processed CSV with the same columns as `bleaching_model_ready.csv` minus the target column.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    uploaded_df = None
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        st.dataframe(uploaded_df.head(10), use_container_width=True)


input_source = st.radio("Prediction source", options=["Manual editor", "Uploaded CSV"], horizontal=True)
source_df = manual_input_df if input_source == "Manual editor" else uploaded_df

predict_col, sample_col = st.columns([1.2, 1.8])
with predict_col:
    run_prediction = st.button("Run Both Models", type="primary", use_container_width=True)
with sample_col:
    st.caption("Tip: if you upload a CSV, extra columns are ignored. Missing required feature columns will stop prediction.")


if run_prediction:
    if source_df is None or source_df.empty:
        st.warning("Provide at least one input row before running prediction.")
    else:
        with st.spinner("Running both models..."):
            try:
                prediction_df, fill_notes = predict_all_models(
                    source_df,
                    reference_features,
                    stacking_artifact,
                    mlp_artifact,
                    mlp_model,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.success(f"Finished prediction for {len(prediction_df):,} row(s).")

                if fill_notes:
                    with st.expander("Auto-filled values"):
                        for note in fill_notes:
                            st.write(f"- {note}")

                if len(prediction_df) == 1:
                    render_single_prediction(prediction_df.iloc[0])

                result_cols = [
                    "stacking_prediction",
                    "bleaching_risk_probability",
                    "bleaching_risk_flag",
                    "two_stage_prediction",
                    "hybrid_prediction",
                    "mlp_prediction",
                    "hybrid_vs_mlp_gap",
                ]

                summary_left, summary_right = st.columns([1.3, 1.0], gap="large")

                with summary_left:
                    st.subheader("Prediction Results")
                    st.dataframe(
                        prediction_df[result_cols].round(4),
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download predictions as CSV",
                        data=build_download_bytes(prediction_df),
                        file_name="coral_bleaching_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                with summary_right:
                    st.subheader("Quick Summary")
                    summary_df = prediction_df[result_cols].copy()
                    summary_frame = pd.DataFrame(
                        {
                            "Metric": [
                                "Average hybrid prediction",
                                "Average MLP prediction",
                                "Average risk probability",
                                "Rows above risk threshold",
                            ],
                            "Value": [
                                f"{summary_df['hybrid_prediction'].mean():.2f}%",
                                f"{summary_df['mlp_prediction'].mean():.2f}%",
                                f"{summary_df['bleaching_risk_probability'].mean() * 100:.2f}%",
                                f"{int(summary_df['bleaching_risk_flag'].sum())} / {len(summary_df)}",
                            ],
                        }
                    )
                    st.dataframe(summary_frame, use_container_width=True, hide_index=True)

                    chart_df = prediction_df[["hybrid_prediction", "mlp_prediction"]].head(20).reset_index(drop=True)
                    st.caption("First 20 rows")
                    st.bar_chart(chart_df, use_container_width=True)


with st.expander("Model Details"):
    st.write("Hybrid Stacking Ensemble")
    st.write(
        "Combines the stacking regressor with a two-stage classifier/regressor flow. "
        "The final prediction blends stacking output with the two-stage severity estimate."
    )
    st.write("Custom Residual MLP")
    st.write(
        "Uses a residual neural network with tabular attention and a saved scaler. "
        "This gives you a neural comparison against the tree-based hybrid pipeline."
    )
