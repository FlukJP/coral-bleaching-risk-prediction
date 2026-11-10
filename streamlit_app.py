from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from model_inference import DATA_PATH, MLP_ARTIFACT_PATH, STACKING_ARTIFACT_PATH
from model_metadata import (
    FRESH,
    UNVERIFIABLE,
    DataStatus,
    get_dataset_status,
    get_model_data_status,
)


st.set_page_config(
    page_title="Coral Bleaching AI Studio",
    layout="wide",
)


def format_size_mb(path: Path) -> str:
    if not path.exists():
        return "-"
    return f"{path.stat().st_size / (1024 * 1024):.1f} MB"


def model_status_html(
    path: Path,
    title: str,
    description: str,
    missing_action: str,
    data_status: DataStatus,
) -> str:
    escaped_title = html.escape(title)
    escaped_description = html.escape(description)
    status_class = "ok" if data_status.status == FRESH else "warn"
    status_details = (
        f"""
            <div class="status-meta"><strong>Training data:</strong> {html.escape(data_status.status)}</div>
            <div class="status-meta">{html.escape(data_status.message)}</div>
        """
    )
    evaluation_details = _evaluation_html(data_status)
    if path.exists():
        return f"""
        <div class="status-card ok">
            <div class="status-icon ok">OK</div>
            <div class="status-body">
                <div class="status-title">{escaped_title}</div>
                <div class="status-text">{escaped_description}</div>
                <div class="status-meta"><strong>Status:</strong> Online | <strong>File size:</strong> {format_size_mb(path)}</div>
                <div class="status-meta"><strong>Data status:</strong> <span class="{status_class}">{html.escape(data_status.status)}</span></div>
                <div class="status-meta">{html.escape(data_status.message)}</div>
                {evaluation_details}
            </div>
        </div>
        """
    return f"""
    <div class="status-card warn">
        <div class="status-icon warn">!</div>
        <div class="status-body">
            <div class="status-title">{escaped_title}</div>
            <div class="status-text">{escaped_description}</div>
            <div class="status-meta"><strong>Status:</strong> Re-training required</div>
            <div class="status-meta"><strong>Artifact:</strong> {path.name} missing</div>
            <div class="status-meta"><strong>Action:</strong> {html.escape(missing_action)}</div>
            {status_details}
        </div>
    </div>
    """


DATA_METADATA_PATH = DATA_PATH.with_name("metadata.json")
STACKING_METADATA_PATH = STACKING_ARTIFACT_PATH.parent / "metadata.json"
MLP_METADATA_PATH = MLP_ARTIFACT_PATH.parent / "metadata.json"

dataset_status = get_dataset_status(DATA_PATH, DATA_METADATA_PATH)
stacking_data_status = get_model_data_status(
    DATA_PATH, DATA_METADATA_PATH, STACKING_METADATA_PATH
)
mlp_data_status = get_model_data_status(
    DATA_PATH, DATA_METADATA_PATH, MLP_METADATA_PATH
)


def _evaluation_html(data_status: DataStatus) -> str:
    model_metadata = data_status.model_metadata
    if model_metadata is None:
        return (
            '<div class="status-meta"><strong>Evaluation:</strong> '
            f'{html.escape(UNVERIFIABLE)} (ไม่มี metadata ที่ตรวจสอบได้)</div>'
        )

    evaluation = model_metadata["evaluation"]
    method = html.escape(evaluation["method"])
    n_folds = evaluation["n_folds"]
    n_validation_rows = evaluation["n_validation_rows"]
    evaluated_at = html.escape(evaluation["evaluated_at"])
    metrics = html.escape(
        json.dumps(evaluation["metrics"], ensure_ascii=False, sort_keys=True)
    )
    coverage = model_metadata["coverage"]
    matched_rows = coverage["matched_rows"]
    denominator = coverage["match_denominator"]
    matched_share = coverage["matched_share"] * 100
    imputed_shares = ", ".join(
        f"{html.escape(feature)} {share * 100:.1f}%"
        for feature, share in coverage["imputed_share_by_feature"].items()
    )
    return f"""
        <div class="status-meta"><strong>Evaluation:</strong> {method}, {n_folds} folds; {n_validation_rows:,} validation rows across folds; evaluated {evaluated_at}</div>
        <div class="status-meta"><strong>Metrics:</strong> {metrics}</div>
        <div class="status-meta"><strong>Auxiliary coverage:</strong> {matched_rows:,}/{denominator:,} rows ({matched_share:.1f}% matched); imputed shares: {imputed_shares}</div>
    """

st.markdown(
    """
    <style>
    :root {
        --navy: #1A365D;
        --cerulean: #2B6CB0;
        --snow: #F7FAFC;
        --slate: #1A202C;
        --white: #FFFFFF;
        --warning: #FFCC00;
        --border: rgba(43, 108, 176, 0.16);
    }
    html, body, .stApp {
        background: var(--snow);
        color: var(--slate);
        font-family: "Segoe UI", Arial, sans-serif;
    }
    #MainMenu, footer, header {
        visibility: hidden;
    }
    .block-container {
        max-width: 1280px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
    }
    .topbar {
        background: var(--navy);
        color: var(--white);
        border-radius: 18px;
        padding: 0.9rem 1.15rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 12px 28px rgba(26, 54, 93, 0.16);
    }
    .topbar-title {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .topbar-nav {
        display: flex;
        gap: 0.45rem;
        flex-wrap: wrap;
    }
    .topbar-pill {
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 999px;
        padding: 0.3rem 0.72rem;
        font-size: 0.84rem;
    }
    .hero {
        background: linear-gradient(135deg, var(--navy), #264975);
        color: var(--white);
        border-radius: 24px;
        padding: 1.85rem 1.95rem;
        margin-bottom: 1rem;
        box-shadow: 0 16px 36px rgba(26, 54, 93, 0.16);
    }
    .hero h1 {
        margin: 0 0 0.45rem 0;
        font-size: 2.1rem;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0;
        line-height: 1.75;
        max-width: 64rem;
    }
    .panel {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.2rem 1.25rem;
        box-shadow: 0 12px 28px rgba(26, 54, 93, 0.06);
    }
    .panel p {
        line-height: 1.75;
    }
    .status-card {
        display: flex;
        gap: 0.8rem;
        align-items: flex-start;
        padding: 0.9rem;
        border-radius: 16px;
        margin-bottom: 0.75rem;
        border: 1px solid transparent;
    }
    .status-card.ok {
        background: rgba(43, 108, 176, 0.06);
        border-color: rgba(43, 108, 176, 0.16);
    }
    .status-card.warn {
        background: rgba(255, 204, 0, 0.12);
        border-color: rgba(255, 204, 0, 0.35);
    }
    .status-icon {
        width: 2rem;
        height: 2rem;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--white);
        font-size: 0.76rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .status-icon.ok {
        background: var(--cerulean);
    }
    .status-icon.warn {
        background: #C28A00;
    }
    .status-title {
        color: var(--navy);
        font-weight: 700;
        margin-bottom: 0.15rem;
    }
    .status-text,
    .status-meta {
        color: var(--slate);
        font-size: 0.94rem;
        line-height: 1.6;
    }
    .footer-note {
        margin-top: 1rem;
        background: rgba(255, 255, 255, 0.92);
        border-left: 4px solid var(--cerulean);
        border-radius: 14px;
        padding: 0.95rem 1rem;
        line-height: 1.7;
    }
    div[data-baseweb="tab-list"] {
        gap: 0.5rem;
        margin-bottom: 0.6rem;
    }
    button[data-baseweb="tab"] {
        background: rgba(43, 108, 176, 0.08);
        border: 1px solid rgba(43, 108, 176, 0.16);
        border-radius: 999px;
        color: var(--navy);
        padding: 0.32rem 0.85rem;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: var(--navy);
        color: var(--white);
        border-color: var(--navy);
    }
    .stPageLink a {
        background: var(--navy);
        color: var(--white) !important;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        display: inline-block;
        text-decoration: none;
        font-weight: 600;
        border: none;
        width: 100%;
        text-align: center;
    }
    .stButton button {
        background: var(--navy) !important;
        color: var(--white) !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.7rem 1.1rem !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="topbar">
        <div class="topbar-title">Coral Bleaching AI Studio</div>
        <div class="topbar-nav">
            <div class="topbar-pill">Dashboard</div>
            <div class="topbar-pill">Prediction</div>
            <div class="topbar-pill">System Logs</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <section class="hero">
        <h1>Coral Bleaching AI Studio</h1>
        <p>
            This application presents two complementary models for coral bleaching severity estimates. Review model
            availability, the training-data status, and evaluation evidence before using the prediction workflow.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


main_col, side_col = st.columns([1.8, 1.0], gap="large")

with main_col:
    tab_stacking, tab_mlp = st.tabs(["Hybrid Stacking Ensemble", "Custom Residual MLP"])

    with tab_stacking:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Hybrid Stacking Ensemble")
        st.write(
            "This Stacking Ensemble combines structured machine-learning models and passes their predictions into a "
            "meta-learner to estimate bleaching severity and associated risk."
        )
        st.write(
            "Its estimates are based on the feature schema and training data recorded in the model metadata."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_mlp:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Custom Residual MLP")
        st.write(
            "The platform also employs a Custom Residual Multi-Layer Perceptron for deep feature analysis. This neural "
            "architecture is designed to capture complex non-linear relationships in processed coral bleaching features "
            "through residual connections and attention-based feature weighting."
        )
        st.write(
            "Its evaluation metrics and validation setup are shown with the model's training-data status."
        )
        st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### System Model Dashboard")
    st.markdown(
        model_status_html(
            STACKING_ARTIFACT_PATH,
            "Core Stacking Ensemble",
            "Ensemble model for severity and risk estimates from processed environmental features.",
            "Check the ensemble training notebook and re-run the final export cells.",
            stacking_data_status,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        model_status_html(
            MLP_ARTIFACT_PATH,
            "Custom Residual MLP",
            "Neural model for severity estimates from processed environmental features.",
            "Check the neural training notebook and re-run the artifact export step.",
            mlp_data_status,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="status-card {'ok' if dataset_status.status == FRESH else 'warn'}">
            <div class="status-icon {'ok' if dataset_status.status == FRESH else 'warn'}">{'OK' if dataset_status.status == FRESH else '!'}</div>
            <div class="status-body">
                <div class="status-title">Model-Ready Dataset</div>
                <div class="status-meta"><strong>Status:</strong> {html.escape(dataset_status.status)}</div>
                <div class="status-meta">{html.escape(dataset_status.message)}</div>
                <div class="status-meta"><strong>Path:</strong> {html.escape(DATA_PATH.name)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


action_col, _ = st.columns([0.5, 1.5], gap="large")
with action_col:
    st.page_link(
        "pages/1_Coral_Bleaching_AI_Studio.py",
        label="Go To AI Prediction",
        icon=":material/arrow_forward:",
    )


st.markdown(
    """
    <div class="footer-note">
        This dashboard provides an overview of both prediction models and their current operational status.
        Select <strong>Go To AI Prediction</strong> to move directly into the prediction interface.
    </div>
    """,
    unsafe_allow_html=True,
)
