from __future__ import annotations

from pathlib import Path

import streamlit as st

from model_inference import DATA_PATH, MLP_ARTIFACT_PATH, STACKING_ARTIFACT_PATH


st.set_page_config(
    page_title="Coral Bleaching AI Studio",
    layout="wide",
)


def format_size_mb(path: Path) -> str:
    if not path.exists():
        return "-"
    return f"{path.stat().st_size / (1024 * 1024):.1f} MB"


def model_status_html(path: Path, title: str, description: str, missing_action: str) -> str:
    if path.exists():
        return f"""
        <div class="status-card ok">
            <div class="status-icon ok">OK</div>
            <div class="status-body">
                <div class="status-title">{title}</div>
                <div class="status-text">{description}</div>
                <div class="status-meta"><strong>Status:</strong> Online | <strong>File size:</strong> {format_size_mb(path)}</div>
            </div>
        </div>
        """
    return f"""
    <div class="status-card warn">
        <div class="status-icon warn">!</div>
        <div class="status-body">
            <div class="status-title">{title}</div>
            <div class="status-text">{description}</div>
            <div class="status-meta"><strong>Status:</strong> Re-training required</div>
            <div class="status-meta"><strong>Artifact:</strong> {path.name} missing</div>
            <div class="status-meta"><strong>Action:</strong> {missing_action}</div>
        </div>
    </div>
    """


data_status = "Fresh" if DATA_PATH.exists() else "Unavailable"

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
            This advanced application presents two complementary artificial intelligence models for high-accuracy coral
            bleaching risk prediction. Analyze processed environmental data, review model availability, and move directly
            into the prediction workflow from this dashboard.
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
            "Our first approach utilizes a sophisticated Stacking Ensemble. This multi-level architecture strategically "
            "combines multiple structured machine learning models and passes their predictions into a meta-learner. "
            "The final system is designed to reduce variance, balance model bias, and deliver strong performance for "
            "complex ecological risk estimation."
        )
        st.write(
            "This approach is particularly suitable when the platform needs both a severity estimate and a robust, "
            "operational prediction workflow that remains stable on processed environmental features."
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
            "The model complements the ensemble pipeline by providing a neural perspective on the same input space, "
            "helping users compare how tree-based and residual deep learning approaches behave under identical conditions."
        )
        st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### System Model Dashboard")
    st.markdown(
        model_status_html(
            STACKING_ARTIFACT_PATH,
            "Core Stacking Ensemble",
            "Primary production model for structured bleaching risk prediction.",
            "Check the ensemble training notebook and re-run the final export cells.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        model_status_html(
            MLP_ARTIFACT_PATH,
            "Custom Residual MLP",
            "Neural comparison model for deep feature interaction analysis.",
            "Check the neural training notebook and re-run the artifact export step.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="status-card {'ok' if DATA_PATH.exists() else 'warn'}">
            <div class="status-icon {'ok' if DATA_PATH.exists() else 'warn'}">{'OK' if DATA_PATH.exists() else '!'}</div>
            <div class="status-body">
                <div class="status-title">Global Environmental Data Source</div>
                <div class="status-meta"><strong>Status:</strong> {data_status}</div>
                <div class="status-meta"><strong>Path:</strong> {DATA_PATH.name}</div>
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
