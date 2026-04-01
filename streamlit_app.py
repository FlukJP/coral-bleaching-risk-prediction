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
            <div class="status-icon ok">✓</div>
            <div class="status-body">
                <div class="status-title">{title}</div>
                <div class="status-text">{description}</div>
                <div class="status-meta"><strong>Status:</strong> <span style="color:#2B6CB0; font-weight:600;">Online</span> | <strong>File size:</strong> {format_size_mb(path)}</div>
            </div>
        </div>
        """
    return f"""
    <div class="status-card warn">
        <div class="status-icon warn">!</div>
        <div class="status-body">
            <div class="status-title">{title}</div>
            <div class="status-text">{description}</div>
            <div class="status-meta" style="color:#C28A00;"><strong>Status:</strong> Re-training required</div>
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
        --snow: #F7FAFC; /* พื้นหลังสว่างสบายตา */
        --slate: #1A202C; /* สีตัวอักษรหลัก อ่านง่าย */
        --white: #FFFFFF;
        --warning: #FFCC00;
        --border: rgba(43, 108, 176, 0.16);
    }
    html, body, .stApp {
        background-color: var(--snow) !important;
        color: var(--slate);
        font-family: "Segoe UI", Arial, sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--navy) !important;
        font-weight: 700;
    }
    p, li, span, div {
        color: var(--slate);
        line-height: 1.6;
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
        border-radius: 12px;
        padding: 0.9rem 1.15rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 12px rgba(26, 54, 93, 0.1);
    }
    .topbar-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--white);
        letter-spacing: 0.5px;
    }
    .topbar-nav {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .topbar-pill {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 6px;
        padding: 0.3rem 0.8rem;
        font-size: 0.85rem;
        color: var(--white);
    }
    .hero {
        background: var(--navy);
        color: var(--white);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(26, 54, 93, 0.15);
    }
    .hero h1 {
        color: var(--white) !important;
        margin: 0 0 0.5rem 0;
        font-size: 2.2rem;
    }
    .hero p {
        color: rgba(255, 255, 255, 0.9) !important;
        margin: 0;
        font-size: 1.05rem;
        max-width: 65rem;
    }
    .panel {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(26, 54, 93, 0.04);
        height: 100%;
    }
    .panel p {
        font-size: 1rem;
        text-align: justify;
    }
    .status-card {
        display: flex;
        gap: 1rem;
        align-items: flex-start;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid transparent;
        background: var(--white);
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    .status-card.ok {
        border-left: 4px solid var(--cerulean);
        border-color: rgba(43, 108, 176, 0.2);
    }
    .status-card.warn {
        border-left: 4px solid #F59E0B;
        border-color: rgba(245, 158, 11, 0.2);
    }
    .status-icon {
        width: 2.2rem;
        height: 2.2rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--white);
        font-size: 1rem;
        font-weight: bold;
        flex-shrink: 0;
    }
    .status-icon.ok {
        background: var(--cerulean);
    }
    .status-icon.warn {
        background: #F59E0B;
    }
    .status-title {
        color: var(--navy);
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.2rem;
    }
    .status-text {
        color: var(--slate);
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }
    .status-meta {
        font-size: 0.85rem;
        color: #4A5568;
    }
    .footer-note {
        margin-top: 1.5rem;
        background: var(--white);
        border-left: 4px solid var(--cerulean);
        border-radius: 8px;
        padding: 1rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    /* Tabs Styling */
    div[data-baseweb="tab-list"] {
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    button[data-baseweb="tab"] {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--navy);
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    button[data-baseweb="tab"]:hover {
        border-color: var(--cerulean);
        background: rgba(43, 108, 176, 0.05);
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: var(--navy);
        color: var(--white) !important;
        border-color: var(--navy);
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: var(--white) !important;
    }
    .stPageLink a {
        background: var(--cerulean);
        color: var(--white) !important;
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        display: inline-flex;
        align-items: center;
        text-decoration: none;
        font-weight: 600;
        border: none;
        transition: background 0.2s ease;
    }
    .stPageLink a:hover {
        background: var(--navy);
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
            This advanced application presents two complementary artificial intelligence models for high-accuracy coral bleaching risk prediction. Analyze processed environmental data, submit findings, and compare combined insights below.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


main_col, side_col = st.columns([1.8, 1.0], gap="large")

with main_col:
    tab_stacking, tab_mlp = st.tabs(["📊 Hybrid Stacking Ensemble", "🧠 Custom Residual MLP"])

    with tab_stacking:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Hybrid Stacking Ensemble")
        st.write(
            "Our first approach utilizes a sophisticated Stacking Ensemble. This multi-level architecture strategically "
            "combines multiple diverse machine learning base models (e.g., Random Forests, Gradient Boosting Machines) "
            "that excel at interpreting structured data. Their base-level predictions are then fed into a higher-level "
            "'meta-learner' model. This second stage effectively learns how to best weigh and combine these predictions, "
            "significantly reducing overall variance and overcoming individual model biases."
        )
        st.write(
            "The result is a robust prediction pipeline that leverages the complementary strengths of different algorithms, "
            "achieving a higher level of accuracy and providing stable, operational workflows for complex ecological risks "
            "like coral bleaching."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_mlp:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Custom Residual MLP")
        st.write(
            "The platform also employs a Custom Residual Multi-Layer Perceptron (MLP) for deep feature analysis. "
            "While tree-based models excel at tabular data, neural networks are superior at capturing intricate, "
            "high-order, non-linear relationships. Our custom-designed architecture goes beyond standard MLPs by "
            "incorporating 'residual connections' (skip connections), inspired by advanced image processing networks."
        )
        st.write(
            "These connections prevent information loss during deep network passes and mitigate the 'vanishing gradient' "
            "problem, allowing the model to learn complex, subtle ecological patterns and interactions that might be missed "
            "by other methods, providing a highly complementary perspective on bleaching risk."
        )
        st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    st.markdown('<div class="panel" style="padding: 1.2rem;">', unsafe_allow_html=True)
    st.markdown("### System Model Dashboard")
    st.markdown("<hr style='margin: 0.5rem 0 1rem 0; border-color: rgba(43, 108, 176, 0.1);'>", unsafe_allow_html=True)
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
    
    # Data Status
    data_icon_class = "ok" if DATA_PATH.exists() else "warn"
    data_icon_char = "✓" if DATA_PATH.exists() else "!"
    data_color = "#2B6CB0" if DATA_PATH.exists() else "#C28A00"
    st.markdown(
        f"""
        <div class="status-card {data_icon_class}">
            <div class="status-icon {data_icon_class}">{data_icon_char}</div>
            <div class="status-body">
                <div class="status-title">Global Environmental Data</div>
                <div class="status-meta"><strong>Status:</strong> <span style="color:{data_color}; font-weight:600;">{data_status}</span></div>
                <div class="status-meta"><strong>Source:</strong> {DATA_PATH.name}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)
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
        <strong>Note:</strong> This dashboard provides an overview of both prediction models and their current operational status.
        Select <span style="color:#2B6CB0; font-weight:600;">Go To AI Prediction</span> to move directly into the prediction interface.
    </div>
    """,
    unsafe_allow_html=True,
)