from __future__ import annotations

import streamlit as st

from model_inference import MLP_ARTIFACT_PATH, STACKING_ARTIFACT_PATH


st.set_page_config(
    page_title="Coral Bleaching Prediction",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --navy: #1A365D;
        --cerulean: #2B6CB0;
        --snow: #F7FAFC;
        --slate: #1A202C;
        --white: #FFFFFF;
    }
    .stApp {
        background: var(--snow);
        color: var(--slate);
    }
    .hero {
        padding: 1.9rem 2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, var(--navy), #234975);
        color: var(--white);
        box-shadow: 0 18px 40px rgba(26, 54, 93, 0.16);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0 0 0.45rem 0;
        font-size: 2.2rem;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0;
        max-width: 60rem;
        line-height: 1.7;
        opacity: 0.96;
    }
    .panel {
        background: var(--white);
        border: 1px solid rgba(43, 108, 176, 0.14);
        border-radius: 20px;
        padding: 1.15rem 1.2rem;
        box-shadow: 0 12px 30px rgba(26, 54, 93, 0.06);
    }
    .section-title {
        color: var(--navy);
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .status {
        display: inline-block;
        margin: 0.1rem 0.35rem 0.1rem 0;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.86rem;
        font-weight: 600;
    }
    .status.ready {
        background: rgba(43, 108, 176, 0.10);
        color: var(--cerulean);
    }
    .status.missing {
        background: rgba(220, 38, 38, 0.10);
        color: #B91C1C;
    }
    .note {
        background: rgba(255, 255, 255, 0.88);
        border-left: 4px solid var(--cerulean);
        border-radius: 14px;
        padding: 0.95rem 1rem;
        margin-top: 0.8rem;
        line-height: 1.65;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def artifact_badge(path) -> str:
    if path.exists():
        return f'<span class="status ready">{path.name} ready</span>'
    return f'<span class="status missing">{path.name} missing</span>'


st.markdown(
    """
    <section class="hero">
        <h1>Coral Bleaching Prediction Platform</h1>
        <p>
            This application presents two complementary artificial intelligence models for coral bleaching risk
            prediction. The first model is a hybrid Stacking Ensemble designed to combine structured tree-based
            predictions with a two-stage risk pipeline. The second model is a Custom Residual MLP designed to
            capture non-linear tabular patterns through a neural architecture. Please use the page menu on the
            left to open the prediction interface after reviewing the model summaries below.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


overview_col, status_col = st.columns([1.6, 1.0], gap="large")

with overview_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Platform Overview</div>', unsafe_allow_html=True)
    st.write(
        "The prediction workflow is intended for processed environmental and reef-condition features. "
        "Both models consume the same prepared schema and return bleaching severity estimates, allowing "
        "users to compare a hybrid ensemble approach with a neural residual approach in a consistent setting."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with status_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Availability</div>', unsafe_allow_html=True)
    st.markdown(artifact_badge(STACKING_ARTIFACT_PATH), unsafe_allow_html=True)
    st.markdown(artifact_badge(MLP_ARTIFACT_PATH), unsafe_allow_html=True)
    st.caption("If a model is marked missing, the relevant training notebook must be executed again.")
    st.markdown("</div>", unsafe_allow_html=True)


tab_stacking, tab_mlp = st.tabs(["Hybrid Stacking Ensemble", "Custom Residual MLP"])

with tab_stacking:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Model Role")
    st.write(
        "The Hybrid Stacking Ensemble is the primary structured model for operational prediction. "
        "It blends a stacking regressor with a two-stage classifier and regressor so that the system "
        "can separately evaluate bleaching likelihood and bleaching severity."
    )
    st.subheader("How It Works")
    st.write(
        "The model first estimates a direct severity signal from the stacking regressor. In parallel, "
        "the two-stage branch estimates bleaching probability and, when the probability exceeds the tuned "
        "threshold, estimates the likely severity among positive-risk cases. These outputs are then combined "
        "through a tuned blend weight to produce the final hybrid prediction."
    )
    st.subheader("Why It Is Useful")
    st.write(
        "This design is suitable when interpretability, structured benchmarking, and balanced performance on "
        "zero-heavy bleaching data are all important. It is especially useful when users need both a final "
        "severity estimate and an intermediate risk probability."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_mlp:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Model Role")
    st.write(
        "The Custom Residual MLP is the neural comparison model in this platform. It applies a residual network "
        "with tabular attention to the same processed feature space and is intended to capture complex interactions "
        "that may not be expressed as strongly in the tree-based ensemble."
    )
    st.subheader("How It Works")
    st.write(
        "The model receives scaled tabular inputs, applies an attention gate to weight feature contributions, "
        "passes the representation through residual blocks, and outputs a log-scale bleaching estimate that is "
        "converted back to the percentage scale after inference."
    )
    st.subheader("Why It Is Useful")
    st.write(
        "This model is useful as a second opinion from a neural architecture. It helps users compare whether "
        "a residual deep learning approach agrees with or diverges from the hybrid ensemble under the same inputs."
    )
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="note">
    Open <strong>Coral Bleaching AI Studio</strong> from the sidebar to enter feature values and run prediction.
    The prediction page will display hybrid ensemble output, residual MLP output, bleaching risk probability,
    and the final comparison summary.
    </div>
    """,
    unsafe_allow_html=True,
)
