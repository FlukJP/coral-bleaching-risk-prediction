from __future__ import annotations

import streamlit as st

from model_inference import MLP_ARTIFACT_PATH, STACKING_ARTIFACT_PATH


st.set_page_config(
    page_title="Coral Bleaching App",
    layout="wide",
)

st.title("Coral Bleaching Prediction")
st.write(
    "Use the page menu on the left to open `Coral Bleaching AI Studio`, "
    "where both the Stacking Ensemble and Custom Residual MLP are available."
)

stacking_ready = STACKING_ARTIFACT_PATH.exists()
mlp_ready = MLP_ARTIFACT_PATH.exists()

status_col1, status_col2 = st.columns(2)
with status_col1:
    st.metric("Stacking Artifact", "Ready" if stacking_ready else "Missing")
with status_col2:
    st.metric("MLP Artifact", "Ready" if mlp_ready else "Missing")

st.info(
    "Run this app with `python -m streamlit run streamlit_app.py` "
    "and then open the page in the sidebar."
)
