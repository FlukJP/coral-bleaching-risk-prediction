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


st.markdown(
    """
    <style>
    :root {
        --navy: #1A365D;
        --cerulean: #2B6CB0;
        --snow: #F7FAFC; /* พื้นหลังสว่างสบายตา */
        --slate: #1A202C; /* สีตัวอักษรหลัก อ่านง่าย */
        --white: #FFFFFF;
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
    }
    .topbar-crumb {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.8);
        background: rgba(255,255,255,0.15);
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
    }
    .hero {
        background: var(--navy);
        color: var(--white);
        border-radius: 16px;
        padding: 1.8rem 2rem;
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
        line-height: 1.6;
    }
    .panel {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(26, 54, 93, 0.04);
        margin-bottom: 1rem;
    }
    .panel h3 {
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .note {
        background: var(--white);
        border-left: 4px solid var(--cerulean);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        line-height: 1.6;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    .status-ok, .status-missing {
        display: inline-flex;
        align-items: center;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem 0.3rem 0.2rem 0;
    }
    .status-ok {
        background: rgba(43, 108, 176, 0.08);
        color: var(--cerulean);
        border: 1px solid rgba(43, 108, 176, 0.2);
    }
    .status-missing {
        background: rgba(245, 158, 11, 0.1);
        color: #B45309;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    details[data-testid="stExpander"] {
        background: var(--white) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        margin-bottom: 0.8rem !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
    }
    details[data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: var(--navy) !important;
        padding: 0.5rem !important;
    }
    .stTextInput > div > div > input {
        border-radius: 8px !important;
        border: 1px solid rgba(43, 108, 176, 0.3) !important;
        background: var(--white) !important;
        color: var(--slate) !important;
        padding: 0.6rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--cerulean) !important;
        box-shadow: 0 0 0 1px var(--cerulean) !important;
    }
    /* Buttons */
    .stFormSubmitButton > button {
        background: var(--navy) !important;
        color: var(--white) !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.2s;
    }
    .stFormSubmitButton > button:hover {
        background: var(--cerulean) !important;
    }
    .stDownloadButton > button {
        background: rgba(43, 108, 176, 0.05) !important;
        color: var(--cerulean) !important;
        border: 1px solid var(--cerulean) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(43, 108, 176, 0.1) !important;
    }
    .stPageLink a {
        background: var(--white);
        color: var(--cerulean) !important;
        border: 1px solid var(--cerulean);
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stPageLink a:hover {
        background: rgba(43, 108, 176, 0.05);
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
        return f'<span class="status-ok">✓ {path.name} | {size_mb:.1f} MB</span>'
    return f'<span class="status-missing">! {path.name} missing</span>'


def is_boolean_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return False
    normalized = {str(value).strip().lower() for value in non_null.unique()}
    allowed = {"1", "0", "true", "false", "t", "f", "yes", "no", "y", "n"}
    return normalized.issubset(allowed)


def default_value_text(series: pd.Series) -> str:
    if is_boolean_like(series):
        mode = series.mode(dropna=True)
        if mode.empty:
            return "False"
        return "True" if str(mode.iloc[0]).strip().lower() in {"1", "true", "t", "yes", "y"} else "False"
    if pd.api.types.is_numeric_dtype(series):
        return f"{float(series.median()):.4f}"
    mode = series.mode(dropna=True)
    return str(mode.iloc[0]) if not mode.empty else ""


def build_template_csv(df: pd.DataFrame) -> str:
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def build_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_feature_groups(columns: list[str]) -> list[tuple[str, list[str]]]:
    grouped = {
        "Location and Date": [],
        "Water and Temperature": [],
        "SSTA Features": [],
        "TSA Features": [],
        "Climate Flags": [],
        "Ocean and Realm": [],
        "Exposure and Substrate": [],
        "Other Features": [],
    }
    for column in columns:
        if column.startswith(("Latitude", "Longitude", "Distance_to_Shore", "Turbidity", "Cyclone_Frequency", "Date_", "Depth_")):
            grouped["Location and Date"].append(column)
        elif column.startswith(("ClimSST", "Temperature_", "Temp_Range_C", "Windspeed")):
            grouped["Water and Temperature"].append(column)
        elif column.startswith("SSTA"):
            grouped["SSTA Features"].append(column)
        elif column.startswith("TSA"):
            grouped["TSA Features"].append(column)
        elif column.startswith(("Is_Tropical", "Aux_")):
            grouped["Climate Flags"].append(column)
        elif column.startswith(("Ocean_Name_", "Realm_Name_")):
            grouped["Ocean and Realm"].append(column)
        elif column.startswith(("Exposure_", "Substrate_")):
            grouped["Exposure and Substrate"].append(column)
        else:
            grouped["Other Features"].append(column)
    return [(name, cols) for name, cols in grouped.items() if cols]


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
        <strong style="color:#1A365D;">Hybrid model status:</strong> <strong>{flag}</strong><br>
        <strong style="color:#1A365D;">Two-model gap:</strong> <strong>{gap:+.2f}</strong> percentage points between hybrid and MLP predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="topbar">
        <div class="topbar-title">Coral Bleaching AI Studio</div>
        <div class="topbar-crumb">Prediction Studio</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <h1>Prediction Studio</h1>
        <p>
            Enter processed environmental feature values below and submit the form to generate predictions from both
            the Hybrid Stacking Ensemble and the Custom Residual MLP models.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


back_col, _ = st.columns([0.45, 1.55], gap="large")
with back_col:
    st.page_link("streamlit_app.py", label="Back To Dashboard", icon=":material/arrow_back:")
st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1.6, 1.0], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Prediction Input")
    st.caption("Enter processed feature values as text. Numeric values and boolean values are both supported.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Model Status")
    st.markdown(artifact_status(STACKING_ARTIFACT_PATH), unsafe_allow_html=True)
    st.markdown(artifact_status(MLP_ARTIFACT_PATH), unsafe_allow_html=True)
    st.caption(f"Reference schema: `{DATA_PATH.name}`")
    st.markdown("</div>", unsafe_allow_html=True)


if not STACKING_ARTIFACT_PATH.exists() or not MLP_ARTIFACT_PATH.exists():
    st.error("At least one model artifact is missing. Re-run the training notebooks before using this web app.")
    st.stop()


reference_features = get_reference_features()
stacking_artifact = get_stacking_artifact()
mlp_artifact, mlp_model = get_mlp_bundle()

feature_columns = reference_features.columns.tolist()
feature_groups = build_feature_groups(feature_columns)
default_values = {column: default_value_text(reference_features[column]) for column in feature_columns}

st.markdown(
    f"""
    <div class="note">
    <strong>Required input fields: {len(feature_columns)}</strong><br>
    <span style="font-size: 0.95rem;">The form is prefilled with representative values from the processed training data so the interface can be tested immediately.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

template_col, helper_col = st.columns([1.0, 1.6], gap="large")
with template_col:
    st.download_button(
        "Download Input Template",
        data=build_template_csv(reference_features.head(1)),
        file_name="coral_bleaching_input_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
with helper_col:
    st.caption("💡 Tip: Use `True/False` or `1/0` for ocean, realm, exposure, substrate, and other boolean-style indicator fields.")
st.markdown("<br>", unsafe_allow_html=True)

with st.form("prediction_form"):
    for section_name, section_fields in feature_groups:
        with st.expander(f"📁 {section_name}", expanded=section_name in {"Location and Date", "Water and Temperature"}):
            col_left, col_right = st.columns(2, gap="large")
            for idx, field in enumerate(section_fields):
                target_col = col_left if idx % 2 == 0 else col_right
                help_text = "Enter True/False or 1/0" if is_boolean_like(reference_features[field]) else None
                with target_col:
                    st.text_input(
                        label=field,
                        value=default_values[field],
                        key=f"field_{field}",
                        help=help_text,
                    )
    st.markdown("<br>", unsafe_allow_html=True)
    run_prediction = st.form_submit_button("🚀 Run AI Prediction", type="primary", use_container_width=True)


if run_prediction:
    input_payload = {column: st.session_state.get(f"field_{column}", "") for column in feature_columns}
    input_df = pd.DataFrame([input_payload])

    with st.spinner("Analyzing environment and running models..."):
        try:
            prediction_df, fill_notes = predict_all_models(
                input_df,
                reference_features,
                stacking_artifact,
                mlp_artifact,
                mlp_model,
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success("✅ Prediction completed successfully.")

            if fill_notes:
                with st.expander("⚠️ Auto-filled missing or invalid values"):
                    for note in fill_notes:
                        st.write(f"- {note}")

            prediction_row = prediction_df.iloc[0]
            st.markdown("---")
            render_single_prediction(prediction_row)

            result_cols = [
                "stacking_prediction",
                "bleaching_risk_probability",
                "bleaching_risk_flag",
                "two_stage_prediction",
                "hybrid_prediction",
                "mlp_prediction",
                "hybrid_vs_mlp_gap",
            ]

            st.markdown("<br>", unsafe_allow_html=True)
            summary_left, summary_right = st.columns([1.3, 1.0], gap="large")

            with summary_left:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.subheader("Prediction Data")
                st.dataframe(
                    prediction_df[result_cols].round(4),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download Result as CSV",
                    data=build_download_bytes(prediction_df),
                    file_name="coral_bleaching_prediction.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with summary_right:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.subheader("Quick Summary")
                summary_frame = pd.DataFrame(
                    {
                        "Metric": [
                            "Hybrid prediction",
                            "Residual MLP",
                            "Risk probability",
                            "Risk flag",
                        ],
                        "Value": [
                            f"{prediction_row['hybrid_prediction']:.2f}%",
                            f"{prediction_row['mlp_prediction']:.2f}%",
                            f"{prediction_row['bleaching_risk_probability'] * 100:.2f}%",
                            "High risk" if int(prediction_row["bleaching_risk_flag"]) == 1 else "Below threshold",
                        ],
                    }
                )
                st.dataframe(summary_frame, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)