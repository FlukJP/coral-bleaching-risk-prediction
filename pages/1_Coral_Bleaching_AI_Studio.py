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
        --snow: #F7FAFC;
        --slate: #1A202C;
        --white: #FFFFFF;
    }
    .stApp {
        background: var(--snow);
        color: var(--slate);
    }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 22px;
        background: linear-gradient(135deg, var(--navy), #234975);
        color: var(--white);
        box-shadow: 0 18px 40px rgba(26, 54, 93, 0.14);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2.15rem;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0;
        max-width: 56rem;
        line-height: 1.7;
        opacity: 0.97;
    }
    .panel {
        background: var(--white);
        border: 1px solid rgba(43, 108, 176, 0.14);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: 0 12px 30px rgba(26, 54, 93, 0.06);
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
        background: rgba(43, 108, 176, 0.10);
        color: var(--cerulean);
    }
    .status-missing {
        background: rgba(220, 38, 38, 0.10);
        color: #B91C1C;
    }
    .note {
        background: rgba(255, 255, 255, 0.92);
        border-left: 4px solid var(--cerulean);
        padding: 0.9rem 1rem;
        border-radius: 14px;
        margin: 0.45rem 0 1rem 0;
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
        Hybrid model status: <strong>{flag}</strong><br>
        Two-model gap: <strong>{gap:+.2f}</strong> percentage points between hybrid and MLP predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <section class="hero">
        <h1>Coral Bleaching AI Studio</h1>
        <p>
            Provide the processed environmental values below, submit the form, and review the combined prediction
            output from both artificial intelligence models.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


left, right = st.columns([1.6, 1.0], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Prediction Input")
    st.caption("Enter processed feature values as text. Numeric values and boolean values are both supported.")
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

feature_columns = reference_features.columns.tolist()
feature_groups = build_feature_groups(feature_columns)
default_values = {column: default_value_text(reference_features[column]) for column in feature_columns}

st.markdown(
    f"""
    <div class="note">
    Required input fields: <strong>{len(feature_columns)}</strong>. The form is prefilled with representative values
    from the processed training data so the interface can be tested immediately.
    </div>
    """,
    unsafe_allow_html=True,
)

template_col, helper_col = st.columns([1.0, 1.6], gap="large")
with template_col:
    st.download_button(
        "Download input template",
        data=build_template_csv(reference_features.head(1)),
        file_name="coral_bleaching_input_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
with helper_col:
    st.caption("Use `True/False` or `1/0` for ocean, realm, exposure, substrate, and other boolean-style indicator fields.")


with st.form("prediction_form"):
    for section_name, section_fields in feature_groups:
        with st.expander(section_name, expanded=section_name in {"Location and Date", "Water and Temperature"}):
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

    run_prediction = st.form_submit_button("Predict Now", type="primary", use_container_width=True)


if run_prediction:
    input_payload = {column: st.session_state.get(f"field_{column}", "") for column in feature_columns}
    input_df = pd.DataFrame([input_payload])

    with st.spinner("Running both models..."):
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
            st.success("Prediction completed.")

            if fill_notes:
                with st.expander("Auto-filled values"):
                    for note in fill_notes:
                        st.write(f"- {note}")

            prediction_row = prediction_df.iloc[0]
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

            summary_left, summary_right = st.columns([1.3, 1.0], gap="large")

            with summary_left:
                st.subheader("Prediction Results")
                st.dataframe(
                    prediction_df[result_cols].round(4),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download prediction as CSV",
                    data=build_download_bytes(prediction_df),
                    file_name="coral_bleaching_prediction.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with summary_right:
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
