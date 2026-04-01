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

# ── Page config (must be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="Prediction Studio · Coral Bleaching AI",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Geist:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
:root {
    --abyss:    #070E1A;
    --deep:     #0B1929;
    --ocean:    #0E2A45;
    --cerulean: #1565C0;
    --sky:      #1E88E5;
    --bio:      #00E5C3;
    --bio-dim:  #00B39A;
    --coral:    #FF6B6B;
    --amber:    #FFB347;
    --snow:     #EFF6FF;
    --mist:     #A8C4E0;
    --border:   rgba(0,229,195,0.14);
    --border-s: rgba(0,229,195,0.28);
    --glass:    rgba(255,255,255,0.04);
}

html, body, .stApp {
    background: var(--abyss) !important;
    color: var(--snow) !important;
    font-family: 'Geist', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 4rem !important; max-width: 1340px !important; }

/* ── Background mesh ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 55% 45% at 15% 25%, rgba(0,229,195,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 45% 55% at 85% 75%, rgba(21,101,192,0.09) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* ── Topbar ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.7rem 1.4rem;
    background: rgba(11,25,41,0.80);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 16px;
    margin-bottom: 1.6rem;
}
.topbar-brand {
    display: flex; align-items: center; gap: 0.55rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700; font-size: 0.95rem;
    color: var(--snow); letter-spacing: -0.01em;
}
.brand-dot {
    width: 7px; height: 7px;
    background: var(--bio); border-radius: 50%;
    box-shadow: 0 0 8px var(--bio);
    animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; box-shadow: 0 0 8px var(--bio); }
    50%      { opacity:.5; box-shadow: 0 0 18px var(--bio); }
}
.topbar-crumb {
    font-size: 0.82rem; color: var(--mist);
    display: flex; align-items: center; gap: 0.5rem;
}
.crumb-sep { opacity: 0.4; }
.crumb-active { color: var(--bio); font-weight: 600; }

/* ── Page header ── */
.page-header {
    margin-bottom: 1.6rem;
    padding: 1.6rem 2rem;
    background: linear-gradient(120deg, rgba(11,25,41,0.95) 0%, rgba(7,14,26,0.90) 100%);
    border: 1px solid var(--border-s);
    border-radius: 22px;
    position: relative; overflow: hidden;
}
.page-header::after {
    content: '';
    position: absolute; right: -5%; top: -40%;
    width: 40%; height: 180%;
    background: radial-gradient(ellipse, rgba(0,229,195,0.09) 0%, transparent 65%);
    pointer-events: none;
}
.page-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--bio-dim);
    margin-bottom: 0.5rem;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--snow); margin: 0 0 0.5rem 0;
}
.page-sub {
    font-size: 0.92rem; line-height: 1.7;
    color: var(--mist); margin: 0;
}

/* ── Glass panel ── */
.gpanel {
    background: rgba(11,25,41,0.65);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 16px 48px rgba(0,0,0,0.28);
    margin-bottom: 1rem;
}
.gpanel-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.96rem; font-weight: 700;
    color: var(--snow); margin: 0 0 0.4rem 0;
}
.gpanel-sub {
    font-size: 0.82rem; color: var(--mist);
    line-height: 1.6; margin: 0;
}

/* ── Status pills ── */
.status-wrap { display: flex; flex-direction: column; gap: 0.5rem; }
.status-pill {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.55rem 0.85rem;
    border-radius: 12px; border: 1px solid transparent;
    font-size: 0.82rem;
}
.s-ok  { background: rgba(0,229,195,0.07); border-color: rgba(0,229,195,0.20); }
.s-err { background: rgba(255,107,107,0.08); border-color: rgba(255,107,107,0.22); }
.s-dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
.s-dot-ok  { background: var(--bio); box-shadow: 0 0 6px var(--bio); }
.s-dot-err { background: var(--coral); box-shadow: 0 0 6px var(--coral); }
.s-name { font-weight: 600; color: var(--snow); }
.s-meta { color: var(--mist); margin-left: auto; font-family: monospace; font-size: 0.75rem; }

/* ── Section heading ── */
.sec-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.11em;
    text-transform: uppercase; color: var(--bio-dim);
    margin: 0 0 0.8rem 0;
}

/* ── Feature group header ── */
.group-hdr {
    font-family: 'Syne', sans-serif;
    font-size: 0.84rem; font-weight: 700;
    color: var(--bio); letter-spacing: 0.04em;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(0,229,195,0.15);
    margin-bottom: 0.5rem;
}

/* ── Info note ── */
.info-note {
    background: rgba(0,229,195,0.06);
    border: 1px solid rgba(0,229,195,0.16);
    border-left: 3px solid var(--bio-dim);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    font-size: 0.84rem;
    line-height: 1.65;
    color: var(--mist);
    margin: 0.6rem 0 1rem 0;
}

/* ── Result cards ── */
.result-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin-bottom: 1.2rem; }
.result-card {
    background: rgba(11,25,41,0.80);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1rem 1.1rem;
}
.result-card.primary { border-color: var(--border-s); background: rgba(0,229,195,0.07); }
.rc-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: var(--mist); margin-bottom: 0.35rem; }
.rc-value { font-family: 'Syne', sans-serif; font-size: 1.85rem; font-weight: 800; color: var(--snow); line-height: 1; margin-bottom: 0.2rem; }
.rc-value.big { font-size: 2.1rem; color: var(--bio); }
.rc-sub { font-size: 0.75rem; color: rgba(168,196,224,0.55); }

/* ── Gap note ── */
.gap-note {
    background: rgba(30,136,229,0.08);
    border: 1px solid rgba(30,136,229,0.20);
    border-radius: 14px;
    padding: 0.85rem 1rem;
    font-size: 0.85rem;
    line-height: 1.65;
    color: var(--mist);
    margin-bottom: 1rem;
}
.gap-note strong { color: var(--sky); }

/* ── Streamlit overrides ── */
.stTextInput > div > div > input {
    background: rgba(11,25,41,0.80) !important;
    border: 1px solid rgba(0,229,195,0.16) !important;
    border-radius: 10px !important;
    color: var(--snow) !important;
    font-family: 'Geist', sans-serif !important;
    font-size: 0.87rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--bio-dim) !important;
    box-shadow: 0 0 0 2px rgba(0,229,195,0.12) !important;
}
.stTextInput label {
    font-size: 0.8rem !important;
    color: var(--mist) !important;
    font-weight: 500 !important;
    font-family: 'Geist', sans-serif !important;
}

/* Expander */
details[data-testid="stExpander"] {
    background: rgba(11,25,41,0.50) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    margin-bottom: 0.6rem !important;
}
details[data-testid="stExpander"] summary {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: var(--snow) !important;
    padding: 0.6rem 0.9rem !important;
}

/* Form submit button */
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #0E5C2E, #1D9E55) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.75rem 2.2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.96rem !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 8px 26px rgba(0,155,70,0.35) !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stFormSubmitButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 32px rgba(0,155,70,0.48) !important;
}

/* Download button */
.stDownloadButton > button {
    background: rgba(30,136,229,0.14) !important;
    color: var(--sky) !important;
    border: 1px solid rgba(30,136,229,0.28) !important;
    border-radius: 10px !important;
    font-family: 'Geist', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}

/* Dataframe */
.stDataFrame { border-radius: 14px; overflow: hidden; }
[data-testid="stDataFrameResizable"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    background: rgba(11,25,41,0.70) !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    color: var(--snow) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Geist', sans-serif !important;
    color: var(--mist) !important;
    font-size: 0.8rem !important;
}

/* Spinner */
[data-testid="stSpinner"] p { color: var(--mist) !important; }

/* Alert */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    font-family: 'Geist', sans-serif !important;
    font-size: 0.88rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data
def get_reference_features() -> pd.DataFrame:
    return load_reference_features()


@st.cache_resource
def get_stacking_artifact():
    return load_stacking_artifact()


@st.cache_resource
def get_mlp_bundle():
    return load_mlp_artifact()


# ── Utilities ─────────────────────────────────────────────────────────────────
def is_boolean_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty:
        return False
    normalized = {str(v).strip().lower() for v in non_null.unique()}
    return normalized.issubset({"1", "0", "true", "false", "t", "f", "yes", "no", "y", "n"})


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
    return df.to_csv(index=False)


def build_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_feature_groups(columns: list[str]) -> list[tuple[str, list[str]]]:
    grouped = {
        "Location & Date": [],
        "Water & Temperature": [],
        "SSTA Features": [],
        "TSA Features": [],
        "Climate Flags": [],
        "Ocean & Realm": [],
        "Exposure & Substrate": [],
        "Other Features": [],
    }
    for col in columns:
        if col.startswith(("Latitude", "Longitude", "Distance_to_Shore", "Turbidity", "Cyclone_Frequency", "Date_", "Depth_")):
            grouped["Location & Date"].append(col)
        elif col.startswith(("ClimSST", "Temperature_", "Temp_", "Windspeed")):
            grouped["Water & Temperature"].append(col)
        elif col.startswith("SSTA"):
            grouped["SSTA Features"].append(col)
        elif col.startswith("TSA"):
            grouped["TSA Features"].append(col)
        elif col.startswith(("Is_Tropical", "Aux_")):
            grouped["Climate Flags"].append(col)
        elif col.startswith(("Ocean_Name_", "Realm_Name_")):
            grouped["Ocean & Realm"].append(col)
        elif col.startswith(("Exposure_", "Substrate_")):
            grouped["Exposure & Substrate"].append(col)
        else:
            grouped["Other Features"].append(col)
    return [(k, v) for k, v in grouped.items() if v]


def risk_color(pct: float) -> str:
    if pct < 5:   return "#00E5C3"
    if pct < 20:  return "#FFB347"
    if pct < 50:  return "#FF6B6B"
    return               "#CF6CE1"


def risk_label(pct: float) -> str:
    if pct < 5:   return "Low Risk"
    if pct < 20:  return "Moderate"
    if pct < 50:  return "High Risk"
    return               "Severe"


def render_result_cards(row: pd.Series):
    hybrid = row["hybrid_prediction"]
    mlp    = row["mlp_prediction"]
    stack  = row["stacking_prediction"]
    proba  = row["bleaching_risk_probability"] * 100
    flag   = int(row["bleaching_risk_flag"])
    gap    = row["hybrid_vs_mlp_gap"]
    color  = risk_color(hybrid)
    rlabel = risk_label(hybrid)

    st.markdown(f"""
    <div class="result-grid">
        <div class="result-card primary">
            <div class="rc-label">Hybrid Prediction</div>
            <div class="rc-value big" style="color:{color}">{hybrid:.2f}%</div>
            <div class="rc-sub">{rlabel}</div>
        </div>
        <div class="result-card">
            <div class="rc-label">Stacking Model</div>
            <div class="rc-value">{stack:.2f}%</div>
            <div class="rc-sub">Ensemble output</div>
        </div>
        <div class="result-card">
            <div class="rc-label">Residual MLP</div>
            <div class="rc-value">{mlp:.2f}%</div>
            <div class="rc-sub">Neural network</div>
        </div>
        <div class="result-card">
            <div class="rc-label">Risk Probability</div>
            <div class="rc-value">{proba:.1f}%</div>
            <div class="rc-sub">{'⚠ High risk flag' if flag == 1 else 'Below threshold'}</div>
        </div>
    </div>
    <div class="gap-note">
        Model status: <strong>{'High risk — bleaching detected' if flag == 1 else 'Below threshold'}</strong>
        &nbsp;·&nbsp;
        Two-model gap: <strong>{gap:+.2f} pp</strong> between hybrid and MLP predictions
    </div>
    """, unsafe_allow_html=True)


# ── Topbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-brand">
        <div class="brand-dot"></div>
        Coral Bleaching AI Studio
    </div>
    <div class="topbar-crumb">
        <span>Dashboard</span>
        <span class="crumb-sep">›</span>
        <span class="crumb-active">Prediction Studio</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-label">🪸 Prediction Studio</div>
    <h1 class="page-title">Enter Environmental Features</h1>
    <p class="page-sub">
        Fill in the processed feature values, submit the form, and compare the combined AI prediction output
        from both the Hybrid Stacking Ensemble and the Custom Residual MLP.
    </p>
</div>
""", unsafe_allow_html=True)


# ── Header columns: info + model status ──────────────────────────────────────
left_hdr, right_hdr = st.columns([1.65, 1.0], gap="large")

with left_hdr:
    st.markdown('<div class="gpanel">', unsafe_allow_html=True)
    st.markdown('<div class="gpanel-title">Prediction Input</div>', unsafe_allow_html=True)
    st.markdown('<p class="gpanel-sub">Enter processed feature values as text. Both numeric and boolean (True/False, 1/0) values are supported. Fields are prefilled with training set medians.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_hdr:
    st.markdown('<div class="gpanel">', unsafe_allow_html=True)
    st.markdown('<div class="gpanel-title">Model Status</div>', unsafe_allow_html=True)

    def status_pill(path, name):
        ok = path.exists()
        dot_cls = "s-dot-ok" if ok else "s-dot-err"
        pill_cls = "s-ok" if ok else "s-err"
        size = f"{path.stat().st_size/(1024*1024):.1f} MB" if ok else "missing"
        return f"""<div class="status-pill {pill_cls}">
            <span class="s-dot {dot_cls}"></span>
            <span class="s-name">{name}</span>
            <span class="s-meta">{size}</span>
        </div>"""

    st.markdown(f"""
    <div class="status-wrap">
        {status_pill(STACKING_ARTIFACT_PATH, "Stacking Ensemble")}
        {status_pill(MLP_ARTIFACT_PATH, "Residual MLP")}
        {status_pill(DATA_PATH, "Reference Data")}
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Guard ─────────────────────────────────────────────────────────────────────
if not STACKING_ARTIFACT_PATH.exists() or not MLP_ARTIFACT_PATH.exists():
    st.error("One or more model artifacts are missing. Re-run the training notebooks before using this studio.")
    st.stop()


# ── Load ──────────────────────────────────────────────────────────────────────
reference_features = get_reference_features()
stacking_artifact  = get_stacking_artifact()
mlp_artifact, mlp_model = get_mlp_bundle()

feature_columns = reference_features.columns.tolist()
feature_groups  = build_feature_groups(feature_columns)
default_values  = {col: default_value_text(reference_features[col]) for col in feature_columns}


st.markdown(f"""
<div class="info-note">
    <strong>{len(feature_columns)} required features</strong> — the form is prefilled with representative training-data values
    so you can test the interface immediately.
    Use <code>True</code>/<code>False</code> or <code>1</code>/<code>0</code> for boolean indicator columns
    (ocean, realm, exposure, substrate).
</div>
""", unsafe_allow_html=True)


tmpl_col, hint_col = st.columns([1, 2], gap="large")
with tmpl_col:
    st.download_button(
        "⬇ Download input template (CSV)",
        data=build_template_csv(reference_features.head(1)),
        file_name="coral_bleaching_input_template.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ── Prediction form ───────────────────────────────────────────────────────────
with st.form("prediction_form"):
    for section_name, section_fields in feature_groups:
        with st.expander(section_name, expanded=section_name in {"Location & Date", "Water & Temperature"}):
            st.markdown(f'<div class="group-hdr">{section_name}</div>', unsafe_allow_html=True)
            col_l, col_r = st.columns(2, gap="large")
            for idx, field in enumerate(section_fields):
                target = col_l if idx % 2 == 0 else col_r
                help_text = "Enter True/False or 1/0" if is_boolean_like(reference_features[field]) else None
                with target:
                    st.text_input(
                        label=field,
                        value=default_values[field],
                        key=f"field_{field}",
                        help=help_text,
                    )

    st.markdown("<br>", unsafe_allow_html=True)
    run_prediction = st.form_submit_button("🔬 Run Prediction", type="primary", use_container_width=True)


# ── Results ───────────────────────────────────────────────────────────────────
if run_prediction:
    input_payload = {col: st.session_state.get(f"field_{col}", "") for col in feature_columns}
    input_df = pd.DataFrame([input_payload])

    with st.spinner("Running both AI models…"):
        try:
            prediction_df, fill_notes = predict_all_models(
                input_df, reference_features, stacking_artifact, mlp_artifact, mlp_model,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    st.success("Prediction complete.")

    if fill_notes:
        with st.expander("ℹ Auto-filled values"):
            for note in fill_notes:
                st.caption(f"• {note}")

    st.markdown("---")
    st.markdown('<div class="sec-label">Prediction Results</div>', unsafe_allow_html=True)
    render_result_cards(prediction_df.iloc[0])

    result_cols = [
        "stacking_prediction", "bleaching_risk_probability", "bleaching_risk_flag",
        "two_stage_prediction", "hybrid_prediction", "mlp_prediction", "hybrid_vs_mlp_gap",
    ]

    res_left, res_right = st.columns([1.4, 1.0], gap="large")

    with res_left:
        st.markdown('<div class="sec-label">Detailed Output Table</div>', unsafe_allow_html=True)
        st.dataframe(
            prediction_df[result_cols].round(4),
            use_container_width=True, hide_index=True,
        )
        st.download_button(
            "⬇ Download prediction (CSV)",
            data=build_download_bytes(prediction_df),
            file_name="coral_bleaching_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with res_right:
        st.markdown('<div class="sec-label">Quick Summary</div>', unsafe_allow_html=True)
        row = prediction_df.iloc[0]
        summary_df = pd.DataFrame({
            "Metric": ["Hybrid prediction", "Residual MLP", "Risk probability", "Risk flag"],
            "Value": [
                f"{row['hybrid_prediction']:.2f}%",
                f"{row['mlp_prediction']:.2f}%",
                f"{row['bleaching_risk_probability']*100:.2f}%",
                "High risk" if int(row["bleaching_risk_flag"]) == 1 else "Below threshold",
            ],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ── Model details ─────────────────────────────────────────────────────────────
with st.expander("ℹ Model Architecture Details"):
    d_left, d_right = st.columns(2, gap="large")
    with d_left:
        st.markdown('<div class="gpanel-title">Hybrid Stacking Ensemble</div>', unsafe_allow_html=True)
        st.caption(
            "Combines a stacking regressor (RF · XGBoost · LightGBM + Ridge meta-learner) "
            "with a two-stage classifier/regressor pipeline. The final output blends the "
            "stacking prediction with the two-stage severity estimate using an optimized alpha weight."
        )
    with d_right:
        st.markdown('<div class="gpanel-title">Custom Residual MLP</div>', unsafe_allow_html=True)
        st.caption(
            "A residual neural network with tabular self-attention, LayerNorm, and Huber loss. "
            "Trained on log1p-transformed targets. Provides a neural complement to the "
            "tree-based hybrid pipeline for capturing high-order non-linear feature interactions."
        )
