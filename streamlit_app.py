from __future__ import annotations

from pathlib import Path

import streamlit as st

from model_inference import DATA_PATH, MLP_ARTIFACT_PATH, STACKING_ARTIFACT_PATH


st.set_page_config(
    page_title="Coral Bleaching AI Studio",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def format_size_mb(path: Path) -> str:
    if not path.exists():
        return "-"
    return f"{path.stat().st_size / (1024 * 1024):.1f} MB"


def model_card(path: Path, model_name: str, model_desc: str, icon: str, missing_action: str) -> str:
    is_ready = path.exists()
    if is_ready:
        badge = f'<span class="badge badge-ok">● Online</span>'
        footer = f'<div class="card-footer">{path.name} &nbsp;·&nbsp; {format_size_mb(path)}</div>'
    else:
        badge = f'<span class="badge badge-warn">● Offline</span>'
        footer = f'<div class="card-footer warn-text">{missing_action}</div>'
    return f"""
    <div class="model-card {'card-ok' if is_ready else 'card-warn'}">
        <div class="card-top">
            <span class="card-icon">{icon}</span>
            <div>
                <div class="card-name">{model_name}</div>
                {badge}
            </div>
        </div>
        <div class="card-desc">{model_desc}</div>
        {footer}
    </div>
    """


data_fresh = DATA_PATH.exists()

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Geist:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
:root {
    --abyss:      #070E1A;
    --deep:       #0B1929;
    --ocean:      #0E2A45;
    --cerulean:   #1565C0;
    --sky:        #1E88E5;
    --bio:        #00E5C3;
    --bio-dim:    #00B39A;
    --coral:      #FF6B6B;
    --amber:      #FFB347;
    --snow:       #EFF6FF;
    --mist:       #A8C4E0;
    --glass:      rgba(255,255,255,0.04);
    --glass-b:    rgba(255,255,255,0.08);
    --border:     rgba(0,229,195,0.14);
    --border-s:   rgba(0,229,195,0.30);
}

html, body, .stApp {
    background: var(--abyss) !important;
    color: var(--snow) !important;
    font-family: 'Geist', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 3rem !important; max-width: 1320px !important; }
[data-testid="stSidebarNav"] { display: none; }

/* ── Animated mesh background ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 20% 30%, rgba(0,229,195,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 80% 70%, rgba(21,101,192,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 40% 60% at 50% 10%, rgba(0,179,154,0.05) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* ── Topbar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.4rem;
    background: rgba(11,25,41,0.80);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 16px;
    margin-bottom: 2rem;
    position: relative;
    z-index: 10;
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: var(--snow);
    letter-spacing: -0.01em;
}
.brand-dot {
    width: 8px; height: 8px;
    background: var(--bio);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--bio);
    animation: pulse-dot 2.4s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--bio); }
    50%       { opacity: 0.55; box-shadow: 0 0 16px var(--bio); }
}
.nav-pills { display: flex; gap: 0.4rem; }
.nav-pill {
    background: var(--glass);
    border: 1px solid var(--border);
    color: var(--mist);
    border-radius: 999px;
    padding: 0.28rem 0.75rem;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.01em;
}

/* ── Hero ── */
.hero {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    padding: 3rem 2.8rem;
    margin-bottom: 2rem;
    background: linear-gradient(130deg, #0B1D33 0%, #0A1628 50%, #071525 100%);
    border: 1px solid var(--border-s);
    box-shadow: 0 32px 72px rgba(0,0,0,0.5), inset 0 1px 0 rgba(0,229,195,0.15);
}
.hero::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 55%;
    height: 180%;
    background: radial-gradient(ellipse, rgba(0,229,195,0.12) 0%, transparent 65%);
    pointer-events: none;
}
.hero-label {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(0,229,195,0.10);
    border: 1px solid rgba(0,229,195,0.25);
    border-radius: 999px;
    padding: 0.28rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--bio);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.4rem;
    line-height: 1.05;
    letter-spacing: -0.04em;
    color: var(--snow);
    margin: 0 0 1rem 0;
}
.hero h1 .accent { color: var(--bio); }
.hero-sub {
    font-size: 1.05rem;
    line-height: 1.75;
    color: var(--mist);
    max-width: 58rem;
    margin: 0 0 0.6rem 0;
}
.hero-sub-th {
    font-size: 0.92rem;
    line-height: 1.7;
    color: rgba(168,196,224,0.70);
    max-width: 58rem;
    margin: 0;
}

/* ── Section label ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--bio-dim);
    margin-bottom: 1rem;
}

/* ── Glass panel ── */
.glass-panel {
    background: rgba(11,25,41,0.60);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1.5rem 1.6rem;
    box-shadow: 0 20px 50px rgba(0,0,0,0.30);
}

/* ── Tab content ── */
.tab-body {
    padding: 1.4rem 0 0.4rem 0;
}
.tab-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--snow);
    margin: 0 0 0.85rem 0;
}
.tab-text {
    font-size: 0.94rem;
    line-height: 1.8;
    color: var(--mist);
    margin-bottom: 0.8rem;
}
.tab-text-th {
    font-size: 0.88rem;
    line-height: 1.75;
    color: rgba(168,196,224,0.65);
    margin: 0;
}

/* ── Model cards ── */
.model-card {
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.8rem;
    border: 1px solid transparent;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.model-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.3);
}
.card-ok {
    background: rgba(0,229,195,0.06);
    border-color: rgba(0,229,195,0.20);
}
.card-warn {
    background: rgba(255,179,71,0.06);
    border-color: rgba(255,179,71,0.22);
}
.card-top {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
}
.card-icon {
    font-size: 1.5rem;
    line-height: 1;
}
.card-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.93rem;
    font-weight: 700;
    color: var(--snow);
    margin-bottom: 0.2rem;
}
.badge {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.18rem 0.6rem;
    border-radius: 999px;
    display: inline-block;
}
.badge-ok {
    background: rgba(0,229,195,0.14);
    color: var(--bio);
    border: 1px solid rgba(0,229,195,0.30);
}
.badge-warn {
    background: rgba(255,179,71,0.14);
    color: var(--amber);
    border: 1px solid rgba(255,179,71,0.30);
}
.card-desc {
    font-size: 0.83rem;
    line-height: 1.6;
    color: var(--mist);
    margin-bottom: 0.65rem;
}
.card-footer {
    font-size: 0.75rem;
    color: rgba(168,196,224,0.55);
    font-family: 'Geist Mono', monospace;
    letter-spacing: 0.01em;
}
.warn-text { color: rgba(255,179,71,0.7); }

/* ── Data source card ── */
.data-card {
    border-radius: 18px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(30,136,229,0.22);
    background: rgba(30,136,229,0.06);
    display: flex;
    align-items: center;
    gap: 0.9rem;
}
.data-icon {
    width: 2.2rem; height: 2.2rem;
    background: rgba(30,136,229,0.18);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.data-meta { flex: 1; }
.data-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--snow);
    margin-bottom: 0.2rem;
}
.data-path {
    font-size: 0.75rem;
    color: rgba(168,196,224,0.55);
    font-family: monospace;
}

/* ── Footer note ── */
.footer-note {
    margin-top: 2rem;
    padding: 1rem 1.3rem;
    background: rgba(0,229,195,0.05);
    border: 1px solid rgba(0,229,195,0.14);
    border-left: 3px solid var(--bio-dim);
    border-radius: 14px;
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--mist);
}

/* ── Streamlit tab override ── */
div[data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0.35rem !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0 !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--mist) !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.55rem 1rem !important;
    font-family: 'Geist', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
}
button[data-baseweb="tab"]:hover {
    color: var(--snow) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--bio) !important;
    border-bottom-color: var(--bio) !important;
}
div[data-baseweb="tab-highlight"] { display: none !important; }

/* ── Streamlit button override ── */
.stButton > button {
    background: linear-gradient(135deg, var(--cerulean), var(--sky)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.8rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 6px 22px rgba(21,101,192,0.38) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(21,101,192,0.50) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Topbar ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-brand">
        <div class="brand-dot"></div>
        Coral Bleaching AI Studio
    </div>
    <div class="nav-pills">
        <div class="nav-pill">Dashboard</div>
        <div class="nav-pill">Prediction</div>
        <div class="nav-pill">System Logs</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<section class="hero">
    <div class="hero-label">🪸 AI-Powered Marine Science</div>
    <h1>Coral Bleaching<br><span class="accent">AI Studio</span></h1>
    <p class="hero-sub">
        Two complementary AI models for high-accuracy coral bleaching risk prediction —
        analyze processed environmental data and compare combined insights in real time.
    </p>
    <p class="hero-sub-th">
        โมเดล AI สองตัวสำหรับการทำนายความเสี่ยงการฟอกขาวของปะการังที่มีความแม่นยำสูง
        วิเคราะห์ข้อมูลสภาพแวดล้อมและเปรียบเทียบผลลัพธ์แบบ real-time
    </p>
</section>
""", unsafe_allow_html=True)


# ── Main layout ──────────────────────────────────────────────────────────────
main_col, side_col = st.columns([1.75, 1.0], gap="large")

with main_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Model Architecture</div>', unsafe_allow_html=True)

    tab_stacking, tab_mlp = st.tabs(["Hybrid Stacking Ensemble", "Custom Residual MLP"])

    with tab_stacking:
        st.markdown("""
        <div class="tab-body">
            <div class="tab-title">Hybrid Stacking Ensemble</div>
            <p class="tab-text">
                Our primary approach utilizes a sophisticated Stacking Ensemble — a multi-level architecture that
                strategically combines diverse machine learning base models highly effective for structured environmental data.
                Base-level predictions are passed into a higher-level meta-learner, which learns to optimally weight and
                combine these signals. This approach reduces overall variance, offsets individual model biases, and creates
                a robust prediction pipeline suited to complex ecological risk estimation.
            </p>
            <p class="tab-text-th">
                แนวทางหลักของเราใช้ Stacking Ensemble ที่ซับซ้อน สถาปัตยกรรมหลายระดับนี้รวมโมเดล Machine Learning
                ที่หลากหลายเข้าด้วยกัน จากนั้นคำทำนายจะถูกส่งต่อไปยัง meta-learner เพื่อเรียนรู้วิธีการรวมผลลัพธ์
                ให้เหมาะสมที่สุด ช่วยลดความแปรปรวนและสร้างการทำนายที่แข็งแกร่ง
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tab_mlp:
        st.markdown("""
        <div class="tab-body">
            <div class="tab-title">Custom Residual MLP</div>
            <p class="tab-text">
                The platform also employs a Custom Residual Multi-Layer Perceptron for deep feature analysis.
                While tree-based models excel at tabular data, this neural architecture captures intricate, high-order,
                non-linear relationships by extending a standard MLP with residual connections — preserving information
                across deeper layers and mitigating vanishing-gradient behavior. The model learns subtle ecological
                interactions that may not be captured by tree-based pipelines alone.
            </p>
            <p class="tab-text-th">
                ใช้ Custom Residual MLP สำหรับการวิเคราะห์คุณลักษณะเชิงลึก สถาปัตยกรรมนี้ต่อยอดจาก MLP ทั่วไป
                ด้วย residual connections ซึ่งช่วยรักษาข้อมูลระหว่างชั้นลึกของเครือข่าย
                และลดปัญหา vanishing gradient ทำให้เรียนรู้รูปแบบเชิงนิเวศที่ละเอียดอ่อนมากขึ้น
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


with side_col:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">System Dashboard</div>', unsafe_allow_html=True)

    st.markdown(model_card(
        STACKING_ARTIFACT_PATH,
        "Hybrid Stacking Ensemble",
        "RF · XGBoost · LightGBM + Ridge meta-learner with two-stage classifier",
        "🤖",
        "Re-run the ensemble training notebook and export cells.",
    ), unsafe_allow_html=True)

    st.markdown(model_card(
        MLP_ARTIFACT_PATH,
        "Custom Residual MLP",
        "Tabular attention · Residual blocks · LayerNorm · Huber loss",
        "🧠",
        "Re-run the neural training notebook and artifact export.",
    ), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="data-card">
        <div class="data-icon">🗃️</div>
        <div class="data-meta">
            <div class="data-name">Environmental Data Source {'✓' if data_fresh else '✗'}</div>
            <div class="data-path">{DATA_PATH.name}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Open Prediction Studio →", type="primary", use_container_width=True):
        if hasattr(st, "switch_page"):
            st.switch_page("pages/1_Coral_Bleaching_AI_Studio.py")
        else:
            st.info("Open **Coral Bleaching AI Studio** from the sidebar menu.")


st.markdown("""
<div class="footer-note">
    This landing page provides an operational overview of both prediction models.
    Open the <strong>Prediction Studio</strong> to enter processed feature values, submit the form,
    and review the combined inference output from both AI models.
</div>
""", unsafe_allow_html=True)
