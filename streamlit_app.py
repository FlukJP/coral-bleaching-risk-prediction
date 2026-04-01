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


def model_dashboard_status(path: Path, model_name: str, missing_action: str) -> str:
    is_ready = path.exists()
    if is_ready:
        return f"""
        <div class="status-item ok">
            <div class="status-icon">OK</div>
            <div class="status-content">
                <div class="status-title">{model_name}</div>
                <div class="status-line"><strong>Status:</strong> Online | <strong>File size:</strong> {format_size_mb(path)}</div>
            </div>
        </div>
        """
    return f"""
    <div class="status-item warn">
        <div class="status-icon">!</div>
        <div class="status-content">
            <div class="status-title">{model_name}</div>
            <div class="status-line"><strong>Status:</strong> Re-training required</div>
            <div class="status-line"><strong>Artifact:</strong> {path.name} missing</div>
            <div class="status-line"><strong>Action:</strong> {missing_action}</div>
        </div>
    </div>
    """


data_fresh = DATA_PATH.exists()

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
    .stApp {
        background: var(--snow);
        color: var(--slate);
    }
    .topbar {
        background: var(--navy);
        color: var(--white);
        border-radius: 18px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 14px 30px rgba(26, 54, 93, 0.18);
    }
    .topbar-title {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .topbar-nav {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .topbar-pill {
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: var(--white);
        border-radius: 999px;
        padding: 0.34rem 0.75rem;
        font-size: 0.88rem;
    }
    .hero {
        background: linear-gradient(135deg, var(--navy), #264975);
        color: var(--white);
        border-radius: 24px;
        padding: 1.9rem 2rem;
        box-shadow: 0 18px 42px rgba(26, 54, 93, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0 0 0.45rem 0;
        font-size: 2.15rem;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0;
        max-width: 66rem;
        line-height: 1.75;
        font-size: 1rem;
        opacity: 0.97;
    }
    .panel {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.2rem 1.25rem;
        box-shadow: 0 14px 36px rgba(26, 54, 93, 0.07);
    }
    .panel h3 {
        color: var(--navy);
        margin-top: 0;
        margin-bottom: 0.7rem;
    }
    .panel p {
        color: var(--slate);
        line-height: 1.8;
        margin-bottom: 0.9rem;
    }
    .status-item {
        display: flex;
        gap: 0.8rem;
        align-items: flex-start;
        border-radius: 16px;
        padding: 0.85rem 0.9rem;
        margin-bottom: 0.8rem;
        border: 1px solid transparent;
    }
    .status-item.ok {
        background: rgba(43, 108, 176, 0.06);
        border-color: rgba(43, 108, 176, 0.14);
    }
    .status-item.warn {
        background: rgba(255, 204, 0, 0.12);
        border-color: rgba(255, 204, 0, 0.40);
    }
    .status-icon {
        min-width: 2rem;
        height: 2rem;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--white);
        background: var(--cerulean);
    }
    .status-item.warn .status-icon {
        background: #C28A00;
    }
    .status-title {
        color: var(--navy);
        font-weight: 700;
        margin-bottom: 0.18rem;
    }
    .status-line {
        color: var(--slate);
        line-height: 1.55;
        font-size: 0.94rem;
    }
    .footer-note {
        margin-top: 1rem;
        background: rgba(255, 255, 255, 0.92);
        border-left: 4px solid var(--cerulean);
        border-radius: 14px;
        padding: 0.95rem 1rem;
        color: var(--slate);
        line-height: 1.7;
    }
    div[data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    button[data-baseweb="tab"] {
        background: rgba(43, 108, 176, 0.08);
        border-radius: 999px;
        color: var(--navy);
        padding: 0.35rem 0.9rem;
        border: 1px solid rgba(43, 108, 176, 0.16);
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: var(--navy);
        color: var(--white);
        border-color: var(--navy);
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
            This advanced application presents two complementary artificial intelligence models for high-accuracy coral bleaching risk prediction.
            Analyze processed environmental data, submit findings, and compare combined insights below.
            <br><br>
            แอปพลิเคชันขั้นสูงนี้มีโมเดลปัญญาประดิษฐ์ที่เสริมกันสองโมเดลเพื่อการทำนายความเสี่ยงของการฟอกขาวของปะการังที่มีความแม่นยำสูง
            วิเคราะห์ข้อมูลสภาพแวดล้อมที่ผ่านการประมวลผล ส่งผลการค้นหา และเปรียบเทียบข้อมูลเชิงลึกแบบผสมผสานด้านล่าง
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
            "combines diverse machine learning base models that are highly effective for structured environmental data. "
            "Their base-level predictions are then passed into a higher-level meta-learner, which learns how to optimally "
            "weight and combine these signals. This approach reduces overall variance, offsets individual model biases, "
            "and creates a robust prediction pipeline suited to complex ecological risk estimation."
        )
        st.write(
            "แนวทางแรกของเราใช้ Stacking Ensemble ที่ซับซ้อน สถาปัตยกรรมหลายระดับนี้รวมโมเดล Machine Learning หลักที่หลากหลาย "
            "ซึ่งเหมาะกับข้อมูลสิ่งแวดล้อมแบบมีโครงสร้าง จากนั้นคำทำนายในระดับฐานจะถูกส่งต่อไปยังโมเดล meta-learner "
            "เพื่อเรียนรู้วิธีการชั่งน้ำหนักและรวมผลลัพธ์ให้เหมาะสมที่สุด วิธีการนี้ช่วยลดความแปรปรวน ลดอคติของโมเดลแต่ละตัว "
            "และสร้างไปป์ไลน์การทำนายที่แข็งแกร่งสำหรับความเสี่ยงเชิงนิเวศที่ซับซ้อน"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_mlp:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Custom Residual MLP")
        st.write(
            "The platform also employs a Custom Residual Multi-Layer Perceptron for deep feature analysis. While tree-based "
            "models excel at tabular data, neural networks are more effective at capturing intricate, high-order, non-linear "
            "relationships. This architecture extends a standard MLP by adding residual connections, which preserve information "
            "across deeper layers and mitigate vanishing-gradient behavior. As a result, the model can learn subtle ecological "
            "interactions that may not be captured by tree-based pipelines alone."
        )
        st.write(
            "แพลตฟอร์มยังใช้ Custom Residual Multi-Layer Perceptron สำหรับการวิเคราะห์คุณลักษณะเชิงลึก แม้ว่าโมเดลแบบต้นไม้จะเด่นในข้อมูลแบบตาราง "
            "แต่เครือข่ายประสาทเหมาะกับการจับความสัมพันธ์ที่ไม่เป็นเส้นตรงและซับซ้อนในระดับสูง สถาปัตยกรรมนี้ต่อยอดจาก MLP ทั่วไปด้วย residual connections "
            "ซึ่งช่วยรักษาข้อมูลระหว่างชั้นลึกของเครือข่ายและลดปัญหา vanishing gradient ทำให้โมเดลเรียนรู้รูปแบบเชิงนิเวศที่ละเอียดอ่อนมากขึ้น"
        )
        st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### System Model Dashboard")
    st.markdown(
        model_dashboard_status(
            STACKING_ARTIFACT_PATH,
            "Core Stacking Ensemble",
            "Check the ensemble training notebook and re-execute the final export cells.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        model_dashboard_status(
            MLP_ARTIFACT_PATH,
            "Custom Residual MLP",
            "Check the neural training notebook and re-execute the artifact export process.",
        ),
        unsafe_allow_html=True,
    )
    data_status_html = f"""
    <div class="status-item {'ok' if data_fresh else 'warn'}">
        <div class="status-icon">{'OK' if data_fresh else '!'}</div>
        <div class="status-content">
            <div class="status-title">Global Environmental Data Source</div>
            <div class="status-line"><strong>Status:</strong> {'Fresh' if data_fresh else 'Unavailable'}</div>
            <div class="status-line"><strong>Path:</strong> {DATA_PATH.name}</div>
        </div>
    </div>
    """
    st.markdown(data_status_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if st.button("Manage App", type="primary", use_container_width=False):
    if hasattr(st, "switch_page"):
        st.switch_page("pages/1_Coral_Bleaching_AI_Studio.py")
    else:
        st.info("Open `Coral Bleaching AI Studio` from the sidebar menu.")


st.markdown(
    """
    <div class="footer-note">
        This landing page is intended to provide a clear operational overview of the two prediction models.
        Open the prediction page to enter processed feature values, submit the form, and review combined inference outputs.
    </div>
    """,
    unsafe_allow_html=True,
)
