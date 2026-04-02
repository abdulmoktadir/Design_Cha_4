import math
import hmac
from pathlib import Path
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog

EPS = 1e-12

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="A Dynamic Fuzzy Delphi-BWM-LBWA-Bonferroni-CoCoSo MCDM Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLES
# ============================================================
CSS = """
<style>
:root {
    --bg: #f4f7fb;
    --surface: rgba(255, 255, 255, 0.92);
    --surface-strong: #ffffff;
    --border: #d9e4f2;
    --text: #10233d;
    --muted: #60748c;
    --primary: #1f5eff;
    --primary-strong: #113a9f;
    --accent: #0f766e;
    --shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
    --shadow-soft: 0 8px 24px rgba(15, 23, 42, 0.05);
    --radius: 18px;
}

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", "Helvetica Neue", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(79, 124, 255, 0.11), transparent 28%),
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.08), transparent 24%),
        linear-gradient(180deg, #f8fbff 0%, #eef3f9 100%);
    color: var(--text);
}

.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2.5rem;
    max-width: 1480px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #13233d 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] * {
    color: #e6edf7;
}

[data-testid="stSidebar"] [data-baseweb="radio"] label {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.45rem;
}

[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
    background: rgba(255, 255, 255, 0.08);
}

.hero-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(245,249,255,0.95) 48%, rgba(238,245,255,0.98) 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 1.55rem 1.7rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.15rem;
}

.hero-eyebrow,
.section-eyebrow,
.info-label,
.kpi-label {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.74rem;
    font-weight: 700;
}

.hero-eyebrow,
.section-eyebrow,
.info-label {
    color: var(--primary-strong);
}

.hero-title {
    margin-top: 0.15rem;
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.03em;
    color: var(--text);
}

.hero-subtitle {
    margin-top: 0.55rem;
    max-width: 900px;
    font-size: 1rem;
    line-height: 1.65;
    color: var(--muted);
}

.hero-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    margin-top: 1.15rem;
}

.info-card,
.kpi-card,
.app-card {
    background: var(--surface-strong);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-soft);
}

.info-card {
    padding: 0.95rem 1rem;
}

.info-card strong {
    display: block;
    margin: 0.3rem 0 0.15rem;
    font-size: 1rem;
    color: var(--text);
}

.info-card small {
    display: block;
    line-height: 1.55;
    color: var(--muted);
}

.section-wrap {
    margin: 0.25rem 0 1rem;
    padding: 1rem 1.15rem;
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(31, 94, 255, 0.12);
    border-radius: 18px;
}

.section-title {
    margin-top: 0.2rem;
    font-size: 1.45rem;
    font-weight: 800;
    line-height: 1.2;
    color: var(--text);
}

.section-subtitle {
    margin-top: 0.35rem;
    max-width: 920px;
    color: var(--muted);
    line-height: 1.6;
}

.app-card {
    padding: 1.05rem 1.15rem;
    margin-bottom: 0.85rem;
}

.app-card strong {
    color: var(--text);
}

.app-card ul {
    margin: 0.55rem 0 0 1.1rem;
    color: var(--muted);
    line-height: 1.7;
}

.kpi-card {
    padding: 1rem 1.05rem;
    min-height: 132px;
}

.kpi-label {
    color: var(--muted);
}

.kpi-value {
    margin-top: 0.45rem;
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--text);
}

.kpi-note,
.small-note {
    margin-top: 0.4rem;
    color: var(--muted);
    font-size: 0.94rem;
    line-height: 1.55;
}

.stButton > button {
    border: 0;
    border-radius: 12px;
    padding: 0.72rem 1.15rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    color: white;
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-strong) 100%);
    box-shadow: 0 10px 24px rgba(31, 94, 255, 0.24);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 28px rgba(31, 94, 255, 0.28);
}

.stButton > button:focus:not(:active) {
    border: none;
}

button[data-baseweb="tab"] {
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.88);
    height: 42px;
    padding: 0 1rem;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(31, 94, 255, 0.12), rgba(17, 58, 159, 0.18));
    border-color: rgba(31, 94, 255, 0.3);
    color: var(--primary-strong);
    font-weight: 700;
}

[data-baseweb="tab-list"] {
    gap: 0.55rem;
    padding-bottom: 0.45rem;
}

div[data-testid="stDataFrame"],
div[data-testid="stTable"] {
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    background: rgba(255,255,255,0.92);
    box-shadow: var(--shadow-soft);
}

div[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 16px;
    background: rgba(255,255,255,0.78);
    box-shadow: var(--shadow-soft);
}

textarea,
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
.stNumberInput input {
    border-radius: 12px !important;
}

hr {
    border-color: rgba(16, 35, 61, 0.08);
}

h1, h2, h3 {
    color: var(--text);
    letter-spacing: -0.02em;
}

@media (max-width: 1200px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }
}

.sidebar-section-title {
    color: #e6edf7 !important;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0.35rem 0 0.75rem 0;
}

.sidebar-profile-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 16px;
    padding: 0.9rem;
    margin-bottom: 0.85rem;
    backdrop-filter: blur(6px);
}

.sidebar-profile-name {
    color: #ffffff;
    font-size: 0.95rem;
    font-weight: 700;
    margin-top: 0.55rem;
    margin-bottom: 0.18rem;
    line-height: 1.35;
}

.sidebar-profile-role {
    color: #cbd5e1;
    font-size: 0.78rem;
    line-height: 1.35;
    margin-bottom: 0.45rem;
}

.sidebar-profile-text {
    color: #94a3b8;
    font-size: 0.76rem;
    line-height: 1.5;
    margin-bottom: 0.45rem;
}

.login-page {
    max-width: 780px;
    margin: 2.25rem auto 0 auto;
    padding: 0 0.75rem 1.1rem 0.75rem;
}

.login-hero-box {
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #06b6d4 100%);
    border-radius: 30px;
    padding: 2.4rem 2.2rem;
    margin-bottom: 1.15rem;
    min-height: 290px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 20px 48px rgba(29, 78, 216, 0.20);
    border: 1px solid rgba(255,255,255,0.14);
    text-align: center;
    position: relative;
    overflow: hidden;
}

.login-hero-box h1 {
    color: #ffffff !important;
    font-size: clamp(2.2rem, 4vw, 3.2rem);
    line-height: 1.14;
    margin: 0 0 0.85rem 0;
    letter-spacing: -0.04em;
    position: relative;
    z-index: 1;
}

.login-hero-box p {
    color: rgba(255,255,255,0.93) !important;
    margin: 0 auto;
    max-width: 580px;
    font-size: 1.08rem;
    line-height: 1.65;
    position: relative;
    z-index: 1;
}

.login-form-note {
    text-align: center;
    color: #475569;
    font-size: 1.02rem;
    font-weight: 500;
    margin-bottom: 1.1rem;
}

div[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #dbe4f0;
    border-radius: 24px;
    padding: 1.55rem 1.55rem 1.15rem 1.55rem;
    box-shadow: 0 16px 36px rgba(15, 23, 42, 0.07);
    max-width: 780px;
    margin: 0 auto;
}

div[data-testid="stForm"] label p {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #0f172a !important;
}

div[data-testid="stForm"] .stTextInput input {
    min-height: 58px;
    font-size: 1rem;
    border-radius: 14px;
    padding: 0.85rem 1rem;
}

div[data-testid="stForm"] .stButton > button,
div[data-testid="stForm"] .stFormSubmitButton > button {
    min-height: 54px;
    border-radius: 14px;
    font-size: 1.05rem;
    font-weight: 600;
}

.app-footer {
    text-align: center;
    margin-top: 2.6rem;
    padding: 1rem 0 0.25rem 0;
    color: #64748b;
    font-size: 0.78rem;
    border-top: 1px solid rgba(16, 35, 61, 0.08);
}

.copyright-pill {
    display: inline-block;
    margin-bottom: 0.55rem;
    padding: 0.38rem 0.9rem;
    border-radius: 999px;
    border: 1px solid #dbe4f0;
    background: #ffffff;
    color: #0f172a;
    font-weight: 700;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
}

@media (max-width: 900px) {
    .login-page {
        max-width: 100%;
        margin-top: 1.5rem;
    }

    .login-hero-box {
        min-height: auto;
        padding: 1.8rem 1.25rem;
        border-radius: 24px;
    }

    .login-hero-box p {
        font-size: 1rem;
    }
}

</style>
"""
st.markdown(CSS, unsafe_allow_html=True)



def logout():
    st.session_state.authenticated = False
    st.rerun()


def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    expected_password = st.secrets.get("APP_PASSWORD", None)

    left, center, right = st.columns([0.45, 3.6, 0.45])
    with center:
        st.markdown(
            """
            <div class="login-page">
                <div class="login-hero-box">
                    <h1>Dynamic Fuzzy Decision Platform</h1>
                    <p>
                        Secure access to the Fuzzy Delphi, Fuzzy BWM, Fuzzy LBWA,
                        Hybrid Integration, and Fuzzy Bonferroni CoCoSo workspace.
                    </p>
                </div>
                <div class="login-form-note">
                    Sign in with the application password
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if expected_password is None:
            st.error(
                "APP_PASSWORD is not configured. Add it in .streamlit/secrets.toml "
                "or Streamlit Cloud Settings → Secrets."
            )
            return False

        with st.form("login_form", clear_on_submit=False):
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter application password",
            )
            submitted = st.form_submit_button("Log in", use_container_width=True)

        if submitted:
            if hmac.compare_digest(password, str(expected_password)):
                st.session_state.authenticated = True
                st.success("Access granted.")
                st.rerun()
            else:
                st.error("Incorrect password.")

    return False


def get_asset_path(filename: str) -> Path:
    return Path(__file__).parent / "assets" / filename


def render_sidebar_profile_card(name, role, institution, image_path, brief_text, full_bio=None, extras=None):
    st.markdown('<div class="sidebar-profile-card">', unsafe_allow_html=True)

    image_path = Path(image_path)
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.caption(f"Image not found: {image_path.name}")

    st.markdown(
        f"""
        <div class="sidebar-profile-name">{name}</div>
        <div class="sidebar-profile-role">{role}<br>{institution}</div>
        <div class="sidebar-profile-text">{brief_text}</div>
        """,
        unsafe_allow_html=True,
    )

    if extras:
        for item in extras:
            st.markdown(
                f'<div class="sidebar-profile-text">• {item}</div>',
                unsafe_allow_html=True,
            )

    if full_bio:
        with st.expander(f"More about {name}", expanded=False):
            st.write(full_bio)

    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar_research_profiles():
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div class="sidebar-section-title">Researcher Profiles</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("👥 View researcher profiles", expanded=False):
        render_sidebar_profile_card(
            name="Prof. J.Z. Ren 任競爭",
            role="Associate Professor",
            institution="The Hong Kong Polytechnic University",
            image_path=get_asset_path("prof_jz_ren.png"),
            brief_text=(
                "Process systems engineering for energy, environment, and sustainability; "
                "recipient of the 2022 APEC ASPIRE Prize."
            ),
            full_bio=(
                "Dr. Jingzheng Ren is an Associate Professor at The Hong Kong Polytechnic "
                "University. His research focuses on process systems engineering for energy, "
                "environment and sustainability, including innovative industrial processes, "
                "decision tools, and optimization models for carbon-neutral industrial systems."
            ),
        )

        render_sidebar_profile_card(
            name="Md. Abdul Moktadir",
            role="Assistant Professor (Leather Products Engineering)",
            institution="University of Dhaka / PolyU Presidential PhD Fellow",
            image_path=get_asset_path("abdul_moktadir.png"),
            brief_text=(
                "Research interests include sustainable supply chains, logistics, risk "
                "management, Industry 4.0, and circular economy."
            ),
            extras=[
                "Affiliation: University of Dhaka",
                "Program: PolyU Presidential PhD Fellow",
            ],
            full_bio=(
                "Md. Abdul Moktadir is pursuing a PhD in Industrial and Systems Engineering "
                "at The Hong Kong Polytechnic University and serves as an Assistant Professor "
                "of Leather Products Engineering at the University of Dhaka."
            ),
        )


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            <span class="copyright-pill">© 2026 Research Toolkit</span><br>
            Developed by <strong>Moktadir M.A.</strong> and <strong>REN J.Z.</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_app_header():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Decision Analytics Workspace</div>
            <div class="hero-title">A Dynamic Fuzzy Delphi-BWM-LBWA-Bonferroni-CoCoSo MCDM Platform</div>
            <div class="hero-subtitle">
                A decision analytics workspace for fuzzy criteria screening, criteria weighting,
                hybrid integration, and alternative ranking.
            </div>
            <div class="hero-grid">
                <div class="info-card">
                    <span class="info-label">Workflow</span>
                    <strong>5 connected modules</strong>
                    <small>Fuzzy Delphi → Modified Fuzzy BWM → Fuzzy LBWA → Hybrid → Fuzzy Bonferroni CoCoSo</small>
                </div>
                <div class="info-card">
                    <span class="info-label">Scope</span>
                    <strong>Main-criteria weighting</strong>
                    <small>Fuzzy BWM and LBWA operate on the main criteria in this version</small>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title, subtitle="", eyebrow="Model Workspace"):
    title_html = escape(str(title))
    eyebrow_html = escape(str(eyebrow))
    subtitle_html = f'<div class="section-subtitle">{escape(str(subtitle))}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-wrap">
            <div class="section-eyebrow">{eyebrow_html}</div>
            <div class="section-title">{title_html}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{escape(str(item['label']))}</div>
                    <div class="kpi-value">{escape(str(item['value']))}</div>
                    <div class="kpi-note">{escape(str(item['note']))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================
# SCALES
# ============================================================
DELHI_SCALE = {
    "VLR": (0.1, 0.1, 0.3),
    "LR":  (0.1, 0.3, 0.5),
    "MR":  (0.3, 0.5, 0.7),
    "HR":  (0.5, 0.7, 0.9),
    "VHR": (0.7, 0.9, 0.9),
}

BWM_SCALE = {
    "EQ": (1, 1, 1),
    "VL": (1, 1, 3),
    "L":  (1, 3, 5),
    "M":  (3, 5, 7),
    "H":  (5, 7, 9),
    "VH": (7, 9, 9),
}

HYBRID_SCALE = {
    "VL": (0.00, 0.00, 0.16),
    "L":  (0.00, 0.16, 0.34),
    "ML": (0.16, 0.34, 0.50),
    "M":  (0.34, 0.50, 0.66),
    "MH": (0.50, 0.66, 0.84),
    "H":  (0.66, 0.84, 1.00),
    "VH": (0.84, 1.00, 1.00),
}

DELHI_MEANING = {
    "VLR": "Very Low Relevance",
    "LR": "Low Relevance",
    "MR": "Moderate Relevance",
    "HR": "High Relevance",
    "VHR": "Very High Relevance",
}

BWM_MEANING = {
    "EQ": "Equal",
    "VL": "Very Low",
    "L": "Low",
    "M": "Medium",
    "H": "High",
    "VH": "Very High",
}

HYBRID_MEANING = {
    "VL": "Very Low",
    "L": "Low",
    "ML": "Medium Low",
    "M": "Moderate",
    "MH": "Medium High",
    "H": "High",
    "VH": "Very High",
}

IB_CR_THRESHOLD_TABLE = {
    3: {3: 0.1667, 4: 0.1667, 5: 0.1667, 6: 0.1667, 7: 0.1667, 8: 0.1667, 9: 0.1667},
    4: {3: 0.1121, 4: 0.1529, 5: 0.1898, 6: 0.2206, 7: 0.2527, 8: 0.2577, 9: 0.2683},
    5: {3: 0.1354, 4: 0.1994, 5: 0.2306, 6: 0.2546, 7: 0.2716, 8: 0.2844, 9: 0.2960},
    6: {3: 0.1330, 4: 0.1990, 5: 0.2643, 6: 0.3044, 7: 0.3144, 8: 0.3221, 9: 0.3262},
    7: {3: 0.1294, 4: 0.2457, 5: 0.2819, 6: 0.3029, 7: 0.3144, 8: 0.3251, 9: 0.3403},
    8: {3: 0.1309, 4: 0.2521, 5: 0.2958, 6: 0.3154, 7: 0.3408, 8: 0.3620, 9: 0.3657},
    9: {3: 0.1359, 4: 0.2681, 5: 0.3062, 6: 0.3337, 7: 0.3517, 8: 0.3620, 9: 0.3662},
}

# ============================================================
# GENERIC HELPERS
# ============================================================
def gmi(tfn):
    return (tfn[0] + 4 * tfn[1] + tfn[2]) / 6.0

def defuzz_tfn(tfn):
    return gmi(tfn)

def safe_pos(x, eps=EPS):
    return max(float(x), eps)

def safe_normalize_to_1(v):
    v = np.asarray(v, dtype=float)
    s = np.nansum(v)
    if len(v) == 0:
        raise ValueError("Vector length cannot be zero.")
    if s <= 0 or np.isclose(s, 0.0) or np.isnan(s):
        return np.ones_like(v) / len(v)
    return v / s

def tfn_div(a, b):
    return (
        a[0] / max(b[2], EPS),
        a[1] / max(b[1], EPS),
        a[2] / max(b[0], EPS),
    )

def fuzzy_multiply(a, b):
    return (
        float(a[0]) * float(b[0]),
        float(a[1]) * float(b[1]),
        float(a[2]) * float(b[2]),
    )

def get_ib_consistency_threshold(criteria_count, scale_max=9):
    try:
        criteria_count = int(criteria_count)
        scale_max = int(scale_max)
    except Exception:
        return np.nan
    return IB_CR_THRESHOLD_TABLE.get(scale_max, {}).get(criteria_count, np.nan)

def get_ib_consistency_status(cr_value, threshold):
    if pd.isna(cr_value) or pd.isna(threshold):
        return "N/A"
    return "Acceptable" if float(cr_value) <= float(threshold) + EPS else "Revise"

def compute_bwm_input_based_consistency(expert_data, factor_names):
    summary_rows = []
    detail_tables = []

    for e_idx, ex in enumerate(expert_data, start=1):
        best_idx = int(ex["best_idx"])
        worst_idx = int(ex["worst_idx"])

        a_bw = ex["bto"][worst_idx]
        crisp_bw = gmi(a_bw)
        crisp_bw_sq = gmi(fuzzy_multiply(a_bw, a_bw))
        denom = crisp_bw_sq - crisp_bw

        rows = []
        for j, code in enumerate(factor_names):
            prod_tfn = fuzzy_multiply(ex["bto"][j], ex["otw"][j])
            crisp_prod = gmi(prod_tfn)

            if abs(denom) <= EPS:
                ib_cr = 0.0 if abs(crisp_prod - crisp_bw) <= EPS else np.nan
            else:
                ib_cr = abs((crisp_prod - crisp_bw) / denom)

            rows.append({
                "Code": code,
                "Best→Other TFN": ex["bto"][j],
                "Other→Worst TFN": ex["otw"][j],
                "R(aBj×ajW)": prod_tfn,
                "Crisp R(aBj×ajW)": crisp_prod,
                "R(aBW)": a_bw,
                "Crisp R(aBW)": crisp_bw,
                "R(aBW×aBW)": fuzzy_multiply(a_bw, a_bw),
                "Crisp R(aBW×aBW)": crisp_bw_sq,
                "IB Consistency Ratio": ib_cr,
            })

        detail_df = pd.DataFrame(rows)
        max_cr = float(np.nanmax(detail_df["IB Consistency Ratio"].values)) if len(detail_df) else np.nan

        summary_rows.append({
            "Expert": f"Ex{e_idx}",
            "Best": factor_names[best_idx],
            "Worst": factor_names[worst_idx],
            "Crisp R(aBW)": crisp_bw,
            "Crisp R(aBW×aBW)": crisp_bw_sq,
            "IB Consistency Ratio": max_cr,
        })

        detail_df.insert(0, "Expert", f"Ex{e_idx}")
        detail_tables.append(detail_df)

    return pd.DataFrame(summary_rows), detail_tables


def geometric_mean(tfns):
    prod_l = prod_m = prod_u = 1.0
    n = len(tfns)
    for t in tfns:
        prod_l *= max(t[0], EPS)
        prod_m *= max(t[1], EPS)
        prod_u *= max(t[2], EPS)
    return (
        prod_l ** (1 / n),
        prod_m ** (1 / n),
        prod_u ** (1 / n),
    )

def tfn_to_str(t, digits=6):
    return f"({t[0]:.{digits}f}, {t[1]:.{digits}f}, {t[2]:.{digits}f})"

def render_scale_table(scale_dict, meaning_dict, title):
    df = pd.DataFrame({
        "Code": list(scale_dict.keys()),
        "Meaning": [meaning_dict[k] for k in scale_dict.keys()],
        "TFN": [str(scale_dict[k]) for k in scale_dict.keys()],
    })
    with st.expander(title, expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

def to_bc_label(x: str) -> str:
    s = str(x).strip().upper()
    if s in {"B", "BENEFIT", "MAX"}:
        return "B"
    return "C"

def normalize_tfn_dict(weight_dict):
    keys = list(weight_dict.keys())
    crisp_vals = np.array([gmi(weight_dict[k]) for k in keys], dtype=float)
    crisp_norm = safe_normalize_to_1(crisp_vals)

    out = {}
    for k, c in zip(keys, crisp_norm):
        t = weight_dict[k]
        base = max(gmi(t), EPS)
        scale = c / base
        out[k] = (t[0] * scale, t[1] * scale, t[2] * scale)
    return out

def validate_unique_nonempty(values, label):
    values = [str(x).strip() for x in values]
    if any(v == "" for v in values):
        raise ValueError(f"{label} cannot contain empty values.")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique.")
    return values

def get_main_criteria_input(prefix, default_n=4):
    n_main = st.number_input(
        "Number of criteria",
        min_value=2,
        value=default_n,
        step=1,
        key=f"{prefix}_n_main"
    )

    rows = []
    for i in range(n_main):
        c1, c2 = st.columns([1.2, 3.2])
        with c1:
            code = st.text_input(
                f"Code {i+1}",
                value=f"C{i+1}",
                key=f"{prefix}_code_{i}"
            ).strip() or f"C{i+1}"
        with c2:
            name = st.text_input(
                f"Criterion {i+1}",
                value=f"Criterion {i+1}",
                key=f"{prefix}_name_{i}"
            ).strip() or f"Criterion {i+1}"

        rows.append({"code": code, "name": name})

    validate_unique_nonempty([r["code"] for r in rows], "Criterion codes")
    validate_unique_nonempty([r["name"] for r in rows], "Criterion names")
    return rows

def criteria_rows_to_df(criteria_rows):
    return pd.DataFrame({
        "Code": [r["code"] for r in criteria_rows],
        "Criterion": [r["name"] for r in criteria_rows],
    })

def sanitize_fuzzy_df(fuzzy_df: pd.DataFrame, criteria, alternatives) -> pd.DataFrame:
    out = fuzzy_df.copy().astype(float)
    for c in criteria:
        trip = out[[f"{c}_l", f"{c}_m", f"{c}_u"]].values
        trip = np.sort(trip, axis=1)
        out[[f"{c}_l", f"{c}_m", f"{c}_u"]] = trip
    return out.reindex(index=alternatives)


# ============================================================
# CoCoSo MIXED INPUT HELPERS
# ============================================================
COCOSO_LINGUISTIC_SCALE = {
    "P":  (8.0, 9.0, 9.0),
    "EH": (7.0, 8.0, 9.0),
    "VH": (6.0, 7.0, 8.0),
    "H":  (5.0, 6.0, 7.0),
    "M":  (4.0, 5.0, 6.0),
    "ML": (3.0, 4.0, 5.0),
    "L":  (2.0, 3.0, 4.0),
    "VL": (1.0, 2.0, 3.0),
    "EL": (1.0, 1.0, 2.0),
}

COCOSO_LINGUISTIC_MEANING = {
    "P":  "Perfect",
    "EH": "Extremely high",
    "VH": "Very high",
    "H":  "High",
    "M":  "Medium",
    "ML": "Medium low",
    "L":  "Low",
    "VL": "Very low",
    "EL": "Extremely low",
}

def render_cocoso_linguistic_scale():
    df = pd.DataFrame({
        "Code": list(COCOSO_LINGUISTIC_SCALE.keys()),
        "Meaning": [COCOSO_LINGUISTIC_MEANING[k] for k in COCOSO_LINGUISTIC_SCALE.keys()],
        "TFN": [str(COCOSO_LINGUISTIC_SCALE[k]) for k in COCOSO_LINGUISTIC_SCALE.keys()],
    })
    with st.expander("Show CoCoSo linguistic scale", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

def crisp_to_tfn_with_sigma(x, sigma=0.10):
    x = float(x)
    return (
        x * (1.0 - sigma),
        x,
        x * (1.0 + sigma),
    )

def aggregate_linguistic_tfn_arithmetic(codes):
    tfns = [COCOSO_LINGUISTIC_SCALE[str(c).strip().upper()] for c in codes]
    n = len(tfns)
    return (
        sum(t[0] for t in tfns) / n,
        sum(t[1] for t in tfns) / n,
        sum(t[2] for t in tfns) / n,
    )

def init_crisp_alt_df(criteria_codes):
    return pd.DataFrame({
        "Criterion": criteria_codes,
        "Value": np.ones(len(criteria_codes), dtype=float),
    })

def init_linguistic_alt_df(criteria_codes, n_exp):
    df = pd.DataFrame({"Criterion": criteria_codes})
    for i in range(n_exp):
        df[f"E{i+1}"] = "M"
    return df

def build_mixed_fuzzy_df_from_inputs(
    alt_names,
    criteria_meta,
    crisp_tables_by_alt,
    linguistic_tables_by_alt,
    sigma=0.10
):
    rows = []

    for alt in alt_names:
        row = {"Alternative": alt}

        for crit in criteria_meta:
            code = crit["Code"]
            mode = crit["Input Mode"]

            if mode == "Crisp":
                df = crisp_tables_by_alt[alt]
                val = float(df.loc[df["Criterion"] == code, "Value"].iloc[0])
                l, m, u = crisp_to_tfn_with_sigma(val, sigma=sigma)
            else:
                df = linguistic_tables_by_alt[alt]
                expert_cols = [c for c in df.columns if c.startswith("E")]
                codes = [str(df.loc[df["Criterion"] == code, c].iloc[0]).strip().upper() for c in expert_cols]
                if any(c not in COCOSO_LINGUISTIC_SCALE for c in codes):
                    raise ValueError(f"Invalid linguistic code found for {alt} / {code}.")
                l, m, u = aggregate_linguistic_tfn_arithmetic(codes)

            vals = sorted([float(l), float(m), float(u)])
            row[f"{code}_l"], row[f"{code}_m"], row[f"{code}_u"] = vals

        rows.append(row)

    fuzzy_df = pd.DataFrame(rows).set_index("Alternative")
    ordered_cols = []
    for crit in criteria_meta:
        code = crit["Code"]
        ordered_cols.extend([f"{code}_l", f"{code}_m", f"{code}_u"])
    return fuzzy_df[ordered_cols]

# ============================================================
# FUZZY DELPHI
# ============================================================
def run_delphi(criteria_tfns, threshold):
    selected, agg_tfns, gmi_vals = [], [], []
    for tfns in criteria_tfns:
        agg = geometric_mean(tfns)
        val = gmi(agg)
        agg_tfns.append(agg)
        gmi_vals.append(val)
        selected.append(val >= threshold)
    return selected, agg_tfns, gmi_vals

# ============================================================
# BWM CORE
# ============================================================
def choose_common_factor(idxs):
    counts = {}
    for idx in idxs:
        counts[idx] = counts.get(idx, 0) + 1
    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]
    return min(winners), counts

def transform_bwm_expert(expert, common_best_idx, common_worst_idx, n):
    orig_bto = expert["bto"]
    orig_otw = expert["otw"]
    divisor_b = orig_bto[common_best_idx]
    divisor_w = orig_otw[common_worst_idx]

    transformed_bto = []
    transformed_otw = []

    for j in range(n):
        transformed_bto.append((1, 1, 1) if j == common_best_idx else tfn_div(orig_bto[j], divisor_b))
    for j in range(n):
        transformed_otw.append((1, 1, 1) if j == common_worst_idx else tfn_div(orig_otw[j], divisor_w))

    return transformed_bto, transformed_otw

def aggregate_bwm_vectors(transformed_experts, n):
    agg_best, agg_worst = [], []
    for j in range(n):
        agg_best.append(geometric_mean([ex["bto_trans"][j] for ex in transformed_experts]))
        agg_worst.append(geometric_mean([ex["otw_trans"][j] for ex in transformed_experts]))
    return agg_best, agg_worst

def solve_bwm_aggregated_lp(agg_best, agg_worst, best_idx, worst_idx):
    n = len(agg_best)
    nvar = 3 * n + 1
    xi_idx = 3 * n

    def idx_l(i): return 3 * i
    def idx_m(i): return 3 * i + 1
    def idx_u(i): return 3 * i + 2

    A_ub, b_ub, A_eq, b_eq = [], [], [], []

    for j in range(n):
        if j == best_idx:
            continue
        lBj, mBj, uBj = agg_best[j]
        for sign in [1, -1]:
            row = np.zeros(nvar)
            row[idx_l(best_idx)] = sign
            row[idx_l(j)] = -sign * uBj
            row[xi_idx] = -1
            A_ub.append(row)
            b_ub.append(0)

            row = np.zeros(nvar)
            row[idx_m(best_idx)] = sign
            row[idx_m(j)] = -sign * mBj
            row[xi_idx] = -1
            A_ub.append(row)
            b_ub.append(0)

            row = np.zeros(nvar)
            row[idx_u(best_idx)] = sign
            row[idx_u(j)] = -sign * lBj
            row[xi_idx] = -1
            A_ub.append(row)
            b_ub.append(0)

    for j in range(n):
        if j == worst_idx:
            continue
        ljW, mjW, ujW = agg_worst[j]
        for sign in [1, -1]:
            row = np.zeros(nvar)
            row[idx_l(j)] = sign
            row[idx_l(worst_idx)] = -sign * ujW
            row[xi_idx] = -1
            A_ub.append(row)
            b_ub.append(0)

            row = np.zeros(nvar)
            row[idx_m(j)] = sign
            row[idx_m(worst_idx)] = -sign * mjW
            row[xi_idx] = -1
            A_ub.append(row)
            b_ub.append(0)

            row = np.zeros(nvar)
            row[idx_u(j)] = sign
            row[idx_u(worst_idx)] = -sign * ljW
            row[xi_idx] = -1
            A_ub.append(row)
            b_ub.append(0)

    row = np.zeros(nvar)
    for i in range(n):
        row[idx_l(i)] = 1 / 6
        row[idx_m(i)] = 4 / 6
        row[idx_u(i)] = 1 / 6
    A_eq.append(row)
    b_eq.append(1.0)

    row = np.zeros(nvar)
    for i in range(n):
        row[idx_m(i)] = 1
    A_eq.append(row)
    b_eq.append(1.0)

    for j in range(n):
        row = np.zeros(nvar)
        row[idx_u(j)] = 1
        for i in range(n):
            if i != j:
                row[idx_l(i)] = 1
        A_ub.append(row)
        b_ub.append(1.0)

    for j in range(n):
        row = np.zeros(nvar)
        row[idx_l(j)] = -1
        for i in range(n):
            if i != j:
                row[idx_u(i)] = -1
        A_ub.append(row)
        b_ub.append(-1.0)

    for i in range(n):
        row = np.zeros(nvar)
        row[idx_l(i)] = 1
        row[idx_m(i)] = -1
        A_ub.append(row)
        b_ub.append(0)

        row = np.zeros(nvar)
        row[idx_m(i)] = 1
        row[idx_u(i)] = -1
        A_ub.append(row)
        b_ub.append(0)

    c = np.zeros(nvar)
    c[xi_idx] = 1.0
    bounds = [(0, None)] * (3 * n) + [(0, 1)]

    res = linprog(
        c=c,
        A_ub=np.array(A_ub, dtype=float),
        b_ub=np.array(b_ub, dtype=float),
        A_eq=np.array(A_eq, dtype=float),
        b_eq=np.array(b_eq, dtype=float),
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise RuntimeError("Linear optimization failed: " + res.message)

    x = res.x
    weights = [(x[3 * i], x[3 * i + 1], x[3 * i + 2]) for i in range(n)]
    return weights, x[xi_idx]

def run_fbwm_group(expert_data, factor_names):
    n = len(factor_names)
    best_idxs = [ex["best_idx"] for ex in expert_data]
    worst_idxs = [ex["worst_idx"] for ex in expert_data]

    common_best_idx, best_counts = choose_common_factor(best_idxs)
    common_worst_idx, worst_counts = choose_common_factor(worst_idxs)

    transformed = []
    for ex in expert_data:
        bto_trans, otw_trans = transform_bwm_expert(ex, common_best_idx, common_worst_idx, n)
        transformed.append({
            **ex,
            "bto_trans": bto_trans,
            "otw_trans": otw_trans,
        })

    agg_best, agg_worst = aggregate_bwm_vectors(transformed, n)
    weights, xi = solve_bwm_aggregated_lp(agg_best, agg_worst, common_best_idx, common_worst_idx)

    return {
        "factor_names": factor_names,
        "common_best_idx": common_best_idx,
        "common_worst_idx": common_worst_idx,
        "best_counts": best_counts,
        "worst_counts": worst_counts,
        "transformed_experts": transformed,
        "agg_best": agg_best,
        "agg_worst": agg_worst,
        "weights_list": weights,
        "weights_dict": {f: w for f, w in zip(factor_names, weights)},
        "xi": xi,
    }

# ============================================================
# FLBWA CORE
# ============================================================
def scalar_divide_tfn(a, tfn_matrix):
    return np.column_stack([
        a / np.maximum(tfn_matrix[:, 2], EPS),
        a / np.maximum(tfn_matrix[:, 1], EPS),
        a / np.maximum(tfn_matrix[:, 0], EPS),
    ])

def defuzzify_weighted(tfn_matrix):
    return (tfn_matrix[:, 0] + 4 * tfn_matrix[:, 1] + tfn_matrix[:, 2]) / 6.0

def run_lbwa_excel_single_table(input_df, num_experts, theta, reference_idx):
    expected_cols = ["Factor", "Qi"] + [f"E{i+1}" for i in range(num_experts)]
    missing_cols = [c for c in expected_cols if c not in input_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    work_df = input_df.copy()
    work_df["Factor"] = work_df["Factor"].astype(str).str.strip()
    work_df["Factor"] = [name if name else f"Factor {i+1}" for i, name in enumerate(work_df["Factor"])]
    work_df["Qi"] = pd.to_numeric(work_df["Qi"], errors="coerce")

    expert_cols = [f"E{i+1}" for i in range(num_experts)]
    for c in expert_cols:
        work_df[c] = pd.to_numeric(work_df[c], errors="coerce")

    if work_df["Qi"].isna().any():
        raise ValueError("Qi contains invalid or empty values.")
    if work_df[expert_cols].isna().any().any():
        raise ValueError("One or more expert score cells contain invalid or empty values.")

    qi_arr = work_df["Qi"].astype(float).values
    data = work_df[expert_cols].astype(float).values

    if np.any(qi_arr < 0):
        raise ValueError("Qi values must be non-negative.")
    if np.any(data < 0):
        raise ValueError("Expert scores must be non-negative.")
    if theta <= 0:
        raise ValueError("Theta (θ) must be greater than zero.")
    if not (0 <= reference_idx < len(work_df)):
        raise ValueError("Reference index is out of range.")

    tfn = np.column_stack([np.min(data, axis=1), np.mean(data, axis=1), np.max(data, axis=1)])

    tfn_df = pd.DataFrame(tfn, columns=["l", "m", "u"])
    tfn_df.insert(0, "Qi", qi_arr)
    tfn_df.insert(0, "Factor", work_df["Factor"])

    denominator_tfn = np.column_stack([
        qi_arr * theta + tfn[:, 0],
        qi_arr * theta + tfn[:, 1],
        qi_arr * theta + tfn[:, 2],
    ])

    influence = scalar_divide_tfn(theta, denominator_tfn)

    influence_df = pd.DataFrame(influence, columns=["l", "m", "u"])
    influence_df.insert(0, "Qi", qi_arr)
    influence_df.insert(0, "Factor", work_df["Factor"])

    n = len(work_df)
    mask_others = np.ones(n, dtype=bool)
    mask_others[reference_idx] = False

    ref_weight = np.array([
        1 / np.maximum(1 + np.sum(influence[mask_others, 2]), EPS),
        1 / np.maximum(1 + np.sum(influence[mask_others, 1]), EPS),
        1 / np.maximum(1 + np.sum(influence[mask_others, 0]), EPS),
    ])

    fuzzy_weights = np.zeros_like(influence)
    fuzzy_weights[reference_idx] = ref_weight

    for i in range(n):
        if i != reference_idx:
            fuzzy_weights[i] = ref_weight * influence[i]

    fuzzy_weight_df = pd.DataFrame(fuzzy_weights, columns=["l", "m", "u"])
    fuzzy_weight_df.insert(0, "Factor", work_df["Factor"])

    crisp_values = defuzzify_weighted(fuzzy_weights)
    crisp_sum = np.sum(crisp_values)
    if crisp_sum <= 0:
        raise ValueError("The sum of crisp values is zero, so normalization cannot be done.")

    normalized_weights = crisp_values / crisp_sum

    result_df = pd.DataFrame({
        "Factor": work_df["Factor"],
        "Qi": qi_arr,
        "Crisp Value": crisp_values,
        "Normalized Weight": normalized_weights,
    })
    result_df["Rank"] = result_df["Normalized Weight"].rank(ascending=False, method="dense").astype(int)
    result_df = result_df[["Rank", "Factor", "Qi", "Crisp Value", "Normalized Weight"]] \
        .sort_values("Normalized Weight", ascending=False).reset_index(drop=True)

    return {
        "input_df": work_df,
        "tfn_df": tfn_df,
        "influence_df": influence_df,
        "fuzzy_weight_df": fuzzy_weight_df,
        "result_df": result_df,
        "weights": [tuple(x) for x in fuzzy_weights],
        "top_factor": str(result_df.iloc[0]["Factor"]),
        "top_weight": float(result_df.iloc[0]["Normalized Weight"]),
        "crisp_sum": float(crisp_sum),
    }

def lbwa_result_to_dict(lbwa_output):
    result_df = lbwa_output["result_df"].copy()
    fuzzy_df = lbwa_output["fuzzy_weight_df"].copy()

    fuzzy_map = {
        str(r["Factor"]).strip(): (float(r["l"]), float(r["m"]), float(r["u"]))
        for _, r in fuzzy_df.iterrows()
    }

    out = {
        str(r["Factor"]).strip(): fuzzy_map[str(r["Factor"]).strip()]
        for _, r in result_df.iterrows()
    }

    return normalize_tfn_dict(out)

# ============================================================
# HYBRID
# ============================================================
def combine_two_tfn_weights(w_bwm, w_lbwa, alpha_tfn, beta_tfn):
    """Return the hybrid numerators for (l, m, u).

    Following the hybridization equations in the user's reference:
    - lower numerator uses FBWM_l and FLBWA_l raised to alpha_u and beta_u
    - middle numerator uses FBWM_m and FLBWA_m raised to alpha_m and beta_m
    - upper numerator uses FBWM_u and FLBWA_u raised to alpha_l and beta_l
    """
    return (
        (safe_pos(w_bwm[0]) ** alpha_tfn[2]) * (safe_pos(w_lbwa[0]) ** beta_tfn[2]),
        (safe_pos(w_bwm[1]) ** alpha_tfn[1]) * (safe_pos(w_lbwa[1]) ** beta_tfn[1]),
        (safe_pos(w_bwm[2]) ** alpha_tfn[0]) * (safe_pos(w_lbwa[2]) ** beta_tfn[0]),
    )


def hybrid_denominator_terms(w_bwm, w_lbwa, alpha_tfn, beta_tfn):
    """Return the hybrid denominator terms for (l, m, u).

    The reference equations use the opposite TFN priority endpoints in the
    denominators for the lower and upper bounds:
    - lower denominator uses alpha_l and beta_l with the lower components
    - middle denominator uses alpha_m and beta_m with the middle components
    - upper denominator uses alpha_u and beta_u with the upper components
    """
    return (
        (safe_pos(w_bwm[0]) ** alpha_tfn[0]) * (safe_pos(w_lbwa[0]) ** beta_tfn[0]),
        (safe_pos(w_bwm[1]) ** alpha_tfn[1]) * (safe_pos(w_lbwa[1]) ** beta_tfn[1]),
        (safe_pos(w_bwm[2]) ** alpha_tfn[2]) * (safe_pos(w_lbwa[2]) ** beta_tfn[2]),
    )


def combine_main_weight_tables(fbwm_df, flbwa_df, alpha_tfn, beta_tfn):
    def _prepare(df, prefix):
        cols = ["Code", "Criterion", "Weight TFN"]
        optional_cols = ["Crisp Weight", "Normalized Weight", "Rank"]
        cols += [c for c in optional_cols if c in df.columns]
        out = df[cols].copy()
        rename_map = {
            "Weight TFN": f"{prefix} TFN",
            "Crisp Weight": f"Crisp {prefix}",
            "Normalized Weight": f"Normalized {prefix}",
            "Rank": f"Rank_{prefix}",
        }
        return out.rename(columns=rename_map)

    df1 = _prepare(fbwm_df, "FBWM")
    df2 = _prepare(flbwa_df, "FLBWA")

    merged = df1.merge(df2, on=["Code", "Criterion"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping criteria were found between FBWM and FLBWA inputs.")

    numerator_terms = []
    denominator_terms = []
    for _, r in merged.iterrows():
        numerator_terms.append(
            combine_two_tfn_weights(
                r["FBWM TFN"],
                r["FLBWA TFN"],
                alpha_tfn,
                beta_tfn,
            )
        )
        denominator_terms.append(
            hybrid_denominator_terms(
                r["FBWM TFN"],
                r["FLBWA TFN"],
                alpha_tfn,
                beta_tfn,
            )
        )

    sum_l = sum(x[0] for x in denominator_terms)
    sum_m = sum(x[1] for x in denominator_terms)
    sum_u = sum(x[2] for x in denominator_terms)

    hybrid_tfns = [
        (x[0] / max(sum_l, EPS), x[1] / max(sum_m, EPS), x[2] / max(sum_u, EPS))
        for x in numerator_terms
    ]

    merged["Hybrid Numerator TFN"] = numerator_terms
    merged["Hybrid Denominator TFN"] = denominator_terms
    merged["Hybrid TFN"] = hybrid_tfns
    merged["Hybrid Crisp"] = merged["Hybrid TFN"].apply(gmi)
    merged["Exact Crisp Value"] = merged["Hybrid Crisp"]
    merged["Normalized Hybrid"] = safe_normalize_to_1(merged["Hybrid Crisp"].values)
    merged["Normalized Value"] = merged["Normalized Hybrid"]
    merged["Rank"] = merged["Normalized Hybrid"].rank(ascending=False, method="dense").astype(int)

    return merged.sort_values("Normalized Hybrid", ascending=False).reset_index(drop=True)


def combine_sub_weight_tables(fbwm_df, flbwa_df, alpha_tfn, beta_tfn):
    """Excel-aligned sub-criteria hybridization.

    The uploaded workbook normalizes the sub-criteria hybrid stage against a
    single denominator built from *all* sub-criteria, not separately within
    each parent group. This function mirrors that workbook exactly:

    1. compute the hybrid numerator for each sub-criterion
    2. compute the denominator term for each sub-criterion
    3. sum denominator terms across the full sub-criteria set
    4. divide each numerator by the global denominator sum component-wise
    """
    def _prepare(df, prefix):
        cols = ["ParentCode", "Code", "Criterion", "Weight TFN"]
        optional_cols = ["Crisp Weight", "Normalized Weight", "Rank"]
        cols += [c for c in optional_cols if c in df.columns]
        out = df[cols].copy()
        rename_map = {
            "Weight TFN": f"{prefix} TFN",
            "Crisp Weight": f"Crisp {prefix}",
            "Normalized Weight": f"Normalized {prefix}",
            "Rank": f"Rank_{prefix}",
        }
        return out.rename(columns=rename_map)

    df1 = _prepare(fbwm_df, "FBWM")
    df2 = _prepare(flbwa_df, "FLBWA")
    merged = df1.merge(df2, on=["ParentCode", "Code", "Criterion"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping sub-criteria were found between FBWM and FLBWA inputs.")

    numerator_terms = []
    denominator_terms = []
    for _, r in merged.iterrows():
        numerator_terms.append(
            combine_two_tfn_weights(
                r["FBWM TFN"],
                r["FLBWA TFN"],
                alpha_tfn,
                beta_tfn,
            )
        )
        denominator_terms.append(
            hybrid_denominator_terms(
                r["FBWM TFN"],
                r["FLBWA TFN"],
                alpha_tfn,
                beta_tfn,
            )
        )

    sum_l = sum(x[0] for x in denominator_terms)
    sum_m = sum(x[1] for x in denominator_terms)
    sum_u = sum(x[2] for x in denominator_terms)

    hybrid_tfns = [
        (x[0] / max(sum_l, EPS), x[1] / max(sum_m, EPS), x[2] / max(sum_u, EPS))
        for x in numerator_terms
    ]

    merged["Hybrid Numerator TFN"] = numerator_terms
    merged["Hybrid Denominator TFN"] = denominator_terms
    merged["Local Hybrid TFN"] = hybrid_tfns
    merged["Local Hybrid Crisp"] = merged["Local Hybrid TFN"].apply(gmi)
    merged["Local Exact Crisp Value"] = merged["Local Hybrid Crisp"]
    merged["Local Normalized Value"] = safe_normalize_to_1(merged["Local Hybrid Crisp"].values)
    merged["Local Rank"] = merged["Local Normalized Value"].rank(ascending=False, method="dense").astype(int)

    return merged.sort_values(["Local Rank", "ParentCode", "Code"], ascending=[True, True, True]).reset_index(drop=True)


def build_hierarchical_hybrid_tables(main_hybrid_df, sub_local_hybrid_df):
    if main_hybrid_df is None or main_hybrid_df.empty:
        raise ValueError("Main hybrid table is empty.")
    if sub_local_hybrid_df is None or sub_local_hybrid_df.empty:
        raise ValueError("Sub-criteria hybrid table is empty.")

    main_map = {
        str(r["Code"]).strip(): r["Hybrid TFN"]
        for _, r in main_hybrid_df.iterrows()
    }

    global_tfns = []
    main_tfns = []
    for _, row in sub_local_hybrid_df.iterrows():
        parent = str(row["ParentCode"]).strip()
        if parent not in main_map:
            raise ValueError(f"Parent code '{parent}' has no matching main hybrid weight.")
        main_tfn = main_map[parent]
        local_tfn = row["Local Hybrid TFN"]
        main_tfns.append(main_tfn)
        global_tfns.append(fuzzy_multiply(main_tfn, local_tfn))

    out = sub_local_hybrid_df.copy()
    out["Main Hybrid TFN"] = main_tfns
    out["Global Hybrid TFN"] = global_tfns
    out["Hybrid TFN"] = out["Global Hybrid TFN"]
    out["Exact Crisp Value"] = out["Global Hybrid TFN"].apply(gmi)
    out["Normalized Value"] = safe_normalize_to_1(out["Exact Crisp Value"].values)
    out["Rank"] = out["Normalized Value"].rank(ascending=False, method="dense").astype(int)
    out = out.sort_values(["Rank", "ParentCode", "Code"], ascending=[True, True, True]).reset_index(drop=True)
    return out

# ============================================================
# FUZZY BONFERRONI CoCoSo
# ============================================================
def fuzzy_df_to_nested_matrix(fuzzy_df, criteria, alternatives):
    matrix = []
    for alt in alternatives:
        row = []
        for c in criteria:
            trip = (
                float(fuzzy_df.loc[alt, f"{c}_l"]),
                float(fuzzy_df.loc[alt, f"{c}_m"]),
                float(fuzzy_df.loc[alt, f"{c}_u"]),
            )
            row.append(tuple(sorted(trip)))
        matrix.append(row)
    return matrix

def normalize_cocoso_bonferroni(decision, types_bc):
    n_alt = len(decision)
    n_crit = len(types_bc)
    norm = [[(0.0, 0.0, 0.0) for _ in range(n_crit)] for _ in range(n_alt)]

    for j in range(n_crit):
        typ = to_bc_label(types_bc[j])

        if typ == "B":
            max_u = max(decision[i][j][2] for i in range(n_alt))
            max_u = safe_pos(max_u)
            for i in range(n_alt):
                l, m, u = decision[i][j]
                norm[i][j] = (l / max_u, m / max_u, u / max_u)
        else:
            min_l = min(decision[i][j][0] for i in range(n_alt))
            min_l = safe_pos(min_l)
            for i in range(n_alt):
                l, m, u = decision[i][j]
                l = safe_pos(l)
                m = safe_pos(m)
                u = safe_pos(u)
                norm[i][j] = (min_l / u, min_l / m, min_l / l)

    return norm

def compute_bonferroni(norm_matrix, weights, phi1=1.0, phi2=1.0):
    weights = safe_normalize_to_1(pd.Series(weights).astype(float).values)

    n_alt = len(norm_matrix)
    n_crit = len(weights)

    if n_crit < 2:
        raise ValueError("At least two criteria are required for fuzzy Bonferroni CoCoSo.")

    scob = []
    pcob = []
    exp_term = 1.0 / safe_pos(phi1 + phi2)

    for a in range(n_alt):
        s_l = 0.0
        s_m = 0.0
        s_u = 0.0

        log_p_l = 0.0
        log_p_m = 0.0
        log_p_u = 0.0

        for i in range(n_crit):
            wi = min(max(weights[i], EPS), 1.0 - EPS)
            denom = 1.0 - wi

            for j in range(n_crit):
                if i == j:
                    continue

                wj = weights[j]
                term = (wi * wj) / denom

                gi_l, gi_m, gi_u = norm_matrix[a][i]
                gj_l, gj_m, gj_u = norm_matrix[a][j]

                s_l += term * (safe_pos(gi_l) ** phi1) * (safe_pos(gj_l) ** phi2)
                s_m += term * (safe_pos(gi_m) ** phi1) * (safe_pos(gj_m) ** phi2)
                s_u += term * (safe_pos(gi_u) ** phi1) * (safe_pos(gj_u) ** phi2)

                base_l = safe_pos(phi1 * gi_l + phi2 * gj_l)
                base_m = safe_pos(phi1 * gi_m + phi2 * gj_m)
                base_u = safe_pos(phi1 * gi_u + phi2 * gj_u)

                log_p_l += term * math.log(base_l)
                log_p_m += term * math.log(base_m)
                log_p_u += term * math.log(base_u)

        s_l = safe_pos(s_l) ** exp_term
        s_m = safe_pos(s_m) ** exp_term
        s_u = safe_pos(s_u) ** exp_term
        scob.append((s_l, s_m, s_u))

        p_l = math.exp(log_p_l) / safe_pos(phi1 + phi2)
        p_m = math.exp(log_p_m) / safe_pos(phi1 + phi2)
        p_u = math.exp(log_p_u) / safe_pos(phi1 + phi2)
        pcob.append((p_l, p_m, p_u))

    return scob, pcob

def relative_significance_excel_style(scob, pcob, pi=0.5):
    sum_scob_l = sum(s[0] for s in scob)
    sum_scob_m = sum(s[1] for s in scob)
    sum_scob_u = sum(s[2] for s in scob)

    sum_pcob_l = sum(p[0] for p in pcob)
    sum_pcob_m = sum(p[1] for p in pcob)
    sum_pcob_u = sum(p[2] for p in pcob)

    min_scob_l = min(s[0] for s in scob)
    min_scob_m = min(s[1] for s in scob)
    min_scob_u = min(s[2] for s in scob)
    max_scob_l = max(s[0] for s in scob)
    max_scob_m = max(s[1] for s in scob)
    max_scob_u = max(s[2] for s in scob)

    min_pcob_l = min(p[0] for p in pcob)
    min_pcob_m = min(p[1] for p in pcob)
    min_pcob_u = min(p[2] for p in pcob)
    max_pcob_l = max(p[0] for p in pcob)
    max_pcob_m = max(p[1] for p in pcob)
    max_pcob_u = max(p[2] for p in pcob)

    psi_a, psi_b, psi_c = [], [], []

    for i in range(len(scob)):
        s = scob[i]
        p = pcob[i]

        a_l = (s[0] + p[0]) / safe_pos(sum_scob_u + sum_pcob_u)
        a_m = (s[1] + p[1]) / safe_pos(sum_scob_m + sum_pcob_m)
        a_u = (s[2] + p[2]) / safe_pos(sum_scob_l + sum_pcob_l)
        psi_a.append((a_l, a_m, a_u))

        b_l = (s[0] / safe_pos(min_scob_u)) + (p[0] / safe_pos(min_pcob_u))
        b_m = (s[1] / safe_pos(min_scob_m)) + (p[1] / safe_pos(min_pcob_m))
        b_u = (s[2] / safe_pos(min_scob_l)) + (p[2] / safe_pos(min_pcob_l))
        psi_b.append((b_l, b_m, b_u))

        c_l = (pi * s[0] + (1 - pi) * p[0]) / safe_pos(pi * max_scob_u + (1 - pi) * max_pcob_u)
        c_m = (pi * s[1] + (1 - pi) * p[1]) / safe_pos(pi * max_scob_m + (1 - pi) * max_pcob_m)
        c_u = (pi * s[2] + (1 - pi) * p[2]) / safe_pos(pi * max_scob_l + (1 - pi) * max_pcob_l)
        psi_c.append((c_l, c_m, c_u))

    return psi_a, psi_b, psi_c

def final_scores_bonferroni(psi_a, psi_b, psi_c, alternative_names=None):
    n_alt = len(psi_a)
    if alternative_names is None:
        alternative_names = [f"A{i+1}" for i in range(n_alt)]

    rows = []
    for i in range(n_alt):
        a = psi_a[i]
        b = psi_b[i]
        c = psi_c[i]

        prod_l = (safe_pos(a[0]) * safe_pos(b[0]) * safe_pos(c[0])) ** (1 / 3)
        prod_m = (safe_pos(a[1]) * safe_pos(b[1]) * safe_pos(c[1])) ** (1 / 3)
        prod_u = (safe_pos(a[2]) * safe_pos(b[2]) * safe_pos(c[2])) ** (1 / 3)

        avg_l = (a[0] + b[0] + c[0]) / 3.0
        avg_m = (a[1] + b[1] + c[1]) / 3.0
        avg_u = (a[2] + b[2] + c[2]) / 3.0

        final_tfn = (prod_l + avg_l, prod_m + avg_m, prod_u + avg_u)
        crisp = defuzz_tfn(final_tfn)

        rows.append([
            alternative_names[i],
            a[0], a[1], a[2],
            b[0], b[1], b[2],
            c[0], c[1], c[2],
            final_tfn[0], final_tfn[1], final_tfn[2],
            crisp,
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "Alternative",
            "psi_a_l", "psi_a_m", "psi_a_u",
            "psi_b_l", "psi_b_m", "psi_b_u",
            "psi_c_l", "psi_c_m", "psi_c_u",
            "Final_l", "Final_m", "Final_u",
            "Crisp",
        ],
    )
    df["Rank"] = df["Crisp"].rank(ascending=False, method="min").astype(int)
    return df.sort_values(["Crisp", "Alternative"], ascending=[False, True]).reset_index(drop=True)

def cocoso_bonferroni_from_app(fuzzy_df, types_bc, final_weights, phi1=1.0, phi2=1.0, pi=0.5):
    criteria = [c[:-2] for c in fuzzy_df.columns if c.endswith("_l")]
    alternatives = fuzzy_df.index.astype(str).tolist()

    fuzzy_df = sanitize_fuzzy_df(fuzzy_df, criteria, alternatives)
    decision = fuzzy_df_to_nested_matrix(fuzzy_df, criteria, alternatives)
    norm_matrix = normalize_cocoso_bonferroni(decision, types_bc)
    weights = pd.Series(final_weights, index=criteria).astype(float)

    scob, pcob = compute_bonferroni(norm_matrix, weights, phi1=phi1, phi2=phi2)
    psi_a, psi_b, psi_c = relative_significance_excel_style(scob, pcob, pi=pi)
    ranking_df = final_scores_bonferroni(psi_a, psi_b, psi_c, alternative_names=alternatives)

    norm_rows = []
    for i, alt in enumerate(alternatives):
        row = {"Alternative": alt}
        for j, c in enumerate(criteria):
            row[f"{c}_l"] = norm_matrix[i][j][0]
            row[f"{c}_m"] = norm_matrix[i][j][1]
            row[f"{c}_u"] = norm_matrix[i][j][2]
        norm_rows.append(row)
    norm_df = pd.DataFrame(norm_rows)

    scob_df = pd.DataFrame(scob, columns=["SCoB_l", "SCoB_m", "SCoB_u"], index=alternatives).reset_index().rename(columns={"index": "Alternative"})
    pcob_df = pd.DataFrame(pcob, columns=["PCoB_l", "PCoB_m", "PCoB_u"], index=alternatives).reset_index().rename(columns={"index": "Alternative"})
    psi_a_df = pd.DataFrame(psi_a, columns=["Ki1_l", "Ki1_m", "Ki1_u"], index=alternatives).reset_index().rename(columns={"index": "Alternative"})
    psi_b_df = pd.DataFrame(psi_b, columns=["Ki2_l", "Ki2_m", "Ki2_u"], index=alternatives).reset_index().rename(columns={"index": "Alternative"})
    psi_c_df = pd.DataFrame(psi_c, columns=["Ki3_l", "Ki3_m", "Ki3_u"], index=alternatives).reset_index().rename(columns={"index": "Alternative"})

    params_df = pd.DataFrame({
        "Parameter": ["phi1", "phi2", "pi"],
        "Value": [phi1, phi2, pi],
    })

    return ranking_df, params_df, norm_df, scob_df, pcob_df, psi_a_df, psi_b_df, psi_c_df

# ============================================================
# UI HELPERS
# ============================================================
def init_bwm_summary(factors, n_exp):
    if len(factors) < 2:
        raise ValueError("At least two factors are required for BWM.")
    return pd.DataFrame(
        [[factors[0], factors[-1]] for _ in range(n_exp)],
        columns=["Best", "Worst"],
        index=[f"E{i+1}" for i in range(n_exp)],
    )

def init_bwm_pairwise_df(factors, best_factor, worst_factor):
    df = pd.DataFrame({
        "Factor": factors,
        "B→j": ["M"] * len(factors),
        "j→W": ["M"] * len(factors),
    })
    df.loc[df["Factor"] == best_factor, "B→j"] = "EQ"
    df.loc[df["Factor"] == worst_factor, "j→W"] = "EQ"
    return df

def init_lbwa_editor_df(factors, n_exp):
    df = pd.DataFrame({"Factor": factors, "Qi": np.ones(len(factors), dtype=int)})
    for i in range(n_exp):
        df[f"E{i+1}"] = np.zeros(len(factors), dtype=float)
    return df


def tfn_components_df(codes, criteria, tfns, label):
    return pd.DataFrame({
        "Code": list(codes),
        "Criterion": list(criteria),
        f"{label}_l": [float(t[0]) for t in tfns],
        f"{label}_m": [float(t[1]) for t in tfns],
        f"{label}_u": [float(t[2]) for t in tfns],
    })


def build_hybrid_editor_from_saved(fbwm_df=None, flbwa_df=None):
    frames = []
    if fbwm_df is not None and not fbwm_df.empty:
        fbwm_part = tfn_components_df(
            fbwm_df["Code"],
            fbwm_df["Criterion"],
            fbwm_df["Weight TFN"],
            "FBWM",
        )
        if "Rank" in fbwm_df.columns:
            fbwm_part["Rank_FBWM"] = fbwm_df["Rank"].values
        frames.append(fbwm_part)
    if flbwa_df is not None and not flbwa_df.empty:
        flbwa_part = tfn_components_df(
            flbwa_df["Code"],
            flbwa_df["Criterion"],
            flbwa_df["Weight TFN"],
            "FLBWA",
        )
        if "Rank" in flbwa_df.columns:
            flbwa_part["Rank_FLBWA"] = flbwa_df["Rank"].values
        frames.append(flbwa_part)

    if not frames:
        return pd.DataFrame({
            "Code": ["C1", "C2", "C3"],
            "Criterion": ["Criterion 1", "Criterion 2", "Criterion 3"],
            "FBWM_l": [0.20, 0.15, 0.10],
            "FBWM_m": [0.24, 0.20, 0.12],
            "FBWM_u": [0.28, 0.25, 0.14],
            "FLBWA_l": [0.18, 0.16, 0.11],
            "FLBWA_m": [0.23, 0.21, 0.13],
            "FLBWA_u": [0.27, 0.26, 0.15],
        })

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["Code", "Criterion"], how="outer")

    for col in [
        "FBWM_l", "FBWM_m", "FBWM_u", "FLBWA_l", "FLBWA_m", "FLBWA_u",
        "Rank_FBWM", "Rank_FLBWA",
    ]:
        if col not in merged.columns:
            merged[col] = np.nan if col.startswith("Rank_") else 0.0

    merged = merged.sort_values(["Code", "Criterion"], kind="stable").reset_index(drop=True)
    return merged


def init_manual_hybrid_editor(n_criteria):
    return pd.DataFrame({
        "Code": [f"C{i+1}" for i in range(n_criteria)],
        "Criterion": [f"Criterion {i+1}" for i in range(n_criteria)],
        "FBWM_l": np.full(n_criteria, 0.10, dtype=float),
        "FBWM_m": np.full(n_criteria, 0.15, dtype=float),
        "FBWM_u": np.full(n_criteria, 0.20, dtype=float),
        "FLBWA_l": np.full(n_criteria, 0.10, dtype=float),
        "FLBWA_m": np.full(n_criteria, 0.15, dtype=float),
        "FLBWA_u": np.full(n_criteria, 0.20, dtype=float),
        "Rank_FBWM": np.full(n_criteria, np.nan),
        "Rank_FLBWA": np.full(n_criteria, np.nan),
    })


def sync_subcriteria_count_df(main_codes, existing_df=None, default_count=3):
    main_codes = [str(c).strip() for c in main_codes if str(c).strip()]
    existing_map = {}
    if existing_df is not None and not existing_df.empty:
        existing_map = {
            str(r["ParentCode"]).strip(): int(max(1, pd.to_numeric(r["SubCount"], errors="coerce") if not pd.isna(pd.to_numeric(r["SubCount"], errors="coerce")) else default_count))
            for _, r in existing_df.iterrows()
        }
    rows = []
    for code in main_codes:
        rows.append({"ParentCode": code, "SubCount": int(existing_map.get(code, default_count))})
    return pd.DataFrame(rows)


def build_subcriteria_editor(count_df, existing_df=None):
    seed_map = {}
    if existing_df is not None and not existing_df.empty:
        seed_map = {
            (str(r["ParentCode"]).strip(), str(r["Code"]).strip()): r
            for _, r in existing_df.iterrows()
        }

    rows = []
    for _, grp in count_df.iterrows():
        parent = str(grp["ParentCode"]).strip()
        count = int(max(1, pd.to_numeric(grp["SubCount"], errors="coerce") if not pd.isna(pd.to_numeric(grp["SubCount"], errors="coerce")) else 1))
        for j in range(1, count + 1):
            code = f"{parent}{j}"
            key = (parent, code)
            seed = seed_map.get(key, {})
            rows.append({
                "ParentCode": parent,
                "Code": code,
                "FBWM_l": float(seed.get("FBWM_l", 0.10)),
                "FBWM_m": float(seed.get("FBWM_m", 0.15)),
                "FBWM_u": float(seed.get("FBWM_u", 0.20)),
                "FLBWA_l": float(seed.get("FLBWA_l", 0.10)),
                "FLBWA_m": float(seed.get("FLBWA_m", 0.15)),
                "FLBWA_u": float(seed.get("FLBWA_u", 0.20)),
            })
    return pd.DataFrame(rows)


def hybrid_editor_to_model_df(editor_df, source_prefix):
    work = editor_df.copy()
    if "Code" not in work.columns:
        raise ValueError(f"{source_prefix} editor must include a 'Code' column.")
    validate_unique_nonempty(work["Code"].tolist(), f"{source_prefix} criterion codes")

    if "Criterion" not in work.columns:
        work["Criterion"] = work["Code"].astype(str)
    if "ParentCode" in work.columns:
        work["ParentCode"] = work["ParentCode"].astype(str).str.strip()
        if work["ParentCode"].eq("").any():
            raise ValueError(f"{source_prefix} sub-criteria rows must include ParentCode values.")

    tfns = []
    for _, row in work.iterrows():
        vals = [
            pd.to_numeric(row[f"{source_prefix}_l"], errors="coerce"),
            pd.to_numeric(row[f"{source_prefix}_m"], errors="coerce"),
            pd.to_numeric(row[f"{source_prefix}_u"], errors="coerce"),
        ]
        if any(pd.isna(v) for v in vals):
            raise ValueError(f"{source_prefix} contains empty TFN cells.")
        if any(float(v) < 0 for v in vals):
            raise ValueError(f"{source_prefix} TFN values must be non-negative.")
        l, m, u = map(float, vals)
        if not (l <= m <= u):
            raise ValueError(f"{source_prefix} TFN values must satisfy l ≤ m ≤ u for every criterion.")
        tfns.append((l, m, u))

    base_cols = []
    if "ParentCode" in work.columns:
        base_cols.append("ParentCode")
    base_cols += ["Code", "Criterion"]

    df = work[base_cols].copy()
    df["Code"] = df["Code"].astype(str).str.strip()
    df["Criterion"] = df["Criterion"].astype(str).str.strip()
    df["Weight TFN"] = tfns
    df["Crisp Weight"] = df["Weight TFN"].apply(gmi)

    if "ParentCode" in df.columns:
        df["Normalized Weight"] = np.nan
        df["Rank"] = np.nan
        for parent_code, idx in df.groupby("ParentCode", sort=False).groups.items():
            crisp_vals = df.loc[list(idx), "Crisp Weight"].astype(float).values
            norm_vals = safe_normalize_to_1(crisp_vals)
            df.loc[list(idx), "Normalized Weight"] = norm_vals
            df.loc[list(idx), "Rank"] = pd.Series(norm_vals, index=list(idx)).rank(ascending=False, method="dense").astype(int)
        return df.sort_values(["ParentCode", "Rank", "Code"], ascending=[True, True, True]).reset_index(drop=True)

    df["Normalized Weight"] = safe_normalize_to_1(df["Crisp Weight"].values)
    df["Rank"] = df["Normalized Weight"].rank(ascending=False, method="dense").astype(int)
    return df.sort_values("Normalized Weight", ascending=False).reset_index(drop=True)

def init_decision_df(criteria_names):
    return pd.DataFrame({
        "Criterion": criteria_names,
        "l": 0.0,
        "m": 0.5,
        "u": 1.0,
    })

def init_cocoso_criteria_weight_df(criteria_rows, weight_df=None):
    base = pd.DataFrame(criteria_rows).copy()

    required = {"Code", "Criterion", "Type"}
    missing = required - set(base.columns)
    if missing:
        raise ValueError(f"Missing columns in criteria_rows: {sorted(missing)}")

    if weight_df is not None and not weight_df.empty:
        if "Hybrid TFN" in weight_df.columns:
            weight_map = {r["Code"]: r["Hybrid TFN"] for _, r in weight_df.iterrows()}
        elif "Global Hybrid TFN" in weight_df.columns:
            weight_map = {r["Code"]: r["Global Hybrid TFN"] for _, r in weight_df.iterrows()}
        else:
            raise ValueError("weight_df must contain 'Hybrid TFN' or 'Global Hybrid TFN'.")

        base["w_l"] = base["Code"].map(lambda x: weight_map.get(x, (0.01, 0.02, 0.03))[0])
        base["w_m"] = base["Code"].map(lambda x: weight_map.get(x, (0.01, 0.02, 0.03))[1])
        base["w_u"] = base["Code"].map(lambda x: weight_map.get(x, (0.01, 0.02, 0.03))[2])
    else:
        base["w_l"] = 0.01
        base["w_m"] = 0.02
        base["w_u"] = 0.03

    return base[["Code", "Criterion", "w_l", "w_m", "w_u", "Type"]]

# ============================================================
# SESSION INIT
# ============================================================
for k, v in {
    "fbwm_main_df": None,
    "flbwa_main_df": None,
    "hybrid_main_df": None,
    "hybrid_sub_local_df": None,
    "hybrid_sub_global_df": None,
    "hybrid_sub_counts": None,
    "hybrid_sub_editor": None,
}.items():
    st.session_state.setdefault(k, v)

if not check_password():
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Model Navigator")
st.sidebar.caption("Move through validation, weighting, integration, and ranking.")
st.sidebar.success("✅ Authenticated")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    logout()

module = st.sidebar.radio(
    "Choose model",
    [
        "🏠 Home",
        "1) Fuzzy Delphi",
        "2) Fuzzy BWM",
        "3) Fuzzy LBWA",
        "4) Hybrid Integration",
        "5) Fuzzy Bonferroni CoCoSo",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown("Recommended flow: **Delphi → BWM → LBWA → Hybrid → CoCoSo**")
render_sidebar_research_profiles()

render_app_header()

# ============================================================
# HOME
# ============================================================
if module == "🏠 Home":
    render_section_header(
        "Overview",
        "Use the navigator on the left to move from criteria screening to weighting, hybrid integration, and final alternative ranking.",
        eyebrow="Platform Overview",
    )

    render_kpi_row([
        {"label": "Modules", "value": "5", "note": "Validation, dual weighting, hybridization, and ranking"},
        {"label": "Weighting engines", "value": "2", "note": "Fuzzy BWM and Fuzzy LBWA for main criteria"},
        {"label": "Ranking model", "value": "1", "note": "Fuzzy Bonferroni CoCoSo for alternatives"},
    ])

    left, right = st.columns([1.55, 1.0])
    with left:
        st.markdown(
            """
            <div class="app-card">
                <strong>Recommended workflow</strong>
                <ul>
                    <li>Run <strong>Fuzzy Delphi</strong> to validate the candidate criteria set.</li>
                    <li>Use <strong>Fuzzy BWM</strong> to derive main-criteria weights from expert judgments.</li>
                    <li>Use <strong>Fuzzy LBWA</strong> to generate a second set of main-criteria weights and save them for the hybrid module.</li>
                    <li>Run <strong>Fuzzy Bonferroni CoCoSo</strong> to rank alternatives with the final criteria weights.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="app-card">
                <strong>Scope note</strong>
                <div class="small-note">
                    Fuzzy BWM and Fuzzy LBWA continue to use the main criteria only. The hybrid module can reuse saved results or accept manual TFN inputs before sending final weights to Fuzzy Bonferroni CoCoSo.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# MODULE 1: DELPHI
# ============================================================
elif module == "1) Fuzzy Delphi":
    render_section_header(
        "Fuzzy Delphi – Criteria Validation",
        "Validate proposed criteria before moving into weighting and ranking stages.",
        eyebrow="Module 1",
    )

    threshold = st.number_input("Threshold (GMI)", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
    n_exp = st.number_input("Number of experts", min_value=1, value=5, step=1)
    n_criteria = st.number_input("Number of criteria", min_value=1, value=5, step=1)

    render_scale_table(DELHI_SCALE, DELHI_MEANING, "Show Delphi code legend")

    criteria_rows = []
    for i in range(n_criteria):
        c1, c2 = st.columns([1.2, 3.0])
        with c1:
            code = st.text_input(f"Criterion code {i+1}", value=f"C{i+1}", key=f"delphi_code_{i}")
        with c2:
            name = st.text_input(f"Criterion name {i+1}", value=f"Criterion {i+1}", key=f"delphi_name_{i}")
        criteria_rows.append({"Code": code.strip() or f"C{i+1}", "Criterion": name.strip() or f"Criterion {i+1}"})

    expert_cols = [f"E{i+1}" for i in range(n_exp)]
    delphi_key = f"delphi_independent_{n_exp}_{n_criteria}"

    if delphi_key not in st.session_state:
        df = pd.DataFrame(criteria_rows)
        for c in expert_cols:
            df[c] = "MR"
        st.session_state[delphi_key] = df

    work_df = st.session_state[delphi_key].copy()
    work_df["Code"] = [r["Code"] for r in criteria_rows]
    work_df["Criterion"] = [r["Criterion"] for r in criteria_rows]

    edited = st.data_editor(
        work_df,
        key="delphi_editor_independent",
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "Code": st.column_config.TextColumn("Code", disabled=True),
            "Criterion": st.column_config.TextColumn("Criterion", disabled=True),
            **{
                c: st.column_config.SelectboxColumn(c, options=list(DELHI_SCALE.keys()), required=True)
                for c in expert_cols
            },
        },
    )
    st.session_state[delphi_key] = edited.copy()

    if st.button("Run Fuzzy Delphi", type="primary"):
        try:
            validate_unique_nonempty(edited["Code"].tolist(), "Criterion codes")

            criteria_tfns = [
                [DELHI_SCALE[str(row[c])] for c in expert_cols]
                for _, row in edited.iterrows()
            ]

            selected, agg_tfns, gmi_vals = run_delphi(criteria_tfns, float(threshold))

            out = edited[["Code", "Criterion"]].copy()
            out["Aggregated TFN"] = [tfn_to_str(x, 4) for x in agg_tfns]
            out["GMI"] = gmi_vals
            out["Selected"] = selected

            st.dataframe(out, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(str(e))

# ============================================================
# MODULE 2: BWM
# ============================================================
elif module == "2) Fuzzy BWM":
    render_section_header(
        "Fuzzy BWM (Main Criteria Only)",
        "Capture expert best-to-others and others-to-worst judgments for main criteria.",
        eyebrow="Module 2",
    )

    n_exp_bwm = st.number_input("Number of experts", min_value=1, value=5, step=1, key="bwm_n_exp")
    render_scale_table(BWM_SCALE, BWM_MEANING, "Show BWM code legend")

    criteria_rows = get_main_criteria_input("bwm", default_n=4)
    criteria_df = criteria_rows_to_df(criteria_rows)
    factor_names = criteria_df["Code"].tolist()
    code_to_name = dict(zip(criteria_df["Code"], criteria_df["Criterion"]))

    st.subheader("Criteria preview")
    st.dataframe(criteria_df, use_container_width=True, hide_index=True)

    threshold_col1, threshold_col2 = st.columns([1, 1])
    with threshold_col1:
        threshold_scale = st.selectbox(
            "IB consistency threshold scale",
            options=list(range(3, 10)),
            index=6,
            help="Published Table-3 scale from Liang et al. (2020). For the current fuzzy BWM code set, use 9.",
        )
    ib_threshold_lookup = get_ib_consistency_threshold(len(factor_names), threshold_scale)
    with threshold_col2:
        if pd.isna(ib_threshold_lookup):
            st.metric("Published threshold", "N/A", help="The paper reports thresholds for 3–9 criteria and 3–9 scales.")
        else:
            st.metric("Published threshold", f"{ib_threshold_lookup:.4f}", help="Input-based consistency threshold looked up from the published Table 3.")
    if pd.isna(ib_threshold_lookup):
        st.warning("Published input-based thresholds are only available for 3–9 criteria and 3–9 scales.")
    else:
        st.caption(f"Using criteria count = {len(factor_names)} and scale = {threshold_scale} for the input-based consistency threshold lookup.")

    summary_key = f"bwm_summary_{len(factor_names)}_{n_exp_bwm}"
    if summary_key not in st.session_state:
        st.session_state[summary_key] = init_bwm_summary(factor_names, n_exp_bwm)

    summary_df = st.data_editor(
        st.session_state[summary_key],
        key=f"editor_{summary_key}",
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Best": st.column_config.SelectboxColumn("Best", options=factor_names, required=True),
            "Worst": st.column_config.SelectboxColumn("Worst", options=factor_names, required=True),
        },
    )
    st.session_state[summary_key] = summary_df.copy()

    pair_tables = []
    tabs = st.tabs([f"Expert {i+1}" for i in range(n_exp_bwm)])
    for e, tab in enumerate(tabs):
        with tab:
            best_factor = str(summary_df.iloc[e]["Best"])
            worst_factor = str(summary_df.iloc[e]["Worst"])
            tbl_key = f"bwm_pair_{e}_{len(factor_names)}"

            base_df = st.session_state.get(tbl_key, init_bwm_pairwise_df(factor_names, best_factor, worst_factor))
            if list(base_df["Factor"]) != factor_names:
                base_df = init_bwm_pairwise_df(factor_names, best_factor, worst_factor)

            base_df.loc[base_df["Factor"] == best_factor, "B→j"] = "EQ"
            base_df.loc[base_df["Factor"] == worst_factor, "j→W"] = "EQ"

            pair_df = st.data_editor(
                base_df,
                key=f"editor_{tbl_key}",
                use_container_width=True,
                num_rows="fixed",
                hide_index=True,
                column_config={
                    "Factor": st.column_config.TextColumn("Factor", disabled=True),
                    "B→j": st.column_config.SelectboxColumn("B→j", options=list(BWM_SCALE.keys()), required=True),
                    "j→W": st.column_config.SelectboxColumn("j→W", options=list(BWM_SCALE.keys()), required=True),
                },
            )

            pair_df.loc[pair_df["Factor"] == best_factor, "B→j"] = "EQ"
            pair_df.loc[pair_df["Factor"] == worst_factor, "j→W"] = "EQ"
            st.session_state[tbl_key] = pair_df.copy()
            pair_tables.append(pair_df)

    if st.button("Solve Fuzzy BWM", type="primary"):
        try:
            expert_data = []
            for e in range(n_exp_bwm):
                best = str(summary_df.iloc[e]["Best"])
                worst = str(summary_df.iloc[e]["Worst"])
                if best == worst:
                    raise ValueError(f"Expert {e+1}: best and worst cannot be identical.")

                table = pair_tables[e]
                expert_data.append({
                    "best_idx": factor_names.index(best),
                    "worst_idx": factor_names.index(worst),
                    "bto": [BWM_SCALE[str(row['B→j'])] for _, row in table.iterrows()],
                    "otw": [BWM_SCALE[str(row['j→W'])] for _, row in table.iterrows()],
                })

            out = run_fbwm_group(expert_data, factor_names)
            ib_summary_df, ib_detail_tables = compute_bwm_input_based_consistency(expert_data, factor_names)
            ib_threshold_value = get_ib_consistency_threshold(len(factor_names), threshold_scale)
            if not pd.isna(ib_threshold_value):
                ib_summary_df["Threshold"] = ib_threshold_value
                ib_summary_df["Status"] = ib_summary_df["IB Consistency Ratio"].apply(lambda x: get_ib_consistency_status(x, ib_threshold_value))
                for i, detail_df in enumerate(ib_detail_tables):
                    detail_df["Threshold"] = ib_threshold_value
                    detail_df["Status"] = detail_df["IB Consistency Ratio"].apply(lambda x: get_ib_consistency_status(x, ib_threshold_value))
                    ib_detail_tables[i] = detail_df
            else:
                ib_summary_df["Threshold"] = np.nan
                ib_summary_df["Status"] = "N/A"
                for i, detail_df in enumerate(ib_detail_tables):
                    detail_df["Threshold"] = np.nan
                    detail_df["Status"] = "N/A"
                    ib_detail_tables[i] = detail_df

            weights = normalize_tfn_dict(out["weights_dict"])

            result_df = pd.DataFrame({
                "Code": factor_names,
                "Criterion": [code_to_name[c] for c in factor_names],
                "Weight TFN": [weights[c] for c in factor_names],
            })
            result_df["Crisp Weight"] = result_df["Weight TFN"].apply(gmi)
            result_df["Normalized Weight"] = safe_normalize_to_1(result_df["Crisp Weight"].values)
            result_df["Rank"] = result_df["Normalized Weight"].rank(ascending=False, method="dense").astype(int)
            result_df = result_df.sort_values("Normalized Weight", ascending=False).reset_index(drop=True)

            st.session_state["fbwm_main_df"] = result_df.copy()

            render_kpi_row([
                {"label": "Common best", "value": factor_names[out["common_best_idx"]], "note": code_to_name[factor_names[out["common_best_idx"]]]},
                {"label": "Common worst", "value": factor_names[out["common_worst_idx"]], "note": code_to_name[factor_names[out["common_worst_idx"]]]},
                {"label": "Consistency ξ", "value": f"{out['xi']:.6f}", "note": "Aggregated LP objective value"},
            ])

            agg_best_df = pd.DataFrame({
                "Code": factor_names,
                "Criterion": [code_to_name[c] for c in factor_names],
                "Aggregated Best→Other TFN": out["agg_best"],
            })
            agg_best_df["Crisp"] = agg_best_df["Aggregated Best→Other TFN"].apply(gmi)
            show_agg_best_df = agg_best_df.copy()
            show_agg_best_df["Aggregated Best→Other TFN"] = show_agg_best_df["Aggregated Best→Other TFN"].apply(tfn_to_str)

            agg_worst_df = pd.DataFrame({
                "Code": factor_names,
                "Criterion": [code_to_name[c] for c in factor_names],
                "Aggregated Other→Worst TFN": out["agg_worst"],
            })
            agg_worst_df["Crisp"] = agg_worst_df["Aggregated Other→Worst TFN"].apply(gmi)
            show_agg_worst_df = agg_worst_df.copy()
            show_agg_worst_df["Aggregated Other→Worst TFN"] = show_agg_worst_df["Aggregated Other→Worst TFN"].apply(tfn_to_str)

            st.subheader("Input-based (IB) consistency ratio by expert")
            if pd.isna(ib_threshold_value):
                st.info("No published threshold is available for this criteria-count / scale combination.")
            else:
                acceptance_count = int((ib_summary_df["Status"] == "Acceptable").sum())
                revise_count = int((ib_summary_df["Status"] == "Revise").sum())
                render_kpi_row([
                    {"label": "IB threshold", "value": f"{ib_threshold_value:.4f}", "note": f"Criteria = {len(factor_names)}, scale = {threshold_scale}"},
                    {"label": "Acceptable experts", "value": str(acceptance_count), "note": "IB CR ≤ threshold"},
                    {"label": "Revise experts", "value": str(revise_count), "note": "IB CR > threshold"},
                ])
            show_ib_summary_df = ib_summary_df.copy()
            st.dataframe(show_ib_summary_df, use_container_width=True, hide_index=True)

            with st.expander("Show each expert input and IB consistency details", expanded=False):
                for e_idx, detail_df in enumerate(ib_detail_tables, start=1):
                    show_detail_df = detail_df.copy()
                    for col in ["Best→Other TFN", "Other→Worst TFN", "R(aBj×ajW)", "R(aBW)", "R(aBW×aBW)"]:
                        show_detail_df[col] = show_detail_df[col].apply(tfn_to_str)
                    st.markdown(f"**Expert {e_idx}**")
                    st.dataframe(show_detail_df, use_container_width=True, hide_index=True)

            st.subheader("Aggregated best-to-other vector")
            st.dataframe(show_agg_best_df, use_container_width=True, hide_index=True)

            st.subheader("Aggregated other-to-worst vector")
            st.dataframe(show_agg_worst_df, use_container_width=True, hide_index=True)

            with st.expander("Show transformed expert vectors", expanded=False):
                for e_idx, ex in enumerate(out["transformed_experts"], start=1):
                    expert_df = pd.DataFrame({
                        "Code": factor_names,
                        "Criterion": [code_to_name[c] for c in factor_names],
                        "Best→Other TFN": ex["bto_trans"],
                        "Other→Worst TFN": ex["otw_trans"],
                    })
                    expert_df["Best→Other TFN"] = expert_df["Best→Other TFN"].apply(tfn_to_str)
                    expert_df["Other→Worst TFN"] = expert_df["Other→Worst TFN"].apply(tfn_to_str)
                    st.markdown(f"**Expert {e_idx}**")
                    st.dataframe(expert_df, use_container_width=True, hide_index=True)

            show_df = result_df.copy()
            show_df["Weight TFN"] = show_df["Weight TFN"].apply(tfn_to_str)

            st.subheader("BWM weights")
            st.dataframe(show_df, use_container_width=True, hide_index=True)
            st.bar_chart(result_df.set_index("Code")[["Normalized Weight"]])

        except Exception as e:
            st.error(str(e))

# ============================================================
# MODULE 3: LBWA
# ============================================================
elif module == "3) Fuzzy LBWA":
    render_section_header(
        "Fuzzy LBWA (Main Criteria Only)",
        "Estimate main-criteria weights with Fuzzy LBWA and save the results for the hybrid integration stage.",
        eyebrow="Module 3",
    )

    n_exp_lbwa = st.number_input("Number of experts", min_value=1, value=4, step=1, key="lbwa_n_exp")
    theta_main = st.number_input(
        "Theta (θ)",
        min_value=0.0001,
        value=2.1,
        step=0.1,
        format="%.4f",
        key="lbwa_theta_main",
    )

    criteria_rows = get_main_criteria_input("lbwa", default_n=4)
    criteria_df = criteria_rows_to_df(criteria_rows)
    factor_names = criteria_df["Code"].tolist()
    code_to_name = dict(zip(criteria_df["Code"], criteria_df["Criterion"]))

    st.subheader("Criteria preview")
    st.dataframe(criteria_df, use_container_width=True, hide_index=True)

    lbwa_key = f"main_lbwa_{len(factor_names)}_{n_exp_lbwa}"
    if lbwa_key not in st.session_state:
        st.session_state[lbwa_key] = init_lbwa_editor_df(factor_names, n_exp_lbwa)

    df = st.session_state[lbwa_key].copy()
    current_exp_cols = [c for c in df.columns if c.startswith("E")]
    if list(df["Factor"]) != factor_names or len(current_exp_cols) != n_exp_lbwa:
        df = init_lbwa_editor_df(factor_names, n_exp_lbwa)

    edited = st.data_editor(
        df,
        key=f"editor_{lbwa_key}",
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "Factor": st.column_config.TextColumn("Factor", disabled=True),
            "Qi": st.column_config.NumberColumn("Qi", min_value=0.0, step=1.0, format="%.2f"),
            **{
                f"E{i+1}": st.column_config.NumberColumn(
                    f"E{i+1}",
                    min_value=0.0,
                    step=0.1,
                    format="%.4f",
                ) for i in range(n_exp_lbwa)
            },
        },
    )
    st.session_state[lbwa_key] = edited.copy()

    reference_idx = st.selectbox(
        "Reference criterion",
        options=list(range(len(factor_names))),
        format_func=lambda i: f"{i+1}. {factor_names[i]} - {code_to_name[factor_names[i]]}",
        key="lbwa_reference_idx",
    )

    if st.button("Compute Fuzzy LBWA Weights", type="primary"):
        try:
            lbwa_out = run_lbwa_excel_single_table(
                edited,
                n_exp_lbwa,
                float(theta_main),
                reference_idx,
            )

            lbwa_weights = lbwa_result_to_dict(lbwa_out)

            result_df = pd.DataFrame({
                "Code": factor_names,
                "Criterion": [code_to_name[c] for c in factor_names],
                "Weight TFN": [lbwa_weights[c] for c in factor_names],
            })
            result_df["Crisp Weight"] = result_df["Weight TFN"].apply(gmi)
            result_df["Normalized Weight"] = safe_normalize_to_1(result_df["Crisp Weight"].values)
            result_df["Rank"] = result_df["Normalized Weight"].rank(ascending=False, method="dense").astype(int)
            result_df = result_df.sort_values("Normalized Weight", ascending=False).reset_index(drop=True)

            st.session_state["flbwa_main_df"] = result_df.copy()

            st.subheader("TFN")
            st.dataframe(lbwa_out["tfn_df"], use_container_width=True, hide_index=True)

            st.subheader("Influence")
            st.dataframe(lbwa_out["influence_df"], use_container_width=True, hide_index=True)

            st.subheader("Fuzzy Weights")
            st.dataframe(lbwa_out["fuzzy_weight_df"], use_container_width=True, hide_index=True)

            st.subheader("Final Results")
            final_results_df = lbwa_out["result_df"].copy()
            final_results_df["Criterion"] = final_results_df["Factor"].map(code_to_name)
            final_results_df = final_results_df[["Rank", "Factor", "Criterion", "Qi", "Crisp Value", "Normalized Weight"]]
            st.dataframe(final_results_df, use_container_width=True, hide_index=True)

            st.subheader("LBWA Weights (Mapped to Criteria)")
            show_lbwa_df = result_df.copy()
            show_lbwa_df["Weight TFN"] = show_lbwa_df["Weight TFN"].apply(tfn_to_str)
            st.dataframe(show_lbwa_df, use_container_width=True, hide_index=True)
            st.bar_chart(result_df.set_index("Code")[["Normalized Weight"]])

            st.success("LBWA results saved. Open the Hybrid Integration module to combine them with FBWM.")

        except Exception as e:
            st.error(f"LBWA computation failed: {e}")

# ============================================================
# MODULE 4: HYBRID
# ============================================================
elif module == "4) Hybrid Integration":
    render_section_header(
        "Hybrid Integration",
        "Hybridize FBWM and FLBWA at the main-criteria and sub-criteria levels, then propagate main hybrid weights to sub-criteria using fuzzy multiplication.",
        eyebrow="Module 4",
    )

    render_scale_table(HYBRID_SCALE, HYBRID_MEANING, "Show hybrid priority code legend")

    source_mode = st.radio(
        "Main-criteria source",
        ["Use saved FBWM and FLBWA results", "Enter or edit main-criteria weights manually"],
        horizontal=True,
    )

    if source_mode == "Use saved FBWM and FLBWA results":
        saved_fbwm = st.session_state.get("fbwm_main_df")
        saved_flbwa = st.session_state.get("flbwa_main_df")
        if saved_fbwm is None and saved_flbwa is None:
            st.info("No saved FBWM or FLBWA results were found in this session. Switch to manual mode or compute the weighting modules first.")
        hybrid_editor_default = build_hybrid_editor_from_saved(saved_fbwm, saved_flbwa)
        hybrid_editor_key = "hybrid_editor_saved"
    else:
        n_hybrid = st.number_input("Number of main criteria", min_value=2, value=4, step=1, key="hybrid_n_criteria")
        hybrid_editor_default = init_manual_hybrid_editor(int(n_hybrid))
        hybrid_editor_key = f"hybrid_editor_manual_{int(n_hybrid)}"

    if hybrid_editor_key not in st.session_state:
        st.session_state[hybrid_editor_key] = hybrid_editor_default

    editor_df = st.session_state[hybrid_editor_key].copy()
    if source_mode == "Use saved FBWM and FLBWA results":
        editor_df = hybrid_editor_default

    st.markdown("**Main-criteria FBWM and FLBWA weights**")
    edited_hybrid = st.data_editor(
        editor_df,
        key=f"editor_{hybrid_editor_key}",
        use_container_width=True,
        num_rows="dynamic" if source_mode != "Use saved FBWM and FLBWA results" else "fixed",
        hide_index=True,
        column_config={
            "Code": st.column_config.TextColumn("Code", required=True),
            "Criterion": st.column_config.TextColumn("Criterion", required=True),
            "FBWM_l": st.column_config.NumberColumn("FBWM l", min_value=0.0, step=0.0001, format="%.6f"),
            "FBWM_m": st.column_config.NumberColumn("FBWM m", min_value=0.0, step=0.0001, format="%.6f"),
            "FBWM_u": st.column_config.NumberColumn("FBWM u", min_value=0.0, step=0.0001, format="%.6f"),
            "FLBWA_l": st.column_config.NumberColumn("FLBWA l", min_value=0.0, step=0.0001, format="%.6f"),
            "FLBWA_m": st.column_config.NumberColumn("FLBWA m", min_value=0.0, step=0.0001, format="%.6f"),
            "FLBWA_u": st.column_config.NumberColumn("FLBWA u", min_value=0.0, step=0.0001, format="%.6f"),
            "Rank_FBWM": st.column_config.NumberColumn("Rank FBWM", disabled=True, format="%.0f"),
            "Rank_FLBWA": st.column_config.NumberColumn("Rank FLBWA", disabled=True, format="%.0f"),
        },
    )
    st.session_state[hybrid_editor_key] = edited_hybrid.copy()

    main_codes = [str(c).strip() for c in edited_hybrid["Code"].tolist() if str(c).strip()]
    validate_unique_nonempty(main_codes, "Main-criteria codes")

    st.markdown("**Sub-criteria structure**")
    sub_count_default = sync_subcriteria_count_df(main_codes, st.session_state.get("hybrid_sub_counts"))
    edited_sub_counts = st.data_editor(
        sub_count_default,
        key="hybrid_sub_counts_editor",
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "ParentCode": st.column_config.TextColumn("Main code", disabled=True),
            "SubCount": st.column_config.NumberColumn("No. of sub-criteria", min_value=1, step=1, format="%.0f"),
        },
    )
    edited_sub_counts["SubCount"] = edited_sub_counts["SubCount"].fillna(1).astype(int).clip(lower=1)
    st.session_state["hybrid_sub_counts"] = edited_sub_counts.copy()

    sub_editor_default = build_subcriteria_editor(edited_sub_counts, st.session_state.get("hybrid_sub_editor"))
    st.markdown("**Sub-criteria local FBWM and FLBWA weights**")
    edited_sub = st.data_editor(
        sub_editor_default,
        key="hybrid_sub_editor_widget",
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "ParentCode": st.column_config.TextColumn("Main code", disabled=True),
            "Code": st.column_config.TextColumn("Sub-code", disabled=True),
            "FBWM_l": st.column_config.NumberColumn("FBWM l", min_value=0.0, step=0.0001, format="%.6f"),
            "FBWM_m": st.column_config.NumberColumn("FBWM m", min_value=0.0, step=0.0001, format="%.6f"),
            "FBWM_u": st.column_config.NumberColumn("FBWM u", min_value=0.0, step=0.0001, format="%.6f"),
            "FLBWA_l": st.column_config.NumberColumn("FLBWA l", min_value=0.0, step=0.0001, format="%.6f"),
            "FLBWA_m": st.column_config.NumberColumn("FLBWA m", min_value=0.0, step=0.0001, format="%.6f"),
            "FLBWA_u": st.column_config.NumberColumn("FLBWA u", min_value=0.0, step=0.0001, format="%.6f"),
        },
    )
    st.session_state["hybrid_sub_editor"] = edited_sub.copy()

    priority_mode = st.radio(
        "Model priority input",
        ["Use linguistic priority codes", "Enter direct TFN priorities"],
        horizontal=True,
    )

    if priority_mode == "Use linguistic priority codes":
        c1, c2 = st.columns(2)
        with c1:
            alpha_code = st.selectbox("Priority of FBWM model (α)", options=list(HYBRID_SCALE.keys()), index=3)
        with c2:
            beta_code = st.selectbox("Priority of FLBWA model (β)", options=list(HYBRID_SCALE.keys()), index=3)
        alpha_tfn = HYBRID_SCALE[alpha_code]
        beta_tfn = HYBRID_SCALE[beta_code]
    else:
        st.caption("Enter the fuzzy priority triplets used to weight the FBWM and FLBWA models in the hybrid integration.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**FBWM priority (α)**")
            alpha_l = st.number_input("α_l", min_value=0.0, value=0.34, step=0.01, format="%.4f")
            alpha_m = st.number_input("α_m", min_value=0.0, value=0.50, step=0.01, format="%.4f")
            alpha_u = st.number_input("α_u", min_value=0.0, value=0.66, step=0.01, format="%.4f")
        with c2:
            st.markdown("**FLBWA priority (β)**")
            beta_l = st.number_input("β_l", min_value=0.0, value=0.34, step=0.01, format="%.4f")
            beta_m = st.number_input("β_m", min_value=0.0, value=0.50, step=0.01, format="%.4f")
            beta_u = st.number_input("β_u", min_value=0.0, value=0.66, step=0.01, format="%.4f")
        alpha_tfn = (float(alpha_l), float(alpha_m), float(alpha_u))
        beta_tfn = (float(beta_l), float(beta_m), float(beta_u))
        if not (alpha_tfn[0] <= alpha_tfn[1] <= alpha_tfn[2] and beta_tfn[0] <= beta_tfn[1] <= beta_tfn[2]):
            st.warning("Priority TFNs should satisfy l ≤ m ≤ u.")

    if st.button("Compute Hierarchical Hybrid Weights", type="primary"):
        try:
            fbwm_model_df = hybrid_editor_to_model_df(edited_hybrid, "FBWM")
            flbwa_model_df = hybrid_editor_to_model_df(edited_hybrid, "FLBWA")
            hybrid_df = combine_main_weight_tables(
                fbwm_model_df,
                flbwa_model_df,
                alpha_tfn,
                beta_tfn,
            )
            st.session_state["hybrid_main_df"] = hybrid_df.copy()

            sub_work = edited_sub.copy()
            sub_work["Criterion"] = sub_work["Code"]
            fbwm_sub_df = hybrid_editor_to_model_df(sub_work, "FBWM")
            flbwa_sub_df = hybrid_editor_to_model_df(sub_work, "FLBWA")
            sub_local_df = combine_sub_weight_tables(
                fbwm_sub_df,
                flbwa_sub_df,
                alpha_tfn,
                beta_tfn,
            )
            st.session_state["hybrid_sub_local_df"] = sub_local_df.copy()

            sub_global_df = build_hierarchical_hybrid_tables(hybrid_df, sub_local_df)
            st.session_state["hybrid_sub_global_df"] = sub_global_df.copy()

            render_kpi_row([
                {"label": "Main criteria", "value": str(len(hybrid_df)), "note": "Rows in the main hybrid table"},
                {"label": "Sub-criteria", "value": str(len(sub_global_df)), "note": "Rows in the final global table"},
                {"label": "Top global code", "value": str(sub_global_df.iloc[0]["Code"]), "note": f"Normalized value = {sub_global_df.iloc[0]['Normalized Value']:.4f}"},
            ])

            st.subheader("Main hybrid weights")
            show_main_df = hybrid_df.copy()
            for col in ["FBWM TFN", "FLBWA TFN", "Hybrid Numerator TFN", "Hybrid Denominator TFN", "Hybrid TFN"]:
                if col in show_main_df.columns:
                    show_main_df[col] = show_main_df[col].apply(tfn_to_str)
            main_cols = [
                "Code",
                "Criterion",
                "FBWM TFN",
                "Rank_FBWM",
                "FLBWA TFN",
                "Rank_FLBWA",
                "Hybrid Numerator TFN",
                "Hybrid Denominator TFN",
                "Hybrid TFN",
                "Exact Crisp Value",
                "Normalized Value",
                "Rank",
            ]
            st.dataframe(show_main_df[[c for c in main_cols if c in show_main_df.columns]], use_container_width=True, hide_index=True)

            st.subheader("Sub-criteria hybrid weights (Excel-aligned local stage)")
            show_sub_local_df = sub_local_df.copy()
            for col in ["FBWM TFN", "FLBWA TFN", "Hybrid Numerator TFN", "Hybrid Denominator TFN", "Local Hybrid TFN"]:
                if col in show_sub_local_df.columns:
                    show_sub_local_df[col] = show_sub_local_df[col].apply(tfn_to_str)
            sub_local_cols = [
                "ParentCode",
                "Code",
                "FBWM TFN",
                "Rank_FBWM",
                "FLBWA TFN",
                "Rank_FLBWA",
                "Hybrid Numerator TFN",
                "Hybrid Denominator TFN",
                "Local Hybrid TFN",
                "Local Exact Crisp Value",
                "Local Normalized Value",
                "Local Rank",
            ]
            st.dataframe(show_sub_local_df[[c for c in sub_local_cols if c in show_sub_local_df.columns]], use_container_width=True, hide_index=True)

            st.subheader("Final global sub-criteria weights")
            show_sub_global_df = sub_global_df.copy()
            for col in ["Main Hybrid TFN", "Local Hybrid TFN", "Global Hybrid TFN"]:
                if col in show_sub_global_df.columns:
                    show_sub_global_df[col] = show_sub_global_df[col].apply(tfn_to_str)
            sub_global_cols = [
                "ParentCode",
                "Code",
                "Main Hybrid TFN",
                "Local Hybrid TFN",
                "Global Hybrid TFN",
                "Exact Crisp Value",
                "Normalized Value",
                "Rank",
            ]
            st.dataframe(show_sub_global_df[[c for c in sub_global_cols if c in show_sub_global_df.columns]], use_container_width=True, hide_index=True)
            st.bar_chart(sub_global_df.set_index("Code")[["Normalized Value"]])

        except Exception as e:
            st.error(f"Hybrid computation failed: {e}")

# ============================================================
# MODULE 5: COCO-SO
# ============================================================
elif module == "5) Fuzzy Bonferroni CoCoSo":
    render_section_header(
        "Fuzzy Bonferroni CoCoSo – Alternative Ranking",
        "Rank alternatives using fuzzy performance inputs, criterion types, and optional saved hybrid weights.",
        eyebrow="Module 5",
    )

    n_criteria = st.number_input(
        "Number of criteria",
        min_value=2,
        value=4,
        step=1,
        key="cocoso_n_criteria"
    )

    n_exp_cocoso = st.number_input(
        "Number of experts for linguistic criteria",
        min_value=1,
        value=3,
        step=1,
        key="cocoso_n_exp"
    )

    sigma_pct = st.number_input(
        "Uncertainty level σ (%) for crisp data",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        key="cocoso_sigma_pct"
    )
    sigma = float(sigma_pct) / 100.0

    render_cocoso_linguistic_scale()

    criteria_rows = []
    for i in range(n_criteria):
        c1, c2, c3, c4 = st.columns([1.0, 2.4, 1.0, 1.2])
        with c1:
            code = st.text_input(
                f"Criterion code {i+1}",
                value=f"C{i+1}",
                key=f"cocoso_code_{i}"
            ).strip() or f"C{i+1}"
        with c2:
            name = st.text_input(
                f"Criterion name {i+1}",
                value=f"Criterion {i+1}",
                key=f"cocoso_name_{i}"
            ).strip() or f"Criterion {i+1}"
        with c3:
            typ = st.selectbox(
                f"Type {i+1}",
                ["benefit", "cost"],
                key=f"cocoso_type_{i}"
            )
        with c4:
            mode = st.selectbox(
                f"Input mode {i+1}",
                ["Crisp", "Linguistic"],
                key=f"cocoso_mode_{i}"
            )

        criteria_rows.append({
            "Code": code,
            "Criterion": name,
            "Type": typ,
            "Input Mode": mode,
        })

    validate_unique_nonempty([r["Code"] for r in criteria_rows], "Criterion codes")

    use_saved_weights = st.checkbox("Use saved hybrid weights if available", value=False)
    if use_saved_weights:
        source_weight_df = st.session_state.get("hybrid_sub_global_df")
        if source_weight_df is None or getattr(source_weight_df, "empty", True):
            source_weight_df = st.session_state.get("hybrid_main_df")
    else:
        source_weight_df = None
    criteria_df = init_cocoso_criteria_weight_df(criteria_rows, source_weight_df)

    criteria_df["Input Mode"] = [r["Input Mode"] for r in criteria_rows]

    edited_criteria = st.data_editor(
        criteria_df,
        key="cocoso_criteria_independent",
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "Code": st.column_config.TextColumn("Code", disabled=True),
            "Criterion": st.column_config.TextColumn("Criterion", disabled=True),
            "w_l": st.column_config.NumberColumn("w_l", min_value=0.0, format="%.6f"),
            "w_m": st.column_config.NumberColumn("w_m", min_value=0.0, format="%.6f"),
            "w_u": st.column_config.NumberColumn("w_u", min_value=0.0, format="%.6f"),
            "Type": st.column_config.SelectboxColumn("Type", options=["benefit", "cost"], required=True),
            "Input Mode": st.column_config.TextColumn("Input Mode", disabled=True),
        },
    )

    criteria_names = edited_criteria["Code"].tolist()
    if len(criteria_names) < 2:
        st.error("At least two criteria are required for CoCoSo.")
        st.stop()

    alt_names = [
        x.strip() for x in st.text_area(
            "Alternative names (one per line)",
            value="A1\nA2\nA3\nA4",
            height=120
        ).splitlines() if x.strip()
    ]

    if not alt_names:
        st.warning("Please define at least one alternative.")
        st.stop()

    crisp_criteria = edited_criteria.loc[edited_criteria["Input Mode"] == "Crisp", "Code"].tolist()
    linguistic_criteria = edited_criteria.loc[edited_criteria["Input Mode"] == "Linguistic", "Code"].tolist()

    st.subheader("Alternative evaluations")

    crisp_tables_by_alt = {}
    linguistic_tables_by_alt = {}

    alt_tabs = st.tabs(alt_names)
    for idx, alt in enumerate(alt_tabs):
        alt_name = alt_names[idx]
        with alt:
            if crisp_criteria:
                st.markdown("**Crisp criteria**")
                crisp_key = f"cocoso_crisp_{idx}_{len(crisp_criteria)}"
                if crisp_key not in st.session_state:
                    st.session_state[crisp_key] = init_crisp_alt_df(crisp_criteria)

                crisp_df = st.session_state[crisp_key].copy()
                crisp_df["Criterion"] = crisp_criteria

                crisp_edited = st.data_editor(
                    crisp_df,
                    key=f"editor_{crisp_key}",
                    use_container_width=True,
                    num_rows="fixed",
                    hide_index=True,
                    column_config={
                        "Criterion": st.column_config.TextColumn("Criterion", disabled=True),
                        "Value": st.column_config.NumberColumn("Value", step=0.01, format="%.6f"),
                    },
                )
                st.session_state[crisp_key] = crisp_edited.copy()
                crisp_tables_by_alt[alt_name] = crisp_edited.copy()

            if linguistic_criteria:
                st.markdown("**Linguistic criteria**")
                ling_key = f"cocoso_ling_{idx}_{len(linguistic_criteria)}_{n_exp_cocoso}"
                if ling_key not in st.session_state:
                    st.session_state[ling_key] = init_linguistic_alt_df(linguistic_criteria, n_exp_cocoso)

                ling_df = st.session_state[ling_key].copy()
                ling_df["Criterion"] = linguistic_criteria

                current_exp_cols = [c for c in ling_df.columns if c.startswith("E")]
                if len(current_exp_cols) != n_exp_cocoso or list(ling_df["Criterion"]) != linguistic_criteria:
                    ling_df = init_linguistic_alt_df(linguistic_criteria, n_exp_cocoso)

                ling_edited = st.data_editor(
                    ling_df,
                    key=f"editor_{ling_key}",
                    use_container_width=True,
                    num_rows="fixed",
                    hide_index=True,
                    column_config={
                        "Criterion": st.column_config.TextColumn("Criterion", disabled=True),
                        **{
                            f"E{i+1}": st.column_config.SelectboxColumn(
                                f"E{i+1}",
                                options=list(COCOSO_LINGUISTIC_SCALE.keys()),
                                required=True
                            ) for i in range(n_exp_cocoso)
                        },
                    },
                )
                st.session_state[ling_key] = ling_edited.copy()
                linguistic_tables_by_alt[alt_name] = ling_edited.copy()

    p1, p2, p3 = st.columns(3)
    with p1:
        phi1 = st.number_input("ϕ1", value=1.0, step=0.1)
    with p2:
        phi2 = st.number_input("ϕ2", value=1.0, step=0.1)
    with p3:
        pi = st.number_input("π", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    if st.button("Run Fuzzy Bonferroni CoCoSo", type="primary"):
        try:
            if phi1 + phi2 <= 0:
                st.error("ϕ1 + ϕ2 must be greater than 0.")
                st.stop()

            fuzzy_weights = list(zip(
                edited_criteria["w_l"].astype(float),
                edited_criteria["w_m"].astype(float),
                edited_criteria["w_u"].astype(float)
            ))
            final_weights = [defuzz_tfn(w) for w in fuzzy_weights]
            types = [to_bc_label(t) for t in edited_criteria["Type"].astype(str).tolist()]

            fuzzy_df = build_mixed_fuzzy_df_from_inputs(
                alt_names=alt_names,
                criteria_meta=edited_criteria[["Code", "Criterion", "Type", "Input Mode"]].to_dict("records"),
                crisp_tables_by_alt=crisp_tables_by_alt,
                linguistic_tables_by_alt=linguistic_tables_by_alt,
                sigma=sigma
            )

            ranking_df, params_df, norm_df, scob_df, pcob_df, psi_a_df, psi_b_df, psi_c_df = cocoso_bonferroni_from_app(
                fuzzy_df,
                types,
                final_weights,
                float(phi1),
                float(phi2),
                float(pi)
            )

            st.subheader("Aggregated fuzzy decision matrix")
            st.dataframe(fuzzy_df.reset_index(), use_container_width=True, hide_index=True)

            st.subheader("Normalized fuzzy decision matrix")
            st.dataframe(norm_df, use_container_width=True, hide_index=True)

            st.subheader("Weighted Bonferroni sequences")

            st.subheader("SCoB")
            st.dataframe(scob_df, use_container_width=True, hide_index=True)

            st.subheader("PCoB")
            st.dataframe(pcob_df, use_container_width=True, hide_index=True)

            st.subheader("Relative significance from defuzzified SCoB and PCoB")

            st.subheader("Kia")
            st.dataframe(psi_a_df, use_container_width=True, hide_index=True)

            st.subheader("Kib")
            st.dataframe(psi_b_df, use_container_width=True, hide_index=True)

            st.subheader("Kic")
            st.dataframe(psi_c_df, use_container_width=True, hide_index=True)

            st.subheader("Final ranking")
            st.dataframe(
                ranking_df[["Rank", "Alternative", "Final_l", "Final_m", "Final_u", "Crisp"]],
                use_container_width=True,
                hide_index=True
            )
            st.bar_chart(ranking_df.set_index("Alternative")[["Crisp"]])

        except Exception as e:
            st.error(f"CoCoSo computation failed: {e}")

render_footer()
