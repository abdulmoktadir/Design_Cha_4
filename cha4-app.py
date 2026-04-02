import base64
import hmac
import io
import math
import zipfile
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import graphviz
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Protected Decision Support Research Studio",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root {
    --bg: #f6f3fb;
    --bg2: #eef8f6;
    --surface: rgba(255,255,255,0.94);
    --surface-strong: #ffffff;
    --border: #e2daf3;
    --text: #231942;
    --muted: #716b88;
    --primary: #6d28d9;
    --primary-strong: #4c1d95;
    --accent: #0f766e;
    --shadow: 0 18px 44px rgba(35, 25, 66, 0.10);
    --shadow-soft: 0 10px 24px rgba(35, 25, 66, 0.06);
}
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }
.stApp {
    background:
        radial-gradient(circle at top left, rgba(109,40,217,0.10), transparent 26%),
        radial-gradient(circle at top right, rgba(15,118,110,0.09), transparent 24%),
        linear-gradient(180deg, var(--bg) 0%, #f9f7fc 52%, var(--bg2) 100%);
    color: var(--text);
}
.block-container { max-width: 1480px; padding-top: 1.1rem; padding-bottom: 2.5rem; }
[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at top, rgba(167,139,250,0.16) 0%, rgba(167,139,250,0.02) 30%),
        linear-gradient(180deg, #1c1533 0%, #17253f 52%, #12343c 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #edf3ff; }
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] .stNumberInput input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #f9fbff !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 0.52rem 0.78rem;
    margin-bottom: 0.48rem;
}
.hero-card {
    background:
        radial-gradient(circle at top right, rgba(167,139,250,0.22) 0%, rgba(167,139,250,0.02) 26%),
        linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(248,245,255,0.98) 54%, rgba(239,251,248,0.96) 100%);
    border: 1px solid rgba(76,29,149,0.10);
    border-radius: 26px;
    padding: 1.55rem 1.65rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.15rem;
}
.hero-eyebrow, .module-badge, .sidebar-section-title, .info-label {
    text-transform: uppercase; letter-spacing: 0.09em; font-size: 0.74rem; font-weight: 800;
}
.hero-eyebrow { color: var(--primary-strong); }
.hero-title { margin-top: 0.2rem; font-size: 2.15rem; font-weight: 850; line-height: 1.05; letter-spacing: -0.035em; color: var(--text); }
.hero-subtitle { margin-top: 0.55rem; max-width: 980px; font-size: 1rem; line-height: 1.7; color: var(--muted); }
.hero-grid { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 0.9rem; margin-top: 1.15rem; }
.info-card, .module-banner { background: var(--surface-strong); border: 1px solid var(--border); border-radius: 18px; box-shadow: var(--shadow-soft); }
.info-card { padding: 0.95rem 1rem; }
.info-card strong { display:block; margin-top:0.32rem; color: var(--text); font-size:1rem; }
.info-card small { display:block; margin-top:0.18rem; color: var(--muted); line-height:1.55; }
.module-banner {
    position: relative; overflow:hidden; padding: 1rem 1.1rem; margin-bottom: 1rem;
    background:
        radial-gradient(circle at 100% 0%, rgba(109,40,217,0.12) 0%, rgba(109,40,217,0.02) 28%),
        linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(250,246,255,0.98) 56%, rgba(244,252,250,0.98) 100%);
}
.module-badge {
    display:inline-flex; align-items:center; gap:0.35rem; color: var(--primary-strong);
    background: rgba(109,40,217,0.08); border:1px solid rgba(109,40,217,0.14); border-radius:999px; padding:0.3rem 0.62rem;
}
.module-title { margin-top:0.5rem; font-size:1.55rem; font-weight:820; line-height:1.15; color: var(--text); }
.module-subtitle { margin-top:0.28rem; max-width:980px; color: var(--muted); line-height:1.65; }
.stButton > button {
    width:100%; border:0; border-radius:14px; padding:0.72rem 1.1rem; font-weight:750; color:white;
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    box-shadow: 0 12px 26px rgba(109,40,217,0.22);
}
button[data-baseweb="tab"] { border-radius:999px; border:1px solid var(--border); background: rgba(255,255,255,0.84); height:42px; padding:0 1rem; }
button[data-baseweb="tab"][aria-selected="true"] { background: linear-gradient(135deg, rgba(109,40,217,0.11), rgba(15,118,110,0.13)); border-color: rgba(109,40,217,0.26); color: var(--primary-strong); font-weight:750; }
[data-baseweb="tab-list"] { gap:0.55rem; padding-bottom:0.45rem; }
[data-testid="metric-container"] { background: rgba(255,255,255,0.9); padding:1rem; border-radius:18px; border:1px solid var(--border); box-shadow: var(--shadow-soft); }
[data-testid="stMetricValue"] { color: var(--text); }
div[data-testid="stDataFrame"], div[data-testid="stTable"] { border:1px solid var(--border); border-radius:18px; overflow:hidden; background:rgba(255,255,255,0.92); box-shadow: var(--shadow-soft); }
div[data-testid="stExpander"] { border:1px solid var(--border); border-radius:18px; background:rgba(255,255,255,0.80); box-shadow: var(--shadow-soft); }
textarea, div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, .stNumberInput input { border-radius:14px !important; }
.sidebar-section-title { color:#f6f3ff !important; margin:0.2rem 0 0.7rem 0; }
.sidebar-section-note { color:#bfd0ef !important; font-size:0.79rem; line-height:1.6; margin:0 0 0.85rem 0; }
.sidebar-profile-card { background: linear-gradient(180deg, rgba(33,24,58,0.98) 0%, rgba(21,47,68,0.98) 100%); border:1px solid rgba(200,193,255,0.16); border-radius:18px; padding:1rem; margin:0.1rem 0 0.9rem 0; box-shadow:0 14px 28px rgba(2,8,23,0.26); }
.sidebar-profile-badge { display:inline-flex; align-items:center; gap:0.35rem; color:#efe9ff !important; background:rgba(167,139,250,0.18); border:1px solid rgba(196,181,253,0.24); border-radius:999px; padding:0.26rem 0.58rem; font-size:0.68rem; font-weight:700; letter-spacing:0.05em; text-transform:uppercase; margin-bottom:0.8rem; }
.sidebar-profile-image-frame { width:100%; border-radius:16px; overflow:hidden; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.10); margin-bottom:0.85rem; }
.sidebar-profile-image { display:block; width:100%; height:auto; }
.sidebar-profile-image-placeholder { padding:1.05rem 0.85rem; text-align:center; color:#e8e6ff !important; font-size:0.78rem; }
.sidebar-profile-name { color:#ffffff !important; font-size:1.08rem; font-weight:820; margin:0 0 0.18rem 0; line-height:1.35; }
.sidebar-profile-role { color:#d9e8ff !important; font-size:0.84rem; font-weight:700; line-height:1.45; margin-bottom:0.16rem; }
.sidebar-profile-institution { color:#ecf3ff !important; font-size:0.8rem; line-height:1.45; margin-bottom:0.62rem; }
.sidebar-profile-text, .sidebar-profile-bullet, .sidebar-profile-bio { color:#dbe8f8 !important; font-size:0.8rem; line-height:1.66; }
.sidebar-profile-divider { border-top:1px solid rgba(255,255,255,0.08); margin:0.72rem 0 0.68rem 0; }
.sidebar-bio-details { margin:-0.2rem 0 1rem 0; border:1px solid rgba(196,181,253,0.18); border-radius:16px; overflow:hidden; background:rgba(17,24,39,0.22); box-shadow:0 12px 22px rgba(2,8,23,0.18); }
.sidebar-bio-summary { list-style:none; cursor:pointer; padding:0.8rem 0.92rem; background:linear-gradient(135deg, rgba(109,40,217,0.26) 0%, rgba(15,118,110,0.22) 100%); color:#f6f1ff !important; font-size:0.83rem; font-weight:760; border:none; }
.sidebar-bio-summary::-webkit-details-marker { display:none; }
.sidebar-bio-summary::before { content:"▸"; display:inline-block; margin-right:0.45rem; color:#ddd6fe; transition:transform 0.2s ease; }
.sidebar-bio-details[open] .sidebar-bio-summary::before { transform:rotate(90deg); }
.sidebar-bio-panel { padding:0.9rem 0.95rem 0.95rem 0.95rem; background:linear-gradient(180deg, rgba(18,28,45,0.96) 0%, rgba(17,43,60,0.96) 100%); color:#e9f2ff !important; font-size:0.8rem; line-height:1.72; border-top:1px solid rgba(255,255,255,0.08); }
.config-shell { background:linear-gradient(180deg, rgba(255,255,255,0.94) 0%, rgba(249,246,255,0.98) 100%); border:1px solid var(--border); border-radius:18px; padding:1rem 1rem 0.7rem 1rem; margin:0 0 1rem 0; box-shadow:var(--shadow-soft); }
.config-lead { margin:0 0 0.9rem 0; color:var(--muted); font-size:0.9rem; line-height:1.6; }
.login-page { position:relative; max-width:960px; margin:1.7rem auto 0 auto; padding:0 0.9rem 1.25rem 0.9rem; }
.login-hero-box {
    position:relative; z-index:1;
    background:
        radial-gradient(circle at 84% 16%, rgba(255,255,255,0.20) 0%, rgba(255,255,255,0.02) 26%),
        radial-gradient(circle at 12% 108%, rgba(45,212,191,0.18) 0%, rgba(45,212,191,0.02) 28%),
        linear-gradient(135deg, #2a145f 0%, #6d28d9 48%, #0f766e 100%);
    border-radius:34px; padding:1.2rem; margin-bottom:1.15rem; box-shadow:0 28px 70px rgba(34,17,64,0.18), 0 12px 26px rgba(15,118,110,0.16); border:1px solid rgba(255,255,255,0.16); overflow:hidden;
}
.login-hero-inner { position:relative; z-index:1; min-height:315px; border-radius:26px; padding:2.15rem 2.05rem 1.95rem 2.05rem; border:1px solid rgba(255,255,255,0.08); background: linear-gradient(180deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%); backdrop-filter: blur(10px); }
.login-badge { display:inline-flex; align-items:center; gap:0.4rem; padding:0.42rem 0.85rem; border-radius:999px; background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.14); color:rgba(255,255,255,0.96); font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; }
.login-hero-box h1 { color:#ffffff !important; font-size:clamp(2.5rem, 4.8vw, 3.9rem); line-height:1.02; margin:1rem 0 0.95rem 0; letter-spacing:-0.055em; max-width:760px; }
.login-hero-box p { color:rgba(255,255,255,0.93) !important; margin:0; max-width:700px; font-size:1.08rem; line-height:1.72; }
.login-pill-row { display:flex; flex-wrap:wrap; gap:0.65rem; margin-top:1.15rem; }
.login-pill { display:inline-flex; align-items:center; padding:0.48rem 0.8rem; border-radius:999px; font-size:0.82rem; font-weight:650; color:#eff6ff; background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.14); }
.login-metrics { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:0.85rem; margin-top:1.35rem; }
.login-metric-card { padding:0.92rem 0.95rem; border-radius:18px; background:rgba(255,255,255,0.10); border:1px solid rgba(255,255,255,0.12); }
.login-metric-label { display:block; color:rgba(255,255,255,0.72); font-size:0.73rem; font-weight:700; letter-spacing:0.07em; text-transform:uppercase; }
.login-metric-value { display:block; margin-top:0.35rem; color:#ffffff; font-size:1.08rem; font-weight:800; }
.login-metric-note { display:block; margin-top:0.25rem; color:rgba(255,255,255,0.74); font-size:0.82rem; line-height:1.45; }
.login-form-note { text-align:center; color:var(--text); font-size:1rem; font-weight:650; margin:0 0 1rem 0; }
div[data-testid="stForm"] { background:linear-gradient(180deg, rgba(255,255,255,0.97) 0%, rgba(252,250,255,0.99) 100%); border:1px solid rgba(225,217,241,0.92); border-radius:28px; padding:1.55rem 1.55rem 1.18rem 1.55rem; box-shadow:0 24px 48px rgba(34,17,64,0.10), 0 12px 26px rgba(109,40,217,0.08); max-width:920px; margin:0 auto; }
div[data-testid="stForm"] label p { font-size:1.05rem !important; font-weight:700 !important; color:var(--text) !important; }
div[data-testid="stForm"] .stTextInput input { min-height:60px; font-size:1rem; border-radius:16px; padding:0.92rem 1rem; border:1px solid #ddd6f3; background:rgba(251,249,255,0.98); }
.login-helper { display:flex; justify-content:center; margin-top:0.75rem; }
.login-helper span { display:inline-flex; align-items:center; gap:0.4rem; border-radius:999px; padding:0.42rem 0.8rem; background:rgba(255,255,255,0.78); color:var(--muted); font-size:0.8rem; border:1px solid rgba(225,217,241,0.92); }
.app-footer { text-align:center; margin-top:2.6rem; padding:1rem 0 0.25rem 0; color:var(--muted); font-size:0.78rem; border-top:1px solid rgba(35,25,66,0.08); }
.copyright-pill { display:inline-block; margin-bottom:0.55rem; padding:0.38rem 0.9rem; border-radius:999px; border:1px solid #dfd7f2; background:#ffffff; color:var(--text); font-weight:750; box-shadow:0 8px 22px rgba(34,17,64,0.05); }
@media (max-width: 1200px) { .hero-grid { grid-template-columns:1fr; } }
@media (max-width: 900px) { .login-metrics { grid-template-columns:1fr; } .login-hero-box h1 { font-size:2.18rem; } }
</style>
"""


def apply_custom_styling():
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
    _, center, _ = st.columns([0.28, 4.3, 0.28])
    with center:
        st.markdown(
            """
            <div class="login-page">
                <div class="login-hero-box">
                    <div class="login-hero-inner">
                        <div class="login-badge">Protected research login</div>
                        <h1>Integrated Decision Research Studio</h1>
                        <p>Enter the secured workspace for stratification modeling, expert-weight analytics, DFS-AHP, DFS-QFD, and MILP-based portfolio optimization in one deployment-ready system.</p>
                        <div class="login-pill-row">
                            <span class="login-pill">Stratification</span>
                            <span class="login-pill">Expert Weights</span>
                            <span class="login-pill">DFS-AHP</span>
                            <span class="login-pill">DFS-QFD</span>
                            <span class="login-pill">MILP</span>
                        </div>
                        <div class="login-metrics">
                            <div class="login-metric-card">
                                <span class="login-metric-label">Security</span>
                                <span class="login-metric-value">Password protected access</span>
                                <span class="login-metric-note">Safer deployment for shared research environments.</span>
                            </div>
                            <div class="login-metric-card">
                                <span class="login-metric-label">Workspace</span>
                                <span class="login-metric-value">5 linked analytical modules</span>
                                <span class="login-metric-note">From scenario logic to final strategy selection.</span>
                            </div>
                            <div class="login-metric-card">
                                <span class="login-metric-label">Outputs</span>
                                <span class="login-metric-value">Decision-ready exports</span>
                                <span class="login-metric-note">Tables, rankings, matrices, and optimization summaries.</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="login-form-note">Sign in with the application password to open the workspace</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if expected_password is None:
            st.error("APP_PASSWORD is not configured. Add it in .streamlit/secrets.toml or Streamlit Cloud Settings → Secrets.")
            return False

        with st.form("login_form", clear_on_submit=False):
            password = st.text_input("Password", type="password", placeholder="Enter application password")
            submitted = st.form_submit_button("Log in", use_container_width=True)

        st.markdown(
            '<div class="login-helper"><span>🔐 Researcher profiles remain in the left sidebar after sign-in</span></div>',
            unsafe_allow_html=True,
        )

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


def get_image_data_uri(image_path: Path) -> Optional[str]:
    image_path = Path(image_path)
    if not image_path.exists():
        return None
    suffix = image_path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(suffix, "image/png")
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def render_sidebar_profile_card(container, name, role, institution, image_path, brief_text, full_bio=None, extras=None, tag="Researcher"):
    image_uri = get_image_data_uri(Path(image_path))
    image_html = f'<img class="sidebar-profile-image" src="{image_uri}" alt="{escape(name)}">' if image_uri else '<div class="sidebar-profile-image-placeholder">Profile image not available</div>'
    extras_html = ""
    if extras:
        extras_html = '<div class="sidebar-profile-divider"></div>' + ''.join(f'<div class="sidebar-profile-bullet">• {escape(item)}</div>' for item in extras)
    container.markdown(
        f"""
        <div class="sidebar-profile-card">
            <div class="sidebar-profile-badge">👤 {escape(tag)}</div>
            <div class="sidebar-profile-image-frame">{image_html}</div>
            <div class="sidebar-profile-name">{escape(name)}</div>
            <div class="sidebar-profile-role">{escape(role)}</div>
            <div class="sidebar-profile-institution">{escape(institution)}</div>
            <div class="sidebar-profile-text">{escape(brief_text)}</div>
            {extras_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if full_bio:
        bio_html = escape(full_bio).replace("\n", "<br>")
        container.markdown(
            f"""
            <details class="sidebar-bio-details">
                <summary class="sidebar-bio-summary">More about {escape(name)}</summary>
                <div class="sidebar-bio-panel">{bio_html}</div>
            </details>
            """,
            unsafe_allow_html=True,
        )

def render_sidebar_research_profiles():
    box = st.sidebar.container()
    box.markdown("---")
    box.markdown('<div class="sidebar-section-title">Researcher Profiles</div>', unsafe_allow_html=True)
    box.markdown('<div class="sidebar-section-note">Profiles remain fixed in the left sidebar for deployment-ready presentation.</div>', unsafe_allow_html=True)
    render_sidebar_profile_card(
        box,
        'Prof. J.Z. Ren 任競爭',
        'Associate Professor',
        'The Hong Kong Polytechnic University',
        get_asset_path('prof_jz_ren.png'),
        'Process systems engineering for energy, environment, and sustainability; recipient of the 2022 APEC ASPIRE Prize.',
        full_bio='Dr. Jingzheng Ren is an Associate Professor at The Hong Kong Polytechnic University. His research focuses on process systems engineering for energy, environment and sustainability, including innovative industrial processes, decision tools, and optimization models for carbon-neutral industrial systems.',
        tag='Lead Researcher',
    )
    render_sidebar_profile_card(
        box,
        'Md. Abdul Moktadir',
        'Assistant Professor (Leather Products Engineering)',
        'University of Dhaka / PolyU Presidential PhD Fellow',
        get_asset_path('abdul_moktadir.png'),
        'Research interests include sustainable supply chains, logistics, risk management, Industry 4.0, and circular economy.',
        full_bio='Md. Abdul Moktadir is pursuing a PhD in Industrial and Systems Engineering at The Hong Kong Polytechnic University and serves as an Assistant Professor of Leather Products Engineering at the University of Dhaka.',
        extras=['Affiliation: University of Dhaka', 'Program: PolyU Presidential PhD Fellow'],
        tag='Co-Researcher',
    )


def render_app_header():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Protected multi-module research workspace</div>
            <div class="hero-title">Dynamic Stratification, DFS, QFD, and MILP Decision Studio</div>
            <div class="hero-subtitle">Deployment-ready integrated environment for scenario stratification, expert covariance analysis, dual-fuzzy synthesis, quality-function deployment, and optimization-driven strategy portfolio selection.</div>
            <div class="hero-grid">
                <div class="info-card"><span class="info-label">Security</span><strong>Password-protected access</strong><small>Safer presentation and sharing in research or academic deployment settings.</small></div>
                <div class="info-card"><span class="info-label">Research flow</span><strong>Five connected analytical modules</strong><small>Move from structured inputs to rankings, weighted matrices, and optimal portfolios.</small></div>
                <div class="info-card"><span class="info-label">Presentation</span><strong>Professional deployable interface</strong><small>Polished visuals, sidebar researcher profiles, and export-ready outputs.</small></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_module_banner(icon: str, title: str, subtitle: str, badge: str = "Module"):
    st.markdown(
        f"""
        <div class="module-banner">
            <div class="module-badge">{escape(badge)}</div>
            <div class="module-title">{escape(icon)} {escape(title)}</div>
            <div class="module-subtitle">{escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <div class="app-footer"><span class="copyright-pill">© 2026 Research Decision Studio</span><br>Developed by <strong>Moktadir M.A.</strong> and <strong>REN J.Z.</strong></div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# MODULE 1: STRATIFICATION MODELER
# ============================================================

class StratificationEngine:
    @staticmethod
    def parse_parents(s: str) -> List[str]:
        if not isinstance(s, str) or not s.strip():
            return []
        return list(dict.fromkeys([p.strip() for p in s.split(",") if p.strip()]))

    @staticmethod
    def topo_sort(nodes: List[str], parents: Dict[str, List[str]]) -> List[str]:
        node_set = set(nodes)
        indeg = {n: 0 for n in nodes}
        children = {n: [] for n in nodes}

        for n, ps in parents.items():
            for p in ps:
                if p not in node_set:
                    continue
                indeg[n] += 1
                children[p].append(n)

        q = [n for n in nodes if indeg[n] == 0]
        out = []

        while q:
            cur = q.pop(0)
            out.append(cur)
            for ch in children[cur]:
                indeg[ch] -= 1
                if indeg[ch] == 0:
                    q.append(ch)

        if len(out) != len(nodes):
            raise ValueError("Circular dependency detected in interactions.")
        return out

    @staticmethod
    def solve_for_p(base_multipliers: Dict[str, float], order: List[str], parents: Dict[str, List[str]]) -> float:
        def f(P: float) -> float:
            probs = {}
            for n in order:
                ps = parents.get(n, [])
                if not ps:
                    probs[n] = base_multipliers.get(n, 0.0) * P
                else:
                    prod = 1.0
                    for p in ps:
                        prod *= probs.get(p, 0.0)
                    probs[n] = prod
            return sum(probs.values()) - 1.0

        lo, hi = 0.0, 1.0
        for _ in range(100):
            if f(hi) > 0:
                break
            hi *= 2.0

        for _ in range(100):
            mid = (lo + hi) / 2.0
            if f(mid) > 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0


def _init_strat_base_df(num_events: int) -> pd.DataFrame:
    root_id = "S1"
    base_ids = [f"S{i}" for i in range(2, num_events + 2)]
    all_base_ids = [root_id] + base_ids
    default_vals = [25.0 if i < 5 else 0.0 for i in range(len(all_base_ids))]

    return pd.DataFrame(
        {
            "ID": all_base_ids,
            "Label": [f"Context {i}" for i in range(1, len(all_base_ids) + 1)],
            "Value (%)": default_vals,
        }
    )


def _init_strat_inter_df(num_events: int) -> pd.DataFrame:
    if num_events == 4:
        return pd.DataFrame(
            [
                {"ID": "S6", "Parents (e.g. S2,S3)": "S2,S3", "Label": "Interaction Alpha"},
                {"ID": "S7", "Parents (e.g. S2,S3)": "S6,S4", "Label": "Final Scenario"},
            ]
        )
    return pd.DataFrame(columns=["ID", "Parents (e.g. S2,S3)", "Label"])


def _ensure_strat_store(num_events: int) -> None:
    base_cols = ["ID", "Label", "Value (%)"]
    inter_cols = ["ID", "Parents (e.g. S2,S3)", "Label"]
    strat_store_key = f"strat|{num_events}"
    base_df = st.session_state.get("strat_base_df")
    inter_df = st.session_state.get("strat_inter_df")

    base_ok = isinstance(base_df, pd.DataFrame) and list(base_df.columns) == base_cols and len(base_df) == (num_events + 1)
    inter_ok = isinstance(inter_df, pd.DataFrame) and all(col in inter_df.columns for col in inter_cols)

    if st.session_state.get("strat_store_key") != strat_store_key or not base_ok or not inter_ok:
        st.session_state["strat_store_key"] = strat_store_key
        st.session_state["strat_base_df"] = _init_strat_base_df(num_events)
        st.session_state["strat_inter_df"] = _init_strat_inter_df(num_events)
        return

    st.session_state["strat_base_df"] = base_df.loc[:, base_cols].copy()
    st.session_state["strat_inter_df"] = inter_df.reindex(columns=inter_cols).copy()


def page_stratification():
    render_module_banner("🧭", "Stratification Modeler", "Network-based stratification dashboard for base events, interaction scenarios, and probability propagation.", badge="Module 1")

    st.sidebar.subheader("Stratification settings")
    num_events = int(st.sidebar.number_input("Base Event Count", 1, 20, 4, key="strat_num_events"))
    precision = st.sidebar.slider("Graph Precision (Decimals)", 2, 6, 4, key="strat_precision")
    show_labels = st.sidebar.checkbox("Show Labels on Graph", value=True, key="strat_show_labels")
    normalize = st.sidebar.checkbox("Normalize to 100%", value=True, key="strat_normalize")
    st.sidebar.info("Input Mode: Percentages (%)")

    _ensure_strat_store(num_events)

    with st.expander("Calculation Methodology & Equations"):
        st.write("The model calculates the root scaling factor $P$ by solving for the point where the sum of all scenario probabilities equals 1.")
        st.latex(r"\sum_{i=1}^{n} Prob(S_i) = 1")
        st.write("For Base Scenarios:")
        st.latex(r"Prob(S_{base}) = \left( \frac{Value_{\%}}{100} \right) \times P")
        st.write("For Interaction Scenarios:")
        st.latex(r"Prob(S_{inter}) = \prod Prob(S_{parents})")

    col_left, col_right = st.columns([1, 1.5], gap="medium")

    with col_left:
        st.subheader("1. Definitions")

        base_df = st.data_editor(
            st.session_state["strat_base_df"],
            use_container_width=True,
            hide_index=True,
            key="strat_base_editor",
        )
        st.session_state["strat_base_df"] = base_df

        st.subheader("2. Interactions")
        inter_df = st.data_editor(
            st.session_state["strat_inter_df"],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="strat_inter_editor",
        )
        st.session_state["strat_inter_df"] = inter_df

    with col_right:
        st.subheader("3. Analytics & Visualization")

        try:
            engine = StratificationEngine()

            root_id = "S1"
            base_ids = [str(v).strip() for v in base_df["ID"].tolist() if str(v).strip() and str(v).strip() != root_id]
            all_base_ids = [root_id] + base_ids

            base_mult = {}
            labels = {}
            for _, row in base_df.iterrows():
                sid = str(row["ID"]).strip()
                if not sid:
                    continue
                base_mult[sid] = float(row["Value (%)"]) / 100.0
                labels[sid] = str(row["Label"]).strip()

            parents_map = {sid: [] for sid in all_base_ids}
            nodes_ordered = list(all_base_ids)

            for _, row in inter_df.iterrows():
                sid = str(row.get("ID", "")).strip()
                if not sid:
                    continue
                ps = engine.parse_parents(str(row.get("Parents (e.g. S2,S3)", "")))
                parents_map[sid] = ps
                if sid not in nodes_ordered:
                    nodes_ordered.append(sid)
                labels[sid] = str(row.get("Label", "")).strip()

            sorted_nodes = engine.topo_sort(nodes_ordered, parents_map)
            P_val = engine.solve_for_p(base_mult, sorted_nodes, parents_map)

            probs = {}
            for n in sorted_nodes:
                ps = parents_map.get(n, [])
                if not ps:
                    probs[n] = base_mult.get(n, 0.0) * P_val
                else:
                    p_prod = 1.0
                    for p in ps:
                        p_prod *= probs.get(p, 0.0)
                    probs[n] = p_prod

            total_p = sum(probs.values())
            if normalize and total_p > 0:
                probs = {k: v / total_p for k, v in probs.items()}
                total_p = 1.0

            m1, m2 = st.columns(2)
            m1.metric("Solved Factor (P)", f"{P_val:.6f}")
            m2.metric("Total Sum", f"{total_p * 100:.2f}%")

            dot = graphviz.Digraph(format="svg")
            dot.attr(rankdir="LR", bgcolor="transparent")

            for n in sorted_nodes:
                display_name = labels.get(n, n) if show_labels else n
                p_str = f"{probs[n]:.{precision}f}"
                color = "#D1E8FF" if n == root_id else "#E1F5FE"
                dot.node(
                    n,
                    f"{display_name}\nP={p_str}",
                    style="filled",
                    fillcolor=color,
                    shape="box",
                    fontname="Arial",
                )

            for child, ps in parents_map.items():
                for p in ps:
                    dot.edge(p, child)

            for b in base_ids:
                if root_id in sorted_nodes and b in sorted_nodes:
                    dot.edge(root_id, b, style="dashed", color="gray", label="base")

            st.graphviz_chart(dot)

            res_list = []
            for n in sorted_nodes:
                res_list.append(
                    {
                        "Scenario ID": n,
                        "Label": labels.get(n, ""),
                        "Probability": round(probs[n], precision + 2),
                        "Percentage (%)": round(probs[n] * 100, 4),
                    }
                )

            final_results_df = pd.DataFrame(res_list)
            st.session_state["strat_results_df"] = final_results_df.copy()

            with st.expander("Detailed Results & CSV Export", expanded=True):
                display_df = final_results_df.copy()
                display_df["Percentage (%)"] = display_df["Percentage (%)"].map(lambda x: f"{x:.2f}%")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                csv = final_results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results (CSV)",
                    data=csv,
                    file_name="stratification_results.csv",
                    mime="text/csv",
                )

            send_candidates_df = final_results_df.reset_index(drop=True)

            st.subheader("Send scenario probabilities to DFS-AHP")
            st.caption("You can send all scenarios, including the base/root scenario S1.")
            if send_candidates_df.empty:
                st.info("No scenarios are available to send.")
            else:
                label_map = {}
                for _, row in send_candidates_df.iterrows():
                    sid = row["Scenario ID"]
                    role = "Base/Root" if sid == root_id else "Scenario"
                    label_map[sid] = f'{sid} - {row["Label"]} [{role}] ({row["Probability"]:.6f})'

                selected_scenarios = st.multiselect(
                    "Select scenarios to send",
                    options=send_candidates_df["Scenario ID"].tolist(),
                    default=send_candidates_df["Scenario ID"].tolist(),
                    format_func=lambda x: label_map.get(x, x),
                    key="strat_send_pick",
                )

                if st.button("Send selected probabilities to DFS-AHP", key="strat_send_btn"):
                    send_df = send_candidates_df[send_candidates_df["Scenario ID"].isin(selected_scenarios)].reset_index(drop=True)
                    if send_df.empty:
                        st.warning("Please select at least one scenario.")
                    else:
                        st.session_state["strat_to_ahp_df"] = send_df.copy()
                        st.session_state["ahp_s"] = int(len(send_df))
                        for key in list(st.session_state.keys()):
                            if key.startswith("ahp_sp_"):
                                del st.session_state[key]
                        for idx, p in enumerate(send_df["Probability"].tolist()):
                            st.session_state[f"ahp_sp_{idx}"] = float(p)
                        st.session_state["ahp_loaded_signature"] = tuple(
                            (str(row["Scenario ID"]), float(row["Probability"]))
                            for _, row in send_df.iterrows()
                        )
                        st.success("Selected scenario probabilities, including base/root scenarios, sent to Module 3 (DFS-AHP).")

        except Exception as e:
            st.error(f"Configuration Error: {e}")


# ============================================================
# MODULE 2: EXPERT WEIGHT DETERMINATION MODEL
# ============================================================

def init_expert_cov_df(n_experts: int, n_dims: int) -> pd.DataFrame:
    cols = [f"X{j+1}" for j in range(n_dims)]
    idx = [f"Ex{i+1}" for i in range(n_experts)]
    return pd.DataFrame(np.zeros((n_experts, n_dims), dtype=float), index=idx, columns=cols)


def minmax_normalize_expert_data(df: pd.DataFrame) -> pd.DataFrame:
    x = df.astype(float).copy()
    col_min = x.min(axis=0)
    col_max = x.max(axis=0)
    denom = (col_max - col_min).replace(0, np.nan)

    normalized = (x - col_min) / denom
    normalized = normalized.fillna(0.0)
    return normalized


def compute_excel_matching_covariance(normalized_df: pd.DataFrame) -> pd.DataFrame:
    x = normalized_df.astype(float).to_numpy()
    cov = np.cov(x, rowvar=False, ddof=0)
    return pd.DataFrame(cov, index=normalized_df.columns, columns=normalized_df.columns)


def compute_sorted_eigenvector_weights(raw_df: pd.DataFrame, cov_df: pd.DataFrame):
    cov_matrix = cov_df.astype(float).to_numpy()

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    max_index = int(np.argmax(eigenvalues))
    max_eigenvalue = float(eigenvalues[max_index])
    principal_eigenvector = eigenvectors[:, max_index].astype(float)

    sorted_eigenvector = np.sort(principal_eigenvector)[::-1]

    lambda_values = raw_df.astype(float).to_numpy() @ sorted_eigenvector
    lambda_sum = float(np.sum(lambda_values))

    if abs(lambda_sum) < 1e-12:
        weights = np.ones(len(lambda_values)) / len(lambda_values)
    else:
        weights = lambda_values / lambda_sum

    eigen_df = pd.DataFrame(
        {
            "Dimension": raw_df.columns,
            "Principal Eigenvector": principal_eigenvector,
            "Sorted Eigenvector (desc)": sorted_eigenvector,
        }
    )

    result_df = pd.DataFrame(
        {
            "Expert": raw_df.index,
            "λ": lambda_values,
            "Weight": weights,
        }
    )

    all_eigen_df = pd.DataFrame({"Eigenvalue": eigenvalues}).sort_values("Eigenvalue", ascending=False).reset_index(drop=True)

    return max_eigenvalue, eigen_df, result_df, all_eigen_df


class DFSAHP:
    def __init__(self):
        self.optimistic_uv = {
            "EEI": (0.50, 0.50),
            "SMI": (0.55, 0.45),
            "WMI": (0.60, 0.40),
            "MI": (0.65, 0.35),
            "StMI": (0.70, 0.30),
            "VSI": (0.75, 0.25),
            "AMI": (0.80, 0.20),
            "PMI": (0.85, 0.15),
            "EMI": (0.90, 0.10),
        }

        self.pessimistic_uv = {
            "EEU": (0.50, 0.50),
            "SMU": (0.45, 0.55),
            "WMU": (0.40, 0.60),
            "MU": (0.35, 0.65),
            "StMU": (0.30, 0.70),
            "VSU": (0.25, 0.75),
            "AMU": (0.20, 0.80),
            "PMU": (0.15, 0.85),
            "EMU": (0.10, 0.90),
        }
        self.pessimistic_uv["EEI"] = (0.50, 0.50)

        self.k = 0.9

        self.intensity = {
            "EEI": 1, "EEU": 1,
            "SMI": 2, "SMU": 2,
            "WMI": 3, "WMU": 3,
            "MI": 4, "MU": 4,
            "StMI": 5, "StMU": 5,
            "VSI": 6, "VSU": 6,
            "AMI": 7, "AMU": 7,
            "PMI": 8, "PMU": 8,
            "EMI": 9, "EMU": 9,
        }

        self.RI = {
            1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
            11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59,
        }

    def term_to_uv(self, term: str) -> Tuple[float, float]:
        if term is None:
            return (0.5, 0.5)
        t = str(term).strip()
        if t in self.optimistic_uv:
            return self.optimistic_uv[t]
        if t in self.pessimistic_uv:
            return self.pessimistic_uv[t]
        return (0.5, 0.5)

    def build_abcd_matrices(self, combined_df: pd.DataFrame, criteria: List[str]):
        n = len(criteria)
        a = np.zeros((n, n), dtype=float)
        b = np.zeros((n, n), dtype=float)
        c = np.zeros((n, n), dtype=float)
        d = np.zeros((n, n), dtype=float)
        invalid_terms = set()

        for i in range(n):
            for j in range(n):
                o_col = f"{criteria[j]} (O)"
                p_col = f"{criteria[j]} (P)"
                o_term = combined_df.iloc[i][o_col]
                p_term = combined_df.iloc[i][p_col]

                uo, vo = self.term_to_uv(o_term)
                up, vp = self.term_to_uv(p_term)

                a[i, j], b[i, j] = float(uo), float(vo)
                c[i, j], d[i, j] = float(up), float(vp)

                for raw in [o_term, p_term]:
                    if raw is None:
                        continue
                    raw = str(raw).strip()
                    if raw and (raw not in self.optimistic_uv) and (raw not in self.pessimistic_uv):
                        invalid_terms.add(raw)

        return a, b, c, d, sorted(invalid_terms)

    def aggregate_a_row(self, row_a: np.ndarray) -> float:
        n = len(row_a)
        row_a = np.clip(row_a, 1e-12, 1.0)
        return float(np.prod(row_a ** (1.0 / n)))

    def aggregate_b_row(self, row_b: np.ndarray) -> float:
        n = len(row_b)
        row_b = np.clip(row_b, 0.0, 1.0 - 1e-12)
        return float(1.0 - np.prod((1.0 - row_b) ** (1.0 / n)))

    def aggregate_c_row_excel(self, row_c: np.ndarray) -> float:
        n = len(row_c)
        row_c = np.clip(row_c, 1e-12, 1.0)
        num = float(np.prod(row_c))
        denom = (1.0 / n) * float(np.sum((row_c ** (n - 1)) * (1.0 - row_c))) + num
        return 0.0 if abs(denom) < 1e-12 else float(num / denom)

    def aggregate_d_row(self, row_d: np.ndarray) -> float:
        return float(np.mean(row_d))

    def ci_dfs_value(self, a, b, c, d) -> float:
        val = 1.0 - np.sqrt(
            ((a - d) ** 2 + (b - c) ** 2 + (1.0 - a - b) ** 2 + (1.0 - c - d) ** 2) / 2.0
        )
        return float(max(min(val, 1.0), 0.0))

    def si_value(self, a, b, c, d, ci_dfs) -> float:
        return float((a + b - c + d) * ci_dfs / (2.0 * self.k))

    def _is_unimportant(self, term: str) -> bool:
        t = (term or "").strip()
        return t.endswith("U") or t == "EEU"

    def term_to_ratio(self, term: str) -> float:
        if term is None:
            return 1.0
        t = str(term).strip()
        if t == "":
            return 1.0
        k = self.intensity.get(t, None)
        if k is None:
            return 1.0
        return (1.0 / float(k)) if self._is_unimportant(t) else float(k)

    def build_cr_matrix_excel_rule(self, combined_df: pd.DataFrame, criteria: List[str]) -> np.ndarray:
        n = len(criteria)
        A = np.ones((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                o_term = combined_df.iloc[i][f"{criteria[j]} (O)"]
                p_term = combined_df.iloc[i][f"{criteria[j]} (P)"]

                aij = self.term_to_ratio(o_term)
                aji = self.term_to_ratio(p_term)

                if aij <= 0:
                    aij = 1.0
                if aji <= 0:
                    aji = 1.0

                A[i, j] = aij
                A[j, i] = aji
        return A

    def ahp_weights_geometric_mean(self, A: np.ndarray) -> np.ndarray:
        n = A.shape[0]
        gm = np.prod(A, axis=1) ** (1.0 / n)
        s = gm.sum()
        return gm / s if s > 0 else np.ones(n) / n

    def cr_gm_method(self, A: np.ndarray) -> Dict[str, object]:
        n = A.shape[0]
        if n <= 2:
            return {
                "lambda_max": 0.0,
                "CI_AHP": 0.0,
                "RI": self.RI.get(n, 0.0),
                "CR": 0.0,
                "weights": np.ones(n) / n,
            }
        w = self.ahp_weights_geometric_mean(A)
        Aw = A.dot(w)
        lam_vec = Aw / np.clip(w, 1e-12, None)
        lambda_max = float(np.mean(lam_vec))
        CI = float((lambda_max - n) / (n - 1))
        RI = float(self.RI.get(n, 0.0))
        CR = float(CI / RI) if RI > 0 else 0.0
        return {"lambda_max": lambda_max, "CI_AHP": CI, "RI": RI, "CR": CR, "weights": w}

    def ahp_weights_eigen(self, A: np.ndarray) -> Tuple[float, np.ndarray]:
        eigvals, eigvecs = np.linalg.eig(A)
        idx = int(np.argmax(np.real(eigvals)))
        lam = float(np.real(eigvals[idx]))
        v = np.real(eigvecs[:, idx]).astype(float)
        if np.all(v <= 0):
            v = -v
        v = np.abs(v)
        s = v.sum()
        w = v / s if s > 0 else np.ones(A.shape[0]) / A.shape[0]
        return lam, w

    def cr_eigen_method(self, A: np.ndarray) -> Dict[str, object]:
        n = A.shape[0]
        if n <= 2:
            return {
                "lambda_max": 0.0,
                "CI_AHP": 0.0,
                "RI": self.RI.get(n, 0.0),
                "CR": 0.0,
                "weights": np.ones(n) / n,
            }
        lambda_max, w = self.ahp_weights_eigen(A)
        CI = float((lambda_max - n) / (n - 1))
        RI = float(self.RI.get(n, 0.0))
        CR = float(CI / RI) if RI > 0 else 0.0
        return {"lambda_max": lambda_max, "CI_AHP": CI, "RI": RI, "CR": CR, "weights": w}

    def compute_expert(self, combined_df: pd.DataFrame, criteria: List[str]):
        a_mat, b_mat, c_mat, d_mat, invalid_terms = self.build_abcd_matrices(combined_df, criteria)
        n = len(criteria)

        rows = []
        for i in range(n):
            a_i = self.aggregate_a_row(a_mat[i, :])
            b_i = self.aggregate_b_row(b_mat[i, :])
            c_i = self.aggregate_c_row_excel(c_mat[i, :])
            d_i = self.aggregate_d_row(d_mat[i, :])

            ci_dfs = self.ci_dfs_value(a_i, b_i, c_i, d_i)
            si = self.si_value(a_i, b_i, c_i, d_i, ci_dfs)

            rows.append(
                {
                    "Criterion": criteria[i],
                    "a (μO)": a_i,
                    "b (νO)": b_i,
                    "c (μP)": c_i,
                    "d (νP)": d_i,
                    "CI (DFS)": ci_dfs,
                    "SI": si,
                }
            )

        df = pd.DataFrame(rows)
        s_sum = float(df["SI"].sum())
        df["Normalized Weight"] = df["SI"] / s_sum if s_sum > 0 else 0.0
        df = df.sort_values("Normalized Weight", ascending=False).reset_index(drop=True)
        df["Rank"] = np.arange(1, len(df) + 1)

        A = self.build_cr_matrix_excel_rule(combined_df, criteria)
        cr_gm = self.cr_gm_method(A)
        cr_ev = self.cr_eigen_method(A)

        return df, invalid_terms, {"A": A, "GM": cr_gm, "EIGEN": cr_ev}


def make_blank_ahp_matrix(criteria: List[str]) -> pd.DataFrame:
    cols = []
    for c in criteria:
        cols.extend([f"{c} (O)", f"{c} (P)"])
    df = pd.DataFrame("", index=criteria, columns=cols)
    for i, c in enumerate(criteria):
        df.at[criteria[i], f"{c} (O)"] = "EEI"
        df.at[criteria[i], f"{c} (P)"] = "EEI"
    return df


def fmt_ratio(x: float) -> str:
    for d in [2, 3, 4, 5, 6, 7, 8, 9]:
        if abs(x - 1.0 / d) < 1e-3:
            return f"1/{d}"
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if abs(x - k) < 1e-3:
            return f"{k:.2f}"
    return f"{x:.2f}"


@dataclass
class DFS:
    mu_O: float
    nu_O: float
    mu_P: float
    nu_P: float

    def clip(self) -> "DFS":
        return DFS(
            float(np.clip(self.mu_O, 0, 1)),
            float(np.clip(self.nu_O, 0, 1)),
            float(np.clip(self.mu_P, 0, 1)),
            float(np.clip(self.nu_P, 0, 1)),
        )


def dfs_dwgm(values: List[DFS], weights: List[float]) -> DFS:
    mu_O = 1.0
    nu_O = 1.0
    mu_P = 1.0
    nu_P = 1.0
    for v, w in zip(values, weights):
        mu_O *= (max(v.mu_O, 1e-12) ** w)
        nu_O *= (max(v.nu_O, 1e-12) ** w)
        mu_P *= (max(v.mu_P, 1e-12) ** w)
        nu_P *= (max(v.nu_P, 1e-12) ** w)
    return DFS(mu_O, nu_O, mu_P, nu_P).clip()


def dfs_multiply(weight_dfs: DFS, rel_dfs: DFS) -> DFS:
    w = weight_dfs
    r = rel_dfs

    a = w.mu_O * r.mu_O
    b = w.nu_O + r.nu_O - (w.nu_O * r.nu_O)

    denom_c = (w.mu_P + r.mu_P - (w.mu_P * r.mu_P))
    c = (w.mu_P * r.mu_P) / denom_c if abs(denom_c) > 1e-12 else 0.0

    denom_d = (1.0 - (w.nu_P * r.nu_P))
    d = (w.nu_P + r.nu_P - 2.0 * w.nu_P * r.nu_P) / denom_d if abs(denom_d) > 1e-12 else 0.0

    return DFS(a, b, c, d).clip()


def score_defuzz_from_weighted(weighted: DFS) -> float:
    a, b, c, d = weighted.mu_O, weighted.nu_O, weighted.mu_P, weighted.nu_P
    term1 = math.sqrt(max((a + d) / 8.0, 0.0))
    term2 = math.sqrt(max((b + c) / 8.0, 0.0))
    return 0.5 + (term1 - term2)


def default_scale_uv() -> Dict[str, Tuple[float, float]]:
    return {
        "EEI": (0.50, 0.50),
        "SMI": (0.55, 0.45),
        "WMI": (0.60, 0.40),
        "MI": (0.65, 0.35),
        "StMI": (0.70, 0.30),
        "VSI": (0.75, 0.25),
        "AMI": (0.80, 0.20),
        "PMI": (0.85, 0.15),
        "EMI": (0.90, 0.10),
        "EEU": (0.50, 0.50),
        "SMU": (0.45, 0.55),
        "WMU": (0.40, 0.60),
        "MU": (0.35, 0.65),
        "StMU": (0.30, 0.70),
        "VSU": (0.25, 0.75),
        "AMU": (0.20, 0.80),
        "PMU": (0.15, 0.85),
        "EMU": (0.10, 0.90),
    }


def term_to_dfs(term_O: str, term_P: str, scale: Dict[str, Tuple[float, float]]) -> DFS:
    muO, nuO = scale.get(term_O, scale["EEI"])
    muP, nuP = scale.get(term_P, scale["EEU"])
    return DFS(muO, nuO, muP, nuP).clip()


def normalize_weights(ws: List[float]) -> List[float]:
    if len(ws) == 0:
        return []
    s = sum(ws)
    if s <= 0:
        return [1.0 / len(ws)] * len(ws)
    return [w / s for w in ws]


def safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return float(x)
    except Exception:
        return default


def init_relationship_df(rc_names: List[str], ms_names: List[str]) -> pd.DataFrame:
    cols = ["RC"]
    for ms in ms_names:
        cols += [f"{ms}_O", f"{ms}_P"]
    df = pd.DataFrame({"RC": rc_names})
    for c in cols[1:]:
        df[c] = "EEI" if c.endswith("_O") else "EEU"
    return df[cols]


def init_rc_weight_df(rc_names: List[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "RC": rc_names,
            "mu_O": [0.50] * len(rc_names),
            "nu_O": [0.50] * len(rc_names),
            "mu_P": [0.50] * len(rc_names),
            "nu_P": [0.50] * len(rc_names),
        }
    )


def init_cost_df(ms_names: List[str]) -> pd.DataFrame:
    return pd.DataFrame({"MS": ms_names, "ICj": [1.0] * len(ms_names)})


def dfs_table_to_map(df: pd.DataFrame, criteria: List[str]) -> Dict[str, DFS]:
    base = df.set_index("Criterion")
    out = {}
    for c in criteria:
        row = base.loc[c]
        out[c] = DFS(
            float(row["a (μO)"]),
            float(row["b (νO)"]),
            float(row["c (μP)"]),
            float(row["d (νP)"]),
        ).clip()
    return out


def build_weight_table_from_dfs_map(criteria: List[str], dfs_map: Dict[str, DFS], engine: DFSAHP) -> pd.DataFrame:
    rows = []
    for c in criteria:
        v = dfs_map[c].clip()
        ci_dfs = engine.ci_dfs_value(v.mu_O, v.nu_O, v.mu_P, v.nu_P)
        si = engine.si_value(v.mu_O, v.nu_O, v.mu_P, v.nu_P, ci_dfs)
        rows.append(
            {
                "Criterion": c,
                "a (μO)": v.mu_O,
                "b (νO)": v.nu_O,
                "c (μP)": v.mu_P,
                "d (νP)": v.nu_P,
                "CI (DFS)": ci_dfs,
                "SI": si,
            }
        )

    df = pd.DataFrame(rows)
    s_sum = float(df["SI"].sum())
    df["Normalized Weight"] = df["SI"] / s_sum if s_sum > 0 else 0.0
    df = df.sort_values("Normalized Weight", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df


def weighted_componentwise_dfs_aggregate(values: List[DFS], weights: List[float]) -> DFS:
    m = len(values)
    W = np.array(normalize_weights([safe_float(w, 1.0) for w in weights]), dtype=float)

    a_vec = np.clip(np.array([v.mu_O for v in values], dtype=float), 1e-12, 1.0)
    b_vec = np.clip(np.array([v.nu_O for v in values], dtype=float), 0.0, 1.0 - 1e-12)
    c_vec = np.clip(np.array([v.mu_P for v in values], dtype=float), 1e-12, 1.0)
    d_vec = np.clip(np.array([v.nu_P for v in values], dtype=float), 0.0, 1.0)

    a = float(np.prod(a_vec ** W))
    b = float(1.0 - np.prod((1.0 - b_vec) ** W))

    c_num = float(np.prod(c_vec))
    c_aux = float(np.sum((c_vec ** (m - 1)) * W * (1.0 - c_vec)))
    c_den = c_aux + c_num
    c = 0.0 if abs(c_den) < 1e-12 else float(c_num / c_den)

    d_num = float(np.sum(d_vec * W))
    d_den = 1.0 + float(np.sum((d_vec * W) - (d_vec / m)))
    d = 0.0 if abs(d_den) < 1e-12 else float(d_num / d_den)

    return DFS(a, b, c, d).clip()


def aggregate_experts_decomposed_dfs(
    expert_tables: List[pd.DataFrame],
    expert_weights: List[float],
    criteria: List[str],
    engine: DFSAHP,
) -> pd.DataFrame:
    expert_maps = [dfs_table_to_map(t, criteria) for t in expert_tables]
    agg_map = {}

    for c in criteria:
        vals = [expert_maps[e][c] for e in range(len(expert_maps))]
        agg_map[c] = weighted_componentwise_dfs_aggregate(vals, expert_weights)

    return build_weight_table_from_dfs_map(criteria, agg_map, engine)


def aggregate_scenarios_decomposed_dfs(
    scenario_tables: List[pd.DataFrame],
    scenario_probs: List[float],
    criteria: List[str],
    engine: DFSAHP,
) -> Tuple[pd.DataFrame, List[float]]:
    probs = normalize_weights([safe_float(p, 1.0) for p in scenario_probs])
    scenario_maps = [dfs_table_to_map(t, criteria) for t in scenario_tables]

    final_map = {}
    for c in criteria:
        vals = [scenario_maps[s][c] for s in range(len(scenario_maps))]
        final_map[c] = weighted_componentwise_dfs_aggregate(vals, probs)

    final_df = build_weight_table_from_dfs_map(criteria, final_map, engine)
    return final_df, probs


def decomposed_weight_table_to_rc_df(df: pd.DataFrame, criteria_order: List[str]) -> pd.DataFrame:
    x = df.copy().set_index("Criterion").loc[criteria_order].reset_index()
    return pd.DataFrame(
        {
            "RC": x["Criterion"].astype(str).tolist(),
            "mu_O": x["a (μO)"].astype(float).tolist(),
            "nu_O": x["b (νO)"].astype(float).tolist(),
            "mu_P": x["c (μP)"].astype(float).tolist(),
            "nu_P": x["d (νP)"].astype(float).tolist(),
        }
    )


def _to_float(x, default=0.0):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return default
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b)
        return float(s)
    except Exception:
        return default


def expected_from_omp(O, ML, P):
    return (O + P + 4.0 * ML) / 6.0


def make_square_df(names, fill=0.0):
    df = pd.DataFrame(fill, index=names, columns=names)
    np.fill_diagonal(df.values, 0.0)
    return df


def subset_cost_time(x, cost, time, rho, zeta):
    x = x.astype(int)
    base_cost = float(np.dot(cost, x))
    base_time = float(np.dot(time, x))

    n = len(x)
    sav_cost = 0.0
    sav_time = 0.0
    for i in range(n):
        if x[i] == 0:
            continue
        for j in range(i + 1, n):
            if x[j] == 0:
                continue
            sav_cost += float(rho[i, j])
            sav_time += float(zeta[i, j])

    total_cost = base_cost - sav_cost
    total_time = base_time - sav_time
    return total_cost, total_time, base_cost, base_time, sav_cost, sav_time


def solve_by_enumeration(rep, cost, time, rho, zeta, budget, time_limit):
    n = len(rep)
    best = None
    for mask in range(1 << n):
        x = np.array([(mask >> k) & 1 for k in range(n)], dtype=int)
        total_cost, total_time, base_cost, base_time, sav_cost, sav_time = subset_cost_time(
            x, cost, time, rho, zeta
        )
        if total_cost <= budget + 1e-9 and total_time <= time_limit + 1e-9:
            obj = float(np.dot(rep, x))
            if (best is None) or (obj > best["obj"] + 1e-12):
                best = {
                    "x": x,
                    "obj": obj,
                    "total_cost": total_cost,
                    "total_time": total_time,
                    "base_cost": base_cost,
                    "base_time": base_time,
                    "sav_cost": sav_cost,
                    "sav_time": sav_time,
                }
    return best


def compute_expected_vector(df_omp, col_O="O", col_ML="ML", col_P="P"):
    O = df_omp[col_O].map(lambda v: _to_float(v, 0.0)).to_numpy(dtype=float)
    ML = df_omp[col_ML].map(lambda v: _to_float(v, 0.0)).to_numpy(dtype=float)
    P = df_omp[col_P].map(lambda v: _to_float(v, 0.0)).to_numpy(dtype=float)
    return expected_from_omp(O, ML, P)


def compute_expected_matrix(df_O, df_ML, df_P):
    n = df_O.shape[0]
    E = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            O = _to_float(df_O.iloc[i, j], 0.0)
            ML = _to_float(df_ML.iloc[i, j], 0.0)
            P = _to_float(df_P.iloc[i, j], 0.0)
            E[i, j] = expected_from_omp(O, ML, P)
    np.fill_diagonal(E, 0.0)
    out = np.zeros_like(E)
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = E[i, j]
    return out


def page_expert_covariance_model():
    render_module_banner("👥", "Expert Weight Determination Model", "Covariance-based expert weighting with normalized data, covariance structure, and eigenvector-driven priority estimation.", badge="Module 2")

    st.markdown("### Configuration")
    cfg1, cfg2 = st.columns(2)
    with cfg1:
        n_experts = int(
            st.number_input(
                "Number of experts",
                min_value=2,
                max_value=100,
                value=4,
                step=1,
                key="cov_n_experts",
            )
        )
    with cfg2:
        n_dims = int(
            st.number_input(
                "Number of dimensions",
                min_value=2,
                max_value=50,
                value=10,
                step=1,
                key="cov_n_dims",
            )
        )

    st.subheader("Expert and dimension names")
    c1, c2 = st.columns(2)

    with c1:
        expert_names = []
        for i in range(n_experts):
            expert_names.append(st.text_input(f"Expert {i+1}", value=f"Ex{i+1}", key=f"cov_expert_{i}"))

    with c2:
        dim_names = []
        for j in range(n_dims):
            dim_names.append(st.text_input(f"Dimension {j+1}", value=f"X{j+1}", key=f"cov_dim_{j}"))

    store_key = f"cov|{n_experts}|{n_dims}|{'|'.join(expert_names)}|{'|'.join(dim_names)}"
    if st.session_state.get("cov_store_key") != store_key:
        st.session_state["cov_store_key"] = store_key
        df0 = init_expert_cov_df(n_experts, n_dims)
        df0.index = expert_names
        df0.columns = dim_names
        st.session_state["cov_input_df"] = df0

    df_current = st.session_state["cov_input_df"].copy()
    df_current.index = expert_names
    df_current.columns = dim_names
    st.session_state["cov_input_df"] = df_current

    st.subheader("Input expert data matrix χ_ij")
    st.caption("Paste expert values directly from Excel.")
    input_df = st.data_editor(
        st.session_state["cov_input_df"],
        use_container_width=True,
        num_rows="fixed",
        height=300,
        key="cov_input_editor",
    )
    input_df.index = expert_names
    input_df.columns = dim_names
    st.session_state["cov_input_df"] = input_df

    if st.button("Compute expert weights", type="primary", key="cov_run"):
        try:
            raw_df = input_df.astype(float).copy()
        except Exception:
            st.error("All input values must be numeric.")
            return

        normalized_df = minmax_normalize_expert_data(raw_df)
        cov_df = compute_excel_matching_covariance(normalized_df)

        max_eigenvalue, eigen_df, result_df, all_eigen_df = compute_sorted_eigenvector_weights(raw_df, cov_df)

        st.session_state["cov_raw_df"] = raw_df
        st.session_state["cov_normalized_df"] = normalized_df
        st.session_state["cov_cov_df"] = cov_df
        st.session_state["cov_eigen_df"] = eigen_df
        st.session_state["cov_result_df"] = result_df
        st.session_state["cov_all_eigen_df"] = all_eigen_df

        st.subheader("Normalized data ψ")
        st.dataframe(normalized_df.round(6), use_container_width=True)

        st.subheader("Covariance matrix η_jj")
        st.caption("Computed from normalized data using population covariance: ddof=0")
        st.dataframe(cov_df.round(4), use_container_width=True)

        st.subheader("Eigenvalues")
        st.dataframe(all_eigen_df.round(6), use_container_width=True)
        st.metric("Maximum Eigenvalue", f"{max_eigenvalue:.6f}")

        st.subheader("Principal and sorted eigenvector")
        st.dataframe(eigen_df.round(6), use_container_width=True)

        st.subheader("Expert scores and weights")
        st.dataframe(result_df.round(6), use_container_width=True)
        st.bar_chart(result_df.set_index("Expert")[["Weight"]], use_container_width=True)

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            raw_df.to_excel(writer, sheet_name="Expert_Data")
            normalized_df.to_excel(writer, sheet_name="Normalized_Psi")
            cov_df.to_excel(writer, sheet_name="Covariance_Eta")
            all_eigen_df.to_excel(writer, index=False, sheet_name="Eigenvalues")
            eigen_df.to_excel(writer, index=False, sheet_name="Eigenvectors")
            result_df.to_excel(writer, index=False, sheet_name="Expert_Weights")

        st.download_button(
            "Download Excel",
            data=out.getvalue(),
            file_name="expert_covariance_sorted_eigenvector.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def page_dfs_ahp():
    render_module_banner("🧮", "DFS-AHP Scenario → Expert → Scenario Fusion", "Integrate scenario probabilities, expert judgments, and decomposed dual-fuzzy weights into a unified fused priority structure.", badge="Module 3")
    engine = DFSAHP()

    strat_sent_df = st.session_state.get("strat_to_ahp_df")
    if isinstance(strat_sent_df, pd.DataFrame) and not strat_sent_df.empty:
        st.info("Stratification probabilities are loaded from Module 1. You can keep them or edit them below.")
        st.dataframe(strat_sent_df[["Scenario ID", "Label", "Probability"]], use_container_width=True, hide_index=True)

        loaded_signature = tuple(
            (str(row["Scenario ID"]), float(row["Probability"]))
            for _, row in strat_sent_df.iterrows()
        )
        if st.session_state.get("ahp_loaded_signature") != loaded_signature:
            st.session_state["ahp_s"] = int(len(strat_sent_df))
            for key in list(st.session_state.keys()):
                if key.startswith("ahp_sp_"):
                    del st.session_state[key]
            for idx, p in enumerate(strat_sent_df["Probability"].tolist()):
                st.session_state[f"ahp_sp_{idx}"] = float(p)
            st.session_state["ahp_loaded_signature"] = loaded_signature

    loaded_probs = []
    loaded_titles = []
    if isinstance(strat_sent_df, pd.DataFrame) and not strat_sent_df.empty:
        loaded_probs = strat_sent_df["Probability"].astype(float).tolist()
        loaded_titles = [
            f'{row["Scenario ID"]} - {row["Label"]}'
            for _, row in strat_sent_df.iterrows()
        ]

    st.markdown('<div class="config-shell">', unsafe_allow_html=True)
    st.markdown("### Configuration")
    st.markdown('<div class="config-lead">Set the DFS-AHP problem size, choose the main CR display rule, and adjust scenario transition probabilities directly inside the module.</div>', unsafe_allow_html=True)
    cfg1, cfg2, cfg3, cfg4 = st.columns([1, 1, 1, 1])
    with cfg1:
        num_criteria = int(st.number_input("Number of Criteria", min_value=2, max_value=15, value=9, step=1, key="ahp_n"))
    with cfg2:
        num_scenarios = int(st.number_input("Number of Scenarios", min_value=1, max_value=10, value=1, step=1, key="ahp_s"))
    with cfg3:
        num_experts = int(st.number_input("Number of Experts per Scenario", min_value=1, max_value=8, value=1, step=1, key="ahp_e"))
    with cfg4:
        cr_threshold = st.number_input("CR threshold", min_value=0.0, max_value=1.0, value=0.10, step=0.01, key="ahp_thr")

    cr_method = st.radio(
        "CR to use as main (display/pass):",
        ["Eigenvalue method (CR′)", "Geometric mean method"],
        index=0,
        key="ahp_cr_method",
        horizontal=True,
    )
    use_key = "EIGEN" if "Eigenvalue" in cr_method else "GM"

    st.markdown("#### Scenario transition probabilities")
    scen_probs_raw = []
    prob_cols = st.columns(min(max(int(num_scenarios), 1), 3))
    for s in range(int(num_scenarios)):
        label = loaded_titles[s] if s < len(loaded_titles) else f"Scenario {s+1}"
        default_value = float(loaded_probs[s]) if s < len(loaded_probs) else 1.0
        with prob_cols[s % len(prob_cols)]:
            scen_probs_raw.append(
                st.number_input(
                    f"{label} probability",
                    min_value=0.0,
                    value=default_value,
                    step=0.1,
                    key=f"ahp_sp_{s}",
                )
            )
    scen_probs = normalize_weights(scen_probs_raw)
    st.caption("Normalized scenario probabilities: " + ", ".join([f"S{s+1}={p:.4f}" for s, p in enumerate(scen_probs)]))
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Criteria names")
    criteria = []
    cols = st.columns(3)
    for i in range(int(num_criteria)):
        with cols[i % 3]:
            criteria.append(st.text_input(f"Criterion {i+1}", value=f"KPI{i+1}", key=f"ahp_crit_{i}"))

    store_key = f"ahp|{num_criteria}|{num_scenarios}|{num_experts}|" + "|".join(criteria)
    if st.session_state.get("ahp_store_key") != store_key:
        st.session_state["ahp_store_key"] = store_key
        st.session_state["ahp_mats"] = {
            f"scenario_{s}_expert_{e}": make_blank_ahp_matrix(criteria)
            for s in range(int(num_scenarios))
            for e in range(int(num_experts))
        }

    st.subheader("Paste-friendly matrix (O & P)")
    scenario_tab_titles = []
    for s in range(int(num_scenarios)):
        if s < len(loaded_titles):
            scenario_tab_titles.append(loaded_titles[s])
        else:
            scenario_tab_titles.append(f"Scenario {s+1}")
    scen_tabs = st.tabs(scenario_tab_titles)

    for s in range(int(num_scenarios)):
        with scen_tabs[s]:
            st.caption(f"Scenario normalized transition probability = {scen_probs[s]:.4f}")
            exp_tabs = st.tabs([f"Expert {e+1}" for e in range(int(num_experts))])

            for e in range(int(num_experts)):
                with exp_tabs[e]:
                    mat_key = f"scenario_{s}_expert_{e}"
                    df = st.data_editor(
                        st.session_state["ahp_mats"][mat_key],
                        height=420,
                        use_container_width=True,
                        key=f"ahp_ed_s{s}_e{e}",
                    )
                    for i, c in enumerate(criteria):
                        df.at[criteria[i], f"{c} (O)"] = "EEI"
                        df.at[criteria[i], f"{c} (P)"] = "EEI"
                    st.session_state["ahp_mats"][mat_key] = df

    st.subheader("Expert weights (used inside each scenario)")
    ew = []
    wcols = st.columns(min(int(num_experts), 4))
    for e in range(int(num_experts)):
        with wcols[e % len(wcols)]:
            ew.append(
                st.number_input(
                    f"Expert {e+1} weight",
                    min_value=0.0,
                    value=1.0,
                    step=0.1,
                    key=f"ahp_w_{e}",
                )
            )
    ew_norm = normalize_weights(ew)
    st.caption("Normalized expert weights: " + ", ".join([f"E{i+1}={w:.4f}" for i, w in enumerate(ew_norm)]))

    if st.button("Run DFS-AHP", type="primary", key="ahp_run"):
        all_invalid = set()
        scenario_expert_results = []
        scenario_expert_cr = []
        scenario_aggregated_results = []

        for s in range(int(num_scenarios)):
            exp_res_this_s = []
            exp_cr_this_s = []

            for e in range(int(num_experts)):
                mat_key = f"scenario_{s}_expert_{e}"
                res, invalid_terms, cr_pack = engine.compute_expert(
                    st.session_state["ahp_mats"][mat_key],
                    criteria,
                )
                exp_res_this_s.append(res)
                exp_cr_this_s.append(cr_pack)
                all_invalid.update(invalid_terms)

            scenario_expert_results.append(exp_res_this_s)
            scenario_expert_cr.append(exp_cr_this_s)

            agg_s = aggregate_experts_decomposed_dfs(exp_res_this_s, ew_norm, criteria, engine)
            scenario_aggregated_results.append(agg_s)

        final_fused_df, scen_probs_norm = aggregate_scenarios_decomposed_dfs(
            scenario_aggregated_results,
            scen_probs,
            criteria,
            engine,
        )

        st.session_state["ahp_criteria"] = criteria
        st.session_state["ahp_scenario_expert_results"] = scenario_expert_results
        st.session_state["ahp_scenario_expert_cr"] = scenario_expert_cr
        st.session_state["ahp_scenario_aggregated_results"] = scenario_aggregated_results
        st.session_state["ahp_final_fused_weights"] = final_fused_df
        st.session_state["ahp_scenario_probs_norm"] = scen_probs_norm
        st.session_state["ahp_final_weights"] = final_fused_df.copy()

        if all_invalid:
            st.warning("Unknown terms treated as neutral: " + ", ".join(sorted(all_invalid)))

        st.subheader("Per-scenario outputs")
        for s in range(int(num_scenarios)):
            st.markdown(f"## {scenario_tab_titles[s]}")
            st.caption(f"Normalized transition probability = {scen_probs_norm[s]:.6f}")

            for e in range(int(num_experts)):
                with st.expander(f"{scenario_tab_titles[s]} • Expert {e+1}", expanded=(s == 0 and e == 0)):
                    A = scenario_expert_cr[s][e]["A"]
                    A_df = pd.DataFrame(A, index=criteria, columns=criteria)
                    A_disp = A_df.copy()
                    for i in range(len(criteria)):
                        for j in range(len(criteria)):
                            A_disp.iloc[i, j] = fmt_ratio(float(A_df.iloc[i, j]))
                    st.markdown("**CR matrix (Excel rule)**")
                    st.dataframe(A_disp, use_container_width=True)

                    gm = scenario_expert_cr[s][e]["GM"]
                    ev = scenario_expert_cr[s][e]["EIGEN"]
                    show_df = pd.DataFrame(
                        [
                            {
                                "Method": "Geometric mean",
                                "lambda_max": gm["lambda_max"],
                                "CI": gm["CI_AHP"],
                                "RI": gm["RI"],
                                "CR": gm["CR"],
                                "Pass (<=thr)": "✅" if gm["CR"] <= cr_threshold else "❌",
                            },
                            {
                                "Method": "Eigenvalue (CR′)",
                                "lambda_max": ev["lambda_max"],
                                "CI": ev["CI_AHP"],
                                "RI": ev["RI"],
                                "CR′": ev["CR"],
                                "Pass (<=thr)": "✅" if ev["CR"] <= cr_threshold else "❌",
                            },
                        ]
                    )
                    for col in ["lambda_max", "CI", "RI", "CR", "CR′"]:
                        if col in show_df.columns:
                            show_df[col] = pd.to_numeric(show_df[col], errors="coerce").round(6)
                    st.dataframe(show_df, use_container_width=True)

                    st.markdown(f"**AHP weights used for selected method: {cr_method}**")
                    w = scenario_expert_cr[s][e][use_key]["weights"]
                    st.dataframe(
                        pd.DataFrame({"Criterion": criteria, "AHP Weight": np.round(w, 6)}),
                        use_container_width=True,
                    )

                    st.markdown("**DFS decomposed results (expert level)**")
                    show = scenario_expert_results[s][e].copy()
                    for col in ["a (μO)", "b (νO)", "c (μP)", "d (νP)", "CI (DFS)", "SI", "Normalized Weight"]:
                        show[col] = show[col].astype(float).round(6)
                    st.dataframe(show, use_container_width=True)
                    st.bar_chart(
                        scenario_expert_results[s][e].set_index("Criterion")[["Normalized Weight"]],
                        use_container_width=True,
                    )

            st.markdown("### Scenario aggregate across experts")
            st.caption(f"For this block, m = number of experts = {int(num_experts)}")
            show_agg = scenario_aggregated_results[s].copy()
            for col in ["a (μO)", "b (νO)", "c (μP)", "d (νP)", "CI (DFS)", "SI", "Normalized Weight"]:
                show_agg[col] = show_agg[col].astype(float).round(6)
            st.dataframe(show_agg, use_container_width=True)
            st.bar_chart(
                scenario_aggregated_results[s].set_index("Criterion")[["Normalized Weight"]],
                use_container_width=True,
            )
            st.divider()

        st.subheader("Final DFS fusion across scenarios")
        st.caption(f"For this block, m = number of scenarios = {int(num_scenarios)}")

        final_show = final_fused_df.copy()
        final_show["Final Weight"] = final_show["Normalized Weight"]
        for col in ["a (μO)", "b (νO)", "c (μP)", "d (νP)", "CI (DFS)", "SI", "Normalized Weight", "Final Weight"]:
            if col in final_show.columns:
                final_show[col] = final_show[col].astype(float).round(6)
        st.dataframe(final_show, use_container_width=True)
        st.bar_chart(final_fused_df.set_index("Criterion")[["Normalized Weight"]], use_container_width=True)

    st.subheader("Send RC decomposed weights (a,b,c,d) to DFS-QFD")
    if "ahp_final_fused_weights" not in st.session_state:
        st.info("Run DFS-AHP first, then you can send decomposed values to DFS-QFD.")
        return

    criteria0 = st.session_state.get("ahp_criteria", criteria)

    send_options = ["Final fused scenario DFS"]
    if "ahp_scenario_aggregated_results" in st.session_state:
        for s in range(len(st.session_state["ahp_scenario_aggregated_results"])):
            send_options.append(f"Scenario {s+1} aggregated DFS")
    if "ahp_scenario_expert_results" in st.session_state:
        for s in range(len(st.session_state["ahp_scenario_expert_results"])):
            for e in range(len(st.session_state["ahp_scenario_expert_results"][s])):
                send_options.append(f"Scenario {s+1} - Expert {e+1}")

    pick = st.selectbox(
        "Choose which decomposed DFS values to send",
        options=send_options,
        index=0,
        key="ahp_send_pick",
    )

    if st.button("Send to DFS-QFD RC weights", key="ahp_send_btn"):
        if pick == "Final fused scenario DFS":
            src_df = st.session_state["ahp_final_fused_weights"].copy()
        elif "aggregated DFS" in pick:
            s_idx = int(pick.split()[1]) - 1
            src_df = st.session_state["ahp_scenario_aggregated_results"][s_idx].copy()
        else:
            s_idx = int(pick.split()[1]) - 1
            e_idx = int(pick.split()[-1]) - 1
            src_df = st.session_state["ahp_scenario_expert_results"][s_idx][e_idx].copy()

        rc_weight_df = decomposed_weight_table_to_rc_df(src_df, criteria0)
        st.session_state["qfd_rc_weight_df"] = rc_weight_df
        st.success("Sent decomposed RC weights to Module 4 (DFS-QFD).")


def page_dfs_qfd():
    render_module_banner("📊", "DFS-QFD Analysis", "Map requirement criteria to mitigation strategies and compute final ReP-based prioritization through dual-fuzzy relationship modeling.", badge="Module 4")

    st.markdown('<div class="config-shell">', unsafe_allow_html=True)
    st.markdown("### Configuration")
    st.markdown('<div class="config-lead">Configure requirement criteria, mitigation strategies, and expert weights directly inside the DFS-QFD module so the sidebar stays focused on navigation and researcher information.</div>', unsafe_allow_html=True)
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        n_rc = int(st.number_input("Number of RCs", min_value=2, max_value=50, value=9, step=1, key="qfd_nrc"))
    with cfg2:
        n_ms = int(st.number_input("Number of MSs", min_value=2, max_value=50, value=9, step=1, key="qfd_nms"))
    with cfg3:
        n_exp = int(st.number_input("Number of Experts", min_value=1, max_value=10, value=1, step=1, key="qfd_nexp"))

    st.markdown("#### Expert weights")
    exp_ws = []
    weight_cols = st.columns(min(max(int(n_exp), 1), 4))
    for e in range(n_exp):
        with weight_cols[e % len(weight_cols)]:
            exp_ws.append(
                float(
                    st.number_input(
                        f"Expert {e+1} weight",
                        min_value=0.0,
                        value=1.0,
                        step=0.1,
                        key=f"qfd_w_{e}",
                    )
                )
            )
    exp_ws = normalize_weights(exp_ws)
    st.caption("Normalized expert weights: " + ", ".join([f"E{idx+1}={w:.3f}" for idx, w in enumerate(exp_ws)]))
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("RC & MS names")
    c1, c2 = st.columns(2)

    with c1:
        rc_names = []
        ahp_rc_defaults = []
        if "qfd_rc_weight_df" in st.session_state:
            ahp_rc_defaults = st.session_state["qfd_rc_weight_df"]["RC"].astype(str).tolist()
        for i in range(n_rc):
            default = ahp_rc_defaults[i] if i < len(ahp_rc_defaults) else f"RC{i+1}"
            rc_names.append(st.text_input(f"RC {i+1}", value=default, key=f"qfd_rc_name_{i}"))

    with c2:
        ms_names = []
        for j in range(n_ms):
            ms_names.append(st.text_input(f"MS {j+1}", value=f"MS{j+1}", key=f"qfd_ms_name_{j}"))

    st.subheader("DFS linguistic scale")
    scale = default_scale_uv()
    scale_df = pd.DataFrame([{"Term": k, "u (μ)": v[0], "v (ν)": v[1]} for k, v in scale.items()]).sort_values("Term")
    st.dataframe(scale_df, use_container_width=True, height=280)

    st.subheader("Paste RC weights (DFS decomposed)")
    if "qfd_rc_weight_df" not in st.session_state:
        st.session_state["qfd_rc_weight_df"] = init_rc_weight_df(rc_names)

    df0 = st.session_state["qfd_rc_weight_df"]
    if len(df0) != len(rc_names) or list(df0["RC"].astype(str)) != rc_names:
        st.session_state["qfd_rc_weight_df"] = init_rc_weight_df(rc_names)

    rc_weight_df = st.data_editor(
        st.session_state["qfd_rc_weight_df"],
        use_container_width=True,
        num_rows="fixed",
        key="qfd_rc_weight_editor",
    )
    st.session_state["qfd_rc_weight_df"] = rc_weight_df

    rc_weights_dfs: List[DFS] = []
    for _, row in rc_weight_df.iterrows():
        rc_weights_dfs.append(
            DFS(
                safe_float(row["mu_O"]),
                safe_float(row["nu_O"]),
                safe_float(row["mu_P"]),
                safe_float(row["nu_P"]),
            ).clip()
        )

    st.subheader("Paste ICj (expected cost) for RePj calculation")
    if "qfd_ic_df" not in st.session_state:
        st.session_state["qfd_ic_df"] = init_cost_df(ms_names)

    ic0 = st.session_state["qfd_ic_df"]
    if len(ic0) != len(ms_names) or list(ic0["MS"].astype(str)) != ms_names:
        st.session_state["qfd_ic_df"] = init_cost_df(ms_names)

    ic_df = st.data_editor(
        st.session_state["qfd_ic_df"],
        use_container_width=True,
        num_rows="fixed",
        key="qfd_ic_editor",
    )
    st.session_state["qfd_ic_df"] = ic_df
    ic_vals = [max(safe_float(v, 1.0), 1e-9) for v in ic_df["ICj"].tolist()]

    st.subheader("Relationship matrices (RC × MS) per expert (O/P terms)")
    rel_dfs_by_expert: List[List[List[DFS]]] = []

    for e in range(n_exp):
        st.markdown(f"### Expert {e+1}")
        key = f"qfd_rel_df_exp_{e}"
        if key not in st.session_state:
            st.session_state[key] = init_relationship_df(rc_names, ms_names)

        dfrel = st.session_state[key]
        if len(dfrel) != len(rc_names) or list(dfrel["RC"].astype(str)) != rc_names:
            st.session_state[key] = init_relationship_df(rc_names, ms_names)

        rel_df = st.data_editor(
            st.session_state[key],
            use_container_width=True,
            height=320,
            num_rows="fixed",
            key=f"qfd_rel_editor_{e}",
        )
        st.session_state[key] = rel_df

        mat_e: List[List[DFS]] = []
        for i in range(n_rc):
            row_dfs = []
            for j in range(n_ms):
                tO = str(rel_df.loc[i, f"{ms_names[j]}_O"]).strip()
                tP = str(rel_df.loc[i, f"{ms_names[j]}_P"]).strip()
                row_dfs.append(term_to_dfs(tO, tP, scale))
            mat_e.append(row_dfs)
        rel_dfs_by_expert.append(mat_e)
        st.divider()

    if st.button("Compute DFS-QFD outputs", type="primary", key="qfd_run"):
        combined_rel = [[None for _ in range(n_ms)] for _ in range(n_rc)]
        for i in range(n_rc):
            for j in range(n_ms):
                values = [rel_dfs_by_expert[e][i][j] for e in range(n_exp)]
                combined_rel[i][j] = weighted_componentwise_dfs_aggregate(values, exp_ws)

        comb_rows = []
        for i, rc in enumerate(rc_names):
            row = {"RC": rc}
            for j, ms in enumerate(ms_names):
                v = combined_rel[i][j]
                row[f"{ms}_muO"] = round(v.mu_O, 6)
                row[f"{ms}_nuO"] = round(v.nu_O, 6)
                row[f"{ms}_muP"] = round(v.mu_P, 6)
                row[f"{ms}_nuP"] = round(v.nu_P, 6)
            comb_rows.append(row)
        combined_df = pd.DataFrame(comb_rows)
        st.session_state["qfd_combined_df"] = combined_df

        st.markdown("### Aggregated relationship matrix across experts (decomposed)")
        st.caption(f"Aggregation rule uses expert weights with m = {n_exp}")
        st.dataframe(combined_df, use_container_width=True, height=320)

        weighted_dfs = [[None for _ in range(n_ms)] for _ in range(n_rc)]
        for i in range(n_rc):
            w_i = rc_weights_dfs[i]
            for j in range(n_ms):
                weighted_dfs[i][j] = dfs_multiply(w_i, combined_rel[i][j])

        weighted_rows = []
        for i, rc in enumerate(rc_names):
            row = {"RC": rc}
            for j, ms in enumerate(ms_names):
                v = weighted_dfs[i][j]
                row[f"{ms}_muO"] = round(v.mu_O, 6)
                row[f"{ms}_nuO"] = round(v.nu_O, 6)
                row[f"{ms}_muP"] = round(v.mu_P, 6)
                row[f"{ms}_nuP"] = round(v.nu_P, 6)
            weighted_rows.append(row)
        weighted_df = pd.DataFrame(weighted_rows)
        st.session_state["qfd_weighted_df"] = weighted_df

        st.markdown("### Weighted aggregated matrix after DFS multiplication (S2)")
        st.dataframe(weighted_df, use_container_width=True, height=320)

        crisp_contrib = np.zeros((n_rc, n_ms), dtype=float)
        for i in range(n_rc):
            for j in range(n_ms):
                crisp_contrib[i, j] = score_defuzz_from_weighted(weighted_dfs[i][j])

        crisp_df = pd.DataFrame(crisp_contrib, index=rc_names, columns=ms_names).round(6)
        st.session_state["qfd_crisp_df"] = crisp_df
        st.markdown("### Defuzzified weighted aggregated matrix")
        st.dataframe(crisp_df, use_container_width=True, height=320)

        AIj = crisp_contrib.sum(axis=0)
        RIj = AIj / (float(AIj.sum()) if float(AIj.sum()) != 0 else 1.0)

        SyP = np.array([AIj[j] / ic_vals[j] for j in range(n_ms)], dtype=float)
        RePj = SyP / (float(SyP.sum()) if float(SyP.sum()) != 0 else 1.0)

        results = pd.DataFrame(
            {
                "MS": ms_names,
                "AIj": AIj,
                "RIj (normalized AIj)": RIj,
                "ICj": ic_vals,
                "SyP = AIj/ICj": SyP,
                "RePj (normalized SyP)": RePj,
            }
        )
        results["Rank (RePj)"] = results["RePj (normalized SyP)"].rank(ascending=False, method="dense").astype(int)
        results = results.sort_values("RePj (normalized SyP)", ascending=False)

        st.session_state["qfd_results"] = results

        st.markdown("### Final RePj results")
        st.dataframe(results.round(6), use_container_width=True)
        st.bar_chart(results.set_index("MS")["RePj (normalized SyP)"], use_container_width=True)

        st.subheader("Send RePj to MILP")
        if st.button("Send RePj to MILP inputs", key="qfd_send_rep"):
            rep_df = pd.DataFrame({"MS": ms_names, "ReP": RePj})
            st.session_state["milp_rep_df"] = rep_df
            st.session_state["milp_ms_names"] = ms_names
            st.success("Sent RePj to Module 5 (MILP).")

        st.subheader("Download DFS-QFD Excel")
        buf = io.BytesIO()
        try:
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                rc_weight_df.to_excel(writer, index=False, sheet_name="RC_Weights_DFS")
                combined_df.to_excel(writer, index=False, sheet_name="Aggregated_Relationship_DFS")
                weighted_df.to_excel(writer, index=False, sheet_name="Weighted_Aggregated_DFS")
                crisp_df.reset_index().rename(columns={"index": "RC"}).to_excel(
                    writer, index=False, sheet_name="Defuzzified_Weighted_Matrix"
                )
                results.to_excel(writer, index=False, sheet_name="Results")
            st.download_button(
                "Download Excel",
                data=buf.getvalue(),
                file_name="DFS_QFD_StageIII_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as ex:
            st.error(f"Excel export failed: {ex}")
            st.download_button(
                "Download Results (CSV)",
                results.to_csv(index=False).encode("utf-8"),
                file_name="results.csv",
                mime="text/csv",
            )


def page_milp():
    render_module_banner("🎯", "MILP Optimization", "Optimize the final strategy portfolio under budget and time constraints using expected values and pairwise savings effects.", badge="Module 5")

    st.markdown("### Configuration")
    cfg1, cfg2, cfg3, cfg4 = st.columns([1, 1, 1.15, 1.15])
    with cfg1:
        m = int(st.number_input("Number of mitigation strategies (m)", min_value=2, max_value=20, value=9, step=1, key="milp_m"))
    with cfg2:
        n_exp = int(st.number_input("Number of experts", min_value=1, max_value=6, value=1, step=1, key="milp_ne"))
    with cfg3:
        budget = st.number_input("Available budget δ", min_value=0.0, value=200000.0, step=1000.0, key="milp_budget")
    with cfg4:
        time_lim = st.number_input("Available time ∄ (months)", min_value=0.0, value=60.0, step=1.0, key="milp_time")

    expert_choices = [f"Expert {i}" for i in range(1, n_exp + 1)]
    cfg5, cfg6 = st.columns([1.6, 1.0])
    with cfg5:
        selected = st.multiselect(
            "Experts to include",
            options=expert_choices,
            default=[expert_choices[0]] if expert_choices else [],
            key="milp_sel",
        )
    with cfg6:
        scale_savings = st.selectbox(
            "Savings cost scaling",
            ["Use as entered", "Multiply ρᵢⱼ by 1000"],
            index=0,
            key="milp_scale",
        )
    selected_ids = [int(s.split()[-1]) for s in selected] if selected else ([1] if n_exp >= 1 else [])

    st.subheader("Mitigation strategy names")
    cols = st.columns(3)
    names = []
    ms_defaults = st.session_state.get("milp_ms_names", [])
    for i in range(m):
        with cols[i % 3]:
            default = ms_defaults[i] if i < len(ms_defaults) else f"MS{i+1}"
            names.append(st.text_input(f"Strategy {i+1}", value=default, key=f"milp_name_{i}"))

    key = f"milp|{m}|{n_exp}|" + "|".join(names)
    if st.session_state.get("milp_store_key") != key:
        st.session_state["milp_store_key"] = key

        if "milp_rep_df" in st.session_state and len(st.session_state["milp_rep_df"]) == m:
            st.session_state["milp_rep_df_local"] = st.session_state["milp_rep_df"].copy()
            st.session_state["milp_rep_df_local"]["MS"] = names
        else:
            st.session_state["milp_rep_df_local"] = pd.DataFrame({"MS": names, "ReP": [1.0] * m})

        st.session_state["milp_experts"] = {}
        for e in range(1, n_exp + 1):
            st.session_state["milp_experts"][e] = {
                "cost_omp": pd.DataFrame({"MS": names, "O": 0.0, "ML": 0.0, "P": 0.0}),
                "time_omp": pd.DataFrame({"MS": names, "O": 0.0, "ML": 0.0, "P": 0.0}),
                "rho_O": make_square_df(names, 0.0),
                "rho_ML": make_square_df(names, 0.0),
                "rho_P": make_square_df(names, 0.0),
                "zeta_O": make_square_df(names, 0.0),
                "zeta_ML": make_square_df(names, 0.0),
                "zeta_P": make_square_df(names, 0.0),
            }

    st.subheader("Input ReP (from DFS-QFD)")
    st.session_state["milp_rep_df_local"] = st.data_editor(
        st.session_state["milp_rep_df_local"],
        use_container_width=True,
        height=240,
        num_rows="fixed",
        key="milp_rep_editor",
    )

    st.caption("Configuration controls for this optimization model are shown here in the main workspace for easier review while editing inputs.")

    st.subheader("Enter expert O/ML/P data")
    st.info("Paste from Excel. For pairwise matrices fill ONLY upper triangle (i<j).")

    for e in range(1, n_exp + 1):
        with st.expander(f"Expert {e} inputs (O / ML / P)", expanded=(e == 1)):
            tabs = st.tabs(["Cost (Ce)", "Time (T)", "Saving cost ρᵢⱼ", "Saving time ζᵢⱼ"])

            with tabs[0]:
                st.session_state["milp_experts"][e]["cost_omp"] = st.data_editor(
                    st.session_state["milp_experts"][e]["cost_omp"],
                    use_container_width=True,
                    height=260,
                    num_rows="fixed",
                    key=f"milp_cost_{e}",
                )
                exp_cost = compute_expected_vector(st.session_state["milp_experts"][e]["cost_omp"])
                show = st.session_state["milp_experts"][e]["cost_omp"][["MS", "O", "ML", "P"]].copy()
                show["Expected"] = exp_cost
                st.caption("Expected cost = (O + P + 4×ML)/6")
                st.dataframe(show, use_container_width=True)

            with tabs[1]:
                st.session_state["milp_experts"][e]["time_omp"] = st.data_editor(
                    st.session_state["milp_experts"][e]["time_omp"],
                    use_container_width=True,
                    height=260,
                    num_rows="fixed",
                    key=f"milp_time_{e}",
                )
                exp_time = compute_expected_vector(st.session_state["milp_experts"][e]["time_omp"])
                show = st.session_state["milp_experts"][e]["time_omp"][["MS", "O", "ML", "P"]].copy()
                show["Expected"] = exp_time
                st.caption("Expected time = (O + P + 4×ML)/6")
                st.dataframe(show, use_container_width=True)

            with tabs[2]:
                sub = st.tabs(["ρ (O)", "ρ (ML)", "ρ (P)"])
                with sub[0]:
                    st.session_state["milp_experts"][e]["rho_O"] = st.data_editor(
                        st.session_state["milp_experts"][e]["rho_O"],
                        use_container_width=True,
                        height=320,
                        key=f"milp_rhoO_{e}",
                    )
                with sub[1]:
                    st.session_state["milp_experts"][e]["rho_ML"] = st.data_editor(
                        st.session_state["milp_experts"][e]["rho_ML"],
                        use_container_width=True,
                        height=320,
                        key=f"milp_rhoML_{e}",
                    )
                with sub[2]:
                    st.session_state["milp_experts"][e]["rho_P"] = st.data_editor(
                        st.session_state["milp_experts"][e]["rho_P"],
                        use_container_width=True,
                        height=320,
                        key=f"milp_rhoP_{e}",
                    )
                st.caption("Only upper triangle (i<j) is used.")

            with tabs[3]:
                sub = st.tabs(["ζ (O)", "ζ (ML)", "ζ (P)"])
                with sub[0]:
                    st.session_state["milp_experts"][e]["zeta_O"] = st.data_editor(
                        st.session_state["milp_experts"][e]["zeta_O"],
                        use_container_width=True,
                        height=320,
                        key=f"milp_zetaO_{e}",
                    )
                with sub[1]:
                    st.session_state["milp_experts"][e]["zeta_ML"] = st.data_editor(
                        st.session_state["milp_experts"][e]["zeta_ML"],
                        use_container_width=True,
                        height=320,
                        key=f"milp_zetaML_{e}",
                    )
                with sub[2]:
                    st.session_state["milp_experts"][e]["zeta_P"] = st.data_editor(
                        st.session_state["milp_experts"][e]["zeta_P"],
                        use_container_width=True,
                        height=320,
                        key=f"milp_zetaP_{e}",
                    )
                st.caption("Only upper triangle (i<j) is used.")

    if st.button("Solve optimization", type="primary", key="milp_solve"):
        if "milp_rep_df_local" not in st.session_state or len(st.session_state["milp_rep_df_local"]) != len(names):
            st.error("ReP table not ready. Please re-check inputs.")
            return

        rep = np.array([_to_float(v, 0.0) for v in st.session_state["milp_rep_df_local"]["ReP"].tolist()], dtype=float)

        exp_cost_list, exp_time_list, exp_rho_list, exp_zeta_list = [], [], [], []
        for e in selected_ids:
            cost_vec = compute_expected_vector(st.session_state["milp_experts"][e]["cost_omp"])
            time_vec = compute_expected_vector(st.session_state["milp_experts"][e]["time_omp"])
            rho = compute_expected_matrix(
                st.session_state["milp_experts"][e]["rho_O"],
                st.session_state["milp_experts"][e]["rho_ML"],
                st.session_state["milp_experts"][e]["rho_P"],
            )
            zeta = compute_expected_matrix(
                st.session_state["milp_experts"][e]["zeta_O"],
                st.session_state["milp_experts"][e]["zeta_ML"],
                st.session_state["milp_experts"][e]["zeta_P"],
            )
            exp_cost_list.append(cost_vec)
            exp_time_list.append(time_vec)
            exp_rho_list.append(rho)
            exp_zeta_list.append(zeta)

        if len(selected_ids) == 1:
            cost, time, rho, zeta = exp_cost_list[0], exp_time_list[0], exp_rho_list[0], exp_zeta_list[0]
            agg_note = f"Using Expert {selected_ids[0]} only (expected values)."
        else:
            cost = np.mean(np.stack(exp_cost_list, axis=0), axis=0)
            time = np.mean(np.stack(exp_time_list, axis=0), axis=0)
            rho = np.mean(np.stack(exp_rho_list, axis=0), axis=0)
            zeta = np.mean(np.stack(exp_zeta_list, axis=0), axis=0)
            agg_note = f"Using average expected values across experts: {', '.join([f'E{i}' for i in selected_ids])}"

        if scale_savings != "Use as entered":
            rho = rho * 1000.0

        best = solve_by_enumeration(rep, cost, time, rho, zeta, budget, time_lim)

        st.subheader("Results")
        st.success(agg_note)

        if best is None:
            st.error("No feasible solution found. Increase budget/time or adjust inputs.")
            return

        x = best["x"]
        selected_ms = [names[i] for i in range(len(names)) if x[i] == 1]

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Objective (Max ReP)", f"{best['obj']:.6f}")
        r2.metric("Total cost", f"{best['total_cost']:.3f}")
        r3.metric("Total time", f"{best['total_time']:.3f}")
        r4.metric("Selected count", f"{int(x.sum())}")

        st.write("**Selected strategies:** " + (", ".join(selected_ms) if selected_ms else "None"))

        breakdown = pd.DataFrame(
            [
                {
                    "Base cost (Σ Ce_j)": best["base_cost"],
                    "Saving cost (Σ ρ_ij)": best["sav_cost"],
                    "Total cost": best["total_cost"],
                    "Base time (Σ T_j)": best["base_time"],
                    "Saving time (Σ ζ_ij)": best["sav_time"],
                    "Total time": best["total_time"],
                }
            ]
        )
        st.dataframe(breakdown, use_container_width=True)

        out = pd.DataFrame(
            {
                "MS": names,
                "l_j": x.astype(int),
                "ReP_j": rep,
                "Ce_used": cost,
                "T_used": time,
                "ReP_j*l_j": rep * x,
            }
        )
        st.dataframe(out, use_container_width=True)

        pairs = []
        n = len(names)
        for i in range(n):
            for j in range(i + 1, n):
                if x[i] == 1 and x[j] == 1 and (rho[i, j] != 0 or zeta[i, j] != 0):
                    pairs.append({"Pair": f"{names[i]} & {names[j]}", "ρ_ij used": rho[i, j], "ζ_ij used": zeta[i, j]})
        st.dataframe(pd.DataFrame(pairs) if pairs else pd.DataFrame([{"Pairs": "None"}]), use_container_width=True)

        st.subheader("Download results (ZIP)")
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("selection.csv", out.to_csv(index=False))
            z.writestr("breakdown.csv", breakdown.to_csv(index=False))
            z.writestr("rho_used_upper.csv", pd.DataFrame(rho, index=names, columns=names).to_csv(index=True))
            z.writestr("zeta_used_upper.csv", pd.DataFrame(zeta, index=names, columns=names).to_csv(index=True))
            z.writestr("rep.csv", st.session_state["milp_rep_df_local"].to_csv(index=False))
            z.writestr("aggregation_note.txt", agg_note)

        st.download_button(
            "Download ZIP",
            data=zbuf.getvalue(),
            file_name="milp_results.zip",
            mime="application/zip",
        )



def main():
    apply_custom_styling()
    if not check_password():
        return

    with st.sidebar:
        st.markdown('<div class="sidebar-section-title">Workspace Navigation</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-note">Choose a module, adjust configuration controls, and export results from each research block.</div>', unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Select module",
        [
            "1) Stratification Modeler",
            "2) Expert Covariance Model",
            "3) DFS-AHP",
            "4) DFS-QFD",
            "5) MILP Optimization",
        ],
        index=0,
        key="nav_page",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Log out", key="logout_btn"):
        logout()

    render_sidebar_research_profiles()
    render_app_header()

    if page.startswith("1)"):
        page_stratification()
    elif page.startswith("2)"):
        page_expert_covariance_model()
    elif page.startswith("3)"):
        page_dfs_ahp()
    elif page.startswith("4)"):
        page_dfs_qfd()
    else:
        page_milp()

    render_footer()


if __name__ == "__main__":
    main()
