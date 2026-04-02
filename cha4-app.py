import base64
import csv
import hmac
import io
import zipfile
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import graphviz
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Protected Stratification → TrFS-BWM → Global Weights → TrFS-QFD → MILP DSS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# SciPy is required for TrFS-BWM LP models
try:
    from scipy.optimize import linprog
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
    linprog = None


# ============================================================
# Styling, authentication, sidebar profiles, and footer
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

.main-header {
    padding: 0.7rem 0 0.65rem 0;
    border-bottom: 2px solid rgba(31, 94, 255, 0.18);
    margin-bottom: 1.6rem;
    color: var(--text);
    letter-spacing: -0.03em;
}

.section-header {
    background: linear-gradient(90deg, rgba(240,247,255,0.95) 0%, rgba(231,241,255,0.95) 100%);
    padding: 0.85rem 1rem;
    border-radius: 14px;
    border-left: 4px solid var(--primary);
    margin: 1.35rem 0 1rem 0;
    color: var(--primary-strong);
    font-weight: 800;
    box-shadow: var(--shadow-soft);
}

.info-box {
    background-color: rgba(255, 255, 255, 0.84);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-soft);
}

.highlight {
    background: linear-gradient(90deg, #fff6d8 0%, #fff9e8 100%);
    padding: 0.7rem 0.85rem;
    border-radius: 12px;
    border-left: 4px solid #f59e0b;
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
.info-label,
.kpi-label {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.74rem;
    font-weight: 700;
    color: var(--primary-strong);
}

.hero-title {
    margin-top: 0.15rem;
    font-size: 2.15rem;
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.03em;
    color: var(--text);
}

.hero-subtitle {
    margin-top: 0.55rem;
    max-width: 920px;
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

.stButton > button {
    width: 100%;
    border: 0;
    border-radius: 12px;
    padding: 0.72rem 1.15rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    color: white;
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-strong) 100%);
    box-shadow: 0 10px 24px rgba(31, 94, 255, 0.24);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 28px rgba(31, 94, 255, 0.28);
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

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.9);
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-soft);
}

[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    color: var(--text);
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

.sidebar-section-title {
    color: #f8fbff !important;
    font-size: 0.82rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.11em;
    margin: 0.35rem 0 0.75rem 0;
}

.sidebar-section-note {
    color: #aac0db !important;
    font-size: 0.78rem;
    line-height: 1.5;
    margin: 0 0 0.85rem 0;
}

[data-testid="stSidebar"] div[data-testid="stExpander"] {
    background: transparent;
    border: none;
    box-shadow: none;
}

[data-testid="stSidebar"] div[data-testid="stExpander"] > details {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 16px;
    overflow: hidden;
}

[data-testid="stSidebar"] div[data-testid="stExpander"] summary {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 16px;
}

[data-testid="stSidebar"] div[data-testid="stExpander"] summary p,
[data-testid="stSidebar"] div[data-testid="stExpander"] summary svg {
    color: #f8fbff !important;
    fill: #f8fbff !important;
    font-weight: 700 !important;
}

.sidebar-profile-card {
    background: linear-gradient(180deg, rgba(19, 36, 67, 0.98) 0%, rgba(14, 28, 53, 0.98) 100%);
    border: 1px solid rgba(148, 163, 184, 0.20);
    border-radius: 18px;
    padding: 1rem;
    margin: 0.1rem 0 0.9rem 0;
    box-shadow: 0 14px 28px rgba(2, 8, 23, 0.28);
}

.sidebar-profile-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    color: #dbeafe !important;
    background: rgba(59, 130, 246, 0.18);
    border: 1px solid rgba(147, 197, 253, 0.22);
    border-radius: 999px;
    padding: 0.26rem 0.58rem;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

.sidebar-profile-image-frame {
    width: 100%;
    border-radius: 16px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.10);
    margin-bottom: 0.85rem;
}

.sidebar-profile-image {
    display: block;
    width: 100%;
    height: auto;
}

.sidebar-profile-image-placeholder {
    padding: 1.1rem 0.85rem;
    text-align: center;
    color: #dbe7f5 !important;
    font-size: 0.78rem;
}

.sidebar-profile-name {
    color: #ffffff !important;
    font-size: 1.08rem;
    font-weight: 800;
    margin: 0 0 0.18rem 0;
    line-height: 1.35;
}

.sidebar-profile-role {
    color: #c7ddff !important;
    font-size: 0.84rem;
    font-weight: 700;
    line-height: 1.45;
    margin-bottom: 0.16rem;
}

.sidebar-profile-institution {
    color: #e5eefb !important;
    font-size: 0.8rem;
    line-height: 1.45;
    margin-bottom: 0.62rem;
}

.sidebar-profile-text,
.sidebar-profile-bullet,
.sidebar-profile-bio {
    color: #d9e5f5 !important;
    font-size: 0.8rem;
    line-height: 1.65;
}

.sidebar-profile-text {
    margin-bottom: 0.55rem;
}

.sidebar-profile-bullet {
    margin-bottom: 0.22rem;
}

.sidebar-profile-divider {
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    margin: 0.72rem 0 0.68rem 0;
}

.login-page {
    position: relative;
    max-width: 920px;
    margin: 1.9rem auto 0 auto;
    padding: 0 0.9rem 1.25rem 0.9rem;
}

.login-page::before,
.login-page::after {
    content: "";
    position: absolute;
    border-radius: 999px;
    filter: blur(12px);
    z-index: 0;
    pointer-events: none;
}

.login-page::before {
    top: 34px;
    left: -36px;
    width: 180px;
    height: 180px;
    background: radial-gradient(circle, rgba(37, 99, 235, 0.18) 0%, rgba(37, 99, 235, 0.02) 72%);
}

.login-page::after {
    bottom: 8px;
    right: -20px;
    width: 220px;
    height: 220px;
    background: radial-gradient(circle, rgba(20, 184, 166, 0.16) 0%, rgba(20, 184, 166, 0.02) 72%);
}

.login-hero-box {
    position: relative;
    z-index: 1;
    background:
        radial-gradient(circle at 86% 18%, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.02) 26%),
        radial-gradient(circle at 16% 110%, rgba(45, 212, 191, 0.18) 0%, rgba(45, 212, 191, 0.02) 28%),
        linear-gradient(135deg, #081225 0%, #153ea8 45%, #0ea5e9 100%);
    border-radius: 34px;
    padding: 1.3rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 28px 70px rgba(15, 23, 42, 0.16), 0 10px 30px rgba(37, 99, 235, 0.16);
    border: 1px solid rgba(255,255,255,0.18);
    overflow: hidden;
}

.login-hero-box::before,
.login-hero-box::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    pointer-events: none;
}

.login-hero-box::before {
    top: -68px;
    right: -52px;
    width: 220px;
    height: 220px;
    background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0.02) 72%);
}

.login-hero-box::after {
    bottom: -90px;
    left: -54px;
    width: 230px;
    height: 230px;
    background: radial-gradient(circle, rgba(129, 140, 248, 0.18) 0%, rgba(129, 140, 248, 0.02) 72%);
}

.login-hero-inner {
    position: relative;
    z-index: 1;
    min-height: 310px;
    border-radius: 26px;
    padding: 2.2rem 2.15rem 2rem 2.15rem;
    border: 1px solid rgba(255,255,255,0.08);
    background: linear-gradient(180deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%);
    backdrop-filter: blur(10px);
}

.login-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.42rem 0.85rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.14);
    color: rgba(255,255,255,0.96);
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.login-hero-box h1 {
    color: #ffffff !important;
    font-size: clamp(2.6rem, 5vw, 4rem);
    line-height: 1.02;
    margin: 1rem 0 0.95rem 0;
    letter-spacing: -0.055em;
    max-width: 760px;
}

.login-hero-box p {
    color: rgba(255,255,255,0.93) !important;
    margin: 0;
    max-width: 700px;
    font-size: 1.1rem;
    line-height: 1.72;
}

.login-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
    margin-top: 1.15rem;
}

.login-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.48rem 0.8rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #eff6ff;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.14);
    backdrop-filter: blur(10px);
}

.login-metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    margin-top: 1.35rem;
}

.login-metric-card {
    padding: 0.92rem 0.95rem;
    border-radius: 18px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
}

.login-metric-label {
    display: block;
    color: rgba(255,255,255,0.72);
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.login-metric-value {
    display: block;
    margin-top: 0.35rem;
    color: #ffffff;
    font-size: 1.08rem;
    font-weight: 800;
}

.login-metric-note {
    display: block;
    margin-top: 0.25rem;
    color: rgba(255,255,255,0.74);
    font-size: 0.82rem;
    line-height: 1.45;
}

.login-form-note {
    position: relative;
    z-index: 1;
    text-align: center;
    color: #475569;
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 1rem 0;
}

div[data-testid="stForm"] {
    position: relative;
    z-index: 1;
    background: linear-gradient(180deg, rgba(255,255,255,0.97) 0%, rgba(248,251,255,0.98) 100%);
    border: 1px solid rgba(219,228,240,0.92);
    border-radius: 28px;
    padding: 1.6rem 1.6rem 1.2rem 1.6rem;
    box-shadow: 0 24px 48px rgba(15, 23, 42, 0.08), 0 12px 26px rgba(31, 94, 255, 0.08);
    max-width: 920px;
    margin: 0 auto;
}

div[data-testid="stForm"] label p {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
}

div[data-testid="stForm"] .stTextInput input {
    min-height: 60px;
    font-size: 1rem;
    border-radius: 16px;
    padding: 0.92rem 1rem;
    border: 1px solid #d8e2ef;
    background: rgba(248,250,252,0.95);
}

div[data-testid="stForm"] .stButton > button,
div[data-testid="stForm"] .stFormSubmitButton > button {
    min-height: 56px;
    border-radius: 16px;
    font-size: 1.05rem;
    font-weight: 700;
}

.login-helper {
    position: relative;
    z-index: 1;
    display: flex;
    justify-content: center;
    margin-top: 0.75rem;
}

.login-helper span {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    border-radius: 999px;
    padding: 0.42rem 0.8rem;
    background: rgba(255,255,255,0.72);
    color: #64748b;
    font-size: 0.8rem;
    border: 1px solid rgba(219,228,240,0.92);
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

@media (max-width: 1200px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 900px) {
    .login-page {
        max-width: 100%;
        margin-top: 1.15rem;
        padding-left: 0.25rem;
        padding-right: 0.25rem;
    }

    .login-hero-box {
        border-radius: 26px;
        padding: 0.8rem;
    }

    .login-hero-inner {
        min-height: auto;
        padding: 1.45rem 1.1rem 1.3rem 1.1rem;
        border-radius: 22px;
    }

    .login-hero-box h1 {
        font-size: 2.2rem;
        line-height: 1.08;
    }

    .login-hero-box p {
        font-size: 0.98rem;
    }

    .login-metrics {
        grid-template-columns: 1fr;
    }
}
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

    left, center, right = st.columns([0.25, 4.4, 0.25])
    with center:
        st.markdown(
            """
            <div class="login-page">
                <div class="login-hero-box">
                    <div class="login-hero-inner">
                        <div class="login-badge">Secure research workspace</div>
                        <h1>Protected System-Level TrFS Decision Support Studio</h1>
                        <p>
                            Access the integrated workspace for stratification, scenario-aware TrFS-BWM,
                            global weight synthesis, TrFS-QFD prioritization, and MILP strategy optimization
                            in one premium environment.
                        </p>
                        <div class="login-pill-row">
                            <span class="login-pill">Stratification</span>
                            <span class="login-pill">TrFS-BWM</span>
                            <span class="login-pill">Global Weights</span>
                            <span class="login-pill">TrFS-QFD</span>
                            <span class="login-pill">MILP Optimization</span>
                        </div>
                        <div class="login-metrics">
                            <div class="login-metric-card">
                                <span class="login-metric-label">Modules</span>
                                <span class="login-metric-value">5 linked analytics modules</span>
                                <span class="login-metric-note">From scenario logic to portfolio selection.</span>
                            </div>
                            <div class="login-metric-card">
                                <span class="login-metric-label">Access</span>
                                <span class="login-metric-value">Password protected</span>
                                <span class="login-metric-note">Controlled sign-in for protected research work.</span>
                            </div>
                            <div class="login-metric-card">
                                <span class="login-metric-label">Outputs</span>
                                <span class="login-metric-value">Decision-ready results</span>
                                <span class="login-metric-note">Weights, rankings, exports, and optimization insights.</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="login-form-note">
                    Sign in with the application password to continue
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

        st.markdown(
            '<div class="login-helper"><span>🔒 Protected workspace · Researcher profile and copyright footer enabled after sign-in</span></div>',
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


def get_image_data_uri(image_path: Path) -> str | None:
    image_path = Path(image_path)
    if not image_path.exists():
        return None

    suffix = image_path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def render_sidebar_profile_card(container, name, role, institution, image_path, brief_text, full_bio=None, extras=None, tag="Researcher"):
    safe_name = escape(name)
    safe_role = escape(role)
    safe_institution = escape(institution)
    safe_brief = escape(brief_text)
    safe_tag = escape(tag)
    image_uri = get_image_data_uri(Path(image_path))

    if image_uri:
        image_html = f'<img class="sidebar-profile-image" src="{image_uri}" alt="{safe_name}">'
    else:
        image_html = '<div class="sidebar-profile-image-placeholder">Profile image not available</div>'

    extras_html = ""
    if extras:
        extras_html = '<div class="sidebar-profile-divider"></div>' + "".join(
            f'<div class="sidebar-profile-bullet">• {escape(item)}</div>' for item in extras
        )

    container.markdown(
        f"""
        <div class="sidebar-profile-card">
            <div class="sidebar-profile-badge">👤 {safe_tag}</div>
            <div class="sidebar-profile-image-frame">{image_html}</div>
            <div class="sidebar-profile-name">{safe_name}</div>
            <div class="sidebar-profile-role">{safe_role}</div>
            <div class="sidebar-profile-institution">{safe_institution}</div>
            <div class="sidebar-profile-text">{safe_brief}</div>
            {extras_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if full_bio:
        with container.expander(f"More about {name}", expanded=False):
            st.markdown(
                f'<div class="sidebar-profile-bio">{escape(full_bio)}</div>',
                unsafe_allow_html=True,
            )


def render_sidebar_research_profiles():
    sidebar_box = st.sidebar.container()
    sidebar_box.markdown("---")
    sidebar_box.markdown(
        '<div class="sidebar-section-title">Researcher Profiles</div>',
        unsafe_allow_html=True,
    )
    sidebar_box.markdown(
        '<div class="sidebar-section-note">Profiles stay fixed in the left panel with improved contrast for readability.</div>',
        unsafe_allow_html=True,
    )

    render_sidebar_profile_card(
        sidebar_box,
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
        tag="Lead Researcher",
    )

    render_sidebar_profile_card(
        sidebar_box,
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
        tag="Co-Researcher",
    )


def render_app_header():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Protected decision analytics workspace</div>
            <div class="hero-title">System-Level TrFS-SBWM-QFD-MILP Decision Support Model</div>
            <div class="hero-subtitle">
                Integrated advanced fuzzy decision-support toolkit for scenario stratification, TrFS weighting,
                global sub-criteria synthesis, QFD prioritization, and portfolio optimization.
            </div>
            <div class="hero-grid">
                <div class="info-card">
                    <span class="info-label">Workflow</span>
                    <strong>5 connected modules</strong>
                    <small>Stratification → TrFS-BWM → Global Weights → TrFS-QFD → MILP Optimization</small>
                </div>
                <div class="info-card">
                    <span class="info-label">Purpose</span>
                    <strong>Research-grade decision support</strong>
                    <small>Move from expert judgment and scenario logic to ranked strategies and optimized portfolios.</small>
                </div>
                <div class="info-card">
                    <span class="info-label">Output</span>
                    <strong>Clear, exportable analytics</strong>
                    <small>Generate weighted models, relationship matrices, crisp scores, and optimization-ready results.</small>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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


# ============================================================
# Small utilities
# ============================================================

def _to_float(x, default=0.0) -> float:
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


def make_square_df(names, fill=0.0):
    df = pd.DataFrame(fill, index=names, columns=names)
    np.fill_diagonal(df.values, 0.0)
    return df


def upper_triangle_only(mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mat)
    n = mat.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = mat[i, j]
    return out


def normalize_weights(ws: List[float]) -> List[float]:
    if len(ws) == 0:
        return []
    s = float(sum(ws))
    if s <= 0:
        return [1.0 / len(ws)] * len(ws)
    return [float(w) / s for w in ws]


def exact_sum_ok(ws: List[float], tol: float = 1e-6) -> bool:
    return abs(float(sum(ws)) - 1.0) <= tol


# ============================================================
# TrFS core
# ============================================================

@dataclass(frozen=True)
class TrFS:
    w1: float
    w2: float
    w3: float
    w4: float

    def clip(self) -> "TrFS":
        a = np.clip([self.w1, self.w2, self.w3, self.w4], 0.0, 1.0)
        a = np.maximum.accumulate(a)  # enforce w1<=w2<=w3<=w4
        return TrFS(float(a[0]), float(a[1]), float(a[2]), float(a[3]))


def trfs_add(a: TrFS, b: TrFS) -> TrFS:
    return TrFS(a.w1 + b.w1, a.w2 + b.w2, a.w3 + b.w3, a.w4 + b.w4).clip()


def trfs_scalar_mul(s: float, a: TrFS) -> TrFS:
    return TrFS(s * a.w1, s * a.w2, s * a.w3, s * a.w4).clip()


def trfs_mul(a: TrFS, b: TrFS) -> TrFS:
    return TrFS(a.w1 * b.w1, a.w2 * b.w2, a.w3 * b.w3, a.w4 * b.w4).clip()


def gmIR(a: TrFS) -> float:
    return float((a.w1 + 2.0 * a.w2 + 2.0 * a.w3 + a.w4) / 6.0)


def trfs_to_str(t: TrFS, nd=4) -> str:
    return f"({t.w1:.{nd}f}, {t.w2:.{nd}f}, {t.w3:.{nd}f}, {t.w4:.{nd}f})"


def weighted_trfs_average(values: List[TrFS], weights: List[float]) -> TrFS:
    """
    Weighted TrFS aggregation using the operators already defined in the code:
      aggregated = Σ (weight_i ⊗ value_i)
    where scalar multiplication is trfs_scalar_mul and sum is trfs_add.
    """
    acc = TrFS(0.0, 0.0, 0.0, 0.0)
    for v, w in zip(values, weights):
        acc = trfs_add(acc, trfs_scalar_mul(float(w), v))
    return acc.clip()


def simple_trfs_average(values: List[TrFS]) -> TrFS:
    """
    Step 11 TrFS aggregation for QFD: arithmetic mean across experts.
    χ_ij = (Ση/e, Σπ/e, Σκ/e, Σγ/e)
    """
    if not values:
        return TrFS(0.0, 0.0, 0.0, 0.0)
    n = float(len(values))
    return TrFS(
        sum(v.w1 for v in values) / n,
        sum(v.w2 for v in values) / n,
        sum(v.w3 for v in values) / n,
        sum(v.w4 for v in values) / n,
    ).clip()


# ============================================================
# QFD Linguistic Scale
# ============================================================

def default_trfs_scale() -> pd.DataFrame:
    data = [
        ("EL", 0.0, 0.1, 0.1, 0.2),
        ("VL", 0.1, 0.2, 0.3, 0.4),
        ("L",  0.2, 0.3, 0.4, 0.5),
        ("ML", 0.3, 0.4, 0.5, 0.6),
        ("M",  0.4, 0.5, 0.6, 0.7),
        ("MH", 0.5, 0.6, 0.7, 0.8),
        ("H",  0.6, 0.7, 0.8, 0.9),
        ("VH", 0.7, 0.8, 0.9, 1.0),
        ("EH", 0.8, 0.9, 1.0, 1.0),
    ]
    return pd.DataFrame(data, columns=["Term", "w1", "w2", "w3", "w4"])


def scale_df_to_dict(scale_df: pd.DataFrame) -> Dict[str, TrFS]:
    out: Dict[str, TrFS] = {}
    for _, r in scale_df.iterrows():
        t = str(r["Term"]).strip()
        out[t] = TrFS(float(r["w1"]), float(r["w2"]), float(r["w3"]), float(r["w4"])).clip()
    return out


def init_weights_df(ch_names: List[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "Challenge": ch_names,
        "w1": ["0.25000"] * len(ch_names),
        "w2": ["0.35000"] * len(ch_names),
        "w3": ["0.45000"] * len(ch_names),
        "w4": ["0.55000"] * len(ch_names),
    })


def weights_df_for_editor(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Challenge" in out.columns:
        out["Challenge"] = out["Challenge"].astype(str)
    for c in ["w1", "w2", "w3", "w4"]:
        if c in out.columns:
            def _fmt(v):
                s = str(v).strip()
                if s == "" or s.lower() in {"nan", "none"}:
                    return ""
                try:
                    return f"{float(s):.5f}"
                except Exception:
                    return s
            out[c] = out[c].apply(_fmt)
    return out


def standardize_qfd_weight_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a QFD-ready Challenge/w1..w4 DataFrame formatted for direct editing."""
    if df is None or len(df) == 0:
        raise ValueError("No weight data found.")

    out = df.copy()
    cols = {str(c).strip().lower(): c for c in out.columns}

    challenge_col = None
    for key in ["challenge", "sub-criterion", "sub criterion", "criterion", "name"]:
        if key in cols:
            challenge_col = cols[key]
            break
    if challenge_col is None:
        challenge_col = out.columns[0]

    needed = []
    for c in ["w1", "w2", "w3", "w4"]:
        if c not in cols:
            raise ValueError(f"Missing required column: {c}")
        needed.append(cols[c])

    qfd_df = out[[challenge_col] + needed].copy()
    qfd_df.columns = ["Challenge", "w1", "w2", "w3", "w4"]
    qfd_df["Challenge"] = qfd_df["Challenge"].astype(str).str.strip()
    for c in ["w1", "w2", "w3", "w4"]:
        qfd_df[c] = qfd_df[c].map(lambda v: f"{_to_float(v, 0.0):.5f}")
    return qfd_df


def make_rij_editor_df(strategy_names: List[str], rij_values) -> pd.DataFrame:
    arr = np.asarray(rij_values, dtype=float).reshape(-1)
    n = len(strategy_names)
    if len(arr) != n:
        if n == 0:
            arr = np.array([], dtype=float)
        else:
            arr = np.ones(n, dtype=float) / n
    df = pd.DataFrame({"Strategy": strategy_names, "RIj": arr})
    df["Rank"] = df["RIj"].rank(ascending=False, method="dense").astype(int)
    return df


def init_rel_df(ch_names: List[str], sy_names: List[str], terms: List[str]) -> pd.DataFrame:
    df = pd.DataFrame({"Challenge": ch_names})
    for s in sy_names:
        df[s] = terms[0] if terms else "M"
    return df


def compute_trfs_qfd(
    ch_weights: List[TrFS],
    rel_by_expert: List[List[List[TrFS]]],  # [e][i][j]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[TrFS]], List[List[TrFS]]]:
    """
    Expert-only QFD aggregation using simple arithmetic averaging across experts.
    """
    n_exp = len(rel_by_expert)
    n_ch = len(ch_weights)
    n_sy = len(rel_by_expert[0][0])

    agg_rel = [[TrFS(0, 0, 0, 0) for _ in range(n_sy)] for _ in range(n_ch)]
    for i in range(n_ch):
        for j in range(n_sy):
            vals = [rel_by_expert[e][i][j] for e in range(n_exp)]
            agg_rel[i][j] = simple_trfs_average(vals)

    weighted_trfs = [[TrFS(0, 0, 0, 0) for _ in range(n_sy)] for _ in range(n_ch)]
    crisp = np.zeros((n_ch, n_sy), dtype=float)
    for i in range(n_ch):
        for j in range(n_sy):
            weighted_trfs[i][j] = trfs_mul(ch_weights[i], agg_rel[i][j])
            crisp[i, j] = gmIR(weighted_trfs[i][j])

    AIj = crisp.sum(axis=0)
    denom = float(AIj.sum()) if float(AIj.sum()) != 0 else 1.0
    RIj = AIj / denom

    return crisp, AIj, RIj, agg_rel, weighted_trfs


# ============================================================
# MILP (Enumeration solver) — Direct Average inputs
# ============================================================

def solve_by_enumeration(RIj, IC, IT, sigma, tau, budget, time_limit):
    n = len(RIj)
    best = None
    for mask in range(1 << n):
        q = np.array([(mask >> k) & 1 for k in range(n)], dtype=int)

        base_cost = float(np.dot(IC, q))
        base_time = float(np.dot(IT, q))

        sav_cost = 0.0
        sav_time = 0.0
        for i in range(n):
            if q[i] == 0:
                continue
            for j in range(i + 1, n):
                if q[j] == 0:
                    continue
                sav_cost += float(sigma[i, j])
                sav_time += float(tau[i, j])

        total_cost = base_cost - sav_cost
        total_time = base_time - sav_time

        if total_cost <= budget + 1e-9 and total_time <= time_limit + 1e-9:
            obj = float(np.dot(RIj, q))
            if (best is None) or (obj > best["obj"] + 1e-12):
                best = {
                    "q": q,
                    "obj": obj,
                    "total_cost": total_cost,
                    "total_time": total_time,
                    "base_cost": base_cost,
                    "base_time": base_time,
                    "sav_cost": sav_cost,
                    "sav_time": sav_time,
                }
    return best


# ============================================================
# 0) TrFS-BWM core
# ============================================================

TrF = Tuple[float, float, float, float]

def bwm_gmir(x: TrF) -> float:
    a, b, c, d = x
    return (a + 2*b + 2*c + d) / 6.0

def bwm_trf_str(x: TrF, nd: int = 4) -> str:
    return f"({x[0]:.{nd}f}, {x[1]:.{nd}f}, {x[2]:.{nd}f}, {x[3]:.{nd}f})"


LINGUISTIC_SCALE_BWM: Dict[str, TrF] = {
    "AI (Absolute importance)": (8, 8.5, 9, 9),
    "EXI (Extreme importance)": (7, 7.5, 8.5, 9),
    "VSI (Very strong importance)": (6, 6.5, 7.5, 8),
    "SI+ (Strong importance plus)": (5, 5.5, 6.5, 7),
    "SI (Strong importance)": (4, 4.5, 5.5, 6),
    "MI+ (Moderate importance plus)": (3, 3.5, 4.5, 5),
    "MI (Moderate importance)": (2, 2.5, 3.5, 4),
    "WI (Weak importance)": (1, 1.5, 2.5, 3),
    "EI (Equal importance)": (1, 1, 1, 1),
}
SCALE_OPTIONS_BWM = list(LINGUISTIC_SCALE_BWM.keys())

PSI_TABLE: Dict[TrF, Tuple[float, float]] = {
    (8, 8.5, 9, 9): (4.836, 5.034),
    (7, 7.5, 8.5, 9): (4.422, 4.457),
    (6, 6.5, 7.5, 8): (3.672, 3.712),
    (5, 5.5, 6.5, 7): (2.94, 2.985),
    (4, 4.5, 5.5, 6): (2.229, 2.281),
    (3, 3.5, 4.5, 5): (1.55, 1.607),
    (2, 2.5, 3.5, 4): (0.898, 0.975),
    (1, 1.5, 2.5, 3): (0.305, 0.406),
    (1, 1, 1, 1): (0.0, 0.0),
}

@dataclass
class ExpertInputBWM:
    best_idx: int
    worst_idx: int
    best_to: List[str]
    to_worst: List[str]


def solve_trfs_bwm_weights(
    mB: List[TrF],
    mW: List[TrF],
    best_idx: int,
    worst_idx: int,
    enforce_gmir_sum_eq: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str]]:
    if not SCIPY_OK:
        return None, None, "SciPy is not available. Install scipy to run TrFS-BWM."

    n = len(mB)

    def vid(i: int, k: int) -> int:
        return 4*i + (k-1)

    phi_id = 4*n
    num_vars = 4*n + 1

    A_ub = []
    b_ub = []

    def add_abs_le_phi(expr: Dict[int, float]):
        row_pos = np.zeros(num_vars)
        row_neg = np.zeros(num_vars)
        for j, coef in expr.items():
            row_pos[j] = coef
            row_neg[j] = -coef
        row_pos[phi_id] = -1.0
        row_neg[phi_id] = -1.0
        A_ub.append(row_pos); b_ub.append(0.0)
        A_ub.append(row_neg); b_ub.append(0.0)

    for j in range(n):
        a1, a2, a3, a4 = mB[j]
        add_abs_le_phi({vid(best_idx, 1): 1.0, vid(j, 4): -a1})
        add_abs_le_phi({vid(best_idx, 2): 1.0, vid(j, 3): -a2})
        add_abs_le_phi({vid(best_idx, 3): 1.0, vid(j, 2): -a3})
        add_abs_le_phi({vid(best_idx, 4): 1.0, vid(j, 1): -a4})

    for j in range(n):
        b1, b2, b3, b4 = mW[j]
        add_abs_le_phi({vid(j, 1): 1.0, vid(worst_idx, 4): -b1})
        add_abs_le_phi({vid(j, 2): 1.0, vid(worst_idx, 3): -b2})
        add_abs_le_phi({vid(j, 3): 1.0, vid(worst_idx, 2): -b3})
        add_abs_le_phi({vid(j, 4): 1.0, vid(worst_idx, 1): -b4})

    for i in range(n):
        row = np.zeros(num_vars); row[vid(i,1)] = 1.0; row[vid(i,2)] = -1.0
        A_ub.append(row); b_ub.append(0.0)
        row = np.zeros(num_vars); row[vid(i,2)] = 1.0; row[vid(i,3)] = -1.0
        A_ub.append(row); b_ub.append(0.0)
        row = np.zeros(num_vars); row[vid(i,3)] = 1.0; row[vid(i,4)] = -1.0
        A_ub.append(row); b_ub.append(0.0)

    for i in range(n):
        row = np.zeros(num_vars)
        row[vid(i,4)] = 1.0
        for k in range(n):
            if k != i:
                row[vid(k,1)] = 1.0
        A_ub.append(row); b_ub.append(1.0)

        row = np.zeros(num_vars)
        row[vid(i,1)] = -1.0
        for k in range(n):
            if k != i:
                row[vid(k,4)] = -1.0
        A_ub.append(row); b_ub.append(-1.0)

        row = np.zeros(num_vars)
        row[vid(i,3)] = 1.0
        for k in range(n):
            if k != i:
                row[vid(k,2)] = 1.0
        A_ub.append(row); b_ub.append(1.0)

        row = np.zeros(num_vars)
        row[vid(i,2)] = -1.0
        for k in range(n):
            if k != i:
                row[vid(k,3)] = -1.0
        A_ub.append(row); b_ub.append(-1.0)

    A_eq = None
    b_eq = None
    if enforce_gmir_sum_eq:
        A_eq = np.zeros((1, num_vars))
        for i in range(n):
            A_eq[0, vid(i,1)] += 1/6
            A_eq[0, vid(i,2)] += 2/6
            A_eq[0, vid(i,3)] += 2/6
            A_eq[0, vid(i,4)] += 1/6
        b_eq = np.array([1.0])

    c = np.zeros(num_vars)
    c[phi_id] = 1.0

    bounds = [(0.0, 1.0)] * (4*n) + [(0.0, None)]

    res = linprog(
        c=c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return None, None, res.message

    x = res.x
    W = x[:4*n].reshape(n, 4)
    phi = float(x[phi_id])
    return W, phi, None


def compute_cr(phi_star: float, used_scales: List[TrF]) -> Optional[float]:
    if phi_star is None or not used_scales:
        return None
    mx = max(used_scales, key=bwm_gmir)
    psi_a, psi_b = PSI_TABLE.get(mx, (None, None))
    if psi_a is None or psi_b is None or psi_a == 0.0 or psi_b == 0.0:
        return None
    return max(phi_star/psi_a, phi_star/psi_b)


def _bwm_criteria_input() -> List[str]:
    default = "C1\nC2\nC3\nC4"
    txt = st.text_area("Criteria (one per line)", value=default, height=120)
    return [c.strip() for c in txt.splitlines() if c.strip()]


def _bwm_default_df(criteria: List[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Best→Others (B2O)": ["MI (Moderate importance)"] * len(criteria),
            "Others→Worst (O2W)": ["MI (Moderate importance)"] * len(criteria),
        },
        index=criteria,
    )


def _bwm_criteria_signature(criteria: List[str]) -> str:
    return "||".join([str(c).strip() for c in criteria])


def _bwm_prepare_df(df: pd.DataFrame, criteria: List[str]) -> pd.DataFrame:
    base = _bwm_default_df(criteria)
    if df is None or len(df) == 0:
        return base

    out = df.copy()
    if "Criterion" in out.columns:
        out = out.set_index("Criterion")
    if list(out.columns) != ["Best→Others (B2O)", "Others→Worst (O2W)"]:
        keep = [c for c in ["Best→Others (B2O)", "Others→Worst (O2W)"] if c in out.columns]
        out = out[keep]
    out = out.reindex(criteria)
    for col in ["Best→Others (B2O)", "Others→Worst (O2W)"]:
        if col not in out.columns:
            out[col] = base[col]
        out[col] = out[col].fillna(base[col])
    return out[["Best→Others (B2O)", "Others→Worst (O2W)"]]


def _bwm_enforced_df(criteria: List[str], best: str, worst: str, df: pd.DataFrame) -> pd.DataFrame:
    out = _bwm_prepare_df(df, criteria)
    if best in out.index:
        out.loc[best, "Best→Others (B2O)"] = "EI (Equal importance)"
    if worst in out.index:
        out.loc[worst, "Others→Worst (O2W)"] = "EI (Equal importance)"
    return out


def _bwm_state_key(s: int, e: int) -> str:
    return f"bwm_state_s{s}_e{e}"


def _ensure_bwm_input_state(s: int, e: int, criteria: List[str]) -> Dict[str, object]:
    key = _bwm_state_key(s, e)
    sig = _bwm_criteria_signature(criteria)
    default_best = criteria[0]
    default_worst = criteria[-1] if len(criteria) > 1 else criteria[0]

    state = st.session_state.get(key)
    if not isinstance(state, dict) or state.get("criteria_sig") != sig:
        state = {
            "criteria_sig": sig,
            "best": default_best,
            "worst": default_worst,
            "df": _bwm_default_df(criteria),
        }
    else:
        if state.get("best") not in criteria:
            state["best"] = default_best
        if state.get("worst") not in criteria:
            state["worst"] = default_worst
        state["df"] = _bwm_prepare_df(state.get("df"), criteria)

    if state["best"] == state["worst"] and len(criteria) > 1:
        state["worst"] = next((c for c in criteria if c != state["best"]), criteria[-1])

    state["df"] = _bwm_enforced_df(criteria, str(state["best"]), str(state["worst"]), state["df"])
    st.session_state[key] = state
    return state


def _bwm_widget_version_key(s: int, e: int) -> str:
    return f"bwm_widget_ver_s{s}_e{e}"


def _bwm_widget_key(prefix: str, s: int, e: int) -> str:
    ver = int(st.session_state.get(_bwm_widget_version_key(s, e), 0))
    return f"{prefix}_s{s}_e{e}_v{ver}"


def _sync_bwm_widget_state(s: int, e: int, criteria: List[str]):
    _ensure_bwm_input_state(s, e, criteria)
    ver_key = _bwm_widget_version_key(s, e)
    st.session_state[ver_key] = int(st.session_state.get(ver_key, 0)) + 1


def _copy_bwm_state(src_s: int, src_e: int, dst_s: int, dst_e: int, criteria: List[str]):
    src = _ensure_bwm_input_state(src_s, src_e, criteria)
    dst = {
        "criteria_sig": _bwm_criteria_signature(criteria),
        "best": src["best"],
        "worst": src["worst"],
        "df": _bwm_prepare_df(src["df"], criteria),
    }
    st.session_state[_bwm_state_key(dst_s, dst_e)] = dst
    _sync_bwm_widget_state(dst_s, dst_e, criteria)


def _bwm_paste_template(criteria: List[str]) -> str:
    lines = ["Criterion	Best→Others (B2O)	Others→Worst (O2W)"]
    for crit in criteria:
        lines.append(f"{crit}	MI (Moderate importance)	MI (Moderate importance)")
    return "\n".join(lines)


def _parse_pasted_bwm_table(text: str, criteria: List[str]) -> pd.DataFrame:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("No pasted text provided.")

    sep = _detect_delimiter_from_text(raw)
    reader = csv.reader(io.StringIO(raw), delimiter=sep)
    rows = [[str(cell).strip() for cell in row] for row in reader]
    rows = [row for row in rows if any(cell != "" for cell in row)]
    if not rows:
        raise ValueError("No usable rows found in pasted text.")

    expected_rows = len(criteria)
    valid_terms = set(SCALE_OPTIONS_BWM)
    first = [c.strip().lower() for c in rows[0]]
    has_header = (
        "best→others (b2o)" in first
        or "best->others (b2o)" in first
        or "others→worst (o2w)" in first
        or "others->worst (o2w)" in first
    )
    data_rows = rows[1:] if has_header else rows
    if len(data_rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, but found {len(data_rows)}.")

    parsed_rows = []
    named_rows = {}
    has_name_col = any(len(row) >= 3 for row in data_rows)

    for idx, row in enumerate(data_rows):
        if len(row) < 2:
            raise ValueError(f"Row {idx + 1} must contain at least two linguistic values.")

        if len(row) >= 3:
            crit_name, b2o, o2w = row[0], row[1], row[2]
        else:
            crit_name, b2o, o2w = criteria[idx], row[0], row[1]

        if crit_name and crit_name not in criteria:
            raise ValueError(f"Unknown criterion '{crit_name}' in pasted data.")
        if b2o not in valid_terms:
            raise ValueError(f"'{b2o}' is not a valid Best→Others term.")
        if o2w not in valid_terms:
            raise ValueError(f"'{o2w}' is not a valid Others→Worst term.")

        if has_name_col:
            named_rows[crit_name] = (b2o, o2w)
        else:
            parsed_rows.append((criteria[idx], b2o, o2w))

    if has_name_col:
        missing = [crit for crit in criteria if crit not in named_rows]
        if missing:
            raise ValueError("Missing criteria in pasted data: " + ", ".join(missing))
        parsed_rows = [(crit, named_rows[crit][0], named_rows[crit][1]) for crit in criteria]

    out = pd.DataFrame(parsed_rows, columns=["Criterion", "Best→Others (B2O)", "Others→Worst (O2W)"])
    return out.set_index("Criterion")


def _bwm_render_expert_block(s: int, e: int, criteria: List[str], num_scenarios: int, num_experts: int) -> ExpertInputBWM:
    state = _ensure_bwm_input_state(s, e, criteria)
    st.markdown(f"### Scenario {s+1} • Expert {e+1}")

    best_key = _bwm_widget_key("bwm_best", s, e)
    worst_key = _bwm_widget_key("bwm_worst", s, e)
    editor_key = _bwm_widget_key("bwm_editor", s, e)

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    with ctrl1:
        best = st.selectbox(
            "Best criterion",
            criteria,
            index=criteria.index(state["best"]),
            key=best_key,
        )
    with ctrl2:
        worst = st.selectbox(
            "Worst criterion",
            criteria,
            index=criteria.index(state["worst"]),
            key=worst_key,
        )
    with ctrl3:
        st.caption("Tip: use the paste loader below or clone an existing block before fine-tuning the table.")

    state["best"] = best
    state["worst"] = worst
    state["df"] = _bwm_enforced_df(criteria, best, worst, state["df"])
    st.session_state[_bwm_state_key(s, e)] = state

    if best == worst:
        st.warning("Best and Worst are the same. Please choose different criteria.")

    with st.expander("Quick input tools", expanded=False):
        st.caption("Paste one table for this expert. Header is optional. Supported formats: TSV, CSV, or semicolon-separated.")
        st.code(_bwm_paste_template(criteria), language="text")
        paste_text = st.text_area(
            f"Paste Best→Others / Others→Worst table for Scenario {s+1} • Expert {e+1}",
            value="",
            height=180,
            key=f"bwm_paste_s{s}_e{e}",
        )

        action_cols = st.columns(4)
        with action_cols[0]:
            if st.button("Load pasted table", key=f"bwm_load_paste_s{s}_e{e}", use_container_width=True):
                try:
                    loaded_df = _parse_pasted_bwm_table(paste_text, criteria)
                    state["df"] = _bwm_enforced_df(criteria, state["best"], state["worst"], loaded_df)
                    st.session_state[_bwm_state_key(s, e)] = state
                    _sync_bwm_widget_state(s, e, criteria)
                    st.rerun()
                except Exception as ex:
                    st.error(f"Paste could not be loaded: {ex}")
        with action_cols[1]:
            if st.button("Reset this block", key=f"bwm_reset_s{s}_e{e}", use_container_width=True):
                st.session_state[_bwm_state_key(s, e)] = {
                    "criteria_sig": _bwm_criteria_signature(criteria),
                    "best": criteria[0],
                    "worst": criteria[-1] if len(criteria) > 1 else criteria[0],
                    "df": _bwm_default_df(criteria),
                }
                _sync_bwm_widget_state(s, e, criteria)
                st.rerun()
        with action_cols[2]:
            copy_label = "Copy previous expert" if e > 0 else "Copy previous scenario"
            if st.button(copy_label, key=f"bwm_copy_prev_s{s}_e{e}", use_container_width=True, disabled=(e == 0 and s == 0)):
                if e > 0:
                    _copy_bwm_state(s, e - 1, s, e, criteria)
                else:
                    _copy_bwm_state(s - 1, e, s, e, criteria)
                st.rerun()
        with action_cols[3]:
            if st.button("Fill all experts in scenario", key=f"bwm_fill_scenario_s{s}_e{e}", use_container_width=True, disabled=num_experts <= 1):
                for dst_e in range(num_experts):
                    _copy_bwm_state(s, e, s, dst_e, criteria)
                st.rerun()

        if num_scenarios > 1:
            if st.button(
                "Fill same expert slot across all scenarios",
                key=f"bwm_fill_column_s{s}_e{e}",
                use_container_width=True,
            ):
                for dst_s in range(num_scenarios):
                    _copy_bwm_state(s, e, dst_s, e, criteria)
                st.rerun()

    editor_df = _bwm_enforced_df(criteria, state["best"], state["worst"], state["df"])
    edited = st.data_editor(
        editor_df,
        use_container_width=True,
        key=editor_key,
        column_config={
            "Best→Others (B2O)": st.column_config.SelectboxColumn("Best→Others (B2O)", options=SCALE_OPTIONS_BWM),
            "Others→Worst (O2W)": st.column_config.SelectboxColumn("Others→Worst (O2W)", options=SCALE_OPTIONS_BWM),
        },
        disabled=criteria[0:0],
    )

    state["df"] = _bwm_enforced_df(criteria, state["best"], state["worst"], edited)
    st.session_state[_bwm_state_key(s, e)] = state

    best_idx = criteria.index(state["best"])
    worst_idx = criteria.index(state["worst"])
    best_to = [state["df"].loc[c, "Best→Others (B2O)"] for c in criteria]
    to_worst = [state["df"].loc[c, "Others→Worst (O2W)"] for c in criteria]

    best_to[best_idx] = "EI (Equal importance)"
    to_worst[worst_idx] = "EI (Equal importance)"

    return ExpertInputBWM(best_idx=best_idx, worst_idx=worst_idx, best_to=best_to, to_worst=to_worst)


def apply_scenario_probability_to_weight_matrix(W: np.ndarray, scenario_prob: float) -> np.ndarray:
    """
    Step 7 of the methodology:
    multiply each scenario-specific TrFS criterion weight by its scenario probability
    using positive scalar multiplication of a TrFS.
    """
    weighted = np.zeros_like(W, dtype=float)
    for i in range(W.shape[0]):
        t = trfs_scalar_mul(float(scenario_prob), TrFS(float(W[i, 0]), float(W[i, 1]), float(W[i, 2]), float(W[i, 3])))
        weighted[i, :] = [t.w1, t.w2, t.w3, t.w4]
    return weighted


def build_weighted_scenario_expert_dataset(
    weights_by_scenario_expert: List[List[np.ndarray]],
    scenario_ps: List[float],
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Returns
    -------
    weighted_dataset : ndarray, shape (q, n_criteria, 4)
        Each q is one scenario-expert dataset after multiplying by the scenario probability.
    q_map : list[(scenario_index, expert_index)]
        Mapping from q back to (scenario, expert).
    """
    stacked = []
    q_map: List[Tuple[int, int]] = []

    for s in range(len(weights_by_scenario_expert)):
        for e in range(len(weights_by_scenario_expert[s])):
            stacked.append(apply_scenario_probability_to_weight_matrix(weights_by_scenario_expert[s][e], scenario_ps[s]))
            q_map.append((s, e))

    return np.stack(stacked, axis=0), q_map


def solve_gams_style_trfs_aggregation(weighted_dataset: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str]]:
    """
    Exact Step 8 / GAMS-style aggregation model.

    weighted_dataset[q, j, :] contains the scenario-probability-weighted TrFS weight
    for dataset q (scenario-expert pair) and criterion j.

    The model is equivalent to the provided GAMS formulation:
      max obj = sum(q,j) sim(q,j)
      sim(q,j) = 1 - 0.25*( |x1(j)+x2(j)-w1(q,j)-w2(q,j)|
                          + |x3(j)+x4(j)-w3(q,j)-w4(q,j)| )
    with the same normalization / ordering constraints.

    We solve an equivalent LP by linearizing the absolute values.
    Because the original model can have multiple optimal solutions, a different solver
    may return a different optimal x1..x4 than GAMS while still matching the same
    optimal objective value.
    """
    if not SCIPY_OK:
        return None, None, "SciPy is required to solve the final GAMS-style TrFS aggregation model."

    if weighted_dataset.ndim != 3 or weighted_dataset.shape[2] != 4:
        return None, None, "weighted_dataset must have shape (q, n_criteria, 4)."

    q_count, n_crit, _ = weighted_dataset.shape
    w1 = weighted_dataset[:, :, 0]
    w2 = weighted_dataset[:, :, 1]
    w3 = weighted_dataset[:, :, 2]
    w4 = weighted_dataset[:, :, 3]

    def idx_x(block: int, j: int) -> int:
        return block * n_crit + j

    sx1 = 4 * n_crit
    sx2 = 4 * n_crit + 1
    sx3 = 4 * n_crit + 2
    sx4 = 4 * n_crit + 3

    def idx_d12(q: int, j: int) -> int:
        return 4 * n_crit + 4 + q * n_crit + j

    def idx_d34(q: int, j: int) -> int:
        return 4 * n_crit + 4 + q_count * n_crit + q * n_crit + j

    num_vars = 4 * n_crit + 4 + 2 * q_count * n_crit
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []

    # Sum variables: sx1 = sum_j x1(j), ..., sx4 = sum_j x4(j)
    row = np.zeros(num_vars)
    row[sx1] = -1.0
    for j in range(n_crit):
        row[idx_x(0, j)] = 1.0
    A_eq.append(row)
    b_eq.append(0.0)

    row = np.zeros(num_vars)
    row[sx2] = -1.0
    for j in range(n_crit):
        row[idx_x(1, j)] = 1.0
    A_eq.append(row)
    b_eq.append(0.0)

    row = np.zeros(num_vars)
    row[sx3] = -1.0
    for j in range(n_crit):
        row[idx_x(2, j)] = 1.0
    A_eq.append(row)
    b_eq.append(0.0)

    row = np.zeros(num_vars)
    row[sx4] = -1.0
    for j in range(n_crit):
        row[idx_x(3, j)] = 1.0
    A_eq.append(row)
    b_eq.append(0.0)

    # GAMS normalization and monotonicity constraints.
    for j in range(n_crit):
        row = np.zeros(num_vars)
        row[idx_x(3, j)] = 1.0
        row[idx_x(0, j)] = -1.0
        row[sx1] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)

        row = np.zeros(num_vars)
        row[idx_x(0, j)] = 1.0
        row[idx_x(3, j)] = -1.0
        row[sx4] = 1.0
        A_ub.append(-row)
        b_ub.append(-1.0)

        row = np.zeros(num_vars)
        row[idx_x(2, j)] = 1.0
        row[idx_x(1, j)] = -1.0
        row[sx2] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)

        row = np.zeros(num_vars)
        row[idx_x(1, j)] = 1.0
        row[idx_x(2, j)] = -1.0
        row[sx3] = 1.0
        A_ub.append(-row)
        b_ub.append(-1.0)

        row = np.zeros(num_vars)
        row[idx_x(0, j)] = 1.0
        row[idx_x(1, j)] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

        row = np.zeros(num_vars)
        row[idx_x(1, j)] = 1.0
        row[idx_x(2, j)] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

        row = np.zeros(num_vars)
        row[idx_x(2, j)] = 1.0
        row[idx_x(3, j)] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # Absolute value linearization.
    for q in range(q_count):
        for j in range(n_crit):
            c12 = float(w1[q, j] + w2[q, j])
            c34 = float(w3[q, j] + w4[q, j])

            row = np.zeros(num_vars)
            row[idx_x(0, j)] = 1.0
            row[idx_x(1, j)] = 1.0
            row[idx_d12(q, j)] = -1.0
            A_ub.append(row)
            b_ub.append(c12)

            row = np.zeros(num_vars)
            row[idx_x(0, j)] = -1.0
            row[idx_x(1, j)] = -1.0
            row[idx_d12(q, j)] = -1.0
            A_ub.append(row)
            b_ub.append(-c12)

            row = np.zeros(num_vars)
            row[idx_x(2, j)] = 1.0
            row[idx_x(3, j)] = 1.0
            row[idx_d34(q, j)] = -1.0
            A_ub.append(row)
            b_ub.append(c34)

            row = np.zeros(num_vars)
            row[idx_x(2, j)] = -1.0
            row[idx_x(3, j)] = -1.0
            row[idx_d34(q, j)] = -1.0
            A_ub.append(row)
            b_ub.append(-c34)

    c = np.zeros(num_vars)
    for q in range(q_count):
        for j in range(n_crit):
            c[idx_d12(q, j)] = 1.0
            c[idx_d34(q, j)] = 1.0

    bounds = [(0.0, None)] * num_vars

    res = linprog(
        c=c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return None, None, res.message

    x = res.x
    agg = np.zeros((n_crit, 4), dtype=float)
    for j in range(n_crit):
        agg[j, 0] = x[idx_x(0, j)]
        agg[j, 1] = x[idx_x(1, j)]
        agg[j, 2] = x[idx_x(2, j)]
        agg[j, 3] = x[idx_x(3, j)]

    obj = float(q_count * n_crit - 0.25 * res.fun)
    return agg, obj, None


def aggregate_bwm_weights_across_scenarios_experts(
    weights_by_scenario_expert: List[List[np.ndarray]],
    scenario_ps: List[float],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[Tuple[int, int]]], Optional[float], Optional[str]]:
    """
    Exact workflow required by the methodology and GAMS code:
    1) solve TrFS-BWM for every scenario and expert,
    2) multiply each scenario-specific TrFS weight by the scenario probability,
    3) aggregate all scenario-expert datasets using the final similarity-maximization model.
    """
    weighted_dataset, q_map = build_weighted_scenario_expert_dataset(weights_by_scenario_expert, scenario_ps)
    agg, obj, err = solve_gams_style_trfs_aggregation(weighted_dataset)
    return agg, weighted_dataset, q_map, obj, err


# ============================================================
# Saved TrFS-BWM Sets + Global Weight Helpers
# ============================================================

def _ensure_bwm_saved_sets():
    if "bwm_saved_sets" not in st.session_state:
        st.session_state["bwm_saved_sets"] = {}


def _standardize_weight_df(df: pd.DataFrame, name_col: str = "Criterion") -> pd.DataFrame:
    required = [name_col, "w1", "w2", "w3", "w4"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[[name_col, "w1", "w2", "w3", "w4"]].copy()
    out = out.rename(columns={name_col: "Criterion"})
    out["Criterion"] = out["Criterion"].astype(str).str.strip()
    for c in ["w1", "w2", "w3", "w4"]:
        out[c] = out[c].map(lambda v: float(_to_float(v, 0.0)))
    return out


def _save_bwm_weight_set(set_name: str, role: str, df: pd.DataFrame):
    _ensure_bwm_saved_sets()
    st.session_state["bwm_saved_sets"][set_name] = {
        "role": role,
        "df": _standardize_weight_df(df),
    }


def _get_saved_set_names(role: Optional[str] = None, exclude: Optional[List[str]] = None) -> List[str]:
    _ensure_bwm_saved_sets()
    exclude = set(exclude or [])
    names = []
    for name, payload in st.session_state["bwm_saved_sets"].items():
        if name in exclude:
            continue
        if role is None or payload.get("role") == role:
            names.append(name)
    return sorted(names)


def compute_global_weights_from_sets(
    main_df: pd.DataFrame,
    subgroup_map: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    main_std = _standardize_weight_df(main_df)

    for _, main_row in main_std.iterrows():
        main_name = str(main_row["Criterion"])
        if main_name not in subgroup_map:
            continue

        main_trfs = TrFS(
            float(main_row["w1"]),
            float(main_row["w2"]),
            float(main_row["w3"]),
            float(main_row["w4"]),
        ).clip()

        sub_std = _standardize_weight_df(subgroup_map[main_name])
        for _, sub_row in sub_std.iterrows():
            local_trfs = TrFS(
                float(sub_row["w1"]),
                float(sub_row["w2"]),
                float(sub_row["w3"]),
                float(sub_row["w4"]),
            ).clip()
            global_trfs = trfs_mul(main_trfs, local_trfs)
            rows.append({
                "Main Criterion": main_name,
                "Sub Criterion": str(sub_row["Criterion"]),
                "Main TrFS": trfs_to_str(main_trfs, nd=5),
                "Local Sub TrFS": trfs_to_str(local_trfs, nd=5),
                "Global TrFS": trfs_to_str(global_trfs, nd=5),
                "main_w1": main_trfs.w1,
                "main_w2": main_trfs.w2,
                "main_w3": main_trfs.w3,
                "main_w4": main_trfs.w4,
                "local_w1": local_trfs.w1,
                "local_w2": local_trfs.w2,
                "local_w3": local_trfs.w3,
                "local_w4": local_trfs.w4,
                "gw1": global_trfs.w1,
                "gw2": global_trfs.w2,
                "gw3": global_trfs.w3,
                "gw4": global_trfs.w4,
                "GMIR (Main)": gmIR(main_trfs),
                "GMIR (Local Sub)": gmIR(local_trfs),
                "GMIR (Global)": gmIR(global_trfs),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "Main Criterion", "Sub Criterion", "Main TrFS", "Local Sub TrFS", "Global TrFS",
            "main_w1", "main_w2", "main_w3", "main_w4",
            "local_w1", "local_w2", "local_w3", "local_w4",
            "gw1", "gw2", "gw3", "gw4",
            "GMIR (Main)", "GMIR (Local Sub)", "GMIR (Global)"
        ])

    return pd.DataFrame(rows)


def _default_weight_df(names: List[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "Criterion": [str(x).strip() for x in names],
        "w1": [0.0] * len(names),
        "w2": [0.0] * len(names),
        "w3": [0.0] * len(names),
        "w4": [0.0] * len(names),
    })


def _sanitize_name_list(raw: str, n_default: int, prefix: str) -> List[str]:
    parts = [p.strip() for p in str(raw).replace("\n", ",").split(",") if p.strip()]
    if parts:
        return parts
    return [f"{prefix}{i+1}" for i in range(n_default)]


def _parse_pasted_weight_table(text: str, fallback_prefix: str = "C") -> pd.DataFrame:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("No pasted text provided.")

    first_line = raw.splitlines()[0]
    if "	" in first_line:
        sep = "	"
    elif first_line.count(",") >= 3:
        sep = ","
    elif first_line.count(";") >= 3:
        sep = ";"
    else:
        raise ValueError("Could not detect delimiter. Please paste tab-, comma-, or semicolon-separated data.")

    lower_cols = {"criterion", "criteria", "challenge", "name", "main criterion", "sub criterion"}

    def _load(header):
        return pd.read_csv(io.StringIO(raw), sep=sep, engine="python", header=header)

    try:
        df_try = _load(0)
        cols_lower = [str(c).strip().lower() for c in df_try.columns]
        if any(c in lower_cols for c in cols_lower) and all(c in cols_lower for c in ["w1", "w2", "w3", "w4"]):
            colmap = {str(c).strip().lower(): c for c in df_try.columns}
            crit_col = next((colmap[c] for c in cols_lower if c in lower_cols), None)
            out = df_try[[crit_col, colmap["w1"], colmap["w2"], colmap["w3"], colmap["w4"]]].copy()
            out.columns = ["Criterion", "w1", "w2", "w3", "w4"]
            return _standardize_weight_df(out)
    except Exception:
        pass

    df = _load(None)
    df = df.dropna(how="all")
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    if df.shape[1] < 4:
        raise ValueError("Need at least 4 numeric weight columns (w1, w2, w3, w4).")

    if df.shape[1] >= 5:
        out = df.iloc[:, :5].copy()
        out.columns = ["Criterion", "w1", "w2", "w3", "w4"]
    else:
        out = df.iloc[:, :4].copy()
        out.columns = ["w1", "w2", "w3", "w4"]
        out.insert(0, "Criterion", [f"{fallback_prefix}{i+1}" for i in range(len(out))])
    return _standardize_weight_df(out)



def _detect_delimiter_from_text(raw: str) -> str:
    first_line = str(raw or '').strip().splitlines()[0] if str(raw or '').strip() else ''
    if "	" in first_line:
        return "	"
    if first_line.count(",") >= 1:
        return ","
    if first_line.count(";") >= 1:
        return ";"
    raise ValueError("Could not detect delimiter. Please paste tab-, comma-, or semicolon-separated data.")


def _parse_pasted_rel_matrix(
    text: str,
    row_names: List[str],
    col_names: List[str],
    valid_terms: List[str],
) -> pd.DataFrame:
    raw = str(text or '').strip()
    if not raw:
        raise ValueError('No pasted text provided.')
    sep = _detect_delimiter_from_text(raw)
    reader = csv.reader(io.StringIO(raw), delimiter=sep)
    rows = [[str(cell).strip() for cell in row] for row in reader]
    rows = [row for row in rows if any(cell != '' for cell in row)]
    if not rows:
        raise ValueError('No usable rows found in pasted text.')

    expected_cols = len(col_names)
    row_name_set = {str(x).strip().lower() for x in row_names}
    col_name_set = {str(x).strip().lower() for x in col_names}

    first = rows[0]
    header_overlap = sum(1 for c in first if c.strip().lower() in col_name_set)
    header_present = header_overlap >= max(1, expected_cols // 2)
    data_rows = rows[1:] if header_present else rows
    if not data_rows:
        raise ValueError('No data rows were found after removing the header row.')

    first_data = data_rows[0]
    row_label_present = False
    if len(first_data) == expected_cols + 1:
        row_label_present = True
    elif len(first_data) > 0 and first_data[0].strip().lower() in row_name_set:
        row_label_present = True

    parsed = []
    for row in data_rows:
        cells = row[1:] if row_label_present else row
        cells = [str(x).strip() for x in cells if str(x).strip() != '']
        if len(cells) < expected_cols:
            raise ValueError(f'Each pasted row must have {expected_cols} relationship values. Found {len(cells)} in row: {row}')
        parsed.append(cells[:expected_cols])

    if len(parsed) != len(row_names):
        raise ValueError(f'Expected {len(row_names)} rows for the current challenge count, but found {len(parsed)} rows.')

    invalid = sorted({cell for row in parsed for cell in row if cell not in valid_terms})
    if invalid:
        raise ValueError('These linguistic terms are not in the current scale: ' + ', '.join(map(str, invalid)))

    out = pd.DataFrame(parsed, columns=col_names)
    out.insert(0, 'Challenge', row_names)
    return out


def _weights_template_text(name_col: str, names: List[str], sep: str = '	') -> str:
    header = sep.join([name_col, 'w1', 'w2', 'w3', 'w4'])
    lines = [header]
    for n in names:
        lines.append(sep.join([str(n), '0.100000', '0.200000', '0.300000', '0.400000']))
    return '\n'.join(lines)


def _rel_template_text(row_names: List[str], col_names: List[str], default_term: str = 'M', sep: str = '	') -> str:
    header = sep.join(['Challenge'] + [str(c) for c in col_names])
    lines = [header]
    for r in row_names:
        lines.append(sep.join([str(r)] + [default_term] * len(col_names)))
    return '\n'.join(lines)

def _stage_qfd_weight_import(df: pd.DataFrame):
    std = _standardize_weight_df(df)
    staged = std.rename(columns={"Criterion": "Challenge"}).copy()
    names = staged["Challenge"].astype(str).tolist()
    st.session_state["qfd_pending_weights_import"] = staged
    st.session_state["qfd_nch"] = len(names)
    for i, name in enumerate(names):
        st.session_state[f"ch_name_{i}"] = str(name)


def _render_manual_weight_table(title: str, key_root: str, names: List[str], help_text: str) -> pd.DataFrame:
    st.markdown(f"**{title}**")
    st.caption(help_text)
    seed_key = f"{key_root}_seed"
    table_key = f"{key_root}_table"
    paste_key = f"{key_root}_paste"

    seed_sig = "|".join(names)
    if st.session_state.get(seed_key) != seed_sig:
        st.session_state[seed_key] = seed_sig
        st.session_state[table_key] = _default_weight_df(names)

    paste_text = st.text_area(
        f"Paste table for {title}",
        value="",
        height=140,
        key=paste_key,
        help="Paste columns as: Criterion, w1, w2, w3, w4 (header optional). Excel tab-separated paste is supported."
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button(f"Load pasted table into {title}", key=f"{key_root}_load_paste", use_container_width=True):
            try:
                pasted_df = _parse_pasted_weight_table(paste_text, fallback_prefix=key_root[:1].upper())
                if len(pasted_df) != len(names):
                    st.warning(f"Pasted table has {len(pasted_df)} rows but this block currently expects {len(names)} rows. Loaded the pasted row count.")
                st.session_state[table_key] = pasted_df
                st.session_state[seed_key] = "|".join(pasted_df["Criterion"].astype(str).tolist())
            except Exception as e:
                st.error(f"Paste could not be loaded: {e}")
    with c2:
        if st.button(f"Reset {title}", key=f"{key_root}_reset", use_container_width=True):
            st.session_state[table_key] = _default_weight_df(names)
            st.session_state[seed_key] = seed_sig

    edited = st.data_editor(
        st.session_state[table_key],
        use_container_width=True,
        num_rows="dynamic",
        key=f"{key_root}_editor",
        column_config={
            "Criterion": st.column_config.TextColumn("Criterion", width="medium"),
            "w1": st.column_config.NumberColumn("w1", format="%.6f"),
            "w2": st.column_config.NumberColumn("w2", format="%.6f"),
            "w3": st.column_config.NumberColumn("w3", format="%.6f"),
            "w4": st.column_config.NumberColumn("w4", format="%.6f"),
        }
    )
    st.session_state[table_key] = edited.copy()

    try:
        std = _standardize_weight_df(edited)
        invalid = std[(std[["w1", "w2", "w3", "w4"]].diff(axis=1).iloc[:, 1:] < -1e-12).any(axis=1)]
        if not invalid.empty:
            st.warning("Some rows do not satisfy w1 ≤ w2 ≤ w3 ≤ w4. The model will clip/order values when computing TrFS products.")
        return std
    except Exception as e:
        st.error(f"Table format issue: {e}")
        return pd.DataFrame(columns=["Criterion", "w1", "w2", "w3", "w4"])



# ============================================================
# 0) Scenario-aware TrFS-BWM Page
# ============================================================

def page_trfs_bwm():
    st.markdown('<h1 class="main-header">🧠 Scenario-aware TrFS-SBWM Model</h1>', unsafe_allow_html=True)
    st.markdown("**Trapezoidal Fuzzy Stratified Best–Worst Method with Scenarios**")

    if not SCIPY_OK:
        st.error("SciPy is required for TrFS-BWM (linprog). Please install `scipy` and rerun.")
        st.stop()

    st.markdown('<div class="section-header">📝 Criteria & Scenario/Expert Setup</div>', unsafe_allow_html=True)

    criteria = _bwm_criteria_input()
    if len(criteria) < 2:
        st.error("Please provide at least 2 criteria.")
        st.stop()

    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ TrFS-BWM Settings</div>', unsafe_allow_html=True)

        num_scenarios = st.slider("Number of scenarios", min_value=1, max_value=10, value=2, step=1)
        num_experts = st.slider("Number of experts", min_value=1, max_value=10, value=3, step=1)

        st.markdown("---")
        st.markdown("**Scenario transition probabilities (must sum to 1)**")
        s_probs = []
        default_sp = round(1.0 / num_scenarios, 3)
        for s in range(num_scenarios):
            s_probs.append(
                float(
                    st.number_input(
                        f"Scenario {s+1} probability",
                        min_value=0.0,
                        value=default_sp,
                        step=0.01,
                        key=f"bwm_sp_{s}",
                    )
                )
            )
        sp_ok = exact_sum_ok(s_probs)
        if sp_ok:
            st.success(f"Scenario probability sum = {sum(s_probs):.6f} (OK)")
        else:
            st.error(f"Scenario probability sum = {sum(s_probs):.6f}. Adjust to exactly 1.000.")

        st.markdown("---")
        st.markdown("**Scale preview (Table S1)**")
        scale_df = pd.DataFrame(
            {"TrFS": [bwm_trf_str(LINGUISTIC_SCALE_BWM[k], nd=2) for k in SCALE_OPTIONS_BWM]},
            index=SCALE_OPTIONS_BWM
        )
        st.dataframe(scale_df, use_container_width=True, height=280)

    can_run = sp_ok

    st.markdown("---")
    scenario_inputs: List[List[ExpertInputBWM]] = []
    scenario_tabs = st.tabs([f"Scenario {s+1}" for s in range(num_scenarios)])

    for s, scen_tab in enumerate(scenario_tabs):
        with scen_tab:
            st.caption(f"Transition probability = {s_probs[s]:.6f}")
            expert_inputs_this_s = []

            expert_tabs = st.tabs([f"Expert {e+1}" for e in range(num_experts)])
            for e, exp_tab in enumerate(expert_tabs):
                with exp_tab:
                    expert_inputs_this_s.append(_bwm_render_expert_block(s, e, criteria, num_scenarios, num_experts))

            scenario_inputs.append(expert_inputs_this_s)

    st.markdown("---")
    compute_btn = st.button(
        "✅ Compute Scenario-aware TrFS-BWM Weights",
        type="primary",
        use_container_width=True,
        disabled=not can_run,
    )

    if not can_run:
        st.warning("Model is locked: adjust scenario probabilities so their SUM = 1.000 to enable computation.")
        return

    if not compute_btn:
        return

    weights_by_scenario_expert: List[List[np.ndarray]] = []
    phi_by_scenario_expert: List[List[float]] = []
    cr_by_scenario_expert: List[List[Optional[float]]] = []
    errors = []

    for s in range(num_scenarios):
        w_list = []
        phi_list = []
        cr_list = []

        for e in range(num_experts):
            ex = scenario_inputs[s][e]

            if ex.best_idx == ex.worst_idx:
                errors.append(f"Scenario {s+1}, Expert {e+1}: Best and Worst must be different.")
                continue

            mB = [LINGUISTIC_SCALE_BWM[lbl] for lbl in ex.best_to]
            mW = [LINGUISTIC_SCALE_BWM[lbl] for lbl in ex.to_worst]

            W, phi, err = solve_trfs_bwm_weights(mB, mW, ex.best_idx, ex.worst_idx)
            if err:
                errors.append(f"Scenario {s+1}, Expert {e+1}: {err}")
                continue

            used = [LINGUISTIC_SCALE_BWM[lbl] for lbl in (ex.best_to + ex.to_worst) if "EI (Equal importance)" not in lbl]
            cr = compute_cr(phi, used)

            w_list.append(W)
            phi_list.append(phi)
            cr_list.append(cr)

        weights_by_scenario_expert.append(w_list)
        phi_by_scenario_expert.append(phi_list)
        cr_by_scenario_expert.append(cr_list)

    if errors:
        st.error("Cannot compute weights:\n- " + "\n- ".join(errors))
        st.stop()

    st.success("✅ Scenario-aware TrFS-BWM solved successfully!")

    st.markdown('<div class="section-header">👥 Scenario × Expert Results</div>', unsafe_allow_html=True)
    out_tabs = st.tabs([f"Scenario {s+1}" for s in range(num_scenarios)])

    for s, tab in enumerate(out_tabs):
        with tab:
            st.caption(f"Transition probability = {s_probs[s]:.6f}")
            subtabs = st.tabs([f"Expert {e+1}" for e in range(num_experts)])

            for e, subtab in enumerate(subtabs):
                with subtab:
                    W = weights_by_scenario_expert[s][e]
                    phi = phi_by_scenario_expert[s][e]
                    cr = cr_by_scenario_expert[s][e]

                    df = pd.DataFrame(W, index=criteria, columns=["w1", "w2", "w3", "w4"])
                    df["TrFS weight"] = df.apply(lambda r: bwm_trf_str((r.w1, r.w2, r.w3, r.w4), nd=5), axis=1)
                    df["GMIR (crisp)"] = df.apply(lambda r: bwm_gmir((r.w1, r.w2, r.w3, r.w4)), axis=1)
                    df = df[["TrFS weight", "GMIR (crisp)"]]
                    st.dataframe(df, use_container_width=True)

                    p = s_probs[s]
                    weighted_df = pd.DataFrame(
                        [[p * float(v) for v in row] for row in W],
                        index=criteria,
                        columns=["w1", "w2", "w3", "w4"],
                    )
                    weighted_df["TrFS weighted by scenario probability"] = weighted_df.apply(
                        lambda r: bwm_trf_str((r.w1, r.w2, r.w3, r.w4), nd=5), axis=1
                    )
                    weighted_df["GMIR (weighted)"] = weighted_df.apply(
                        lambda r: bwm_gmir((r.w1, r.w2, r.w3, r.w4)), axis=1
                    )
                    with st.expander("Show scenario-probability multiplied weights"):
                        st.dataframe(
                            weighted_df[["TrFS weighted by scenario probability", "GMIR (weighted)"]],
                            use_container_width=True
                        )

                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Optimal φ* (Eq. S11)", f"{phi:.6f}")
                    with c2:
                        if cr is None:
                            st.info("CR not available.")
                        else:
                            st.metric("Consistency Ratio, CR (Eq. S12)", f"{cr:.6f}")

    st.markdown('<div class="section-header">🧩 Final Aggregated Group Weights</div>', unsafe_allow_html=True)
    st.caption(
        "Exact methodology / GAMS workflow:\n"
        "1) solve TrFS-BWM for each expert in each scenario;\n"
        "2) multiply each scenario-specific TrFS weight by its scenario probability;\n"
        "3) aggregate all scenario-expert datasets with the final similarity-maximization model (no expert weights)."
    )

    aggW, weighted_dataset, q_map, agg_obj, agg_err = aggregate_bwm_weights_across_scenarios_experts(weights_by_scenario_expert, s_probs)
    if agg_err:
        st.error(f"Final GAMS-style aggregation failed: {agg_err}")
        st.stop()

    agg_df = pd.DataFrame(aggW, index=criteria, columns=["w1", "w2", "w3", "w4"])
    agg_df["TrFS weight"] = agg_df.apply(lambda r: bwm_trf_str((r.w1, r.w2, r.w3, r.w4), nd=5), axis=1)
    agg_df["GMIR (crisp)"] = agg_df.apply(lambda r: bwm_gmir((r.w1, r.w2, r.w3, r.w4)), axis=1)
    agg_out = agg_df[["TrFS weight", "GMIR (crisp)"]].copy()

    st.dataframe(agg_out, use_container_width=True)
    c1m, c2m = st.columns(2)
    with c1m:
        st.metric("Aggregation objective", f"{agg_obj:.6f}")
    with c2m:
        st.caption("Note: the final GAMS-style model can have multiple optimal solutions with the same objective value.")
    st.subheader("Crisp weights (GMIR)")
    chart_df = agg_out[["GMIR (crisp)"]].reset_index().rename(columns={"index": "Criterion"})
    st.bar_chart(chart_df, x="Criterion", y="GMIR (crisp)")

    with st.expander("Show the scenario-probability weighted datasets used in the final aggregation model"):
        rows = []
        for q_idx, (s_idx, e_idx) in enumerate(q_map):
            dfq = pd.DataFrame(weighted_dataset[q_idx], index=criteria, columns=["w1", "w2", "w3", "w4"])
            dfq.insert(0, "Criterion", dfq.index)
            dfq.insert(0, "Expert", e_idx + 1)
            dfq.insert(0, "Scenario", s_idx + 1)
            dfq.insert(0, "q", q_idx + 1)
            rows.append(dfq.reset_index(drop=True))
        st.dataframe(pd.concat(rows, ignore_index=True), use_container_width=True)

    st.session_state["bwm_criteria"] = criteria
    st.session_state["bwm_aggW"] = aggW
    st.session_state["bwm_agg_df"] = pd.DataFrame(
        aggW, index=criteria, columns=["w1", "w2", "w3", "w4"]
    ).reset_index().rename(columns={"index": "Criterion"})

    st.markdown('<div class="section-header">💾 Save This Aggregated Weight Set</div>', unsafe_allow_html=True)
    _ensure_bwm_saved_sets()
    save_cols = st.columns([2, 1, 1])
    with save_cols[0]:
        set_name = st.text_input(
            "Set name",
            value=st.session_state.get("bwm_last_set_name", ""),
            key="bwm_save_set_name",
            placeholder="e.g., Main Criteria / Cost Subcriteria / Quality Subcriteria"
        )
    with save_cols[1]:
        set_role = st.selectbox(
            "Set role",
            ["Main criteria", "Sub-criteria group"],
            key="bwm_save_set_role"
        )
    with save_cols[2]:
        st.write("")
        if st.button("Save aggregated set", use_container_width=True):
            if not str(set_name).strip():
                st.error("Please enter a set name before saving.")
            else:
                _save_bwm_weight_set(str(set_name).strip(), set_role, st.session_state["bwm_agg_df"])
                st.session_state["bwm_last_set_name"] = str(set_name).strip()
                st.success(f"Saved '{str(set_name).strip()}' for the global-weight module ✅")

    saved_sets_preview = []
    for nm, payload in st.session_state.get("bwm_saved_sets", {}).items():
        saved_sets_preview.append({
            "Set Name": nm,
            "Role": payload.get("role", ""),
            "Items": len(payload.get("df", pd.DataFrame())),
        })
    if saved_sets_preview:
        with st.expander("Show saved aggregated TrFS-BWM sets"):
            st.dataframe(pd.DataFrame(saved_sets_preview), use_container_width=True, hide_index=True)

    with st.expander("➡️ Send aggregated TrFS weights to TrFS-QFD challenge weights", expanded=True):
        st.markdown(
            "This will make the **TrFS-QFD page** able to load these weights into the *Challenge Weights* table "
            "(matching by **names** if possible, otherwise by **order** if lengths match)."
        )
        st.success("Saved in session ✅. Now open **4) TrFS-QFD (experts only)** and click **Load from TrFS-BWM**.")

    c1, c2 = st.columns(2)
    with c1:
        csv_bytes = agg_out.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download aggregated weights (CSV)",
            data=csv_bytes,
            file_name="scenario_aware_trfs_bwm_aggregated_weights.csv",
            mime="text/csv",
            use_container_width=True
        )

    with c2:
        all_rows = []
        for s in range(num_scenarios):
            for e in range(num_experts):
                W = weights_by_scenario_expert[s][e]
                tmp = pd.DataFrame(W, columns=["w1", "w2", "w3", "w4"])
                tmp.insert(0, "Criterion", criteria)
                tmp.insert(0, "Expert", e + 1)
                tmp.insert(0, "Scenario", s + 1)
                tmp.insert(0, "ScenarioProb", s_probs[s])
                weighted_tmp = pd.DataFrame(apply_scenario_probability_to_weight_matrix(W, s_probs[s]), columns=["pw1", "pw2", "pw3", "pw4"])
                tmp = pd.concat([tmp, weighted_tmp], axis=1)
                tmp["GMIR"] = tmp.apply(lambda r: bwm_gmir((r.w1, r.w2, r.w3, r.w4)), axis=1)
                tmp["WeightedGMIR"] = tmp.apply(lambda r: bwm_gmir((r.pw1, r.pw2, r.pw3, r.pw4)), axis=1)
                all_rows.append(tmp)
        expert_csv = pd.concat(all_rows, ignore_index=True).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download scenario-expert weights (CSV)",
            data=expert_csv,
            file_name="scenario_expert_trfs_bwm_and_probability_weighted_weights.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# 1) TrFS-QFD Page (Experts only, NO scenario)
# ============================================================

def page_trfs_qfd():

    st.markdown('<h1 class="main-header">📊 TrFS-QFD Analysis</h1>', unsafe_allow_html=True)
    st.markdown("**Challenge → Strategy Relationship Mapping Using TrFS Lingustic Terms**")

    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)

        st.markdown("**Model Dimensions**")
        col1, col2 = st.columns(2)
        with col1:
            n_ch = int(st.number_input("Challenges", min_value=2, max_value=80, value=10, step=1, key="qfd_nch"))
        with col2:
            n_sy = int(st.number_input("Strategies", min_value=2, max_value=40, value=10, step=1, key="qfd_nsy"))

        n_exp = int(st.number_input("Experts", min_value=1, max_value=10, value=3, step=1, key="qfd_nexp"))
        st.caption("Expert judgments are aggregated in Module 4 by simple TrFS arithmetic averaging, so no expert-weight inputs are required here.")

    st.markdown('<div class="section-header">📝 Input Definitions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Challenge Names**")
        ch_names = [st.text_input(f"Challenge {i+1}", value=f"Ch{i+1}", key=f"ch_name_{i}",
                                  label_visibility="collapsed") for i in range(n_ch)]
    with col2:
        st.markdown("**Strategy Names**")
        sy_names = [st.text_input(f"Strategy {j+1}", value=f"Sy{j+1}", key=f"sy_name_{j}",
                                  label_visibility="collapsed") for j in range(n_sy)]

    wkey = f"wdf|{n_ch}|" + "|".join(ch_names)
    if st.session_state.get("wkey") != wkey:
        st.session_state["wkey"] = wkey
        st.session_state["weights_df"] = init_weights_df(ch_names)

    with st.expander("🎯 TrFS Linguistic Scale", expanded=True):
        if "scale_df" not in st.session_state:
            st.session_state["scale_df"] = default_trfs_scale()

        scale_df = st.data_editor(
            st.session_state["scale_df"],
            use_container_width=True,
            num_rows="dynamic",
            key="scale_editor",
            column_config={
                "Term": st.column_config.TextColumn("Term", width="small"),
                "w1": st.column_config.NumberColumn("w1", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
                "w2": st.column_config.NumberColumn("w2", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
                "w3": st.column_config.NumberColumn("w3", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
                "w4": st.column_config.NumberColumn("w4", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
            }
        )
        st.session_state["scale_df"] = scale_df
        scale = scale_df_to_dict(scale_df)

    term_list = sorted(list(scale.keys()))
    if not term_list:
        st.error("Your scale table is empty. Add at least one term.")
        return

    st.markdown('<div class="section-header">⚖️ Challenge Weights (TrFS)</div>', unsafe_allow_html=True)
    load_col, note_col = st.columns([1.2, 2.8])
    with load_col:
        if st.button("📥 Load weights from Module 3", key="qfd_load_module3_weights", use_container_width=True):
            src_df = st.session_state.get("global_weights_df")
            if src_df is None or len(src_df) == 0:
                st.warning("No saved global weights found in Module 3.")
            else:
                try:
                    load_df = standardize_qfd_weight_df(src_df)
                    loaded_names = load_df["Challenge"].astype(str).tolist()
                    st.session_state["qfd_nch"] = len(loaded_names)
                    for i, name in enumerate(loaded_names):
                        st.session_state[f"ch_name_{i}"] = name
                    st.session_state["weights_df"] = load_df.copy()
                    st.session_state["wkey"] = f"wdf|{len(loaded_names)}|" + "|".join(loaded_names)
                    st.rerun()
                except Exception as ex:
                    st.error(f"Module 3 weights could not be loaded: {ex}")
    with note_col:
        st.caption("You can either load the global sub-criteria weights saved in Module 3 or paste/type directly into the table below. For direct paste, click the first cell under w1 and paste a 4-column Excel block (w1, w2, w3, w4).")

    editor_df = weights_df_for_editor(st.session_state["weights_df"])
    weights_df = st.data_editor(
        editor_df,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        disabled=["Challenge"],
        height=min(700, 60 + max(1, len(editor_df)) * 35),
        key="weights_editor",
        column_config={
            "Challenge": st.column_config.TextColumn("Challenge", width="medium"),
            "w1": st.column_config.TextColumn("w1", width="small", help="Paste decimal values directly, e.g. 0.00310"),
            "w2": st.column_config.TextColumn("w2", width="small", help="Paste decimal values directly, e.g. 0.00310"),
            "w3": st.column_config.TextColumn("w3", width="small", help="Paste decimal values directly, e.g. 0.07660"),
            "w4": st.column_config.TextColumn("w4", width="small", help="Paste decimal values directly, e.g. 0.07660"),
        }
    )
    st.session_state["weights_df"] = weights_df.copy()

    invalid_rows = []
    for idx, row in weights_df.iterrows():
        vals = [_to_float(row[c], default=np.nan) for c in ["w1", "w2", "w3", "w4"]]
        if any(np.isnan(v) for v in vals):
            invalid_rows.append(idx + 1)
    if invalid_rows:
        st.warning(f"Rows with non-numeric weight entries: {invalid_rows}. Fix them before computing the TrFS-QFD analysis.")

    ch_weights = [
        TrFS(_to_float(r["w1"], 0), _to_float(r["w2"], 0), _to_float(r["w3"], 0), _to_float(r["w4"], 0)).clip()
        for _, r in weights_df.iterrows()
    ]

    st.markdown('<div class="section-header">🔗 Relationship Matrices</div>', unsafe_allow_html=True)
    st.caption("Step 11 aggregation uses the simple arithmetic mean of the experts' TrFS relationship assessments for each challenge-strategy pair.")
    rel_by_expert_terms: List[pd.DataFrame] = []
    expert_tabs = st.tabs([f"Expert {e+1}" for e in range(n_exp)])

    for e, tab in enumerate(expert_tabs):
        with tab:
            rkey = f"rel_df_e{e}|{n_ch}|{n_sy}|{'|'.join(ch_names)}|{'|'.join(sy_names)}"
            if st.session_state.get(f"rkey_{e}") != rkey:
                st.session_state[f"rkey_{e}"] = rkey
                st.session_state[f"rel_df_{e}"] = init_rel_df(ch_names, sy_names, term_list)

            with st.expander(f"📋 Paste relationship matrix for Expert {e+1}", expanded=False):
                st.caption("Paste one table with rows = challenges and columns = strategies using the linguistic terms from the current scale. Header and first Challenge column are optional. Excel tab-separated paste is supported.")
                st.code(_rel_template_text(ch_names, sy_names, default_term=term_list[0] if term_list else 'M'), language="text")
                rel_paste = st.text_area(
                    f"Paste relationship matrix for Expert {e+1}",
                    value="",
                    height=180,
                    key=f"rel_paste_{e}",
                )
                rp1, rp2 = st.columns(2)
                with rp1:
                    if st.button(f"Load pasted matrix for Expert {e+1}", key=f"rel_load_{e}", use_container_width=True):
                        try:
                            st.session_state[f"rel_df_{e}"] = _parse_pasted_rel_matrix(rel_paste, ch_names, sy_names, term_list)
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Paste could not be loaded: {ex}")
                with rp2:
                    if st.button(f"Reset Expert {e+1} matrix", key=f"rel_reset_{e}", use_container_width=True):
                        st.session_state[f"rel_df_{e}"] = init_rel_df(ch_names, sy_names, term_list)
                        st.rerun()

            col_cfg = {s: st.column_config.SelectboxColumn(s, options=term_list, width="small") for s in sy_names}
            rel_df = st.data_editor(
                st.session_state[f"rel_df_{e}"],
                use_container_width=True,
                height=400,
                num_rows="fixed",
                column_config=col_cfg,
                key=f"rel_editor_{e}",
            )
            st.session_state[f"rel_df_{e}"] = rel_df
            rel_by_expert_terms.append(rel_df)

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        compute_btn = st.button(
            "🚀 **Compute TrFS-QFD Analysis**",
            type="primary",
            use_container_width=True,
        )

    if compute_btn:
        with st.spinner("Computing TrFS-QFD analysis..."):
            rel_by_expert: List[List[List[TrFS]]] = []
            for e in range(n_exp):
                mat_e: List[List[TrFS]] = []
                rdf = rel_by_expert_terms[e].copy()
                for i in range(n_ch):
                    row = []
                    for j in range(n_sy):
                        term = str(rdf.loc[i, sy_names[j]]).strip()
                        row.append(scale.get(term, list(scale.values())[0]).clip())
                    mat_e.append(row)
                rel_by_expert.append(mat_e)

            crisp, AIj, RIj, agg_rel, weighted_trfs = compute_trfs_qfd(ch_weights, rel_by_expert)

            crisp_df = pd.DataFrame(crisp, index=ch_names, columns=sy_names).round(6)
            agg_df = pd.DataFrame(
                [[trfs_to_str(agg_rel[i][j], nd=4) for j in range(n_sy)] for i in range(n_ch)],
                index=ch_names, columns=sy_names
            )
            weighted_df = pd.DataFrame(
                [[trfs_to_str(weighted_trfs[i][j], nd=4) for j in range(n_sy)] for i in range(n_ch)],
                index=ch_names, columns=sy_names
            )

            res_df = pd.DataFrame({"Strategy": sy_names, "AIj": AIj, "RIj": RIj})
            res_df["Rank"] = res_df["AIj"].rank(ascending=False, method="dense").astype(int)
            res_df = res_df.sort_values("Rank").reset_index(drop=True)

            st.session_state["crisp_df"] = crisp_df
            st.session_state["agg_df"] = agg_df
            st.session_state["weighted_df"] = weighted_df
            st.session_state["qfd_results"] = res_df
            st.session_state["RIj"] = RIj
            st.session_state["sy_names"] = sy_names

        st.success("✅ Analysis completed successfully!")

        st.markdown('<div class="section-header">📈 Results</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Strategies", len(sy_names))
        with col2:
            st.metric("Challenges", len(ch_names))
        with col3:
            st.metric("Experts", n_exp)
        with col4:
            st.metric("Top Strategy", st.session_state["qfd_results"].loc[0, "Strategy"])
        style_metric_cards()

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 AIj/RIj Results",
            "🧩 Aggregated Matrix (TrFS)",
            "⚙️ Weighted Matrix (TrFS)",
            "📋 Crisp Score Matrix",
            "📥 Export Data"
        ])

        with tab1:
            st.dataframe(st.session_state["qfd_results"].style.background_gradient(subset=["AIj", "RIj"], cmap="Blues"),
                         use_container_width=True)

        with tab2:
            st.dataframe(st.session_state["agg_df"], use_container_width=True)

        with tab3:
            st.dataframe(st.session_state["weighted_df"], use_container_width=True)

        with tab4:
            st.dataframe(st.session_state["crisp_df"].style.background_gradient(cmap="YlOrBr"), use_container_width=True)

        with tab5:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                weights_df.to_excel(writer, index=False, sheet_name="Challenge_Weights")
                st.session_state["scale_df"].to_excel(writer, index=False, sheet_name="Linguistic_Scale")
                for e in range(n_exp):
                    rel_by_expert_terms[e].to_excel(writer, index=False, sheet_name=f"Expert_{e+1}")
                st.session_state["agg_df"].reset_index().rename(columns={"index": "Challenge"})                     .to_excel(writer, index=False, sheet_name="Aggregated_TrFS")
                st.session_state["weighted_df"].reset_index().rename(columns={"index": "Challenge"})                     .to_excel(writer, index=False, sheet_name="Weighted_TrFS")
                st.session_state["crisp_df"].reset_index().rename(columns={"index": "Challenge"})                     .to_excel(writer, index=False, sheet_name="Crisp_Scores")
                st.session_state["qfd_results"].to_excel(writer, index=False, sheet_name="Results")

            st.download_button(
                "📥 Download Excel Workbook",
                data=buf.getvalue(),
                file_name="TrFS_QFD_Analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )


# ============================================================
# 2) Global Weight Page
# ============================================================

def page_global_weights():
    st.markdown('<h1 class="main-header">🌐 Global TrFS Weight Builder</h1>', unsafe_allow_html=True)
    st.markdown("**Compute global sub-criteria weights for TrFS-QFD using aggregated main and local sub-criteria weights**")
    
    entry_mode = st.radio(
        "Input mode",
        ["Paste / edit aggregated weights manually", "Use saved Module 2 sets"],
        index=0,
        horizontal=True,
        key="global_weight_input_mode"
    )

    main_df = pd.DataFrame()
    subgroup_map: Dict[str, pd.DataFrame] = {}
    subgroup_selection: Dict[str, str] = {}

    if entry_mode == "Paste / edit aggregated weights manually":
        st.markdown('<div class="section-header">🧩 Main-Criteria Aggregated Weights</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            n_main = int(st.number_input("Number of main criteria", min_value=1, value=3, step=1, key="global_n_main"))
        with c2:
            main_names_raw = st.text_input(
                "Main criterion names (comma-separated)",
                value=st.session_state.get("global_main_names_raw", "M1, M2, M3"),
                key="global_main_names_raw"
            )
        main_names = _sanitize_name_list(main_names_raw, n_main, "M")
        if len(main_names) != n_main:
            st.warning(f"You entered {len(main_names)} names; the current number of main criteria is {n_main}. The page will use the names you provided.")

        main_df = _render_manual_weight_table(
            title="Aggregated main-criteria weights",
            key_root="global_main",
            names=main_names,
            help_text="Paste the aggregated main-criteria TrFS weights extracted from Module 2."
        )
        main_names = main_df["Criterion"].astype(str).tolist()

        st.markdown('<div class="section-header">🧩 Local Aggregated Sub-Criteria Weights by Main Criterion</div>', unsafe_allow_html=True)
        st.info("For each main criterion, define how many sub-criteria belong to that group, then paste or edit the aggregated local TrFS weights extracted from Module 2.")

        for idx, main_name in enumerate(main_names):
            with st.expander(f"Sub-criteria group under main criterion: {main_name}", expanded=(idx == 0)):
                c1, c2 = st.columns([1, 2])
                with c1:
                    n_sub = int(st.number_input(
                        f"Number of sub-criteria for {main_name}",
                        min_value=1,
                        value=int(st.session_state.get(f"global_n_sub_{main_name}", 2)),
                        step=1,
                        key=f"global_n_sub_{main_name}"
                    ))
                with c2:
                    sub_names_raw = st.text_input(
                        f"Sub-criteria names for {main_name} (comma-separated)",
                        value=st.session_state.get(f"global_sub_names_raw_{main_name}", ", ".join([f"{main_name}_S1", f"{main_name}_S2"])),
                        key=f"global_sub_names_raw_{main_name}"
                    )
                sub_names = _sanitize_name_list(sub_names_raw, n_sub, f"{main_name}_S")
                if len(sub_names) != n_sub:
                    st.warning(f"You entered {len(sub_names)} sub-criteria names for {main_name}; the table will use the names you provided.")

                subgroup_map[main_name] = _render_manual_weight_table(
                    title=f"Aggregated local sub-criteria weights for {main_name}",
                    key_root=f"global_sub_{idx}",
                    names=sub_names,
                    help_text="Paste the aggregated local sub-criteria TrFS weights extracted from Module 2 for this main criterion's subgroup."
                )
                subgroup_selection[main_name] = "Manual entry"

    else:
        _ensure_bwm_saved_sets()
        saved_sets = st.session_state.get("bwm_saved_sets", {})
        if not saved_sets:
            st.info("No saved aggregated TrFS-BWM sets found. Switch to manual mode to paste weights directly, or run Module 2 and save aggregated sets.")
            return

        main_candidates = _get_saved_set_names(role="Main criteria") or sorted(saved_sets.keys())
        with st.sidebar:
            st.markdown('<div class="section-header">⚙️ Global Weight Setup</div>', unsafe_allow_html=True)
            main_set_name = st.selectbox("Aggregated main-criteria set", main_candidates, key="global_main_set")

        main_df = saved_sets[main_set_name]["df"].copy()
        main_names = main_df["Criterion"].astype(str).tolist()

        st.markdown('<div class="section-header">📦 Selected Main-Criteria Aggregated Weights</div>', unsafe_allow_html=True)
        preview_main = main_df.copy()
        preview_main["TrFS weight"] = preview_main.apply(lambda r: trfs_to_str(TrFS(r.w1, r.w2, r.w3, r.w4), nd=5), axis=1)
        preview_main["GMIR (crisp)"] = preview_main.apply(lambda r: gmIR(TrFS(r.w1, r.w2, r.w3, r.w4)), axis=1)
        st.dataframe(preview_main[["Criterion", "TrFS weight", "GMIR (crisp)"]], use_container_width=True)

        st.markdown('<div class="section-header">🔗 Map Each Main Criterion to Its Aggregated Sub-Criteria Set</div>', unsafe_allow_html=True)
        subgroup_candidates = [nm for nm in sorted(saved_sets.keys()) if nm != main_set_name]
        if not subgroup_candidates:
            st.warning("No saved sub-criteria sets were found. Switch to manual mode if you want to paste sub-criteria weights directly.")
            return

        for crit in main_names:
            default_options = [nm for nm in subgroup_candidates if saved_sets[nm].get("role") == "Sub-criteria group"] or subgroup_candidates
            subgroup_selection[crit] = st.selectbox(
                f"Sub-criteria set for main criterion: {crit}",
                subgroup_candidates,
                index=0 if default_options else 0,
                key=f"global_sub_map_{crit}"
            )
            subgroup_map[crit] = saved_sets[subgroup_selection[crit]]["df"].copy()
            with st.expander(f"Preview local aggregated sub-criteria weights for {crit}", expanded=False):
                sub_preview = subgroup_map[crit].copy()
                sub_preview["TrFS weight"] = sub_preview.apply(lambda r: trfs_to_str(TrFS(r.w1, r.w2, r.w3, r.w4), nd=5), axis=1)
                sub_preview["GMIR (crisp)"] = sub_preview.apply(lambda r: gmIR(TrFS(r.w1, r.w2, r.w3, r.w4)), axis=1)
                st.dataframe(sub_preview[["Criterion", "TrFS weight", "GMIR (crisp)"]], use_container_width=True)

    if main_df.empty:
        st.warning("Please enter or load the aggregated main-criteria weights first.")
        return

    st.markdown("---")
    compute_global_btn = st.button("✅ Compute Global Sub-Criteria Weights", type="primary", use_container_width=True)
    if not compute_global_btn:
        return

    empty_groups = [k for k, v in subgroup_map.items() if v is None or v.empty]
    if empty_groups:
        st.error("Missing local sub-criteria weights for: " + ", ".join(empty_groups))
        return

    global_df = compute_global_weights_from_sets(main_df, subgroup_map)
    if global_df.empty:
        st.error("No global weights were produced. Check the entered main and sub-criteria tables.")
        return

    global_df.insert(1, "Subgroup Source", global_df["Main Criterion"].map(subgroup_selection))

    st.success("✅ Global TrFS weights computed successfully!")
    st.markdown('<div class="section-header">📊 Global Weight Results</div>', unsafe_allow_html=True)

    display_cols = [
        "Main Criterion", "Subgroup Source", "Sub Criterion", "Main TrFS",
        "Local Sub TrFS", "Global TrFS", "GMIR (Global)"
    ]
    st.dataframe(global_df[display_cols], use_container_width=True)

    st.subheader("Global crisp weights (GMIR)")
    chart_df = global_df[["Sub Criterion", "GMIR (Global)"]].copy()
    st.bar_chart(chart_df, x="Sub Criterion", y="GMIR (Global)")

    st.markdown('<div class="section-header">💾 Send Global Weights to TrFS-QFD</div>', unsafe_allow_html=True)
    qfd_load_df = global_df[["Sub Criterion", "gw1", "gw2", "gw3", "gw4"]].copy()
    qfd_load_df = qfd_load_df.rename(columns={
        "Sub Criterion": "Challenge",
        "gw1": "w1",
        "gw2": "w2",
        "gw3": "w3",
        "gw4": "w4",
    })
    st.session_state["global_weights_df"] = qfd_load_df
    st.session_state["global_weights_results"] = global_df.copy()
    qfd_tsv_text, qfd_csv_text = _df_to_qfd_paste_blocks(qfd_load_df, name_col="Challenge")
    st.session_state["global_weights_qfd_tsv"] = qfd_tsv_text
    st.session_state["global_weights_qfd_csv"] = qfd_csv_text
    st.success("Saved in session ✅. Open Module 4 (TrFS-QFD) and load the global weights into the Challenge Weights table.")

    st.markdown('<div class="section-header">📋 Module 4 Ready-to-Paste Block</div>', unsafe_allow_html=True)
    st.caption("Copy either block below and paste it directly into Module 4 → Direct paste global sub-criteria weights. TSV is best for Excel-style paste.")
    tsv_col, csv_col = st.columns(2)
    with tsv_col:
        st.markdown("**TSV block (recommended)**")
        st.code(qfd_tsv_text, language=None)
        st.text_area(
            "TSV text area for quick select-all copy",
            value=qfd_tsv_text,
            height=180,
            key="global_weights_qfd_tsv_preview",
        )
    with csv_col:
        st.markdown("**CSV block**")
        st.code(qfd_csv_text, language=None)
        st.text_area(
            "CSV text area for quick select-all copy",
            value=qfd_csv_text,
            height=180,
            key="global_weights_qfd_csv_preview",
        )

    with st.expander("Show detailed global-weight calculations"):
        st.dataframe(global_df, use_container_width=True)

    csv_bytes = global_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download global weights (CSV)",
        data=csv_bytes,
        file_name="trfs_global_subcriteria_weights.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# 2) MILP Page (Direct Average)
# ============================================================

def page_milp():
    st.markdown('<h1 class="main-header">🎯 Strategy Portfolio Optimization</h1>', unsafe_allow_html=True)
    st.markdown("**MILP-based Strategy Porfolio Selection with Budgetary & Time Constraints**")
    st.caption("Uses **Direct Average (Expected)** Values for IC, IT, σ, and τ.")

    if "sy_names" not in st.session_state:
        st.info("ℹ️ Please run the TrFS-QFD analysis first to compute strategy importance scores.")
        return

    sy_names = st.session_state["sy_names"]
    m = len(sy_names)

    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Optimization Settings</div>', unsafe_allow_html=True)
        budget = st.number_input("💰 Available Budget (δ)", min_value=0.0, value=200000.0, step=1000.0, format="%.0f")
        time_lim = st.number_input("⏱️ Available Time (∄)", min_value=0.0, value=60.0, step=1.0, format="%.1f")
        st.markdown("---")
        n_exp = int(st.number_input("Number of Experts (avg inputs)", min_value=1, max_value=10, value=3, step=1))
        agg_mode = st.radio(
            "Aggregation",
            ["Mean across experts"] + [f"Use only Expert {k}" for k in range(1, n_exp + 1)],
            index=0
        )

    st.markdown('<div class="section-header">📊 Strategy Importance Scores</div>', unsafe_allow_html=True)

    rij_store_key = f"milp_rij_df|{m}|{'|'.join(sy_names)}"
    if st.session_state.get("milp_rij_key") != rij_store_key:
        st.session_state["milp_rij_key"] = rij_store_key
        st.session_state["milp_rij_df"] = make_rij_editor_df(sy_names, st.session_state.get("RIj", np.ones(m) / m))

    load_col, note_col = st.columns([1.2, 2.8])
    with load_col:
        if st.button("📥 Load RIj from Module 4", key="milp_load_rij", use_container_width=True):
            if "qfd_results" in st.session_state and len(st.session_state["qfd_results"]) > 0:
                qfd_df = st.session_state["qfd_results"].copy()
                qfd_map = {str(r["Strategy"]): _to_float(r["RIj"], 0.0) for _, r in qfd_df.iterrows()}
                loaded_rij = [qfd_map.get(name, 0.0) for name in sy_names]
                st.session_state["milp_rij_df"] = make_rij_editor_df(sy_names, loaded_rij)
                st.rerun()
            elif "RIj" in st.session_state:
                st.session_state["milp_rij_df"] = make_rij_editor_df(sy_names, st.session_state["RIj"])
                st.rerun()
            else:
                st.warning("No RIj values found from Module 4. Run TrFS-QFD first.")
    with note_col:
        st.caption("You can edit RIj manually here, or click the button to reload the latest RIj values computed in Module 4.")

    edited_df = st.data_editor(
        st.session_state["milp_rij_df"],
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        key="milp_rij_editor",
        column_config={
            "Strategy": st.column_config.TextColumn("Strategy", width="medium", disabled=True),
            "RIj": st.column_config.NumberColumn("RIj", min_value=0.0, max_value=1.0, format="%.6f"),
            "Rank": st.column_config.NumberColumn("Rank", disabled=True)
        }
    )
    edited_df["Strategy"] = sy_names
    edited_df["RIj"] = edited_df["RIj"].map(lambda v: _to_float(v, 0.0))
    edited_df["Rank"] = edited_df["RIj"].rank(ascending=False, method="dense").astype(int)
    st.session_state["milp_rij_df"] = edited_df.copy()
    RIj = edited_df["RIj"].to_numpy(dtype=float)

    st.markdown('<div class="section-header">👥 Expert Inputs (Direct Averages)</div>', unsafe_allow_html=True)
    st.info("Fill ONLY upper triangle (i<j) for σ and τ.")

    store_key = f"milp_avg_store|{m}|{n_exp}|{'|'.join(sy_names)}"
    if st.session_state.get("milp_avg_store_key") != store_key:
        st.session_state["milp_avg_store_key"] = store_key
        st.session_state["milp_avg_experts"] = {}
        for e in range(1, n_exp + 1):
            st.session_state["milp_avg_experts"][e] = {
                "IC_avg": pd.DataFrame({"Strategy": sy_names, "Avg": 0.0}),
                "IT_avg": pd.DataFrame({"Strategy": sy_names, "Avg": 0.0}),
                "sigma_avg": make_square_df(sy_names, 0.0),
                "tau_avg": make_square_df(sy_names, 0.0),
            }

    expert_tabs = st.tabs([f"Expert {e}" for e in range(1, n_exp + 1)])
    for e, tab in enumerate(expert_tabs, 1):
        with tab:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**IC (Avg)**")
                st.session_state["milp_avg_experts"][e]["IC_avg"] = st.data_editor(
                    st.session_state["milp_avg_experts"][e]["IC_avg"],
                    use_container_width=True,
                    height=260,
                    num_rows="fixed",
                    key=f"IC_avg_{e}",
                    column_config={
                        "Strategy": st.column_config.TextColumn("Strategy", width="medium"),
                        "Avg": st.column_config.NumberColumn("Avg", min_value=0.0, format="%.4f"),
                    }
                )
                st.markdown("**IT (Avg)**")
                st.session_state["milp_avg_experts"][e]["IT_avg"] = st.data_editor(
                    st.session_state["milp_avg_experts"][e]["IT_avg"],
                    use_container_width=True,
                    height=260,
                    num_rows="fixed",
                    key=f"IT_avg_{e}",
                    column_config={
                        "Strategy": st.column_config.TextColumn("Strategy", width="medium"),
                        "Avg": st.column_config.NumberColumn("Avg", min_value=0.0, format="%.4f"),
                    }
                )

            with c2:
                st.markdown("**σ (Avg)**")
                st.session_state["milp_avg_experts"][e]["sigma_avg"] = st.data_editor(
                    st.session_state["milp_avg_experts"][e]["sigma_avg"],
                    use_container_width=True,
                    height=260,
                    key=f"sigma_avg_{e}",
                )
                st.markdown("**τ (Avg)**")
                st.session_state["milp_avg_experts"][e]["tau_avg"] = st.data_editor(
                    st.session_state["milp_avg_experts"][e]["tau_avg"],
                    use_container_width=True,
                    height=260,
                    key=f"tau_avg_{e}",
                )

    st.markdown("---")
    solve_btn = st.button("🔍 **Solve Optimization Problem**", type="primary", use_container_width=True)

    if not solve_btn:
        return

    def get_expert_arrays(eid: int):
        IC = st.session_state["milp_avg_experts"][eid]["IC_avg"]["Avg"].map(lambda v: _to_float(v, 0.0)).to_numpy(dtype=float)
        IT = st.session_state["milp_avg_experts"][eid]["IT_avg"]["Avg"].map(lambda v: _to_float(v, 0.0)).to_numpy(dtype=float)

        sigma = st.session_state["milp_avg_experts"][eid]["sigma_avg"].applymap(lambda v: _to_float(v, 0.0)).to_numpy(dtype=float)
        tau = st.session_state["milp_avg_experts"][eid]["tau_avg"].applymap(lambda v: _to_float(v, 0.0)).to_numpy(dtype=float)

        np.fill_diagonal(sigma, 0.0)
        np.fill_diagonal(tau, 0.0)
        sigma = upper_triangle_only(sigma)
        tau = upper_triangle_only(tau)
        return IC, IT, sigma, tau

    with st.spinner("Solving MILP optimization..."):
        if agg_mode.startswith("Use only Expert"):
            eid = int(agg_mode.split()[-1])
            IC, IT, sigma, tau = get_expert_arrays(eid)
            note = f"Using Expert {eid} only (direct averages)."
        else:
            IC_list, IT_list, sigma_list, tau_list = [], [], [], []
            for eid in range(1, n_exp + 1):
                IC_e, IT_e, sigma_e, tau_e = get_expert_arrays(eid)
                IC_list.append(IC_e); IT_list.append(IT_e)
                sigma_list.append(sigma_e); tau_list.append(tau_e)

            IC = np.mean(np.stack(IC_list, axis=0), axis=0)
            IT = np.mean(np.stack(IT_list, axis=0), axis=0)
            sigma = np.mean(np.stack(sigma_list, axis=0), axis=0)
            tau = np.mean(np.stack(tau_list, axis=0), axis=0)
            np.fill_diagonal(sigma, 0.0)
            np.fill_diagonal(tau, 0.0)
            sigma = upper_triangle_only(sigma)
            tau = upper_triangle_only(tau)
            note = "Aggregated across experts by mean (direct averages)."

        best = solve_by_enumeration(RIj, IC, IT, sigma, tau, budget, time_lim)

    if best is None:
        st.error("❌ No feasible solution found. Increase budget/time or adjust inputs.")
        return

    st.success("✅ Optimization completed successfully!")
    q = best["q"]

    st.markdown('<div class="section-header">📊 Optimization Results</div>', unsafe_allow_html=True)
    st.caption(f"*{note}*")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Objective Value", f"{best['obj']:.6f}")
    with col2:
        st.metric("Total Cost", f"${best['total_cost']:,.2f}",
                  delta=f"-{best['sav_cost']:,.2f}" if best['sav_cost'] > 0 else None)
    with col3:
        st.metric("Total Time", f"{best['total_time']:.2f}",
                  delta=f"-{best['sav_time']:.2f}" if best['sav_time'] > 0 else None)
    with col4:
        st.metric("Selected", f"{int(q.sum())}/{m}", delta=f"{int(q.sum())/m*100:.0f}%")
    style_metric_cards()

    solution_df = pd.DataFrame({
        "Strategy": sy_names,
        "Selected": ["✅" if x == 1 else "❌" for x in q],
        "RIj": RIj,
        "Contribution": RIj * q,
        "IC_used": IC,
        "IT_used": IT
    })

    tab1, tab2, tab3 = st.tabs(["📋 Solution Details", "🧾 Inputs Used", "📥 Export"])
    with tab1:
        st.dataframe(solution_df.style.background_gradient(subset=["Contribution"], cmap="Greens"),
                     use_container_width=True)

    with tab2:
        st.dataframe(pd.DataFrame({"Strategy": sy_names, "IC": IC, "IT": IT}), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(pd.DataFrame(sigma, index=sy_names, columns=sy_names), use_container_width=True, height=380)
        with c2:
            st.dataframe(pd.DataFrame(tau, index=sy_names, columns=sy_names), use_container_width=True, height=380)

    with tab3:
        zbuf = io.BytesIO()
        sigma_df = pd.DataFrame(sigma, index=sy_names, columns=sy_names)
        tau_df = pd.DataFrame(tau, index=sy_names, columns=sy_names)

        with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("solution.csv", solution_df.to_csv(index=False))
            z.writestr("sigma_used.csv", sigma_df.to_csv())
            z.writestr("tau_used.csv", tau_df.to_csv())

        st.download_button(
            "📦 Download ZIP Archive",
            data=zbuf.getvalue(),
            file_name="TrFS_MILP_Results_DirectAvg.zip",
            mime="application/zip",
            use_container_width=True
        )


# ============================================================
# 3) Stratification Modeler
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
    def solve_for_p(base_multipliers, order, parents):
        def f(P):
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


def page_stratification():
    st.markdown('<h1 class="main-header">🧭 Stratification Modeler</h1>', unsafe_allow_html=True)
    st.markdown("**Network-based stratification dashboard for base and interaction scenarios**")

    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Stratification Settings</div>', unsafe_allow_html=True)
        st.info("Input Mode: **Percentages (%)**")

        num_events = int(st.number_input("Base Event Count", 1, 20, 4, key="strat_num_events"))
        st.divider()
        precision = st.slider("Graph Precision (Decimals)", 2, 6, 4, key="strat_precision")
        show_labels = st.checkbox("Show Labels on Graph", value=True, key="strat_show_labels")
        normalize = st.checkbox("Normalize to 100%", value=True, key="strat_normalize")

    with st.expander("📊 Calculation Methodology & Equations"):
        st.write("The model calculates the root scaling factor $P$ by solving for the point where the sum of all scenario probabilities equals 1.")
        st.latex(r"\sum_{i=1}^{n} Prob(S_i) = 1")
        st.write("**For Base Scenarios:**")
        st.latex(r"Prob(S_{base}) = \left( \frac{Value_{\%}}{100} \right) \times P")
        st.write("**For Interaction Scenarios:**")
        st.latex(r"Prob(S_{inter}) = \prod Prob(S_{parents})")

    col_left, col_right = st.columns([1, 1.5], gap="medium")

    with col_left:
        st.subheader("1. Definitions")

        root_id = "S1"
        base_ids = [f"S{i}" for i in range(2, num_events + 2)]
        all_base_ids = [root_id] + base_ids

        with st.expander("Base Scenarios (%)", expanded=True):
            init_data = {
                "ID": all_base_ids,
                "Label": [f"Context {i}" for i in range(1, len(all_base_ids) + 1)],
                "Value (%)": [25.0 if i < 5 else 0.0 for i in range(len(all_base_ids))]
            }

            base_df = st.data_editor(
                pd.DataFrame(init_data),
                use_container_width=True,
                hide_index=True,
                key="strat_base_df"
            )

        st.subheader("2. Interactions")
        with st.expander("Custom Logic (Parents)", expanded=True):
            inter_init = pd.DataFrame(columns=["ID", "Parents (e.g. S2,S3)", "Label"])
            if num_events == 4:
                inter_init = pd.DataFrame([
                    {"ID": "S6", "Parents (e.g. S2,S3)": "S2,S3", "Label": "Interaction Alpha"},
                    {"ID": "S7", "Parents (e.g. S2,S3)": "S6,S4", "Label": "Final Scenario"}
                ])

            inter_df = st.data_editor(
                inter_init,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                key="strat_inter_df"
            )

    with col_right:
        st.subheader("3. Analytics & Visualization")

        try:
            engine = StratificationEngine()
            base_mult = {row["ID"]: float(row["Value (%)"]) / 100.0 for _, row in base_df.iterrows()}
            labels = {row["ID"]: row["Label"] for _, row in base_df.iterrows()}

            parents_map = {sid: [] for sid in all_base_ids}
            nodes_ordered = list(all_base_ids)

            for _, row in inter_df.iterrows():
                sid = str(row["ID"]).strip()
                if not sid:
                    continue
                ps = engine.parse_parents(str(row["Parents (e.g. S2,S3)"]))
                parents_map[sid] = ps
                nodes_ordered.append(sid)
                labels[sid] = row["Label"]

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

            dot = graphviz.Digraph(format='svg')
            dot.attr(rankdir='LR', bgcolor='transparent')

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
                    fontname="Arial"
                )

            for child, ps in parents_map.items():
                for p in ps:
                    dot.edge(p, child)

            for b in base_ids:
                dot.edge(root_id, b, style="dashed", color="gray", label="base")

            st.graphviz_chart(dot)

            with st.expander("Detailed Results & CSV Export", expanded=True):
                res_list = []
                for n in sorted_nodes:
                    res_list.append({
                        "Scenario ID": n,
                        "Label": labels.get(n, ""),
                        "Probability": round(probs[n], precision + 2),
                        "Percentage (%)": f"{probs[n] * 100:.2f}%"
                    })

                final_results_df = pd.DataFrame(res_list)
                st.dataframe(final_results_df, use_container_width=True, hide_index=True)

                csv = final_results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name='stratification_results.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"Configuration Error: {e}")

    st.divider()
    st.caption("Stratification Logic Engine | Developed by Md Abdul MOKTADIR")


# ============================================================
# Main
# ============================================================

def main():
    apply_custom_styling()

    if not check_password():
        st.stop()

    with st.sidebar:
        st.markdown('<div class="section-header">🧭 Navigation</div>', unsafe_allow_html=True)
        st.markdown('### Modules')
        st.success('✅ Authenticated')
        if st.button('🚪 Logout', use_container_width=True):
            logout()

        page = st.radio(
            'Select Module',
            [
                '1) Stratification Modeler',
                '2) TrFS-BWM (scenario-aware)',
                '3) Global TrFS Weights',
                '4) TrFS-QFD',
                '5) MILP Optimization'
            ],
            index=0,
            label_visibility='collapsed'
        )

        st.markdown('---')
        st.markdown('### Workflow')
        st.markdown("""
        1. **Stratification Modeler** → Define and normalize scenario probabilities  
        2. **TrFS-BWM (scenario-aware)** → Compute aggregated TrFS weights for main or sub-criteria groups  
        3. **Global TrFS Weights** → Multiply aggregated main and local sub-criteria weights to get global sub-criteria weights  
        4. **TrFS-QFD** → Compute AIj and RIj rankings  
        5. **MILP Optimization** → Optimize the strategy portfolio using direct averages  
        """)

        st.markdown('---')
        st.markdown('Recommended flow: **Stratification → TrFS-BWM → Global Weights → TrFS-QFD → MILP**')

    render_sidebar_research_profiles()
    render_app_header()

    if page.startswith('1)'):
        page_stratification()
    elif page.startswith('2)'):
        page_trfs_bwm()
    elif page.startswith('3)'):
        page_global_weights()
    elif page.startswith('4)'):
        page_trfs_qfd()
    else:
        page_milp()

    render_footer()


if __name__ == "__main__":
    main()
