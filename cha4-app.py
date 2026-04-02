import math
import hmac
from pathlib import Path

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIG — MUST BE THE FIRST STREAMLIT COMMAND
# =========================================================
st.set_page_config(
    page_title="IVFFS Delphi · WINGS · TODIM Toolkit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

EPS = 1e-12

# =========================================================
# GLOBAL STYLING
# =========================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        * {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background-color: #f8fafc;
        }

        [data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: none;
        }

        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] div {
            color: #e2e8f0;
        }

        [data-testid="stSidebar"] .stButton > button {
            background: transparent;
            color: #f8fafc;
            border: 1px solid #94a3b8;
            border-radius: 10px;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: #60a5fa;
            color: white;
        }

        .suite-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 45%, #06b6d4 100%);
            border-radius: 24px;
            padding: 1.55rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 18px 40px rgba(29, 78, 216, 0.18);
            border: 1px solid rgba(255,255,255,0.14);
            position: relative;
            overflow: hidden;
        }

        .suite-hero::before {
            content: "";
            position: absolute;
            top: -42px;
            right: -42px;
            width: 190px;
            height: 190px;
            background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0.02) 72%);
            border-radius: 50%;
        }

        .suite-hero-title {
            color: #ffffff;
            font-size: 2rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.03em;
            position: relative;
            z-index: 1;
        }

        .suite-hero-subtitle {
            color: rgba(255,255,255,0.92);
            font-size: 1rem;
            line-height: 1.55;
            max-width: 900px;
            margin: 0;
            position: relative;
            z-index: 1;
        }

        .suite-native-card {
            background: linear-gradient(180deg, #ffffff 0%, #eff6ff 100%);
            border: 1px solid #bfdbfe;
            border-radius: 18px;
            padding: 1rem;
            min-height: 132px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            margin-bottom: 1rem;
        }

        .suite-native-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 16px 28px rgba(15, 23, 42, 0.10);
        }

        .suite-native-title {
            color: #1e3a8a;
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .suite-native-text {
            color: #334155;
            font-size: 0.88rem;
            line-height: 1.55;
        }

        .suite-badge-row {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
            margin-top: 0.9rem;
        }

        .suite-badge {
            display: inline-block;
            padding: 0.36rem 0.75rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            color: #ffffff;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.16);
        }

        .page-banner {
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 55%, #ec4899 100%);
            border-radius: 20px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1.15rem;
            box-shadow: 0 12px 30px rgba(59, 130, 246, 0.18);
            border: 1px solid rgba(255,255,255,0.18);
        }

        .page-banner.green {
            background: linear-gradient(135deg, #059669 0%, #0ea5e9 55%, #2563eb 100%);
        }

        .page-banner.orange {
            background: linear-gradient(135deg, #f59e0b 0%, #ef4444 55%, #ec4899 100%);
        }

        .page-banner.slate {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #06b6d4 100%);
        }

        .page-banner-title {
            color: white;
            font-size: 1.65rem;
            font-weight: 800;
            margin: 0 0 0.3rem 0;
            letter-spacing: -0.02em;
        }

        .page-banner-subtitle {
            color: rgba(255,255,255,0.92);
            font-size: 0.96rem;
            line-height: 1.5;
            margin: 0;
        }

        .sidebar-section-title {
            color: #e2e8f0 !important;
            font-size: 0.88rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 0.25rem 0 0.75rem 0;
        }

        .sidebar-profile-card {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 16px;
            padding: 0.85rem;
            margin-bottom: 0.85rem;
            backdrop-filter: blur(6px);
        }

        .sidebar-profile-name {
            color: #f8fafc;
            font-size: 0.95rem;
            font-weight: 700;
            margin-top: 0.55rem;
            margin-bottom: 0.2rem;
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

        .app-footer {
            text-align: center;
            margin-top: 2.5rem;
            padding: 1rem;
            color: #64748b;
            font-size: 0.75rem;
            border-top: 1px solid #e2e8f0;
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
            font-size: clamp(2.6rem, 4vw, 3.55rem);
            line-height: 1.12;
            margin: 0 0 0.85rem 0;
            letter-spacing: -0.04em;
            position: relative;
            z-index: 1;
        }

        .login-hero-box p {
            color: rgba(255,255,255,0.93) !important;
            margin: 0 auto;
            max-width: 560px;
            font-size: 1.18rem;
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
                font-size: 1.02rem;
            }
        }

        .stDataFrame, .stDataEditor {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }

        .stMetric {
            background: #f8fafc;
            border-radius: 12px;
            padding: 0.8rem;
            text-align: center;
            border: 1px solid #eef2f6;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# AUTHENTICATION
# =========================================================
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
                    <h1>IVFFS Decision Analytics Toolkit</h1>
                    <p>
                        Secure access to the integrated IVFFS-Delphi, IVFFS-WINGS,
                        and IVFFS-TODIM research workspace.
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


# =========================================================
# UI HELPERS
# =========================================================
def get_asset_path(filename: str) -> Path:
    return Path(__file__).parent / "assets" / filename


def render_page_banner(title: str, subtitle: str, theme: str = "default"):
    theme_class = "" if theme == "default" else f" {theme}"
    st.markdown(
        f"""
        <div class="page-banner{theme_class}">
            <div class="page-banner-title">{title}</div>
            <div class="page-banner-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_banner():
    st.markdown(
        """
        <div class="suite-hero">
            <div class="suite-hero-title">🧠 Integrated IVFFS Delphi · WINGS · TODIM Toolkit</div>
            <div class="suite-hero-subtitle">
                A unified research interface for screening criteria, mapping causal structure,
                and ranking alternatives using interval-valued Fermatean fuzzy sets.
            </div>
            <div class="suite-badge-row">
                <span class="suite-badge">Secure Access</span>
                <span class="suite-badge">Research Profiles</span>
                <span class="suite-badge">Interactive Analytics</span>
                <span class="suite-badge">CSV Export</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="suite-native-card">
                <div class="suite-native-title">📘 IVFFS-Delphi</div>
                <div class="suite-native-text">
                    Screen and validate candidate criteria through linguistic Delphi responses,
                    IVFFDWA aggregation, and threshold-based selection.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="suite-native-card">
                <div class="suite-native-title">🕸️ IVFFS-WINGS</div>
                <div class="suite-native-text">
                    Explore component strength, causal interaction, total relation structure,
                    and final importance weights for the system.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="suite-native-card">
                <div class="suite-native-title">📊 IVFFS-TODIM</div>
                <div class="suite-native-text">
                    Evaluate alternatives with expert linguistic judgments, normalization,
                    dominance comparison, and final superiority ranking.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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
            © 2026 Developed by <strong>Moktadir M.A.</strong> and <strong>REN J.Z.</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# IVFFS REPRESENTATION
#   IVFFS = ([mu_L, mu_U], [nu_L, nu_U])
#   store as (mu_L, mu_U, nu_L, nu_U)
# =========================================================
def clamp01(x):
    return float(min(1.0 - EPS, max(EPS, float(x))))


def make_ivffs(muL, muU, nuL, nuU):
    muL, muU, nuL, nuU = map(float, (muL, muU, nuL, nuU))
    muL, muU = min(muL, muU), max(muL, muU)
    nuL, nuU = min(nuL, nuU), max(nuL, nuU)
    return (clamp01(muL), clamp01(muU), clamp01(nuL), clamp01(nuU))


def format_ivffs(v):
    muL, muU, nuL, nuU = v
    return f"([{muL:.6f},{muU:.6f}],[{nuL:.6f},{nuU:.6f}])"


# =========================================================
# IVFFDWA (Dombi-based) AGGREGATION
# =========================================================
def _safe_pow(x, p):
    return float(x) ** float(p)


def agg_membership_bound(x_list, w_list, alpha, power=3.0):
    alpha = max(float(alpha), EPS)
    s = 0.0
    for x, w in zip(x_list, w_list):
        x = clamp01(x)
        xp = _safe_pow(x, power)
        frac = xp / max(EPS, (1.0 - xp))
        s += float(w) * _safe_pow(frac, alpha)

    inner = _safe_pow(s, 1.0 / alpha)
    val = 1.0 - (1.0 / (1.0 + inner))
    return _safe_pow(val, 1.0 / power)


def agg_nonmembership_bound(y_list, w_list, alpha, power=3.0):
    alpha = max(float(alpha), EPS)
    s = 0.0
    for y, w in zip(y_list, w_list):
        y = clamp01(y)
        yp = _safe_pow(y, power)
        frac = (1.0 - yp) / max(EPS, yp)
        s += float(w) * _safe_pow(frac, alpha)

    inner = _safe_pow(s, 1.0 / alpha)
    denom = _safe_pow((1.0 + inner), 1.0 / power)
    return 1.0 / max(EPS, denom)


def ivffdwa_aggregate(ivffs_list, w_list, alpha):
    muL_list = [v[0] for v in ivffs_list]
    muU_list = [v[1] for v in ivffs_list]
    nuL_list = [v[2] for v in ivffs_list]
    nuU_list = [v[3] for v in ivffs_list]

    a = agg_membership_bound(muL_list, w_list, alpha, power=3.0)
    b = agg_membership_bound(muU_list, w_list, alpha, power=3.0)
    c = agg_nonmembership_bound(nuL_list, w_list, alpha, power=3.0)
    d = agg_nonmembership_bound(nuU_list, w_list, alpha, power=3.0)

    return make_ivffs(a, b, c, d)


# =========================================================
# IVFFS SCORE FUNCTION Ψ
# =========================================================
def ivffs_score(v):
    muL, muU, nuL, nuU = v
    return 0.5 * (0.5 * (muL ** 3 + muU ** 3 - nuL ** 3 - nuU ** 3) + 1.0)


# =========================================================
# LINGUISTIC SCALES
# =========================================================
DELPHI_LINGUISTIC = {
    "VLI": make_ivffs(0.10, 0.20, 0.80, 0.90),
    "LI": make_ivffs(0.20, 0.50, 0.70, 0.80),
    "MI": make_ivffs(0.50, 0.70, 0.50, 0.70),
    "HI": make_ivffs(0.70, 0.80, 0.20, 0.50),
    "VHI": make_ivffs(0.80, 0.90, 0.10, 0.20),
}
DELPHI_LINGUISTIC_FULL = {
    "VLI": "Very Low Important",
    "LI": "Low Important",
    "MI": "Medium Important",
    "HI": "High Important",
    "VHI": "Very High Important",
}

WINGS_STRENGTH = {
    "VLR": make_ivffs(0.10, 0.20, 0.80, 0.90),
    "LR": make_ivffs(0.20, 0.50, 0.70, 0.80),
    "MR": make_ivffs(0.50, 0.70, 0.50, 0.70),
    "HR": make_ivffs(0.70, 0.80, 0.20, 0.50),
    "VHR": make_ivffs(0.80, 0.90, 0.10, 0.20),
}
WINGS_STRENGTH_FULL = {
    "VLR": "Very Low Relevance",
    "LR": "Low Relevance",
    "MR": "Medium Relevance",
    "HR": "High Relevance",
    "VHR": "Very High Relevance",
}

WINGS_INFLUENCE = {
    "ELI": make_ivffs(0.05, 0.15, 0.85, 0.95),
    "VLI": make_ivffs(0.15, 0.25, 0.75, 0.85),
    "LI": make_ivffs(0.25, 0.35, 0.65, 0.75),
    "MI": make_ivffs(0.50, 0.50, 0.50, 0.50),
    "HI": make_ivffs(0.65, 0.75, 0.25, 0.35),
    "VHI": make_ivffs(0.75, 0.85, 0.15, 0.25),
    "EHI": make_ivffs(0.85, 0.95, 0.05, 0.15),
}
WINGS_INFLUENCE_FULL = {
    "ELI": "Extremely Low Influence",
    "VLI": "Very Low Influence",
    "LI": "Low Influence",
    "MI": "Medium Influence",
    "HI": "High Influence",
    "VHI": "Very High Influence",
    "EHI": "Extremely High Influence",
}

TODIM_LINGUISTIC = {
    "VP": make_ivffs(0.10, 0.15, 0.90, 0.95),
    "P": make_ivffs(0.20, 0.25, 0.80, 0.85),
    "MP": make_ivffs(0.30, 0.35, 0.70, 0.75),
    "F": make_ivffs(0.50, 0.55, 0.40, 0.45),
    "MG": make_ivffs(0.70, 0.75, 0.30, 0.35),
    "G": make_ivffs(0.80, 0.85, 0.20, 0.25),
    "VG": make_ivffs(0.90, 0.95, 0.10, 0.15),
}
TODIM_FULL = {
    "VP": "Very Poor",
    "P": "Poor",
    "MP": "Medium Poor",
    "F": "Fair",
    "MG": "Medium Good",
    "G": "Good",
    "VG": "Very Good",
}


# =========================================================
# GENERAL HELPERS
# =========================================================
def parse_csv_names(s):
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def normalize_ivffs_todim(v, crit_type):
    crit_type = (crit_type or "").strip().lower()
    if crit_type.startswith("c"):
        muL, muU, nuL, nuU = v
        return make_ivffs(nuL, nuU, muL, muU)
    return v


def round_df(df, d=6):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(d)
    return out


def sync_strength_df(old_df, comps):
    old_df = old_df.copy()
    if "Component" not in old_df.columns:
        old_df["Component"] = old_df.index.astype(str)
    if "Strength" not in old_df.columns:
        old_df["Strength"] = "MR"

    lookup = dict(zip(old_df["Component"].astype(str), old_df["Strength"].astype(str)))
    new_strength = [lookup.get(c, "MR") for c in comps]
    return pd.DataFrame({"Component": comps, "Strength": new_strength})


def sync_influence_df(old_df, comps):
    old_df = old_df.copy()
    old_df.index = old_df.index.astype(str)
    old_df.columns = old_df.columns.astype(str)

    new_df = pd.DataFrame("ELI", index=comps, columns=comps)
    common_r = [c for c in comps if c in old_df.index]
    common_c = [c for c in comps if c in old_df.columns]
    if common_r and common_c:
        new_df.loc[common_r, common_c] = old_df.loc[common_r, common_c].values

    for i in range(len(comps)):
        new_df.iloc[i, i] = "—"
    return new_df


def show_expert_weights(prefix, k):
    st.markdown("**Expert weights (must sum to 1.0)**")
    if k == 1:
        st.info("Single expert → weight = 1.0")
        return [1.0]

    cols = st.columns(k)
    w_exp = []
    for i in range(k):
        with cols[i]:
            w_exp.append(
                st.number_input(
                    f"E{i + 1}",
                    min_value=0.0,
                    max_value=1.0,
                    value=round(1.0 / k, 6),
                    step=0.00001,
                    format="%.6f",
                    key=f"{prefix}_wexp_{i}",
                )
            )
    return w_exp


def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def make_download_button(label, df, filename):
    st.download_button(
        label=label,
        data=df_to_csv_bytes(df),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def generate_wings_interaction_graph(infl_df, comps, title="Interaction Graph"):
    graph = graphviz.Digraph(comment=title)
    graph.attr(rankdir="LR")

    for comp in comps:
        graph.node(comp, label=comp, shape="box", style="rounded,filled", fillcolor="lightblue")

    for i, src in enumerate(comps):
        for j, dst in enumerate(comps):
            if i == j:
                continue
            term = str(infl_df.iloc[i, j]).strip()
            if term in WINGS_INFLUENCE and term != "ELI":
                graph.edge(src, dst, label=term)

    return graph


def render_wings_plots(results_df):
    st.markdown("### Cause–Effect Map")
    fig, ax = plt.subplots(figsize=(9, 7))

    x_mean = results_df["Engagement"].mean()
    for _, row in results_df.iterrows():
        ax.scatter(row["Engagement"], row["Role"], s=120)
        ax.text(row["Engagement"], row["Role"], f" {row['Component']}", fontsize=10)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(x_mean, linestyle="--", linewidth=1)
    ax.set_xlabel("Engagement (TI + TR)")
    ax.set_ylabel("Role (TI - TR)")
    ax.set_title("Cause–Effect Diagram")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.markdown("### Component Weights")
    weight_df = results_df[["Component", "Weight"]].sort_values("Weight", ascending=False).copy()
    st.dataframe(weight_df.round(6), use_container_width=True, hide_index=True)
    st.bar_chart(weight_df.set_index("Component")[["Weight"]])


# =========================================================
# MODULE 1: IVFFS-DELPHI
# =========================================================
def ivffs_delphi_module():
    render_page_banner(
        "📘 IVFFS-Delphi Screening Module",
        "Screen criteria using linguistic Delphi inputs, IVFFDWA aggregation, and threshold-based selection.",
        theme="default",
    )

    with st.expander("Delphi linguistic scale", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Code": k,
                        "Meaning": DELPHI_LINGUISTIC_FULL[k],
                        "IVFFS": format_ivffs(v),
                    }
                    for k, v in DELPHI_LINGUISTIC.items()
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Step 1: Criteria, respondents, threshold")
    col1, col2, col3, col4 = st.columns(4)
    n = int(col1.number_input("Number of criteria", min_value=1, max_value=100, value=5, step=1, key="d_n"))
    k = int(col2.number_input("Number of respondents", min_value=1, max_value=100, value=4, step=1, key="d_k"))
    alpha = col3.number_input("Dombi parameter (α)", min_value=0.01, max_value=50.0, value=0.5, step=0.01, key="d_alpha")
    threshold = col4.number_input("Selection threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01, key="d_threshold")

    items_in = st.text_input("Criteria names (comma-separated)", value="F1,F2,F3,F4,F5", key="d_items_in")
    items = parse_csv_names(items_in)
    if len(items) != n:
        items = [f"F{i + 1}" for i in range(n)]

    respondent_cols = [f"R{i + 1}" for i in range(k)]
    w_exp = [1.0 / k] * k

    c1, c2 = st.columns(2)
    with c1:
        st.info(f"Equal respondent weight applied automatically: 1/{k} = {1.0 / k:.6f}")
    with c2:
        st.metric("Total weight", f"{sum(w_exp):.6f}")

    sk_terms = "d_terms_matrix"
    if sk_terms not in st.session_state:
        st.session_state[sk_terms] = pd.DataFrame("MI", index=items, columns=respondent_cols)

    old_df = st.session_state[sk_terms].copy()
    old_df.index = old_df.index.astype(str)
    old_df.columns = old_df.columns.astype(str)
    need_reset = list(old_df.index) != items or list(old_df.columns) != respondent_cols

    if need_reset:
        new_df = pd.DataFrame("MI", index=items, columns=respondent_cols)
        common_rows = [r for r in items if r in old_df.index]
        common_cols = [c for c in respondent_cols if c in old_df.columns]
        if common_rows and common_cols:
            new_df.loc[common_rows, common_cols] = old_df.loc[common_rows, common_cols]
        st.session_state[sk_terms] = new_df

    st.subheader("Step 2: Delphi linguistic evaluation table")
    st.markdown("**Rows = criteria, columns = respondents**")

    edited_df = st.data_editor(
        st.session_state[sk_terms],
        use_container_width=True,
        column_config={c: st.column_config.SelectboxColumn(c, options=list(DELPHI_LINGUISTIC.keys())) for c in respondent_cols},
        key="d_editor_matrix",
    )

    edited_df = edited_df.copy().astype(str)
    for c in respondent_cols:
        edited_df[c] = edited_df[c].str.strip().apply(lambda x: x if x in DELPHI_LINGUISTIC else "MI")
    edited_df.index = items
    st.session_state[sk_terms] = edited_df

    if st.button("✅ Run IVFFS-Delphi", type="primary", use_container_width=True, key="d_run_btn"):
        with st.spinner("Computing IVFFS-Delphi..."):
            ling_mat = st.session_state[sk_terms].copy()

            agg_ivffs = {}
            for item in items:
                vals = []
                for resp in respondent_cols:
                    term = str(ling_mat.loc[item, resp]).strip()
                    vals.append(DELPHI_LINGUISTIC.get(term, DELPHI_LINGUISTIC["MI"]))
                agg_ivffs[item] = ivffdwa_aggregate(vals, w_exp, alpha)

            df_agg = pd.DataFrame({
                "Criterion": items,
                "Aggregated IVFFS": [format_ivffs(agg_ivffs[item]) for item in items],
            })

            scores = [ivffs_score(agg_ivffs[item]) for item in items]
            decisions = ["Selected" if s >= threshold else "Rejected" for s in scores]
            res = pd.DataFrame({
                "Criterion": items,
                "Aggregated IVFFS": [format_ivffs(agg_ivffs[item]) for item in items],
                "Score Ψ": scores,
                "Threshold": [threshold] * len(items),
                "Decision": decisions,
            })

            accepted_df = res[res["Decision"] == "Selected"].copy()
            rejected_df = res[res["Decision"] == "Rejected"].copy()
            t1, t2, t3 = st.tabs(["📋 Results", "📈 Plot", "📝 Export"])

            with t1:
                st.markdown("### Delphi linguistic evaluation table")
                st.dataframe(ling_mat, use_container_width=True)

                with st.expander("Transformed respondent IVFFS table"):
                    trans_df = ling_mat.copy()
                    for c in respondent_cols:
                        trans_df[c] = trans_df[c].apply(lambda t: format_ivffs(DELPHI_LINGUISTIC.get(t, DELPHI_LINGUISTIC["MI"])))
                    st.dataframe(trans_df, use_container_width=True)

                st.markdown("### Aggregated IVFFS evaluations (IVFFDWA)")
                st.dataframe(df_agg, hide_index=True, use_container_width=True)

                st.markdown("### Final IVFFS-Delphi screening results")
                st.dataframe(round_df(res, 6), hide_index=True, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Accepted criteria", len(accepted_df))
                c2.metric("Rejected criteria", len(rejected_df))
                c3.metric("Threshold", f"{threshold:.2f}")

                if len(accepted_df) > 0:
                    st.success("Accepted criteria:\n\n" + "\n".join([f"- {x}" for x in accepted_df["Criterion"].tolist()]))
                else:
                    st.warning("No criterion passed the current threshold.")

                if len(rejected_df) > 0:
                    with st.expander("Rejected criteria"):
                        st.write("\n".join([f"- {x}" for x in rejected_df["Criterion"].tolist()]))

            with t2:
                st.markdown("### Plot")
                st.bar_chart(res.set_index("Criterion")[["Score Ψ"]])

            with t3:
                st.markdown("### Export")
                make_download_button("⬇️ Download Delphi results as CSV", round_df(res, 6), "ivffs_delphi_results.csv")
                make_download_button(
                    "⬇️ Download Delphi linguistic table as CSV",
                    ling_mat.reset_index().rename(columns={"index": "Criterion"}),
                    "ivffs_delphi_linguistic_table.csv",
                )

            st.session_state["delphi_selected_items"] = accepted_df["Criterion"].tolist()


# =========================================================
# MODULE 2: IVFFS-WINGS
# =========================================================
def ivffs_wings_module():
    render_page_banner(
        "🕸️ IVFFS-WINGS Analysis Module",
        "Model component strength, influence structure, total relation matrix, and final system weights.",
        theme="orange",
    )

    with st.expander("Strength scale (Table A1)", expanded=False):
        st.dataframe(
            pd.DataFrame([
                {"Code": k, "Meaning": WINGS_STRENGTH_FULL[k], "IVFFS": format_ivffs(v)}
                for k, v in WINGS_STRENGTH.items()
            ]),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Influence scale (Table A4)", expanded=False):
        st.dataframe(
            pd.DataFrame([
                {"Code": k, "Meaning": WINGS_INFLUENCE_FULL[k], "IVFFS": format_ivffs(v)}
                for k, v in WINGS_INFLUENCE.items()
            ]),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Step 1: Components and experts")
    n = int(st.number_input("Number of components", min_value=2, max_value=30, value=5, step=1, key="w_n"))
    k = int(st.number_input("Number of experts", min_value=1, max_value=20, value=4, step=1, key="w_k"))

    colA, colB = st.columns(2)
    default_comp = "C1,C2,C3,C4,C5"
    if st.session_state.get("delphi_selected_items"):
        default_comp = ",".join(st.session_state["delphi_selected_items"])

    comp_names_in = colA.text_input("Component names (comma-separated)", value=default_comp, key="w_comps_in")
    alpha = colB.number_input("Dombi parameter (α)", min_value=0.01, max_value=50.0, value=0.5, step=0.01, key="w_alpha")

    comps = parse_csv_names(comp_names_in)
    if len(comps) != n:
        comps = [f"C{i + 1}" for i in range(n)]

    w_exp = show_expert_weights("w", k)
    if not np.isclose(sum(w_exp), 1.0):
        st.error(f"Expert weights must sum to 1.0 (now {sum(w_exp):.6f}).")
        return

    sk_strength = "w_strength_terms"
    sk_infl = "w_infl_terms"
    valid_strength = list(WINGS_STRENGTH.keys())
    valid_influence = list(WINGS_INFLUENCE.keys())

    if sk_strength not in st.session_state or len(st.session_state[sk_strength]) != k:
        st.session_state[sk_strength] = [pd.DataFrame({"Component": comps, "Strength": ["MR"] * len(comps)}) for _ in range(k)]

    if sk_infl not in st.session_state or len(st.session_state[sk_infl]) != k:
        st.session_state[sk_infl] = [pd.DataFrame("ELI", index=comps, columns=comps) for _ in range(k)]
        for e in range(k):
            for i in range(len(comps)):
                st.session_state[sk_infl][e].iloc[i, i] = "—"

    for e in range(k):
        st.session_state[sk_strength][e] = sync_strength_df(st.session_state[sk_strength][e], comps)
        st.session_state[sk_infl][e] = sync_influence_df(st.session_state[sk_infl][e], comps)

        st.session_state[sk_strength][e]["Strength"] = (
            st.session_state[sk_strength][e]["Strength"].astype(str).str.strip().apply(lambda x: x if x in valid_strength else "MR")
        )

        infl_df = st.session_state[sk_infl][e].copy().astype(str)
        for i in range(len(comps)):
            for j in range(len(comps)):
                if i == j:
                    infl_df.iloc[i, j] = "—"
                else:
                    val = infl_df.iloc[i, j].strip()
                    infl_df.iloc[i, j] = val if val in valid_influence else "ELI"
        st.session_state[sk_infl][e] = infl_df

    st.subheader("Step 2: Expert inputs")
    tabs = st.tabs([f"Expert {i + 1}" for i in range(k)])

    for e, tab in enumerate(tabs):
        with tab:
            st.markdown("### (A) Strength of each component (diagonal)")
            dfS = st.data_editor(
                st.session_state[sk_strength][e],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Component": st.column_config.TextColumn("Component", disabled=True),
                    "Strength": st.column_config.SelectboxColumn("Strength", options=valid_strength),
                },
                key=f"w_strength_editor_{e}",
            )
            dfS = sync_strength_df(dfS, comps)
            dfS["Strength"] = dfS["Strength"].astype(str).str.strip().apply(lambda x: x if x in valid_strength else "MR")
            st.session_state[sk_strength][e] = dfS

            st.markdown("### (B) Influence matrix (row → column) — off-diagonal")
            col_cfg = {c: st.column_config.SelectboxColumn(c, options=["—"] + valid_influence) for c in comps}
            editedI = st.data_editor(
                st.session_state[sk_infl][e],
                use_container_width=True,
                column_config=col_cfg,
                key=f"w_infl_editor_{e}",
            )

            editedI = sync_influence_df(editedI, comps).astype(str)
            for i in range(len(comps)):
                for j in range(len(comps)):
                    if i == j:
                        editedI.iloc[i, j] = "—"
                    else:
                        val = editedI.iloc[i, j].strip()
                        editedI.iloc[i, j] = val if val in valid_influence else "ELI"
            st.session_state[sk_infl][e] = editedI

            st.markdown("### Interaction graph")
            graph = generate_wings_interaction_graph(st.session_state[sk_infl][e], comps, title=f"Expert {e + 1} Graph")
            st.graphviz_chart(graph, use_container_width=True)

    if st.button("✅ Run IVFFS-WINGS", type="primary", use_container_width=True, key="w_run_btn"):
        with st.spinner("Computing IVFFS-WINGS..."):
            n_comp = len(comps)
            expert_sidrm = []

            for e in range(k):
                strength_terms = st.session_state[sk_strength][e]["Strength"].astype(str).str.strip().tolist()
                infl_terms = st.session_state[sk_infl][e].copy().astype(str)
                mat = [[None] * n_comp for _ in range(n_comp)]

                for i in range(n_comp):
                    for j in range(n_comp):
                        if i == j:
                            term = strength_terms[i]
                            mat[i][j] = WINGS_STRENGTH.get(term, WINGS_STRENGTH["MR"])
                        else:
                            term = infl_terms.iloc[i, j].strip()
                            if term == "—" or term not in WINGS_INFLUENCE:
                                term = "ELI"
                            mat[i][j] = WINGS_INFLUENCE[term]
                expert_sidrm.append(mat)

            agg = [[None] * n_comp for _ in range(n_comp)]
            for i in range(n_comp):
                for j in range(n_comp):
                    cell_list = [expert_sidrm[e][i][j] for e in range(k)]
                    agg[i][j] = ivffdwa_aggregate(cell_list, w_exp, alpha)

            agg_df = pd.DataFrame(
                [[format_ivffs(agg[i][j]) for j in range(n_comp)] for i in range(n_comp)],
                index=comps,
                columns=comps,
            )

            C = np.array([[ivffs_score(agg[i][j]) for j in range(n_comp)] for i in range(n_comp)], dtype=float)
            C_df = pd.DataFrame(C, index=comps, columns=comps)

            sC = float(C.sum())
            if abs(sC) < EPS:
                st.error("Normalization failed: sum(C) is zero.")
                return

            N = C / sC
            N_df = pd.DataFrame(N, index=comps, columns=comps)
            I = np.eye(n_comp)
            T = N @ np.linalg.pinv(I - N)
            T_df = pd.DataFrame(T, index=comps, columns=comps)

            TI = T.sum(axis=1)
            TR = T.sum(axis=0)
            ENG = TI + TR
            ROLE = TI - TR
            EV = np.sqrt(ENG ** 2 + ROLE ** 2)
            W = EV / max(EPS, float(EV.sum()))

            out = pd.DataFrame({
                "Component": comps,
                "TI": TI,
                "TR": TR,
                "Engagement": ENG,
                "Role": ROLE,
                "Expected value": EV,
                "Weight": W,
            })
            out["Type"] = np.where(out["Role"] >= 0, "Cause", "Effect")
            t1, t2, t3 = st.tabs(["📋 Results", "📈 Plot", "📝 Export"])

            with t1:
                st.markdown("### Aggregated IVFFS-SIDRM (IVFFDWA)")
                st.dataframe(agg_df, use_container_width=True)
                st.markdown("### Score Matrix C (Ψ)")
                st.dataframe(C_df.round(6), use_container_width=True)
                st.markdown("### Normalized Score Matrix N")
                st.dataframe(N_df.round(6), use_container_width=True)
                st.markdown("### Total Relation Matrix T")
                st.dataframe(T_df.round(6), use_container_width=True)
                st.markdown("### Final Results")
                st.dataframe(out.round(6), use_container_width=True, hide_index=True)

            with t2:
                render_wings_plots(out)

            with t3:
                st.markdown("### Export")
                make_download_button("⬇️ Download WINGS final results as CSV", round_df(out, 6), "ivffs_wings_results.csv")
                make_download_button(
                    "⬇️ Download WINGS total relation matrix as CSV",
                    round_df(T_df.reset_index().rename(columns={"index": "Component"}), 6),
                    "ivffs_wings_total_relation_matrix.csv",
                )


# =========================================================
# MODULE 3: IVFFS-TODIM
# =========================================================
def ivffs_todim_module():
    render_page_banner(
        "📊 IVFFS-TODIM Ranking Module",
        "Aggregate expert evaluations, normalize by criterion type, compute dominance, and rank alternatives.",
        theme="green",
    )

    with st.expander("TODIM linguistic scale (VP…VG)", expanded=False):
        st.dataframe(
            pd.DataFrame([
                {"Code": k, "Meaning": TODIM_FULL[k], "IVFFS": format_ivffs(v)}
                for k, v in TODIM_LINGUISTIC.items()
            ]),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Step 1: Alternatives, criteria, types, weights")
    c1, c2 = st.columns(2)
    alts_in = c1.text_input("Alternatives (comma-separated)", "S1,S2,S3,S4", key="t_alts_in")
    crits_in = c2.text_input("Criteria (comma-separated)", "C1,C2,C3", key="t_crits_in")

    alts = parse_csv_names(alts_in)
    crits = parse_csv_names(crits_in)
    if len(alts) < 2 or len(crits) < 1:
        st.warning("Provide at least 2 alternatives and at least 1 criterion.")
        return

    cfg_key = "t_cfg"
    if cfg_key not in st.session_state or list(st.session_state[cfg_key]["Criterion"]) != crits:
        w0 = [round(1 / len(crits), 5)] * len(crits)
        if len(crits) > 1:
            w0[-1] = round(1.0 - sum(w0[:-1]), 5)
        st.session_state[cfg_key] = pd.DataFrame({"Criterion": crits, "Type": ["Benefit"] * len(crits), "Weight": w0})

    cfg = st.data_editor(
        st.session_state[cfg_key],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Benefit", "Cost"]),
            "Weight": st.column_config.NumberColumn("Weight", format="%.5f", min_value=0.0, max_value=1.0, step=0.00001),
        },
        key="t_cfg_editor",
    )
    st.session_state[cfg_key] = cfg

    types = cfg["Type"].tolist()
    w_crit = cfg["Weight"].astype(float).tolist()
    if not np.isclose(sum(w_crit), 1.0):
        st.error(f"Criteria weights must sum to 1.0 (now {sum(w_crit):.5f}).")
        return

    st.subheader("Step 2: Experts, weights, evaluations")
    k = int(st.number_input("Number of experts", min_value=1, max_value=30, value=4, step=1, key="t_k"))
    alpha = st.number_input("Dombi parameter (α)", min_value=0.01, max_value=50.0, value=0.5, step=0.01, key="t_alpha")

    w_exp = show_expert_weights("t", k)
    if not np.isclose(sum(w_exp), 1.0):
        st.error(f"Expert weights must sum to 1.0 (now {sum(w_exp):.6f}).")
        return

    mat_key = "t_terms"
    need_reset = (mat_key not in st.session_state) or (len(st.session_state[mat_key]) != k)
    if not need_reset:
        df0 = st.session_state[mat_key][0]
        need_reset = list(df0.index) != alts or list(df0.columns) != crits
    if need_reset:
        st.session_state[mat_key] = [pd.DataFrame("F", index=alts, columns=crits) for _ in range(k)]

    tabs = st.tabs([f"Expert {i + 1}" for i in range(k)])
    for i, tab in enumerate(tabs):
        with tab:
            st.session_state[mat_key][i] = st.data_editor(
                st.session_state[mat_key][i],
                use_container_width=True,
                column_config={c: st.column_config.SelectboxColumn(c, options=list(TODIM_LINGUISTIC.keys())) for c in crits},
                key=f"t_editor_{i}",
            )

    st.subheader("Step 3: TODIM parameters")
    theta = st.number_input("Loss attenuation factor (θ)", min_value=0.01, max_value=50.0, value=1.0, step=0.01, key="t_theta")
    ref_mode = st.selectbox("Reference criterion (ωr)", options=["Max weight (default)"] + crits, index=0, key="t_ref")

    if st.button("✅ Run IVFFS-TODIM", type="primary", use_container_width=True, key="t_run_btn"):
        with st.spinner("Computing IVFFS-TODIM..."):
            agg = {}
            for a in alts:
                for c in crits:
                    vals = []
                    for e in range(k):
                        term = st.session_state[mat_key][e].loc[a, c]
                        vals.append(TODIM_LINGUISTIC[term])
                    agg[(a, c)] = ivffdwa_aggregate(vals, w_exp, alpha)

            agg_df = pd.DataFrame(
                [[format_ivffs(agg[(a, c)]) for c in crits] for a in alts],
                index=alts,
                columns=crits,
            )

            norm = {}
            for j, c in enumerate(crits):
                for a in alts:
                    norm[(a, c)] = normalize_ivffs_todim(agg[(a, c)], types[j])

            norm_df = pd.DataFrame(
                [[format_ivffs(norm[(a, c)]) for c in crits] for a in alts],
                index=alts,
                columns=crits,
            )

            tau = np.array([[ivffs_score(norm[(a, c)]) for c in crits] for a in alts], dtype=float)
            tau_df = pd.DataFrame(tau, index=alts, columns=crits).round(6)

            if ref_mode == "Max weight (default)":
                w_r = max(w_crit)
            else:
                w_r = float(w_crit[crits.index(ref_mode)])

            if abs(w_r) < EPS:
                st.error("Reference weight ωr is zero.")
                return

            w_rel = np.array([w / w_r for w in w_crit], dtype=float)
            sum_w_rel = float(w_rel.sum())
            if sum_w_rel <= EPS:
                st.error("Sum of relative weights is zero.")
                return

            nA, nC = len(alts), len(crits)
            dominance = np.zeros((nA, nA), dtype=float)
            for i in range(nA):
                for q in range(nA):
                    if i == q:
                        continue
                    s = 0.0
                    for j in range(nC):
                        diff = tau[i, j] - tau[q, j]
                        ad = abs(diff)
                        if ad <= 0:
                            continue
                        if diff > 0:
                            s += math.sqrt((w_rel[j] * ad) / sum_w_rel)
                        else:
                            s += (-1.0 / theta) * math.sqrt((sum_w_rel * ad) / max(EPS, w_rel[j]))
                    dominance[i, q] = s

            dom_df = pd.DataFrame(dominance, index=alts, columns=alts).round(6)
            Y = dominance.sum(axis=1)
            y_min = float(np.min(Y))
            y_max = float(np.max(Y))
            if abs(y_max - y_min) < EPS:
                Y_star = np.ones_like(Y, dtype=float)
            else:
                Y_star = (Y - y_min) / (y_max - y_min)

            df_phi = pd.DataFrame({
                "Alternative": alts,
                "Υ (comprehensive superiority)": Y,
                "Υ* (normalized superiority)": Y_star,
            })
            df_phi["Rank"] = df_phi["Υ* (normalized superiority)"].rank(ascending=False, method="min").astype(int)
            df_phi = df_phi.sort_values(["Rank", "Υ* (normalized superiority)"], ascending=[True, False]).reset_index(drop=True)
            t1, t2, t3 = st.tabs(["📋 Results", "📈 Plot", "📝 Export"])

            with t1:
                st.markdown("### Aggregated IVFFS Decision Matrix (IVFFDWA)")
                st.dataframe(agg_df, use_container_width=True)
                st.markdown("### Normalized IVFFS Matrix")
                st.dataframe(norm_df, use_container_width=True)
                st.markdown("### Score Matrix τ (Ψ)")
                st.dataframe(tau_df, use_container_width=True)
                st.markdown("### Dominance matrix δ(i,q)")
                st.dataframe(dom_df, use_container_width=True)
                st.markdown("### Final IVFFS-TODIM Ranking")
                st.dataframe(df_phi.round(6), use_container_width=True, hide_index=True)

            with t2:
                st.markdown("### Plot")
                st.bar_chart(df_phi.set_index("Alternative")[["Υ* (normalized superiority)"]])

            with t3:
                st.markdown("### Export")
                make_download_button("⬇️ Download TODIM ranking as CSV", round_df(df_phi, 6), "ivffs_todim_ranking.csv")
                make_download_button(
                    "⬇️ Download TODIM dominance matrix as CSV",
                    dom_df.reset_index().rename(columns={"index": "Alternative"}),
                    "ivffs_todim_dominance_matrix.csv",
                )


# =========================================================
# MAIN APP
# =========================================================
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a Module",
        ["IVFFS-DELPHI", "IVFFS-WINGS", "IVFFS-TODIM"],
    )

    st.sidebar.success("✅ Authenticated")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        logout()

    render_workspace_banner()

    if page == "IVFFS-DELPHI":
        ivffs_delphi_module()
    elif page == "IVFFS-WINGS":
        ivffs_wings_module()
    else:
        ivffs_todim_module()

    render_sidebar_research_profiles()
    render_footer()


if check_password():
    main()
else:
    st.stop()
