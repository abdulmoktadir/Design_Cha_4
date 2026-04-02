import base64
import hmac
import io
import math
import zipfile
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from functools import lru_cache

import graphviz
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Decision Support Research Studio | Advanced Analytics Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# ENHANCED PROFESSIONAL CSS
# ============================================================
CSS = """
<style>
:root {
    --bg: #f8fafc;
    --bg2: #f1f5f9;
    --surface: rgba(255,255,255,0.96);
    --surface-strong: #ffffff;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #475569;
    --primary: #4f46e5;
    --primary-strong: #4338ca;
    --accent: #0d9488;
    --success: #059669;
    --warning: #d97706;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
}
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
.stApp {
    background: linear-gradient(135deg, var(--bg) 0%, var(--bg2) 100%);
    color: var(--text);
}
.block-container { max-width: 1440px; padding-top: 1.5rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #0f172a 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e2e8f0; }
[data-testid="stSidebar"] .stSelectbox label, 
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stRadio label {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] .stNumberInput input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #f8fafc !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.5rem;
}
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #4f46e5, #0d9488);
    border: none;
    color: white;
}
.hero-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(248,250,252,0.98) 100%);
    border: 1px solid var(--border);
    border-radius: 28px;
    padding: 1.75rem 2rem;
    box-shadow: var(--shadow-xl);
    margin-bottom: 1.5rem;
}
.hero-eyebrow {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--primary);
    background: linear-gradient(135deg, #e0e7ff, #ccfbf1);
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
}
.hero-title {
    margin-top: 0.75rem;
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    background-clip: text;
    -webkit-background-clip: text;
    color: transparent;
}
.hero-subtitle {
    margin-top: 0.75rem;
    max-width: 900px;
    font-size: 1rem;
    line-height: 1.6;
    color: var(--muted);
}
.hero-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1.5rem;
}
.info-card {
    background: var(--surface-strong);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1rem 1.25rem;
    box-shadow: var(--shadow-sm);
    transition: transform 0.2s, box-shadow 0.2s;
}
.info-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}
.info-card strong {
    display: block;
    margin-top: 0.5rem;
    font-size: 1.1rem;
    color: var(--text);
}
.info-card small {
    display: block;
    margin-top: 0.25rem;
    color: var(--muted);
    font-size: 0.85rem;
    line-height: 1.5;
}
.module-banner {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow-sm);
}
.module-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #e0e7ff;
    color: var(--primary-strong);
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.module-title {
    margin-top: 0.75rem;
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text);
}
.module-subtitle {
    margin-top: 0.25rem;
    color: var(--muted);
    font-size: 0.9rem;
}
.stButton > button {
    width: 100%;
    border-radius: 14px;
    padding: 0.6rem 1rem;
    font-weight: 600;
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    color: white;
    border: none;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.9;
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
}
div[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 16px;
    background: rgba(255,255,255,0.9);
}
.sidebar-profile-card {
    background: linear-gradient(135deg, rgba(30,27,75,0.95) 0%, rgba(15,23,42,0.95) 100%);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 1.25rem;
    margin: 0.5rem 0 1rem 0;
}
.sidebar-profile-name {
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0.5rem 0 0.2rem 0;
    color: white;
}
.sidebar-profile-role {
    font-size: 0.8rem;
    color: #a5b4fc;
}
.sidebar-profile-institution {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}
.dashboard-stat {
    background: white;
    border-radius: 20px;
    padding: 1rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
}
.dashboard-stat-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--primary);
}
.dashboard-stat-label {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.25rem;
}
.app-footer {
    text-align: center;
    margin-top: 3rem;
    padding: 1.5rem 0;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.8rem;
}
@media (max-width: 768px) {
    .hero-grid { grid-template-columns: 1fr; }
    .hero-title { font-size: 1.8rem; }
}
</style>
"""

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
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
    if expected_password is None:
        st.error("APP_PASSWORD is not configured. Add it in .streamlit/secrets.toml")
        return False

    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; margin: 2rem 0;">
                <div style="background: linear-gradient(135deg, #4f46e5, #0d9488); width: 80px; height: 80px; border-radius: 24px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
                    <span style="font-size: 2.5rem;">🔬</span>
                </div>
                <h1 style="font-size: 2.2rem; font-weight: 800;">Decision Support Studio</h1>
                <p style="color: #475569; max-width: 400px; margin: 0.5rem auto;">Advanced analytics for research professionals</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("login_form"):
                password = st.text_input("Password", type="password", placeholder="Enter application password")
                submitted = st.form_submit_button("Access Workspace", use_container_width=True)
            
            if submitted:
                if hmac.compare_digest(password, str(expected_password)):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid password")
    return False

def get_asset_path(filename: str) -> Path:
    return Path(__file__).parent / "assets" / filename

def get_image_data_uri(image_path: Path) -> Optional[str]:
    if not image_path.exists():
        return None
    suffix = image_path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(suffix, "image/png")
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"

def render_sidebar_profile_card(container, name, role, institution, image_path, brief_text, full_bio=None, extras=None, tag="Researcher"):
    image_uri = get_image_data_uri(Path(image_path))
    image_html = f'<img style="width: 100%; border-radius: 12px;" src="{image_uri}" alt="{escape(name)}">' if image_uri else '<div style="background: #334155; border-radius: 12px; padding: 2rem; text-align: center; color: white;">No image</div>'
    extras_html = ""
    if extras:
        extras_html = '<div style="margin-top: 0.75rem;">' + ''.join(f'<div style="font-size: 0.75rem; color: #94a3b8;">• {escape(item)}</div>' for item in extras) + '</div>'
    container.markdown(
        f"""
        <div class="sidebar-profile-card">
            {image_html}
            <div class="sidebar-profile-name">{escape(name)}</div>
            <div class="sidebar-profile-role">{escape(role)}</div>
            <div class="sidebar-profile-institution">{escape(institution)}</div>
            <div style="font-size: 0.75rem; color: #cbd5e1; line-height: 1.4;">{escape(brief_text)}</div>
            {extras_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if full_bio:
        with container.expander(f"About {name}"):
            st.markdown(f'<div style="color: #e2e8f0; font-size: 0.85rem;">{escape(full_bio)}</div>', unsafe_allow_html=True)

def render_sidebar_research_profiles():
    box = st.sidebar
    box.markdown("---")
    box.markdown('<div style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; margin-bottom: 0.75rem;">👥 Research Team</div>', unsafe_allow_html=True)
    render_sidebar_profile_card(
        box,
        'Prof. J.Z. Ren',
        'Associate Professor',
        'The Hong Kong Polytechnic University',
        get_asset_path('prof_jz_ren.png'),
        'Process systems engineering for energy, environment, and sustainability.',
        full_bio='Dr. Jingzheng Ren leads research in process systems engineering, decision tools, and optimization models for carbon-neutral industrial systems.',
        tag='Lead',
    )
    render_sidebar_profile_card(
        box,
        'Md. Abdul Moktadir',
        'Assistant Professor & PhD Fellow',
        'University of Dhaka / PolyU',
        get_asset_path('abdul_moktadir.png'),
        'Sustainable supply chains, risk management, Industry 4.0, circular economy.',
        extras=['PolyU Presidential PhD Fellow', 'Leather Products Engineering'],
        tag='Co-Researcher',
    )

def render_app_header():
    st.markdown("""
    <div class="hero-card">
        <div class="hero-eyebrow">Integrated Research Workspace</div>
        <div class="hero-title">Stratification · DFS · AHP · QFD · MILP</div>
        <div class="hero-subtitle">A professional-grade decision support studio for multi-criteria analysis, fuzzy synthesis, and portfolio optimization.</div>
        <div class="hero-grid">
            <div class="info-card"><strong>📊 Stratification</strong><small>Network-based probability propagation</small></div>
            <div class="info-card"><strong>👥 Expert Weights</strong><small>Covariance & eigenvector analysis</small></div>
            <div class="info-card"><strong>🧮 DFS-AHP</strong><small>Dual-fuzzy analytic hierarchy</small></div>
            <div class="info-card"><strong>📈 DFS-QFD</strong><small>Quality function deployment</small></div>
            <div class="info-card"><strong>🎯 MILP</strong><small>Portfolio optimization under constraints</small></div>
            <div class="info-card"><strong>📤 Exports</strong><small>Excel, CSV, ZIP downloads</small></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_module_banner(icon: str, title: str, subtitle: str, badge: str = "Module"):
    st.markdown(f"""
    <div class="module-banner">
        <div class="module-badge">{badge}</div>
        <div class="module-title">{icon} {title}</div>
        <div class="module-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    st.markdown("""
    <div class="app-footer">
        <span>© 2026 Decision Support Research Studio</span><br>
        <span style="font-size: 0.7rem;">Developed by Moktadir M.A. & Ren J.Z.</span>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# DASHBOARD PAGE
# ============================================================
def page_dashboard():
    render_module_banner("📊", "Analytics Dashboard", "Overview of all module outputs and key metrics", badge="Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    strat_df = st.session_state.get("strat_results_df")
    expert_df = st.session_state.get("cov_result_df")
    ahp_df = st.session_state.get("ahp_final_fused_weights")
    qfd_df = st.session_state.get("qfd_results")
    milp_df = st.session_state.get("milp_last_results")
    
    with col1:
        status = "✅" if strat_df is not None else "⏳"
        st.markdown(f'<div class="dashboard-stat"><div class="dashboard-stat-value">{status}</div><div class="dashboard-stat-label">Stratification</div></div>', unsafe_allow_html=True)
    with col2:
        status = "✅" if expert_df is not None else "⏳"
        st.markdown(f'<div class="dashboard-stat"><div class="dashboard-stat-value">{status}</div><div class="dashboard-stat-label">Expert Weights</div></div>', unsafe_allow_html=True)
    with col3:
        status = "✅" if ahp_df is not None else "⏳"
        st.markdown(f'<div class="dashboard-stat"><div class="dashboard-stat-value">{status}</div><div class="dashboard-stat-label">DFS-AHP</div></div>', unsafe_allow_html=True)
    with col4:
        status = "✅" if qfd_df is not None else "⏳"
        st.markdown(f'<div class="dashboard-stat"><div class="dashboard-stat-value">{status}</div><div class="dashboard-stat-label">DFS-QFD</div></div>', unsafe_allow_html=True)
    with col5:
        status = "✅" if milp_df is not None else "⏳"
        st.markdown(f'<div class="dashboard-stat"><div class="dashboard-stat-value">{status}</div><div class="dashboard-stat-label">MILP</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    tabs = st.tabs(["📈 Stratification", "👥 Expert Weights", "🧮 DFS-AHP", "📊 DFS-QFD", "🎯 MILP"])
    
    with tabs[0]:
        if strat_df is not None:
            st.dataframe(strat_df.head(10), use_container_width=True)
            fig = px.bar(strat_df, x="Label", y="Percentage (%)", title="Scenario Probabilities", color="Percentage (%)", color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run Stratification Modeler to see results here.")
    
    with tabs[1]:
        if expert_df is not None:
            st.dataframe(expert_df, use_container_width=True)
            fig = px.bar(expert_df, x="Expert", y="Weight", title="Expert Weights", color="Weight", color_continuous_scale="Greens")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run Expert Covariance Model to see results here.")
    
    with tabs[2]:
        if ahp_df is not None:
            st.dataframe(ahp_df, use_container_width=True)
            fig = px.bar(ahp_df, x="Criterion", y="Normalized Weight", title="DFS-AHP Final Weights", color="Normalized Weight", color_continuous_scale="Purples")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run DFS-AHP to see results here.")
    
    with tabs[3]:
        if qfd_df is not None:
            st.dataframe(qfd_df, use_container_width=True)
            fig = px.bar(qfd_df, x="MS", y="RePj (normalized SyP)", title="Mitigation Strategy Priorities (RePj)", color="RePj (normalized SyP)", color_continuous_scale="Oranges")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run DFS-QFD to see results here.")
    
    with tabs[4]:
        if milp_df is not None:
            st.dataframe(milp_df, use_container_width=True)
        else:
            st.info("Run MILP Optimization to see results here.")

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
            raise ValueError("Circular dependency detected.")
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

@st.cache_data(ttl=3600)
def compute_stratification(base_df: pd.DataFrame, inter_df: pd.DataFrame, normalize: bool = True) -> Tuple[pd.DataFrame, float, Dict]:
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
    results = []
    for n in sorted_nodes:
        results.append({
            "Scenario ID": n,
            "Label": labels.get(n, ""),
            "Probability": probs[n],
            "Percentage (%)": probs[n] * 100,
        })
    return pd.DataFrame(results), P_val, {"parents": parents_map, "labels": labels, "sorted_nodes": sorted_nodes}

def page_stratification():
    render_module_banner("🧭", "Stratification Modeler", "Network-based probability propagation for scenario analysis.", badge="Module 1")
    
    st.sidebar.subheader("⚙️ Configuration")
    num_events = st.sidebar.number_input("Base Events", 1, 20, 4, key="strat_num_events")
    precision = st.sidebar.slider("Decimal Precision", 2, 6, 4, key="strat_precision")
    show_labels = st.sidebar.checkbox("Show Labels on Graph", True, key="strat_show_labels")
    normalize = st.sidebar.checkbox("Normalize to 100%", True, key="strat_normalize")
    
    if st.session_state.get("strat_store_key") != num_events:
        st.session_state["strat_store_key"] = num_events
        st.session_state["strat_base_df"] = pd.DataFrame({
            "ID": ["S1"] + [f"S{i}" for i in range(2, num_events + 2)],
            "Label": [f"Context {i}" for i in range(1, num_events + 2)],
            "Value (%)": [25.0 if i < 5 else 0.0 for i in range(num_events + 1)],
        })
        st.session_state["strat_inter_df"] = pd.DataFrame(columns=["ID", "Parents (e.g. S2,S3)", "Label"])
    
    with st.expander("📖 Methodology", expanded=False):
        st.markdown(r"""
        The model calculates the root scaling factor $P$ by solving for the point where the sum of all scenario probabilities equals 1.
        $$\sum_{i=1}^{n} \text{Prob}(S_i) = 1$$
        - **Base Scenarios**: $\text{Prob}(S_{\text{base}}) = (\text{Value}_{\%}/100) \times P$
        - **Interaction Scenarios**: $\text{Prob}(S_{\text{inter}}) = \prod \text{Prob}(S_{\text{parents}})$
        """)
    
    col_left, col_right = st.columns([1, 1.5], gap="large")
    
    with col_left:
        st.subheader("1. Base Events")
        base_df = st.data_editor(st.session_state["strat_base_df"], use_container_width=True, hide_index=True, key="strat_base_editor")
        st.session_state["strat_base_df"] = base_df
        
        st.subheader("2. Interactions")
        inter_df = st.data_editor(st.session_state["strat_inter_df"], num_rows="dynamic", use_container_width=True, hide_index=True, key="strat_inter_editor")
        st.session_state["strat_inter_df"] = inter_df
    
    with col_right:
        st.subheader("3. Results")
        with st.spinner("Computing probabilities..."):
            try:
                results_df, P_val, meta = compute_stratification(base_df, inter_df, normalize)
                st.session_state["strat_results_df"] = results_df
                
                m1, m2 = st.columns(2)
                m1.metric("Solved Factor (P)", f"{P_val:.6f}")
                m2.metric("Total Sum", f"{results_df['Percentage (%)'].sum():.2f}%")
                
                dot = graphviz.Digraph(format="svg")
                dot.attr(rankdir="LR", bgcolor="transparent")
                for n in meta["sorted_nodes"]:
                    display = meta["labels"].get(n, n) if show_labels else n
                    prob = results_df[results_df["Scenario ID"] == n]["Probability"].values[0] if len(results_df[results_df["Scenario ID"] == n]) > 0 else 0
                    dot.node(n, f"{display}\n{prob:.{precision}f}", style="filled", fillcolor="#E0E7FF", shape="box")
                for child, parents in meta["parents"].items():
                    for p in parents:
                        dot.edge(p, child)
                st.graphviz_chart(dot)
                
                st.dataframe(results_df[["Scenario ID", "Label", "Percentage (%)"]].style.format({"Percentage (%)": "{:.2f}%"}), use_container_width=True)
                
                csv = results_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", csv, "stratification_results.csv", "text/csv")
                
                if st.button("Send probabilities to DFS-AHP", type="primary"):
                    st.session_state["strat_to_ahp_df"] = results_df.copy()
                    st.session_state["ahp_s"] = len(results_df)
                    for idx, p in enumerate(results_df["Probability"].tolist()):
                        st.session_state[f"ahp_sp_{idx}"] = p
                    st.success("Sent to Module 3 (DFS-AHP).")
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================
# MODULE 2: EXPERT COVARIANCE MODEL
# ============================================================
@st.cache_data(ttl=3600)
def compute_expert_weights(raw_df: pd.DataFrame) -> Dict:
    def minmax_normalize(df):
        x = df.astype(float).copy()
        col_min = x.min(axis=0)
        col_max = x.max(axis=0)
        denom = (col_max - col_min).replace(0, np.nan)
        normalized = (x - col_min) / denom
        return normalized.fillna(0.0)
    
    normalized = minmax_normalize(raw_df)
    cov = np.cov(normalized.astype(float).to_numpy(), rowvar=False, ddof=0)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    max_idx = int(np.argmax(eigenvalues))
    principal = eigenvectors[:, max_idx]
    sorted_eigen = np.sort(principal)[::-1]
    lambda_vals = raw_df.astype(float).to_numpy() @ sorted_eigen
    lambda_sum = float(np.sum(lambda_vals))
    weights = lambda_vals / lambda_sum if abs(lambda_sum) > 1e-12 else np.ones(len(lambda_vals)) / len(lambda_vals)
    return {
        "normalized": normalized,
        "covariance": pd.DataFrame(cov, index=raw_df.columns, columns=raw_df.columns),
        "eigenvalues": eigenvalues,
        "principal_eigenvector": principal,
        "sorted_eigenvector": sorted_eigen,
        "lambda": lambda_vals,
        "weights": weights,
    }

def page_expert_covariance_model():
    render_module_banner("👥", "Expert Weight Determination", "Covariance-based expert weighting with eigenvector priority.", badge="Module 2")
    
    st.sidebar.subheader("⚙️ Setup")
    n_experts = st.sidebar.number_input("Experts", 2, 100, 4, key="cov_n_experts")
    n_dims = st.sidebar.number_input("Dimensions", 2, 50, 10, key="cov_n_dims")
    
    c1, c2 = st.columns(2)
    with c1:
        expert_names = [st.text_input(f"Expert {i+1}", f"Ex{i+1}", key=f"cov_expert_{i}") for i in range(n_experts)]
    with c2:
        dim_names = [st.text_input(f"Dimension {j+1}", f"X{j+1}", key=f"cov_dim_{j}") for j in range(n_dims)]
    
    if st.session_state.get("cov_store_key") != (n_experts, n_dims):
        st.session_state["cov_store_key"] = (n_experts, n_dims)
        st.session_state["cov_input_df"] = pd.DataFrame(np.zeros((n_experts, n_dims)), index=expert_names, columns=dim_names)
    
    input_df = st.data_editor(st.session_state["cov_input_df"], use_container_width=True, height=300, key="cov_input_editor")
    input_df.index = expert_names
    input_df.columns = dim_names
    st.session_state["cov_input_df"] = input_df
    
    if st.button("Compute Expert Weights", type="primary", key="cov_run"):
        try:
            raw_df = input_df.astype(float)
            with st.spinner("Computing covariance and eigenvector weights..."):
                res = compute_expert_weights(raw_df)
                st.session_state["cov_normalized_df"] = res["normalized"]
                st.session_state["cov_cov_df"] = res["covariance"]
                st.session_state["cov_result_df"] = pd.DataFrame({"Expert": expert_names, "λ": res["lambda"], "Weight": res["weights"]})
                
                st.subheader("Normalized Data")
                st.dataframe(res["normalized"].round(6), use_container_width=True)
                st.subheader("Covariance Matrix")
                st.dataframe(res["covariance"].round(4), use_container_width=True)
                st.subheader("Expert Weights")
                st.dataframe(st.session_state["cov_result_df"].round(6), use_container_width=True)
                fig = px.bar(st.session_state["cov_result_df"], x="Expert", y="Weight", title="Expert Weights", color="Weight", color_continuous_scale="Teal")
                st.plotly_chart(fig, use_container_width=True)
                
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as writer:
                    raw_df.to_excel(writer, sheet_name="Raw_Data")
                    res["normalized"].to_excel(writer, sheet_name="Normalized")
                    res["covariance"].to_excel(writer, sheet_name="Covariance")
                    st.session_state["cov_result_df"].to_excel(writer, index=False, sheet_name="Weights")
                st.download_button("📥 Download Excel", out.getvalue(), "expert_weights.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================
# MODULE 3: DFS-AHP (from original, with minimal changes)
# ============================================================
class DFSAHP:
    def __init__(self):
        self.optimistic_uv = {
            "EEI": (0.50, 0.50), "SMI": (0.55, 0.45), "WMI": (0.60, 0.40),
            "MI": (0.65, 0.35), "StMI": (0.70, 0.30), "VSI": (0.75, 0.25),
            "AMI": (0.80, 0.20), "PMI": (0.85, 0.15), "EMI": (0.90, 0.10),
        }
        self.pessimistic_uv = {
            "EEU": (0.50, 0.50), "SMU": (0.45, 0.55), "WMU": (0.40, 0.60),
            "MU": (0.35, 0.65), "StMU": (0.30, 0.70), "VSU": (0.25, 0.75),
            "AMU": (0.20, 0.80), "PMU": (0.15, 0.85), "EMU": (0.10, 0.90),
        }
        self.pessimistic_uv["EEI"] = (0.50, 0.50)
        self.k = 0.9
        self.intensity = {
            "EEI": 1, "EEU": 1, "SMI": 2, "SMU": 2, "WMI": 3, "WMU": 3,
            "MI": 4, "MU": 4, "StMI": 5, "StMU": 5, "VSI": 6, "VSU": 6,
            "AMI": 7, "AMU": 7, "PMI": 8, "PMU": 8, "EMI": 9, "EMU": 9,
        }
        self.RI = {1:0.00,2:0.00,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49,11:1.51,12:1.48,13:1.56,14:1.57,15:1.59}

    def term_to_uv(self, term: str) -> Tuple[float, float]:
        if term is None: return (0.5,0.5)
        t = str(term).strip()
        if t in self.optimistic_uv: return self.optimistic_uv[t]
        if t in self.pessimistic_uv: return self.pessimistic_uv[t]
        return (0.5,0.5)

    def build_abcd_matrices(self, combined_df: pd.DataFrame, criteria: List[str]):
        n = len(criteria)
        a = np.zeros((n,n)); b = np.zeros((n,n)); c = np.zeros((n,n)); d = np.zeros((n,n))
        invalid_terms = set()
        for i in range(n):
            for j in range(n):
                o_col = f"{criteria[j]} (O)"
                p_col = f"{criteria[j]} (P)"
                o_term = combined_df.iloc[i][o_col]
                p_term = combined_df.iloc[i][p_col]
                uo, vo = self.term_to_uv(o_term)
                up, vp = self.term_to_uv(p_term)
                a[i,j], b[i,j] = uo, vo
                c[i,j], d[i,j] = up, vp
                for raw in [o_term, p_term]:
                    if raw and str(raw).strip() and (str(raw).strip() not in self.optimistic_uv) and (str(raw).strip() not in self.pessimistic_uv):
                        invalid_terms.add(str(raw).strip())
        return a,b,c,d,invalid_terms

    def aggregate_a_row(self, row_a: np.ndarray) -> float:
        n = len(row_a); row_a = np.clip(row_a, 1e-12, 1.0)
        return float(np.prod(row_a ** (1.0/n)))

    def aggregate_b_row(self, row_b: np.ndarray) -> float:
        n = len(row_b); row_b = np.clip(row_b, 0.0, 1.0-1e-12)
        return float(1.0 - np.prod((1.0 - row_b) ** (1.0/n)))

    def aggregate_c_row_excel(self, row_c: np.ndarray) -> float:
        n = len(row_c); row_c = np.clip(row_c, 1e-12, 1.0)
        num = float(np.prod(row_c))
        denom = (1.0/n) * float(np.sum((row_c ** (n-1)) * (1.0 - row_c))) + num
        return 0.0 if abs(denom) < 1e-12 else float(num / denom)

    def aggregate_d_row(self, row_d: np.ndarray) -> float:
        return float(np.mean(row_d))

    def ci_dfs_value(self, a,b,c,d) -> float:
        val = 1.0 - np.sqrt(((a-d)**2 + (b-c)**2 + (1.0-a-b)**2 + (1.0-c-d)**2)/2.0)
        return float(max(min(val,1.0),0.0))

    def si_value(self, a,b,c,d, ci_dfs) -> float:
        return float((a+b-c+d) * ci_dfs / (2.0 * self.k))

    def _is_unimportant(self, term: str) -> bool:
        t = (term or "").strip()
        return t.endswith("U") or t == "EEU"

    def term_to_ratio(self, term: str) -> float:
        if term is None: return 1.0
        t = str(term).strip()
        if t == "": return 1.0
        k = self.intensity.get(t, None)
        if k is None: return 1.0
        return (1.0 / float(k)) if self._is_unimportant(t) else float(k)

    def build_cr_matrix_excel_rule(self, combined_df: pd.DataFrame, criteria: List[str]) -> np.ndarray:
        n = len(criteria)
        A = np.ones((n,n))
        for i in range(n):
            for j in range(i+1, n):
                o_term = combined_df.iloc[i][f"{criteria[j]} (O)"]
                p_term = combined_df.iloc[i][f"{criteria[j]} (P)"]
                aij = self.term_to_ratio(o_term)
                aji = self.term_to_ratio(p_term)
                if aij <= 0: aij = 1.0
                if aji <= 0: aji = 1.0
                A[i,j] = aij; A[j,i] = aji
        return A

    def ahp_weights_geometric_mean(self, A: np.ndarray) -> np.ndarray:
        n = A.shape[0]
        gm = np.prod(A, axis=1) ** (1.0/n)
        s = gm.sum()
        return gm / s if s > 0 else np.ones(n)/n

    def cr_gm_method(self, A: np.ndarray) -> Dict:
        n = A.shape[0]
        if n <= 2:
            return {"lambda_max":0.0, "CI_AHP":0.0, "RI":self.RI.get(n,0.0), "CR":0.0, "weights":np.ones(n)/n}
        w = self.ahp_weights_geometric_mean(A)
        Aw = A.dot(w)
        lam_vec = Aw / np.clip(w, 1e-12, None)
        lambda_max = float(np.mean(lam_vec))
        CI = float((lambda_max - n) / (n - 1))
        RI = float(self.RI.get(n,0.0))
        CR = float(CI / RI) if RI > 0 else 0.0
        return {"lambda_max":lambda_max, "CI_AHP":CI, "RI":RI, "CR":CR, "weights":w}

    def ahp_weights_eigen(self, A: np.ndarray) -> Tuple[float, np.ndarray]:
        eigvals, eigvecs = np.linalg.eig(A)
        idx = int(np.argmax(np.real(eigvals)))
        lam = float(np.real(eigvals[idx]))
        v = np.real(eigvecs[:, idx]).astype(float)
        if np.all(v <= 0): v = -v
        v = np.abs(v)
        s = v.sum()
        w = v / s if s > 0 else np.ones(A.shape[0])/A.shape[0]
        return lam, w

    def cr_eigen_method(self, A: np.ndarray) -> Dict:
        n = A.shape[0]
        if n <= 2:
            return {"lambda_max":0.0, "CI_AHP":0.0, "RI":self.RI.get(n,0.0), "CR":0.0, "weights":np.ones(n)/n}
        lambda_max, w = self.ahp_weights_eigen(A)
        CI = float((lambda_max - n) / (n - 1))
        RI = float(self.RI.get(n,0.0))
        CR = float(CI / RI) if RI > 0 else 0.0
        return {"lambda_max":lambda_max, "CI_AHP":CI, "RI":RI, "CR":CR, "weights":w}

    def compute_expert(self, combined_df: pd.DataFrame, criteria: List[str]):
        a_mat, b_mat, c_mat, d_mat, invalid_terms = self.build_abcd_matrices(combined_df, criteria)
        n = len(criteria)
        rows = []
        for i in range(n):
            a_i = self.aggregate_a_row(a_mat[i,:])
            b_i = self.aggregate_b_row(b_mat[i,:])
            c_i = self.aggregate_c_row_excel(c_mat[i,:])
            d_i = self.aggregate_d_row(d_mat[i,:])
            ci_dfs = self.ci_dfs_value(a_i,b_i,c_i,d_i)
            si = self.si_value(a_i,b_i,c_i,d_i, ci_dfs)
            rows.append({"Criterion":criteria[i], "a (μO)":a_i, "b (νO)":b_i, "c (μP)":c_i, "d (νP)":d_i, "CI (DFS)":ci_dfs, "SI":si})
        df = pd.DataFrame(rows)
        s_sum = float(df["SI"].sum())
        df["Normalized Weight"] = df["SI"] / s_sum if s_sum > 0 else 0.0
        df = df.sort_values("Normalized Weight", ascending=False).reset_index(drop=True)
        df["Rank"] = np.arange(1,len(df)+1)
        A = self.build_cr_matrix_excel_rule(combined_df, criteria)
        cr_gm = self.cr_gm_method(A)
        cr_ev = self.cr_eigen_method(A)
        return df, invalid_terms, {"A":A, "GM":cr_gm, "EIGEN":cr_ev}

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
    for d in [2,3,4,5,6,7,8,9]:
        if abs(x - 1.0/d) < 1e-3: return f"1/{d}"
    for k in [1,2,3,4,5,6,7,8,9]:
        if abs(x - k) < 1e-3: return f"{k:.2f}"
    return f"{x:.2f}"

@dataclass
class DFS:
    mu_O: float
    nu_O: float
    mu_P: float
    nu_P: float
    def clip(self): return DFS(np.clip(self.mu_O,0,1), np.clip(self.nu_O,0,1), np.clip(self.mu_P,0,1), np.clip(self.nu_P,0,1))

def dfs_dwgm(values: List[DFS], weights: List[float]) -> DFS:
    mu_O = 1.0; nu_O = 1.0; mu_P = 1.0; nu_P = 1.0
    for v,w in zip(values,weights):
        mu_O *= (max(v.mu_O,1e-12)**w)
        nu_O *= (max(v.nu_O,1e-12)**w)
        mu_P *= (max(v.mu_P,1e-12)**w)
        nu_P *= (max(v.nu_P,1e-12)**w)
    return DFS(mu_O, nu_O, mu_P, nu_P).clip()

def dfs_multiply(weight_dfs: DFS, rel_dfs: DFS) -> DFS:
    w = weight_dfs; r = rel_dfs
    a = w.mu_O * r.mu_O
    b = w.nu_O + r.nu_O - (w.nu_O * r.nu_O)
    denom_c = (w.mu_P + r.mu_P - (w.mu_P * r.mu_P))
    c = (w.mu_P * r.mu_P) / denom_c if abs(denom_c) > 1e-12 else 0.0
    denom_d = (1.0 - (w.nu_P * r.nu_P))
    d = (w.nu_P + r.nu_P - 2.0 * w.nu_P * r.nu_P) / denom_d if abs(denom_d) > 1e-12 else 0.0
    return DFS(a,b,c,d).clip()

def score_defuzz_from_weighted(weighted: DFS) -> float:
    a,b,c,d = weighted.mu_O, weighted.nu_O, weighted.mu_P, weighted.nu_P
    term1 = math.sqrt(max((a+d)/8.0,0.0))
    term2 = math.sqrt(max((b+c)/8.0,0.0))
    return 0.5 + (term1 - term2)

def default_scale_uv() -> Dict[str, Tuple[float,float]]:
    return {
        "EEI":(0.50,0.50),"SMI":(0.55,0.45),"WMI":(0.60,0.40),"MI":(0.65,0.35),
        "StMI":(0.70,0.30),"VSI":(0.75,0.25),"AMI":(0.80,0.20),"PMI":(0.85,0.15),"EMI":(0.90,0.10),
        "EEU":(0.50,0.50),"SMU":(0.45,0.55),"WMU":(0.40,0.60),"MU":(0.35,0.65),
        "StMU":(0.30,0.70),"VSU":(0.25,0.75),"AMU":(0.20,0.80),"PMU":(0.15,0.85),"EMU":(0.10,0.90),
    }

def term_to_dfs(term_O: str, term_P: str, scale: Dict[str, Tuple[float,float]]) -> DFS:
    muO, nuO = scale.get(term_O, scale["EEI"])
    muP, nuP = scale.get(term_P, scale["EEU"])
    return DFS(muO, nuO, muP, nuP).clip()

def normalize_weights(ws: List[float]) -> List[float]:
    if len(ws)==0: return []
    s = sum(ws)
    if s <= 0: return [1.0/len(ws)]*len(ws)
    return [w/s for w in ws]

def safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x,str) and x.strip()==""): return default
        return float(x)
    except: return default

def init_relationship_df(rc_names: List[str], ms_names: List[str]) -> pd.DataFrame:
    cols = ["RC"]
    for ms in ms_names:
        cols += [f"{ms}_O", f"{ms}_P"]
    df = pd.DataFrame({"RC": rc_names})
    for c in cols[1:]:
        df[c] = "EEI" if c.endswith("_O") else "EEU"
    return df[cols]

def init_rc_weight_df(rc_names: List[str]) -> pd.DataFrame:
    return pd.DataFrame({"RC": rc_names, "mu_O":[0.50]*len(rc_names), "nu_O":[0.50]*len(rc_names), "mu_P":[0.50]*len(rc_names), "nu_P":[0.50]*len(rc_names)})

def init_cost_df(ms_names: List[str]) -> pd.DataFrame:
    return pd.DataFrame({"MS": ms_names, "ICj": [1.0]*len(ms_names)})

def dfs_table_to_map(df: pd.DataFrame, criteria: List[str]) -> Dict[str, DFS]:
    base = df.set_index("Criterion")
    out = {}
    for c in criteria:
        row = base.loc[c]
        out[c] = DFS(float(row["a (μO)"]), float(row["b (νO)"]), float(row["c (μP)"]), float(row["d (νP)"])).clip()
    return out

def build_weight_table_from_dfs_map(criteria: List[str], dfs_map: Dict[str, DFS], engine: DFSAHP) -> pd.DataFrame:
    rows = []
    for c in criteria:
        v = dfs_map[c].clip()
        ci_dfs = engine.ci_dfs_value(v.mu_O, v.nu_O, v.mu_P, v.nu_P)
        si = engine.si_value(v.mu_O, v.nu_O, v.mu_P, v.nu_P, ci_dfs)
        rows.append({"Criterion":c, "a (μO)":v.mu_O, "b (νO)":v.nu_O, "c (μP)":v.mu_P, "d (νP)":v.nu_P, "CI (DFS)":ci_dfs, "SI":si})
    df = pd.DataFrame(rows)
    s_sum = float(df["SI"].sum())
    df["Normalized Weight"] = df["SI"] / s_sum if s_sum > 0 else 0.0
    df = df.sort_values("Normalized Weight", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1,len(df)+1)
    return df

def weighted_componentwise_dfs_aggregate(values: List[DFS], weights: List[float]) -> DFS:
    m = len(values)
    W = np.array(normalize_weights([safe_float(w,1.0) for w in weights]), dtype=float)
    a_vec = np.clip(np.array([v.mu_O for v in values]), 1e-12, 1.0)
    b_vec = np.clip(np.array([v.nu_O for v in values]), 0.0, 1.0-1e-12)
    c_vec = np.clip(np.array([v.mu_P for v in values]), 1e-12, 1.0)
    d_vec = np.clip(np.array([v.nu_P for v in values]), 0.0, 1.0)
    a = float(np.prod(a_vec ** W))
    b = float(1.0 - np.prod((1.0 - b_vec) ** W))
    c_num = float(np.prod(c_vec))
    c_aux = float(np.sum((c_vec ** (m-1)) * W * (1.0 - c_vec)))
    c_den = c_aux + c_num
    c = 0.0 if abs(c_den) < 1e-12 else float(c_num / c_den)
    d_num = float(np.sum(d_vec * W))
    d_den = 1.0 + float(np.sum((d_vec * W) - (d_vec / m)))
    d = 0.0 if abs(d_den) < 1e-12 else float(d_num / d_den)
    return DFS(a,b,c,d).clip()

def aggregate_experts_decomposed_dfs(expert_tables: List[pd.DataFrame], expert_weights: List[float], criteria: List[str], engine: DFSAHP) -> pd.DataFrame:
    expert_maps = [dfs_table_to_map(t, criteria) for t in expert_tables]
    agg_map = {}
    for c in criteria:
        vals = [expert_maps[e][c] for e in range(len(expert_maps))]
        agg_map[c] = weighted_componentwise_dfs_aggregate(vals, expert_weights)
    return build_weight_table_from_dfs_map(criteria, agg_map, engine)

def aggregate_scenarios_decomposed_dfs(scenario_tables: List[pd.DataFrame], scenario_probs: List[float], criteria: List[str], engine: DFSAHP) -> Tuple[pd.DataFrame, List[float]]:
    probs = normalize_weights([safe_float(p,1.0) for p in scenario_probs])
    scenario_maps = [dfs_table_to_map(t, criteria) for t in scenario_tables]
    final_map = {}
    for c in criteria:
        vals = [scenario_maps[s][c] for s in range(len(scenario_maps))]
        final_map[c] = weighted_componentwise_dfs_aggregate(vals, probs)
    final_df = build_weight_table_from_dfs_map(criteria, final_map, engine)
    return final_df, probs

def decomposed_weight_table_to_rc_df(df: pd.DataFrame, criteria_order: List[str]) -> pd.DataFrame:
    x = df.copy().set_index("Criterion").loc[criteria_order].reset_index()
    return pd.DataFrame({
        "RC": x["Criterion"].astype(str).tolist(),
        "mu_O": x["a (μO)"].astype(float).tolist(),
        "nu_O": x["b (νO)"].astype(float).tolist(),
        "mu_P": x["c (μP)"].astype(float).tolist(),
        "nu_P": x["d (νP)"].astype(float).tolist(),
    })

def _to_float(x, default=0.0):
    try:
        if x is None: return default
        s = str(x).strip()
        if s == "" or s.lower() in {"nan","none"}: return default
        if "/" in s:
            a,b = s.split("/")
            return float(a)/float(b)
        return float(s)
    except: return default

def expected_from_omp(O, ML, P):
    return (O + P + 4.0*ML)/6.0

def make_square_df(names, fill=0.0):
    df = pd.DataFrame(fill, index=names, columns=names)
    np.fill_diagonal(df.values, 0.0)
    return df

def subset_cost_time(x, cost, time, rho, zeta):
    x = x.astype(int)
    base_cost = float(np.dot(cost, x))
    base_time = float(np.dot(time, x))
    n = len(x)
    sav_cost = 0.0; sav_time = 0.0
    for i in range(n):
        if x[i]==0: continue
        for j in range(i+1, n):
            if x[j]==0: continue
            sav_cost += float(rho[i,j])
            sav_time += float(zeta[i,j])
    total_cost = base_cost - sav_cost
    total_time = base_time - sav_time
    return total_cost, total_time, base_cost, base_time, sav_cost, sav_time

def solve_by_enumeration(rep, cost, time, rho, zeta, budget, time_limit):
    n = len(rep)
    best = None
    for mask in range(1<<n):
        x = np.array([(mask>>k)&1 for k in range(n)], dtype=int)
        total_cost, total_time, base_cost, base_time, sav_cost, sav_time = subset_cost_time(x, cost, time, rho, zeta)
        if total_cost <= budget+1e-9 and total_time <= time_limit+1e-9:
            obj = float(np.dot(rep, x))
            if (best is None) or (obj > best["obj"]+1e-12):
                best = {"x":x, "obj":obj, "total_cost":total_cost, "total_time":total_time, "base_cost":base_cost, "base_time":base_time, "sav_cost":sav_cost, "sav_time":sav_time}
    return best

def compute_expected_vector(df_omp, col_O="O", col_ML="ML", col_P="P"):
    O = df_omp[col_O].map(lambda v: _to_float(v,0.0)).to_numpy(dtype=float)
    ML = df_omp[col_ML].map(lambda v: _to_float(v,0.0)).to_numpy(dtype=float)
    P = df_omp[col_P].map(lambda v: _to_float(v,0.0)).to_numpy(dtype=float)
    return expected_from_omp(O, ML, P)

def compute_expected_matrix(df_O, df_ML, df_P):
    n = df_O.shape[0]
    E = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(n):
            O = _to_float(df_O.iloc[i,j],0.0)
            ML = _to_float(df_ML.iloc[i,j],0.0)
            P = _to_float(df_P.iloc[i,j],0.0)
            E[i,j] = expected_from_omp(O, ML, P)
    np.fill_diagonal(E, 0.0)
    out = np.zeros_like(E)
    for i in range(n):
        for j in range(i+1, n):
            out[i,j] = E[i,j]
    return out

def page_dfs_ahp():
    render_module_banner("🧮", "DFS-AHP Scenario → Expert → Scenario Fusion", "Integrate scenario probabilities, expert judgments, and decomposed dual-fuzzy weights.", badge="Module 3")
    engine = DFSAHP()

    strat_sent_df = st.session_state.get("strat_to_ahp_df")
    if isinstance(strat_sent_df, pd.DataFrame) and not strat_sent_df.empty:
        st.info("Stratification probabilities loaded from Module 1.")
        st.dataframe(strat_sent_df[["Scenario ID", "Label", "Probability"]], use_container_width=True, hide_index=True)
        loaded_signature = tuple((str(row["Scenario ID"]), float(row["Probability"])) for _, row in strat_sent_df.iterrows())
        if st.session_state.get("ahp_loaded_signature") != loaded_signature:
            st.session_state["ahp_s"] = int(len(strat_sent_df))
            for key in list(st.session_state.keys()):
                if key.startswith("ahp_sp_"): del st.session_state[key]
            for idx, p in enumerate(strat_sent_df["Probability"].tolist()):
                st.session_state[f"ahp_sp_{idx}"] = float(p)
            st.session_state["ahp_loaded_signature"] = loaded_signature

    st.sidebar.subheader("DFS-AHP setup")
    num_criteria = st.sidebar.number_input("Number of Criteria", 2, 15, 9, key="ahp_n")
    num_scenarios = st.sidebar.number_input("Number of Scenarios", 1, 10, 1, key="ahp_s")
    num_experts = st.sidebar.number_input("Number of Experts per Scenario", 1, 8, 1, key="ahp_e")
    cr_threshold = st.sidebar.number_input("CR threshold", 0.0, 1.0, 0.10, 0.01, key="ahp_thr")
    cr_method = st.sidebar.radio("CR method:", ["Eigenvalue method (CR′)", "Geometric mean method"], index=0, key="ahp_cr_method")
    use_key = "EIGEN" if "Eigenvalue" in cr_method else "GM"

    st.sidebar.divider()
    st.sidebar.subheader("Scenario probabilities")
    scen_probs_raw = []
    loaded_probs = []; loaded_titles = []
    if isinstance(strat_sent_df, pd.DataFrame) and not strat_sent_df.empty:
        loaded_probs = strat_sent_df["Probability"].astype(float).tolist()
        loaded_titles = [f'{row["Scenario ID"]} - {row["Label"]}' for _, row in strat_sent_df.iterrows()]
    for s in range(int(num_scenarios)):
        label = loaded_titles[s] if s < len(loaded_titles) else f"Scenario {s+1}"
        default_value = float(loaded_probs[s]) if s < len(loaded_probs) else 1.0
        scen_probs_raw.append(st.sidebar.number_input(f"{label} probability", 0.0, value=default_value, step=0.1, key=f"ahp_sp_{s}"))
    scen_probs = normalize_weights(scen_probs_raw)
    st.sidebar.caption("Normalized: " + ", ".join([f"S{s+1}={p:.4f}" for s,p in enumerate(scen_probs)]))

    st.subheader("Criteria names")
    criteria = []
    cols = st.columns(3)
    for i in range(int(num_criteria)):
        with cols[i % 3]:
            criteria.append(st.text_input(f"Criterion {i+1}", value=f"KPI{i+1}", key=f"ahp_crit_{i}"))

    store_key = f"ahp|{num_criteria}|{num_scenarios}|{num_experts}|" + "|".join(criteria)
    if st.session_state.get("ahp_store_key") != store_key:
        st.session_state["ahp_store_key"] = store_key
        st.session_state["ahp_mats"] = {f"scenario_{s}_expert_{e}": make_blank_ahp_matrix(criteria) for s in range(int(num_scenarios)) for e in range(int(num_experts))}

    scenario_tab_titles = []
    for s in range(int(num_scenarios)):
        if s < len(loaded_titles): scenario_tab_titles.append(loaded_titles[s])
        else: scenario_tab_titles.append(f"Scenario {s+1}")
    scen_tabs = st.tabs(scenario_tab_titles)

    for s in range(int(num_scenarios)):
        with scen_tabs[s]:
            st.caption(f"Scenario probability = {scen_probs[s]:.4f}")
            exp_tabs = st.tabs([f"Expert {e+1}" for e in range(int(num_experts))])
            for e in range(int(num_experts)):
                with exp_tabs[e]:
                    mat_key = f"scenario_{s}_expert_{e}"
                    df = st.data_editor(st.session_state["ahp_mats"][mat_key], height=420, use_container_width=True, key=f"ahp_ed_s{s}_e{e}")
                    for i, c in enumerate(criteria):
                        df.at[criteria[i], f"{c} (O)"] = "EEI"
                        df.at[criteria[i], f"{c} (P)"] = "EEI"
                    st.session_state["ahp_mats"][mat_key] = df

    st.subheader("Expert weights")
    ew = []
    wcols = st.columns(min(int(num_experts),4))
    for e in range(int(num_experts)):
        with wcols[e % len(wcols)]:
            ew.append(st.number_input(f"Expert {e+1} weight", 0.0, value=1.0, step=0.1, key=f"ahp_w_{e}"))
    ew_norm = normalize_weights(ew)
    st.caption("Normalized expert weights: " + ", ".join([f"E{i+1}={w:.4f}" for i,w in enumerate(ew_norm)]))

    if st.button("Run DFS-AHP", type="primary", key="ahp_run"):
        all_invalid = set()
        scenario_expert_results = []
        scenario_expert_cr = []
        scenario_aggregated_results = []
        for s in range(int(num_scenarios)):
            exp_res_this_s = []; exp_cr_this_s = []
            for e in range(int(num_experts)):
                mat_key = f"scenario_{s}_expert_{e}"
                res, invalid_terms, cr_pack = engine.compute_expert(st.session_state["ahp_mats"][mat_key], criteria)
                exp_res_this_s.append(res); exp_cr_this_s.append(cr_pack)
                all_invalid.update(invalid_terms)
            scenario_expert_results.append(exp_res_this_s)
            scenario_expert_cr.append(exp_cr_this_s)
            agg_s = aggregate_experts_decomposed_dfs(exp_res_this_s, ew_norm, criteria, engine)
            scenario_aggregated_results.append(agg_s)
        final_fused_df, scen_probs_norm = aggregate_scenarios_decomposed_dfs(scenario_aggregated_results, scen_probs, criteria, engine)

        st.session_state["ahp_criteria"] = criteria
        st.session_state["ahp_scenario_expert_results"] = scenario_expert_results
        st.session_state["ahp_scenario_expert_cr"] = scenario_expert_cr
        st.session_state["ahp_scenario_aggregated_results"] = scenario_aggregated_results
        st.session_state["ahp_final_fused_weights"] = final_fused_df
        st.session_state["ahp_scenario_probs_norm"] = scen_probs_norm
        st.session_state["ahp_final_weights"] = final_fused_df.copy()

        if all_invalid: st.warning("Unknown terms: " + ", ".join(sorted(all_invalid)))

        st.subheader("Per-scenario outputs")
        for s in range(int(num_scenarios)):
            st.markdown(f"## {scenario_tab_titles[s]}")
            st.caption(f"Probability = {scen_probs_norm[s]:.6f}")
            for e in range(int(num_experts)):
                with st.expander(f"Expert {e+1}", expanded=(s==0 and e==0)):
                    A = scenario_expert_cr[s][e]["A"]
                    A_df = pd.DataFrame(A, index=criteria, columns=criteria)
                    A_disp = A_df.copy()
                    for i in range(len(criteria)):
                        for j in range(len(criteria)):
                            A_disp.iloc[i,j] = fmt_ratio(float(A_df.iloc[i,j]))
                    st.markdown("**CR matrix**")
                    st.dataframe(A_disp, use_container_width=True)
                    gm = scenario_expert_cr[s][e]["GM"]
                    ev = scenario_expert_cr[s][e]["EIGEN"]
                    show_df = pd.DataFrame([
                        {"Method":"Geometric mean","lambda_max":gm["lambda_max"],"CI":gm["CI_AHP"],"RI":gm["RI"],"CR":gm["CR"],"Pass":"✅" if gm["CR"]<=cr_threshold else "❌"},
                        {"Method":"Eigenvalue","lambda_max":ev["lambda_max"],"CI":ev["CI_AHP"],"RI":ev["RI"],"CR′":ev["CR"],"Pass":"✅" if ev["CR"]<=cr_threshold else "❌"},
                    ])
                    st.dataframe(show_df.round(6), use_container_width=True)
                    st.markdown(f"**AHP weights ({cr_method})**")
                    w = scenario_expert_cr[s][e][use_key]["weights"]
                    st.dataframe(pd.DataFrame({"Criterion":criteria, "AHP Weight":np.round(w,6)}), use_container_width=True)
                    st.markdown("**DFS decomposed results**")
                    show = scenario_expert_results[s][e].copy()
                    for col in ["a (μO)","b (νO)","c (μP)","d (νP)","CI (DFS)","SI","Normalized Weight"]:
                        show[col] = show[col].astype(float).round(6)
                    st.dataframe(show, use_container_width=True)
                    fig = px.bar(show, x="Criterion", y="Normalized Weight", title=f"Expert {e+1} Weights", color="Normalized Weight", color_continuous_scale="Viridis")
                    st.plotly_chart(fig, use_container_width=True)
            st.markdown("### Scenario aggregate")
            show_agg = scenario_aggregated_results[s].copy()
            for col in ["a (μO)","b (νO)","c (μP)","d (νP)","CI (DFS)","SI","Normalized Weight"]:
                show_agg[col] = show_agg[col].astype(float).round(6)
            st.dataframe(show_agg, use_container_width=True)
            fig = px.bar(show_agg, x="Criterion", y="Normalized Weight", title=f"Aggregated Weights - {scenario_tab_titles[s]}", color="Normalized Weight", color_continuous_scale="Plasma")
            st.plotly_chart(fig, use_container_width=True)
            st.divider()

        st.subheader("Final DFS fusion across scenarios")
        final_show = final_fused_df.copy()
        final_show["Final Weight"] = final_show["Normalized Weight"]
        for col in ["a (μO)","b (νO)","c (μP)","d (νP)","CI (DFS)","SI","Normalized Weight","Final Weight"]:
            if col in final_show.columns:
                final_show[col] = final_show[col].astype(float).round(6)
        st.dataframe(final_show, use_container_width=True)
        fig = px.bar(final_fused_df, x="Criterion", y="Normalized Weight", title="Final Fused Weights", color="Normalized Weight", color_continuous_scale="Turbo")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Send RC decomposed weights to DFS-QFD")
    if "ahp_final_fused_weights" not in st.session_state:
        st.info("Run DFS-AHP first.")
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
    pick = st.selectbox("Choose which decomposed DFS values to send", options=send_options, index=0, key="ahp_send_pick")
    if st.button("Send to DFS-QFD RC weights", key="ahp_send_btn"):
        if pick == "Final fused scenario DFS":
            src_df = st.session_state["ahp_final_fused_weights"].copy()
        elif "aggregated DFS" in pick:
            s_idx = int(pick.split()[1]) - 1
            src_df = st.session_state["ahp_scenario_aggregated_results"][s_idx].copy()
        else:
            parts = pick.split()
            s_idx = int(parts[1]) - 1
            e_idx = int(parts[-1]) - 1
            src_df = st.session_state["ahp_scenario_expert_results"][s_idx][e_idx].copy()
        rc_weight_df = decomposed_weight_table_to_rc_df(src_df, criteria0)
        st.session_state["qfd_rc_weight_df"] = rc_weight_df
        st.success("Sent to Module 4 (DFS-QFD).")

def page_dfs_qfd():
    render_module_banner("📊", "DFS-QFD Analysis", "Map requirement criteria to mitigation strategies and compute ReP prioritization.", badge="Module 4")
    st.sidebar.subheader("DFS-QFD setup")
    n_rc = int(st.sidebar.number_input("Number of RCs", 2, 50, 9, key="qfd_nrc"))
    n_ms = int(st.sidebar.number_input("Number of MSs", 2, 50, 9, key="qfd_nms"))
    n_exp = int(st.sidebar.number_input("Number of Experts", 1, 10, 1, key="qfd_nexp"))
    st.sidebar.divider()
    st.sidebar.subheader("Expert weights")
    exp_ws = []
    for e in range(n_exp):
        exp_ws.append(float(st.sidebar.number_input(f"Expert {e+1} weight", 0.0, value=1.0, step=0.1, key=f"qfd_w_{e}")))
    exp_ws = normalize_weights(exp_ws)
    st.sidebar.caption("Normalized: " + ", ".join([f"{w:.3f}" for w in exp_ws]))

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
        ms_names = [st.text_input(f"MS {j+1}", value=f"MS{j+1}", key=f"qfd_ms_name_{j}") for j in range(n_ms)]

    st.subheader("DFS linguistic scale")
    scale = default_scale_uv()
    scale_df = pd.DataFrame([{"Term":k, "u (μ)":v[0], "v (ν)":v[1]} for k,v in scale.items()]).sort_values("Term")
    st.dataframe(scale_df, use_container_width=True, height=280)

    st.subheader("RC weights (DFS decomposed)")
    if "qfd_rc_weight_df" not in st.session_state:
        st.session_state["qfd_rc_weight_df"] = init_rc_weight_df(rc_names)
    df0 = st.session_state["qfd_rc_weight_df"]
    if len(df0) != len(rc_names) or list(df0["RC"].astype(str)) != rc_names:
        st.session_state["qfd_rc_weight_df"] = init_rc_weight_df(rc_names)
    rc_weight_df = st.data_editor(st.session_state["qfd_rc_weight_df"], use_container_width=True, num_rows="fixed", key="qfd_rc_weight_editor")
    st.session_state["qfd_rc_weight_df"] = rc_weight_df
    rc_weights_dfs = [DFS(safe_float(row["mu_O"]), safe_float(row["nu_O"]), safe_float(row["mu_P"]), safe_float(row["nu_P"])).clip() for _, row in rc_weight_df.iterrows()]

    st.subheader("ICj (expected cost)")
    if "qfd_ic_df" not in st.session_state:
        st.session_state["qfd_ic_df"] = init_cost_df(ms_names)
    ic0 = st.session_state["qfd_ic_df"]
    if len(ic0) != len(ms_names) or list(ic0["MS"].astype(str)) != ms_names:
        st.session_state["qfd_ic_df"] = init_cost_df(ms_names)
    ic_df = st.data_editor(st.session_state["qfd_ic_df"], use_container_width=True, num_rows="fixed", key="qfd_ic_editor")
    st.session_state["qfd_ic_df"] = ic_df
    ic_vals = [max(safe_float(v,1.0),1e-9) for v in ic_df["ICj"].tolist()]

    st.subheader("Relationship matrices (RC × MS) per expert")
    rel_dfs_by_expert = []
    for e in range(n_exp):
        st.markdown(f"### Expert {e+1}")
        key = f"qfd_rel_df_exp_{e}"
        if key not in st.session_state:
            st.session_state[key] = init_relationship_df(rc_names, ms_names)
        dfrel = st.session_state[key]
        if len(dfrel) != len(rc_names) or list(dfrel["RC"].astype(str)) != rc_names:
            st.session_state[key] = init_relationship_df(rc_names, ms_names)
        rel_df = st.data_editor(st.session_state[key], use_container_width=True, height=320, num_rows="fixed", key=f"qfd_rel_editor_{e}")
        st.session_state[key] = rel_df
        mat_e = []
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
                row[f"{ms}_muO"] = round(v.mu_O,6); row[f"{ms}_nuO"] = round(v.nu_O,6)
                row[f"{ms}_muP"] = round(v.mu_P,6); row[f"{ms}_nuP"] = round(v.nu_P,6)
            comb_rows.append(row)
        combined_df = pd.DataFrame(comb_rows)
        st.session_state["qfd_combined_df"] = combined_df
        st.markdown("### Aggregated relationship matrix")
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
                row[f"{ms}_muO"] = round(v.mu_O,6); row[f"{ms}_nuO"] = round(v.nu_O,6)
                row[f"{ms}_muP"] = round(v.mu_P,6); row[f"{ms}_nuP"] = round(v.nu_P,6)
            weighted_rows.append(row)
        weighted_df = pd.DataFrame(weighted_rows)
        st.session_state["qfd_weighted_df"] = weighted_df
        st.markdown("### Weighted aggregated matrix")
        st.dataframe(weighted_df, use_container_width=True, height=320)

        crisp_contrib = np.zeros((n_rc, n_ms))
        for i in range(n_rc):
            for j in range(n_ms):
                crisp_contrib[i,j] = score_defuzz_from_weighted(weighted_dfs[i][j])
        crisp_df = pd.DataFrame(crisp_contrib, index=rc_names, columns=ms_names).round(6)
        st.session_state["qfd_crisp_df"] = crisp_df
        st.markdown("### Defuzzified weighted matrix")
        st.dataframe(crisp_df, use_container_width=True, height=320)
        fig = px.imshow(crisp_df, text_auto=True, aspect="auto", color_continuous_scale="Blues", title="Defuzzified Contributions")
        st.plotly_chart(fig, use_container_width=True)

        AIj = crisp_contrib.sum(axis=0)
        RIj = AIj / (float(AIj.sum()) if float(AIj.sum())!=0 else 1.0)
        SyP = np.array([AIj[j]/ic_vals[j] for j in range(n_ms)])
        RePj = SyP / (float(SyP.sum()) if float(SyP.sum())!=0 else 1.0)
        results = pd.DataFrame({
            "MS": ms_names, "AIj": AIj, "RIj (normalized AIj)": RIj,
            "ICj": ic_vals, "SyP = AIj/ICj": SyP, "RePj (normalized SyP)": RePj
        })
        results["Rank (RePj)"] = results["RePj (normalized SyP)"].rank(ascending=False, method="dense").astype(int)
        results = results.sort_values("RePj (normalized SyP)", ascending=False)
        st.session_state["qfd_results"] = results
        st.markdown("### Final RePj results")
        st.dataframe(results.round(6), use_container_width=True)
        fig = px.bar(results, x="MS", y="RePj (normalized SyP)", title="Mitigation Strategy Priorities", color="RePj (normalized SyP)", color_continuous_scale="Oranges")
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Send RePj to MILP", key="qfd_send_rep"):
            rep_df = pd.DataFrame({"MS": ms_names, "ReP": RePj})
            st.session_state["milp_rep_df"] = rep_df
            st.session_state["milp_ms_names"] = ms_names
            st.success("Sent to Module 5 (MILP).")

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            rc_weight_df.to_excel(writer, index=False, sheet_name="RC_Weights_DFS")
            combined_df.to_excel(writer, index=False, sheet_name="Aggregated_Relationship_DFS")
            weighted_df.to_excel(writer, index=False, sheet_name="Weighted_Aggregated_DFS")
            crisp_df.reset_index().rename(columns={"index":"RC"}).to_excel(writer, index=False, sheet_name="Defuzzified_Weighted_Matrix")
            results.to_excel(writer, index=False, sheet_name="Results")
        st.download_button("📥 Download Excel", buf.getvalue(), "DFS_QFD_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def page_milp():
    render_module_banner("🎯", "MILP Optimization", "Optimize strategy portfolio under budget and time constraints with pairwise savings.", badge="Module 5")
    st.sidebar.subheader("Model size")
    m = int(st.sidebar.number_input("Number of mitigation strategies", 2, 20, 9, key="milp_m"))
    n_exp = int(st.sidebar.number_input("Number of experts", 1, 6, 1, key="milp_ne"))

    st.subheader("Mitigation strategy names")
    cols = st.columns(3)
    names = []
    ms_defaults = st.session_state.get("milp_ms_names", [])
    for i in range(m):
        with cols[i%3]:
            default = ms_defaults[i] if i < len(ms_defaults) else f"MS{i+1}"
            names.append(st.text_input(f"Strategy {i+1}", value=default, key=f"milp_name_{i}"))

    key = f"milp|{m}|{n_exp}|" + "|".join(names)
    if st.session_state.get("milp_store_key") != key:
        st.session_state["milp_store_key"] = key
        if "milp_rep_df" in st.session_state and len(st.session_state["milp_rep_df"]) == m:
            st.session_state["milp_rep_df_local"] = st.session_state["milp_rep_df"].copy()
            st.session_state["milp_rep_df_local"]["MS"] = names
        else:
            st.session_state["milp_rep_df_local"] = pd.DataFrame({"MS": names, "ReP": [1.0]*m})
        st.session_state["milp_experts"] = {}
        for e in range(1, n_exp+1):
            st.session_state["milp_experts"][e] = {
                "cost_omp": pd.DataFrame({"MS": names, "O":0.0, "ML":0.0, "P":0.0}),
                "time_omp": pd.DataFrame({"MS": names, "O":0.0, "ML":0.0, "P":0.0}),
                "rho_O": make_square_df(names,0.0), "rho_ML": make_square_df(names,0.0), "rho_P": make_square_df(names,0.0),
                "zeta_O": make_square_df(names,0.0), "zeta_ML": make_square_df(names,0.0), "zeta_P": make_square_df(names,0.0),
            }

    st.subheader("ReP (from DFS-QFD)")
    st.session_state["milp_rep_df_local"] = st.data_editor(st.session_state["milp_rep_df_local"], use_container_width=True, height=240, num_rows="fixed", key="milp_rep_editor")

    st.sidebar.subheader("Aggregation")
    expert_choices = [f"Expert {i}" for i in range(1, n_exp+1)]
    selected = st.sidebar.multiselect("Experts to include", options=expert_choices, default=[expert_choices[0]] if expert_choices else [], key="milp_sel")
    selected_ids = [int(s.split()[-1]) for s in selected] if selected else ([1] if n_exp>=1 else [])
    scale_savings = st.sidebar.selectbox("Savings cost scaling", ["Use as entered", "Multiply ρᵢⱼ by 1000"], index=0, key="milp_scale")

    st.subheader("Budget & time limits")
    cA, cB = st.columns(2)
    with cA:
        budget = st.number_input("Available budget δ", 0.0, value=200000.0, step=1000.0, key="milp_budget")
    with cB:
        time_lim = st.number_input("Available time (months)", 0.0, value=60.0, step=1.0, key="milp_time")

    st.subheader("Enter expert O/ML/P data")
    st.info("Paste from Excel. For pairwise matrices fill only upper triangle (i<j).")
    for e in range(1, n_exp+1):
        with st.expander(f"Expert {e} inputs", expanded=(e==1)):
            tabs = st.tabs(["Cost (Ce)", "Time (T)", "Saving cost ρ", "Saving time ζ"])
            with tabs[0]:
                st.session_state["milp_experts"][e]["cost_omp"] = st.data_editor(st.session_state["milp_experts"][e]["cost_omp"], use_container_width=True, height=260, num_rows="fixed", key=f"milp_cost_{e}")
                exp_cost = compute_expected_vector(st.session_state["milp_experts"][e]["cost_omp"])
                show = st.session_state["milp_experts"][e]["cost_omp"][["MS","O","ML","P"]].copy()
                show["Expected"] = exp_cost
                st.dataframe(show, use_container_width=True)
            with tabs[1]:
                st.session_state["milp_experts"][e]["time_omp"] = st.data_editor(st.session_state["milp_experts"][e]["time_omp"], use_container_width=True, height=260, num_rows="fixed", key=f"milp_time_{e}")
                exp_time = compute_expected_vector(st.session_state["milp_experts"][e]["time_omp"])
                show = st.session_state["milp_experts"][e]["time_omp"][["MS","O","ML","P"]].copy()
                show["Expected"] = exp_time
                st.dataframe(show, use_container_width=True)
            with tabs[2]:
                sub = st.tabs(["ρ (O)","ρ (ML)","ρ (P)"])
                with sub[0]: st.session_state["milp_experts"][e]["rho_O"] = st.data_editor(st.session_state["milp_experts"][e]["rho_O"], use_container_width=True, height=320, key=f"milp_rhoO_{e}")
                with sub[1]: st.session_state["milp_experts"][e]["rho_ML"] = st.data_editor(st.session_state["milp_experts"][e]["rho_ML"], use_container_width=True, height=320, key=f"milp_rhoML_{e}")
                with sub[2]: st.session_state["milp_experts"][e]["rho_P"] = st.data_editor(st.session_state["milp_experts"][e]["rho_P"], use_container_width=True, height=320, key=f"milp_rhoP_{e}")
            with tabs[3]:
                sub = st.tabs(["ζ (O)","ζ (ML)","ζ (P)"])
                with sub[0]: st.session_state["milp_experts"][e]["zeta_O"] = st.data_editor(st.session_state["milp_experts"][e]["zeta_O"], use_container_width=True, height=320, key=f"milp_zetaO_{e}")
                with sub[1]: st.session_state["milp_experts"][e]["zeta_ML"] = st.data_editor(st.session_state["milp_experts"][e]["zeta_ML"], use_container_width=True, height=320, key=f"milp_zetaML_{e}")
                with sub[2]: st.session_state["milp_experts"][e]["zeta_P"] = st.data_editor(st.session_state["milp_experts"][e]["zeta_P"], use_container_width=True, height=320, key=f"milp_zetaP_{e}")

    if st.button("Solve optimization", type="primary", key="milp_solve"):
        if "milp_rep_df_local" not in st.session_state or len(st.session_state["milp_rep_df_local"]) != len(names):
            st.error("ReP table not ready.")
            return
        rep = np.array([_to_float(v,0.0) for v in st.session_state["milp_rep_df_local"]["ReP"].tolist()], dtype=float)
        exp_cost_list, exp_time_list, exp_rho_list, exp_zeta_list = [], [], [], []
        for e in selected_ids:
            cost_vec = compute_expected_vector(st.session_state["milp_experts"][e]["cost_omp"])
            time_vec = compute_expected_vector(st.session_state["milp_experts"][e]["time_omp"])
            rho = compute_expected_matrix(st.session_state["milp_experts"][e]["rho_O"], st.session_state["milp_experts"][e]["rho_ML"], st.session_state["milp_experts"][e]["rho_P"])
            zeta = compute_expected_matrix(st.session_state["milp_experts"][e]["zeta_O"], st.session_state["milp_experts"][e]["zeta_ML"], st.session_state["milp_experts"][e]["zeta_P"])
            exp_cost_list.append(cost_vec); exp_time_list.append(time_vec); exp_rho_list.append(rho); exp_zeta_list.append(zeta)
        if len(selected_ids)==1:
            cost, time, rho, zeta = exp_cost_list[0], exp_time_list[0], exp_rho_list[0], exp_zeta_list[0]
            agg_note = f"Using Expert {selected_ids[0]} only."
        else:
            cost = np.mean(np.stack(exp_cost_list, axis=0), axis=0)
            time = np.mean(np.stack(exp_time_list, axis=0), axis=0)
            rho = np.mean(np.stack(exp_rho_list, axis=0), axis=0)
            zeta = np.mean(np.stack(exp_zeta_list, axis=0), axis=0)
            agg_note = f"Average across experts: {', '.join([f'E{i}' for i in selected_ids])}"
        if scale_savings != "Use as entered":
            rho = rho * 1000.0
        best = solve_by_enumeration(rep, cost, time, rho, zeta, budget, time_lim)

        st.subheader("Results")
        st.success(agg_note)
        if best is None:
            st.error("No feasible solution found. Increase budget/time or adjust inputs.")
            return
        x = best["x"]
        selected_ms = [names[i] for i in range(len(names)) if x[i]==1]
        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Objective (Max ReP)", f"{best['obj']:.6f}")
        r2.metric("Total cost", f"{best['total_cost']:.3f}")
        r3.metric("Total time", f"{best['total_time']:.3f}")
        r4.metric("Selected count", f"{int(x.sum())}")
        st.write("**Selected strategies:** " + (", ".join(selected_ms) if selected_ms else "None"))
        breakdown = pd.DataFrame([{
            "Base cost (Σ Ce_j)": best["base_cost"], "Saving cost (Σ ρ_ij)": best["sav_cost"], "Total cost": best["total_cost"],
            "Base time (Σ T_j)": best["base_time"], "Saving time (Σ ζ_ij)": best["sav_time"], "Total time": best["total_time"],
        }])
        st.dataframe(breakdown, use_container_width=True)
        out = pd.DataFrame({"MS": names, "l_j": x.astype(int), "ReP_j": rep, "Ce_used": cost, "T_used": time, "ReP_j*l_j": rep*x})
        st.dataframe(out, use_container_width=True)
        st.session_state["milp_last_results"] = out
        pairs = []
        n = len(names)
        for i in range(n):
            for j in range(i+1, n):
                if x[i]==1 and x[j]==1 and (rho[i,j]!=0 or zeta[i,j]!=0):
                    pairs.append({"Pair": f"{names[i]} & {names[j]}", "ρ_ij used": rho[i,j], "ζ_ij used": zeta[i,j]})
        if pairs:
            st.dataframe(pd.DataFrame(pairs), use_container_width=True)
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("selection.csv", out.to_csv(index=False))
            z.writestr("breakdown.csv", breakdown.to_csv(index=False))
            z.writestr("rho_used_upper.csv", pd.DataFrame(rho, index=names, columns=names).to_csv(index=True))
            z.writestr("zeta_used_upper.csv", pd.DataFrame(zeta, index=names, columns=names).to_csv(index=True))
            z.writestr("rep.csv", st.session_state["milp_rep_df_local"].to_csv(index=False))
            z.writestr("aggregation_note.txt", agg_note)
        st.download_button("📥 Download ZIP", zbuf.getvalue(), "milp_results.zip", "application/zip")

# ============================================================
# MAIN
# ============================================================
def main():
    apply_custom_styling()
    if not check_password():
        return
    
    with st.sidebar:
        st.markdown('<div style="margin-bottom: 1rem;"><span style="font-size: 1.2rem; font-weight: 700;">🔬 Research Studio</span></div>', unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "1️⃣ Stratification", "2️⃣ Expert Covariance", "3️⃣ DFS-AHP", "4️⃣ DFS-QFD", "5️⃣ MILP"],
            index=0,
            key="nav_page",
            label_visibility="collapsed",
        )
        st.markdown("---")
        if st.button("🚪 Log out", use_container_width=True):
            logout()
    
    render_sidebar_research_profiles()
    render_app_header()
    
    if page == "📊 Dashboard":
        page_dashboard()
    elif page == "1️⃣ Stratification":
        page_stratification()
    elif page == "2️⃣ Expert Covariance":
        page_expert_covariance_model()
    elif page == "3️⃣ DFS-AHP":
        page_dfs_ahp()
    elif page == "4️⃣ DFS-QFD":
        page_dfs_qfd()
    elif page == "5️⃣ MILP":
        page_milp()
    
    render_footer()

if __name__ == "__main__":
    main()
