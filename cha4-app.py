# ============================================================
# IMPORTS AND SETUP
# ============================================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import base64
import io
from pathlib import Path
import hmac
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Set page config at the top
st.set_page_config(page_title="IT2TrFS-Delphi - WINGS - CoCoSo Toolkit", layout="wide")

# ============================================================
# RESEARCHER PROFILE HELPERS
# ============================================================
def get_asset_path(filename):
    """
    Resolve image file path.
    Put the images inside an 'assets' folder beside your app file.
    Example:
        app.py
        assets/
            prof_jz_ren.png
            abdul_moktadir.png
    """
    return Path(__file__).parent / "assets" / filename


def render_inventor_card(name, role, institution, image_path, brief_text, full_bio=None, extras=None):
    st.markdown('<div class="inventor-card">', unsafe_allow_html=True)

    if Path(image_path).exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.warning(f"Image not found: {image_path}")

    st.markdown(
        f"""
        <div class="inventor-name">{name}</div>
        <div class="inventor-role">{role}<br>{institution}</div>
        <div class="inventor-mini">{brief_text}</div>
        """,
        unsafe_allow_html=True,
    )

    if extras:
        for item in extras:
            st.markdown(
                f'<div class="inventor-mini">{item}</div>',
                unsafe_allow_html=True,
            )

    if full_bio:
        with st.expander("View full profile"):
            st.write(full_bio)

    st.markdown("</div>", unsafe_allow_html=True)


def render_inventor_profiles():
    st.markdown('<div class="inventor-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="inventor-section-title">Researcher Profiles</div>',
        unsafe_allow_html=True,
    )

    render_inventor_card(
        name="Prof. J.Z. Ren 任競爭",
        role="Associate Professor",
        institution="The Hong Kong Polytechnic University",
        image_path=get_asset_path("prof_jz_ren.png"),
        brief_text=(
            "Research focus: process systems engineering for energy, environment, "
            "and sustainability. Awarded the 2022 APEC ASPIRE Prize."
        ),
        full_bio=(
            "Dr. Jingzheng Ren is currently an Associate Professor at The Hong Kong "
            "Polytechnic University. His research focuses on process systems engineering "
            "for energy, environment and sustainability, including innovative industrial "
            "processes, decision-making tools, and optimization models for sustainable "
            "and carbon-neutral industrial systems. He has published extensively and has "
            "received major international recognition, including the 2022 APEC Science "
            "Prize for Innovation, Research and Education (ASPIRE Prize)."
        ),
    )

    render_inventor_card(
        name="Md. Abdul Moktadir",
        role="Assistant Professor (Leather Products Engineering)",
        institution="University of Dhaka / PolyU Presidential PhD Fellow",
        image_path=get_asset_path("abdul_moktadir.png"),
        brief_text=(
            "Current PhD researcher in Industrial and Systems Engineering at PolyU "
            "with research interests in sustainable supply chains, logistics, "
            "risk management, Industry 4.0, and circular economy."
        ),
        extras=[
            "Affiliation: University of Dhaka",
            "Program: PolyU Presidential PhD Fellow",
        ],
        full_bio=(
            "Md. Abdul Moktadir is currently pursuing a PhD in Industrial and Systems "
            "Engineering at The Hong Kong Polytechnic University. He also serves as an "
            "Assistant Professor of Leather Products Engineering at the University of "
            "Dhaka. His work has appeared in several international journals, and his "
            "research interests include sustainable supply chain management, risk "
            "management, energy-efficient supply chain planning and design, logistics, "
            "Industry 4.0, and circular economy."
        ),
    )

# ============================================================
# AUTHENTICATION
# ============================================================
def logout():
    st.session_state.authenticated = False
    st.rerun()


def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    expected_password = st.secrets.get("APP_PASSWORD", None)

    st.markdown(
        """
        <div class="login-shell">
            <div class="hero" style="margin-bottom:0.9rem;">
                <h2>IT2TrFS-Delphi - WINGS - CoCoSo Toolkit</h2>
                <p class="hero-sub">
                    Secure access to a decision analytics workspace for IT2TrFS optimization and ranking.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="login-shell"><div class="login-card">', unsafe_allow_html=True)
    st.markdown("### Sign in")
    st.caption("Enter the application password to continue.")

    if expected_password is None:
        st.error(
            "APP_PASSWORD is not configured. Add it in Streamlit secrets "
            "(.streamlit/secrets.toml locally or Settings → Secrets on Streamlit Cloud)."
        )
        st.markdown("</div></div>", unsafe_allow_html=True)
        return False

    with st.form("login_form", clear_on_submit=False):
        password = st.text_input("Password", type="password", placeholder="Enter password")
        submitted = st.form_submit_button("Log in", use_container_width=True)

    if submitted:
        if hmac.compare_digest(password, str(expected_password)):
            st.session_state.authenticated = True
            st.success("Access granted.")
            st.rerun()
        else:
            st.error("Incorrect password.")

    st.markdown("</div></div>", unsafe_allow_html=True)
    return False

def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            © 2026 Developed by <strong>Moktadir M.A.</strong> and <strong>REN J.Z.</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# IT2TrFS OPERATIONS
# ============================================================
def zero_it2():
    return ((0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0))

def add_it2(A, B):
    Au, Al = A
    Bu, Bl = B
    new_u = (Au[0] + Bu[0], Au[1] + Bu[1], Au[2] + Bu[2], Au[3] + Bu[3], Au[4], Au[5])
    new_l = (Al[0] + Bl[0], Al[1] + Bl[1], Al[2] + Bl[2], Al[3] + Bl[3], Al[4], Al[5])
    return (new_u, new_l)

def sub_it2(A, B):
    Au, Al = A
    Bu, Bl = B
    new_u = (Au[0] - Bu[3], Au[1] - Bu[2], Au[2] - Bu[1], Au[3] - Bu[0], Au[4], Au[5])
    new_l = (Al[0] - Bl[3], Al[1] - Bl[2], Al[2] - Bl[1], Al[3] - Bl[0], Al[4], Al[5])
    return (new_u, new_l)

def scalar_mul_it2(k, A):
    Au, Al = A
    new_u = (k * Au[0], k * Au[1], k * Au[2], k * Au[3], Au[4], Au[5])
    new_l = (k * Al[0], k * Al[1], k * Al[2], k * Al[3], Al[4], Al[5])
    return (new_u, new_l)

def format_it2(A):
    Au, Al = A
    return f"({Au[0]:.4f}, {Au[1]:.4f}, {Au[2]:.4f}, {Au[3]:.4f}; {Al[0]:.4f}, {Al[1]:.4f}, {Al[2]:.4f}, {Al[3]:.4f})"

# ============================================================
# DELPHI MODULE
# ============================================================
def delphi_app():
    st.title("📊 IT2TrFS-Delphi Consensus Analysis")
    st.write("IT2TrFS-Delphi module for consensus reaching")
    
    # Add footer at the end
    render_footer()

# ============================================================
# WINGS MODULE
# ============================================================
def wings_app():
    st.title("📊 IT2TrFS WINGS Method Analysis Platform")
    st.write("IT2TrFS-WINGS module")
    
    # Add footer at the end
    render_footer()

# ============================================================
# CoCoSo MODULE
# ============================================================
def cocoso_app():
    st.title("📊 IT2TrFS-CoCoSo Ranking Method")
    st.write("IT2TrFS-CoCoSo module for multi-criteria ranking")
    
    # Add footer at the end
    render_footer()

# ============================================================
# MAIN NAVIGATION
# ============================================================
def main():
    # Check authentication first
    if not check_password():
        return  # Stop if not authenticated
    
    # Main navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a Module", 
                           ["IT2TrFS-Delphi", "IT2TrFS-WINGS", "IT2TrFS-CoCoSo", "Researcher Profiles"])
    
    # Logout button in sidebar
    if st.sidebar.button("Logout", type="secondary"):
        logout()
    
    if page == "IT2TrFS-Delphi":
        delphi_app()
    elif page == "IT2TrFS-WINGS":
        wings_app()
    elif page == "IT2TrFS-CoCoSo":
        cocoso_app()
    elif page == "Researcher Profiles":
        render_inventor_profiles()
        render_footer()

if __name__ == "__main__":
    main()
