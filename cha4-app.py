import io
import re
import base64
from datetime import datetime

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH


APP_TITLE = "IT2TrFS Decision Studio"


# =========================================================
# Shared UI helpers
# =========================================================

def inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        .hero {
            padding: 1.25rem 1.35rem;
            border: 1px solid rgba(120,120,120,.18);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(56, 189, 248, .10), rgba(16, 185, 129, .08));
            margin-bottom: 1rem;
        }
        .hero h1, .hero h2, .hero h3 {margin: 0 0 .35rem 0;}
        .muted {color: rgba(120,120,120,.95);}
        div[data-testid="stMetric"] {
            border: 1px solid rgba(120,120,120,.18);
            border-radius: 14px;
            padding: 0.55rem 0.7rem;
            background: rgba(255,255,255,.02);
        }
        .section-card {
            border: 1px solid rgba(120,120,120,.18);
            border-radius: 16px;
            padding: 1rem;
            background: rgba(255,255,255,.02);
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero">
            <h2>{title}</h2>
            <div class="muted">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def csv_download_button(df: pd.DataFrame, label: str, file_name: str):
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )


def docx_download_button(doc: Document, label: str, file_name: str):
    stream = io.BytesIO()
    doc.save(stream)
    stream.seek(0)
    st.download_button(
        label,
        data=stream.getvalue(),
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )


# =========================================================
# IT2TrFS primitives
# =========================================================

def format_it2(it2):
    u, l = it2
    return (
        f"(({u[0]:.6f},{u[1]:.6f},{u[2]:.6f},{u[3]:.6f};{u[4]:.1f},{u[5]:.1f}), "
        f"({l[0]:.6f},{l[1]:.6f},{l[2]:.6f},{l[3]:.6f};{l[4]:.1f},{l[5]:.1f}))"
    )


def zero_it2():
    return ((0, 0, 0, 0, 1, 1), (0, 0, 0, 0, 0.9, 0.9))


def add_it2(a, b):
    au, al = a
    bu, bl = b
    return (
        (au[0] + bu[0], au[1] + bu[1], au[2] + bu[2], au[3] + bu[3], min(au[4], bu[4]), min(au[5], bu[5])),
        (al[0] + bl[0], al[1] + bl[1], al[2] + bl[2], al[3] + bl[3], min(al[4], bl[4]), min(al[5], bl[5])),
    )


def sub_it2(a, b):
    au, al = a
    bu, bl = b
    return (
        (au[0] - bu[0], au[1] - bu[1], au[2] - bu[2], au[3] - bu[3], min(au[4], bu[4]), min(au[5], bu[5])),
        (al[0] - bl[0], al[1] - bl[1], al[2] - bl[2], al[3] - bl[3], min(al[4], bl[4]), min(al[5], bl[5])),
    )


def scalar_mul_it2(k, a):
    au, al = a
    return (
        (k * au[0], k * au[1], k * au[2], k * au[3], au[4], au[5]),
        (k * al[0], k * al[1], k * al[2], k * al[3], al[4], al[5]),
    )


def it2_pow(a, w):
    au, al = a

    def pw(x):
        if x == 0 and w == 0:
            return 1.0
        return float(x) ** float(w)

    return (
        (pw(au[0]), pw(au[1]), pw(au[2]), pw(au[3]), au[4], au[5]),
        (pw(al[0]), pw(al[1]), pw(al[2]), pw(al[3]), al[4], al[5]),
    )


def it2_to_row(it2):
    au, al = it2
    return {
        "a": au[0], "b": au[1], "c": au[2], "d": au[3], "uh1": au[4], "uh2": au[5],
        "e": al[0], "f": al[1], "g": al[2], "h": al[3], "lh1": al[4], "lh2": al[5],
    }


def format_it2_table(matrix_dict, alternatives, criteria):
    df = pd.DataFrame(index=alternatives, columns=criteria, dtype=object)
    for alt in alternatives:
        for crit in criteria:
            df.loc[alt, crit] = format_it2(matrix_dict[(alt, crit)])
    return df


def cocoso_crisp_score(it2):
    au, al = it2
    a, b, c, d, uh1, uh2 = au
    e, f, g, h, lh1, lh2 = al
    score_u = (((d - a) + ((uh2 * c) - a) + ((uh1 * b) - a)) / 4.0) + a
    score_l = (((h - e) + ((lh2 * g) - e) + ((lh1 * f) - e)) / 4.0) + e
    return (score_u + score_l) / 2.0


def it2_score_components(it2):
    au, al = it2
    a, b, c, d, uh1, uh2 = au
    e, f, g, h, lh1, lh2 = al
    score_u = (((d - a) + ((uh2 * c) - a) + ((uh1 * b) - a)) / 4.0) + a
    score_l = (((h - e) + ((lh2 * g) - e) + ((lh1 * f) - e)) / 4.0) + e
    return score_u, score_l, (score_u + score_l) / 2.0


def parse_loose_numeric(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    m = re.search(r"-?\d+(?:\.\d+)?", str(x).strip())
    if not m:
        return np.nan
    try:
        return float(m.group())
    except Exception:
        return np.nan


# =========================================================
# Delphi
# =========================================================

DELPHI_NUMERIC_SCALE = {
    1: ((0, 0.1, 0.1, 0.1, 1, 1), (0.0, 0.1, 0.1, 0.05, 0.9, 0.9)),
    2: ((0.2, 0.3, 0.3, 0.4, 1, 1), (0.25, 0.3, 0.3, 0.35, 0.9, 0.9)),
    3: ((0.4, 0.5, 0.5, 0.6, 1, 1), (0.45, 0.5, 0.5, 0.55, 0.9, 0.9)),
    4: ((0.6, 0.7, 0.7, 0.8, 1, 1), (0.65, 0.7, 0.7, 0.75, 0.9, 0.9)),
    5: ((0.8, 0.9, 0.9, 1.0, 1, 1), (0.85, 0.9, 0.9, 0.95, 0.9, 0.9)),
}

DELPHI_TERM_LABELS = {1: "VLR", 2: "LR", 3: "MR", 4: "HR", 5: "VHR"}
DELPHI_TERM_FULL = {
    1: "Very Low Relevance",
    2: "Low Relevance",
    3: "Medium Relevance",
    4: "High Relevance",
    5: "Very High Relevance",
}


def format_delphi_scale_table():
    rows = []
    for k, v in DELPHI_NUMERIC_SCALE.items():
        rows.append(
            {
                "Score": k,
                "Abbreviation": DELPHI_TERM_LABELS[k],
                "Meaning": DELPHI_TERM_FULL[k],
                "IT2TrFS": format_it2(v),
            }
        )
    return pd.DataFrame(rows)


def aggregate_it2_delphi_from_scores(scores_df: pd.DataFrame):
    aggregated = {}
    summary_rows = []

    for crit in scores_df.columns:
        col = scores_df[crit].apply(parse_loose_numeric)
        valid_scores = col[col.isin([1, 2, 3, 4, 5])].astype(int).tolist()

        if not valid_scores:
            aggregated[crit] = zero_it2()
            summary_rows.append(
                {
                    "Criterion": crit,
                    "N valid": 0,
                    "Mean rating": np.nan,
                    "Std rating": np.nan,
                    "Mode rating": np.nan,
                    "a": np.nan, "b": np.nan, "c": np.nan, "d": np.nan, "uh1": np.nan, "uh2": np.nan,
                    "e": np.nan, "f": np.nan, "g": np.nan, "h": np.nan, "lh1": np.nan, "lh2": np.nan,
                    "Score(UMF)": np.nan, "Score(LMF)": np.nan, "Crisp score": np.nan,
                }
            )
            continue

        n = len(valid_scores)
        agg = (
            (
                sum(DELPHI_NUMERIC_SCALE[s][0][0] for s in valid_scores) / n,
                sum(DELPHI_NUMERIC_SCALE[s][0][1] for s in valid_scores) / n,
                sum(DELPHI_NUMERIC_SCALE[s][0][2] for s in valid_scores) / n,
                sum(DELPHI_NUMERIC_SCALE[s][0][3] for s in valid_scores) / n,
                min(DELPHI_NUMERIC_SCALE[s][0][4] for s in valid_scores),
                min(DELPHI_NUMERIC_SCALE[s][0][5] for s in valid_scores),
            ),
            (
                sum(DELPHI_NUMERIC_SCALE[s][1][0] for s in valid_scores) / n,
                sum(DELPHI_NUMERIC_SCALE[s][1][1] for s in valid_scores) / n,
                sum(DELPHI_NUMERIC_SCALE[s][1][2] for s in valid_scores) / n,
                sum(DELPHI_NUMERIC_SCALE[s][1][3] for s in valid_scores) / n,
                min(DELPHI_NUMERIC_SCALE[s][1][4] for s in valid_scores),
                min(DELPHI_NUMERIC_SCALE[s][1][5] for s in valid_scores),
            ),
        )
        aggregated[crit] = agg
        score_u, score_l, crisp = it2_score_components(agg)

        valid_numeric = np.array(valid_scores, dtype=float)
        mode_val = pd.Series(valid_numeric).mode()
        mode_rating = float(mode_val.iloc[0]) if not mode_val.empty else np.nan

        row = {
            "Criterion": crit,
            "N valid": len(valid_scores),
            "Mean rating": float(np.mean(valid_numeric)),
            "Std rating": float(np.std(valid_numeric, ddof=1)) if len(valid_numeric) > 1 else 0.0,
            "Mode rating": mode_rating,
            **it2_to_row(agg),
            "Score(UMF)": score_u,
            "Score(LMF)": score_l,
            "Crisp score": crisp,
        }
        summary_rows.append(row)

    return aggregated, pd.DataFrame(summary_rows)


def detect_delphi_columns(df: pd.DataFrame):
    candidates = []
    for col in df.columns:
        if str(col).lower().startswith("unnamed"):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        first_chunk = s.iloc[:100].dropna()
        if len(first_chunk) == 0:
            continue
        if first_chunk.between(1, 5, inclusive="both").mean() >= 0.80:
            candidates.append(col)
    return candidates


def extract_contiguous_response_block(df: pd.DataFrame, criteria_cols):
    if len(criteria_cols) == 0:
        return df.iloc[0:0].copy()

    temp = df[criteria_cols].apply(pd.to_numeric, errors="coerce")
    row_has_response = temp.notna().any(axis=1)

    started = False
    keep_idx = []
    for idx, flag in row_has_response.items():
        if flag and not started:
            started = True
        if started:
            if flag:
                keep_idx.append(idx)
            else:
                break

    if not keep_idx:
        return df.iloc[0:0].copy()
    return df.loc[keep_idx].copy()


def render_delphi_results(clean_scores_df: pd.DataFrame, summary_df: pd.DataFrame, threshold: float):
    tabs = st.tabs(["Summary", "Clean matrix", "Chart", "Export"])

    out = summary_df.copy()
    out["Aggregated IT2TrFS"] = out.apply(
        lambda row: format_it2(
            (
                (row["a"], row["b"], row["c"], row["d"], row["uh1"], row["uh2"]),
                (row["e"], row["f"], row["g"], row["h"], row["lh1"], row["lh2"]),
            )
        ),
        axis=1,
    )
    out["Status"] = np.where(out["Crisp score"] >= threshold, "Accepted", "Rejected")
    out["Rank"] = out["Crisp score"].rank(ascending=False, method="min").astype("Int64")
    out = out.sort_values(["Status", "Crisp score"], ascending=[False, False]).reset_index(drop=True)

    with tabs[0]:
        accepted = out.loc[out["Status"] == "Accepted", "Criterion"].tolist()
        rejected = out.loc[out["Status"] == "Rejected", "Criterion"].tolist()
        c1, c2, c3 = st.columns(3)
        c1.metric("Accepted criteria", len(accepted))
        c2.metric("Rejected criteria", len(rejected))
        c3.metric("Threshold", f"{threshold:.2f}")

        display_df = out[
            ["Criterion", "Aggregated IT2TrFS", "Mode rating", "Score(UMF)", "Score(LMF)", "Crisp score", "Status", "Rank"]
        ].copy()
        st.dataframe(
            display_df.style.format(
                {"Mode rating": "{:.0f}", "Score(UMF)": "{:.6f}", "Score(LMF)": "{:.6f}", "Crisp score": "{:.6f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        if accepted:
            st.success("Accepted criteria: " + ", ".join(accepted))
        if rejected:
            st.info("Rejected criteria: " + ", ".join(rejected))

    with tabs[1]:
        clean_it2_df = clean_scores_df.copy().applymap(
            lambda x: format_it2(DELPHI_NUMERIC_SCALE[int(x)]) if pd.notna(x) and int(x) in DELPHI_NUMERIC_SCALE else ""
        )
        st.dataframe(clean_it2_df, use_container_width=True)

    with tabs[2]:
        plot_df = out[["Criterion", "Crisp score"]].copy()
        st.bar_chart(plot_df.set_index("Criterion"))

    with tabs[3]:
        csv_download_button(out, "Download Delphi summary (CSV)", "it2trfs_delphi_summary.csv")


def delphi_page():
    hero("IT2TrFS-Delphi", "Criterion screening with a cleaner upload flow, better summaries, and more readable outputs.")

    howto_tab, excel_tab, manual_tab = st.tabs(["How it works", "Excel upload", "Manual entry"])

    with howto_tab:
        st.markdown(
            """
            **Workflow**
            1. Convert 1–5 Delphi ratings into IT2TrFS values.
            2. Aggregate valid responses criterion-wise.
            3. Compute Score(UMF), Score(LMF), and crisp score.
            4. Screen criteria using a configurable threshold.
            """
        )
        st.dataframe(format_delphi_scale_table(), use_container_width=True, hide_index=True)

    with excel_tab:
        c1, c2 = st.columns([1.7, 1])
        with c1:
            uploaded_file = st.file_uploader("Upload Delphi workbook", type=["xlsx", "xls"], key="delphi_upload")
        with c2:
            threshold = st.number_input("Acceptance threshold", min_value=0.0, max_value=1.0, value=0.60, step=0.01)

        if uploaded_file is not None:
            try:
                xls = pd.ExcelFile(uploaded_file)
                sheet_name = st.selectbox("Worksheet", options=xls.sheet_names, index=0)
                df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name)

                st.markdown("### Sheet preview")
                st.dataframe(df_raw.head(10), use_container_width=True)

                detected_cols = detect_delphi_columns(df_raw)
                response_block = extract_contiguous_response_block(df_raw, detected_cols)

                with st.expander("Advanced settings"):
                    criteria_cols = st.multiselect(
                        "Detected Delphi columns",
                        options=list(df_raw.columns),
                        default=detected_cols,
                    )
                    start_default = 1 if response_block.empty else int(response_block.index.min()) + 2
                    end_default = min(len(df_raw), int(response_block.index.max()) + 2) if not response_block.empty else min(10, len(df_raw))
                    start_row = st.number_input("Start row (1-based after header)", min_value=1, max_value=max(len(df_raw), 1), value=start_default, step=1)
                    end_row = st.number_input("End row (1-based after header)", min_value=1, max_value=max(len(df_raw), 1), value=end_default, step=1)

                if len(criteria_cols) == 0:
                    st.warning("No Delphi columns selected.")
                elif st.button("Run Delphi analysis", type="primary", use_container_width=True):
                    response_block = df_raw.iloc[int(start_row) - 1:int(end_row)].copy()
                    scores_df = response_block[criteria_cols].applymap(parse_loose_numeric)
                    scores_df = scores_df.where(scores_df.isin([1, 2, 3, 4, 5]), np.nan)
                    scores_df = scores_df.loc[scores_df.notna().any(axis=1)].reset_index(drop=True)

                    if len(scores_df) == 0:
                        st.error("No valid 1–5 Delphi responses were found.")
                    else:
                        _, summary_df = aggregate_it2_delphi_from_scores(scores_df)
                        render_delphi_results(scores_df, summary_df, threshold)

            except Exception as e:
                st.error(f"Could not read the workbook: {e}")

    with manual_tab:
        c1, c2, c3 = st.columns(3)
        n_respondents = c1.number_input("Respondents", min_value=1, max_value=50, value=5, step=1)
        n_criteria = c2.number_input("Criteria", min_value=1, max_value=50, value=6, step=1)
        threshold = c3.number_input("Acceptance threshold", min_value=0.0, max_value=1.0, value=0.60, step=0.01, key="delphi_threshold_manual")

        criteria_names = []
        cols = st.columns(min(4, n_criteria))
        for i in range(n_criteria):
            with cols[i % len(cols)]:
                criteria_names.append(st.text_input(f"Criterion {i+1}", value=f"C{i+1}", key=f"delphi_crit_{i}"))

        expected_index = [f"R{i+1}" for i in range(n_respondents)]
        key = "delphi_manual_df"
        if key not in st.session_state or list(st.session_state[key].index) != expected_index or list(st.session_state[key].columns) != criteria_names:
            st.session_state[key] = pd.DataFrame(3, index=expected_index, columns=criteria_names)

        edited_scores = st.data_editor(
            st.session_state[key],
            use_container_width=True,
            column_config={c: st.column_config.NumberColumn(c, min_value=1, max_value=5, step=1, format="%d") for c in criteria_names},
            key="delphi_editor",
        )

        if st.button("Run manual Delphi analysis", type="primary", use_container_width=True):
            clean_scores = edited_scores.applymap(parse_loose_numeric)
            clean_scores = clean_scores.where(clean_scores.isin([1, 2, 3, 4, 5]), np.nan)
            _, summary_df = aggregate_it2_delphi_from_scores(clean_scores)
            render_delphi_results(clean_scores, summary_df, threshold)


# =========================================================
# CoCoSo
# =========================================================

COCOSO_LINGUISTIC_TERMS = {
    "VP": ((0, 0, 0, 0.1, 1, 1), (0.05, 0, 0, 0.05, 0.9, 0.9)),
    "P": ((0, 0.1, 0.1, 0.3, 1, 1), (0.05, 0.1, 0.1, 0.25, 0.9, 0.9)),
    "MP": ((0.1, 0.3, 0.3, 0.5, 1, 1), (0.15, 0.3, 0.3, 0.45, 0.9, 0.9)),
    "F": ((0.3, 0.5, 0.5, 0.7, 1, 1), (0.35, 0.5, 0.5, 0.65, 0.9, 0.9)),
    "MG": ((0.5, 0.7, 0.7, 0.9, 1, 1), (0.55, 0.7, 0.7, 0.85, 0.9, 0.9)),
    "G": ((0.7, 0.9, 0.9, 1.0, 1, 1), (0.75, 0.9, 0.9, 0.95, 0.9, 0.9)),
    "VG": ((0.9, 1.0, 1.0, 1.0, 1, 1), (0.95, 1.0, 1.0, 0.95, 0.9, 0.9)),
}
COCOSO_FULL = {"VP": "Very Poor", "P": "Poor", "MP": "Medium Poor", "F": "Fair", "MG": "Medium Good", "G": "Good", "VG": "Very Good"}


def is_benefit_type(t):
    s = str(t).strip().lower()
    return s.startswith("b") or "benefit" in s or s == "ben" or "max" in s


def safe_div(num, den):
    den = float(den)
    return 0.0 if den == 0 else float(num) / den


def normalize_it2_matrix_excel(agg_matrix, criteria_types, alternatives, criteria):
    norm = {}
    for j, crit in enumerate(criteria):
        all_a, all_b, all_c, all_d = [], [], [], []
        for alt in alternatives:
            au, _ = agg_matrix[(alt, crit)]
            all_a.append(float(au[0]))
            all_b.append(float(au[1]))
            all_c.append(float(au[2]))
            all_d.append(float(au[3]))

        delta_plus = max(max(all_a), max(all_b), max(all_c), max(all_d)) if alternatives else 1.0
        delta_minus = min(min(all_a), min(all_b), min(all_c), min(all_d)) if alternatives else 0.0

        if is_benefit_type(criteria_types[j]):
            div = delta_plus if delta_plus != 0 else 1.0
            for alt in alternatives:
                au, al = agg_matrix[(alt, crit)]
                a, b, c, d, uh1, uh2 = au
                e, f, g, h, lh1, lh2 = al
                norm[(alt, crit)] = (
                    (safe_div(a, div), safe_div(b, div), safe_div(c, div), safe_div(d, div), uh1, uh2),
                    (safe_div(e, div), safe_div(f, div), safe_div(g, div), safe_div(h, div), lh1, lh2),
                )
        else:
            m = delta_minus
            for alt in alternatives:
                au, al = agg_matrix[(alt, crit)]
                aU, bU, cU, dU, uh1, uh2 = au
                eL, fL, gL, hL, lh1, lh2 = al
                norm[(alt, crit)] = (
                    (safe_div(m, dU), safe_div(m, cU), safe_div(m, bU), safe_div(m, aU), uh1, uh2),
                    (safe_div(m, hL), safe_div(m, gL), safe_div(m, fL), safe_div(m, eL), lh1, lh2),
                )
    return norm


def cocoso_page():
    hero("IT2TrFS-CoCoSo", "Professional data-entry flow for alternatives, criteria, expert judgments, normalization, and final ranking.")

    with st.expander("Linguistic scale"):
        scale_df = pd.DataFrame(
            [{"Abbreviation": k, "Meaning": COCOSO_FULL[k], "IT2TrFS": format_it2(v)} for k, v in COCOSO_LINGUISTIC_TERMS.items()]
        )
        st.dataframe(scale_df, use_container_width=True, hide_index=True)

    st.markdown("### Step 1 · Decision structure")
    c1, c2 = st.columns(2)
    alternatives = [a.strip() for a in c1.text_input("Alternatives", "T1, T2, T3").split(",") if a.strip()]
    criteria = [c.strip() for c in c2.text_input("Criteria", "C1, C2, C3").split(",") if c.strip()]

    if len(alternatives) < 1 or len(criteria) < 1:
        st.warning("Provide at least one alternative and one criterion.")
        return

    state_key = "cocoso_criteria_df"
    if state_key not in st.session_state or list(st.session_state[state_key]["Criterion"]) != criteria:
        w = [round(1 / len(criteria), 6)] * len(criteria)
        if len(criteria) > 0:
            w[-1] = 1.0 - sum(w[:-1])
        st.session_state[state_key] = pd.DataFrame({"Criterion": criteria, "Type": ["Benefit"] * len(criteria), "Weight": w})

    crit_df = st.data_editor(
        st.session_state[state_key],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Type": st.column_config.SelectboxColumn("Type", options=["Benefit", "Cost"]),
            "Weight": st.column_config.NumberColumn("Weight", min_value=0.0, max_value=1.0, step=0.00001, format="%.5f"),
        },
        key="cocoso_criteria_editor",
    )

    criteria_types = crit_df["Type"].tolist()
    criteria_weights = crit_df["Weight"].astype(float).tolist()

    if not np.isclose(sum(criteria_weights), 1.0):
        st.error(f"Criteria weights must sum to 1.0. Current sum: {sum(criteria_weights):.5f}")
        return

    st.markdown("### Step 2 · Expert evaluations")
    num_experts = st.number_input("Number of experts", min_value=1, max_value=30, value=2, step=1)
    expert_weights = []
    if num_experts == 1:
        expert_weights = [1.0]
        st.info("Single expert mode: expert weight = 1.0")
    else:
        cols = st.columns(num_experts)
        for i in range(num_experts):
            with cols[i]:
                expert_weights.append(
                    st.number_input(
                        f"E{i+1} weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=round(1 / num_experts, 6),
                        step=0.01,
                        format="%.6f",
                        key=f"cocoso_weight_{i}",
                    )
                )
        if not np.isclose(sum(expert_weights), 1.0):
            st.error(f"Expert weights must sum to 1.0. Current sum: {sum(expert_weights):.5f}")
            return

    key = "cocoso_expert_dfs"
    reset_needed = (
        key not in st.session_state
        or len(st.session_state[key]) != num_experts
        or list(st.session_state[key].get(0, pd.DataFrame()).index) != alternatives
        or list(st.session_state[key].get(0, pd.DataFrame()).columns) != criteria
    )
    if reset_needed:
        st.session_state[key] = {i: pd.DataFrame("F", index=alternatives, columns=criteria) for i in range(num_experts)}

    tabs = st.tabs([f"Expert {i+1}" for i in range(num_experts)])
    for i, tab in enumerate(tabs):
        with tab:
            st.session_state[key][i] = st.data_editor(
                st.session_state[key][i],
                use_container_width=True,
                column_config={c: st.column_config.SelectboxColumn(c, options=list(COCOSO_LINGUISTIC_TERMS.keys())) for c in criteria},
                key=f"cocoso_editor_{i}",
            )

    st.markdown("### Step 3 · Run model")
    tau = st.number_input("τ (tau)", min_value=0.0, max_value=1.0, value=0.50, step=0.05)

    if st.button("Run CoCoSo analysis", type="primary", use_container_width=True):
        agg_matrix = {}
        for alt in alternatives:
            for crit in criteria:
                acc = None
                for e in range(num_experts):
                    term = st.session_state[key][e].loc[alt, crit]
                    it2 = COCOSO_LINGUISTIC_TERMS[term]
                    it2w = scalar_mul_it2(expert_weights[e], it2)
                    acc = it2w if acc is None else add_it2(acc, it2w)
                agg_matrix[(alt, crit)] = acc

        norm_matrix = normalize_it2_matrix_excel(agg_matrix, criteria_types, alternatives, criteria)

        SBi, PBi = {}, {}
        for alt in alternatives:
            s_acc = zero_it2()
            p_acc = zero_it2()
            for j, crit in enumerate(criteria):
                r = norm_matrix[(alt, crit)]
                wj = float(criteria_weights[j])
                s_acc = add_it2(s_acc, scalar_mul_it2(wj, r))
                p_acc = add_it2(p_acc, it2_pow(r, wj))
            SBi[alt] = s_acc
            PBi[alt] = p_acc

        crisp_S = {alt: cocoso_crisp_score(SBi[alt]) for alt in alternatives}
        crisp_P = {alt: cocoso_crisp_score(PBi[alt]) for alt in alternatives}

        sumS, sumP = sum(crisp_S.values()), sum(crisp_P.values())
        minS, minP = min(crisp_S.values()), min(crisp_P.values())
        maxS, maxP = max(crisp_S.values()), max(crisp_P.values())
        denom_kic = tau * maxS + (1.0 - tau) * maxP
        denom_kic = denom_kic if denom_kic != 0 else 1.0

        rows = []
        for alt in alternatives:
            S, P = crisp_S[alt], crisp_P[alt]
            Kia = (S + P) / (sumS + sumP) if (sumS + sumP) != 0 else 0.0
            Kib = (S / minS if minS != 0 else 0.0) + (P / minP if minP != 0 else 0.0)
            Kic = ((tau * S) + ((1.0 - tau) * P)) / denom_kic
            K = (Kia * Kib * Kic) ** (1 / 3) + ((Kia + Kib + Kic) / 3)
            rows.append({"Alternative": alt, "Kia": Kia, "Kib": Kib, "Kic": Kic, "K": K})

        dfK = pd.DataFrame(rows)
        dfK["Rank"] = dfK["K"].rank(ascending=False, method="min").astype(int)
        dfK = dfK.sort_values("Rank").reset_index(drop=True)

        tabs = st.tabs(["Ranking", "Matrices", "Intermediate values", "Export"])

        with tabs[0]:
            c1, c2 = st.columns(2)
            c1.metric("Best alternative", dfK.iloc[0]["Alternative"])
            c2.metric("Top K value", f"{dfK.iloc[0]['K']:.6f}")
            st.dataframe(dfK.style.format(precision=6), use_container_width=True, hide_index=True)
            st.bar_chart(dfK.set_index("Alternative")["K"])

        with tabs[1]:
            st.markdown("### Aggregated IT2TrFS decision matrix")
            st.dataframe(format_it2_table(agg_matrix, alternatives, criteria), use_container_width=True)
            st.markdown("### Normalized IT2TrFS matrix")
            st.dataframe(format_it2_table(norm_matrix, alternatives, criteria), use_container_width=True)

        with tabs[2]:
            sbi_df = pd.DataFrame([{"Alternative": alt, **it2_to_row(SBi[alt])} for alt in alternatives])
            pbi_df = pd.DataFrame([{"Alternative": alt, **it2_to_row(PBi[alt])} for alt in alternatives])
            crisp_df = pd.DataFrame({"Alternative": alternatives, "Crisp SBi": [crisp_S[a] for a in alternatives], "Crisp PBi": [crisp_P[a] for a in alternatives]})
            st.markdown("### SBi")
            st.dataframe(sbi_df.style.format(precision=6), use_container_width=True, hide_index=True)
            st.markdown("### PBi")
            st.dataframe(pbi_df.style.format(precision=6), use_container_width=True, hide_index=True)
            st.markdown("### Crisp scores")
            st.dataframe(crisp_df.style.format(precision=6), use_container_width=True, hide_index=True)

        with tabs[3]:
            csv_download_button(dfK, "Download final ranking (CSV)", "cocoso_ranking.csv")


# =========================================================
# WINGS
# =========================================================

LINGUISTIC_TERMS = {
    "strength": {
        "VLR": ((0, 0.1, 0.1, 0.1, 1, 1), (0.0, 0.1, 0.1, 0.05, 0.9, 0.9)),
        "LR": ((0.2, 0.3, 0.3, 0.4, 1, 1), (0.25, 0.3, 0.3, 0.35, 0.9, 0.9)),
        "MR": ((0.4, 0.5, 0.5, 0.6, 1, 1), (0.45, 0.5, 0.5, 0.55, 0.9, 0.9)),
        "HR": ((0.6, 0.7, 0.7, 0.8, 1, 1), (0.65, 0.7, 0.7, 0.75, 0.9, 0.9)),
        "VHR": ((0.8, 0.9, 0.9, 1.0, 1, 1), (0.85, 0.90, 0.90, 0.95, 0.9, 0.9)),
    },
    "influence": {
        "ELI": ((0, 0.1, 0.1, 0.2, 1, 1), (0.05, 0.1, 0.1, 0.15, 0.9, 0.9)),
        "VLI": ((0.1, 0.2, 0.2, 0.35, 1, 1), (0.15, 0.2, 0.2, 0.3, 0.9, 0.9)),
        "LI": ((0.2, 0.35, 0.35, 0.5, 1, 1), (0.25, 0.35, 0.35, 0.45, 0.9, 0.9)),
        "MI": ((0.35, 0.5, 0.5, 0.65, 1, 1), (0.40, 0.5, 0.5, 0.6, 0.9, 0.9)),
        "HI": ((0.5, 0.65, 0.65, 0.8, 1, 1), (0.55, 0.65, 0.65, 0.75, 0.9, 0.9)),
        "VHI": ((0.65, 0.80, 0.80, 0.9, 1, 1), (0.7, 0.8, 0.8, 0.85, 0.9, 0.9)),
        "EHI": ((0.8, 0.9, 0.9, 1.0, 1, 1), (0.85, 0.9, 0.9, 0.95, 0.9, 0.9)),
    },
}

FULL_FORMS = {
    "VLR": "Very Low Relevance",
    "LR": "Low Relevance",
    "MR": "Medium Relevance",
    "HR": "High Relevance",
    "VHR": "Very High Relevance",
    "ELI": "Extremely Low Influence",
    "VLI": "Very Low Influence",
    "LI": "Low Influence",
    "MI": "Medium Influence",
    "HI": "High Influence",
    "VHI": "Very High Influence",
    "EHI": "Extremely High Influence",
}


def defuzz_it2(a):
    return cocoso_crisp_score(a)


def compute_total_relation_matrix(normalized_matrix):
    n = len(normalized_matrix)
    Z_4d = np.zeros((2, 2, n, n, 4))

    for i in range(n):
        for j in range(n):
            au, al = normalized_matrix[i][j]
            Z_4d[0, 0, i, j, :] = au[:4]
            Z_4d[0, 1, i, j, :2] = au[4:]
            Z_4d[1, 0, i, j, :] = al[:4]
            Z_4d[1, 1, i, j, :2] = al[4:]

    for i in range(2):
        for j in range(2):
            if j == 0:
                for k in range(4):
                    Z_component = Z_4d[i, j, :, :, k]
                    try:
                        T_component = Z_component @ np.linalg.pinv(np.eye(n) - Z_component)
                    except np.linalg.LinAlgError:
                        T_component = np.zeros((n, n))
                    Z_4d[i, j, :, :, k] = T_component

    T = [[zero_it2() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            T[i][j] = (
                (
                    Z_4d[0, 0, i, j, 0],
                    Z_4d[0, 0, i, j, 1],
                    Z_4d[0, 0, i, j, 2],
                    Z_4d[0, 0, i, j, 3],
                    Z_4d[0, 1, i, j, 0],
                    Z_4d[0, 1, i, j, 1],
                ),
                (
                    Z_4d[1, 0, i, j, 0],
                    Z_4d[1, 0, i, j, 1],
                    Z_4d[1, 0, i, j, 2],
                    Z_4d[1, 0, i, j, 3],
                    Z_4d[1, 1, i, j, 0],
                    Z_4d[1, 1, i, j, 1],
                ),
            )
    return T


def calculate_TI_TR(T):
    n = len(T)
    TI = [zero_it2() for _ in range(n)]
    TR = [zero_it2() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            TI[i] = add_it2(TI[i], T[i][j])
            TR[j] = add_it2(TR[j], T[i][j])
    return TI, TR


def wings_method_experts(strengths_list, influence_matrices_list, weights=None):
    n = len(strengths_list[0])
    num_experts = len(strengths_list)
    if weights is None:
        weights = [1.0 / num_experts] * num_experts

    avg_sidrm = [[zero_it2() for _ in range(n)] for _ in range(n)]
    for exp in range(num_experts):
        w = weights[exp]
        for i in range(n):
            avg_sidrm[i][i] = add_it2(avg_sidrm[i][i], scalar_mul_it2(w, strengths_list[exp][i]))
            for j in range(n):
                if i != j:
                    avg_sidrm[i][j] = add_it2(avg_sidrm[i][j], scalar_mul_it2(w, influence_matrices_list[exp][i][j]))

    s1U = s2U = s3U = s4U = s1L = s2L = s3L = s4L = 0.0
    for i in range(n):
        for j in range(n):
            au, al = avg_sidrm[i][j]
            s1U += au[0]; s2U += au[1]; s3U += au[2]; s4U += au[3]
            s1L += al[0]; s2L += al[1]; s3L += al[2]; s4L += al[3]
    s = s1U + s2U + s3U + s4U + s1L + s2L + s3L + s4L

    Z_mat = [[zero_it2() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            au, al = avg_sidrm[i][j]
            Z_mat[i][j] = (
                (au[0] / s if s else 0.0, au[1] / s if s else 0.0, au[2] / s if s else 0.0, au[3] / s if s else 0.0, au[4], au[5]),
                (al[0] / s if s else 0.0, al[1] / s if s else 0.0, al[2] / s if s else 0.0, al[3] / s if s else 0.0, al[4], al[5]),
            )

    T_mat = compute_total_relation_matrix(Z_mat)
    TI, TR = calculate_TI_TR(T_mat)
    engagement = [add_it2(TI[i], TR[i]) for i in range(n)]
    role = [sub_it2(TI[i], TR[i]) for i in range(n)]

    TI_defuzz = np.array([defuzz_it2(x) for x in TI], dtype=float)
    TR_defuzz = np.array([defuzz_it2(x) for x in TR], dtype=float)
    engagement_defuzz = np.array([defuzz_it2(x) for x in engagement], dtype=float)
    role_defuzz = np.array([defuzz_it2(x) for x in role], dtype=float)

    Ew = np.sqrt(np.square(engagement_defuzz) + np.square(role_defuzz))
    normalized_weights = Ew / np.sum(Ew) if float(np.sum(Ew)) > 0 else np.zeros_like(Ew)

    return {
        "average_sidrm": avg_sidrm,
        "scaling_factor": s,
        "normalized_matrix": Z_mat,
        "total_matrix": T_mat,
        "total_impact": TI,
        "total_receptivity": TR,
        "engagement": engagement,
        "role": role,
        "total_impact_defuzz": TI_defuzz,
        "total_receptivity_defuzz": TR_defuzz,
        "engagement_defuzz": engagement_defuzz,
        "role_defuzz": role_defuzz,
        "Ew": Ew,
        "normalized_weights": normalized_weights,
    }


def format_it2_df(mat, index, columns):
    df = pd.DataFrame(index=index, columns=columns)
    for i in range(len(index)):
        for j in range(len(columns)):
            df.iloc[i, j] = format_it2(mat[i][j])
    return df


def generate_flowchart_for_expert(expert_data, component_names, expert_idx=None):
    graph = graphviz.Digraph(comment=f"WINGS Expert {expert_idx+1}" if expert_idx is not None else "WINGS")
    graph.attr(rankdir="TD", size="8,8")
    for comp_idx, comp_name in enumerate(component_names):
        strength = expert_data["strengths_linguistic"][comp_idx]
        graph.node(comp_name, label=f"{comp_name} ({strength})", shape="box", style="rounded,filled", fillcolor="lightblue")
    for i, from_comp in enumerate(component_names):
        for j, to_comp in enumerate(component_names):
            if i == j:
                continue
            influence = expert_data["influence_matrix_linguistic"][i][j]
            if influence != "ELI":
                graph.edge(from_comp, to_comp, label=influence)
    return graph


def create_word_report(results, component_names, n_experts=1, expert_weights=None):
    doc = Document()
    title = doc.add_heading("IT2TrFS WINGS Analysis Report", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph(f"Number of experts: {n_experts}")
    if expert_weights and n_experts > 1:
        doc.add_paragraph("Expert weights: " + ", ".join([f"Expert {i+1}: {w:.4f}" for i, w in enumerate(expert_weights)]))

    table = doc.add_table(rows=1, cols=7)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    hdr[0].text = "Component"
    hdr[1].text = "Total Impact (TI)"
    hdr[2].text = "Total Receptivity (TR)"
    hdr[3].text = "TI + TR"
    hdr[4].text = "TI - TR"
    hdr[5].text = "E(wj)"
    hdr[6].text = "Normalized Weight"

    for i, name in enumerate(component_names):
        row = table.add_row().cells
        row[0].text = name
        row[1].text = f"{results['total_impact_defuzz'][i]:.6f}"
        row[2].text = f"{results['total_receptivity_defuzz'][i]:.6f}"
        row[3].text = f"{results['engagement_defuzz'][i]:.6f}"
        row[4].text = f"{results['role_defuzz'][i]:.6f}"
        row[5].text = f"{results['Ew'][i]:.6f}"
        row[6].text = f"{results['normalized_weights'][i]:.6f}"
    return doc


def wings_page():
    hero("IT2TrFS-WINGS", "Component interaction analysis with better input ergonomics, flowcharts, cause-effect view, and exports.")

    with st.expander("Linguistic references"):
        c1, c2 = st.columns(2)
        with c1:
            strength_df = pd.DataFrame(
                [{"Abbreviation": abbr, "Meaning": FULL_FORMS[abbr], "IT2TrFS": format_it2(it2)} for abbr, it2 in LINGUISTIC_TERMS["strength"].items()]
            )
            st.dataframe(strength_df, use_container_width=True, hide_index=True)
        with c2:
            influence_df = pd.DataFrame(
                [{"Abbreviation": abbr, "Meaning": FULL_FORMS[abbr], "IT2TrFS": format_it2(it2)} for abbr, it2 in LINGUISTIC_TERMS["influence"].items()]
            )
            st.dataframe(influence_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    n_components = c1.number_input("Number of components", min_value=2, max_value=20, value=3, step=1)
    n_experts = c2.number_input("Number of experts", min_value=1, max_value=10, value=1, step=1)

    component_names = []
    cols = st.columns(min(4, n_components))
    for i in range(n_components):
        with cols[i % len(cols)]:
            component_names.append(st.text_input(f"Component {i+1}", value=f"C{i+1}", key=f"wing_comp_{i}"))

    expert_weights = None
    if n_experts > 1:
        st.markdown("### Expert weights")
        weight_cols = st.columns(n_experts)
        weights = []
        for i in range(n_experts):
            with weight_cols[i]:
                weights.append(
                    st.number_input(
                        f"E{i+1}",
                        min_value=0.0,
                        max_value=1.0,
                        value=round(1 / n_experts, 4),
                        step=0.01,
                        key=f"wing_weight_{i}",
                    )
                )
        if not np.isclose(sum(weights), 1.0):
            st.error(f"Expert weights must sum to 1.0. Current sum: {sum(weights):.4f}")
            return
        expert_weights = weights

    state_key = "wings_experts_data"
    reset_needed = (
        state_key not in st.session_state
        or len(st.session_state[state_key]) != n_experts
        or st.session_state.get("wings_components_cache") != component_names
    )
    if reset_needed:
        st.session_state[state_key] = {}
        for e in range(n_experts):
            st.session_state[state_key][e] = {
                "strengths_linguistic": ["MR"] * n_components,
                "influence_matrix_linguistic": [["ELI" if i != j else "ELI" for j in range(n_components)] for i in range(n_components)],
            }
        st.session_state["wings_components_cache"] = component_names

    strengths_list, influence_list = [], []
    expert_tabs = st.tabs([f"Expert {e+1}" for e in range(n_experts)])

    for e, tab in enumerate(expert_tabs):
        with tab:
            st.markdown(f"### Expert {e+1}")
            st.markdown("**Component strengths / relevance**")
            strength_cols = st.columns(min(n_components, 4))
            strengths = []
            for i in range(n_components):
                with strength_cols[i % len(strength_cols)]:
                    cur = st.session_state[state_key][e]["strengths_linguistic"][i]
                    term = st.selectbox(
                        component_names[i],
                        options=list(LINGUISTIC_TERMS["strength"].keys()),
                        index=list(LINGUISTIC_TERMS["strength"].keys()).index(cur),
                        key=f"wings_strength_{e}_{i}",
                    )
                    st.session_state[state_key][e]["strengths_linguistic"][i] = term
                    strengths.append(LINGUISTIC_TERMS["strength"][term])

            st.markdown("**Influence matrix**")
            inf_mat = [[None] * n_components for _ in range(n_components)]

            head_cols = st.columns(n_components + 1)
            with head_cols[0]:
                st.markdown("**From → To**")
            for j in range(n_components):
                with head_cols[j + 1]:
                    st.markdown(f"**{component_names[j]}**")

            for i in range(n_components):
                row_cols = st.columns(n_components + 1)
                with row_cols[0]:
                    st.markdown(f"**{component_names[i]}**")
                for j in range(n_components):
                    with row_cols[j + 1]:
                        if i == j:
                            st.markdown("—")
                            inf_mat[i][j] = zero_it2()
                            st.session_state[state_key][e]["influence_matrix_linguistic"][i][j] = "ELI"
                        else:
                            cur = st.session_state[state_key][e]["influence_matrix_linguistic"][i][j]
                            term = st.selectbox(
                                f"{component_names[i]}→{component_names[j]}",
                                options=list(LINGUISTIC_TERMS["influence"].keys()),
                                index=list(LINGUISTIC_TERMS["influence"].keys()).index(cur),
                                key=f"wings_influence_{e}_{i}_{j}",
                                label_visibility="collapsed",
                            )
                            st.session_state[state_key][e]["influence_matrix_linguistic"][i][j] = term
                            inf_mat[i][j] = LINGUISTIC_TERMS["influence"][term]

            strengths_list.append(strengths)
            influence_list.append(inf_mat)
            st.graphviz_chart(generate_flowchart_for_expert(st.session_state[state_key][e], component_names, e), use_container_width=True)

    if st.button("Run WINGS analysis", type="primary", use_container_width=True):
        results = wings_method_experts(strengths_list, influence_list, weights=expert_weights)
        top_component = component_names[int(np.argsort(results["normalized_weights"])[::-1][0])] if len(component_names) else "—"

        tabs = st.tabs(["Results", "Cause-effect map", "Export"])

        with tabs[0]:
            c1, c2, c3 = st.columns(3)
            c1.metric("Top weighted component", top_component)
            c2.metric("Scaling factor", f"{results['scaling_factor']:.6f}")
            c3.metric("Components analysed", len(component_names))

            results_df = pd.DataFrame({
                "Component": component_names,
                "TI (Defuzzified)": results["total_impact_defuzz"],
                "TR (Defuzzified)": results["total_receptivity_defuzz"],
                "TI+TR": results["engagement_defuzz"],
                "TI-TR": results["role_defuzz"],
                "E(wj)": results["Ew"],
                "Normalized Weight": results["normalized_weights"],
            }).sort_values(by="Normalized Weight", ascending=False)

            class_df = pd.DataFrame({
                "Component": component_names,
                "Type": ["Cause" if x > 0 else "Effect" for x in results["role_defuzz"]],
                "TI-TR": results["role_defuzz"],
                "TI+TR": results["engagement_defuzz"],
                "E(wj)": results["Ew"],
                "Normalized Weight": results["normalized_weights"],
            }).sort_values(by="Normalized Weight", ascending=False)

            st.markdown("### Average SIDRM")
            st.dataframe(format_it2_df(results["average_sidrm"], component_names, component_names), use_container_width=True)
            st.markdown("### Final weights")
            st.dataframe(results_df.style.format(precision=6), use_container_width=True, hide_index=True)
            st.markdown("### Cause-effect classification")
            st.dataframe(class_df.style.format(precision=6), use_container_width=True, hide_index=True)

        with tabs[1]:
            plot_df = pd.DataFrame({
                "Component": component_names,
                "TI+TR": results["engagement_defuzz"],
                "TI-TR": results["role_defuzz"],
            })
            fig, ax = plt.subplots(figsize=(8.5, 6))
            for _, row in plot_df.iterrows():
                ax.scatter(row["TI+TR"], row["TI-TR"], s=120)
                ax.text(row["TI+TR"], row["TI-TR"], f" {row['Component']}", fontsize=10)
            ax.axhline(0, linestyle="--", linewidth=1)
            ax.axvline(plot_df["TI+TR"].mean(), linestyle="--", linewidth=1)
            ax.set_xlabel("TI + TR")
            ax.set_ylabel("TI - TR")
            ax.set_title("Cause-effect map")
            st.pyplot(fig)

        with tabs[2]:
            results_df = pd.DataFrame({
                "Component": component_names,
                "TI (Defuzzified)": results["total_impact_defuzz"],
                "TR (Defuzzified)": results["total_receptivity_defuzz"],
                "TI+TR": results["engagement_defuzz"],
                "TI-TR": results["role_defuzz"],
                "E(wj)": results["Ew"],
                "Normalized Weight": results["normalized_weights"],
            }).sort_values(by="Normalized Weight", ascending=False)
            csv_download_button(results_df, "Download WINGS results (CSV)", "wings_results.csv")
            docx_download_button(create_word_report(results, component_names, n_experts, expert_weights), "Download WINGS report (DOCX)", "wings_report.docx")


# =========================================================
# Home + main
# =========================================================

def overview_page():
    hero(APP_TITLE, "A cleaner, more professional Streamlit interface for IT2TrFS-Delphi, IT2TrFS-WINGS, and IT2TrFS-CoCoSo.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="section-card">
                <h4>Delphi</h4>
                <div class="muted">Upload or enter 1–5 ratings, aggregate them into IT2TrFS values, and screen criteria using crisp thresholds.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="section-card">
                <h4>WINGS</h4>
                <div class="muted">Model interdependencies, examine cause-effect behavior, and export results to CSV and DOCX.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="section-card">
                <h4>CoCoSo</h4>
                <div class="muted">Structure decision problems, collect expert linguistic evaluations, normalize, and rank alternatives.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        **Suggested package install**
        ```bash
        pip install streamlit pandas numpy matplotlib graphviz python-docx openpyxl
        ```
        **Run**
        ```bash
        streamlit run app.py
        ```
        """
    )


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    inject_css()

    with st.sidebar:
        st.title("IT2TrFS Studio")
        st.caption("Professional decision support app")
        page = st.radio("Choose module", ["Overview", "IT2TrFS-Delphi", "IT2TrFS-WINGS", "IT2TrFS-CoCoSo"], index=0)

    if page == "Overview":
        overview_page()
    elif page == "IT2TrFS-Delphi":
        delphi_page()
    elif page == "IT2TrFS-WINGS":
        wings_page()
    else:
        cocoso_page()


if __name__ == "__main__":
    main()
