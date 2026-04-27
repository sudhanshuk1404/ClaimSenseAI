"""ClaimSense AI — Streamlit Frontend.

Run:
    streamlit run streamlit_app.py

Requires the FastAPI backend to be running at http://localhost:8000
    python main.py
"""

from __future__ import annotations

import json
import time
from typing import Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000/api/v1"
st.set_page_config(
    page_title="ClaimSense AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Minimal CSS — clean, professional, no heavy theming
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label { font-size: 0.78rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #1e293b; margin-top: 0.2rem; }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-green  { background: #dcfce7; color: #166534; }
    .badge-red    { background: #fee2e2; color: #991b1b; }
    .badge-yellow { background: #fef9c3; color: #854d0e; }
    .badge-high   { background: #fee2e2; color: #991b1b; }
    .badge-medium { background: #fef9c3; color: #854d0e; }
    .badge-low    { background: #f1f5f9; color: #475569; }
    .cluster-card {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        background: #fff;
    }
    .cluster-card.top-pick { border: 2px solid #f59e0b; background: #fffbeb; }
    .section-header { font-size: 0.9rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem; }
    .evidence-item { padding: 0.35rem 0; border-bottom: 1px solid #f1f5f9; font-size: 0.88rem; color: #374151; }
    .evidence-item:last-child { border-bottom: none; }
    hr.divider { border: none; border-top: 1px solid #e2e8f0; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _api_get(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the API. Make sure the backend is running: `python main.py`")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.json().get('detail', str(e))}")
        st.stop()


def _api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the API. Make sure the backend is running: `python main.py`")
        st.stop()
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e))
        st.error(f"API error {e.response.status_code}: {detail}")
        st.stop()


def _health() -> dict | None:
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reusable UI components
# ---------------------------------------------------------------------------

def _recoverability_badge(value: str) -> str:
    colour = {"recoverable": "green", "not_recoverable": "red", "needs_review": "yellow"}.get(value, "yellow")
    label  = value.replace("_", " ").title()
    return f'<span class="badge badge-{colour}">{label}</span>'


def _priority_badge(value: str) -> str:
    colour = {"high": "high", "medium": "medium", "low": "low"}.get(value.lower(), "low")
    return f'<span class="badge badge-{colour}">{value.upper()}</span>'


def _metric_card(label: str, value: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>"""


def _render_root_cause(analysis: dict):
    """Render Problem 1 results."""
    st.markdown("#### 🔍 Root Cause Analysis")

    # Top metrics
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            _metric_card("Claim Amount", f"${analysis['recoverability_rationale'] and analysis.get('root_cause_analysis',{}).get('confidence_score',0) or 0:.0%}"),
            unsafe_allow_html=True,
        )

    recov = analysis["recoverability"]
    confidence = analysis["confidence_score"]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Recoverability** &nbsp; {_recoverability_badge(recov)}", unsafe_allow_html=True)
    with col2:
        st.metric("Confidence", f"{confidence:.0%}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Root cause
    st.markdown('<div class="section-header">Root Cause</div>', unsafe_allow_html=True)
    st.info(analysis["denial_root_cause"])

    # CARC interpretation
    st.markdown('<div class="section-header">CARC Interpretation</div>', unsafe_allow_html=True)
    st.write(analysis["carc_interpretation"])

    # Recoverability rationale
    st.markdown('<div class="section-header">Recoverability Rationale</div>', unsafe_allow_html=True)
    st.write(analysis["recoverability_rationale"])

    # Supporting evidence
    st.markdown('<div class="section-header">Supporting Evidence</div>', unsafe_allow_html=True)
    evidence_html = "".join(
        f'<div class="evidence-item">📌 {e}</div>'
        for e in analysis.get("supporting_evidence", [])
    )
    st.markdown(evidence_html, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Recommended action
    st.markdown('<div class="section-header">Recommended Action</div>', unsafe_allow_html=True)
    rec_color = {"recoverable": "success", "not_recoverable": "error", "needs_review": "warning"}.get(recov, "info")
    getattr(st, rec_color)(f"➡️ {analysis['recommended_action']}")


def _render_pattern(pattern: dict):
    """Render Problem 2 results."""
    st.markdown("#### 🔄 Historical Pattern Matching")

    appeal_rate = pattern.get("historical_appeal_success_rate", 0)
    similar     = pattern.get("similar_claims", [])
    systemic    = pattern.get("systemic_pattern")

    # Appeal rate + similar count
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Historical Appeal Success Rate", f"{appeal_rate:.0%}",
                  help="Based on similar paid vs denied claims in the dataset")
    with col2:
        st.metric("Similar Claims Found", len(similar))

    # Systemic pattern
    if systemic:
        st.warning(f"⚠️ **Systemic Pattern Detected:** {systemic}")

    # Similar claims table
    if similar:
        st.markdown('<div class="section-header">Top Similar Claims</div>', unsafe_allow_html=True)
        rows = []
        for s in similar:
            outcome_icon = "✅" if s["outcome"] == "paid" else "❌"
            rows.append({
                "Claim ID": s["claim_id"],
                "Outcome": f"{outcome_icon} {s['outcome'].title()}",
                "Similarity": f"{s['similarity_score']:.1%}",
                "Payer": s.get("payer_name") or "—",
                "Procedure": s.get("procedure_code") or "—",
                "Match Reasons": " · ".join(s.get("match_reasons", [])[:2]),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # LLM narrative
    st.markdown('<div class="section-header">Pattern Analysis</div>', unsafe_allow_html=True)
    st.write(pattern.get("pattern_analysis", "—"))


def _render_cluster_card(cluster: dict, is_top: bool):
    """Render a single denial cluster."""
    priority = cluster.get("priority", "medium")
    top_label = "⭐ Top Opportunity  " if is_top else ""
    card_class = "cluster-card top-pick" if is_top else "cluster-card"

    header_html = f"""
    <div class="{card_class}">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.7rem;">
            <span style="font-size:1rem; font-weight:700; color:#1e293b;">{top_label}{cluster['label']}</span>
            {_priority_badge(priority)}
        </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    # Metrics row
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Claims", len(cluster["claim_ids"]))
    with mc2:
        st.metric("Denied Amount", f"${cluster['total_denied_amount']:,.0f}")
    with mc3:
        st.metric("Payer", cluster.get("payer") or "Mixed")
    with mc4:
        rate = cluster.get("historical_appeal_success_rate")
        st.metric("Appeal Rate", f"{rate:.0%}" if rate is not None else "—")

    # Summary
    st.markdown(f'<div style="margin-top:0.6rem; color:#374151; font-size:0.9rem;">{cluster["summary"]}</div>', unsafe_allow_html=True)

    # Action
    st.markdown(
        f'<div style="margin-top:0.7rem; padding:0.6rem 0.9rem; background:#f0fdf4; border-left:3px solid #22c55e; border-radius:4px; font-size:0.88rem; color:#15803d;"><strong>Action:</strong> {cluster["recommended_action"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")   # spacing


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital-2.png", width=60)
    st.markdown("## ClaimSense AI")
    st.markdown("AI-powered healthcare claim denial analysis")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠  Overview", "🔍  Analyze a Claim", "📊  Batch Intelligence"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Live health indicator
    health = _health()
    if health:
        st.success(f"✅ API online · {health['denied_claims']} denied claims loaded")
    else:
        st.error("❌ API offline")
        st.code("python main.py", language="bash")


# ===========================================================================
# PAGE 1 — Overview
# ===========================================================================

if page == "🏠  Overview":
    st.title("🏥 ClaimSense AI")
    st.markdown("#### AI-powered healthcare claim denial analysis for Revenue Cycle Management")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔍 Problem 1 + 2
        **Claim Analysis**

        Submit a denied claim (EDI 835 + 837) and get:
        - Root cause of the denial
        - Recoverability verdict with confidence
        - Supporting evidence from actual claim fields
        - Historical similar claims
        - Appeal strategy recommendation
        """)

    with col2:
        st.markdown("""
        ### 📊 Problem 3
        **Batch Intelligence**

        Cluster all denied claims and get:
        - Denial clusters by payer + CARC code
        - Dollar value at risk per cluster
        - Estimated appeal success rates
        - Actionable billing team summaries
        - Top opportunity cluster highlighted
        """)

    with col3:
        st.markdown("""
        ### ⚙️ How it works
        **Data flow:**
        1. EDI 835 (remittance) joined with EDI 837 (original claim)
        2. CARC code enriched with domain context
        3. GPT-4o reasons about root cause & recoverability
        4. Embeddings + structured matching find similar claims
        5. K-means clusters denials for batch prioritisation
        """)

    st.markdown("---")

    # Dataset stats
    if health:
        st.markdown("### 📈 Loaded Dataset")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Claims", health["total_claims"])
        c2.metric("Denied", health["denied_claims"])
        c3.metric("Paid (historical)", health["paid_claims"])
        c4.metric("LLM Model", health["model"])

    st.markdown("---")
    st.markdown("**API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)")


# ===========================================================================
# PAGE 2 — Analyze a Claim  (Problem 1 + 2)
# ===========================================================================

elif page == "🔍  Analyze a Claim":
    st.title("🔍 Analyze a Denied Claim")
    st.markdown("Select a claim from the dataset or paste your own 835+837 JSON. Runs **Problem 1** (root cause) + **Problem 2** (pattern matching) together.")
    st.markdown("---")

    # ---- Input method ----
    input_mode = st.radio(
        "Input method",
        ["Select from dataset", "Paste custom JSON"],
        horizontal=True,
    )

    claim_payload: dict | None = None

    if input_mode == "Select from dataset":
        # Fetch denied claims list
        with st.spinner("Loading claims..."):
            claims_list = _api_get("/claims", params={"outcome": "denied"})

        if not claims_list:
            st.warning("No denied claims found in the dataset.")
            st.stop()

        # Build display options
        options = {
            f"{c['claim_id']} — {c['payer'] or '?'} · CPT {c['procedure_code'] or '?'} · CARC {c['carc_code'] or '?'} · ${c['claim_amount']:,.0f}": c
            for c in claims_list
        }

        selected_label = st.selectbox("Select a denied claim", list(options.keys()))
        selected = options[selected_label]

        # Show quick claim info
        info_cols = st.columns(5)
        info_cols[0].metric("Claim ID",     selected["claim_id"])
        info_cols[1].metric("Payer",        selected.get("payer") or "—")
        info_cols[2].metric("Procedure",    selected.get("procedure_code") or "—")
        info_cols[3].metric("CARC Code",    selected.get("carc_code") or "—")
        info_cols[4].metric("Denied",       f"${selected['claim_amount']:,.0f}")

        # Build payload from the raw dataset
        raw_claims = _api_get("/claims")  # all claims (to get the full 835/837 data)
        # For dataset claims we fetch the raw claim detail via the public endpoint
        # We re-build the payload from the stored synthetic data via the full claim list
        # Since we don't expose a GET /claims/{id} endpoint anymore, we load from file
        import json as _json
        from pathlib import Path
        raw_data = _json.loads((Path(__file__).parent / "data" / "synthetic_claims.json").read_text())
        raw_map = {c["claim_id"]: c for c in raw_data["claims"]}
        raw = raw_map.get(selected["claim_id"])
        if raw:
            claim_payload = {"edi835": raw["edi835"], "edi837": raw["edi837"]}

    else:
        st.markdown("Paste a JSON object with `edi835` and `edi837` keys:")
        default_example = {
            "edi835": {
                "pc_ClaimID": "CLM-2026-00142",
                "pc_ClaimStatus": "4",
                "pc_ClaimAmount": 4500.00,
                "pc_ClaimPaid": 0.00,
                "pc_InsuranceType": "Commercial",
                "pc_ReceivedDate": "2026-03-20",
                "cp_PayerName": "Blue Cross Blue Shield",
                "pcla_AdjustmentGroup": "CO",
                "pcla_AdjustmentReason": "29",
                "pcla_AdjustmentAmount": 4500.00,
                "pcl_ProcedureCode": "99214",
            },
            "edi837": {
                "ec_ClaimNo": "CLM-2026-00142",
                "ec_PayerName": "Blue Cross Blue Shield",
                "ec_InsuranceType": "Commercial",
                "ec_ServiceDateFrom": "2025-06-15",
                "ec_PrincipalDiagnosis": "J06.9",
                "ec_BillProvNPI": "1234567890",
                "ec_DelayReasonCode": "",
                "ec_ClaimFrequency": "1",
            },
        }
        raw_text = st.text_area("Claim JSON", value=json.dumps(default_example, indent=2), height=320)
        try:
            claim_payload = json.loads(raw_text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            claim_payload = None

    # ---- Run Analysis ----
    st.markdown("---")
    if claim_payload and st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Running root cause analysis + pattern matching (this takes ~10–20 seconds)..."):
            t0 = time.time()
            result = _api_post("/claims/analyze", claim_payload)
            elapsed = time.time() - t0

        st.success(f"✅ Analysis complete in {elapsed:.1f}s · Estimated cost: ${result.get('estimated_cost_usd', 0):.4f}")
        st.markdown("---")

        tab1, tab2 = st.tabs(["🔍 Problem 1 — Root Cause", "🔄 Problem 2 — Pattern Match"])

        with tab1:
            _render_root_cause(result["root_cause_analysis"])

        with tab2:
            if result.get("pattern_match"):
                _render_pattern(result["pattern_match"])
            else:
                st.info("Pattern matching was not run for this request.")


# ===========================================================================
# PAGE 3 — Batch Intelligence  (Problem 3)
# ===========================================================================

elif page == "📊  Batch Intelligence":
    st.title("📊 Batch Denial Intelligence")
    st.markdown("Clusters all denied claims by payer + CARC code, estimates appeal rates, and surfaces the highest-value recovery opportunity.")
    st.markdown("---")

    col_btn, col_refresh = st.columns([3, 1])
    with col_btn:
        run = st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True)
    with col_refresh:
        force_refresh = st.checkbox("Force refresh", value=False, help="Re-run even if cached results exist")

    if run:
        with st.spinner("Clustering denied claims and generating batch intelligence report (30–60 seconds)..."):
            t0 = time.time()
            report = _api_get("/batch/cluster", params={"refresh": str(force_refresh).lower()})
            elapsed = time.time() - t0

        st.success(f"✅ Report ready in {elapsed:.1f}s")
        st.markdown("---")

        # ---- Summary metrics ----
        st.markdown("### Executive Summary")
        st.info(report["executive_summary"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Denied Claims Analyzed", report["total_claims_analyzed"])
        m2.metric("Total Denied Amount",     f"${report['total_denied_amount']:,.0f}")
        m3.metric("Clusters Found",          len(report["clusters"]))
        top_cluster = next(
            (c for c in report["clusters"] if c["cluster_id"] == report["top_opportunity_cluster_id"]),
            None,
        )
        if top_cluster:
            m4.metric("Top Opportunity", f"${top_cluster['total_denied_amount']:,.0f}")

        st.markdown("---")

        # ---- Cluster cards ----
        st.markdown("### Denial Clusters")

        # Sort: top opportunity first, then by denied amount
        top_id = report["top_opportunity_cluster_id"]
        clusters = sorted(
            report["clusters"],
            key=lambda c: (c["cluster_id"] != top_id, -c["total_denied_amount"])
        )

        for cluster in clusters:
            is_top = cluster["cluster_id"] == top_id
            _render_cluster_card(cluster, is_top)

    else:
        # Placeholder when no analysis has been run yet
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#94a3b8;">
            <div style="font-size:3rem;">📊</div>
            <div style="font-size:1.1rem; margin-top:0.5rem;">Click <strong>Run Batch Analysis</strong> to cluster all denied claims.</div>
            <div style="font-size:0.85rem; margin-top:0.5rem;">Results are cached — subsequent runs are instant unless you force refresh.</div>
        </div>
        """, unsafe_allow_html=True)
