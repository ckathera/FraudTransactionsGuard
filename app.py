"""
FraudGuard AI — Streamlit Dashboard
Bank Transaction Fraud Detection powered by LangGraph + XGBoost + RAG + MCP

Run: streamlit run app.py
"""
import json
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.decision-block  { background:#fde8e8; border-left:6px solid #e53e3e; padding:16px; border-radius:8px; }
.decision-flag   { background:#fef3cd; border-left:6px solid #d69e2e; padding:16px; border-radius:8px; }
.decision-approve{ background:#e8f5e9; border-left:6px solid #38a169; padding:16px; border-radius:8px; }
.metric-card { background:#f7fafc; border-radius:10px; padding:12px; text-align:center; }
h1 { color: #1a202c; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bank.png", width=60)
    st.title("FraudGuard AI")
    st.caption("Powered by LangGraph · XGBoost · RAG · MCP")
    st.divider()
    st.markdown("### How it works")
    st.markdown("""
1. **Transaction Loader** — fetches tx details via MCP
2. **Fraud Scorer** — XGBoost + Isolation Forest ML
3. **Account Investigator** — velocity & history check
4. **Pattern Analyzer** — RAG from fraud policy docs
5. **Decision Maker** — LLM final BLOCK/FLAG/APPROVE
6. **Alert Writer** — generates structured report
    """)
    st.divider()
    st.markdown("### Tech Stack")
    st.markdown("🤖 `qwen3-32b` via Groq")
    st.markdown("🌲 XGBoost + Isolation Forest")
    st.markdown("📚 FAISS + sentence-transformers")
    st.markdown("🔧 FastMCP (5 tools)")
    st.markdown("🕸️ LangGraph StateGraph")


# ── Main ───────────────────────────────────────────────────────────────────────
st.title("🛡️ FraudGuard — Transaction Intelligence System")
st.caption("Agentic AI fraud detection with real-time investigation and policy-aware decisions")

# Load available transactions
@st.cache_data
def load_transactions():
    path = os.path.join(os.path.dirname(__file__), "data", "transactions.csv")
    return pd.read_csv(path)


@st.cache_resource
def get_agent():
    from agent import build_agent
    return build_agent()


try:
    df = load_transactions()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    st.warning("⚠️ No transaction data found. Run `python src/generate_data.py` first.")

# ── Transaction Selector ───────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader("Select Transaction")
    if data_loaded:
        # Offer quick picks
        fraud_txns = df[df["fraud"] == 1]["transaction_id"].tolist()
        legit_txns = df[df["fraud"] == 0]["transaction_id"].tolist()

        tab1, tab2, tab3 = st.tabs(["🔴 Fraud Samples", "🟢 Legit Samples", "🔍 Enter ID"])

        with tab1:
            selected_fraud = st.selectbox("Known fraud transactions:", fraud_txns[:20])
            if st.button("🔍 Investigate (Fraud)", key="btn_fraud", type="primary"):
                st.session_state["selected_txn"] = selected_fraud

        with tab2:
            selected_legit = st.selectbox("Known legitimate transactions:", legit_txns[:20])
            if st.button("🔍 Investigate (Legit)", key="btn_legit"):
                st.session_state["selected_txn"] = selected_legit

        with tab3:
            manual_id = st.text_input("Transaction ID (e.g. TXN00042):", value="TXN00001")
            if st.button("🔍 Investigate", key="btn_manual"):
                st.session_state["selected_txn"] = manual_id
    else:
        st.info("Load data first.")

# ── Analysis Panel ─────────────────────────────────────────────────────────────
if "selected_txn" in st.session_state and data_loaded:
    txn_id = st.session_state["selected_txn"]
    st.divider()
    st.subheader(f"Investigating: `{txn_id}`")

    # Show raw transaction details
    row = df[df["transaction_id"] == txn_id]
    if not row.empty:
        tx = row.iloc[0].to_dict()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Amount", f"${tx['amount']:,.2f}")
        c2.metric("Merchant", tx['merchant_name'])
        c3.metric("Category", tx['merchant_category'])
        c4.metric("Online", "Yes" if tx['is_online'] else "No")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Hour", f"{int(tx['hour']):02d}:00")
        c6.metric("Distance from Home", f"{tx['distance_from_home_km']:.1f} km")
        c7.metric("Tx in 24h", int(tx['num_transactions_24h']))
        c8.metric("Account Age", f"{int(tx['account_age_days'])} days")

    st.divider()

    # Run agent with progress
    with st.status("Running FraudGuard Agent...", expanded=True) as status:
        st.write("🔌 Connecting to MCP tools...")
        st.write("🤖 Loading LangGraph agent...")
        agent = get_agent()

        st.write("📦 Node 1 — Transaction Loader")
        st.write("🧠 Node 2 — Fraud Scorer (XGBoost + Isolation Forest)")

        try:
            result = agent.invoke({"transaction_id": txn_id})
            status.update(label="Investigation complete!", state="complete", expanded=False)
        except Exception as e:
            status.update(label=f"Error: {e}", state="error")
            st.error(str(e))
            st.stop()

    # ── Results ────────────────────────────────────────────────────────────────
    score = result.get("fraud_score", {})
    decision = result.get("decision", "FLAG")
    prob = score.get("fraud_probability", 0)
    risk = result.get("risk_level", "MEDIUM")

    st.subheader("Results")
    r1, r2, r3 = st.columns(3)

    # Fraud probability gauge
    with r1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            title={"text": "Fraud Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e53e3e" if prob > 0.7 else "#d69e2e" if prob > 0.4 else "#38a169"},
                "steps": [
                    {"range": [0, 40], "color": "#e8f5e9"},
                    {"range": [40, 70], "color": "#fef3cd"},
                    {"range": [70, 100], "color": "#fde8e8"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": prob * 100},
            },
            number={"suffix": "%"},
        ))
        fig.update_layout(height=250, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    # Decision badge
    with r2:
        st.markdown("#### Decision")
        if decision == "BLOCK":
            st.markdown('<div class="decision-block"><h2>🔴 BLOCK</h2><p>Transaction blocked. Card frozen pending verification.</p></div>', unsafe_allow_html=True)
        elif decision == "FLAG":
            st.markdown('<div class="decision-flag"><h2>🟡 FLAG</h2><p>Flagged for manual review. Customer notification sent.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="decision-approve"><h2>🟢 APPROVE</h2><p>Transaction approved. Logged for pattern analysis.</p></div>', unsafe_allow_html=True)

        st.metric("Anomaly Score", f"{score.get('anomaly_score', 0):.3f}", help="0=normal, 1=highly anomalous")
        st.metric("Risk Level", risk)

    # Feature importance
    with r3:
        st.markdown("#### Top Risk Factors")
        factors = score.get("top_risk_factors", {})
        if factors:
            factor_df = pd.DataFrame(
                list(factors.items()), columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)
            fig2 = go.Figure(go.Bar(
                x=factor_df["Importance"],
                y=factor_df["Feature"],
                orientation="h",
                marker_color="#4299e1",
            ))
            fig2.update_layout(height=230, margin=dict(t=10, b=10, l=10, r=10), xaxis_title="Importance")
            st.plotly_chart(fig2, use_container_width=True)

    # Final report
    st.divider()
    st.subheader("📋 Fraud Alert Report")
    st.markdown(result.get("final_report", "No report generated."))

    # Agent trace
    with st.expander("🕵️ Agent Investigation Trace"):
        for i, msg in enumerate(result.get("messages", [])):
            role = msg.get("role", "system") if isinstance(msg, dict) else getattr(msg, "role", "system")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if role == "system":
                st.info(f"**Step {i+1}:** {content}")
            else:
                st.markdown(f"**Analyst:** {content[:500]}...")

    # Account history table
    if result.get("account_history") and result["account_history"].get("transactions"):
        with st.expander("📊 Account Transaction History"):
            hist_df = pd.DataFrame(result["account_history"]["transactions"])
            hist_df["fraud"] = hist_df["fraud"].map({1: "🔴 Fraud", 0: "🟢 Legit"})
            st.dataframe(hist_df, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("FraudGuard AI · Built with LangGraph · XGBoost · FAISS · FastMCP · Groq")
