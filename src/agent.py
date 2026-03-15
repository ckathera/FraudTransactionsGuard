"""
FraudGuard Agent — LangGraph Multi-Node Fraud Investigation Agent
-----------------------------------------------------------------
Graph flow:
  START
    → transaction_loader   (MCP: get_transaction_details)
    → fraud_scorer         (MCP: score_fraud → ML model)
        ↓ conditional route
    HIGH/MEDIUM → account_investigator (MCP: get_account_history + check_velocity)
               → pattern_analyzer     (RAG: fraud patterns + compliance rules)
               → decision_maker       (LLM: BLOCK / FLAG / APPROVE)
    LOW        → decision_maker       (skip investigation)
    → alert_writer  (LLM: final structured report)
  END
"""
import json
import os
import sys
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Allow running agent.py directly
sys.path.insert(0, os.path.dirname(__file__))

from mcp_server import (
    check_velocity,
    get_account_history,
    get_merchant_risk,
    get_transaction_details,
    score_fraud,
)
from rag_engine import retrieve

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="qwen/qwen3-32b")


# ── State ─────────────────────────────────────────────────────────────────────
class FraudAgentState(TypedDict):
    transaction_id: str
    transaction_data: dict
    fraud_score: dict
    risk_level: str                 # HIGH / MEDIUM / LOW
    account_history: dict
    velocity_data: dict
    merchant_risk: dict
    retrieved_context: str
    investigation_notes: str
    decision: str                   # BLOCK / FLAG / APPROVE
    final_report: str
    messages: Annotated[list, add_messages]


# ── Node 1: Transaction Loader ─────────────────────────────────────────────────
def transaction_loader(state: FraudAgentState) -> dict:
    """Fetch full transaction details via MCP tool."""
    raw = get_transaction_details(state["transaction_id"])
    tx_data = json.loads(raw)
    return {
        "transaction_data": tx_data,
        "messages": [{"role": "system", "content": f"Loaded transaction {state['transaction_id']}"}],
    }


# ── Node 2: Fraud Scorer ───────────────────────────────────────────────────────
def fraud_scorer(state: FraudAgentState) -> dict:
    """Run XGBoost + Isolation Forest scoring via MCP tool."""
    raw = score_fraud(state["transaction_id"])
    score = json.loads(raw)

    # Also get merchant risk for context
    merchant_cat = state["transaction_data"].get("merchant_category", "unknown")
    merchant_raw = get_merchant_risk(merchant_cat)
    merchant_risk = json.loads(merchant_raw)

    return {
        "fraud_score": score,
        "risk_level": score.get("risk_level", "LOW"),
        "merchant_risk": merchant_risk,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Fraud score: {score.get('fraud_probability', 0):.1%} — "
                    f"Risk: {score.get('risk_level')} — "
                    f"Merchant risk: {merchant_risk.get('risk')}"
                ),
            }
        ],
    }


# ── Node 3: Account Investigator ──────────────────────────────────────────────
def account_investigator(state: FraudAgentState) -> dict:
    """Fetch account history and velocity data via MCP tools."""
    account_id = state["transaction_data"].get("account_id", "")
    history_raw = get_account_history(account_id, limit=10)
    velocity_raw = check_velocity(account_id)

    history = json.loads(history_raw)
    velocity = json.loads(velocity_raw)

    notes = []
    if velocity.get("velocity_flags"):
        notes.extend(velocity["velocity_flags"])
    if history.get("fraud_count_in_history", 0) > 0:
        notes.append(f"Account has {history['fraud_count_in_history']} prior fraud transactions.")

    return {
        "account_history": history,
        "velocity_data": velocity,
        "investigation_notes": " | ".join(notes) if notes else "No immediate velocity flags.",
        "messages": [
            {
                "role": "system",
                "content": f"Account investigation complete. Flags: {len(velocity.get('velocity_flags', []))}",
            }
        ],
    }


# ── Node 4: Pattern Analyzer (RAG) ────────────────────────────────────────────
def pattern_analyzer(state: FraudAgentState) -> dict:
    """Retrieve relevant fraud patterns and compliance rules from vector store."""
    tx = state["transaction_data"]
    score = state["fraud_score"]
    risk = state["risk_level"]

    query = (
        f"{risk} risk fraud transaction, "
        f"merchant category {tx.get('merchant_category', '')}, "
        f"amount ${tx.get('amount', 0):.2f}, "
        f"distance {tx.get('distance_from_home_km', 0):.0f}km from home, "
        f"fraud probability {score.get('fraud_probability', 0):.0%}"
    )
    context = retrieve(query, k=4)

    return {
        "retrieved_context": context,
        "messages": [{"role": "system", "content": "RAG retrieval complete — fraud patterns loaded."}],
    }


# ── Node 5: Decision Maker ────────────────────────────────────────────────────
def decision_maker(state: FraudAgentState) -> dict:
    """LLM makes BLOCK / FLAG / APPROVE decision based on all gathered evidence."""
    tx = state["transaction_data"]
    score = state["fraud_score"]

    context_block = ""
    if state.get("retrieved_context"):
        context_block = f"""
Relevant Fraud Policies & Patterns:
{state['retrieved_context'][:1200]}
"""

    investigation_block = ""
    if state.get("investigation_notes"):
        investigation_block = f"""
Account Investigation Notes:
{state['investigation_notes']}
Velocity flags: {state.get('velocity_data', {}).get('velocity_flags', [])}
Prior fraud in account history: {state.get('account_history', {}).get('fraud_count_in_history', 0)}
"""

    prompt = f"""You are a senior bank fraud analyst. Analyze this transaction and make a final decision.

Transaction Details:
- ID: {tx.get('transaction_id')}
- Amount: ${tx.get('amount', 0):,.2f}
- Merchant: {tx.get('merchant_name')} ({tx.get('merchant_category')})
- Online: {'Yes' if tx.get('is_online') else 'No'}
- Time: {tx.get('hour', 0):02d}:00  |  Day: {tx.get('day_of_week')}
- Distance from home: {tx.get('distance_from_home_km', 0):.1f} km
- Transactions last 24h: {tx.get('num_transactions_24h')}
- Account age: {tx.get('account_age_days')} days
- Amount vs 30-day avg: {tx.get('amount_vs_avg_ratio', 1):.2f}x

ML Fraud Score:
- Fraud Probability: {score.get('fraud_probability', 0):.1%}
- Anomaly Score: {score.get('anomaly_score', 0):.3f}
- Risk Level: {score.get('risk_level')}
- Merchant Risk: {state.get('merchant_risk', {}).get('risk', 'UNKNOWN')}
{investigation_block}{context_block}
Based on the above evidence, respond with:
1. DECISION: [BLOCK / FLAG / APPROVE]
2. CONFIDENCE: [HIGH / MEDIUM / LOW]
3. REASON: 2-3 sentences explaining your decision.
4. RECOMMENDED_ACTIONS: 2-3 bullet points of what should happen next.

Be concise and decisive. Use the decision matrix from the investigation playbook.
"""
    response = llm.invoke(prompt)
    content = response.content

    # Parse decision keyword
    decision = "FLAG"  # default
    upper = content.upper()
    if "DECISION: BLOCK" in upper or "**BLOCK**" in upper:
        decision = "BLOCK"
    elif "DECISION: APPROVE" in upper or "**APPROVE**" in upper:
        decision = "APPROVE"
    elif "DECISION: FLAG" in upper or "**FLAG**" in upper:
        decision = "FLAG"

    return {
        "decision": decision,
        "messages": [{"role": "assistant", "content": content}],
    }


# ── Node 6: Alert Writer ──────────────────────────────────────────────────────
def alert_writer(state: FraudAgentState) -> dict:
    """Format final structured fraud alert report."""
    tx = state["transaction_data"]
    score = state["fraud_score"]
    decision = state["decision"]

    decision_colors = {"BLOCK": "🔴", "FLAG": "🟡", "APPROVE": "🟢"}
    icon = decision_colors.get(decision, "⚪")

    prompt = f"""Write a concise, professional fraud alert report for a bank operations team.

Transaction: {tx.get('transaction_id')} | Account: {tx.get('account_id')}
Amount: ${tx.get('amount', 0):,.2f} at {tx.get('merchant_name')} ({tx.get('merchant_category')})
Decision: {icon} {decision}
Fraud Probability: {score.get('fraud_probability', 0):.1%} | Risk: {score.get('risk_level')}

Analyst notes from investigation:
{(state['messages'][-1].content if hasattr(state['messages'][-1], 'content') else state['messages'][-1].get('content', '')) if state.get('messages') else 'N/A'}

Format as:
## {icon} FRAUD ALERT — {decision}
**Transaction:** [details]
**Risk Assessment:** [1 sentence]
**Decision Rationale:** [2 sentences]
**Immediate Actions Required:** [2-3 bullets]
**Case Reference:** [generate a fake reference number]

Keep it under 200 words. Professional tone. /no_think
"""
    response = llm.invoke(prompt)
    # Strip any residual <think>...</think> blocks just in case
    import re
    report = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
    return {
        "final_report": report,
        "messages": [{"role": "assistant", "content": report}],
    }


# ── Conditional Routing ───────────────────────────────────────────────────────
def route_by_risk(state: FraudAgentState) -> Literal["account_investigator", "decision_maker"]:
    """Route HIGH/MEDIUM risk to full investigation; LOW risk directly to decision."""
    if state["risk_level"] in ("HIGH", "MEDIUM"):
        return "account_investigator"
    return "decision_maker"


# ── Build Graph ───────────────────────────────────────────────────────────────
def build_agent() -> StateGraph:
    graph = StateGraph(FraudAgentState)

    graph.add_node("transaction_loader", transaction_loader)
    graph.add_node("fraud_scorer", fraud_scorer)
    graph.add_node("account_investigator", account_investigator)
    graph.add_node("pattern_analyzer", pattern_analyzer)
    graph.add_node("decision_maker", decision_maker)
    graph.add_node("alert_writer", alert_writer)

    graph.add_edge(START, "transaction_loader")
    graph.add_edge("transaction_loader", "fraud_scorer")
    graph.add_conditional_edges("fraud_scorer", route_by_risk)
    graph.add_edge("account_investigator", "pattern_analyzer")
    graph.add_edge("pattern_analyzer", "decision_maker")
    graph.add_edge("decision_maker", "alert_writer")
    graph.add_edge("alert_writer", END)

    return graph.compile()


agent = build_agent()


def run(transaction_id: str) -> FraudAgentState:
    """Run the fraud investigation agent for a given transaction ID."""
    result = agent.invoke({"transaction_id": transaction_id})
    return result


if __name__ == "__main__":
    import sys
    txn_id = sys.argv[1] if len(sys.argv) > 1 else "TXN00001"
    output = run(txn_id)
    print("\n" + "=" * 60)
    print(output["final_report"])
    print(f"\nDecision: {output['decision']}")
    print(f"Fraud Probability: {output['fraud_score'].get('fraud_probability', 0):.1%}")
