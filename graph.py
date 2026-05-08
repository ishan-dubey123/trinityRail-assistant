from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from typing import TypedDict, Optional
import json

from database import run_query
from rag import search_documents
from confidence import calculate_confidence, should_trigger_human_checkpoint

# ============================================================
# 1. STATE
# ============================================================
class AgentState(TypedDict):
    question: str
    route: Optional[str]
    sql_query: Optional[str]
    sql_result: Optional[dict]
    rag_result: Optional[dict]
    confidence: Optional[dict]
    checkpoint: Optional[dict]
    awaiting_human: Optional[bool]
    human_decision: Optional[str]
    final_answer: Optional[str]
    follow_ups: Optional[list]

# ============================================================
# 2. LLM (Groq free tier)
# ============================================================
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ============================================================
# 3. ROUTER
# ============================================================
def router_node(state: AgentState) -> AgentState:
    print(f"\n🔀 ROUTER: {state['question']}")
    prompt = f"""
You are a routing assistant for a railcar company.
Classify into exactly one: "sql", "rag", or "both".
- sql: needs numbers, counts, or database data.
- rag: needs policy rules or documents.
- both: needs both data and policy.
Question: {state['question']}
Reply with only one word.
"""
    route = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    if route not in ["sql", "rag", "both"]:
        route = "both"
    print(f"   → Route: {route}")
    return {**state, "route": route}

# ============================================================
# 4. SQL AGENT – STRICT PROMPT WITH EXAMPLES
# ============================================================
def sql_agent_node(state: AgentState) -> AgentState:
    print(f"\n📊 SQL AGENT")
    question = state["question"]

    schema_prompt = f"""
You are a SQL expert for TrinityRail. Write ONLY a SELECT query. No markdown, no backticks, no extra words.

Schema:
- railcars: car_id (int), car_type ('tank','boxcar','flatcar'), status ('idle','leased','maintenance'), region ('north','south','east','west'), last_inspection_date (text YYYY-MM-DD), commodity (text)
- leases: lease_id (int), car_id (int), customer_name (text), start_date (text), end_date (text), monthly_rate (real)

CRITICAL RULES – FOLLOW EXACTLY:
1. Add a WHERE clause ONLY if the user explicitly mentions a condition.
2. Do NOT add car_type, region, commodity, or any other column unless the user names it.
3. For "how many cars are idle?" → SELECT COUNT(car_id) FROM railcars WHERE status='idle'
4. For "list idle cars" → SELECT car_id, car_type, region FROM railcars WHERE status='idle'
5. For "how many tank cars are idle?" → SELECT COUNT(car_id) FROM railcars WHERE car_type='tank' AND status='idle'
6. For "overdue for maintenance" → SELECT car_id, last_inspection_date FROM railcars WHERE last_inspection_date < date('now', '-90 days')
   (Do NOT add status='maintenance' unless the user asks for maintenance status specifically.)
7. Use date('now') for today. Use date('now', '-90 days') for 90 days ago.

Question: {question}

SQL query:
"""
    response = llm.invoke([HumanMessage(content=schema_prompt)])
    sql_query = response.content.strip()
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    print(f"   → SQL: {sql_query}")

    sql_result = run_query(sql_query)
    print(f"   → Rows: {sql_result.get('row_count', 0)}")
    return {**state, "sql_query": sql_query, "sql_result": sql_result}

# ============================================================
# 5. RAG AGENT
# ============================================================
def rag_agent_node(state: AgentState) -> AgentState:
    print(f"\n📄 RAG AGENT")
    rag_result = search_documents(state["question"], top_k=3, threshold=0.7)
    print(f"   → Found {rag_result['count']} docs, best score {rag_result['best_score']}")
    return {**state, "rag_result": rag_result}

# ============================================================
# 6. CONFIDENCE
# ============================================================
def confidence_node(state: AgentState) -> AgentState:
    print(f"\n🎯 CONFIDENCE")
    confidence = calculate_confidence(
        sql_result=state.get("sql_result"),
        rag_result=state.get("rag_result"),
        question=state["question"]
    )
    checkpoint = should_trigger_human_checkpoint(
        sql_result=state.get("sql_result"),
        confidence=confidence,
        question=state["question"]
    )
    print(f"   → {confidence['level']} ({confidence['score']}/100), checkpoint: {checkpoint['trigger']}")
    return {
        **state,
        "confidence": confidence,
        "checkpoint": checkpoint,
        "awaiting_human": checkpoint["trigger"]
    }

# ============================================================
# 7. HUMAN CHECKPOINT
# ============================================================
def human_checkpoint_node(state: AgentState) -> AgentState:
    print(f"\n⏸️  HUMAN CHECKPOINT")
    return state

# ============================================================
# 8. SYNTHESIZER – handles COUNT and empty results
# ============================================================
def synthesizer_node(state: AgentState) -> AgentState:
    print(f"\n✍️  SYNTHESIZER")
    question = state["question"]
    sql_result = state.get("sql_result")
    rag_result = state.get("rag_result")

    # Simplify COUNT results
    if sql_result and sql_result.get("success") and sql_result.get("row_count", 0) == 1:
        first_row = sql_result["results"][0]
        keys = list(first_row.keys())
        if len(keys) == 1 and any(w in keys[0].lower() for w in ["count", "total"]):
            sql_result["results"] = [{"count": first_row[keys[0]]}]

    # Build context
    context_parts = []
    if sql_result and sql_result.get("success") and sql_result.get("row_count", 0) > 0:
        rows = sql_result["results"][:10]
        context_parts.append(f"DATABASE RESULTS ({sql_result['row_count']} rows):\n{json.dumps(rows, indent=2)}")
    elif sql_result and sql_result.get("success") and sql_result.get("row_count", 0) == 0:
        context_parts.append("DATABASE RESULTS: No matching records found.")

    if rag_result and rag_result.get("count", 0) > 0:
        docs = [item["document"] for item in rag_result["matched"]]
        context_parts.append(f"POLICY DOCUMENTS:\n" + "\n".join(f"- {d}" for d in docs))

    if not context_parts:
        context = "No data or policy documents found."
    else:
        context = "\n\n".join(context_parts)

    synthesis_prompt = f"""
You are a helpful assistant for TrinityRail.
Answer the user's question using ONLY the context below.
Be concise, clear, and direct.
If the context says "No matching records found", answer "None found" or "Zero".
Never invent information.

Context:
{context}

User Question: {question}

Answer:
"""
    answer = llm.invoke([HumanMessage(content=synthesis_prompt)]).content.strip()

    # Follow-ups
    follow_prompt = f"""
Suggest exactly 2 short follow-up questions for: "{question}"
Return ONLY JSON array, e.g. ["q1", "q2"]
"""
    try:
        follow_ups = json.loads(llm.invoke([HumanMessage(content=follow_prompt)]).content.strip())
    except:
        follow_ups = ["Which region has the most idle cars?", "Which cars are overdue for inspection?"]

    print(f"   → Answer: {answer[:100]}...")
    return {**state, "final_answer": answer, "follow_ups": follow_ups}

# ============================================================
# 9. CONDITIONAL EDGES
# ============================================================
def route_decision(state: AgentState) -> str:
    r = state.get("route", "both")
    if r == "sql":
        return "sql_only"
    elif r == "rag":
        return "rag_only"
    return "both"

def checkpoint_decision(state: AgentState) -> str:
    return "needs_human" if state.get("awaiting_human") else "skip_human"

def human_decision(state: AgentState) -> str:
    return "loop_back" if state.get("human_decision") == "refine" else "proceed"

# ============================================================
# 10. BUILD GRAPH
# ============================================================
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("confidence", confidence_node)
    graph.add_node("human_checkpoint", human_checkpoint_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision, {
        "sql_only": "sql_agent",
        "rag_only": "rag_agent",
        "both": "sql_agent"
    })
    graph.add_edge("sql_agent", "rag_agent")
    graph.add_edge("rag_agent", "confidence")
    graph.add_conditional_edges("confidence", checkpoint_decision, {
        "needs_human": "human_checkpoint",
        "skip_human": "synthesizer"
    })
    graph.add_conditional_edges("human_checkpoint", human_decision, {
        "proceed": "synthesizer",
        "loop_back": "router"
    })
    graph.add_edge("synthesizer", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory, interrupt_before=["human_checkpoint"])

trinity_graph = build_graph()

# ============================================================
# 11. API HELPERS
# ============================================================
def ask_agent(question: str, thread_id: str = "default") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    initial = {
        "question": question, "route": None, "sql_query": None, "sql_result": None,
        "rag_result": None, "confidence": None, "checkpoint": None, "awaiting_human": False,
        "human_decision": None, "final_answer": None, "follow_ups": []
    }
    result = trinity_graph.invoke(initial, config=config)
    if result.get("awaiting_human"):
        return {"status": "awaiting_human", "checkpoint_message": result["checkpoint"]["message"], "thread_id": thread_id}
    return {
        "status": "complete", "answer": result.get("final_answer", "I could not generate an answer."),
        "confidence": result.get("confidence", {}).get("level", "Unknown"),
        "follow_ups": result.get("follow_ups", []), "sql_used": result.get("sql_query"), "thread_id": thread_id
    }

def resume_agent(thread_id: str, decision: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    trinity_graph.update_state(config, {"human_decision": decision, "awaiting_human": False})
    result = trinity_graph.invoke(None, config=config)
    return {
        "status": "complete", "answer": result.get("final_answer", "I could not generate an answer."),
        "confidence": result.get("confidence", {}).get("level", "Unknown"),
        "follow_ups": result.get("follow_ups", []), "sql_used": result.get("sql_query"), "thread_id": thread_id
    }
