from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Optional
import json

from database import run_query
from rag import search_documents
from confidence import calculate_confidence, should_trigger_human_checkpoint

# ─────────────────────────────────────────────
# 1. THE STATE — shared memory across all nodes
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    question          : str
    route             : Optional[str]
    sql_query         : Optional[str]
    sql_result        : Optional[dict]
    rag_result        : Optional[dict]
    confidence        : Optional[dict]
    checkpoint        : Optional[dict]
    awaiting_human    : Optional[bool]
    human_decision    : Optional[str]
    final_answer      : Optional[str]
    follow_ups        : Optional[list]

# ─────────────────────────────────────────────
# 2. THE LLM — Groq + Llama 3.3 (free)
# ─────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ─────────────────────────────────────────────
# 3. ROUTER NODE
# ─────────────────────────────────────────────
def router_node(state: AgentState) -> AgentState:
    print(f"\n🔀 ROUTER: Analyzing question...")
    question = state["question"]

    prompt = f"""
You are a routing assistant for a railcar company's AI system.

Classify this question into EXACTLY one of these three categories:
- "sql"  → question needs numbers, counts, or data from the database
- "rag"  → question needs policy rules, guidelines, or documents
- "both" → question needs both data AND policy knowledge

Question: {question}

Reply with ONLY one word: sql, rag, or both.
No explanation. No punctuation. Just the single word.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    route = response.content.strip().lower()

    if route not in ["sql", "rag", "both"]:
        route = "both"

    print(f"   → Route decided: {route}")
    return {**state, "route": route}

# ─────────────────────────────────────────────
# 4. SQL AGENT — IMPROVED PROMPT
# ─────────────────────────────────────────────
def sql_agent_node(state: AgentState) -> AgentState:
    print(f"\n📊 SQL AGENT: Generating query...")
    question = state["question"]

    schema_prompt = f"""
You are a SQL expert for TrinityRail. Write ONLY a SELECT query. No markdown, no backticks, no explanation.

Database schema:
- railcars: car_id (int), car_type ('tank','boxcar','flatcar'), status ('idle','leased','maintenance'), region ('north','south','east','west'), last_inspection_date (text YYYY-MM-DD), commodity ('grain','chemicals','coal','none')
- leases: lease_id (int), car_id (int), customer_name (text), start_date (text), end_date (text), monthly_rate (real)

Rules:
1. Answer exactly what the user asks. Do NOT add extra filters (e.g., commodity, car_type) unless the user explicitly mentions them.
2. For "how many cars are idle?" → SELECT COUNT(car_id) FROM railcars WHERE status='idle'
   For "list idle cars" → SELECT car_id, car_type, region FROM railcars WHERE status='idle'
3. Date logic:
   - Use date('now') for today.
   - "Overdue for inspection" → last_inspection_date < date('now', '-90 days')  (applies to all cars; ignore status unless asked)
   - "Idle for more than X days" → last_inspection_date < date('now', '-X days') AND status='idle'
4. Never include commodity in WHERE unless the user asks about a specific commodity (e.g., "cars carrying grain").

Question: {question}

SQL query:
"""
    response = llm.invoke([HumanMessage(content=schema_prompt)])
    sql_query = response.content.strip()
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    print(f"   → Generated SQL: {sql_query}")
    sql_result = run_query(sql_query)
    print(f"   → Rows returned: {sql_result.get('row_count', 0)}")

    return {**state, "sql_query": sql_query, "sql_result": sql_result}

# ─────────────────────────────────────────────
# 5. RAG AGENT (uses Mistral embeddings via rag.py)
# ─────────────────────────────────────────────
def rag_agent_node(state: AgentState) -> AgentState:
    print(f"\n📄 RAG AGENT: Searching policy documents...")
    question = state["question"]
    rag_result = search_documents(question, top_k=3, threshold=0.7)
    print(f"   → Documents found: {rag_result['count']}")
    if rag_result["count"] > 0:
        print(f"   → Best match score: {rag_result['best_score']}")
    return {**state, "rag_result": rag_result}

# ─────────────────────────────────────────────
# 6. CONFIDENCE CHECKER
# ─────────────────────────────────────────────
def confidence_node(state: AgentState) -> AgentState:
    print(f"\n🎯 CONFIDENCE: Scoring results...")
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
    print(f"   → Confidence: {confidence['level']} ({confidence['score']}/100)")
    print(f"   → Human checkpoint needed: {checkpoint['trigger']}")
    return {
        **state,
        "confidence": confidence,
        "checkpoint": checkpoint,
        "awaiting_human": checkpoint["trigger"]
    }

# ─────────────────────────────────────────────
# 7. HUMAN CHECKPOINT NODE
# ─────────────────────────────────────────────
def human_checkpoint_node(state: AgentState) -> AgentState:
    print(f"\n⏸️  HUMAN CHECKPOINT: Waiting for user decision...")
    return state

# ─────────────────────────────────────────────
# 8. SYNTHESIZER — with COUNT result handling
# ─────────────────────────────────────────────
def synthesizer_node(state: AgentState) -> AgentState:
    print(f"\n✍️  SYNTHESIZER: Building final answer...")
    question = state["question"]
    sql_result = state.get("sql_result")
    rag_result = state.get("rag_result")
    confidence = state.get("confidence", {})

    # --- Preprocess SQL result: simplify COUNT queries ---
    if sql_result and sql_result.get("success") and sql_result.get("row_count", 0) == 1:
        first_row = sql_result["results"][0]
        keys = list(first_row.keys())
        if len(keys) == 1 and any(word in keys[0].lower() for word in ["count", "total"]):
            count_val = first_row[keys[0]]
            sql_result["results"] = [{"count": count_val}]
            sql_result["simplified_count"] = count_val

    # Build context
    context_parts = []
    if sql_result and sql_result.get("success") and sql_result.get("row_count", 0) > 0:
        rows = sql_result["results"][:10]
        context_parts.append(f"DATABASE RESULTS ({sql_result['row_count']} total rows):\n{json.dumps(rows, indent=2)}")

    if rag_result and rag_result.get("count", 0) > 0:
        docs = [item["document"] for item in rag_result["matched"]]
        context_parts.append(f"RELEVANT POLICY DOCUMENTS:\n" + "\n".join(f"- {d}" for d in docs))

    if not context_parts:
        context = "No relevant data or documents were found."
    else:
        context = "\n\n".join(context_parts)

    synthesis_prompt = f"""
You are a helpful assistant for TrinityRail, a North American railcar company.

Answer the user's question using ONLY the context provided below.
Be concise, clear, and professional.
If the context contains a simplified "count" value, just state the number.
Do not make up information beyond what is in the context.

Context:
{context}

User Question: {question}

Write a clear, direct answer in 2-4 sentences:
"""
    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
    answer = response.content.strip()

    # Generate follow-ups
    followup_prompt = f"""
The user asked: "{question}"
The answer was: "{answer}"

Suggest exactly 2 short follow-up questions they might ask next.
Keep them specific to railcar operations.
Return ONLY a JSON array of 2 strings. Example: ["question 1", "question 2"]
No explanation. Just the JSON array.
"""
    followup_response = llm.invoke([HumanMessage(content=followup_prompt)])
    try:
        follow_ups = json.loads(followup_response.content.strip())
    except:
        follow_ups = [
            "Which region has the most idle railcars?",
            "Which cars are overdue for inspection?"
        ]

    print(f"   → Answer generated. Confidence: {confidence.get('level', 'N/A')}")
    return {**state, "final_answer": answer, "follow_ups": follow_ups}

# ─────────────────────────────────────────────
# 9. CONDITIONAL EDGES
# ─────────────────────────────────────────────
def route_decision(state: AgentState) -> str:
    route = state.get("route", "both")
    if route == "sql":
        return "sql_only"
    elif route == "rag":
        return "rag_only"
    else:
        return "both"

def after_sql_decision(state: AgentState) -> str:
    return "confidence"

def after_rag_decision(state: AgentState) -> str:
    return "confidence"

def checkpoint_decision(state: AgentState) -> str:
    if state.get("awaiting_human"):
        return "needs_human"
    return "skip_human"

def human_decision(state: AgentState) -> str:
    decision = state.get("human_decision", "proceed")
    if decision == "refine":
        return "loop_back"
    return "proceed"

# ─────────────────────────────────────────────
# 10. BUILD GRAPH
# ─────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("confidence", confidence_node)
    graph.add_node("human_checkpoint", human_checkpoint_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "sql_only": "sql_agent",
            "rag_only": "rag_agent",
            "both": "sql_agent"
        }
    )

    graph.add_edge("sql_agent", "rag_agent")
    graph.add_edge("rag_agent", "confidence")

    graph.add_conditional_edges(
        "confidence",
        checkpoint_decision,
        {
            "needs_human": "human_checkpoint",
            "skip_human": "synthesizer"
        }
    )

    graph.add_conditional_edges(
        "human_checkpoint",
        human_decision,
        {
            "proceed": "synthesizer",
            "loop_back": "router"
        }
    )

    graph.add_edge("synthesizer", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory, interrupt_before=["human_checkpoint"])

trinity_graph = build_graph()

# ─────────────────────────────────────────────
# 11. HELPER FUNCTIONS for FastAPI
# ─────────────────────────────────────────────
def ask_agent(question: str, thread_id: str = "default") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "question": question,
        "route": None,
        "sql_query": None,
        "sql_result": None,
        "rag_result": None,
        "confidence": None,
        "checkpoint": None,
        "awaiting_human": False,
        "human_decision": None,
        "final_answer": None,
        "follow_ups": []
    }
    result = trinity_graph.invoke(initial_state, config=config)
    if result.get("awaiting_human"):
        return {
            "status": "awaiting_human",
            "checkpoint_message": result["checkpoint"]["message"],
            "thread_id": thread_id
        }
    return {
        "status": "complete",
        "answer": result.get("final_answer", "I could not generate an answer."),
        "confidence": result.get("confidence", {}).get("level", "Unknown"),
        "follow_ups": result.get("follow_ups", []),
        "sql_used": result.get("sql_query"),
        "thread_id": thread_id
    }

def resume_agent(thread_id: str, decision: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    trinity_graph.update_state(config, {"human_decision": decision, "awaiting_human": False})
    result = trinity_graph.invoke(None, config=config)
    return {
        "status": "complete",
        "answer": result.get("final_answer", "I could not generate an answer."),
        "confidence": result.get("confidence", {}).get("level", "Unknown"),
        "follow_ups": result.get("follow_ups", []),
        "sql_used": result.get("sql_query"),
        "thread_id": thread_id
    }
