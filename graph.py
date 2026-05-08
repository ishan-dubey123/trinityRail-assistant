from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from typing import TypedDict, Optional
import json
import re

from database import run_query
from rag import search_documents
from confidence import calculate_confidence, should_trigger_human_checkpoint

# ========== 1. STATE ==========
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

# ========== 2. LLM (Groq, free) ==========
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ========== 3. ROUTER (STRONG BIAS FOR RAG ON POLICY WORDS) ==========
def router_node(state: AgentState) -> AgentState:
    print(f"\n🔀 ROUTER: {state['question']}")
    question_lower = state["question"].lower()
    
    # Force RAG if question contains policy-related keywords
    policy_keywords = ["policy", "policies", "document", "rule", "guideline", "regulation", "inspection rule", "maintenance policy"]
    if any(kw in question_lower for kw in policy_keywords):
        print("   → Force route: rag (policy keywords detected)")
        return {**state, "route": "rag"}

    prompt = f"""
You are a router for a railcar assistant. Decide if the user's question needs:
- "sql" → ONLY for database data: counts, lists of cars, lease info, specific car attributes.
- "rag" → ONLY for policies, rules, documents, guidelines, or any question that asks "what is the rule/policy for...".
- "both" → only if they need both data and policy (e.g., "show idle tank cars and their inspection rule").

IMPORTANT: If the question contains words like "policy", "rule", "document", "guideline", "regulation", answer "rag".

Question: {state['question']}
Reply with exactly one word: sql, rag, or both.
"""
    route = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    if route not in ["sql", "rag", "both"]:
        route = "both"
    print(f"   → Route: {route}")
    return {**state, "route": route}

# ========== 4. SQL AGENT (with validation to remove extra car_type filters) ==========
def sql_agent_node(state: AgentState) -> AgentState:
    print(f"\n📊 SQL AGENT")
    question = state["question"]
    question_lower = question.lower()

    # Prompt for SQL generation
    schema_prompt = f"""
You are a SQL expert for TrinityRail. Write ONLY a SELECT query. No markdown, no backticks.

Schema:
- railcars: car_id, car_type ('tank','boxcar','flatcar'), status ('idle','leased','maintenance'), region, last_inspection_date (YYYY-MM-DD), commodity
- leases: lease_id, car_id, customer_name, start_date, end_date, monthly_rate

Rules:
- Add WHERE only if user explicitly mentions.
- Do NOT add car_type unless user says "tank", "boxcar", or "flatcar".
- For "how many cars are idle?" → SELECT COUNT(car_id) FROM railcars WHERE status='idle'
- For "overdue for maintenance" → SELECT car_id, last_inspection_date FROM railcars WHERE last_inspection_date < date('now', '-90 days')
- Use date('now', '-90 days') for 90 days ago.

Question: {question}

SQL query:
"""
    response = llm.invoke([HumanMessage(content=schema_prompt)])
    sql_query = response.content.strip()
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    # ----- Validation: remove car_type filter if user didn't mention a car type -----
    car_type_keywords = ['tank', 'boxcar', 'flatcar']
    user_mentioned_car_type = any(word in question_lower for word in car_type_keywords)

    if not user_mentioned_car_type:
        # Remove patterns like `car_type='tank'` or `car_type = 'boxcar'` from WHERE clause
        pattern = r"(?:\bAND\s+)?car_type\s*=\s*'[^']+'(?:\s+AND\s+|\s*$)"
        sql_query = re.sub(pattern, '', sql_query, flags=re.IGNORECASE)
        # Clean up double WHERE, trailing AND, etc.
        sql_query = re.sub(r"WHERE\s+AND\s+", "WHERE ", sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r"WHERE\s*$", "", sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r"AND\s+WHERE", "WHERE", sql_query, flags=re.IGNORECASE)
        print(f"   → Validation: removed car_type (not in question)")

    print(f"   → Final SQL: {sql_query}")
    sql_result = run_query(sql_query)
    print(f"   → Rows returned: {sql_result.get('row_count', 0)}")
    return {**state, "sql_query": sql_query, "sql_result": sql_result}

# ========== 5. RAG AGENT ==========
def rag_agent_node(state: AgentState) -> AgentState:
    print(f"\n📄 RAG AGENT")
    rag_result = search_documents(state["question"], top_k=3, threshold=0.7)
    print(f"   → Found {rag_result['count']} docs, best score {rag_result['best_score']}")
    return {**state, "rag_result": rag_result}

# ========== 6. CONFIDENCE ==========
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

# ========== 7. HUMAN CHECKPOINT (placeholder) ==========
def human_checkpoint_node(state: AgentState) -> AgentState:
    print(f"\n⏸️  HUMAN CHECKPOINT")
    return state

# ========== 8. SYNTHESIZER (handles COUNT and empty results) ==========
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
        context_parts.append("POLICY DOCUMENTS:\n" + "\n".join(f"- {d}" for d in docs))

    if not context_parts:
        context = "No data or policy documents found."
    else:
        context = "\n\n".join(context_parts)

    synthesis_prompt = f"""
Answer the user's question using ONLY the context below.
Be concise and direct.
If the context says "No matching records found", answer "None found".

Context:
{context}

User Question: {question}

Answer:
"""
    answer = llm.invoke([HumanMessage(content=synthesis_prompt)]).content.strip()

    # Generate follow‑up questions
    follow_prompt = f"""
Suggest exactly 2 short follow-up questions for: "{question}"
Return ONLY a JSON array, e.g. ["q1", "q2"]
"""
    try:
        follow_ups = json.loads(llm.invoke([HumanMessage(content=follow_prompt)]).content.strip())
    except:
        follow_ups = ["Which region has the most idle cars?", "Which cars are overdue for inspection?"]

    print(f"   → Answer: {answer[:100]}...")
    return {**state, "final_answer": answer, "follow_ups": follow_ups}

# ========== 9. CONDITIONAL EDGES ==========
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

# ========== 10. BUILD THE GRAPH ==========
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

# ========== 11. API HELPER FUNCTIONS ==========
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
