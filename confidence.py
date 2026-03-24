def calculate_confidence(
    sql_result: dict = None,
    rag_result: dict = None,
    question: str = ""
) -> dict:
    """
    Calculates a confidence score based on:
    - Did SQL return results?
    - Did RAG find strong document matches?
    - Was the question specific enough?

    Returns a dict with:
    - level : 'High', 'Medium', or 'Low'
    - score : numeric score 0-100
    - reason: plain English explanation
    """

    score = 100  # start perfect, subtract for problems
    reasons = []

    # --- Check SQL Results ---
    if sql_result is not None:
        if not sql_result.get("success"):
            score -= 40
            reasons.append("SQL query failed or had an error")
        elif sql_result.get("row_count", 0) == 0:
            score -= 30
            reasons.append("SQL query returned no matching records")
        elif sql_result.get("row_count", 0) > 500:
            score -= 10
            reasons.append("SQL returned very large result set")

    # --- Check RAG Results ---
    if rag_result is not None:
        if rag_result.get("count", 0) == 0:
            score -= 30
            reasons.append("No relevant policy documents found")
        elif rag_result.get("best_score", 0) < 0.7:
            score -= 20
            reasons.append("Document matches were weak")

    # --- Check Question Quality ---
    word_count = len(question.strip().split())
    if word_count < 4:
        score -= 15
        reasons.append("Question was very short or vague")

    # --- Assign Level ---
    if score >= 80:
        level = "High"
    elif score >= 50:
        level = "Medium"
    else:
        level = "Low"

    return {
        "level": level,
        "score": max(score, 0),  # never go below 0
        "reasons": reasons if reasons else ["All checks passed"]
    }

def should_trigger_human_checkpoint(
    sql_result: dict = None,
    confidence: dict = None,
    question: str = ""
) -> dict:
    """
    Decides whether to pause and ask the human to confirm.

    Triggers when:
    1. Result set is too large (>500 rows)
    2. Confidence is Low
    3. Question contains sensitive keywords

    Returns dict with:
    - trigger  : True or False
    - reason   : why it triggered
    - message  : what to show the user
    """

    # --- Check 1: Large result set ---
    if sql_result and sql_result.get("row_count", 0) > 500:
        return {
            "trigger": True,
            "reason": "large_result",
            "message": (
                f"I found {sql_result['row_count']} railcars matching your query. "
                "That's a large dataset. Do you want to proceed or filter further?"
            )
        }

    # --- Check 2: Low confidence ---
    if confidence and confidence.get("level") == "Low":
        return {
            "trigger": True,
            "reason": "low_confidence",
            "message": (
                "I'm not very confident in this answer. "
                "The question may be too vague or no matching data was found. "
                "Do you want to rephrase or proceed anyway?"
            )
        }

    # --- Check 3: Sensitive keywords ---
    sensitive_words = [
        "delete", "remove", "financial", "revenue",
        "cost", "salary", "profit", "loss", "write"
    ]
    question_lower = question.lower()
    for word in sensitive_words:
        if word in question_lower:
            return {
                "trigger": True,
                "reason": "sensitive_data",
                "message": (
                    f"Your question involves sensitive data ('{word}'). "
                    "Please confirm you want to proceed."
                )
            }

    # --- No trigger needed ---
    return {
        "trigger": False,
        "reason": None,
        "message": None
    }