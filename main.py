from fastapi import FastAPI
from typing import Optional
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uuid
import os

from database import create_tables, seed_data
from rag import load_documents
from graph import ask_agent, resume_agent

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = FastAPI(title="TrinityRail Assistant")

# Run setup on startup

@app.on_event("startup")
def startup():
    print("🚀 Starting TrinityRail Assistant...")
    # This tells the code to use the only folder Vercel lets us write to
    os.environ["DB_PATH"] = "/tmp/railcar.db" 
    os.environ["CHROMA_PATH"] = "/tmp/chroma_db"
    
    create_tables()
    seed_data()
    load_documents()
    print("✅ All systems ready.\n")


# ─────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question : str
    thread_id: Optional[str] = None

class ResumeRequest(BaseModel):
    thread_id: str
    decision : str         # "proceed" or "refine"

# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.post("/ask")
def ask(request: QuestionRequest):
    """
    Main endpoint — takes a question, runs the agent, returns answer.
    """
    # Generate a unique thread ID if not provided
    # Thread ID keeps conversation memory separate per user
    thread_id = request.thread_id or str(uuid.uuid4())

    result = ask_agent(
        question=request.question,
        thread_id=thread_id
    )
    return result

@app.post("/resume")
def resume(request: ResumeRequest):
    """
    Resume endpoint — called after human checkpoint.
    User sends "proceed" or "refine".
    """
    result = resume_agent(
        thread_id=request.thread_id,
        decision=request.decision
    )
    return result

@app.get("/health")
def health():
    return {"status": "running", "agent": "TrinityRail Assistant"}

# ─────────────────────────────────────────────
# Simple Chat UI — served at localhost:8000
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>TrinityRail Assistant</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f1117;
            color: #e0e0e0;
            height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .header h1 { color: #4f9cf9; font-size: 24px; }
        .header p  { color: #888; font-size: 13px; margin-top: 4px; }

        #chat-box {
            width: 100%;
            max-width: 800px;
            flex: 1;
            overflow-y: auto;
            padding: 10px 0;
        }
        .message {
            margin: 10px 0;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 80%;
            line-height: 1.5;
            font-size: 14px;
        }
        .user-msg {
            background: #1e3a5f;
            color: #cce0ff;
            margin-left: auto;
            text-align: right;
        }
        .agent-msg {
            background: #1a1f2e;
            border: 1px solid #2a2f45;
            color: #e0e0e0;
        }
        .confidence {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: bold;
            margin-top: 6px;
        }
        .High   { background: #1a4731; color: #4caf50; }
        .Medium { background: #3d3200; color: #ffc107; }
        .Low    { background: #4a1010; color: #f44336; }

        .follow-ups {
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        .followup-btn {
            background: #1e2a3a;
            border: 1px solid #3a4a5a;
            color: #7ab3e0;
            padding: 5px 10px;
            border-radius: 8px;
            font-size: 12px;
            cursor: pointer;
        }
        .followup-btn:hover { background: #2a3a4a; }

        .checkpoint-msg {
            background: #3d2a00;
            border: 1px solid #8a5c00;
            color: #ffd070;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: 14px;
        }
        .checkpoint-btns { margin-top: 8px; display: flex; gap: 8px; }
        .btn-proceed {
            background: #1a4731; color: #4caf50;
            border: none; padding: 6px 14px;
            border-radius: 6px; cursor: pointer; font-size: 13px;
        }
        .btn-refine {
            background: #4a1010; color: #f44336;
            border: none; padding: 6px 14px;
            border-radius: 6px; cursor: pointer; font-size: 13px;
        }

        .input-area {
            width: 100%;
            max-width: 800px;
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        #question-input {
            flex: 1;
            padding: 12px 16px;
            background: #1a1f2e;
            border: 1px solid #2a2f45;
            border-radius: 10px;
            color: #e0e0e0;
            font-size: 14px;
            outline: none;
        }
        #question-input:focus { border-color: #4f9cf9; }
        #send-btn {
            padding: 12px 20px;
            background: #4f9cf9;
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
        }
        #send-btn:hover { background: #3a8ae0; }
        #send-btn:disabled { background: #2a3a5a; cursor: not-allowed; }

        .sql-detail {
            font-size: 11px;
            color: #555;
            margin-top: 6px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚂 TrinityRail Assistant</h1>
        <p>Ask questions about railcar fleet, maintenance, leases, and policies</p>
    </div>

    <div id="chat-box">
        <div class="message agent-msg">
            Hello! I'm the TrinityRail Assistant. Ask me anything about railcars — 
            fleet status, maintenance schedules, lease info, or company policies.<br><br>
            <strong>Try asking:</strong><br>
            • "How many tank cars are idle?"<br>
            • "What is the inspection rule for tank cars?"<br>
            • "Which cars are overdue for maintenance?"
        </div>
    </div>

    <div class="input-area">
        <input id="question-input" type="text" placeholder="Ask about railcars..." />
        <button id="send-btn" onclick="sendQuestion()">Send</button>
    </div>

<script>
    let currentThreadId = null;

    // Allow Enter key to send
    document.getElementById('question-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendQuestion();
    });

    function addMessage(content, isUser = false) {
        const box = document.getElementById('chat-box');
        const div = document.createElement('div');
        div.className = 'message ' + (isUser ? 'user-msg' : 'agent-msg');
        div.innerHTML = content;
        box.appendChild(div);
        box.scrollTop = box.scrollHeight;
    }

    async function sendQuestion() {
        const input = document.getElementById('question-input');
        const btn   = document.getElementById('send-btn');
        const question = input.value.trim();
        if (!question) return;

        // Show user message
        addMessage(question, true);
        input.value = '';
        btn.disabled = true;
        btn.textContent = 'Thinking...';

        try {
            const res = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, thread_id: currentThreadId })
            });
            const data = await res.json();
            currentThreadId = data.thread_id;

            if (data.status === 'awaiting_human') {
                showCheckpoint(data.checkpoint_message, data.thread_id);
            } else {
                showAnswer(data);
            }
        } catch (err) {
            addMessage('❌ Error connecting to the agent. Is the server running?');
        }

        btn.disabled = false;
        btn.textContent = 'Send';
    }

    function showAnswer(data) {
        let html = `<div>${data.answer}</div>`;
        html += `<span class="confidence ${data.confidence}">${data.confidence} Confidence</span>`;

        if (data.sql_used) {
            html += `<div class="sql-detail">SQL: ${data.sql_used}</div>`;
        }

        if (data.follow_ups && data.follow_ups.length > 0) {
            html += `<div class="follow-ups">`;
            data.follow_ups.forEach(q => {
                html += `<button class="followup-btn" onclick="askFollowUp('${q.replace(/'/g,"&#39;")}')">${q}</button>`;
            });
            html += `</div>`;
        }
        addMessage(html);
    }

    function showCheckpoint(message, threadId) {
        const html = `
            <div class="checkpoint-msg">
                ⏸️ <strong>Confirmation needed:</strong> ${message}
                <div class="checkpoint-btns">
                    <button class="btn-proceed" onclick="resumeAgent('proceed', '${threadId}')">✅ Proceed</button>
                    <button class="btn-refine"  onclick="resumeAgent('refine',  '${threadId}')">✏️ Refine question</button>
                </div>
            </div>`;
        addMessage(html);
    }

    async function resumeAgent(decision, threadId) {
        const btn = document.getElementById('send-btn');
        btn.disabled = true;

        addMessage(decision === 'proceed' ? '✅ Proceeding...' : '✏️ Please rephrase your question.', true);

        if (decision === 'refine') {
            btn.disabled = false;
            return;
        }

        try {
            const res = await fetch('/resume', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ thread_id: threadId, decision })
            });
            const data = await res.json();
            showAnswer(data);
        } catch (err) {
            addMessage('❌ Error resuming agent.');
        }

        btn.disabled = false;
    }

    function askFollowUp(question) {
        document.getElementById('question-input').value = question;
        sendQuestion();
    }
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────
# Run the server
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
