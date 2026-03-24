# 🚂 TrinityRail Assistant — Multi-Agent AI System

A multi-agent AI system built with LangGraph, ChromaDB, and FastAPI
that answers natural language questions about railcar fleet data,
maintenance policies, and lease information.

## Architecture

- **Router Node** — classifies question as SQL, RAG, or both
- **SQL Agent** — generates and runs SQLite queries using LLM
- **RAG Agent** — searches policy documents using ChromaDB embeddings
- **Confidence Scorer** — scores result quality (High/Medium/Low)
- **Human-in-the-Loop** — pauses for confirmation on sensitive queries
- **Synthesizer** — combines results into a clean natural language answer

## Tech Stack

- LangGraph (multi-agent orchestration)
- ChromaDB (vector store for RAG)
- Ollama + Mistral (local LLM, no API key needed)
- FastAPI (REST API + chat UI)
- SQLite (mock railcar database)

## Setup

### 1. Install dependencies
pip install -r requirements.txt

### 2. Start Ollama with Mistral
ollama pull mistral
ollama serve

### 3. Run the server
python main.py

### 4. Open browser
http://localhost:8000

## Sample Questions

- "How many tank cars are idle?"
- "What is the inspection rule for tank cars?"
- "Which cars are overdue for maintenance per policy?"
- "What is the financial cost of idle railcars?" ← triggers human checkpoint

## Project Structure

- main.py — FastAPI server and chat UI
- graph.py — LangGraph multi-agent graph
- database.py — SQLite setup and mock data
- rag.py — ChromaDB document store
- confidence.py — confidence scoring and human-in-the-loop logic