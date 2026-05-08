# 🚂 TrinityRail Assistant — Multi-Agent AI System

A multi-agent AI system built with LangGraph, Pinecone, and FastAPI
that answers natural language questions about railcar fleet data,
maintenance policies, and lease information.

## Architecture

- **Router Node** — classifies question as SQL, RAG, or both
- **SQL Agent** — generates and runs SQLite queries using LLM
- **RAG Agent** — searches policy documents using Mistral AI embeddings + Pinecone vector DB
- **Confidence Scorer** — scores result quality (High/Medium/Low)
- **Human-in-the-Loop** — pauses for confirmation on sensitive queries or Low confidence
- **Synthesizer** — combines results into a clean natural language answer
  
## Tech Stack

- LangGraph (multi-agent orchestration)
- Pinecone (Cloud vector store for RAG)
- Mistral AI (embedding API)
- Grog (LLM for SQL & reasoning)
- FastAPI (REST API + chat UI)
- SQLite (mock railcar database)
- Vercel (Serverless Deployment)

## Live Demo
https://trinity-rail-assistant.vercel.app/

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
