import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

INDEX_NAME = "trinity-rail-index-1536"

def get_vectorstore():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

def search_documents(question: str, top_k: int = 3, threshold: float = 0.5) -> dict:
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(question, k=top_k)
    matched = []
    for doc, score in results:
        if score >= threshold:
            matched.append({"document": doc.page_content, "similarity": round(score, 3)})
    best_score = matched[0]["similarity"] if matched else 0.0
    return {"matched": matched, "count": len(matched), "best_score": best_score}
