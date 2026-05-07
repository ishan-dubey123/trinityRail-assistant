import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# 1. Get API key from environment variable (set in Vercel)
# ============================================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set")

INDEX_NAME = "trinity-rail-index"

# ============================================================
# 2. Initialize real embedding model (384 dimensions)
# ============================================================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ============================================================
# 3. Connect to Pinecone and return vector store
# ============================================================
def get_vectorstore():
    """
    Creates and returns a PineconeVectorStore connected to your index.
    Raises an error if the index does not exist.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        raise RuntimeError(f"Index '{INDEX_NAME}' does not exist. Please run upload script.")
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

# ============================================================
# 4. Search function used by graph.py
# ============================================================
def search_documents(question: str, top_k: int = 3, threshold: float = 0.5) -> dict:
    """
    Searches Pinecone for documents relevant to the question.

    Args:
        question: User's query string
        top_k: Number of top documents to retrieve
        threshold: Minimum similarity score (0 to 1) to include result

    Returns:
        dict with keys: matched (list), count (int), best_score (float)
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(question, k=top_k)

    matched = []
    for doc, score in results:
        if score >= threshold:
            matched.append({
                "document": doc.page_content,
                "similarity": round(score, 3)
            })

    best_score = matched[0]["similarity"] if matched else 0.0

    return {
        "matched": matched,
        "count": len(matched),
        "best_score": best_score
    }

# ============================================================
# Optional: quick test when run directly
# ============================================================
if __name__ == "__main__":
    test_question = "What are the inspection rules for tank cars?"
    result = search_documents(test_question)
    print(f"Question: {test_question}")
    print(f"Found {result['count']} documents (best score: {result['best_score']})")
    for item in result['matched']:
        print(f"  - Similarity {item['similarity']}: {item['document'][:100]}...")
