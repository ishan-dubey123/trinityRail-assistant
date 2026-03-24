import os
os.environ["HF_HOME"] = r"D:\Fleet_Asset_Intelligence_Agent_Railcars_Trinity_Industries\.cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\Fleet_Asset_Intelligence_Agent_Railcars_Trinity_Industries\.cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ChromaDB client
client = chromadb.PersistentClient(path="./chroma_store")

# Embedder
embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create collection
collection = client.get_or_create_collection(
    name="trinity_policies",
    embedding_function=embedder
)

# --- Trinity Policy Documents ---
# These are realistic policy statements a railcar company would have
# In a real project, these would come from actual PDF/Word documents
POLICY_DOCUMENTS = [
    # Maintenance policies
    "Tank cars must be inspected every 90 days per federal safety regulations.",
    "Boxcars carrying grain require fumigation and cleaning before reuse.",
    "Flatcars must have all tie-down equipment inspected before each load.",
    "Idle railcars parked for more than 60 days require a full wheel inspection.",
    "Any railcar involved in a derailment must undergo full structural inspection before returning to service.",
    "Brake systems on all car types must be tested every 180 days.",

    # Lease policies
    "Lease agreements expire automatically unless renewed at least 30 days prior to end date.",
    "Monthly lease rates for tank cars range from $1,500 to $3,000 depending on car age and condition.",
    "Customers must notify Trinity at least 14 days before returning a leased railcar.",
    "Early lease termination incurs a penalty of two months of the monthly lease rate.",

    # Operational policies
    "Railcars carrying hazardous chemicals require special placarding per DOT regulations.",
    "Cars transporting coal must be cleaned within 7 days of unloading to prevent corrosion.",
    "Grain cars must be sealed and verified before departure from loading facilities.",
    "Railcar utilization rate is calculated as leased cars divided by total available cars.",

    # Regional policies
    "Southern region railcars face higher humidity corrosion risk and require quarterly coating inspections.",
    "Northern region cars require winterization checks before November each year.",
    "Eastern region cars are subject to port authority regulations when near coastal terminals.",
    "Western region flatcars serving lumber yards require load securing certification.",

    # Safety policies
    "All maintenance personnel must log inspection results in the digital tracking system within 24 hours.",
    "Safety violations found during inspection must be reported to operations within 4 hours.",
    "Railcars with structural damage are immediately flagged as out-of-service until repaired.",
]

def load_documents():
    """
    Loads policy documents into ChromaDB.
    Only loads if collection is empty — avoids duplicates.
    """
    existing = collection.count()
    if existing > 0:
        print(f"✅ ChromaDB already has {existing} documents loaded.")
        return

    # Add all documents with unique IDs
    collection.add(
        documents=POLICY_DOCUMENTS,
        ids=[f"doc_{i}" for i in range(len(POLICY_DOCUMENTS))]
    )
    print(f"✅ Loaded {len(POLICY_DOCUMENTS)} policy documents into ChromaDB.")

def search_documents(question: str, top_k: int = 3, threshold: float = 0.7):
    """
    Searches ChromaDB for documents relevant to the question.

    How it works:
    1. Converts the question to a vector (embedding)
    2. Finds the top_k most similar documents
    3. Filters out anything below the threshold score

    Args:
        question  : the user's question
        top_k     : how many results to return max
        threshold : minimum similarity score (0.0 to 1.0)

    Returns a dict with matched documents and their scores.
    """
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    # ChromaDB returns distances (lower = more similar)
    # We convert distance to similarity: similarity = 1 - distance
    documents = results["documents"][0]   # list of matched text
    distances = results["distances"][0]   # list of distance scores

    matched = []
    for doc, distance in zip(documents, distances):
        similarity = round(1 - distance, 3)  # convert to similarity score

        if similarity >= threshold:
            matched.append({
                "document": doc,
                "similarity": similarity
            })

    # Sort by highest similarity first
    matched.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "matched": matched,
        "count": len(matched),
        "best_score": matched[0]["similarity"] if matched else 0.0
    }

# --- Run setup when this file is executed directly ---
if __name__ == "__main__":
    load_documents()
    print("\n--- Test Search ---")
    result = search_documents("inspection rule for tank cars")
    for item in result["matched"]:
        print(f"Score: {item['similarity']} | {item['document']}")