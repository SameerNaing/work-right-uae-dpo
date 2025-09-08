from typing import Optional, Literal
from pydantic import BaseModel, Field

import chromadb


from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool


client = chromadb.PersistentClient(path="./chroma_store")
collection_name="wr-uae"
collection = client.get_or_create_collection(collection_name)
embed_model = OllamaEmbeddings(model="nomic-embed-text:v1.5")


vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embed_model,
    client=client
)

Category = Literal[
    "domestic-worker", "labour-inspection", "labour-disputes", "establishments",
    "fees-and-guarantees", "grievances", "alternative-end-of-service-benefits-system",
    "occupational-health-and-safety-and-labour-accommodation",
    "training-and-employment-of-students", "private-employment-agencies",
    "wage-protection", "workpermit-and-contract", "emiratisation", "mohre-services",
    "uae-jobs", "mohre-faq", "uae-visa-emirates-id", "uae-passport-travel"
]

class FindRelevantArgs(BaseModel):
    query: str = Field(..., description="User query to search for semantically similar documents.")
    category: Optional[Category] = Field(
        None,
        description="Optional category filter; limits results to this category."
    )
    k: int = Field(
        5,
        ge=1, le=20,
        description="Number of top results to return (defaults to 5; max 20)."
    )
    
def cut(text: str, max_chars: int = 1200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " …[truncated]"

@tool("find_relevant",args_schema=FindRelevantArgs)
def find_relevant(query: str, category: str | None = None, k: int = 5) -> str:
    """Search the vector store and return a compact RESULTS list plus a CONTEXT block."""
    filt = {"category": category} if category else None
    hits = vector_store.similarity_search(query, k=k, filter=filt)

    lines = []
    ctx = []
    for i, d in enumerate(hits, 1):
        cat = (d.metadata or {}).get("category", "unknown")
        doc_id = (d.metadata or {}).get("id", d.id)
        lines.append(f"[{i}] id={doc_id} category={cat}")
        ctx.append(f"[{i}] {cut(d.page_content)}")

    return "RESULTS:\n" + "\n".join(lines) + "\n\nCONTEXT:\n" + "\n\n---\n\n".join(ctx)


class GetFullArgs(BaseModel):
    doc_id: str = Field(..., description="Document ID to fetch the full, untruncated text for.")

@tool("get_full", args_schema=GetFullArgs)
def get_full(doc_id: str) -> str:
    """Fetch the full text of a single document by its ID from the vector store."""
    try:
        docs = vector_store.get_by_ids([doc_id])
        return docs[0].page_content if docs else f"Not found: {doc_id}"
    except Exception as e:
        return f"Error fetching {doc_id}: {e}"
    
    
tools = [
    find_relevant, 
    get_full
]