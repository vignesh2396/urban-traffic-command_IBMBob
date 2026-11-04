"""
RAG Tools — UrbanPulse
Retrieves relevant traffic regulations and emergency protocols
from a ChromaDB vector store backed by Gemini embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import chromadb
from google import genai
from chromadb import EmbeddingFunction, Documents, Embeddings

from utils.config import settings
from utils.audit_logger import log_action

logger = logging.getLogger("urbanpulse")

_client: Optional[chromadb.PersistentClient] = None
_collection = None


class _GeminiEmbeddingFunction(EmbeddingFunction):
    """Thin wrapper that calls the Gemini embedding API directly,
    bypassing chromadb's built-in wrapper which uses a stale SDK API."""

    def __call__(self, input: Documents) -> Embeddings:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        result = client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=input,
        )
        # New SDK returns EmbedContentResponse with .embeddings list of ContentEmbedding
        if isinstance(input, str):
            return [result.embeddings[0].values]
        return [e.values for e in result.embeddings]


def _get_collection():
    """Lazy-initialise ChromaDB client and collection."""
    global _client, _collection
    if _collection is not None:
        return _collection

    chroma_path = Path(settings.CHROMA_PATH)
    chroma_path.mkdir(parents=True, exist_ok=True)

    _client = chromadb.PersistentClient(path=str(chroma_path))

    # Use Gemini embeddings if key is available, else fall back to default
    if settings.GEMINI_API_KEY:
        ef = _GeminiEmbeddingFunction()
    else:
        from chromadb.utils import embedding_functions
        ef = embedding_functions.DefaultEmbeddingFunction()

    _collection = _client.get_or_create_collection(
        name="traffic_regulations",
        embedding_function=ef,
        metadata={"description": "Urban traffic regulations and emergency protocols"},
    )
    return _collection


def retrieve_regulations(
    query: str,
    zone_id: Optional[str] = None,
    n_results: int = None,
) -> list[dict]:
    """
    Retrieve the top-k most relevant regulation chunks for a given query.

    Args:
        query:     Natural language query from the Incident Investigation Agent.
        zone_id:   Optional zone filter (passed to audit log).
        n_results: Number of results; defaults to settings.RAG_TOP_K.

    Returns:
        List of dicts: [{source, content, distance}]
    """
    if n_results is None:
        n_results = settings.RAG_TOP_K

    collection = _get_collection()

    if collection.count() == 0:
        logger.warning("RAG collection is empty — ingestion may not have run")
        log_action(
            agent_name="RAGTool",
            action="RAG_COLLECTION_EMPTY",
            entity_type="zone",
            entity_id=zone_id,
            payload={"query": query},
        )
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            docs.append(
                {
                    "source": meta.get("source", "Unknown"),
                    "content": doc,
                    "distance": round(dist, 4),
                }
            )

        log_action(
            agent_name="RAGTool",
            action="RAG_QUERY_EXECUTED",
            entity_type="zone",
            entity_id=zone_id,
            payload={"query": query, "hits": len(docs)},
        )
        return docs

    except Exception as exc:
        logger.error("RAG query failed: %s", exc)
        log_action(
            agent_name="RAGTool",
            action="RAG_QUERY_ERROR",
            entity_type="zone",
            entity_id=zone_id,
            payload={"error": str(exc)},
        )
        return []
