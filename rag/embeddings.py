"""
RAG Document Ingestion Script — UrbanPulse
Embeds traffic regulation documents and stores them in ChromaDB.

Usage:
    python -m rag.embeddings

Place .txt or .pdf (if pypdf installed) documents in rag/documents/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from google import genai
from chromadb import EmbeddingFunction, Documents, Embeddings

from utils.config import settings

logger = logging.getLogger("urbanpulse")


class _GeminiEmbeddingFunction(EmbeddingFunction):
    """Thin wrapper that calls the Gemini embedding API directly,
    bypassing chromadb's built-in wrapper which uses a stale SDK API."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        texts = [input] if isinstance(input, str) else list(input)
        embeddings = []
        for text in texts:
            result = client.models.embed_content(
                model=settings.EMBEDDING_MODEL,
                contents=text,
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings



# ── Sample built-in knowledge chunks ─────────────────────────────────────────
# These are used if no documents are found in rag/documents/.
BUILT_IN_CHUNKS: list[dict] = [
    {
        "id": "reg-001",
        "source": "Urban Traffic Management Manual §4.2",
        "content": (
            "During high-density congestion events exceeding 80% road capacity, "
            "traffic management authorities should activate pre-designated diversion "
            "routes and adjust signal timings to extend green phases on alternative corridors. "
            "Emergency services must be notified when average speeds fall below 10 km/h for "
            "more than 5 minutes on arterial roads."
        ),
    },
    {
        "id": "reg-002",
        "source": "Emergency Response Protocol §7.1",
        "content": (
            "A CRITICAL traffic incident is defined as: vehicle density >90%, average speed "
            "<5 km/h, confirmed accident report, or natural disaster. Upon CRITICAL declaration, "
            "the Traffic Management Center (TMC) must: (1) notify police and emergency services "
            "within 2 minutes, (2) activate all available diversion routes, (3) broadcast "
            "commuter alerts via all available channels."
        ),
    },
    {
        "id": "reg-003",
        "source": "Adverse Weather Traffic Protocol §2.4",
        "content": (
            "During adverse weather conditions (heavy rain, fog, reduced visibility), standard "
            "congestion thresholds are reduced by 15%. If visibility drops below 100 metres, "
            "mandatory speed reduction advisories must be issued. Signal cycle times should be "
            "extended by 20% to account for increased stopping distances. Traffic density "
            "monitoring frequency should be doubled."
        ),
    },
    {
        "id": "reg-004",
        "source": "Smart City Traffic Regulation Act §12",
        "content": (
            "Automated traffic management systems must provide human-readable justifications "
            "for any automated intervention. All automated signal changes must be logged with "
            "timestamps, operator ID, and system version. Critical interventions affecting "
            "emergency vehicle routing must be reviewed by a certified Traffic Management "
            "Officer before implementation."
        ),
    },
    {
        "id": "reg-005",
        "source": "Incident Prioritisation Framework §3.1",
        "content": (
            "Traffic incidents shall be prioritised using a severity scoring matrix (1-10): "
            "Score 1-3: Minor congestion, automated response permitted. "
            "Score 4-6: Moderate incident, operator notification required. "
            "Score 7-8: Major incident, immediate response team activation. "
            "Score 9-10: Critical emergency, all-hands response, TMC Commander notification."
        ),
    },
    {
        "id": "reg-006",
        "source": "Urban Diversion Management Guideline §5.3",
        "content": (
            "Pre-designated diversion routes must be reviewed and updated quarterly. "
            "Primary diversions should add no more than 15 minutes to normal journey time. "
            "Secondary diversions (for when primary is also congested) should add no more "
            "than 25 minutes. Diversion signs must be activated within 3 minutes of "
            "incident classification. Night-time diversions require additional lighting checks."
        ),
    },
    {
        "id": "reg-007",
        "source": "Signal Coordination Manual §8.2",
        "content": (
            "Adaptive signal control should prioritise emergency vehicle corridors. "
            "During incidents, the affected zone's signals should shift to a 'clearance cycle' "
            "reducing green time toward the incident zone by 40%. Adjacent intersections on "
            "diversion routes should receive extended green phases of 25-35 seconds. "
            "All signal changes must revert to normal scheduling within 2 hours of incident clearance."
        ),
    },
]


def ingest_documents() -> int:
    """
    Ingest documents from rag/documents/ AND built-in chunks into ChromaDB.
    Returns the total number of chunks indexed.
    """
    chroma_path = Path(settings.CHROMA_PATH)
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_path))

    if settings.GEMINI_API_KEY:
        ef = _GeminiEmbeddingFunction()
    else:
        from chromadb.utils import embedding_functions
        ef = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name="traffic_regulations",
        embedding_function=ef,
    )

    # Only ingest if collection is empty (idempotent)
    if collection.count() > 0:
        logger.info("RAG collection already has %d chunks — skipping ingestion", collection.count())
        return collection.count()

    # Add built-in chunks
    all_ids = [c["id"] for c in BUILT_IN_CHUNKS]
    all_docs = [c["content"] for c in BUILT_IN_CHUNKS]
    all_meta = [{"source": c["source"]} for c in BUILT_IN_CHUNKS]

    # Scan documents directory for .txt files
    docs_dir = Path(__file__).parent / "documents"
    docs_dir.mkdir(exist_ok=True)
    for txt_file in docs_dir.glob("*.txt"):
        content = txt_file.read_text(encoding="utf-8")
        chunks = _chunk_text(content, chunk_size=500)
        for i, chunk in enumerate(chunks):
            all_ids.append(f"file-{txt_file.stem}-{i}")
            all_docs.append(chunk)
            all_meta.append({"source": txt_file.name})

    collection.add(ids=all_ids, documents=all_docs, metadatas=all_meta)
    logger.info("RAG: Ingested %d chunks into ChromaDB", len(all_ids))
    return len(all_ids)


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    count = ingest_documents()
    print(f"✅ RAG ingestion complete: {count} chunks indexed.")
