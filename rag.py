"""
rag.py - PDF ingestion + semantic retrieval via Qdrant Cloud
SSL fix included for Windows certifi path issues.
"""

import os
import ssl
import certifi
import uuid
import fitz
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# ── SSL fix for Windows (certifi path issue) ──────────────────────────────
os.environ["SSL_CERT_FILE"]      = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

load_dotenv()

COLLECTION  = "cleaning_faq"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE  = 600
OVERLAP     = 100
TOP_K       = 5
VECTOR_DIM  = 384

_model  = None
_client = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_client():
    global _client
    if _client is None:
        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_key = os.getenv("QDRANT_API_KEY", "").strip()

        if not qdrant_url:
            raise ValueError(
                "QDRANT_URL is not set. "
                "Add it to your .env file or paste it in the sidebar Config."
            )
        if not qdrant_key:
            raise ValueError(
                "QDRANT_API_KEY is not set. "
                "Add it to your .env file or paste it in the sidebar Config."
            )

        _client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_key,
            https=True,
            verify=certifi.where(),   # ← explicit cert bundle path
        )
    return _client


def ensure_collection():
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_DIM, distance=models.Distance.COSINE
            ),
        )
    # Always ensure payload index exists — handles both new and pre-existing collections.
    # Qdrant requires an index on bool fields before they can be used in filters.
    try:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="is_sentinel",
            field_schema=models.PayloadSchemaType.BOOL,
        )
    except Exception:
        pass  # Index already exists — fine to ignore


def _chunk_text(text: str) -> list:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if len(chunk) > 60:
            chunks.append(chunk)
        start += CHUNK_SIZE - OVERLAP
    return chunks


def _extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def ingest_pdf(pdf_path: str) -> int:
    ensure_collection()
    client   = get_client()
    model    = get_model()
    filename = os.path.basename(pdf_path)

    sentinel_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, filename))
    try:
        existing = client.retrieve(collection_name=COLLECTION, ids=[sentinel_id])
        if existing:
            return 0
    except Exception:
        pass

    text = _extract_text(pdf_path)
    if not text.strip():
        raise ValueError(f"No text could be extracted from '{filename}'. Is it a scanned PDF?")

    chunks     = _chunk_text(text)
    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=False)

    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i].tolist(),
            payload={"text": chunks[i], "filename": filename, "is_sentinel": False},
        )
        for i in range(len(chunks))
    ]

    points.append(
        models.PointStruct(
            id=sentinel_id,
            vector=[0.0] * VECTOR_DIM,
            payload={"text": "__sentinel__", "filename": filename, "is_sentinel": True},
        )
    )

    client.upsert(collection_name=COLLECTION, points=points)
    return len(chunks)


def retrieve(query: str) -> list[str]:
    ensure_collection()
    client = get_client()
    model  = get_model()

    total = client.get_collection(COLLECTION).points_count
    if total == 0:
        return []

    query_vec = model.encode(query).tolist()

    not_sentinel = models.Filter(
        must=[
            models.FieldCondition(
                key="is_sentinel",
                match=models.MatchValue(value=False),
            )
        ]
    )

    try:
        response = client.query_points(
            collection_name=COLLECTION,
            query=query_vec,
            limit=TOP_K,
            score_threshold=0.20,
            query_filter=not_sentinel,
        )
        hits = response.points
    except AttributeError:
        hits = client.search(
            collection_name=COLLECTION,
            query_vector=query_vec,
            limit=TOP_K,
            score_threshold=0.20,
            query_filter=not_sentinel,
        )

    return [h.payload["text"] for h in hits]


def collection_count() -> int:
    try:
        ensure_collection()
        client = get_client()
        result = client.count(
            collection_name=COLLECTION,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="is_sentinel",
                        match=models.MatchValue(value=False),
                    )
                ]
            ),
            exact=True,
        )
        return result.count
    except Exception:
        return 0