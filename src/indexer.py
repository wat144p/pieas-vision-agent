"""
Module 3: Vector Indexing (ChromaDB)

Embeds image descriptions using Sentence Transformers and stores them in ChromaDB
for semantic search. Provides incremental indexing and search functions.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any

# ---- workaround for old sqlite3 on some systems ----
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # system sqlite3 is fine

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from . import database

logger = logging.getLogger(__name__)

# Configuration
CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "pieas_images"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Load embedding model (runs on CPU)
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Initialise ChromaDB persistent client
client = chromadb.PersistentClient(
    path=str(CHROMA_PATH),
    settings=Settings(anonymized_telemetry=False),
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}  # cosine similarity
)


def _build_document_text(desc: Dict[str, Any]) -> str:
    """Combine relevant description fields into a single text for embedding."""
    parts = []

    scene = desc.get("scene_description")
    if scene:
        parts.append(str(scene))

    category = desc.get("category")
    if category:
        parts.append("Category: " + str(category))

    tags = desc.get("relevant_tags")
    if tags:
        if isinstance(tags, list):
            parts.append("Tags: " + ", ".join(tags))
        else:
            parts.append("Tags: " + str(tags))

    visible = desc.get("visible_text")
    if visible:
        if isinstance(visible, list):
            parts.append("Visible text: " + ", ".join(visible))
        else:
            parts.append("Visible text: " + str(visible))

    return ". ".join(parts)


def index_single(image_hash: str, description: Dict[str, Any]) -> None:
    """
    Embed a single image description and upsert into ChromaDB.
    If the hash already exists, it will be updated.
    """
    doc_text = _build_document_text(description)
    embedding = embedder.encode(doc_text).tolist()

    # Metadata to store alongside the vector
    metadata = {
        "image_hash": image_hash,
        "category": description.get("category", "Other"),
        # You can add more metadata fields if needed
    }

    collection.upsert(
        ids=[image_hash],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[doc_text],
    )
    logger.debug(f"Indexed {image_hash[:12]}...")


def index_all_images() -> int:
    """
    One‑time batch index: embed all images that already have descriptions.
    Skips images already present in ChromaDB (based on ID).
    Returns the number of newly indexed images.
    """
    conn = database.get_connection()
    rows = conn.execute("""
        SELECT i.hash, d.description_json
        FROM images i
        JOIN descriptions d ON i.hash = d.image_hash
        WHERE i.analyzed = 1
    """).fetchall()
    conn.close()

    # Check which hashes already exist in ChromaDB
    existing_ids = set()
    try:
        existing_data = collection.get(include=[])  # only IDs
        existing_ids = set(existing_data["ids"])
    except Exception:
        pass  # collection might be empty

    count = 0
    for row in rows:
        img_hash = row["hash"]
        if img_hash in existing_ids:
            continue
        try:
            desc = json.loads(row["description_json"])
            index_single(img_hash, desc)
            count += 1
        except Exception as e:
            logger.error(f"Failed to index {img_hash[:12]}: {e}")

    logger.info(f"Bulk index complete. {count} new images indexed.")
    return count


def search(
    query_text: str,
    top_k: int = 10,
    category_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Perform semantic search over indexed images.
    Returns a list of dicts with keys: id, distance, metadata, document.
    """
    query_embedding = embedder.encode(query_text).tolist()

    where_filter = None
    if category_filter:
        where_filter = {"category": category_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["metadatas", "documents", "distances"]
    )

    # Flatten results (we always pass a single query embedding)
    out = []
    if results["ids"] and results["ids"][0]:
        for i, img_id in enumerate(results["ids"][0]):
            out.append({
                "id": img_id,
                "distance": results["distances"][0][i] if results["distances"] else None,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "document": results["documents"][0][i] if results["documents"] else "",
            })
    return out


# ---------- CLI for bulk indexing ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting one‑time bulk index of all analyzed images...")
    indexed = index_all_images()
    logger.info(f"Done. {indexed} images indexed.")