"""
ingest.py
---------
Ingestion pipeline for the Unofficial Guide RAG system.

Reads:  data/rmp_raw.json
Chunks: splits each review into overlapping text chunks
Embeds: uses sentence-transformers all-MiniLM-L6-v2 (local, no API key)
Stores: upserts chunks into ChromaDB (local, persisted to chroma_db/)

Install:
  pip install sentence-transformers chromadb langchain-text-splitters
"""

import html
import json
import hashlib
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Config

RMP_PATH      = Path("documents/rmp_raw.json")
CHROMA_DIR    = "chroma_db"
COLLECTION    = "unofficial_guide"

# Chunking config — single source of truth, matches planning.md.
# 1200 chars ≈ 300 tokens at ~4 chars/token. These reviews average ~392 chars,
# so a 1200-char window holds an entire review (header stats + comment) in one
# chunk; the 200-char overlap only matters if a longer source is added later.
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200
EMBED_MODEL   = "all-MiniLM-L6-v2"
BATCH_SIZE    = 64

# Helper functions

def clean_text(text: str) -> str:
    """
    Final cleaning pass before chunking. The RMP GraphQL API returns comments
    with un-decoded HTML entities (e.g. `Worst &quot;teacher&quot; I&#39;ve had`);
    if left in, they survive into the embeddings and show up verbatim in cited
    answers. Decode them and normalize whitespace.
    """
    text = html.unescape(text)            # &quot; -> "  ·  &amp; -> &  ·  &#39; -> '
    text = text.replace(" ", " ")    # non-breaking space -> regular space
    text = re.sub(r"[ \t]+", " ", text)   # collapse runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excess blank lines
    return text.strip()


def make_chunk_id(text: str, index: int) -> str:
    """Stable unique ID for a chunk based on its content + position."""
    hash_str = hashlib.md5(f"{text[:80]}_{index}".encode()).hexdigest()[:12]
    return f"chunk_{hash_str}"


def flatten_metadata(meta: dict) -> dict:
    """
    ChromaDB only accepts str/int/float/bool metadata values.
    Flatten lists and None values so nothing gets rejected.
    """
    flat = {}
    for k, v in meta.items():
        if v is None:
            flat[k] = ""
        elif isinstance(v, list):
            flat[k] = ", ".join(str(x) for x in v)
        elif isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            flat[k] = str(v)
    return flat

# Main

def main():
    # Load raw documents
    print(f"Loading {RMP_PATH}...")
    with open(RMP_PATH, encoding="utf-8") as f:
        documents = json.load(f)
    print(f"  {len(documents)} raw documents loaded")

    # Set up text splitter
    # RecursiveCharacterTextSplitter tries to split on paragraphs first,
    # then sentences, then words — keeps chunks semantically coherent.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Chunk all documents
    print("\nChunking documents...")
    all_chunks   = []
    all_ids      = []
    all_metadata = []

    for doc in documents:
        text   = clean_text(doc.get("text", ""))
        source = doc.get("source", "ratemyprofessor")
        url    = doc.get("url", "")
        date   = doc.get("date", "")
        topic  = doc.get("topic", "professors")
        meta   = doc.get("metadata", {})

        if not text:
            continue

        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(chunk, len(all_chunks))

            # Merge doc-level fields + nested metadata into one flat dict
            chunk_meta = flatten_metadata({
                "source":     source,
                "url":        url,
                "date":       date,
                "topic":      topic,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **meta,
            })

            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadata.append(chunk_meta)

    print(f"  {len(all_chunks)} chunks created from {len(documents)} documents")

    # Load embedding model (downloads once, cached locally after)
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    print("  (first run downloads ~90MB — subsequent runs are instant)")
    model = SentenceTransformer(EMBED_MODEL)

    # Set up ChromaDB
    print(f"\nSetting up ChromaDB at ./{CHROMA_DIR}/")
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    # Get or create collection
    # Delete and recreate if re-running so we don't get duplicate chunks
    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        print(f"  Collection '{COLLECTION}' exists — clearing for fresh ingest")
        client.delete_collection(COLLECTION)

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for retrieval
    )
    print(f"  Collection '{COLLECTION}' created")

    # Embed and upsert in batches
    print(f"\nEmbedding and upserting {len(all_chunks)} chunks (batch size={BATCH_SIZE})...")
    total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_num  = i // BATCH_SIZE + 1
        batch_text = all_chunks[i : i + BATCH_SIZE]
        batch_ids  = all_ids[i : i + BATCH_SIZE]
        batch_meta = all_metadata[i : i + BATCH_SIZE]

        print(f"  Batch {batch_num}/{total_batches} — embedding {len(batch_text)} chunks...")

        # Embed
        embeddings = model.encode(
            batch_text,
            show_progress_bar=False,
        ).tolist()

        # Upsert into ChromaDB
        collection.upsert(
            ids=batch_ids,
            documents=batch_text,
            embeddings=embeddings,
            metadatas=batch_meta,
        )

    # Verify
    count = collection.count()
    print(f"\nDone.")
    print(f"  Chunks in ChromaDB : {count}")
    print(f"  Collection name    : {COLLECTION}")
    print(f"  Persisted to       : ./{CHROMA_DIR}/")

    # Quick retrieval test
    print("\nRunning a quick retrieval test...")
    test_query = "Which professor gives the most useful feedback?"
    test_embedding = model.encode(test_query).tolist()
    results = collection.query(
        query_embeddings=[test_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    print(f"\nQuery: '{test_query}'")
    print("Top 3 retrieved chunks:\n")
    for j, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        prof = meta.get("professor", "unknown")
        score = round(1 - dist, 3)
        print(f"  [{j+1}] Professor: {prof}  |  similarity: {score}")
        print(f"       {doc[:120]}...")
        print()


if __name__ == "__main__":
    main()