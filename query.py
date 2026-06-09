"""
query.py
--------
Retrieval + grounded generation for the Unofficial Guide RAG system.

Flow (Milestones 4 + 5 of the assignment):
  1. retrieve(question, k)  -> top-k chunks from ChromaDB by cosine similarity
  2. build_context(chunks)  -> a numbered, source-labeled context block
  3. generate(...)          -> Groq llama-3.3-70b-versatile, grounded ONLY in
                               that context (refuses when context is insufficient)
  4. ask(question)          -> {"answer", "sources", "chunks"} for the interface

Grounding is enforced two ways:
  - a strict system prompt that forbids outside knowledge and mandates a fixed
    refusal string when the context doesn't answer the question, and
  - source attribution that is appended PROGRAMMATICALLY from each retrieved
    chunk's metadata, so citations never depend on the model remembering to add
    them.

Run directly for a CLI:
    python query.py "Why do students avoid Professor Baugh for chemistry?"
    python query.py            # interactive prompt loop
"""

import os
import sys

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Config

CHROMA_DIR  = "chroma_db"
COLLECTION  = "unofficial_guide"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "llama-3.3-70b-versatile"
TOP_K       = 5

# Exact string the model must emit when the retrieved context can't answer the
# question. We detect this verbatim to suppress (irrelevant) source citations.
REFUSAL = "I don't have enough information on that."

SYSTEM_PROMPT = f"""You are the Unofficial Guide, a question-answering assistant for \
UCLA students. You answer questions about UCLA professors and courses using ONLY \
the student reviews provided in the context below.

Rules:
- Use ONLY the information in the provided reviews. Do not use any outside or prior \
knowledge about these professors, courses, or UCLA.
- If the reviews are about a different professor/topic than the question, or do not \
contain enough information to answer, reply with EXACTLY this sentence and nothing \
else: "{REFUSAL}"
- When the reviews disagree, say so explicitly and summarize the range of opinion \
rather than presenting one side as the consensus.
- Refer to professors by name. Keep the answer concise (2-5 sentences) and concrete, \
quoting or paraphrasing what reviewers actually said.
- Do not invent ratings, courses, or quotes that are not in the context."""

# Lazily-initialized singletons (loaded once, reused across calls)

_model = None
_collection = None
_groq_client = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_collection(COLLECTION)
    return _collection


def _get_groq():
    """Create the Groq client on first use so retrieval-only usage needs no key."""
    global _groq_client
    if _groq_client is None:
        load_dotenv()
        key = os.getenv("GROQ_API_KEY")
        if not key or key == "your_key_here":
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
                "free key from https://console.groq.com (retrieval works without it, "
                "but generating answers does not)."
            )
        from groq import Groq
        _groq_client = Groq(api_key=key)
    return _groq_client


# Retrieval

def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """
    Embed the question and return the top-k most similar chunks from ChromaDB.

    Each result: {text, metadata, distance, similarity}.
    distance is cosine distance (0 = identical); similarity = 1 - distance.
    """
    collection = _get_collection()
    embedding = _get_model().encode(question).tolist()
    res = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append(
            {
                "text": doc,
                "metadata": meta,
                "distance": round(dist, 4),
                "similarity": round(1 - dist, 4),
            }
        )
    return out


def build_context(chunks: list[dict]) -> str:
    """Numbered, source-labeled context block handed to the LLM."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        header = f"[Review {i} — {m.get('professor', 'Unknown')}, {m.get('department', '')}]"
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n".join(blocks)


def format_source(meta: dict) -> str:
    """One human-readable citation line from a chunk's metadata."""
    prof = meta.get("professor", "Unknown professor")
    dept = meta.get("department", "")
    rating = meta.get("overall_rating", "")
    date = meta.get("date", "")
    url = meta.get("url", "")
    label = prof
    if dept:
        label += f" ({dept})"
    if rating != "":
        label += f" — {rating}/5.0"
    extra = " · ".join(x for x in [date, url] if x)
    return f"{label}" + (f" — {extra}" if extra else "")


def dedupe_sources(chunks: list[dict]) -> list[str]:
    """Unique source lines, preserving retrieval order (best match first)."""
    seen, sources = set(), []
    for c in chunks:
        line = format_source(c["metadata"])
        if line not in seen:
            seen.add(line)
            sources.append(line)
    return sources


# Generation

def generate(question: str, chunks: list[dict]) -> str:
    """Call Groq with the grounded system prompt + retrieved context."""
    context = build_context(chunks)
    user_msg = (
        f"Student reviews (context):\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the reviews above."
    )
    client = _get_groq()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,          # low — we want faithful, not creative, answers
        max_tokens=400,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()


def ask(question: str, k: int = TOP_K) -> dict:
    """
    End-to-end: retrieve -> generate -> attach sources.

    Returns {"answer", "sources", "chunks"}. Sources are appended
    programmatically from retrieved metadata, and suppressed when the model
    declines to answer (so an out-of-scope question doesn't show bogus citations).
    """
    chunks = retrieve(question, k=k)
    answer = generate(question, chunks)

    declined = REFUSAL.rstrip(".") in answer
    sources = [] if declined else dedupe_sources(chunks)

    return {"answer": answer, "sources": sources, "chunks": chunks}


# CLI

def _print_result(question: str, result: dict) -> None:
    print(f"\nQ: {question}\n")
    print(f"A: {result['answer']}\n")
    if result["sources"]:
        print("Retrieved from:")
        for s in result["sources"]:
            print(f"  • {s}")
    else:
        print("Retrieved from: (none — system declined / out of scope)")
    print("-" * 70)


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        _print_result(question, ask(question))
        return

    print("Unofficial Guide — ask about UCLA professors (Ctrl-C or 'quit' to exit)")
    try:
        while True:
            question = input("\n> ").strip()
            if question.lower() in {"quit", "exit", "q", ""}:
                break
            _print_result(question, ask(question))
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
