# The Unofficial Guide — Project 1

A Retrieval-Augmented Generation (RAG) system that makes the unofficial, student-to-student knowledge in **UCLA Rate My Professors reviews** searchable and answerable. Ask a plain-language question — *"Why do students say to avoid Professor Baugh for chemistry?"* — and get a grounded, cited answer drawn from real reviews.

> **Status:** Complete. Full pipeline built and run end-to-end — ingestion → chunking → embedding → retrieval → grounded generation → Gradio interface (`scrap_rmp.py`, `ingest.py`, `query.py`, `app.py`; ChromaDB has 596 chunks). All 5 evaluation questions plus a stress-test failure case have been run and documented below; reproduce with `python eval.py` (needs a free `GROQ_API_KEY` in `.env`). *Remaining for the student: record the 3–5 min demo video.*

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your GROQ_API_KEY (free at console.groq.com)

python scrap_rmp.py           # (optional) re-collect reviews → documents/rmp_raw.json
python ingest.py              # chunk + embed + load into ChromaDB (chroma_db/)

# Ask a question (needs GROQ_API_KEY):
python query.py "Why do students avoid Professor Baugh for chemistry?"
python query.py               # interactive prompt loop
python app.py                 # launch the Gradio web UI at http://localhost:7860
python eval.py                # run all 5 evaluation questions end-to-end
```

---

## Domain

**UCLA professor and course reviews.** This system covers the experiential knowledge students share about who to take a class with and what the class is actually like — teaching style, exam fairness, grading harshness, and workload. This knowledge is valuable because the official course catalog only lists titles and descriptions; it never tells you that a Chem 20A professor's exams come "out of left field" relative to the readings, or that a math professor's midterm proofs are "not in the same atmosphere" as lecture. It's hard to find through official channels because the university has no reason to publish it, and hard to use even on Rate My Professors itself, where opinions are unstructured, contradictory, and scattered one professor page at a time with no way to query across all of them at once.

---

## Document Sources

All documents come from **Rate My Professors** (`ratemyprofessors.com`), collected through its public GraphQL API by `scrap_rmp.py`. One "document" = one individual student review, stored as a text blob combining the professor's aggregate stats (rating, difficulty, would-take-again, course, grade) with the student's free-text comment.

- **Scope:** UCLA (school ID 1075), the 60 most-reviewed professors with ≥5 ratings, up to 10 reviews each.
- **Total:** **596 review-documents · 60 professors · 26 departments · dates 2005–2026.** Mean length ≈ 392 characters.
- **Why the GraphQL API:** the `RateMyProfessorAPI` PyPI package now returns HTTP 403; the GraphQL endpoint still responds with a browser `User-Agent` and RMP's public `test:test` token.

| # | Source | Type | URL or file path |
|---|--------|------|------------------|
| 1 | RMP — Mathematics reviews (70) | API → JSON | `documents/rmp_raw.json` |
| 2 | RMP — English reviews (60) | API → JSON | `documents/rmp_raw.json` |
| 3 | RMP — Chemistry reviews (59) | API → JSON | `documents/rmp_raw.json` |
| 4 | RMP — History reviews (49) | API → JSON | `documents/rmp_raw.json` |
| 5 | RMP — Anthropology reviews (49) | API → JSON | `documents/rmp_raw.json` |
| 6 | RMP — Psychology reviews (40) | API → JSON | `documents/rmp_raw.json` |
| 7 | RMP — Physics reviews (20) | API → JSON | `documents/rmp_raw.json` |
| 8 | RMP — Computer Science reviews (20) | API → JSON | `documents/rmp_raw.json` |
| 9 | RMP — Accounting / Communication / Sociology / Life Science (20 each) | API → JSON | `documents/rmp_raw.json` |
| 10 | RMP — Economics, Mech. Eng., Dental, Humanities, +others | API → JSON | `documents/rmp_raw.json` |

**Source URL pattern:** each document links back to `https://www.ratemyprofessors.com/professor/{legacy_id}`.

---

## Chunking Strategy

**Strategy (of the three covered in class — fixed-size, semantic, recursive): Recursive.** I use LangChain's `RecursiveCharacterTextSplitter`, which tries a hierarchy of separators in order — paragraph `\n\n` → line `\n` → sentence `. ` → word ` ` → character — and only falls back to a hard character cut as a last resort, so it respects natural boundaries (unlike *fixed-size*, which cuts blindly every N chars) without the cost/complexity of *semantic* chunking (embedding the text to group it by meaning). Because my `chunk_size` (1200) is larger than essentially every review (mean ≈ 392 chars, max 554), the splitter rarely fires an actual split — each review lands in one chunk. So in practice this is **recursive splitting configured to yield one-review-per-chunk**, which is the intent: one review = one unit of meaning.

**Chunk size:** 1200 characters (≈ 300 tokens at ~4 chars/token)
**Overlap:** 200 characters
**Splitter:** `RecursiveCharacterTextSplitter` (langchain-text-splitters), separators `["\n\n", "\n", ". ", " ", ""]` — it breaks on paragraph → line → sentence → word boundaries, never mid-word.

**Preprocessing before chunking:** each review is assembled by `scrap_rmp.py` into a clean labeled blob (`Professor: … / Department: … / Overall rating: … / Student review: <comment>`); empty / "No Comments" reviews are dropped. `ingest.py` then runs a final `clean_text()` pass that **decodes HTML entities** (`html.unescape()` turns `Worst &quot;teacher&quot; I&#39;ve had` into `Worst "teacher" I've had`) and normalizes whitespace before chunking — this removed entities from **16 of 596** reviews that would otherwise have survived into the embeddings and citations. Metadata (professor, department, rating, difficulty, would-take-again, course, grade, tags, source URL, date) is flattened to Chroma-compatible types and attached to every chunk.

**Why these choices fit the documents:** the corpus is **short, self-contained reviews** (mean ≈ 392 chars, max 554), not long-form guides. A 1200-char window holds an entire review — header stats *and* comment — in a single chunk, which keeps each student's opinion attached to the professor it's about. Splitting smaller would orphan a sentence like *"his tests are unrelated to the readings"* from the professor's name, making it un-retrievable and un-citable. The 200-char overlap rarely triggers (reviews seldom exceed 1200 chars) but is retained so the same config degrades gracefully if a longer review or a long-form source is added later.

**Final chunk count:** **596 chunks** in ChromaDB (collection `unofficial_guide`) — roughly one chunk per review, confirmed with `collection.count()`.

> **Implementation note:** the `CHUNK_SIZE` / `CHUNK_OVERLAP` constants in `ingest.py` are now set to `1200` / `200` and passed directly into the `RecursiveCharacterTextSplitter`, so the config has a single source of truth that matches this spec. (Earlier in development these constants were leftover `300` / `50` values that the splitter ignored in favor of hard-coded `1200` / `200`; reconciling them is noted in *Spec Reflection* below.)

### Sample chunks (with source)

Five representative chunks straight from the store, each labeled with its source document (professor + RMP URL). Each is a complete, self-contained review — header stats + one student comment — which is the whole point of the one-review-per-chunk strategy.

1. **Source: Sihao Ma — Mathematics** · `ratemyprofessors.com/professor/3145448`
   > Professor: Sihao Ma / Department: Mathematics / Overall rating: 1.7 / Difficulty: 4.7 / Would take again: 20.7% / Course: Math115A / Grade: Drop/Withdrawal / **Student review:** `..............` *(a content-free review — see the retrieval-noise note in the Failure Case)*

2. **Source: Delroy A. Baugh — Chemistry** · `ratemyprofessors.com/professor/473701`
   > … Overall rating: 1.5 / Difficulty: 4.1 / Course: CHEM20A / **Student review:** "His lectures relate very little to the textbook and homework problems. He only teaches things that he finds interesting. He started off the General Chemistry course talking about quantum mechanics rather than actual general chemistry. He is lazy and makes his TAs do all the work for him."

3. **Source: Jordan Mendler — Computer Science** · `ratemyprofessors.com/professor/1909266`
   > … Overall rating: 5.0 / Difficulty: 2.4 / Course: CM224 / Grade: A+ / **Student review:** "Great"

4. **Source: Stephen Ross — Economics** · `ratemyprofessors.com/professor/1576103`
   > … Overall rating: 4.8 / Difficulty: 1.3 / Would take again: 91.3% / Course: ECON185 / Grade: B / **Student review:** "Professor Ross is engaging and passionate. His lectures are informative and relevant… Overall, a great course to take."

5. **Source: Mary Corey — History** · `ratemyprofessors.com/professor/675237`
   > … Overall rating: 4.4 / Difficulty: 2.7 / Course: HIST140 / Grade: A / **Student review:** "One of my favorite professors at UCLA! I learned so much from her lectures. She's awesome. :)"

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (384-dimensional, runs locally with no API key and no rate limits). Vectors are stored in **ChromaDB** with cosine similarity (`hnsw:space = "cosine"`), persisted to `chroma_db/`.

**Why:** it's fast, free, private, and good enough on short English opinion text — ideal for an iterate-locally student project where the corpus is small.

**Production tradeoff reflection:** if I were deploying for real users and cost weren't a constraint, I'd weigh a larger hosted model (e.g. OpenAI `text-embedding-3-large` or a Voyage domain model) against MiniLM on:
- **Domain accuracy** — MiniLM is weakest exactly where this corpus lives: slang, professor nicknames, and course codes ("Chem 20A", "weed-out"). A larger model embeds these more faithfully, improving recall on the hardest queries.
- **Context length** — irrelevant for short reviews today, but it matters if I add syllabi/FAQs later.
- **Multilingual** — unnecessary for an English-only UCLA corpus, so not worth paying for.
- **Latency & local vs. API** — MiniLM's zero-cost, no-rate-limit, privacy-preserving local inference is a genuine advantage; a hosted model adds per-call latency and a key to manage. At this scale MiniLM's quality is sufficient, so the local tradeoff wins.

---

## Retrieval — Test Examples

Three test queries with their **top-3 retrieved chunks** (cosine distance; lower = closer). Reproduce with `query.retrieve(q, k=3)`.

**Example 1 — "Why do students recommend avoiding Professor Delroy A. Baugh for general chemistry?"**
| # | dist | source | chunk (excerpt) |
|---|------|--------|-----------------|
| 1 | 0.273 | Baugh · CHEM20A | "His lectures relate very little to the textbook and homework… started off the General Chemistry course talking about quantum mechanics… makes his TAs do all the work." |
| 2 | 0.293 | Baugh · CHEM20A | "Worst 'teacher' I've ever had… His test questions were completely from out of left field, and his lectures make [no sense]…" |
| 3 | 0.300 | Baugh · CHEM20A | "A good chemist but a bad teacher. The tests are entirely from lecture slides, but the slides have nothing to do with the book…" |

> **Why these are relevant:** all three are Baugh's **CHEM20A (general chemistry)** reviews, and each independently states the *reason to avoid him* the query asks for — lectures disconnected from the book, exams "out of left field," over-reliance on TAs. The query shares almost no exact words with the chunks ("avoiding" / "recommend" appear nowhere), yet semantic similarity surfaces them because the *meaning* — a bad-teaching complaint — matches. Distances 0.27–0.30 are tight.

**Example 2 — "Which Computer Science professor do students describe as caring and chill, and why?"**
| # | dist | source | chunk (excerpt) |
|---|------|--------|-----------------|
| 1 | 0.383 | Jordan Mendler · CS131 | "Super chill professor… He treats us like adults…" |
| 2 | 0.397 | Jordan Mendler · CS131 | "Jordan (as he likes to be called) is awesome… always doing anything he can to help. He even [has] an Ice Cream party at the end of class…" |
| 3 | 0.429 | Jordan Mendler · MGMNTX417 | "Mr Mendler is very caring. Really great teacher… he loves mentoring students." |

> **Why these are relevant:** the query's two attributes — "caring" and "chill" — map directly onto chunk 1 ("super chill", "treats us like adults") and chunk 3 ("very caring"), and chunk 2 supplies the *why* ("always doing anything he can to help"). All three are the **same professor** (Jordan Mendler), so the system can answer with one confident subject rather than a blend. Note the embedding even pulls a chunk from a *different* course (MGMNTX417) for the same professor — fine here, but the root of the aggregation/cross-course issues discussed in the Failure Case.

**Example 3 — "What do students say about Professor Sihao Ma's exams?"**
| # | dist | source | chunk (excerpt) |
|---|------|--------|-----------------|
| 1 | 0.364 | Sihao Ma · Math32A | "I don't know why all other reviewers are so furious… His homework and exams are not hard at all if you understand the concepts…" |
| 2 | 0.378 | Sihao Ma · Math115A | "I would avoid this professor at all cost… Exams are extremely difficult and nowhere near what we do in class…" |
| 3 | 0.416 | Sihao Ma · Math115A | `..............` (content-free review) |

> Retrieval correctly pulls **both sides** of Ma's polarized reviews (chunk 1 defends, chunk 2 attacks) — which is why generation produced a balanced answer. Chunk 3 is a content-free review that still scored into the top-3; see the retrieval-noise note in the Failure Case.

---

## Grounded Generation

Implemented in `query.py` (`ask()` → `retrieve()` → `generate()`).

**LLM:** Groq `llama-3.3-70b-versatile` (free tier, OpenAI-compatible), keyed from `GROQ_API_KEY` in `.env`, called at `temperature=0.1` so answers stay faithful rather than creative.

**Grounding is enforced two ways, not one:**

1. **A strict system prompt** (see `SYSTEM_PROMPT` in `query.py`) that tells the model to use *only* the supplied reviews, forbids outside knowledge, requires it to surface disagreement instead of picking a side, and mandates an **exact refusal string** when the context can't answer: *"I don't have enough information on that."* The retrieved chunks are passed in a numbered, professor-labeled context block.
2. **Programmatic source attribution.** Citations are *not* left to the model to volunteer — `ask()` builds the "Retrieved from" list directly from each retrieved chunk's metadata (professor, department, overall rating, date, RMP URL) and de-duplicates it. So every answer carries its sources even if the model forgets to mention them.

**Out-of-scope handling:** when the model emits the refusal string (detected verbatim), `ask()` **suppresses the source list** — so a question the corpus can't answer doesn't get decorated with irrelevant professor citations. The Gradio UI then shows "*(no sources — the system declined to answer this question)*".

### Example responses (source attribution visible)

These are outputs from a real run of `python query.py` / `app.py` (wording varies slightly run-to-run since generation is non-deterministic; the sources and grounding do not). Sources are appended programmatically from retrieved metadata.

**Example A — grounded answer with citations**
```
Q: Which Computer Science professor do students describe as caring and chill, and why?

A: Students describe Jordan Mendler as caring and chill. According to the reviews,
   Jordan Mendler is "very caring" and "super chill", with one reviewer noting that
   "he treats us like adults" and another stating that "he is very nice and always
   doing anything he can to help".

Retrieved from:
  • Jordan Mendler (Computer Science) — 5/5.0 — 2019-01-17 · ratemyprofessors.com/professor/1909266
  • Jordan Mendler (Computer Science) — 5/5.0 — 2019-11-02 · ratemyprofessors.com/professor/1909266
  • Jordan Mendler (Computer Science) — 5/5.0 — 2019-05-14 · ratemyprofessors.com/professor/1909266
  • Jordan Mendler (Computer Science) — 5/5.0 — 2018-11-14 · ratemyprofessors.com/professor/1909266
```

**Example B — grounded answer with citations**
```
Q: Why do students recommend avoiding Professor Delroy A. Baugh for general chemistry?

A: Students recommend avoiding Professor Delroy A. Baugh for general chemistry because
   he "cannot teach to save his life" and his lectures "make no sense whatsoever." He
   teaches topics not relevant to the course (he "started off the General Chemistry
   course talking about quantum mechanics"), his test questions are "completely from
   out of left field," and he relies heavily on his TAs.

Retrieved from:
  • Delroy A. Baugh (Chemistry) — 1.5/5.0 — 2017-10-24 · ratemyprofessors.com/professor/473701
  • Delroy A. Baugh (Chemistry) — 1.5/5.0 — 2011-12-20 · ratemyprofessors.com/professor/473701
  • Delroy A. Baugh (Chemistry) — 1.5/5.0 — 2010-11-05 · ratemyprofessors.com/professor/473701
  • Delroy A. Baugh (Chemistry) — 1.5/5.0 — 2011-11-15 · ratemyprofessors.com/professor/473701
```

**Example C — out-of-scope query (refusal, no fabrication)**
```
Q: Which UCLA dorm has the best dining hall?

A: I don't have enough information on that.

Retrieved from: (none — system declined / out of scope)
```

---

## Query Interface

Two interchangeable front-ends, both calling `query.ask()`:

**Gradio web UI (`python app.py` → http://localhost:7860).** Fields:

| Field | Type | Role |
|-------|------|------|
| **Your question** | input textbox | the plain-language question (also submits on Enter) |
| **Ask** | button | triggers retrieval + generation |
| **Answer** | output textbox (8 lines) | the grounded answer |
| **Retrieved from (sources)** | output textbox (6 lines) | bulleted citations, or the decline notice on out-of-scope questions |

Five clickable example questions are wired under the input so a viewer can drive it with no instructions.

**CLI (`python query.py "<question>"`, or no-arg for an interactive loop)** — prints the answer followed by a `Retrieved from:` bullet list.

**Sample interaction transcript (Gradio):**
```
[ Your question ]  Is Professor Stephen Ross's economics course easy or hard?
        ( Ask )

[ Answer ]
  According to the reviews, Professor Stephen Ross's economics course is considered
  easy. Reviewers mention that "not much work is required" and describe it as an
  "easy 1 unit course" with a difficulty rating of 1.3/5.0.

[ Retrieved from (sources) ]
  • Stephen Ross (Economics) — 4.8/5.0 — 2022-03-29 · ratemyprofessors.com/professor/1576103
  • Stephen Ross (Economics) — 4.8/5.0 — 2023-09-04 · ratemyprofessors.com/professor/1576103
  • Stephen Ross (Economics) — 4.8/5.0 — 2024-11-21 · ratemyprofessors.com/professor/1576103
```

---

## Evaluation Report

The 5 test questions and expected answers are fixed in `planning.md` (grounded in real review text). The table below is the **actual** end-to-end run — reproduce it with `python eval.py`. Full answers and the exact retrieved chunks (with distances) are in the per-question notes that follow.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What do students say about Prof. Sihao Ma's exams? | Polarized; mostly "extremely difficult, Qs unlike lecture, harsh grading," minority defend him. 1.7/5, diff 4.7. | Correctly reported **mixed opinions**: quoted both *"not hard if you understand the concepts"* and *"extremely difficult… averages in the 50s… nowhere near what we do in class."* | **Relevant** — 5/5 chunks are Ma; the set included both a defender and detractors | **Accurate** — captured the polarization rather than picking a side |
| 2 | Why avoid Prof. Delroy A. Baugh for gen chem? | Lectures ≠ textbook/homework; teaches above Chem 20A level; exams "out of left field." 1.5/5, diff 4.1. | Avoid him — *"cannot teach to save his life,"* lectures unrelated to the book, relies on his TAs, and *"talked about quantum mechanics rather than actual general chemistry"* | **Relevant** — 4/5 chunks are Chem 20A; 1 is his 113A course, but it didn't mislead the answer | **Accurate** — every claim, including the quantum-mechanics detail, traces to a real Chem 20A review |
| 3 | Which CS prof is "caring and chill," and why? | Jordan Mendler — caring, super chill, treats students like adults, ice-cream party. 5.0/5. | **Jordan Mendler** — *"very caring," "super chill," "treats us like adults,"* always willing to help | **Relevant** — 5/5 chunks are Mendler | **Accurate** (didn't surface the ice-cream-party detail, but everything stated is grounded) |
| 4 | Is Prof. Stephen Ross's econ course easy or hard? | Easy — 4.8/5, diff 1.3, ~91% would take again; engaging. | **Easy** — difficulty *1.3/5*, *"not much work required," "easy 1 unit course"* | **Relevant** — 5/5 chunks are Ross | **Accurate** — right conclusion, well-grounded. *(In some runs the model also frames the repeated aggregate 1.3 as "consistent across all reviews" — the intermittent aggregate-conflation slip described in the Failure Case.)* |
| 5 | Which UCLA dorm has the best dining hall? | Out of scope — system should decline (no housing/dining in corpus). | *"I don't have enough information on that."* — declined; **no sources shown** | **Off-target** (correctly) — nearest chunks are unrelated professor reviews at distance ≥ 0.50 | **Accurate** — correct refusal instead of a hallucinated answer |

**Retrieval quality:** Relevant / Partially relevant / Off-target  ·  **Response accuracy:** Accurate / Partially accurate / Inaccurate

**Per-question retrieval detail** (cosine distance; lower = closer):

- **Q1 Ma** — 5/5 Sihao Ma, distances **0.31–0.36**. Both sides present (chunk 1 defends, chunk 2 "avoid at all cost"). One low-content chunk (`".............."`, a Drop/Withdrawal review) still got retrieved at rank 5 — see the secondary failure below.
- **Q2 Baugh** — distances **0.27–0.31**; chunks 1–4 are Chem 20A, chunk 5 is his 113A Quantum Mechanics course. The answer's "quantum mechanics" detail came from the *Chem 20A* chunk 1 (*"started off the General Chemistry course talking about quantum mechanics"*), not the 113A chunk, so it's correctly grounded.
- **Q3 Mendler** — 5/5 Jordan Mendler, distances **0.38–0.47**.
- **Q4 Ross** — 5/5 Stephen Ross, distances **0.27–0.31**.
- **Q5 dorm** — 5 unrelated professor reviews, distances **0.50–0.54** (all above the ~0.5 weak-match line) → correctly triggered refusal.

---

## Failure Case Analysis

**First, two honest notes on what did *not* fail.** (1) I expected **Q1 (Sihao Ma)** to fail from *retrieval-sampling bias* — top-k landing on one side of his polarized reviews. It didn't: retrieval returned both a defender and detractors, and the prompt's "surface disagreement, don't pick a side" instruction produced a genuinely *mixed* answer. (2) While first writing this report I flagged **Q2 (Baugh)** as a "grounded hallucination" because the answer mentioned *quantum mechanics in a general chemistry class* and a 113A chunk was in the retrieval set. Inspecting the actual chunks proved me wrong — the Chem 20A chunk itself says *"started off the General Chemistry course talking about quantum mechanics rather than actual general chemistry,"* so the detail was correctly grounded. **Lesson: verify a suspected hallucination against the retrieved chunk before calling it one.**

**The real failure (primary): global-aggregation questions.** A deliberately harder probe (run it via `python eval.py`, which includes it as a stress test):

> **Q: "Which UCLA professor has the highest overall rating?"**
> **A: "Stephen Dickey has the highest overall rating of 4.8 out of 5.0."** ❌

This is **confidently wrong**. The corpus contains **four professors rated 5.0** (Amr Shahat, Jay Grossman, Jordan Mendler, Sanjhra Banks); 4.8 isn't even close to the max.

**Root cause (retrieval architecture):** "highest rating" is a question about the **whole corpus**, but top-k retrieval only ever hands the LLM **5 chunks** — whichever embed nearest to the phrase *"highest overall rating."* The model then reports the maximum *within that 5-chunk window* and presents it as the global maximum. No amount of prompt-tuning fixes this: the answer is wrong because the relevant evidence (the 5.0-rated professors' chunks) was **never retrieved**. Semantic top-k retrieval is the wrong tool for aggregation/superlative/counting queries; those need a structured query over metadata (e.g. `MAX(overall_rating)` across the collection), not nearest-neighbor search.

**Secondary failure (intermittent): aggregate-vs-per-review conflation.** Generation is non-deterministic, so this one surfaces in *some* runs, not all. Asked *"How many students would take Sihao Ma again?"*, one run answered *"only 20.7%… consistently mentioned in all five reviews."* The 20.7% is correct, but it is **one aggregate figure echoed in every chunk header**, not five independent reports — yet the model treats the repetition as corroboration (some Q4 runs phrase the 1.3 difficulty the same way). Root cause is an **ingestion decision**: I embed each professor's *aggregate* stats into every per-review chunk (good for attribution), which makes repeated metadata look like independent evidence to the LLM. This is exactly *Anticipated Challenge #3* in `planning.md` coming true.

**Also visible (retrieval noise):** the rank-5 chunk for Q1 is a content-free review (`".............."`, a Drop/Withdrawal). Near-empty reviews still embed close enough to occupy a top-k slot; a minimum-content filter at ingestion (length / alphanumeric ratio) would drop them.

**What I'd change:** (a) **route aggregation/superlative/count questions to a structured query** over the metadata instead of semantic retrieval (or compute and surface corpus-level stats separately); (b) **de-duplicate aggregate stats** out of the per-chunk text (keep them only in metadata) so repetition stops reading as corroboration, and have the prompt distinguish "the professor's overall rating" from "what individual reviewers said"; (c) add a **minimum-content filter** in `ingest.py`.

**Resolved during this build (data-quality):** 16 of 596 reviews originally carried un-decoded HTML entities (`&quot;`, `&#39;`) from the RMP API. I added `html.unescape()` to `ingest.py`'s `clean_text()` before chunking and re-ingested — **0 of 596 chunks** now contain entities, so they no longer leak into embeddings or citations.

---

## Spec Reflection

**One way the spec helped during implementation:** Deciding in `planning.md` that *one review = one chunk* (size 1200 for a corpus that averages ~392 chars) meant the chunking code was a near-direct translation of the spec, and the chunk count (596 ≈ one per review) was predictable rather than something to debug after the fact. Likewise, committing in the spec to *programmatic* source attribution (rather than asking the model to cite) made the grounding requirement a concrete coding task — `dedupe_sources()` over retrieved metadata — instead of a hope that the LLM would behave.

**One way the implementation diverged from the spec, and why:** The spec named a chunk size of **1200/200**, but the first cut of `ingest.py` carried leftover `CHUNK_SIZE = 300` / `CHUNK_OVERLAP = 50` constants that the splitter silently *ignored* (it was hard-coded to 1200/200). The running behavior matched the spec, but the code didn't — a future reader tuning `CHUNK_SIZE` would have changed nothing. I reconciled this by setting the constants to **1200/200** and passing them into the splitter, giving the config one source of truth. I also added a step the spec only listed as an *anticipated challenge* rather than a pipeline stage: a `clean_text()` pass in `ingest.py` that runs `html.unescape()` before chunking, after I confirmed 16 reviews carried raw entities like `&quot;`. The spec treated this as a risk to watch; implementation promoted it to an explicit cleaning stage.

---

## AI Usage

> These are the real instances from building this project. Edit the phrasing to match your own voice before submitting.

**Instance 1 — Scraper (overrode a broken approach).**
- *What I gave the AI:* the goal of collecting ≥10 documents of UCLA professor reviews, plus the observation that the `RateMyProfessorAPI` PyPI package was returning HTTP 403.
- *What it produced:* `scrap_rmp.py` hitting the RMP **GraphQL endpoint** directly with a browser `User-Agent` and RMP's public `test:test` token, with pagination, a ≥5-ratings filter, and a `build_document()` that fuses each professor's aggregate stats with one student comment.
- *What I changed or directed differently:* I set `MAX_PROFESSORS=60` / `REVIEWS_PER_PROF=10` to land near 596 documents across 26 departments, had it **drop "No Comments"/empty reviews**, and decided to keep the aggregate rating *inside* each document's text so a retrieved chunk stays attributable on its own.

**Instance 2 — Generation + grounding (tightened what it produced).**
- *What I gave the AI:* the *Retrieval Approach* + *Grounded Generation* design from `planning.md` — answer from retrieved context only, refuse when context is insufficient, and attach sources programmatically.
- *What it produced:* `query.py` (`retrieve()` → Groq `generate()` → `ask()`) and the Gradio `app.py`.
- *What I changed or overrode:* the first draft left **citations to the model** to include in prose; I redirected it to append sources *programmatically* from chunk metadata (`dedupe_sources()`) so attribution can't be forgotten, and added **refusal-string detection** that suppresses the source list on out-of-scope questions (so Q5 doesn't show bogus professor citations). I also lowered `temperature` to `0.1` for faithfulness.

**Instance 3 — Pipeline cleanup (caught a spec/code mismatch).**
- *What I gave the AI:* the `ingest.py` it had generated, plus the note that the README flagged unused `CHUNK_SIZE=300`/`50` constants.
- *What it produced:* a reconciliation setting the constants to `1200`/`200` and wiring them into the splitter, plus a `clean_text()` `html.unescape()` step.
- *What I changed or directed differently:* I had it **re-run ingestion and verify** the result rather than trust the edit — confirming `collection.count() == 596` and that 0 of 596 chunks still contained HTML entities (down from 16).
