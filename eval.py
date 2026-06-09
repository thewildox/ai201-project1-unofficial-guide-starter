"""
eval.py
-------
Reproducible evaluation harness for the Unofficial Guide (Milestone 6).

Runs all 5 test questions from planning.md end-to-end (retrieve -> generate ->
cite) and prints, for each: the question, the expected answer, the retrieved
chunks with distances, the system's actual answer, and its cited sources.

Requires a real GROQ_API_KEY in .env (generation step). Run:
    python eval.py
"""

from query import ask

# The 5 test questions and their expected (ground-truth) answers, copied from
# planning.md. Expected answers are grounded in actual review text.
EVAL = [
    {
        "q": "What do students say about Professor Sihao Ma's exams in his math classes?",
        "expected": (
            "Polarized. Most reviews warn the exams are extremely difficult — proof "
            "questions 'not in the same atmosphere' as lecture, plus harsh grading; a "
            "minority defend him as clear if you understand the concepts. Aggregate "
            "1.7/5 rating, 4.7/5 difficulty."
        ),
    },
    {
        "q": "Why do students recommend avoiding Professor Delroy A. Baugh for general chemistry?",
        "expected": (
            "Lectures relate little to the textbook/homework, he teaches above the level "
            "of Chem 20A, and exam questions are 'out of left field' / unrelated to the "
            "assigned readings. Aggregate 1.5/5 rating, 4.1/5 difficulty."
        ),
    },
    {
        "q": "Which Computer Science professor do students describe as caring and chill, and why?",
        "expected": (
            "Jordan Mendler — awesome, caring, 'super chill'; treats students like adults, "
            "always willing to help, even throws an end-of-class ice-cream party. 5.0/5."
        ),
    },
    {
        "q": "Is Professor Stephen Ross's economics course considered easy or hard?",
        "expected": (
            "Easy. 4.8/5 rating, 1.3/5 difficulty, ~91% would take again; students call "
            "him engaging/passionate and the course great to take."
        ),
    },
    {
        "q": "Which UCLA dorm has the best dining hall?",
        "expected": (
            "Out of scope — the corpus contains only professor reviews, no housing/dining "
            "content. Correct behavior: the system should decline, not invent an answer."
        ),
    },
    {
        # Stress test (documented FAILURE case). Top-k semantic retrieval can't
        # answer a corpus-wide superlative — it only sees 5 chunks, so it reports
        # the max within that window. Truth: four professors are rated 5.0
        # (Amr Shahat, Jay Grossman, Jordan Mendler, Sanjhra Banks).
        "q": "Which UCLA professor has the highest overall rating?",
        "expected": (
            "FAILS BY DESIGN — a global-aggregation question. There are four 5.0-rated "
            "professors; the system instead names the best of its 5 retrieved chunks "
            "(e.g. 'Stephen Dickey, 4.8'), which is wrong. See README Failure Case."
        ),
    },
]


def run():
    for i, item in enumerate(EVAL, 1):
        q = item["q"]
        print("=" * 90)
        print(f"Q{i}: {q}")
        print(f"\nEXPECTED:\n  {item['expected']}\n")

        result = ask(q)

        print("RETRIEVED CHUNKS:")
        for j, c in enumerate(result["chunks"], 1):
            m = c["metadata"]
            comment = c["text"].split("Student review:")[-1].strip().replace("\n", " ")
            print(
                f"  [{j}] dist={c['distance']:.3f}  {m.get('professor')} "
                f"({m.get('department')}, {m.get('overall_rating')}/5)"
            )
            print(f'      "{comment[:140]}"')

        print(f"\nSYSTEM ANSWER:\n  {result['answer']}\n")
        print("CITED SOURCES:")
        if result["sources"]:
            for s in result["sources"]:
                print(f"  • {s}")
        else:
            print("  (none — system declined)")
        print()


if __name__ == "__main__":
    run()
