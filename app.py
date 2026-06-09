"""
app.py
------
Gradio web interface for the Unofficial Guide RAG system (Milestone 5).

Run:
    python app.py
then open http://localhost:7860

Type a plain-language question about a UCLA professor or course. The system
retrieves the most relevant student reviews, generates a grounded answer with
Groq (using only those reviews), and lists the reviews it drew from.
"""

import gradio as gr

from query import ask, TOP_K


def handle_query(question: str):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""

    result = ask(question)
    answer = result["answer"]

    if result["sources"]:
        sources = "\n".join(f"• {s}" for s in result["sources"])
    else:
        sources = "(no sources — the system declined to answer this question)"

    return answer, sources


EXAMPLES = [
    "What do students say about Professor Sihao Ma's exams?",
    "Why do students recommend avoiding Professor Delroy A. Baugh for chemistry?",
    "Which Computer Science professor do students describe as caring and chill?",
    "Is Professor Stephen Ross's economics course easy or hard?",
    "Which UCLA dorm has the best dining hall?",
]


with gr.Blocks(title="The Unofficial Guide — UCLA Professor Reviews") as demo:
    gr.Markdown(
        "# 🎓 The Unofficial Guide\n"
        "Ask a plain-language question about a **UCLA professor or course**. "
        "Answers are grounded **only** in real Rate My Professors student reviews "
        f"(top-{TOP_K} retrieved), with the reviews cited below each answer.\n\n"
        "If the reviews don't cover your question, the system will say so rather "
        "than make something up."
    )

    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. What do students say about Professor Sihao Ma's exams?",
        lines=2,
    )
    btn = gr.Button("Ask", variant="primary")

    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from (sources)", lines=6)

    gr.Examples(examples=EXAMPLES, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
