"""Make sample-llm-notes.pdf: a short stand-in document (original text, not the course guide) so the RAG lab can run
before you have the course PDF. Run it once:  python make_sample_pdf.py
"""
from pathlib import Path

from fpdf import FPDF

SECTIONS = {
    "What is a large language model?": (
        "A large language model (LLM) is a neural network trained on a very large amount of text to predict the next "
        "token in a sequence. A token is a small piece of text, often a word or part of a word. By repeating that "
        "prediction many times, a model can write paragraphs, answer questions, summarise documents and write code. "
        "Models such as GPT, Claude, Llama and Qwen are LLMs. They differ in size, training data and licence."),
    "The transformer architecture": (
        "Most LLMs are built on the transformer, a neural network architecture introduced in 2017. A transformer uses "
        "self-attention: for every token it decides how much to focus on every other token in the input, so it can "
        "relate a word to context far away in the text. Because attention processes all tokens in parallel, "
        "transformers train efficiently on GPUs, which made today's very large models possible."),
    "Training and fine-tuning": (
        "Pre-training teaches a model general language from huge text collections and takes weeks on many GPUs. "
        "Fine-tuning then continues training on a smaller, task-specific dataset, for example support conversations or "
        "medical summaries, so the model follows a style or format more reliably. Instruction tuning and learning from "
        "human feedback make a model follow requests and refuse unsafe ones."),
    "Embeddings and vector search": (
        "An embedding turns a piece of text into a list of numbers so that texts with similar meaning get similar "
        "numbers. Storing the embeddings of many passages in a vector database lets a program find the passages "
        "closest in meaning to a question, even when they share no keywords. Embedding models are separate from "
        "chat models and are much smaller."),
    "Retrieval-augmented generation (RAG)": (
        "RAG gives a model facts it was never trained on. The program first retrieves the passages most relevant to "
        "the question from a document collection, then places them in the prompt, and the model writes the answer "
        "from those passages. This reduces made-up answers and lets the knowledge be updated by changing the "
        "documents instead of retraining the model. Agentic RAG adds a decision step: the system checks whether the "
        "retrieved passages are good enough and searches the web when they are not."),
    "Limits and risks": (
        "LLMs can hallucinate, which means stating false things with confidence. They have a context window, a limit "
        "on how much text they can read at once, and they do not know events after their training date. Answers "
        "should be checked against trusted sources, and private data should be kept out of third-party services "
        "unless the provider's terms allow it. Running a model locally keeps the data on your own machine."),
}

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "Sample notes on large language models", new_x="LMARGIN", new_y="NEXT")
for title, text in SECTIONS.items():
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, text)

out = Path(__file__).parent / "sample-llm-notes.pdf"
pdf.output(str(out))
print(f"Wrote {out} ({pdf.page_no()} pages)")
