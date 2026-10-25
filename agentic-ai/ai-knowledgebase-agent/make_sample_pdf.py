"""Make the stand-in PDFs used while you do not have the real documents. All text is original, written for this project.

  compact-guide-to-large-language-models.pdf   stands in for the course guide (it is NOT that guide)
  open-source-models.pdf                       a second document, about open-source and open-weight models

Run it once:  python make_sample_pdf.py     Replace any of them with a real document of the same name whenever you like.
"""
from pathlib import Path

from fpdf import FPDF

DOCUMENTS = {
    "compact-guide-to-large-language-models.pdf": (
        "Sample notes on large language models",
        "Stand-in text written for this project, not the course guide.",
        {
            "What is a large language model?": (
                "A large language model (LLM) is a neural network trained on a very large amount of text to predict the "
                "next token in a sequence. A token is a small piece of text, often a word or part of a word. By repeating "
                "that prediction many times, a model can write paragraphs, answer questions, summarise documents and write "
                "code."),
            "The transformer architecture": (
                "Most LLMs are built on the transformer, a neural network architecture introduced in 2017. A transformer "
                "uses self-attention: for every token it decides how much to focus on every other token in the input, so it "
                "can relate a word to context far away in the text. Because attention processes all tokens in parallel, "
                "transformers train efficiently on GPUs."),
            "Training and fine-tuning": (
                "Pre-training teaches a model general language from huge text collections. Fine-tuning then continues "
                "training on a smaller, task-specific dataset so the model follows a style or format more reliably. "
                "Instruction tuning and learning from human feedback make a model follow requests and refuse unsafe ones."),
            "Embeddings and vector search": (
                "An embedding turns a piece of text into a list of numbers so that texts with similar meaning get similar "
                "numbers. Storing the embeddings of many passages in a vector database lets a program find the passages "
                "closest in meaning to a question, even when they share no keywords."),
            "Retrieval-augmented generation (RAG)": (
                "RAG gives a model facts it was never trained on. The program first retrieves the passages most relevant "
                "to the question from a document collection, then places them in the prompt, and the model writes the "
                "answer from those passages. The knowledge can be updated by changing the documents instead of "
                "retraining the model."),
            "Limits and risks": (
                "LLMs can hallucinate, which means stating false things with confidence. They have a context window, a "
                "limit on how much text they can read at once, and they do not know events after their training date. "
                "Running a model locally keeps the data on your own machine."),
        },
    ),
    "open-source-models.pdf": (
        "Sample notes on open-source language models",
        "Stand-in text written for this project.",
        {
            "What are open-source models?": (
                "An open-source or open-weight language model is one whose trained weights are published, so anyone can "
                "download and run it. Closed models, by contrast, are available only through the owner's service. Examples "
                "of open-weight families include Llama, Qwen, Mistral and Gemma. The weights are usually free, but each "
                "model has a licence that says how it may be used."),
            "Licences matter": (
                "Open-weight does not always mean open-source in the strict sense. Some licences allow any use, including "
                "commercial use. Others limit commercial use, require attribution, or forbid certain applications. Read "
                "the licence before building a product on a model, and keep a record of which licence each model uses."),
            "Running a model locally": (
                "Tools such as Ollama download an open-weight model and serve it on your own computer. A model with a few "
                "billion parameters runs well on a modern GPU. The text you send stays on your machine, and there is no "
                "per-request fee. The limits are your hardware and the model's size: a small local model is usually less "
                "capable than the largest hosted models."),
            "Why teams choose open models": (
                "Teams pick open models for privacy, because data never leaves their systems; for cost, because there is no "
                "per-token charge; for control, because the model can be fine-tuned and version-pinned; and for "
                "reliability, because it keeps working when a hosted service changes. The trade-off is that the team runs "
                "and updates the model itself."),
        },
    ),
}


def make_pdf(path, title, note, sections):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, note, new_x="LMARGIN", new_y="NEXT")
    for heading, text in sections.items():
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, text)
    pdf.output(str(path))
    return pdf.page_no()


for name, (title, note, sections) in DOCUMENTS.items():
    out = Path(__file__).parent / name
    print(f"Wrote {out.name} ({make_pdf(out, title, note, sections)} page)")
