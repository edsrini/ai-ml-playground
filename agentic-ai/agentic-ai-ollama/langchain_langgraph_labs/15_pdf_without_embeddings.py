"""Read a PDF without embeddings: put the whole text in the prompt and let the model answer from it.

Same graph pattern as 14_agentic_rag.py, minus the vector store: read the PDF, build the prompt, generate the answer.
This only works while the text fits in the model's context window. For a longer document you need a way to pick the
relevant parts, which is what the embeddings and vector store in 14_agentic_rag.py are for.
"""
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState

import truststore
truststore.inject_into_ssl()  # trust the Windows certificate store, so the graph picture can be drawn over HTTPS

from dotenv import load_dotenv
load_dotenv()

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

# Instantiate your chat model
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
    num_ctx=8192,  # the model reads at most 8,192 tokens at once: the prompt and the answer must fit in that
)

MAX_CHARS = 20_000  # about 5,000 tokens: leaves room in the 8,192-token window for the question and the answer

# Same file choice as 14_agentic_rag.py: the course PDF if it is there, otherwise the small stand-in
data_dir = Path(__file__).parent / "data"
file_path = data_dir / "compact-guide-to-large-language-models.pdf"
if not file_path.exists():
    file_path = data_dir / "sample-llm-notes.pdf"  # made by data/make_sample_pdf.py


# ---- Graph state and nodes ----
class State(MessagesState):
    question: str
    text: str
    answer: str


# Node: read the PDF and keep all of its text. No embeddings, no vector store.
def read_pdf(state: State):
    if not file_path.exists():
        raise SystemExit(f"Put a PDF at {file_path} (or run data/make_sample_pdf.py) and run again.")
    pages = PyPDFLoader(str(file_path)).load()
    text = "\n\n".join(page.page_content for page in pages)
    if len(text) > MAX_CHARS:  # plain code checks the limit, so the model never receives a prompt that overflows
        raise ValueError(
            f"The PDF has {len(text):,} characters, more than the {MAX_CHARS:,} that fit in the prompt. "
            "Use 14_agentic_rag.py, which searches for the relevant pieces instead of sending everything."
        )
    return {"text": text}


# Node: the "augment" step: put the whole document and the question into one prompt
def augment(state: State):
    template = ChatPromptTemplate([
        ("system", "You are an assistant for question-answering tasks."),
        ("human", "Use the following document to answer the question. "
                  "If the answer is not in the document, just say that you don't know. "
                  "Use three sentences maximum and keep the answer concise. "
                  "\n# Question: \n-> {question} "
                  "\n# Document: \n-> {document} "
                  "\n# Answer: "),
    ])
    messages = template.invoke(
        {"document": state["text"], "question": state["question"]}
    ).to_messages()
    return {"messages": messages}


# Node: the model writes the answer from that prompt
def generate(state: State):
    ai_message = llm.invoke(state["messages"])
    return {"answer": ai_message.content, "messages": ai_message}


# ---- The graph ----
workflow = StateGraph(State)

workflow.add_node("read_pdf", read_pdf)
workflow.add_node("augment", augment)
workflow.add_node("generate", generate)

workflow.add_edge(START, "read_pdf")
workflow.add_edge("read_pdf", "augment")
workflow.add_edge("augment", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()


# ---- Draw the graph ----
# In a notebook display(Image(...)) shows the picture; in a script we save it as a file instead.
# mermaid.ink renders the PNG over the internet, so fall back to an ASCII drawing if that fails.
try:
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "pdf_without_embeddings_graph.png").write_bytes(graph.get_graph().draw_mermaid_png())
    print(f"Saved {output_dir / 'pdf_without_embeddings_graph.png'}")
except Exception as error:
    print("Could not render the PNG:", type(error).__name__)
    graph.get_graph().print_ascii()


# ---- Run it ----
output = graph.invoke(
    input={"question": "What is a transformer in an LLM?"},
)
print(output["answer"])
