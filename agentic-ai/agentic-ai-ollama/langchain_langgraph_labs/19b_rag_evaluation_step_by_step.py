"""Lab 19 without the shortcuts: the same RAG evaluation, written one visible step at a time.

Lab 19 chains the steps with `prompt | llm | StrOutputParser()`. Here every step is a separate line and is printed, so you
can see what the retriever finds, what the model is sent, what it answers and what RAGAS receives for grading.
"""
import sys
import types

# ragas 0.4 imports ChatVertexAI from a langchain_community module that newer langchain_community versions no longer
# have, so `import ragas` fails. Google Vertex AI is not used here, so register an empty stand-in before the import.
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    _stand_in = types.ModuleType("langchain_community.chat_models.vertexai")
    _stand_in.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _stand_in

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.run_config import RunConfig

from config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, OLLAMA_JUDGE_MODEL, OLLAMA_MODEL

load_dotenv()


# ---- Sample documents (the same five headlines as lab 19) ----
documents = [
    Document(
        page_content="Meta drops multimodal Llama 3.2 — here's why it's such a big deal",
        metadata={"company": "Meta", "topic": "llama"}
    ),
    Document(
        page_content="Chip giant Nvidia acquires OctoAI, a Seattle startup that helps companies run AI models",
        metadata={"company": "Nvidia", "topic": "acquisition"}
    ),
    Document(
        page_content="Google is bringing Gemini to all older Pixel Buds",
        metadata={"company": "Google", "topic": "gemini"}
    ),
    Document(
        page_content="The first Intel Battlmage GPU benchmarks have leaked",
        metadata={"company": "Intel", "topic": "gpu"}
    ),
    Document(
        page_content="Dell partners with Nvidia to accelerate AI adoption in telecoms",
        metadata={"company": "Dell", "topic": "partnership"}
    ),
]
ids = ["id1", "id2", "id3", "id4", "id5"]


# ---- Vector store: where the documents are searched ----
embeddings_fn = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

vector_store = Chroma(
    collection_name="agent-eval-19b",  # its own collection, so it never mixes with lab 19's
    embedding_function=embeddings_fn,
)
_ = vector_store.add_documents(documents=documents, ids=ids)
print(f"Stored {len(vector_store.get()['ids'])} documents in the vector store")


# ---- The model that writes the answers, and the prompt it is given ----
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
    num_ctx=8192,  # keeps the model on the GPU
)

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
{context}

Question: {query}
"""
)


# ---- One question, step by step ----
# Each numbered step below is what `chain.invoke(...)` does inside lab 19, here written out and printed
def answer_question(query: str, show_steps: bool = False) -> dict:
    # Step 1: search. The question is turned into a vector and compared with every stored document.
    # The score is a distance: the smaller it is, the closer the document is to the question.
    results = vector_store.similarity_search_with_score(query, k=1)  # k=1: only the single closest document
    relevant_docs = [doc for doc, _ in results]

    # Step 2: the context is the retrieved documents joined into one block of text
    context = "\n".join(doc.page_content for doc in relevant_docs)

    # Step 3: fill the two blanks in the prompt template
    prompt_text = prompt.format(context=context, query=query)

    # Step 4: send it to the model; the reply is a message object, and its .content is the plain text answer
    reply = llm.invoke(prompt_text)
    response = reply.content

    if show_steps:
        print("\n--- Step 1: retrieved ---")
        for doc, distance in results:
            print(f"distance {distance:.3f}  ->  {doc.page_content}")
        print("\n--- Step 2 and 3: the text sent to the model ---")
        print(prompt_text)
        print("--- Step 4: the model's answer ---")
        print(response)

    # The four things RAGAS needs for each question (the reference is added by the caller)
    return {
        "user_input": query,
        "retrieved_contexts": [doc.page_content for doc in relevant_docs],
        "response": response,
    }


# Look inside one question first
answer_question("Who is partnering with Nvidia?", show_steps=True)


# ---- Ragas Evaluation ----
sample_queries = [
    "What is Meta's latest development in AI?",
    "Which company did Nvidia acquire?",
    "What AI feature is Google adding to older Pixel Buds?",
    "What recent information has leaked about Intel GPUs?",
    "Which company did Dell partner with to accelerate AI adoption?"
]

# The answer each question should get, in the same order
expected_responses = [
    "Meta drops multimodal Llama 3.2 — here's why it's such a big deal",
    "Chip giant Nvidia acquires OctoAI, a Seattle startup that helps companies run AI models",
    "Google is bringing Gemini to all older Pixel Buds",
    "The first Intel Battlemage GPU benchmarks have leaked",
    "Dell partners with Nvidia to accelerate AI adoption in telecoms"
]

dataset = []
for query, reference in zip(sample_queries, expected_responses):
    row = answer_question(query)
    row["reference"] = reference
    dataset.append(row)

evaluation_dataset = EvaluationDataset.from_list(dataset)

# The judge: a separate, stronger non-thinking model (OLLAMA_JUDGE_MODEL in config.py)
evaluator_llm = LangchainLLMWrapper(ChatOllama(
    model=OLLAMA_JUDGE_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
    num_ctx=8192,
))

# 3 metrics x 5 questions = 15 grading jobs; 4 workers matches how many requests Ollama serves at once
result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm,
    run_config=RunConfig(timeout=600, max_workers=4),
)

print(result)  # the average of each metric over the five questions

# One row per question: its answer and its own scores, to see which answer lost points (faithfulness was 0.7)
scores = result.to_pandas()
print(scores[["user_input", "response", "context_recall", "faithfulness", "factual_correctness(mode=f1)"]].to_string())
