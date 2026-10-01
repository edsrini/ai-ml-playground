from anyio.itertools import chain

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel
from langchain_core.tracers.context import collect_runs
from dotenv import load_dotenv

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

load_dotenv()

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
)

prompt = PromptTemplate(
    template="Tell me a joke about {topic}"
)
parser = StrOutputParser()

print(parser.invoke(
    llm.invoke(
        prompt.invoke(
            {"topic": "Python"}
        )
    )
))


runnables = [prompt, llm, parser]

for runnable in runnables:
    print(f"{repr(runnable).split('(')[0]}")
    print(f"\tINVOKE: {repr(runnable.invoke)}")
    print(f"\tBATCH: {repr(runnable.batch)}")
    print(f"\tSTREAM: {repr(runnable.stream)}\n")

for runnable in runnables:
    print(f"{repr(runnable).split('(')[0]}")
    print(f"\tINPUT: {repr(runnable.get_input_schema())}")
    print(f"\tOUTPUT: {repr(runnable.get_output_schema())}")
    print(f"\tCONFIG: {repr(runnable.config_schema())}\n")


chain = RunnableSequence(prompt, llm, parser)

print(type(chain))

for chunk in chain.stream("Python"):
    print(chunk, end="", flush=True)

print("#####################################################")
results = chain.batch([
    {"topic": "Python"},
    {"topic": "Java"},
    {"topic": "JavaScript"},
    {"topic": "Rust"},
    {"topic": "C++"},
    {"topic": "C#"},
    {"topic": "Go"},
    {"topic": "Swift"},
    {"topic": "Kotlin"},
    {"topic": "PHP"},
])
for joke in results:
    print(joke)
    print("---")

chain.get_graph().print_ascii()

def doubler(x) -> int:
    return x * 2

runnable = RunnableLambda(doubler)
print(runnable.invoke(5))

def add_one(x) -> int:
    return x + 1

parallel_chain = RunnableParallel({"doubled": runnable, "plus_one": add_one})
print(parallel_chain.invoke(100))

print(parallel_chain.get_graph().print_ascii())

#LCE
chain = prompt | llm | parser
print(chain.invoke(
    {"topic": "Computer"}))




