# AI Business Advisor

A multi-step [LCEL](https://python.langchain.com/docs/concepts/lcel/) workflow that helps brainstorm and evaluate business ideas, running on a local open model through [Ollama](https://ollama.com).

Built as an exercise from Udacity's Agentic AI Engineer with LangChain and LangGraph Nanodegree (ND901).

## Workflow

```
industry -> [idea prompt] -> idea -> [analysis prompt] -> analysis -> [report prompt] -> final report
```

Each step is a `prompt | llm | parser` chain. `RunnablePassthrough.assign()` carries earlier results forward, so the report step can see the industry, the idea and the analysis.

## Run

```bash
pip install -r requirements.txt
ollama pull qwen3:4b
python advisor.py "Agriculture"
```

The industry argument is optional and defaults to `Healthcare`. To use another local model, set `OLLAMA_MODEL` in a `.env` file, for example `OLLAMA_MODEL=llama3.2:1b`.
