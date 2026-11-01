import os
from pathlib import Path
from typing import Dict, Any, List, Annotated
from IPython.display import Image, display
import nest_asyncio
from dotenv import load_dotenv
from urllib.parse import urlparse
from langchain_core.tools import tool
from langchain_ollama import ChatOllama  # local replacement for ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient
from pydantic import BaseModel, Field

# Lets asyncio code run again inside an already-running event loop (notebooks, some IDE consoles)
nest_asyncio.apply()

# Reads .env into environment variables (the Tavily key will come from here); returns True if a .env file was found
print(load_dotenv())

# Which local model to use and where Ollama is running (both can be changed in .env)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# The course uses ChatOpenAI(model="gpt-4o-mini", temperature=0.0, base_url=..., api_key=os.getenv(...));
# a local Ollama model needs no key
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
    num_ctx=8192,  # context window; without it Ollama may reserve 256K tokens, fill the GPU and spill into RAM (slow)
)

# Client for the Tavily web-search service; the web_search tool below uses it
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")  # the key lives in .env, never in the code
)


# The shared notebook of the whole workflow: every step reads what it needs from here and writes its result back.
# `messages` (from MessagesState) keeps the chat history; the other fields hold the plan, research, draft and so on.
class BlogState(MessagesState):
    """State for the blog creation workflow."""
    query: HumanMessage
    topic: str
    title: str
    audience: str
    sections: List[str]
    keywords: List[str]
    research: List[Dict[str, Any]]
    draft_markdown: str
    references_markdown: str
    final_markdown: str
    seo_score: int
    publish_status: str


# ---- Pydantic Models for parsing outputs ----
# Each Pydantic model describes the shape of one agent's answer: the field names and their types.
# The agent's text is turned into one of these objects, and a wrong shape raises an error instead of passing silently.
class BlogPlan(BaseModel):
    """Output from the intake agent"""
    topic: Annotated[str, Field(description="The main topic of the blog post")]
    title: Annotated[str, Field(description="A clear, working title for the blog post")]
    audience: Annotated[str, Field(description="Target audience description")]
    sections: Annotated[List[str], Field(description="List of section headings for the blog outline")]
    keywords: Annotated[List[str], Field(description="List of relevant keywords for SEO")]


class ResearchItem(BaseModel):
    """Individual research item with metadata"""
    url: Annotated[str, Field(description="URL of the research source")]
    title: Annotated[str, Field(description="Title of the research content")]
    content: Annotated[str, Field(description="Extracted content from the source")]
    score: Annotated[float, Field(description="Relevance score from 0.0 to 1.0")]


class ResearchData(BaseModel):
    """Output from the researcher agent"""
    research: Annotated[List[ResearchItem], Field(description="List of research items")]


class ContentDraft(BaseModel):
    """Output from the writer agent"""
    draft_markdown: Annotated[str, Field(description="The complete blog post in markdown format")]
    # The screenshot cuts off this line; the description text is my guess
    references_markdown: Annotated[str, Field(description="The references section in markdown footnote format")]


class ReviewedContent(BaseModel):
    """Output from the reviewer agent"""
    final_markdown: Annotated[str, Field(description="The final, reviewed blog post")]
    seo_score: Annotated[int, Field(description="SEO score from 0-100")]
    lint_issues: Annotated[List[str], Field(description="List of any linting issues found and fixed")]


class PublicationStatus(BaseModel):
    """Output from the publisher agent"""
    publish_status: Annotated[str, Field(description="Status of the publication process")]


# ============================================================
# TOOLS
# ============================================================

# ---- Intake Agent tools ----
# Builds a fixed title, audience and section list from the topic (plain string templates, no model call)
@tool
def outline_topic(topic: str) -> Dict[str, Any]:
    """
    Creates a pragmatic outline for a blog post given a topic.

    Args:
        topic (str): The domain/area/technology (e.g., "LangGraph multi-agent systems").

    Returns:
        Dict[str, Any]: A dictionary with:
            - title (str): Suggested working title.
            - audience (str): Target reader profile.
            - sections (List[str]): Ordered list of section headings.

    Example:
        >>> outline_topic(topic="LangGraph multi-agent systems")
        {'title': 'LangGraph multi-agent systems: What It Is, Why It Matters, and How To Start',
         'audience': 'Practitioners and curious newcomers',
         'sections': ['Introduction', 'Why ... matters right now', ...]}
    """
    base = topic.strip().rstrip(".")
    title = f"{base}: What It Is, Why It Matters, and How To Start"
    audience = "Practitioners and curious newcomers"
    sections = [
        "Introduction",
        f"Why {base} matters right now",
        f"Core concepts in {base}",
        f"Practical getting-started guide for {base}",
        "Common pitfalls and best practices",
        f"Trends and the future of {base}",
        "Conclusion and next steps",
    ]
    return {"title": title, "audience": audience, "sections": sections}


# Makes a list of keyword phrases by adding words like 'tutorial' or 'guide' to the topic
@tool
def generate_keywords(topic: str, limit: int = 12) -> List[str]:
    """
    Generates a compact keyword set for SEO checks based on a topic.

    Args:
        topic (str): The domain/area/technology to derive keywords from.
        limit (int, optional): Maximum number of keywords to return. Defaults to 12.

    Returns:
        List[str]: A list of keyword phrases ordered by relevance heuristics.

    Example:
        >>> generate_keywords(topic="LangGraph multi-agent systems", limit=8)
        ['langgraph multi-agent systems', 'langgraph multi-agent systems tutorial', ...]
    """
    root = topic.lower().strip()
    extras = [
        f"{root} tutorial", f"{root} guide", f"{root} best practices",
        f"{root} examples", f"{root} use cases", f"{root} vs alternatives",
        f"{root} performance", f"{root} scalability",
        f"{root} architecture", f"{root} tools", f"{root} pitfalls"
    ]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for k in [root] + extras:
        if k not in seen:
            seen.add(k)
            out.append(k)
        if len(out) >= limit: break
    return out


# ---- Researcher Agent tools ----
# Searches the web with Tavily and returns the top 5 results (url, title, snippet, score)
@tool
def web_search(question: str) -> Dict[str, Any]:
    """
    Performs a web search and returns top search results for a given query.

    Args:
        question (str): A query to be searched.

    Returns:
        Dict[str, Any]: A dictionary with a 'results' list where each item is a dict containing:
            - url (str): The result URL.
            - title (str): The page title.
            - content (str): A content snippet or summary.
            - score (float): Relevance score from the search provider.

    Example:
        >>> web_search(question="LangGraph multi-agent patterns")
        {'results': [{'url': 'https://...', 'title': '...', 'content': '...', 'score': 0.82}, ...]}
    """
    response = tavily_client.search(
        query=question,
        max_results=5,
    )
    return response


# Keeps only the best result from each website, so one site cannot fill the whole list
@tool
def dedupe_by_domain(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicates research results by domain, keeping the strongest items per domain.

    Args:
        results (List[Dict[str, Any]]): A list of search result objects where each item contains:
            - url (str): Result URL (used to extract domain).
            - title (str): Page title.
            - content (str): Snippet or summary.
            - score (float): Relevance score.

    Returns:
        List[Dict[str, Any]]: A list of results with at most 'max_per_domain' per domain,
        sorted within each domain by descending 'score' (then title length as tie-breaker).

    Notes:
        If a URL is missing or unparsable, the domain is treated as "unknown".

    Example:
        >>> dedupe_by_domain(search_payload['results'])
        [{'url': 'https://a.com/...', ...}, {'url': 'https://b.org/...', ...}]
    """
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for r in results or []:
        domain = urlparse(r.get("url", "")).netloc or "unknown"
        by_domain.setdefault(domain, []).append(r)
    cleaned = []
    # keep one result per website
    max_per_domain = 1

    for _, items in by_domain.items():
        items = sorted(items, key=lambda x: (x.get("score", 0), len(x.get("title", ""))), reverse=True)
        cleaned.extend(items[:max_per_domain])
    return cleaned


# ---- Writer Agent tools ----
# Turns the sources into numbered Markdown footnotes like [^1]: Title — url
@tool
def format_references(results: List[Dict[str, Any]]) -> str:
    """
    Formats research results into a Markdown footnotes block.

    Args:
        results (List[Dict[str, Any]]): A list of research items with fields:
            - title (str)
            - url (str)

    Returns:
        str: Markdown string with one footnote per line using the format:
            "[^n]: {title} — {url}"

    Example:
        >>> format_references([{'title':'Doc','url':'https://...'}])
        '[^1]: Doc — https://...'
    """
    lines = []
    for i, r in enumerate(results or [], start=1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        lines.append(f"[^{i}]: {title} — {url}")
    return "\n".join(lines)


# Counts the words and divides by a reading speed to get the minutes (at least 1)
@tool
def estimate_read_time(markdown_text: str, wpm: int = 225) -> Dict[str, Any]:
    """
    Estimates reading time and word count for a Markdown draft.

    Args:
        markdown_text (str): The full Markdown content to analyze.
        wpm (int, optional): Reading speed in words per minute. Defaults to 225.

    Returns:
        Dict[str, Any]: A dictionary with:
            - word_count (int): Total number of words.
            - minutes (int): Estimated whole minutes to read (minimum of 1).

    Example:
        >>> estimate_read_time("# Title\\n\\nSome text.")
        {'word_count': 3, 'minutes': 1}
    """
    import re
    words = re.findall(r"\b\w+\b", markdown_text or "")
    wc = len(words)
    minutes = max(1, round(wc / max(100, wpm)))
    return {"word_count": wc, "minutes": minutes}


# ---- Reviewer Agent tools ----
# Checks the post for simple mistakes: no title at the top, footnote marks without notes, very long paragraphs
@tool
def markdown_lint(md: str) -> Dict[str, Any]:
    """
    Runs a simple Markdown lint pass for structure and readability issues.

    Args:
        md (str): Markdown content to lint.

    Returns:
        Dict[str, Any]: A dictionary with:
            - ok (bool): True if no issues found, else False.
            - issues (List[str]): Human-readable problem descriptions.

    Checks:
        - Ensures the document starts with a single H1 ('# ').
        - Warns if footnote markers are present but definitions are missing.
        - Flags overly long paragraphs (very rough heuristic).

    Example:
        >>> markdown_lint("No title")
        {'ok': False, 'issues': ['Missing H1 title at the very top (use '# Title').']}
    """
    issues = []
    if not md.strip().startswith("# "):
        issues.append("Missing H1 title at the very top (use '# Title').")

    if "[^" in md and "]: " not in md:
        issues.append("Footnote marker used but no footnote definitions present.")

    for idx, para in enumerate([p for p in md.split("\n\n") if p.strip()], start=1):
        if len(para) > 1200:
            issues.append(f"Paragraph {idx} is very long; consider splitting.")

    return {"ok": len(issues) == 0, "issues": issues}


# Scores keyword use: more points when the keywords are in the title, then headings, then the text
@tool
def seo_score(md: str, keywords: List[str]) -> Dict[str, Any]:
    """
    Computes a naive SEO score (0-100) based on keyword distribution.

    Args:
        md (str): Markdown content to score.
        keywords (List[str]): Target keywords/phrases to check for.

    Returns:
        Dict[str, Any]: A dictionary with:
            - score (int): SEO score (capped at 100).
            - title_hits (int): Count of keywords present in the H1 title.
            - heading_hits (int): Count of keywords found across headings.
            - body_hits (int): Count of keywords found in the full text.
            - keywords_checked (List[str]): Echo of the keywords analyzed.

    Scoring:
        - Title matches weighted 20 points each.
        - Heading matches weighted 6 points each.
        - Body matches weighted 2 points each.

    Example:
        >>> seo_score(md="# LangGraph Guide", keywords=["langgraph"])
        {'score': 28, 'title_hits': 1, 'heading_hits': 0, 'body_hits': 1, 'keywords_checked': ['langgraph']}
    """
    import re
    text = md.lower()
    # The screenshot ends here. The lines below are my completion of what the docstring describes
    lines = text.splitlines()
    title = next((line[2:] for line in lines if line.startswith("# ")), "")
    headings = " ".join(line for line in lines if re.match(r"#{2,6} ", line))
    wanted = [k.lower() for k in keywords]
    title_hits = sum(1 for k in wanted if k in title)
    heading_hits = sum(1 for k in wanted if k in headings)
    body_hits = sum(1 for k in wanted if k in text)
    # 20 points per keyword in the title, 6 per keyword in the headings, 2 per keyword in the text; 100 at most
    score = min(100, title_hits * 20 + heading_hits * 6 + body_hits * 2)
    return {
        "score": score,
        "title_hits": title_hits,
        "heading_hits": heading_hits,
        "body_hits": body_hits,
        "keywords_checked": keywords[:],
    }


# ---- Publisher Agent tools ----
# A pretend publish: it only prints the post and says 'published'. Nothing is really posted anywhere
@tool
def publish_blog(body_md: str) -> Dict[str, Any]:
    """
    Publishes the final blog post (mock implementation).

    Args:
        body_md (str): Final Markdown body content to publish.

    Returns:
        Dict[str, Any]: A dictionary indicating publication status:
            - status (str): Always 'published' in this mock implementation.

    Side Effects:
        Prints the blog title and body to stdout, followed by a status line.

    Example:
        >>> publish_blog(title="My Post", body_md="# My Post\\n\\nHello")
        {'status': 'published'}
    """
    print("\n=== PUBLISHING BLOG ===")
    print(body_md)
    print("\n=== STATUS: published ===")
    return {"status": "published"}


# ============================================================
# AGENTS
# ============================================================

# ---- 1. Intake Agent ----
# An agent = a model + a system prompt (its job) + the tools it may use. create_react_agent runs the loop:
# think, call a tool, read the result, repeat, then give the final answer.
# The intake agent plans the post: title, audience, sections and keywords
intake_agent = create_react_agent(
    name="intake_agent",
    prompt=SystemMessage(
        content=(
            "You are a pragmatic content planner. "
            "Given a domain/area/technology topic, produce a blog plan: "
            "a clear working title, target audience, and a section outline. "
            "Use tools to create the outline and initial keyword set. "
            "Return ONLY JSON with keys: topic, title, audience, sections, keywords."
        )
    ),
    model=llm,
    tools=[outline_topic, generate_keywords],
)


# ---- 2. Researcher Agent ----
# The researcher agent searches the web, then removes duplicate sites
researcher_agent = create_react_agent(
    name="researcher_agent",
    prompt=SystemMessage(
        content=(
            "You're a senior marketing researcher. "
            "Given the topic and optionally other sections, "
            "find RECENT, high-signal web information about the topic. "
            "Call web_search at least once with a good query; "
            "then call dedupe_by_domain to remove duplicates. "
            "Return ONLY JSON with key 'research', an array of items "
            "each like {url,title,content,score} (deduped)."
        )
    ),
    model=llm,
    tools=[web_search, dedupe_by_domain],
)


# ---- 3. Writer Agent ----
# The writer agent writes the Markdown post, estimates the reading time and builds the references list
writer_agent = create_react_agent(
    name="writer_agent",
    prompt=SystemMessage(
        content=(
            "You are a seasoned technical blogger. "
            "Write a polished Markdown blog post using title, sections, "
            "and research (array of {url,title,content,score}). "
            "Requirements:\n"
            " - Begin with an H1 (= the title).\n"
            " - Add a short abstract (1-2 sentences), then follow the sections.\n"
            " - Use bullets, code fences, and examples when helpful.\n"
            " - Use footnote-style citations [^1], [^2] inline whenever claims come from research.\n"
            " - Append '## References' followed by footnotes generated via the references tool.\n"
            " - Add an estimated reading time line near the top.\n"
            "Workflow:\n"
            " 1) Draft the post body.\n"
            " 2) Call estimate_read_time on the draft and insert a line like '_Estimated reading time: X min_'.\n"
            " 3) Call format_references with the same research list to create footnotes.\n"
            "Return ONLY JSON with keys: draft_markdown, references_markdown."
        )
    ),
    model=llm,
    tools=[estimate_read_time, format_references],
)


# ---- 4. Reviewer Agent ----
# The reviewer agent checks the draft with the lint and SEO tools and fixes the problems
reviewer_agent = create_react_agent(
    name="reviewer_agent",
    prompt=SystemMessage(
        content=(
            "You are a meticulous copy editor. "
            "Input: draft_markdown, keywords, references_markdown. "
            "Tasks:\n"
            " 1) Ensure '## References' exists at the end and merge references_markdown if missing.\n"
            " 2) Call markdown_lint on the draft; then fix issues inline (H1 title, footnotes, long paragraphs).\n"
            " 3) Call seo_score; make small, high-quality edits to improve the score "
            "    without keyword stuffing (preserve voice and clarity).\n"
            "Output: Return ONLY JSON with keys: final_markdown, seo, lint."
        )
    ),
    model=llm,
    tools=[markdown_lint, seo_score],
)


# ---- 5. Publisher Agent ----
# The publisher agent publishes the final post with the publish_blog tool
publisher_agent = create_react_agent(
    name="publisher_agent",
    prompt=SystemMessage(
        content=(
            "You are a release manager. "
            "Publish the finalized blog post using the provided tool. "
            "Input: title and final_markdown (produced by the reviewer). "
            "Return ONLY JSON with keys: publish_status."
        )
    ),
    model=llm,
    tools=[publish_blog],
)


# ============================================================
# NODES
# ============================================================

# A node is one step of the graph: it reads the shared state, runs its agent, and returns only the fields it changes.
# Every node below follows the same four steps: build the message, run the agent, turn its text into an object, update the state.
def intake_node(state: BlogState) -> BlogState:
    """Node for the intake agent to plan the blog."""
    print("Starting intake agent...")

    # Run the agent; it may call its tools several times before it gives its final answer
    result = intake_agent.invoke(
        input={
            "messages": [
                state["query"]
            ]
        }
    )
    messages = result["messages"]
    # The last message is the agent's final answer (JSON text)
    last_ai_message: AIMessage = messages[-1]

    # The agent's last reply is JSON text; a second call asks the model to fill the BlogPlan model from it
    llm_with_structure = llm.with_structured_output(BlogPlan)
    structured_output: BlogPlan = llm_with_structure.invoke(last_ai_message.content)

    print(f"message: {last_ai_message.content[:20]}...")
    print("Done!\n")

    return {
        # keep the original request, then everything the agent said, in the history
        "messages": [state["query"]] + messages,
        "topic": structured_output.topic,
        "title": structured_output.title,
        "audience": structured_output.audience,
        "sections": structured_output.sections,
        "keywords": structured_output.keywords,
    }


def research_node(state: BlogState) -> BlogState:
    """Node for the researcher agent to gather information."""
    print("Starting researcher agent...")

    # What the researcher is told: the topic and the section names, taken from the state
    context_message = f"Research topic: {state['topic']}\nSections: {', '.join(state['sections'])}"

    # The course sends the context as an AIMessage, which leaves the chat ending on an assistant message; qwen3 on
    # Ollama then answers with an empty reply and no tool calls (as seen in the first researcher run), so it is a
    # HumanMessage here
    result = researcher_agent.invoke(
        input={
            "messages": [
                state["query"],
                HumanMessage(content=context_message)
            ]
        }
    )
    messages = result["messages"]
    last_ai_message: AIMessage = messages[-1]

    # Second model call: turn the agent's text into a ResearchData object
    llm_with_structure = llm.with_structured_output(ResearchData)
    structured_output: ResearchData = llm_with_structure.invoke(last_ai_message.content)

    print(f"message: {last_ai_message.content[:20]}...")
    print("Done!\n")

    return {
        "messages": messages,
        "research": structured_output.research,
    }


def writer_node(state: BlogState) -> BlogState:
    """Node for the writer agent to create content."""
    print("Starting writer agent...")

    # The screenshot cuts this line off after "Research: {len(state['"; the end is my guess
    # What the writer is told: the title, the sections and how many sources were found (the source texts are not sent)
    context_message = f"Title: {state['title']}\nSections: {', '.join(state['sections'])}\nResearch: {len(state['research'])} items"

    # HumanMessage instead of the course's AIMessage, for the same reason as in research_node
    result = writer_agent.invoke(
        input={
            "messages": [
                state["query"],
                HumanMessage(content=context_message)
            ]
        }
    )
    messages = result["messages"]
    last_ai_message: AIMessage = messages[-1]

    # Second model call: turn the agent's text into a ContentDraft object
    llm_with_structure = llm.with_structured_output(ContentDraft)
    structured_output: ContentDraft = llm_with_structure.invoke(last_ai_message.content)

    print(f"message: {last_ai_message.content[:20]}...")
    print("Done!\n")

    return {
        "messages": messages,
        "draft_markdown": structured_output.draft_markdown,
        "references_markdown": structured_output.references_markdown,
    }


def reviewer_node(state: BlogState) -> BlogState:
    """Node for the reviewer agent to edit and optimize."""
    print("Starting review agent...")

    # What the reviewer is told: the draft and the keywords to check
    context_message = f"Draft: {state['draft_markdown']}...\nKeywords: {', '.join(state['keywords'])}"

    # HumanMessage instead of the course's AIMessage, for the same reason as in research_node
    result = reviewer_agent.invoke(
        input={
            "messages": [
                state["query"],
                HumanMessage(content=context_message)
            ]
        }
    )
    messages = result["messages"]
    last_ai_message: AIMessage = messages[-1]

    # Second model call: turn the agent's text into a ReviewedContent object
    llm_with_structure = llm.with_structured_output(ReviewedContent)
    structured_output: ReviewedContent = llm_with_structure.invoke(last_ai_message.content)

    print(f"message: {last_ai_message.content[:20]}...")
    print("Done!\n")

    return {
        "messages": messages,
        "final_markdown": structured_output.final_markdown,
        "seo_score": structured_output.seo_score,
    }


def publisher_node(state: BlogState) -> BlogState:
    """Node for the publisher agent to finalize and publish."""
    print("Starting publisher agent...")

    # What the publisher is told: the title and the final text
    context_message = f"Title: {state['title']}\nContent: {state['final_markdown']}"

    # HumanMessage instead of the course's AIMessage, for the same reason as in research_node
    result = publisher_agent.invoke(
        input={
            "messages": [
                state["query"],
                HumanMessage(content=context_message)
            ]
        }
    )
    messages = result["messages"]
    last_ai_message: AIMessage = messages[-1]

    # Second model call: turn the agent's text into a PublicationStatus object
    llm_with_structure = llm.with_structured_output(PublicationStatus)
    structured_output: PublicationStatus = llm_with_structure.invoke(last_ai_message.content)

    print(f"message: {last_ai_message.content[:20]}...")
    print("Done!\n")

    return {
        "messages": messages,
        "publish_status": structured_output.publish_status,
    }


# ============================================================
# GRAPH
# ============================================================

# Build the graph: each node is a step and each edge says which step comes next
workflow = StateGraph(BlogState)

workflow.add_node("intake", intake_node)
workflow.add_node("research", research_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("publisher", publisher_node)

# The order is fixed here in code: intake -> research -> writer -> reviewer -> publisher (no model chooses it)
workflow.add_edge(START, "intake")
workflow.add_edge("intake", "research")
workflow.add_edge("research", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_edge("reviewer", "publisher")
workflow.add_edge("publisher", END)


# ---- Compile and test the workflow ----
# compile() turns the blueprint into something that can run
graph = workflow.compile()

# A script cannot display the picture inline like a notebook, so save it as a file (nothing is drawn if offline).
# The course draws it with MermaidDrawMethod.PYPPETEER, which needs pyppeteer and a Chromium download; the default
# method draws through the mermaid.ink web service instead.
try:
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "blog_graph.png").write_bytes(graph.get_graph().draw_mermaid_png())
    print(f"Saved {output_dir / 'blog_graph.png'}")
except Exception as error:
    print("Could not render the PNG:", type(error).__name__)


# The request that starts the workflow
user_message = HumanMessage(
    content="I want a blog post about AI Agents"
)

# Run the whole pipeline; `result` is the final state with every field filled in
result = graph.invoke(
    input={
        "query": user_message,
    }
)

# A notebook cell would show result; a script has to print what it wants to see
print("Title:", result["title"])
print("SEO score:", result["seo_score"])
print("Publish status:", result["publish_status"])
print("\n=== FINAL MARKDOWN ===")
print(result["final_markdown"])


