import os
from typing import Dict, Any, List
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


# ---- Intake Agent ----
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


# The request that starts the intake agent
user_trigger_message = HumanMessage(content="I want a blog post about MCP")

# Run the agent; it may call its tools several times before it gives its final answer
intake_agent_result = intake_agent.invoke(
    input={
        "messages": [
            user_trigger_message
        ]
    }
)

# A notebook cell would show the result; a script prints each message so the tool calls are visible too
for message in intake_agent_result["messages"]:
    message.pretty_print()


# Client for the Tavily web-search service; the web_search tool below uses it
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")  # the key lives in .env, never in the code
)


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


# ---- Researcher Agent ----
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

# The researcher gets the user's request plus the intake agent's final message (the JSON plan).
# The course passes intake_agent_result["messages"][-1] as it is, an AIMessage. qwen3 on Ollama answers a chat that
# ends with an assistant message with an empty reply (no tool calls), so the plan is passed as a HumanMessage instead.
researcher_agent_result = researcher_agent.invoke(
    input={
        "messages": [
            user_trigger_message,
            HumanMessage(content=intake_agent_result["messages"][-1].content),
        ]
    }
)

for message in researcher_agent_result["messages"]:
    message.pretty_print()


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


# ---- Writer Agent ----
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

# The writer gets the request, the intake plan and the research. As with the researcher, the course passes the last
# messages as they are (AIMessages); here they go in as HumanMessages so the chat does not end on an assistant message.
writer_agent_result = writer_agent.invoke(
    input={
        "messages": [
            user_trigger_message,
            HumanMessage(content=intake_agent_result["messages"][-1].content),
            HumanMessage(content=researcher_agent_result["messages"][-1].content),
        ]
    }
)

for message in writer_agent_result["messages"]:
    message.pretty_print()


# ---- Reviewer Agent ----
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

# The reviewer gets the request plus the intake plan (with the keywords), the research and the writer's draft.
# As before, the earlier agents' final messages go in as HumanMessages so the chat does not end on an assistant message.
reviewer_agent_result = reviewer_agent.invoke(
    input={
        "messages": [
            user_trigger_message,
            HumanMessage(content=intake_agent_result["messages"][-1].content),
            HumanMessage(content=researcher_agent_result["messages"][-1].content),
            HumanMessage(content=writer_agent_result["messages"][-1].content),
        ]
    }
)

for message in reviewer_agent_result["messages"]:
    message.pretty_print()


# ---- Publisher Agent ----
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

# The publisher gets the request plus the final messages of all four earlier agents, as HumanMessages for the same
# reason as before (the chat must not end on an assistant message)
publisher_agent_result = publisher_agent.invoke(
    input={
        "messages": [
            user_trigger_message,
            HumanMessage(content=intake_agent_result["messages"][-1].content),
            HumanMessage(content=researcher_agent_result["messages"][-1].content),
            HumanMessage(content=writer_agent_result["messages"][-1].content),
            HumanMessage(content=reviewer_agent_result["messages"][-1].content),
        ]
    }
)

for message in publisher_agent_result["messages"]:
    message.pretty_print()
