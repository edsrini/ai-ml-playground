from typing import TypedDict, List, Optional, Literal, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
import uuid
import operator
from pathlib import Path


# Shared state every node reads from and writes to.
class CustomerServiceState(TypedDict):
    messages: Annotated[list, add_messages]  # reducer: new messages are appended, not overwritten
    department: Optional[Literal["billing", "technical", "sales"]]  # set by routing; overwritten on update
    user_id: str  # who is asking
    issue_id: Optional[str]  # unique id for this support issue
    user_name: Optional[str]  # customer's name, if known
    issue_history: Annotated[List[str], operator.add]  # reducer: lists are concatenated across updates
    resolved: bool  # True once the issue is closed


# Routing node: picks a department from keywords in the customer's latest message.
def route_customer(state: CustomerServiceState) -> CustomerServiceState:
    department = None
    ai_message = None
    if not state["messages"]:  # nothing to route yet
        return state
    last_message = state["messages"][-1].content.lower()  # latest customer message, lowercased for matching
    if any(word in last_message for word in ["bill", "payment", "charge"]):  # billing keywords
        department = "billing"
        ai_message = AIMessage(content="I'll help you with your billing issue. Let me check your account.")
    elif any(word in last_message for word in ["broken", "error", "not working"]):  # technical keywords
        department = "technical"
        ai_message = AIMessage(content="Let's troubleshoot your technical issue.")
    elif any(word in last_message for word in ["buy", "purchase", "upgrade"]):  # sales keywords
        department = "sales"
        ai_message = AIMessage(content="I'd be happy to help with your purchase.")
    # Only the returned keys are updated; messages is appended via its reducer, the rest overwrite.
    return {"messages": [ai_message], "department": department, "issue_id": str(uuid.uuid4()), "resolved": False}

# Billing handler: replies with account info, closes the issue, and logs a "billing-<id>" history entry.
def handle_billing(state: CustomerServiceState) -> CustomerServiceState:
    new_issues_id = "billing-" + state["issue_id"]  # tag the issue id with its department
    return {"messages": [AIMessage(content="Retrieved billing history: Last payment $99")],  # appended via reducer
            "resolved": True, "issue_history": [new_issues_id]}  # history list is concatenated via operator.add

# Technical handler: runs "diagnostics", closes the issue, and logs a "technical-<id>" history entry.
def handle_technical(state: CustomerServiceState) -> CustomerServiceState:
    new_issues_id = "technical-" + state["issue_id"]
    return {"messages": [AIMessage(content="Diagnostics complete. No hardware issues detected.")],
            "resolved": True, "issue_history": [new_issues_id]}

# Sales handler: lists products, closes the issue, and logs a "sales-<id>" history entry.
def handle_sales(state: CustomerServiceState) -> CustomerServiceState:
    new_issues_id = "sales-" + state["issue_id"]
    return {"messages": [AIMessage(content="Products available: Premium Upgrade, Family Plan.")],
            "resolved": True, "issue_history": [new_issues_id]}

workflow = StateGraph(CustomerServiceState)  # graph builder; CustomerServiceState is the shared state schema
workflow.add_node("router", route_customer)  # entry node: classifies the message into a department

# Routing function: END if already resolved, else go to the chosen department (or back to router if none).
def next_step(state: CustomerServiceState):
    if state["resolved"]:
        return END
    return state["department"] or "router"  # department is None when no keyword matched

workflow.add_node("billing", handle_billing)  # one handler node per department
workflow.add_node("technical", handle_technical)
workflow.add_node("sales", handle_sales)

workflow.set_entry_point("router")  # same as add_edge(START, "router")
workflow.add_conditional_edges("router", next_step)  # branch after router using next_step's return value
workflow.add_edge("billing", END)  # every handler closes the issue and finishes the run
workflow.add_edge("technical", END)
workflow.add_edge("sales", END)

memory = InMemorySaver()  # keeps checkpoints in RAM, so a thread_id can resume its state; lost on exit
app = workflow.compile(checkpointer=memory)  # compile validates the graph; checkpointer enables per-thread memory

# Draw the graph: PNG via the remote mermaid.ink API (flaky here), else fall back to a local ASCII drawing.
try:
    output_dir = Path(__file__).resolve().parent / "output"  # output/ next to this file
    output_dir.mkdir(exist_ok=True)
    png_path = output_dir / "customer_service_graph.png"
    png_path.write_bytes(app.get_graph().draw_mermaid_png())  # needs internet; raises if mermaid.ink is unreachable
    print(f"Graph image saved to {png_path}")
except Exception as e:
    print(f"Could not render PNG ({e}); showing ASCII graph instead:")
    app.get_graph().print_ascii()  # local, needs the grandalf package

config = {"configurable": {"thread_id": "user123"}}  # same thread_id = same saved conversation state

# Turn 1: full initial state for a new thread (note issue_id is left for route_customer to create).
state1 = CustomerServiceState(
    messages=[HumanMessage(content="My bill seems too high this month")],  # contains "bill" -> billing
    department=None,
    user_id="user123",
    user_name="John Doe",
    issue_history=[],
    resolved=False
)

result1 = app.invoke(state1, config=config)
print("Department:", result1['department'])
print("Final response:", result1['messages'][-1].content)  # last message = the department handler's reply
print("Customer issue history:", result1['issue_history'])

# Turn 2: only the new message is passed; the checkpointer restores the rest of the state for this thread.
result2 = app.invoke({"messages": [HumanMessage(content="My internet is not working properly")]},
                     config=config)  # "not working" -> technical
print("Department:", result2['department'])
print("Final response:", result2['messages'][-1].content)
print("Customer issue history:", result2['issue_history'])  # now holds both turns: operator.add concatenated the lists