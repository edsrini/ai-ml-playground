"""The healthcare appointment workflow with real worker agents: each worker is a create_react_agent with tools.

healthcare_appointment_system.py mocks every worker with random.choice. Here the same coordinator and graph shape are
kept, but each worker is a model that calls tools (fake schedule, patient and booking data stand in for real systems),
reads what they return and answers YES or NO.
"""
import os
import re
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ---- Fake systems the tools talk to (a hospital's real calendar, patient records and messaging would be here) ----
DOCTOR_SCHEDULES = {
    "dr. khan": ["09:00", "11:30"],
    "dr. rao": [],  # fully booked
}
PATIENTS = {
    "P001": {"name": "Aisha Noor", "insured": True},
    "P002": {"name": "Omar Ali", "insured": False},
}
bookings = []  # filled by book_slot
sent_messages = []  # filled by send_notification


# ---- Tools: what the workers can do. The docstring is what the model reads to decide when to call each one ----
@tool
def get_doctor_schedule(doctor: str) -> str:
    """Returns the free appointment times of a doctor, for example for 'Dr. Khan'."""
    slots = DOCTOR_SCHEDULES.get(doctor.lower().strip())
    if slots is None:
        return f"There is no doctor named {doctor}."
    if not slots:
        return f"{doctor} has no free times."
    return f"{doctor} is free at: {', '.join(slots)}."


@tool
def lookup_patient(patient_id: str) -> str:
    """Looks up a patient by id (for example 'P001') and says whether the patient is registered and insured."""
    patient = PATIENTS.get(patient_id.upper().strip())
    if patient is None:
        return f"No patient with id {patient_id} is registered."
    insured = "insured" if patient["insured"] else "NOT insured"
    return f"{patient['name']} (id {patient_id}) is registered and {insured}."


@tool
def book_slot(doctor: str, time: str, patient_id: str) -> str:
    """Books an appointment: the doctor's name, a time such as '11:30', and the patient id."""
    slots = DOCTOR_SCHEDULES.get(doctor.lower().strip(), [])
    if time not in slots:
        return f"Cannot book: {doctor} is not free at {time}."
    slots.remove(time)  # the slot is taken
    bookings.append({"doctor": doctor, "time": time, "patient_id": patient_id})
    return f"Booked {doctor} at {time} for patient {patient_id}."


@tool
def send_notification(patient_id: str, message: str) -> str:
    """Sends a text message to the patient with the given id."""
    sent_messages.append({"patient_id": patient_id, "message": message})
    return f"Message sent to patient {patient_id}."


# ---- The state and the model ----
# Four yes/no flags, one per stage; the coordinator reads them to decide which stage runs next.
# `messages` (from MessagesState) holds the chat history.
class State(MessagesState):
    doctor_available: bool
    patient_verified: bool
    appointment_scheduled: bool
    notifications_sent: bool


# One local model shared by all four workers
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)

ANSWER_FORMAT = " Use your tools, then begin your final answer with the single word YES or NO, followed by a short reason."

# One ReAct agent per worker: the same model, a different job and a different set of tools
doctor_agent = create_react_agent(
    model=llm,
    tools=[get_doctor_schedule],
    prompt="You check doctor availability. Is the doctor free at the time the patient asked for?" + ANSWER_FORMAT,
)
patient_agent = create_react_agent(
    model=llm,
    tools=[lookup_patient],
    prompt="You verify patients. Is the patient registered and insured?" + ANSWER_FORMAT,
)
scheduling_agent = create_react_agent(
    model=llm,
    tools=[book_slot],
    prompt="You book appointments with the doctor, time and patient from the request. Was the booking made?"
           + ANSWER_FORMAT,
)
notification_agent = create_react_agent(
    model=llm,
    tools=[send_notification],
    prompt="You notify the patient that the appointment is confirmed, with the doctor and the time. Was the message sent?"
           + ANSWER_FORMAT,
)


def make_worker_node(title: str, worker, flag: str):
    """Wraps a ReAct agent as a graph node: run the agent, read its YES or NO, set the flag, go back to the coordinator."""

    def node(state: State) -> Command[Literal["appointment_coordinator", "__end__"]]:
        print(f"{title}: working...")
        # Give the worker the whole chat so far; it may call its tool before it answers
        result = worker.invoke({"messages": state["messages"]})
        reply = result["messages"][-1].content

        # The first YES or NO in the reply is the answer; no clear answer counts as NO
        match = re.search(r"\b(YES|NO)\b", reply.upper())
        succeeded = bool(match) and match.group(1) == "YES"
        print(f"{'✅' if succeeded else '❌'} {title}: {reply.strip()}")

        update = {flag: succeeded, "messages": [AIMessage(content=reply, name=title)]}
        # The data is real now, so asking again would give the same answer: a failed stage ends the workflow
        return Command(goto="appointment_coordinator" if succeeded else END, update=update)

    return node


# The coordinator is the same fixed logic as before: the first unfinished stage runs next
def appointment_coordinator(state: State) -> Command[Literal[
    "doctor_availability_agent",
    "patient_verification_agent",
    "scheduling_agent",
    "notification_agent",
    "__end__",
]]:
    """Supervisor agent that orchestrates the appointment workflow"""
    print("🔄 Appointment Coordinator: Processing appointment request...")

    if not state["doctor_available"]:
        next_step = "doctor_availability_agent"
    elif not state["patient_verified"]:
        next_step = "patient_verification_agent"
    elif not state["appointment_scheduled"]:
        next_step = "scheduling_agent"
    elif not state["notifications_sent"]:
        next_step = "notification_agent"
    else:
        next_step = END
    print(f"📋 Routing to: {next_step}")

    return Command(goto=next_step)


# Build the graph: the coordinator and the four workers are the nodes
workflow = StateGraph(State)

workflow.add_node(appointment_coordinator)
workflow.add_node("doctor_availability_agent", make_worker_node("Doctor Availability Agent", doctor_agent, "doctor_available"))
workflow.add_node("patient_verification_agent", make_worker_node("Patient Verification Agent", patient_agent, "patient_verified"))
workflow.add_node("scheduling_agent", make_worker_node("Scheduling Agent", scheduling_agent, "appointment_scheduled"))
workflow.add_node("notification_agent", make_worker_node("Notification Agent", notification_agent, "notifications_sent"))

# The graph starts at the coordinator. After that the nodes move the flow themselves with Command(goto=...)
# (coordinator -> a worker -> back to the coordinator), so there are no other edges
workflow.add_edge(START, "appointment_coordinator")

# compile() turns the blueprint into something that can run
graph = workflow.compile()

# A script cannot display the picture inline like a notebook, so save it as a file (nothing is drawn if offline)
try:
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "appointment_graph_b.png").write_bytes(graph.get_graph().draw_mermaid_png())
    print(f"Saved {output_dir / 'appointment_graph_b.png'}")
except Exception as error:
    print("Could not render the PNG:", type(error).__name__)


# ---- Test the workflow: one request that should work, one that should stop at patient verification ----
requests = [
    "Please book an appointment for patient P001 with Dr. Khan at 11:30.",
    "Please book an appointment for patient P002 with Dr. Khan at 09:00.",
]

# Run the whole workflow once per request, starting with every flag set to False
for request in requests:
    print(f"\nRequest: {request}")
    result = graph.invoke(
        input={
            "messages": [HumanMessage(request)],
            "doctor_available": False,
            "patient_verified": False,
            "appointment_scheduled": False,
            "notifications_sent": False,
        }
    )
    print("Flags:", {key: result[key] for key in
                     ("doctor_available", "patient_verified", "appointment_scheduled", "notifications_sent")})

print("\nBookings made:", bookings)
print("Messages sent:", sent_messages)
print("✅ Workflow execution completed!")
