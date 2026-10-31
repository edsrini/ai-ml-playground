import random
from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command


# The shared state of the workflow: the conversation plus one true/false flag per stage
class State(MessagesState):
    doctor_available: bool
    patient_verified: bool
    appointment_scheduled: bool
    notifications_sent: bool


# Why a coordinator (supervisor) pattern suits this workflow:
# - The stages depend on each other and must run in order: the doctor must be free, the patient verified, the slot
#   booked, and only then are notifications sent. One node that owns that order keeps it in a single place.
# - Any stage can fail and must be retried without restarting the rest. Every agent hands control back to the
#   coordinator, which reads the flags again and sends the work to the first unfinished stage, so a retry needs no
#   extra wiring.
# - Each worker agent stays small and does one job, so it can be tested alone and replaced (a mock now, a real
#   calendar or messaging service later) without touching the others.
# - Adding a step, such as insurance checks, means one more flag and one more branch here, not rewiring every agent
#   to every other agent.
# - All routing decisions pass through one function, so logging, approvals and audit trails (important in healthcare)
#   can be added at a single point.
# Trade-offs: every step makes an extra trip through the coordinator, and the coordinator is a single point to get
# right. For a short fixed sequence with no retries or branches, a plain chain of steps would be simpler.
def appointment_coordinator(state: State) -> Command[Literal[
    "doctor_availability_agent",
    "patient_verification_agent",
    "scheduling_agent",
    "notification_agent",
    "__end__",
]]:
    """Supervisor agent that orchestrates the appointment workflow"""
    print("🔄 Appointment Coordinator: Processing appointment request...")

    # Check current state to determine next step
    if not state["doctor_available"]:
        next_step = "doctor_availability_agent"
        print(f"📋 Routing to: {next_step}")
    elif not state["patient_verified"]:
        next_step = "patient_verification_agent"
        print(f"📋 Routing to: {next_step}")
    elif not state["appointment_scheduled"]:
        next_step = "scheduling_agent"
        print(f"📋 Routing to: {next_step}")
    elif not state["notifications_sent"]:
        next_step = "notification_agent"
        print(f"📋 Routing to: {next_step}")
    else:
        # The screenshot ends before this branch; every stage is done, so the workflow finishes
        next_step = END
        print("📋 All steps complete")

    return Command(goto=next_step)


def doctor_availability_agent(state: State) -> Command[Literal["appointment_coordinator"]]:
    """Checks doctor availability and schedules"""
    print("👨‍⚕️ Doctor Availability Agent: Checking schedules...")

    # MOCK: this does not read any real schedule or calendar. The availability is a coin flip, True or False at random,
    # standing in for the answer a hospital scheduling system or database would give. Replace this line with a real
    # lookup (a calendar API or a database query, perhaps as a tool the agent calls) when the system becomes real.
    available = random.choice([True, False])
    if available:
        print("✅ Doctor is available for requested time")
    else:
        print("❌ Doctor not available, suggesting alternatives")

    return Command(
        goto="appointment_coordinator",
        update={"doctor_available": available}
    )


def patient_verification_agent(state: State) -> Command[Literal["appointment_coordinator"]]:
    """Verifies patient identity and eligibility"""
    print("🆔 Patient Verification Agent: Verifying patient...")

    # Simulate patient verification
    verified = random.choice([True, False])
    if verified:
        print("✅ Patient verified and eligible")
    else:
        print("❌ Patient verification failed")

    return Command(
        goto="appointment_coordinator",
        update={"patient_verified": verified}
    )


def scheduling_agent(state: State) -> Command[Literal["appointment_coordinator"]]:
    """Books the actual appointment"""
    print("📅 Scheduling Agent: Booking appointment...")

    # Simulate appointment booking
    booked = random.choice([True, False])
    if booked:
        print("✅ Appointment successfully booked")
    else:
        print("❌ Appointment booking failed")

    return Command(
        goto="appointment_coordinator",
        update={"appointment_scheduled": booked}
    )


def notification_agent(state: State) -> Command[Literal["appointment_coordinator"]]:
    """Sends notifications and reminders"""
    print("📢 Notification Agent: Sending notifications...")

    # Simulate sending notifications
    notifications_sent = random.choice([True, False])
    if notifications_sent:
        print("✅ Notifications sent successfully")
    else:
        print("❌ Failed to send notifications")

    return Command(
        goto="appointment_coordinator",
        update={"notifications_sent": notifications_sent}
    )


workflow = StateGraph(State)

# Add all nodes
workflow.add_node(appointment_coordinator)
workflow.add_node(doctor_availability_agent)
workflow.add_node(patient_verification_agent)
workflow.add_node(scheduling_agent)
workflow.add_node(notification_agent)

# Only the entry is wired with an edge: every other move is decided by the Command each node returns (goto)
workflow.add_edge(START, "appointment_coordinator")


# Compile and visualize:
graph = workflow.compile()

# A script cannot display the picture inline like a notebook, so save it as a file (nothing is drawn if offline)
try:
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "appointment_graph.png").write_bytes(graph.get_graph().draw_mermaid_png())
    print(f"Saved {output_dir / 'appointment_graph.png'}")
except Exception as error:
    print("Could not render the PNG:", type(error).__name__)


# Test the workflow
print("Testing Healthcare Appointment System...")
result = graph.invoke(
    input={
        "doctor_available": False,
        "patient_verified": False,
        "appointment_scheduled": False,
        "notifications_sent": False,
    }
)
print("✅ Workflow execution completed!")
