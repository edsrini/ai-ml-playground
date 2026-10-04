"""Score the agent's triage against the known category of each synthetic incident."""
from data import INCIDENTS
from diagnosis import diagnose_incident

correct = 0
for incident in INCIDENTS:
    result = diagnose_incident(incident)
    ok = result["category"] == incident["expected_category"]
    correct += ok
    print(f"{incident['incident_id']}: expected={incident['expected_category']:<20} got={result['category']:<20} {'OK' if ok else 'MISS'}")

print(f"\ntriage accuracy: {correct}/{len(INCIDENTS)}")
