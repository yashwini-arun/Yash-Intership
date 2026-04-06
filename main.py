import json
import time
import sys
from datetime import datetime
from agents import (
    triage_agent, rca_agent, remediation_agent,
    comms_agent, incident_review_agent
)
from scenarios import SCENARIOS


def print_separator(title: str):
    width = 60
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def print_json(data: dict):
    print(json.dumps(data, indent=2))


def run_incident(scenario_key: str):
    if scenario_key not in SCENARIOS:
        print(f"Unknown scenario. Choose from: {', '.join(SCENARIOS.keys())}")
        sys.exit(1)

    scenario = SCENARIOS[scenario_key]
    alert = scenario["alert"]
    logs = scenario["logs"]

    print_separator(f"INCIDENT FIRED: {alert['id']}")
    print(f"  Service   : {alert['service']}")
    print(f"  Message   : {alert['message']}")
    print(f"  Timestamp : {alert['timestamp']}")
    print(f"  Scenario  : {scenario_key}")

    start_time = time.time()

    # ── Agent 1: Triage ──────────────────────────────────────
    triage = triage_agent(alert)
    print_separator("TRIAGE RESULT")
    print_json(triage)

    # ── Agent 2: Root Cause Analysis ─────────────────────────
    rca = rca_agent(alert, triage, logs)
    print_separator("ROOT CAUSE ANALYSIS")
    print_json(rca)

    # ── Agent 3: Remediation ─────────────────────────────────
    remediation = remediation_agent(triage, rca)
    print_separator("REMEDIATION ACTIONS")
    print_json(remediation)

    # ── Agent 4: Communications ──────────────────────────────
    comms = comms_agent(alert, triage, rca, remediation)
    print_separator("COMMUNICATIONS DRAFTED")
    print(f"\n[Engineering Slack]\n{comms.get('slack_engineering', '')}")
    print(f"\n[Stakeholder Slack]\n{comms.get('slack_stakeholders', '')}")
    print(f"\n[Status Page]\n{comms.get('status_page_update', '')}")

    # ── Agent 5: Incident Review ──────────────────────────────
    elapsed_mins = round((time.time() - start_time) / 60, 1)
    review = incident_review_agent(alert, triage, rca, remediation, comms, elapsed_mins)
    print_separator("INCIDENT REVIEW DOCUMENT")
    print_json(review)

    # ── Final Summary ─────────────────────────────────────────
    total_elapsed = time.time() - start_time
    print_separator("PIPELINE COMPLETE")
    print(f"  Incident ID     : {alert['id']}")
    print(f"  Severity        : {triage.get('severity', 'Unknown')}")
    print(f"  Root cause      : {rca.get('root_cause', 'Unknown')[:80]}...")
    print(f"  Auto-healed     : {remediation.get('auto_healed', False)}")
    print(f"  Pipeline time   : {total_elapsed:.1f}s")
    print(f"  Industry avg    : ~2-4 hours manual")
    print()


def main():
    print("\n  IT Incident Management — A2A Pipeline")
    print("  Powered by Claude agent-to-agent architecture\n")

    if len(sys.argv) > 1:
        scenario = sys.argv[1]
    else:
        print("Available scenarios:")
        for i, key in enumerate(SCENARIOS.keys(), 1):
            s = SCENARIOS[key]
            print(f"  {i}. {key:20s} — {s['alert']['message'][:55]}...")
        print()
        choice = input("Enter scenario name or number (default: db_overload): ").strip()

        keys = list(SCENARIOS.keys())
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            scenario = keys[int(choice) - 1]
        elif choice in SCENARIOS:
            scenario = choice
        else:
            scenario = "db_overload"

    run_incident(scenario)


if __name__ == "__main__":
    main()