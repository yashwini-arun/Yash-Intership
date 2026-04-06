import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

def call_agent(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content


def parse_json(raw: str, fallback: dict) -> dict:
    try:
        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned.strip())
    except Exception:
        return fallback


def triage_agent(alert: dict) -> dict:
    print("\n[1/5] Triage Agent running...")
    system = """You are a senior SRE triage expert. Given a production alert, you must:
1. Assign severity: P0 (critical/full outage), P1 (major degradation), P2 (partial impact), P3 (minor)
2. List all affected services
3. Estimate blast radius (how many users/systems impacted)
4. Give a one-line incident title

Respond ONLY in valid JSON with keys: severity, title, affected_services (list), blast_radius, needs_immediate_action (bool)
No explanation, no markdown, just raw JSON."""

    user = f"Production alert received:\n{json.dumps(alert, indent=2)}"
    raw = call_agent(system, user)
    return parse_json(raw, {
        "severity": "P1",
        "title": alert.get("message", "Unknown incident"),
        "affected_services": [alert.get("service", "unknown")],
        "blast_radius": "Unknown",
        "needs_immediate_action": True
    })


def rca_agent(alert: dict, triage: dict, logs: str) -> dict:
    print("[2/5] RCA Agent running...")
    system = """You are a staff engineer specializing in root cause analysis. You read logs like a detective.
Given an incident triage and log data, identify:
1. The most likely root cause (be specific)
2. Contributing factors
3. Confidence score 0-100
4. Timeline of what went wrong

Respond ONLY in valid JSON with keys: root_cause, contributing_factors (list), confidence (int), timeline (list of {time, event}), recommendation
No explanation, no markdown wrapper, just raw JSON."""

    user = f"""Triage result:
{json.dumps(triage, indent=2)}

Original alert:
{json.dumps(alert, indent=2)}

Log data:
{logs}"""
    raw = call_agent(system, user)
    return parse_json(raw, {
        "root_cause": "Unable to determine automatically",
        "contributing_factors": [],
        "confidence": 0,
        "timeline": [],
        "recommendation": "Manual investigation required"
    })


def remediation_agent(triage: dict, rca: dict) -> dict:
    print("[3/5] Remediation Agent running...")
    system = """You are an SRE with production access. Based on the root cause analysis, propose remediation steps.
Only suggest safe, reversible actions from this playbook:
restart_service, rollback_deployment, scale_replicas, flush_cache, toggle_feature_flag, block_traffic

Respond ONLY in valid JSON with keys:
- actions_taken: list of {action, target, result, reversible}
- auto_healed: bool
- manual_steps_needed: list of strings
- estimated_recovery_time: string
No explanation, no markdown, just raw JSON."""

    user = f"""Triage:
{json.dumps(triage, indent=2)}

Root cause analysis:
{json.dumps(rca, indent=2)}"""
    raw = call_agent(system, user)
    return parse_json(raw, {
        "actions_taken": [],
        "auto_healed": False,
        "manual_steps_needed": ["Manual investigation required"],
        "estimated_recovery_time": "Unknown"
    })


def comms_agent(alert: dict, triage: dict, rca: dict, remediation: dict) -> dict:
    print("[4/5] Comms Agent running...")
    system = """You are a clear communicator who translates technical incidents into stakeholder language.
Write two messages — one technical (for engineers), one non-technical (for business stakeholders).

Respond ONLY in valid JSON with keys:
- slack_engineering: string (Slack formatted, technical details)
- slack_stakeholders: string (plain language, no jargon, reassuring)
- status_page_update: string (one line, public-facing)
No explanation, no markdown wrapper, just raw JSON."""

    user = f"""Alert: {json.dumps(alert, indent=2)}
Triage: {json.dumps(triage, indent=2)}
RCA: {json.dumps(rca, indent=2)}
Remediation: {json.dumps(remediation, indent=2)}"""
    raw = call_agent(system, user)
    return parse_json(raw, {
        "slack_engineering": "Incident in progress. Investigation underway.",
        "slack_stakeholders": "We are aware of an issue and working to resolve it.",
        "status_page_update": "Investigating reports of service degradation."
    })


def incident_review_agent(alert: dict, triage: dict, rca: dict, remediation: dict, comms: dict, duration_mins: float) -> dict:
    print("[5/5] Incident Review Agent running...")
    system = """You are a blameless incident review writer following Google SRE practices.
Write a structured incident review. Focus on systems, not people. Action items must be specific and time-bound.

Respond ONLY in valid JSON with keys:
- summary: string
- impact: string
- timeline: list of {time, event}
- root_cause_summary: string
- what_went_well: list of strings
- what_went_wrong: list of strings
- action_items: list of {item, owner, due_in_days, priority}
- prevention_measures: list of strings
No explanation, no markdown, just raw JSON."""

    user = f"""Full incident context:
Alert: {json.dumps(alert, indent=2)}
Triage: {json.dumps(triage, indent=2)}
RCA: {json.dumps(rca, indent=2)}
Remediation: {json.dumps(remediation, indent=2)}
Total duration: {duration_mins} minutes"""
    raw = call_agent(system, user)
    return parse_json(raw, {
        "summary": "Incident occurred and was resolved.",
        "impact": "Unknown",
        "timeline": [],
        "root_cause_summary": rca.get("root_cause", "Unknown"),
        "what_went_well": [],
        "what_went_wrong": [],
        "action_items": [],
        "prevention_measures": []
    })