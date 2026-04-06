# IT Incident Management — A2A Pipeline

> **2–4 hours of manual incident response, automated in ~5 seconds.**  
> A production-grade, agent-to-agent (A2A) pipeline that triages, diagnoses, remediates, communicates, and documents IT incidents — with a single human approval gate before any action is taken on your servers.

---

## Overview

When production breaks at 2 AM, the traditional workflow is painful: an engineer gets paged, spends 30–90 minutes reading logs, pings colleagues, attempts a fix, and writes a ticket. Users are down the entire time.

This pipeline replaces that workflow with five specialized AI agents that hand off to each other automatically. A human is only required at **Agent 3** — the remediation step — before anything touches production infrastructure.

```
Alert → Triage → RCA → Remediation (✋ Human Gate) → Comms → Incident Review
```

---

## Architecture

```
scenarios.py  ──►  main.py (Orchestrator)  ──►  agents.py (5 AI Agents)
     │                      │
  Alert +              Sequential A2A
   Logs              (each output feeds
                       the next agent)
```

### The Five Agents

| # | Agent | Input | Output |
|---|-------|-------|--------|
| 1 | **Triage Agent** | Raw alert | Severity (P0–P3), affected services, blast radius |
| 2 | **RCA Agent** | Alert + triage + logs | Root cause, confidence score, event timeline |
| 3 | **Remediation Agent** | Triage + RCA | Proposed safe/reversible actions *(requires human approval)* |
| 4 | **Comms Agent** | All prior outputs | Slack messages for engineers & stakeholders, status page copy |
| 5 | **Incident Review Agent** | Full incident context | Blameless postmortem with action items and owners |

Each agent is prompted to respond in structured JSON only, and every agent has a safe fallback if the LLM call fails.

---

## Quickstart

### Prerequisites

- Python 3.9+
- A [Groq](https://console.groq.com) API key (free tier works)

### Installation

```bash
git clone <your-repo-url>
cd incident-pipeline

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Run

```bash
# Interactive mode — choose a scenario from the menu
python main.py

# Run a specific scenario directly
python main.py db_overload
python main.py memory_leak
python main.py bad_deployment
```

---

## Scenarios

Three production-realistic incident scenarios are included out of the box:

| Key | Service | Description |
|-----|---------|-------------|
| `db_overload` | payment-service | DB CPU at 98%, query timeouts, connection pool exhausted due to table bloat and a competing batch job |
| `memory_leak` | api-gateway | Pod memory at 94%, OOMKill imminent — traced to a middleware regression in v2.4.1 that doesn't close connections |
| `bad_deployment` | user-service | Error rate spiked to 34% after deployment v3.2.0 introduced a NullPointerException for users missing a new optional field |

Each scenario includes a realistic alert payload and timestamped log stream.

---

## Adding a Custom Scenario

Add an entry to `scenarios.py`:

```python
SCENARIOS = {
    "your_scenario": {
        "alert": {
            "id": "INC-2024-004",
            "service": "your-service-name",
            "message": "Description of what went wrong",
            "metric": "metric.name",
            "value": 99.0,
            "threshold": 80.0,
            "environment": "production",
            "region": "us-east-1",
            "timestamp": "2024-01-18T10:00:00Z",
            "tags": ["tag1", "tag2"]
        },
        "logs": """
2024-01-18T09:55:00Z [your-service] ERROR Something went wrong...
        """
    }
}
```

---

## Sample Output

```
============================================================
  INCIDENT FIRED: INC-2024-002
============================================================
  Service   : api-gateway
  Message   : api-gateway pod memory at 94% (7.5GB/8GB), OOMKill imminent
  Scenario  : memory_leak

[1/5] Triage Agent running...
[2/5] RCA Agent running...
[3/5] Remediation Agent running...
[4/5] Comms Agent running...
[5/5] Incident Review Agent running...

============================================================
  PIPELINE COMPLETE
============================================================
  Incident ID     : INC-2024-002
  Severity        : P1
  Root cause      : deploy v2.4.1 middleware didn't close connections → 2.1M objects in heap
  Auto-healed     : True
  Pipeline time   : 4.8s
  Industry avg    : ~2-4 hours manual
```

## Project Structure

```
.
├── main.py           # Orchestrator — runs the pipeline end-to-end
├── agents.py         # All five AI agent definitions
├── scenarios.py      # Incident scenarios (alerts + log data)
├── requirements.txt  # Python dependencies
└── .env              # API keys (not committed)
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `groq` | LLM inference via Groq API (llama-3.3-70b-versatile) |
| `python-dotenv` | Load `GROQ_API_KEY` from `.env` |

---

## Model

All agents use **`llama-3.3-70b-versatile`** via Groq's inference API. The model is configured centrally in `agents.py` — swap to any Groq-supported model by changing the `MODEL` constant.

---
