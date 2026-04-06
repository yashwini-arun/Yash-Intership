# MCP Server 

A fully local MCP (Model Context Protocol) server with **12 tools**, controlled through a terminal chat agent powered by **Groq's free Llama 3.3 70B**. 

---

## How it works

```
You type a question
      ↓
agent.py sends it to Groq (Llama 3.3 70B) — free
      ↓
Groq decides which tool(s) to call
      ↓
server.py runs the tool and returns the result
      ↓
Groq forms a final answer and prints it in your terminal
```

`agent.py` starts `server.py` automatically as a subprocess — you never need to run `server.py` manually.

---

## Files

| File | Purpose |
|------|---------|
| `server.py` | The MCP server — contains all 12 tools |
| `agent.py` | Terminal chat agent — connects to server.py via MCP protocol |
| `requirements.txt` | Python dependencies |

---

## Tools

| # | Tool | What it does | How |
|---|------|-------------|-----|
| 1 | `nlp_analyse` | Sentiment, emotion, keywords, entities, Flesch readability | Local |
| 2 | `analyse_code` | Code quality score /100, smells, suggestions | Local |
| 3 | `research_topic` | Academic summary + APA citations | DuckDuckGo |
| 4 | `network_intel` | IP, geolocation, DNS (A/MX/NS), port scan, threat score | ip-api.com |
| 5 | `encrypt_text` | Encrypt using base64, caesar (ROT13), reverse, or morse | Local |
| 6 | `decrypt_text` | Decrypt base64, caesar (ROT13), reverse, or morse | Local |
| 7 | `track_asset` | Live stock & crypto prices + BUY/SELL/HOLD signal | Yahoo Finance |
| 8 | `get_news` | Top headlines by category and country | newsapi.org |
| 9 | `start_personal_api` | Launch a REST API at localhost:8000 | Local |
| 10 | `stop_personal_api` | Stop the REST API | Local |
| 11 | `get_weather` | Live weather for any city | open-meteo.com |
| 12 | `generate_random` | Passwords, UUIDs, numbers, dice, coin, shuffle | Local |

---

## Setup

### Step 1 — Requirements

- Python 3.9 or higher
- pip

### Step 2 — Install dependencies

```bash
pip install fastmcp mcp httpx
```

### Step 3 — Get a free Groq API key

1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign in with Google (no credit card needed)
3. Click **Create API Key** and copy it

### Step 4 — Paste your key into agent.py

Open `agent.py` and find **line 28**:

```python
GROQ_API_KEY = "your_key_here"
```

Replace the placeholder with your actual key.

### Step 5 — (Optional) Add a NewsAPI key for `get_news`

Get a free key at [newsapi.org](https://newsapi.org) and paste it into `server.py`:

```python
NEWS_KEY = "your_newsapi_key_here"
```

---

## Running

```bash
python agent.py
```

The agent starts `server.py` automatically, loads all 12 tools, and drops you into a chat loop:

```
╔══════════════════════════════════════════════════════╗
║       MCP AGENT  —  Terminal Edition                 ║
║  Powered by Groq (Llama 3.3 70B) — Free              ║
║  No VS Code  ·  No Cline  ·  No extensions           ║
╚══════════════════════════════════════════════════════╝

You ▶
```

---

## Terminal commands

| Command | What it does |
|---------|-------------|
| `help` | Show example prompts |
| `tools` | List all 12 loaded tools |
| `exit` / `quit` / `bye` / `q` | Shut down the agent and server |

---

## Example prompts

### Weather
```
What is the weather in Hyderabad?
What is the weather in London?
```

### News
```
Show me today's technology news in India
What are the top business headlines in the US?
Get me sports news in the UK
```

### Stocks & Crypto
```
Track the stock price of AAPL and GOOGL
What is the current price of BTC-USD?
Track TCS.NS and INFY.NS
Should I buy or sell AAPL right now?
```

### Encryption & Decryption
```
Encrypt "meet me at 5pm" using base64
Decrypt "bWVldCBtZSBhdCA1cG0=" using base64
Encrypt "Hello World" using morse
Encrypt "secret" using caesar
Decrypt "URYYB" using caesar
```

### NLP analysis
```
Analyse this text: I love building things with Python!
What is the sentiment of: This project is terrible and full of bugs.
```

### Code analysis
```
Analyse this code: def add(a, b): return a + b
```

### Research
```
Research the topic: artificial intelligence in detailed depth
Give me a brief summary on: climate change
Research quantum computing in standard depth
```

### Network intelligence
```
Run network intelligence on google.com
Check the threat score of 8.8.8.8
Scan network info for github.com
```

### Random generator
```
Generate a strong 20 character password
Give me a random UUID
Roll a 20 sided dice
Pick randomly from: Pizza, Biryani, Burger, Dosa
Flip a coin
```

### Personal REST API
```
Start my personal REST API
```

Then open your browser while the agent is still running:

```
http://localhost:8000
http://localhost:8000/weather?city=Hyderabad
http://localhost:8000/news?category=technology
http://localhost:8000/random?type=password
http://localhost:8000/analyse?text=hello+world
http://localhost:8000/sysinfo
```

To stop it:
```
Stop my personal REST API
```
