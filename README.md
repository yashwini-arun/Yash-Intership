# MCP Server 

A feature-rich, fully free MCP (Model Context Protocol) server built with Python and FastMCP. Packed with **12 tools** across 6 categories — works with **Cline**, **Continue**.

---

## Tools at a glance

| # | Tool | Description | Free? |
|---|------|-------------|-------|
| 1 | `nlp_analyse` | Sentiment, emotion, keywords, entities, readability | ✅ Local |
| 2 | `analyse_code` | Code quality score, smells, suggestions | ✅ Local |
| 3 | `research_topic` | Academic summary with APA citations | ✅ DuckDuckGo |
| 4 | `network_intel` | IP, DNS, ports, geolocation, threat score | ✅ ip-api.com |
| 5 | `encrypt_text` | Encrypt via base64, caesar, reverse, morse | ✅ Local |
| 6 | `decrypt_text` | Decrypt via base64, caesar, reverse, morse | ✅ Local |
| 7 | `track_asset` | Live stock and crypto prices with signals | ✅ Yahoo Finance |
| 8 | `get_news` | Top headlines by category and country | ✅ newsapi.org |
| 9 | `start_personal_api` | Launch your own REST API at localhost:8000 | ✅ Local |
| 10 | `stop_personal_api` | Stop the personal REST API | ✅ Local |
| 11 | `get_weather` | Live weather for any city | ✅ open-meteo.com |
| 12 | `generate_random` | Numbers, passwords, UUIDs, dice, coin | ✅ Local |

---

## Requirements

- Python 3.9 or higher
- pip

---

## Installation

**Step 1 — Clone or download the project**

Place all files in one folder, for example:
```
C:\Users\YourName\my_mcp_server\
```

**Step 2 — Install dependencies**

```bash
pip install fastmcp httpx
```

**Step 3 — Add your NewsAPI key** (optional — only for `get_news`)

Open `server.py` and find this line:
```python
NEWS_KEY = "YOUR_NEWSAPI_KEY_HERE"
```
Replace with your free key from [newsapi.org](https://newsapi.org) (no credit card needed).

**Step 4 — Test the server runs**

```bash
python server.py
```

You should see all 12 tools listed in the terminal. Press `Ctrl+C` to stop.

---

## Connecting to Cline (VS Code)

Cline is a powerful autonomous coding agent that supports MCP natively.

**Step 1 — Install Cline**

Open VS Code → Extensions (`Ctrl+Shift+X`) → search **Cline** → Install.

**Step 2 — Open Cline settings**

Click the Cline icon in the left sidebar → click the settings/gear icon → select **MCP Servers**.

**Step 3 — Add your server**

Click **Edit Config** and paste the following (update the path to match your system):

```json
{
  "mcpServers": {
    "my-mcp-server": {
      "command": "python",
      "args": ["C:/Users/YourName/my_mcp_server/server.py"],
      "env": {}
    }
  }
}
```

> **Windows tip:** Use forward slashes `/` not backslashes `\` in the path.
>
> **Quick way to get the path:** Right-click `server.py` in VS Code Explorer → Copy Path.

**Step 4 — Restart Cline**

Click the refresh icon in Cline. Your 12 tools will appear automatically.

**Step 5 — Start chatting**

Type in the Cline chat box:
```
What is the weather in Hyderabad?
```

Cline will call your tool and show the result.

---

## Connecting to Continue (VS Code)

Continue is a free, open-source AI code assistant with MCP support.

**Step 1 — Install Continue**

Open VS Code → Extensions (`Ctrl+Shift+X`) → search **Continue** → Install.

**Step 2 — Get a free Gemini API key**

Go to [aistudio.google.com](https://aistudio.google.com) → Sign in with Google → click **Get API Key** → Create API key → Copy it. No credit card required.

**Step 3 — Configure Continue**

Run this command in the VS Code terminal (replace both placeholder values):

```powershell
python -c "
import json, pathlib
config = {
    'models': [{'title': 'Gemini', 'provider': 'gemini', 'model': 'gemini-2.0-flash', 'apiKey': 'YOUR_GEMINI_KEY_HERE'}],
    'mcpServers': [{'name': 'my-mcp-server', 'command': 'python', 'args': ['C:/Users/YourName/my_mcp_server/server.py']}]
}
p = pathlib.Path.home() / '.continue' / 'config.json'
p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps(config, indent=2))
print('Done! Config saved to', p)
"
```

**Step 4 — Reload Continue**

Press `Ctrl+Shift+P` → type `Continue: Reload Config` → Enter.

**Step 5 — Start chatting**

Click the Continue icon in the left sidebar → switch mode to **Agent** → type:
```
What is the weather in Hyderabad?
```

---

## Example prompts

### Weather
```
What is the weather in Hyderabad?
What is the weather in Tokyo?
```

### News
```
Show me today's technology news in India
What are the top sports headlines in the US?
Show me business news in the UK
```

### Random generator
```
Generate a strong 20 character password
Give me a random UUID
Roll a 20 sided dice
Pick randomly from: Pizza, Biryani, Burger, Dosa
Flip a coin
```

### Encryption
```
Encrypt "meet me at 5pm" using base64
Decrypt "bWVldCBtZSBhdCA1cG0=" using base64
Encrypt "Hello World" using morse code
Encrypt "secret" using caesar
```

### NLP text intelligence
```
Analyse this text: I love building MCP servers! It is amazing and I feel so excited.
What is the sentiment of: This project is terrible and full of bugs.
```

### Stock & crypto tracker
```
Track the stock price of AAPL and GOOGL
What is the current price of BTC-USD?
Track TCS.NS and INFY.NS
Should I buy or sell AAPL right now?
```

### Network intelligence
```
Run network intelligence on google.com
Check the threat score of 8.8.8.8
Scan network info for github.com
```

### Code analyser
```
Analyse this code: def add(a, b): return a + b
Check the quality of my Python script
```

### Research
```
Research the topic: artificial intelligence in detailed depth
Give me a brief summary on: climate change
Research quantum computing in standard depth
```

### Personal REST API
```
Start my personal REST API server
```
Then open your browser and visit:
```
http://localhost:8000
http://localhost:8000/weather?city=Hyderabad
http://localhost:8000/news?category=technology
http://localhost:8000/random?type=password
http://localhost:8000/analyse?text=hello+world
http://localhost:8000/sysinfo
```

---


