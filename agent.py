"""
MCP Agent — Terminal Chat (Powered by Groq — Free, No Rate Limits)
Works directly with your server.py using the official MCP protocol.
No VS Code, no Cline, no Continue, no extensions needed.

How it works:
  1. Starts YOUR server.py as a proper MCP server subprocess
  2. Connects to it over the MCP protocol (STDIO transport)
  3. You type a question in the terminal
  4. Groq AI (Llama 3.3 70B) decides which tool to call
  5. Your server runs the tool and returns the real result
  6. The final answer prints in the terminal

Setup (one time only):
  1. Get free Groq API key at: https://console.groq.com/keys
  2. Paste it below on line 28
  3. pip install mcp httpx fastmcp
  4. python agent.py
"""

import asyncio, sys, json, os
import httpx
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = "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"   # get free key at console.groq.com
GROQ_MODEL   = "llama-3.3-70b-versatile"  # free model, 14400 req/day
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
SERVER_PATH  = str(Path(__file__).parent / "server.py")

# ─── TERMINAL COLOURS ─────────────────────────────────────────────────────────
R     = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YEL   = "\033[93m"
BLUE  = "\033[94m"
RED   = "\033[91m"

def clr(text, *codes): return "".join(codes) + str(text) + R

def banner():
    print(clr("""
╔══════════════════════════════════════════════════════╗
║       MCP AGENT  —  Terminal Edition                 ║
║  Powered by Groq (Llama 3.3 70B) — Free             ║
║  No VS Code  ·  No Cline  ·  No extensions          ║
╚══════════════════════════════════════════════════════╝
""", CYAN, BOLD))

def step(icon, label, detail="", color=CYAN):
    d = clr(f"  {detail}", DIM) if detail else ""
    print(f"{clr(icon, color, BOLD)}  {clr(label, color)}{d}")

def show_result(text):
    print()
    print(clr("┌─ ANSWER " + "─"*44, GREEN, BOLD))
    for line in text.strip().split("\n"):
        print(clr("│ ", GREEN, BOLD) + line)
    print(clr("└" + "─"*53, GREEN, BOLD))
    print()

def show_tool_call(name, args):
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
    print(f"  {clr('⚡ calling tool:', YEL, BOLD)} {clr(name, CYAN)}({clr(args_str, DIM)})")

def show_tool_result(output):
    preview = output[:200].replace("\n", " ") + ("..." if len(output) > 200 else "")
    print(f"  {clr('📦 tool result:', GREEN, BOLD)} {clr(preview, DIM)}")

# ─── BUILD GROQ MESSAGES FROM CONVERSATION HISTORY ────────────────────────────
def build_groq_messages(conversation: list) -> list:
    """Convert internal conversation format to Groq/OpenAI message format."""
    system = {
        "role": "system",
        "content": (
            "You are a helpful AI assistant connected to a custom MCP server with 12 powerful tools:\n"
            "1. get_weather(city) — live weather for any city\n"
            "2. get_news(category, country) — top headlines\n"
            "3. track_asset(symbols) — live stocks and crypto with BUY/SELL/HOLD signals\n"
            "4. nlp_analyse(text) — sentiment, emotion, keywords, readability\n"
            "5. analyse_code(code) — code quality score out of 100\n"
            "6. network_intel(target) — cybersecurity recon on domain or IP\n"
            "7. research_topic(topic, depth) — academic summary + APA citations\n"
            "8. encrypt_text(text, method) — base64/caesar/reverse/morse\n"
            "9. decrypt_text(text, method) — reverse any encryption\n"
            "10. generate_random(type, options) — passwords, UUIDs, dice, coin\n"
            "11. start_personal_api(port) — launch REST API at localhost:8000\n"
            "12. stop_personal_api() — stop the REST API\n\n"
            "RULES:\n"
            "- Always use tools for live data (weather, stocks, news) — never guess.\n"
            "- After receiving tool results, give a clear, friendly, readable answer.\n"
            "- For casual messages (hello, thanks, bye) just reply normally without tools.\n"
            "- Be concise and helpful."
        )
    }
    messages = [system]
    for turn in conversation:
        role  = turn["role"]
        parts = turn["parts"]
        if role == "user":
            text_parts = [p["text"] for p in parts if "text" in p]
            fn_parts   = [p for p in parts if "functionResponse" in p]
            if text_parts:
                messages.append({"role": "user", "content": text_parts[0]})
            for fp in fn_parts:
                fr = fp["functionResponse"]
                messages.append({
                    "role": "tool",
                    "tool_call_id": fr["name"],
                    "content": str(fr["response"]["result"])
                })
        elif role == "model":
            tool_calls_out = []
            text_out = []
            for p in parts:
                if "functionCall" in p:
                    fc = p["functionCall"]
                    tool_calls_out.append({
                        "id": fc["name"],
                        "type": "function",
                        "function": {
                            "name": fc["name"],
                            "arguments": json.dumps(fc.get("args", {}))
                        }
                    })
                elif "text" in p:
                    text_out.append(p["text"])
            if tool_calls_out:
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls_out
                })
            elif text_out:
                messages.append({"role": "assistant", "content": "\n".join(text_out)})
    return messages

# ─── BUILD TOOL DECLARATIONS FOR GROQ ─────────────────────────────────────────
def build_tool_declarations(tools: list) -> list:
    """Convert MCP tool list to Groq/OpenAI function declarations."""
    declarations = []
    for tool in tools:
        props, required = {}, []
        schema = tool.inputSchema or {}
        for pname, pinfo in schema.get("properties", {}).items():
            props[pname] = {
                "type": pinfo.get("type", "string"),
                "description": pinfo.get("description", "")
            }
            if pname in schema.get("required", []):
                required.append(pname)
        declarations.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": (tool.description or "").split("\n")[0][:200],
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required
                }
            }
        })
    return declarations

# ─── CALL GROQ API ────────────────────────────────────────────────────────────
async def call_groq(conversation: list, tool_declarations: list) -> dict:
    """Send conversation to Groq and get back text or tool calls."""
    messages = build_groq_messages(conversation)
    payload  = {
        "model":       GROQ_MODEL,
        "messages":    messages,
        "tools":       tool_declarations,
        "tool_choice": "auto",
        "temperature": 0.3
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json"
            },
            json=payload
        )
        if resp.status_code != 200:
            raise Exception(f"Groq error {resp.status_code}: {resp.text[:400]}")
        data   = resp.json()
        choice = data["choices"][0]
        msg    = choice["message"]

        # Convert Groq response back to our internal format
        parts_out = []
        if msg.get("content"):
            parts_out.append({"text": msg["content"]})
        for tc in msg.get("tool_calls", []):
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                args = {}
            parts_out.append({
                "functionCall": {
                    "name": tc["function"]["name"],
                    "args": args
                }
            })
        finish = "STOP" if not msg.get("tool_calls") else ""
        return {"candidates": [{"content": {"parts": parts_out}, "finishReason": finish}]}

# ─── ONE FULL CONVERSATION TURN ───────────────────────────────────────────────
async def one_turn(user_message: str, session: ClientSession, tools: list, tool_declarations: list) -> str:
    """
    Handle one full turn:
    user message → Groq → (tool calls → results)* → final text answer
    """
    conversation = [{"role": "user", "parts": [{"text": user_message}]}]

    for round_num in range(8):  # max 8 tool-call rounds
        response  = await call_groq(conversation, tool_declarations)
        candidate = response["candidates"][0]
        parts     = candidate["content"]["parts"]
        finish    = candidate.get("finishReason", "")

        tool_calls = [p for p in parts if "functionCall" in p]
        text_parts = [p for p in parts if "text" in p]

        if tool_calls:
            # Add model's response to history
            conversation.append({"role": "model", "parts": parts})
            tool_responses = []

            for tc in tool_calls:
                fn_name = tc["functionCall"]["name"]
                fn_args = tc["functionCall"].get("args", {})

                show_tool_call(fn_name, fn_args)

                # Actually call the tool on your server.py
                try:
                    result   = await session.call_tool(fn_name, fn_args)
                    tool_out = result.content[0].text if result.content else "(no output)"
                except Exception as ex:
                    tool_out = f"Tool error: {ex}"

                show_tool_result(tool_out)

                tool_responses.append({
                    "functionResponse": {
                        "name": fn_name,
                        "response": {"result": tool_out}
                    }
                })

            # Feed results back for Groq to form final answer
            conversation.append({"role": "user", "parts": tool_responses})

        elif text_parts:
            return "\n".join(p.get("text", "") for p in text_parts).strip()

        elif finish == "STOP":
            return "(Completed — no text response)"

        else:
            return "(Unexpected response from Groq)"

    return "(Maximum tool-call rounds reached)"

# ─── MAIN INTERACTIVE CHAT LOOP ───────────────────────────────────────────────
async def chat():
    banner()

    # Validate config
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        print(clr("  ERROR: Paste your Groq API key on line 28 of agent.py", RED, BOLD))
        print(clr("  Get a FREE key (no credit card) at: https://console.groq.com/keys\n", YEL))
        sys.exit(1)

    if not Path(SERVER_PATH).exists():
        print(clr(f"  ERROR: server.py not found at:\n    {SERVER_PATH}", RED, BOLD))
        print(clr("  Put agent.py and server.py in the same folder.\n", YEL))
        sys.exit(1)

    step("🔌", "Starting your MCP server...", SERVER_PATH)

    server_params = StdioServerParameters(
        command="python",
        args=[SERVER_PATH],
        env={**os.environ}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()
            step("✅", "MCP server connected!", "", GREEN)

            # Load all tools from your server
            tools_result     = await session.list_tools()
            tools            = tools_result.tools
            tool_declarations = build_tool_declarations(tools)

            print()
            print(clr(f"  {len(tools)} tools loaded from server.py:", BLUE, BOLD))
            for i, t in enumerate(tools, 1):
                desc = (t.description or "").split("\n")[0][:55]
                print(f"    {clr(str(i).rjust(2), DIM)}. {clr(t.name, CYAN)}  {clr(desc, DIM)}")
            print()
            print(clr("─" * 56, DIM))
            print(clr("  Type your question and press Enter.", YEL))
            print(clr("  Commands: 'help' · 'tools' · 'exit'", DIM))
            print(clr("─" * 56, DIM))
            print()

            while True:
                try:
                    user_input = input(clr("You ▶  ", CYAN, BOLD)).strip()
                except (EOFError, KeyboardInterrupt):
                    print(clr("\n  Goodbye!\n", DIM))
                    break

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "q", "bye"):
                    print(clr("\n  Goodbye! MCP server shutting down.\n", DIM))
                    break

                if user_input.lower() == "tools":
                    print()
                    for i, t in enumerate(tools, 1):
                        print(f"  {clr(str(i).rjust(2), DIM)}. {clr(t.name, CYAN)}")
                    print()
                    continue

                if user_input.lower() == "help":
                    print(clr("""
  Example questions you can ask:
  ──────────────────────────────────────────────────────
  what is the weather in Hyderabad?
  track AAPL and BTC-USD
  get me technology news in India
  run network intel on google.com
  encrypt "meet me" using morse
  decrypt "-- . . -" using morse
  encrypt "hello" using base64
  decrypt "aGVsbG8=" using base64
  analyse this code: def add(a,b): return a+b
  analyse this text: I love building things with Python
  research the topic artificial intelligence
  generate a secure password of 20 characters
  generate a UUID
  roll a dice
  start my personal API on port 8000
  ──────────────────────────────────────────────────────
""", DIM))
                    continue

                print()
                step("🤖", "Thinking...", "", YEL)

                try:
                    answer = await one_turn(user_input, session, tools, tool_declarations)
                    show_result(answer)
                except Exception as ex:
                    print(clr(f"\n  ERROR: {ex}\n", RED))

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(chat())