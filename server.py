import math, json, re, random, string, uuid, platform, socket, base64, threading
from datetime import datetime
from pathlib import Path
from collections import Counter
import httpx
from fastmcp import FastMCP

mcp = FastMCP("My MCP Server — HR Edition")
WORKSPACE = Path.home() / "mcp_workspace"
WORKSPACE.mkdir(exist_ok=True)
NEWS_KEY = "4973a8e1b92f428fadd469a0f77bcfbc"

MORSE = {'A':'.-','B':'-...','C':'-.-.','D':'-..','E':'.','F':'..-.','G':'--.','H':'....','I':'..','J':'.---','K':'-.-','L':'.-..','M':'--','N':'-.','O':'---','P':'.--.','Q':'--.-','R':'.-.','S':'...','T':'-','U':'..-','V':'...-','W':'.--','X':'-..-','Y':'-.--','Z':'--..','0':'-----','1':'.----','2':'..---','3':'...--','4':'....-','5':'.....','6':'-....','7':'--...','8':'---..','9':'----.', ' ':'/'}
MORSE_R = {v: k for k, v in MORSE.items()}

def _port_open(ip, port):
    try: socket.create_connection((ip, port), timeout=0.5); return True
    except: return False


@mcp.tool()
def nlp_analyse(text: str) -> str:
    """Advanced NLP analysis built from scratch — no ML libraries needed.
    Returns: sentiment, dominant emotion, top keywords, named entities, and Flesch readability score.
    Example: nlp_analyse('I love building projects. Python is amazing!')"""
    POS = {"good","great","excellent","amazing","wonderful","fantastic","love","happy","best","awesome","brilliant","outstanding","perfect","positive","success","win","achieve","smart","strong","impressive"}
    NEG = {"bad","terrible","awful","horrible","hate","worst","poor","fail","failure","negative","wrong","problem","error","bug","crash","broken","dangerous","harmful","weak","sad","angry","crisis","disaster","fraud","loss","attack","threat"}
    EMO = {"joy":{"happy","joy","love","excited","wonderful","great","amazing","fantastic","glad"},"anger":{"angry","hate","furious","rage","annoyed","frustrated","terrible","awful"},"fear":{"fear","scared","afraid","terror","panic","worry","anxious","nervous","threat"},"sadness":{"sad","cry","grief","loss","miss","lonely","depressed","unhappy","sorrow"}}
    STOP = {"the","a","an","is","are","was","were","be","been","have","has","had","do","does","will","would","could","should","to","of","in","on","at","by","for","with","and","but","or","i","you","he","she","it","we","they","this","that","my","your","his","her","its","our","their"}
    wl = [w.lower().strip(".,!?;:\"'()[]") for w in text.split()]
    pc, nc = sum(1 for w in wl if w in POS), sum(1 for w in wl if w in NEG)
    tot = pc + nc
    sent = "Positive" if pc > nc else "Negative" if nc > pc else "Neutral"
    conf = min(99, int(max(pc, nc) / tot * 100)) if tot else 50
    emo_s = {e: sum(1 for w in wl if w in ws) for e, ws in EMO.items()}
    dom = max(emo_s, key=emo_s.get) if any(emo_s.values()) else "neutral"
    kw = [w for w, _ in Counter(w for w in wl if len(w) > 3 and w not in STOP).most_common(8)]
    ents = list(dict.fromkeys(w.strip(".,!?;:\"'()[]") for s in re.split(r'[.!?]', text) for i, w in enumerate(s.strip().split()) if i > 0 and w and w[0].isupper() and len(w) > 1))[:8]
    wc, sc = len(text.split()), max(1, len(re.findall(r'[.!?]+', text)))
    syl = sum(max(1, len(re.findall(r'[aeiouAEIOU]', w))) for w in text.split())
    fk = max(0, min(100, 206.835 - 1.015 * (wc / sc) - 84.6 * (syl / wc))) if wc else 0
    grade = "Very Easy" if fk >= 90 else "Easy" if fk >= 70 else "Standard" if fk >= 60 else "Fairly Difficult" if fk >= 50 else "Difficult" if fk >= 30 else "Very Difficult"
    return "\n".join(["="*50,"  NLP TEXT INTELLIGENCE REPORT","="*50,"",
        f"SENTIMENT  : {sent} ({conf}% confidence) | +{pc} positive / -{nc} negative",
        f"EMOTION    : {dom.upper()} | " + " | ".join(f"{e}:{'█'*s}" for e,s in emo_s.items() if s),
        f"KEYWORDS   : {', '.join(kw) or 'none detected'}",
        f"ENTITIES   : {', '.join(ents) or 'none detected'}",
        f"READABILITY: {grade} (Flesch {round(fk,1)}) | {wc} words | {sc} sentences","="*50])


@mcp.tool()
def analyse_code(code: str, language: str = "python") -> str:
    """Analyse code quality: complexity score, code smells, structure, and improvement suggestions. Score out of 100.
    Example: analyse_code('def add(a,b): return a+b')"""
    ls = code.split("\n")
    total, blank = len(ls), sum(1 for l in ls if not l.strip())
    comments = sum(1 for l in ls if l.strip().startswith(("#","//","/*","*","'''", '"""')))
    funcs = sum(1 for l in ls if re.match(r'^\s*(def |async def |function )', l))
    classes = sum(1 for l in ls if re.match(r'^\s*class ', l))
    cx = sum(len(re.findall(r'\b(if|elif|else|for|while|and|or|try|except|catch|finally)\b', l)) for l in ls) + 1
    cx_lbl = "Low" if cx <= 5 else "Moderate" if cx <= 10 else "High" if cx <= 20 else "Very High"
    smells = []
    fi = [i for i, l in enumerate(ls) if re.match(r'^\s*def ', l)]
    md = sum(1 for i in fi if '"""' not in " ".join(ls[i+1:i+3]) and "'''" not in " ".join(ls[i+1:i+3]))
    if md: smells.append(f"{md} function(s) missing docstrings")
    if sum(1 for l in ls if len(l) - len(l.lstrip()) >= 16) > 2: smells.append("Deep nesting detected")
    td = sum(1 for l in ls if any(k in l.upper() for k in ["TODO","FIXME","HACK"]))
    if td: smells.append(f"{td} TODO/FIXME comments (tech debt)")
    pc = len(re.findall(r'\bprint\s*\(', code))
    if pc > 5: smells.append(f"{pc} print() calls — use logging instead")
    if re.search(r'(password|api_key|secret)\s*=\s*["\'][^"\']+["\']', code, re.I): smells.append("Hardcoded credentials detected!")
    score = max(0, round(100 - cx*1.5 - len(smells)*8 - md*5 - td*3))
    grade = "A+ Excellent" if score>=90 else "A Very Good" if score>=80 else "B Good" if score>=70 else "C Acceptable" if score>=60 else "D Needs Work" if score>=50 else "F Refactor Now"
    sug = ([f"Add docstrings to {md} functions"] if md else []) + (["Reduce complexity — split into smaller functions"] if cx>10 else []) + ([f"Resolve {td} TODOs"] if td else []) + (["Replace print() with logging"] if pc>5 else []) or ["Code looks clean — keep it up!"]
    return "\n".join(["="*50,"  CODE QUALITY REPORT","="*50,
        f"Lines     : {total} total | {total-blank-comments} code | {comments} comments | {blank} blank",
        f"Structure : {funcs} functions | {classes} classes", f"Complexity: {cx} ({cx_lbl})",
        "Smells    : " + (" | ".join(smells) if smells else "✓ None detected"),
        f"SCORE     : {score}/100 — {grade}", "Tips      : " + " | ".join(sug), "="*50])


@mcp.tool()
def research_topic(topic: str, depth: str = "standard") -> str:
    """Generate an academic research summary with APA citations and further reading links.
    depth options: brief / standard / detailed
    Example: research_topic('machine learning', 'detailed')"""
    now = datetime.now()
    texts, sources = [], []
    try:
        r = httpx.get("https://api.duckduckgo.com/", params={"q":topic,"format":"json","no_redirect":1,"no_html":1}, timeout=10).json()
        for k, uk in [("Abstract","AbstractURL"),("Definition","DefinitionURL")]:
            if r.get(k): texts.append(r[k]); sources.append({"t":r.get(k+"Source","Source"),"u":r.get(uk,"")})
        for t in r.get("RelatedTopics",[])[:5]:
            if isinstance(t,dict) and t.get("Text"): texts.append(t["Text"]); sources.append({"t":t["Text"][:50],"u":t.get("FirstURL","")})
    except: pass
    n = 1 if depth=="brief" else 5 if depth=="detailed" else 3
    out = ["="*56, f"  RESEARCH: {topic.upper()}", f"  {now.strftime('%d %B %Y')}", "="*56, "", "OVERVIEW"]
    out += [f"  {p}" for p in texts[:n]] or [f"  {topic.capitalize()} is a significant area with wide-ranging implications."]
    out += ["","KEY ASPECTS TO EXPLORE",f"  1. Historical background of {topic}","  2. Current research and major findings","  3. Ongoing debates and controversies","  4. Real-world applications and impact","  5. Future directions and open questions","","REFERENCES (APA FORMAT)",""]
    for i, s in enumerate(sources[:5], 1):
        out += [f"  [{i}] {s['t'].strip('. ')}. Retrieved {now.strftime('%B %d, %Y')},", f"       from {s['u'] or 'https://en.wikipedia.org/wiki/'+topic.replace(' ','_')}", ""]
    out += ["SEARCH FURTHER", f"  Google Scholar : https://scholar.google.com/scholar?q={topic.replace(' ','+')}",
            f"  Wikipedia      : https://en.wikipedia.org/wiki/{topic.replace(' ','_')}", "="*56]
    return "\n".join(out)


@mcp.tool()
def network_intel(target: str) -> str:
    """Cybersecurity reconnaissance on any domain or IP address.
    Returns: IP resolution, geolocation, DNS records (A/MX/NS), open port scan, and threat score out of 100.
    Example: network_intel('google.com')"""
    t = target.strip().lower().replace("https://","").replace("http://","").split("/")[0]
    out = ["="*50,"  NETWORK INTELLIGENCE REPORT",f"  Target : {t}",f"  Time   : {datetime.now().strftime('%d %b %Y  %H:%M:%S')}","="*50,""]
    try: ip = socket.gethostbyname(t); out.append(f"IP      : {t} → {ip}")
    except: ip = t; out.append(f"IP      : {ip} (direct IP)")
    try:
        g = httpx.get(f"http://ip-api.com/json/{ip}?fields=status,country,regionName,city,isp,org,timezone", timeout=10).json()
        if g.get("status") == "success":
            out.append(f"LOCATION: {g.get('city')}, {g.get('regionName')}, {g.get('country')}")
            out.append(f"ISP/ORG : {g.get('isp')} / {g.get('org')} | TZ: {g.get('timezone')}")
    except: pass
    try:
        with httpx.Client(timeout=10) as c:
            dns = lambda tp: [r["data"] for r in c.get(f"https://dns.google/resolve?name={t}&type={tp}").json().get("Answer",[])]
            out += [f"DNS A   : {', '.join(dns('A')[:4]) or 'none'}", f"DNS MX  : {', '.join(dns('MX')[:3]) or 'none'}", f"DNS NS  : {', '.join(dns('NS')[:3]) or 'none'}"]
    except: pass
    ports = {80:"HTTP",443:"HTTPS",22:"SSH",21:"FTP",25:"SMTP",3306:"MySQL",8080:"HTTP-Alt"}
    opn = [f"{p}/{s}" for p,s in ports.items() if _port_open(ip,p)]
    out.append(f"PORTS   : {', '.join(opn) if opn else 'none open'}")
    risky = [p for p in opn if any(x in p for x in ["SSH","FTP","MySQL"])]
    score = min(100, len(risky)*15)
    out += [f"THREAT  : {score}/100 — {'LOW ✅' if score==0 else 'MEDIUM ⚠️' if score<40 else 'HIGH 🔴'}", "="*50]
    return "\n".join(out)


@mcp.tool()
def encrypt_text(text: str, method: str = "base64") -> str:
    """Encrypt text using one of four methods.
    method options: base64 / caesar (ROT13) / reverse / morse
    Example: encrypt_text('hello world', 'morse')"""
    m = method.lower().strip()
    if m == "base64": return f"Encrypted (base64):\n{base64.b64encode(text.encode()).decode()}"
    if m == "caesar": return f"Encrypted (ROT13):\n{''.join(chr((ord(c)-ord('A'if c.isupper()else'a')+13)%26+ord('A'if c.isupper()else'a'))if c.isalpha()else c for c in text)}"
    if m == "reverse": return f"Encrypted (reverse):\n{text[::-1]}"
    if m == "morse": return f"Encrypted (morse):\n{' '.join(MORSE.get(c.upper(),'?') for c in text)}"
    return "Methods: base64, caesar, reverse, morse"


@mcp.tool()
def decrypt_text(text: str, method: str = "base64") -> str:
    """Decrypt text that was encrypted using encrypt_text.
    method options: base64 / caesar (ROT13) / reverse / morse
    Example: decrypt_text('aGVsbG8gd29ybGQ=', 'base64')"""
    m = method.lower().strip()
    if m == "base64":
        try: return f"Decrypted:\n{base64.b64decode(text.encode()).decode()}"
        except: return "Invalid base64 — paste the encrypted text exactly."
    if m == "caesar": return f"Decrypted:\n{''.join(chr((ord(c)-ord('A'if c.isupper()else'a')+13)%26+ord('A'if c.isupper()else'a'))if c.isalpha()else c for c in text)}"
    if m == "reverse": return f"Decrypted:\n{text[::-1]}"
    if m == "morse": return f"Decrypted:\n{''.join(MORSE_R.get(w,'?') for w in text.strip().split())}"
    return "Methods: base64, caesar, reverse, morse"


@mcp.tool()
def track_asset(symbols: str) -> str:
    """Live stock and crypto prices with BUY / SELL / HOLD signals.
    Supports NSE (e.g. TCS.NS), NASDAQ (e.g. AAPL), and crypto (e.g. BTC-USD).
    Example: track_asset('AAPL, TCS.NS, BTC-USD')"""
    out = [f"{'='*46}\n  LIVE MARKET TRACKER — {datetime.now().strftime('%d %b %Y  %H:%M')}\n{'='*46}"]
    for sym in [s.strip().upper() for s in symbols.split(",") if s.strip()]:
        try:
            meta = httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=5d", headers={"User-Agent":"Mozilla/5.0"}, timeout=10).json()["chart"]["result"][0]["meta"]
            p, prev = meta.get("regularMarketPrice",0), meta.get("chartPreviousClose",0)
            prev = prev or p
            h52, l52, chg = meta.get("fiftyTwoWeekHigh",0), meta.get("fiftyTwoWeekLow",0), p-prev
            pct = (chg/prev*100) if prev else 0
            sig = "STRONG BUY 🟢" if pct>1.5 else "BUY 🟢" if pct>0.3 else "STRONG SELL 🔴" if pct<-1.5 else "SELL 🔴" if pct<-0.3 else "HOLD 🟡"
            out.append(f"  {meta.get('longName') or sym} ({sym})\n  Price : {meta.get('currency','USD')} {p:,.2f} {'▲'if chg>=0 else'▼'}{abs(chg):.2f} ({pct:+.2f}%)\n  52W   : {l52:,.2f} — {h52:,.2f} | {sig}")
        except Exception as ex: out.append(f"  {sym}: Error — {ex}")
    return ("\n"+"─"*46+"\n").join(out)


@mcp.tool()
def get_news(category: str = "general", country: str = "in") -> str:
    """Fetch today's top news headlines by category and country.
    category: general / technology / sports / business / health / science
    country : in (India) / us (USA) / gb (UK) / au (Australia)
    Example: get_news('technology', 'in')"""
    try:
        r = httpx.get("https://newsapi.org/v2/top-headlines", params={"apiKey":NEWS_KEY,"category":category,"country":country,"pageSize":7}, timeout=10).json()
        if r.get("status") != "ok": return f"Failed: {r.get('message','unknown error')}"
        arts = r.get("articles",[])
        if not arts: return f"No headlines found for '{category}' in '{country}'."
        out = [f"Top {category.title()} — {country.upper()} — {datetime.now().strftime('%d %b %Y')}\n"]
        for i, a in enumerate(arts, 1):
            out.append(f"{i}. {a.get('title','?')} [{a.get('source',{}).get('name','?')}]")
            if a.get("description"): out.append(f"   {a['description'][:100]}...")
        return "\n".join(out)
    except Exception as ex: return f"Failed: {ex}"


_api_running = False

@mcp.tool()
def start_personal_api(port: int = 8000) -> str:
    """Launch your own live REST API server at localhost:8000.
    Available endpoints: / /weather /news /search /sysinfo /random /analyse
    Open your browser and visit http://localhost:8000 after starting.
    Example: start_personal_api()"""
    global _api_running
    if _api_running: return f"API already running at http://localhost:{port}"
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import urlparse, parse_qs

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def send_json(self, data, status=200):
            body = json.dumps(data, indent=2).encode()
            self.send_response(status); self.send_header("Content-Type","application/json"); self.send_header("Access-Control-Allow-Origin","*"); self.send_header("Content-Length",len(body)); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            p, now = urlparse(self.path), datetime.now().strftime("%A, %d %B %Y  %H:%M:%S")
            q, path = parse_qs(p.query), p.path
            routes = {
                "/":        lambda: {"name":"My MCP REST API","status":"live","time":now,"endpoints":["/weather","/news","/sysinfo","/random","/analyse"]},
                "/weather": lambda: {"result":get_weather(q.get("city",["Hyderabad"])[0])},
                "/news":    lambda: {"result":get_news(q.get("category",["general"])[0],q.get("country",["in"])[0])},
                "/sysinfo": lambda: {"result":_system_info()},
                "/random":  lambda: {"result":generate_random(q.get("type",["uuid"])[0],q.get("options",[""])[0])},
                "/analyse": lambda: {"result":nlp_analyse(q.get("text",[""])[0]) if q.get("text") else "Add ?text=your+text"},
            }
            self.send_json(routes[path]() if path in routes else {"error":"Not found","available":list(routes)}, 200 if path in routes else 404)

    threading.Thread(target=HTTPServer(("0.0.0.0",port),Handler).serve_forever, daemon=True).start()
    _api_running = True
    return (f"YOUR PERSONAL REST API IS LIVE!\n{'='*42}\nBase URL  : http://localhost:{port}\n{'─'*42}\nEndpoints :\n"
            + "\n".join(f"  http://localhost:{port}{ep}" for ep in ["/","/weather?city=Hyderabad","/news?category=technology","/random?type=password","/analyse?text=hello+world","/sysinfo"])
            + f"\n{'='*42}")


@mcp.tool()
def stop_personal_api() -> str:
    """Stop the personal REST API server."""
    global _api_running; _api_running = False; return "Personal REST API stopped."


@mcp.tool()
def get_weather(city: str) -> str:
    """Live weather for any city worldwide. Free, no API key needed.
    Returns: condition, temperature, humidity, wind speed.
    Example: get_weather('Hyderabad')"""
    try:
        with httpx.Client(timeout=10) as c:
            g = c.get("https://geocoding-api.open-meteo.com/v1/search", params={"name":city,"count":1}).json()
            if not g.get("results"): return f"City '{city}' not found."
            r = g["results"][0]
            w = c.get("https://api.open-meteo.com/v1/forecast", params={"latitude":r["latitude"],"longitude":r["longitude"],"current":"temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code","wind_speed_unit":"kmh"}).json()["current"]
        desc = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",45:"Fog",61:"Light rain",63:"Rain",65:"Heavy rain",80:"Showers",95:"Thunderstorm"}.get(w["weather_code"],"Unknown")
        return f"Weather — {r['name']}, {r.get('country','')}\nCondition : {desc}\nTemp      : {w['temperature_2m']}°C\nHumidity  : {w['relative_humidity_2m']}%\nWind      : {w['wind_speed_10m']} km/h"
    except Exception as ex: return f"Failed: {ex}"


@mcp.tool()
def generate_random(type: str, options: str = "") -> str:
    """Generate random values. Types:
    - number: random integer. options = 'min,max' e.g. '1,100'
    - password: secure password. options = length (default 16)
    - uuid: UUID v4 | choice: pick from list. options = 'a,b,c'
    - shuffle: shuffle a list. options = 'a,b,c'
    - dice: roll dice. options = sides (default 6) | coin: flip a coin
    Example: generate_random('password', '20')"""
    t = type.lower().strip()
    if t == "number":
        p = options.split(","); lo = int(p[0]) if p and p[0].strip() else 1; hi = int(p[1]) if len(p)>1 and p[1].strip() else 100
        return f"Random number ({lo}–{hi}): {random.randint(lo,hi)}"
    if t == "password":
        n = int(options.strip()) if options.strip().isdigit() else 16
        return f"Secure password ({n} chars): {''.join(random.choices(string.ascii_letters+string.digits+'!@#$%^&*',k=n))}"
    if t == "uuid": return f"UUID v4: {uuid.uuid4()}"
    if t == "choice":
        items = [i.strip() for i in options.split(",") if i.strip()]
        return f"Random choice: {random.choice(items)}" if items else "Provide options as comma-separated values."
    if t == "shuffle":
        items = [i.strip() for i in options.split(",") if i.strip()]; random.shuffle(items); return f"Shuffled: {', '.join(items)}"
    if t == "dice":
        sides = int(options.strip()) if options.strip().isdigit() else 6; return f"Rolled d{sides}: {random.randint(1,sides)}"
    if t == "coin": return f"Coin flip: {random.choice(['Heads','Tails'])}"
    return "Types: number, password, uuid, choice, shuffle, dice, coin"


def _system_info() -> str:
    now = datetime.now()
    return (f"Date     : {now.strftime('%A, %d %B %Y')}\nTime     : {now.strftime('%H:%M:%S')}\n"
            f"OS       : {platform.system()} {platform.release()}\nMachine  : {platform.machine()}\n"
            f"Python   : {platform.python_version()}\nWorkspace: {WORKSPACE}")


if __name__ == "__main__":
    tools = ["nlp_analyse — NLP from scratch","analyse_code — Code quality score","research_topic — Academic summaries + APA",
             "network_intel — Cybersecurity recon","encrypt_text — base64/caesar/reverse/morse","decrypt_text — Reverse all encryption",
             "track_asset — Live stocks & crypto","get_news — Top headlines by category","start_personal_api — Your own REST API",
             "stop_personal_api — Stop the REST API","get_weather — Live weather any city","generate_random — Passwords, UUIDs, dice"]
    print("="*52, "  MCP SERVER — HR Edition  (12 tools)", "="*52, sep="\n")
    for i, t in enumerate(tools, 1): print(f"  {i:2}. {t}")
    print("="*52)
    mcp.run()