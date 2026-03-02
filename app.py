"""
app.py - Fake News Detector with LLM Parameter Explorer
Features: Context Window, Temperature, Top-P, Top-K, Tiktoken Cost Estimation
"""
import os
import streamlit as st
from dotenv import load_dotenv
from agent import run_detection, MODEL_CONTEXT_WINDOWS, count_tokens

load_dotenv()

st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Sora:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family:'Sora',sans-serif; background:#07090f; color:#dde3f0; }
.block-container { max-width:780px !important; padding-top:1.5rem; }
section[data-testid="stSidebar"] { background:#0c0f1a; border-right:1px solid #1a1f2e; }

.step-card  { border-radius:10px; padding:12px 16px; margin:5px 0; font-size:13px; line-height:1.7; word-break:break-word; }
.step-label { font-family:'IBM Plex Mono',monospace; font-size:9px; letter-spacing:.1em; font-weight:600; margin-bottom:6px; }
.code-line  { font-family:'IBM Plex Mono',monospace; font-size:12px; background:#07090f; border-radius:6px; padding:6px 11px; margin-top:4px; overflow-x:auto; white-space:nowrap; }
.obs-text   { font-family:'IBM Plex Mono',monospace; font-size:11.5px; color:#6b7a99; line-height:1.75; white-space:pre-wrap; word-break:break-word; max-height:160px; overflow-y:auto; }
.connector  { width:1px; height:10px; background:#1a1f2e; margin:0 0 0 20px; }

/* param explorer cards */
.param-card {
    background:#0c0f1a; border:1px solid #1a1f2e; border-radius:12px;
    padding:16px; margin:8px 0;
}
.param-title {
    font-family:'IBM Plex Mono',monospace; font-size:10px;
    letter-spacing:.1em; font-weight:600; margin-bottom:4px;
}
.param-desc  { font-size:12px; color:#4a5568; line-height:1.7; margin-bottom:8px; }
.param-value {
    font-family:'IBM Plex Mono',monospace; font-size:18px;
    font-weight:700; text-align:center; margin:6px 0;
}

/* context window bar */
.cw-track { height:8px; background:#1a1f2e; border-radius:100px; overflow:hidden; margin:8px 0 4px 0; }

/* token cost table */
.token-row  { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1a1f2e; font-size:12px; }
.token-key  { color:#4a5568; font-family:'IBM Plex Mono',monospace; }
.token-val  { color:#dde3f0; font-family:'IBM Plex Mono',monospace; font-weight:600; }

/* verdict card */
.verdict-wrap  { border-radius:14px; padding:24px; margin-top:10px; }
.verdict-badge { display:inline-flex; align-items:center; gap:8px; border-radius:100px; padding:7px 18px; font-size:15px; font-weight:800; }
.conf-badge    { display:inline-flex; align-items:center; font-family:'IBM Plex Mono',monospace; font-size:9px; letter-spacing:.08em; background:#1a1f2e; border:1px solid #252c3d; color:#6b7a99; border-radius:100px; padding:4px 11px; margin-left:8px; vertical-align:middle; }
.claim-box     { background:#07090f; border:1px solid #1a1f2e; border-radius:8px; padding:10px 14px; font-size:13px; font-style:italic; color:#c8d0e0; line-height:1.65; margin:12px 0; }
.score-row     { display:flex; justify-content:space-between; font-family:'IBM Plex Mono',monospace; font-size:9px; color:#2a3248; margin-bottom:3px; }
.bar-track     { height:5px; background:#1a1f2e; border-radius:100px; overflow:hidden; margin-bottom:3px; }
.reasoning-text{ font-size:13px; color:#8a96b0; line-height:1.85; margin:14px 0; }
.key-finding   { border-radius:8px; padding:9px 13px; font-family:'IBM Plex Mono',monospace; font-size:11px; line-height:1.6; margin-bottom:14px; }
.sources-header{ font-family:'IBM Plex Mono',monospace; font-size:9px; letter-spacing:.09em; color:#2a3248; margin-bottom:8px; }
.source-row    { display:flex; gap:8px; align-items:flex-start; background:#07090f; border:1px solid #1a1f2e; border-radius:7px; padding:8px 12px; margin:4px 0; font-size:12px; color:#6b7a99; line-height:1.5; }
.divider-line  { display:flex; align-items:center; gap:12px; margin:18px 0 8px 0; }
.divider-line span { font-family:'IBM Plex Mono',monospace; font-size:9px; letter-spacing:.1em; color:#1a1f2e; white-space:nowrap; }
.divider-line div  { flex:1; height:1px; background:#1a1f2e; }

.stButton>button { background:linear-gradient(135deg,#4f46e5,#7c3aed) !important; color:white !important; border:none !important; border-radius:10px !important; font-family:'Sora',sans-serif !important; font-weight:700 !important; }
.stButton>button:hover { opacity:.9 !important; }
textarea, .stTextInput input { background:#0c0f1a !important; border:1px solid #1a1f2e !important; color:#dde3f0 !important; border-radius:8px !important; }
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
TOOL_CFG = {
    "analyze_sentiment": {"color":"#38bdf8","bg":"#071724","icon":"💬","label":"SENTIMENT ANALYSIS"},
    "search_news":       {"color":"#a78bfa","bg":"#100d1e","icon":"🔎","label":"NEWS SEARCH"},
    "fact_check":        {"color":"#fb923c","bg":"#160d05","icon":"✅","label":"FACT-CHECK SEARCH"},
}
VERDICT_CFG = {
    "Likely True":  {"color":"#4ade80","bg":"#041510","border":"#4ade8028","icon":"✓"},
    "Misleading":   {"color":"#fbbf24","bg":"#140f02","border":"#fbbf2428","icon":"⚠"},
    "Likely False": {"color":"#f87171","bg":"#140404","border":"#f8717128","icon":"✗"},
    "Unverified":   {"color":"#94a3b8","bg":"#0c0f1a","border":"#94a3b828","icon":"?"},
}
MODELS = list(MODEL_CONTEXT_WINDOWS.keys())

# ── render helpers ─────────────────────────────────────────────────────────────
def thought(text):
    st.markdown(f"""
<div class="step-card" style="background:#0c1220;border-left:3px solid #4f46e5;">
  <div class="step-label" style="color:#6366f1;">💭 THOUGHT</div>
  <div style="color:#8a96b0;font-size:13px;">{text}</div>
</div><div class="connector"></div>""", unsafe_allow_html=True)

def action(tool, inp):
    c = TOOL_CFG.get(tool, {"color":"#64748b","bg":"#0c0f1a","icon":"⚙","label":tool.upper()})
    safe = inp.replace("<","&lt;").replace(">","&gt;")
    st.markdown(f"""
<div class="step-card" style="background:{c['bg']};border:1px solid {c['color']}22;">
  <div class="step-label" style="color:{c['color']};">{c['icon']} ACTION &nbsp;·&nbsp; {c['label']}</div>
  <div class="code-line" style="color:{c['color']};">{tool}[<span style="color:#c8d0e0;">{safe}</span>]</div>
</div><div class="connector"></div>""", unsafe_allow_html=True)

def observation(text, is_err=False):
    col = "#f87171" if is_err else "#4ade80"
    bg  = "#120404" if is_err else "#04120a"
    bdr = "#f8717120" if is_err else "#4ade8020"
    lbl = "⚠ OBSERVATION · ERROR" if is_err else "📋 OBSERVATION · RESULT"
    safe = text.replace("<","&lt;").replace(">","&gt;")
    st.markdown(f"""
<div class="step-card" style="background:{bg};border:1px solid {bdr};">
  <div class="step-label" style="color:{col};">{lbl}</div>
  <div class="obs-text">{safe}</div>
</div><div class="connector"></div>""", unsafe_allow_html=True)

def render_token_report(tr: dict, model: str):
    ctx  = tr["context_window"]
    used = tr["total_tokens"]
    pct  = tr["context_used_pct"]
    bar_color = "#4ade80" if pct < 50 else "#fbbf24" if pct < 80 else "#f87171"

    st.markdown(f"""
<div class="divider-line"><div></div><span>TOKEN USAGE & COST ESTIMATE</span><div></div></div>

<div class="param-card">
  <div class="param-title" style="color:#6366f1;">📊 TIKTOKEN ANALYSIS · {model.upper()}</div>

  <div style="margin:10px 0 4px 0;">
    <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#2a3248;margin-bottom:4px;">
      <span>CONTEXT WINDOW USED</span>
      <span style="color:{bar_color};">{used:,} / {ctx:,} tokens ({pct}%)</span>
    </div>
    <div class="cw-track">
      <div style="height:100%;width:{min(pct,100)}%;background:{bar_color};border-radius:100px;box-shadow:0 0 8px {bar_color}70;"></div>
    </div>
  </div>

  <div style="margin-top:12px;">
    <div class="token-row"><span class="token-key">Prompt tokens</span><span class="token-val">{tr['prompt_tokens']:,}</span></div>
    <div class="token-row"><span class="token-key">Completion tokens</span><span class="token-val">{tr['completion_tokens']:,}</span></div>
    <div class="token-row"><span class="token-key">Total tokens</span><span class="token-val">{tr['total_tokens']:,}</span></div>
    <div class="token-row"><span class="token-key">Input cost (USD)</span><span class="token-val">${tr['input_cost_usd']:.8f}</span></div>
    <div class="token-row"><span class="token-key">Output cost (USD)</span><span class="token-val">${tr['output_cost_usd']:.8f}</span></div>
    <div class="token-row" style="border-bottom:none;"><span class="token-key" style="color:#dde3f0;">Total cost (USD)</span><span class="token-val" style="color:#4ade80;">${tr['total_cost_usd']:.8f}</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

def final_verdict(verdict, claim, news_txt, fact_txt):
    v   = verdict.get("verdict","Unverified")
    vc  = VERDICT_CFG.get(v, VERDICT_CFG["Unverified"])
    sc  = int(verdict.get("score",0))
    pct = sc * 10

    srcs = []
    for block in [news_txt, fact_txt]:
        for line in (block or "").split("\n"):
            line = line.strip()
            if line.startswith("[") and "]" in line:
                title = line.split("]",1)[-1].strip()
                if " (" in title: title = title[:title.rindex(" (")].strip()
                if title and title not in srcs: srcs.append(title)
    srcs = srcs[:5]

    src_rows = "".join(
        f'<div class="source-row"><span style="color:#2a3248;margin-top:2px;">→</span>{s}</div>'
        for s in srcs
    ) or '<div class="source-row"><span style="color:#2a3248;">No sources extracted.</span></div>'

    kf = verdict.get("key_finding","")
    kf_html = f'<div class="key-finding" style="background:{vc["color"]}0a;border:1px solid {vc["color"]}22;color:{vc["color"]};">🔑 {kf}</div>' if kf else ""

    st.markdown(f"""
<div class="divider-line"><div></div><span>FINAL VERDICT</span><div></div></div>
<div class="verdict-wrap" style="background:{vc['bg']};border:1px solid {vc['border']};">
  <div style="margin-bottom:14px;">
    <span class="verdict-badge" style="background:{vc['color']}18;border:1.5px solid {vc['color']}55;color:{vc['color']};">{vc['icon']} {v}</span>
    <span class="conf-badge">{verdict.get('confidence','').upper()} CONFIDENCE</span>
  </div>
  <div class="claim-box">"{claim}"</div>
  <div class="score-row"><span>CREDIBILITY SCORE</span><span style="color:{vc['color']};font-weight:700;">{sc} / 10</span></div>
  <div class="bar-track"><div style="height:100%;width:{pct}%;background:{vc['color']};border-radius:100px;box-shadow:0 0 8px {vc['color']}70;"></div></div>
  <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:8px;color:#2a3248;margin-bottom:14px;"><span>0 · FALSE</span><span>10 · TRUE</span></div>
  <div class="reasoning-text">{verdict.get('reasoning','')}</div>
  {kf_html}
  <div class="sources-header">SOURCES REFERENCED</div>
  {src_rows}
  <div style="margin-top:12px;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#1a1f2e;text-align:center;">⚠ AI-assisted analysis only · Always verify with trusted sources</div>
</div>
""", unsafe_allow_html=True)

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password",
                            value=os.getenv("GROQ_API_KEY",""), placeholder="gsk_...")
    st.caption("🆓 Free at [console.groq.com](https://console.groq.com)")

    model = st.selectbox("Model", MODELS)
    ctx_size = MODEL_CONTEXT_WINDOWS[model]
    st.markdown(f"""
<div style="background:#0c0f1a;border:1px solid #1a1f2e;border-radius:8px;padding:10px 12px;margin:4px 0;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#334155;letter-spacing:.08em;margin-bottom:4px;">CONTEXT WINDOW</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:16px;font-weight:700;color:#6366f1;">{ctx_size:,} <span style="font-size:10px;color:#2a3248;">tokens</span></div>
  <div style="font-size:11px;color:#2a3248;margin-top:2px;">≈ {ctx_size*0.75/1000:.0f}K words</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎛️ LLM Parameters")

    # Temperature
    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0, max_value=2.0, value=0.1, step=0.05,
        help="Controls randomness. 0 = deterministic, 2 = very creative/random"
    )
    temp_desc = (
        "🧊 Deterministic — same answer every time" if temperature < 0.2 else
        "✅ Balanced — slight variation, good for fact-checking" if temperature < 0.7 else
        "🎨 Creative — varied answers, less reliable for facts" if temperature < 1.3 else
        "🔥 Very random — unpredictable, not ideal for analysis"
    )
    st.caption(temp_desc)

    # Top-P
    top_p = st.slider(
        "🎯 Top-P (Nucleus Sampling)",
        min_value=0.0, max_value=1.0, value=0.9, step=0.05,
        help="Only consider tokens whose cumulative probability ≤ top_p. Lower = more focused."
    )
    tp_desc = (
        "Very focused — only top tokens considered" if top_p < 0.5 else
        "Balanced — good nucleus of probable tokens" if top_p < 0.85 else
        "Open — considers wide range of tokens"
    )
    st.caption(f"Nucleus: {tp_desc}")

    # Top-K
    top_k = st.slider(
        "🔢 Top-K",
        min_value=1, max_value=100, value=40, step=1,
        help="Only consider the top K most probable next tokens. Lower = more focused."
    )
    tk_desc = (
        "Very narrow — only most likely tokens" if top_k < 10 else
        "Focused — limited but diverse choices" if top_k < 30 else
        "Balanced — good mix of options" if top_k < 60 else
        "Wide — many token candidates considered"
    )
    st.caption(f"Sampling pool: {tk_desc}")

    # Max tokens
    max_tokens = st.slider(
        "📏 Max Output Tokens",
        min_value=100, max_value=2000, value=800, step=50,
        help="Maximum tokens the LLM can generate in its response."
    )
    st.caption(f"≈ {int(max_tokens * 0.75)} words max output")

    st.markdown("---")
    st.markdown("**ReAct Steps**")
    st.markdown("💬 Sentiment  \n🔎 News Search  \n✅ Fact-Check  \n⚖️ Verdict")
    st.markdown("---")
    st.success("💰 100% Free")

# ── header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:24px;">
  <h1 style="font-size:1.9rem;font-weight:800;letter-spacing:-.03em;
    background:linear-gradient(135deg,#f1f5f9 0%,#818cf8 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:6px;">
    🔍 Fake News Detector
  </h1>
  <p style="color:#3a4460;font-size:13px;">
    ReAct Agent · Context Window Explorer · Tiktoken Cost Estimator · Parameter Tuning
  </p>
</div>
""", unsafe_allow_html=True)

# ── LLM Parameter Explorer panel (always visible) ─────────────────────────────
with st.expander("🧪 LLM Parameter Explorer — What do these settings mean?", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
<div class="param-card">
  <div class="param-title" style="color:#6366f1;">🌡️ TEMPERATURE = {temperature}</div>
  <div class="param-desc">
    Controls how <b>creative vs deterministic</b> the model is.<br>
    Think of it like a dice: temperature 0 always picks the most likely word.
    Temperature 2 picks almost randomly.
  </div>
  <div class="param-value" style="color:#6366f1;">{temperature}</div>
  <div style="height:4px;background:#1a1f2e;border-radius:100px;overflow:hidden;">
    <div style="height:100%;width:{(temperature/2)*100}%;background:#6366f1;border-radius:100px;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:8px;color:#2a3248;margin-top:3px;">
    <span>0 Deterministic</span><span>2 Random</span>
  </div>
  <div style="margin-top:8px;font-size:11px;color:#4a5568;background:#07090f;border-radius:6px;padding:7px 10px;">
    For fact-checking: use <b style="color:#6366f1;">0.0–0.2</b><br>
    For creative writing: use <b style="color:#6366f1;">0.7–1.2</b>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="param-card">
  <div class="param-title" style="color:#38bdf8;">🔢 TOP-K = {top_k}</div>
  <div class="param-desc">
    At each step, only look at the <b>top K most probable</b> next words.
    Like shortlisting candidates before making a final choice.
  </div>
  <div class="param-value" style="color:#38bdf8;">{top_k}</div>
  <div style="height:4px;background:#1a1f2e;border-radius:100px;overflow:hidden;">
    <div style="height:100%;width:{(top_k/100)*100}%;background:#38bdf8;border-radius:100px;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:8px;color:#2a3248;margin-top:3px;">
    <span>1 Narrowest</span><span>100 Widest</span>
  </div>
  <div style="margin-top:8px;font-size:11px;color:#4a5568;background:#07090f;border-radius:6px;padding:7px 10px;">
    Low K (1–10) → very repetitive<br>
    Mid K (20–50) → good balance ✅<br>
    High K (80+) → noisy outputs
  </div>
</div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
<div class="param-card">
  <div class="param-title" style="color:#fb923c;">🎯 TOP-P = {top_p}</div>
  <div class="param-desc">
    Only consider tokens until their probabilities sum to P.
    Like saying "only pick from words that together make up {int(top_p*100)}% of the probability mass."
  </div>
  <div class="param-value" style="color:#fb923c;">{top_p}</div>
  <div style="height:4px;background:#1a1f2e;border-radius:100px;overflow:hidden;">
    <div style="height:100%;width:{top_p*100}%;background:#fb923c;border-radius:100px;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:8px;color:#2a3248;margin-top:3px;">
    <span>0 Narrowest</span><span>1.0 All tokens</span>
  </div>
  <div style="margin-top:8px;font-size:11px;color:#4a5568;background:#07090f;border-radius:6px;padding:7px 10px;">
    0.1 → very focused<br>
    0.9 → standard ✅<br>
    1.0 → no filtering
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="param-card">
  <div class="param-title" style="color:#4ade80;">📐 CONTEXT WINDOW · {model}</div>
  <div class="param-desc">
    Maximum tokens the model can "see" at once — input + output combined.
    Like the model's working memory.
  </div>
  <div class="param-value" style="color:#4ade80;">{ctx_size:,}</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#2a3248;text-align:center;margin-bottom:6px;">tokens</div>
  <div style="font-size:11px;color:#4a5568;background:#07090f;border-radius:6px;padding:7px 10px;">
    ≈ {ctx_size * 3 // 4:,} words &nbsp;·&nbsp; ≈ {ctx_size // 750} books (avg novel)<br>
    <b style="color:#4ade80;">gemma2-9b</b> = 8,192 (small)<br>
    <b style="color:#4ade80;">llama-3.x</b> = 131,072 (huge)
  </div>
</div>
""", unsafe_allow_html=True)

# ── Context Window comparison table ───────────────────────────────────────────
with st.expander("📊 Context Window Comparison — All Groq Models", expanded=False):
    st.markdown("""
<div class="param-card">
  <div class="param-title" style="color:#818cf8;">ALL GROQ MODELS · CONTEXT WINDOW SIZE</div>
  <div style="margin-top:10px;">
""", unsafe_allow_html=True)
    for m, cw in MODEL_CONTEXT_WINDOWS.items():
        pct   = (cw / 131072) * 100
        color = "#4ade80" if cw >= 131072 else "#fbbf24" if cw >= 32000 else "#f87171"
        active = "→ " if m == model else "&nbsp;&nbsp;"
        st.markdown(f"""
<div style="margin:8px 0;">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:{'#dde3f0' if m==model else '#4a5568'};">{active}{m}</span>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:{color};">{cw:,} tokens</span>
  </div>
  <div style="height:4px;background:#1a1f2e;border-radius:100px;overflow:hidden;">
    <div style="height:100%;width:{pct}%;background:{color};border-radius:100px;"></div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

# ── Live token counter ─────────────────────────────────────────────────────────
st.markdown("---")
EXAMPLES = [
    "The moon landing in 1969 was faked by NASA",
    "A new vaccine causes autism in children",
    "5G towers are being used to spread COVID-19",
    "Climate change is causing more frequent hurricanes",
    "Drinking coffee cures cancer completely",
]
with st.expander("💡 Try an example"):
    for ex in EXAMPLES:
        if st.button(ex, key=ex):
            st.session_state["claim"] = ex
            st.rerun()

claim_val = st.text_area(
    "Claim or headline to fact-check",
    value=st.session_state.get("claim",""),
    height=90,
    placeholder="Paste any news headline or claim here…",
    label_visibility="collapsed",
    key="claim_text",
)

# Live token count while typing
if claim_val.strip():
    live_tokens = count_tokens(claim_val)
    remaining   = ctx_size - live_tokens
    pct_used    = (live_tokens / ctx_size) * 100
    tok_color   = "#4ade80" if pct_used < 10 else "#fbbf24" if pct_used < 50 else "#f87171"
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-top:4px;margin-bottom:2px;">
  <div style="flex:1;height:3px;background:#1a1f2e;border-radius:100px;overflow:hidden;">
    <div style="height:100%;width:{pct_used:.2f}%;background:{tok_color};border-radius:100px;"></div>
  </div>
  <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{tok_color};white-space:nowrap;">
    {live_tokens} tokens · {remaining:,} remaining of {ctx_size:,}
  </span>
</div>
""", unsafe_allow_html=True)

_, col_btn, _ = st.columns([1,2,1])
with col_btn:
    run = st.button("🚀 Run ReAct Agent", use_container_width=True)

# ── run ────────────────────────────────────────────────────────────────────────
if run:
    if not api_key:
        st.error("❌ Enter your Groq API key in the sidebar.")
        st.stop()
    if len(claim_val.strip()) < 10:
        st.warning("⚠️ Enter at least 10 characters.")
        st.stop()

    clean = claim_val.strip()
    st.markdown("---")
    st.markdown(f"""
<div style="background:#0c0f1a;border:1px solid #1a1f2e;border-radius:10px;padding:12px 16px;margin-bottom:18px;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#2a3248;letter-spacing:.08em;margin-bottom:4px;">ANALYZING CLAIM</div>
  <div style="color:#c8d0e0;font-size:13px;font-style:italic;">"{clean}"</div>
  <div style="margin-top:8px;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#2a3248;">
    model={model} &nbsp;·&nbsp; temp={temperature} &nbsp;·&nbsp; top_p={top_p} &nbsp;·&nbsp; top_k={top_k} &nbsp;·&nbsp; max_tokens={max_tokens}
  </div>
</div>
""", unsafe_allow_html=True)

    with st.spinner("Running ReAct agent…"):
        try:
            result = run_detection(
                claim=clean, api_key=api_key, model=model,
                temperature=temperature, max_tokens=max_tokens,
                top_p=top_p, top_k=top_k,
            )
        except Exception as e:
            err = str(e)
            st.error(f"❌ {err}")
            if "401" in err or "auth" in err.lower(): st.info("💡 Invalid API key.")
            elif "decommissioned" in err.lower():     st.info("💡 Model removed — pick another.")
            st.stop()

    obs_list = [s for s in result["steps"] if s["type"] == "observation"]
    news_txt = obs_list[1]["text"] if len(obs_list) > 1 else ""
    fact_txt = obs_list[2]["text"] if len(obs_list) > 2 else ""

    for step in result["steps"]:
        if   step["type"] == "thought":     thought(step["text"])
        elif step["type"] == "action":      action(step["tool"], step["input"])
        elif step["type"] == "observation":
            observation(step["text"], any(w in step["text"].lower() for w in ["error","failed","unavailable"]))
        elif step["type"] == "final":
            final_verdict(step["verdict"], step["claim"], news_txt, fact_txt)
            if step.get("token_report"):
                render_token_report(step["token_report"], model)

    st.markdown("---")
    _, cb, _ = st.columns([1,2,1])
    with cb:
        if st.button("🔄 Analyze Another", use_container_width=True):
            st.session_state["claim"] = ""
            st.rerun()

st.markdown("<div style='text-align:center;margin-top:20px;font-size:11px;color:#1a1f2e;'>Streamlit · Groq · Google News RSS · TextBlob · Tiktoken</div>", unsafe_allow_html=True)