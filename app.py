import os, re
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from retriever import retrieve_with_rrf_details
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

st.set_page_config(page_title="BugIQ", page_icon="🔬", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #0D1117 !important;
    color: #C9D1D9 !important;
    font-family: 'Inter', sans-serif !important;
}
#MainMenu, footer, header, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stAppViewContainer"] > .main > .block-container {
    max-width: 820px !important;
    padding: 0 24px 80px !important;
    margin: 0 auto !important;
}

/* ── TOPBAR ── */
.topbar {
    width: 100%; background: #010409;
    border-bottom: 1px solid #21262D;
    padding: 0 0 20px; margin-bottom: 44px;
    display: flex; align-items: center; justify-content: space-between;
    padding-top: 20px;
}
.brand { display: flex; align-items: center; gap: 10px; }
.brand-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #3FB950; box-shadow: 0 0 8px rgba(63,185,80,0.5);
}
.brand-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 15px; font-weight: 600;
    color: #E6EDF3; letter-spacing: 2px;
}
.rrf-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; font-weight: 700; color: #F39C12;
    border: 1px solid rgba(243,156,18,0.4);
    background: rgba(243,156,18,0.08);
    padding: 3px 10px; border-radius: 20px; letter-spacing: 1px;
}
.pipe { display: flex; align-items: center; gap: 5px; }
.pc {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; padding: 3px 9px; border-radius: 20px; letter-spacing: 0.5px;
}
.pc-bm25  { color: #58A6FF; background: rgba(88,166,255,0.08); border: 1px solid rgba(88,166,255,0.2); }
.pc-faiss { color: #3FB950; background: rgba(63,185,80,0.08);  border: 1px solid rgba(63,185,80,0.2); }
.pc-rrf   { color: #F39C12; background: rgba(243,156,18,0.12); border: 1px solid rgba(243,156,18,0.45); font-weight:700; font-size:10px; }
.pc-arr   { color: #30363D; font-size: 11px; }
.pc-llm   { color: #8B949E; background: rgba(139,148,158,0.08); border: 1px solid rgba(139,148,158,0.2); }

/* ── HEADING ── */
.page-heading { text-align: center; margin-bottom: 36px; }
.page-heading h1 {
    font-size: 32px; font-weight: 700;
    color: #E6EDF3; letter-spacing: -0.5px;
    margin-bottom: 10px; line-height: 1.2;
}
.page-heading h1 span { color: #F39C12; }
.page-heading p {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #484F58; line-height: 1.7;
}

/* ── INPUT ── */
.lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; letter-spacing: 2px;
    text-transform: uppercase; color: #484F58; margin-bottom: 10px;
}
.stTextArea label { display: none !important; }
.stTextArea > div > div {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 8px !important;
}
.stTextArea > div > div:focus-within {
    border-color: rgba(243,156,18,0.4) !important;
    box-shadow: 0 0 0 3px rgba(243,156,18,0.05) !important;
}
.stTextArea > div > div > textarea {
    background: transparent !important; border: none !important;
    color: #C9D1D9 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important; line-height: 1.75 !important;
    padding: 18px !important; resize: none !important;
    caret-color: #F39C12 !important;
}
.stTextArea > div > div > textarea::placeholder { color: #30363D !important; }
.stTextArea > div > div > textarea:focus { box-shadow: none !important; outline: none !important; }

/* ── BUTTON ── */
.stButton > button {
    width: 100% !important; margin-top: 10px !important; height: 46px !important;
    background: rgba(243,156,18,0.08) !important;
    border: 1px solid rgba(243,156,18,0.35) !important;
    border-radius: 8px !important; color: #F39C12 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important; font-weight: 600 !important;
    letter-spacing: 2.5px !important; text-transform: uppercase !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(243,156,18,0.15) !important;
    border-color: #F39C12 !important;
}

/* ── DIVIDER ── */
.divider { height: 1px; background: #21262D; margin: 32px 0; }

/* ── STATS ── */
.stats { display: flex; gap: 10px; margin-bottom: 28px; }
.stat {
    flex: 1; background: #161B22;
    border: 1px solid #21262D; border-radius: 8px;
    padding: 14px 16px;
}
.stat-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 16px; font-weight: 600; color: #F39C12; display: block;
}
.stat-key {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px; letter-spacing: 1.5px;
    text-transform: uppercase; color: #484F58;
    margin-top: 4px; display: block;
}

/* ── SECTION LABEL ── */
.sec-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; letter-spacing: 2.5px;
    text-transform: uppercase; color: #484F58;
    margin: 28px 0 12px;
    display: flex; align-items: center; gap: 10px;
}
.sec-lbl::after { content: ''; flex: 1; height: 1px; background: #21262D; }
.sec-lbl.rrf-lbl { color: #F39C12; }
.sec-lbl.rrf-lbl::after { background: rgba(243,156,18,0.2); }

/* ── RRF DOMINANT BLOCK ── */
.rrf-block {
    background: linear-gradient(135deg, rgba(243,156,18,0.07) 0%, rgba(13,17,23,0.0) 60%);
    border: 1px solid rgba(243,156,18,0.25);
    border-radius: 10px; padding: 20px; margin-bottom: 16px;
}
.rrf-block-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; letter-spacing: 3px; text-transform: uppercase;
    color: #F39C12; margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
}
.rrf-block-title::after { content: ''; flex: 1; height: 1px; background: rgba(243,156,18,0.2); }

/* RRF rank rows */
.rrf-row {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 14px; border-radius: 6px; margin-bottom: 6px;
    background: #161B22; border: 1px solid #21262D;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8B949E;
    transition: border-color 0.15s;
}
.rrf-row:first-of-type { border-color: rgba(243,156,18,0.3); background: rgba(243,156,18,0.05); }
.rrf-num { font-size: 13px; font-weight: 700; color: #F39C12; min-width: 28px; }
.rrf-preview { flex: 1; color: #8B949E; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rrf-row:first-of-type .rrf-preview { color: #C9D1D9; }
.rrf-bm25  { color: #58A6FF; font-size: 10px; white-space: nowrap; }
.rrf-faiss { color: #3FB950; font-size: 10px; white-space: nowrap; }
.rrf-score { color: #F39C12; font-weight: 600; font-size: 10px; white-space: nowrap; }
.boost-tag {
    font-size: 9px; padding: 2px 8px; border-radius: 10px; white-space: nowrap;
}
.b-both  { background: rgba(243,156,18,0.15); color: #F39C12; border: 1px solid rgba(243,156,18,0.3); }
.b-bm25  { background: rgba(88,166,255,0.1);  color: #58A6FF; border: 1px solid rgba(88,166,255,0.2); }
.b-faiss { background: rgba(63,185,80,0.1);   color: #3FB950; border: 1px solid rgba(63,185,80,0.2); }

/* ── DIAGNOSIS BOX ── */
.diag-box {
    background: #161B22; border: 1px solid #21262D;
    border-radius: 8px; padding: 24px 26px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px; line-height: 1.9;
    color: #8B949E; white-space: pre-wrap; word-break: break-word;
}

/* ── EXPANDERS ── */
div[data-testid="stExpander"] {
    background: #161B22 !important; border: 1px solid #21262D !important;
    border-radius: 6px !important; margin-bottom: 6px !important;
}
div[data-testid="stExpander"] > details > summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important; color: #484F58 !important; padding: 12px 16px !important;
}
div[data-testid="stExpander"] > details > summary:hover { color: #8B949E !important; }

.stSpinner > div { border-top-color: #F39C12 !important; }
div[data-testid="stAlert"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important; border-radius: 6px !important;
}

/* dataframe dark theme */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Prompt ────────────────────────────────────────────────────────────────────
DIAGNOSIS_PROMPT = PromptTemplate(
    input_variables=["error", "context"],
    template="""You are an expert software debugger.

ERROR / STACK TRACE:
{error}

RETRIEVED CONTEXT (reranked by Cross-Encoder):
{context}

Respond in EXACTLY this format — no extra text:

🔴 ROOT CAUSE:
<one clear sentence>

📁 SUSPICIOUS LOCATION:
<file / function / line, or "Not determinable">

✅ FIX #1 [Confidence: HIGH/MEDIUM/LOW]:
<explanation and code snippet>

✅ FIX #2 [Confidence: HIGH/MEDIUM/LOW]:
<explanation and code snippet>

✅ FIX #3 [Confidence: HIGH/MEDIUM/LOW]:
<explanation and code snippet>

⚠️ RELATED ISSUES TO WATCH:
<brief note>

💻 CORRECTED CODE:
<Rewrite the ENTIRE function or class block where the error occurred. Include all imports at the top. Add a comment on every changed line. Code must be complete and runnable.>
"""
)

def format_context(docs):
    if not docs:
        return "No relevant context found."
    return "\n\n---\n\n".join(
        f"[{i}] {doc.metadata.get('source','?')}\n{doc.page_content.strip()}"
        for i, doc in enumerate(docs, 1)
    )

def run_analysis(error_input):
    data    = retrieve_with_rrf_details(error_input)
    context = format_context(data["rrf_results"])
    llm     = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=GROQ_API_KEY)
    chain   = DIAGNOSIS_PROMPT | llm | StrOutputParser()
    report  = chain.invoke({"error": error_input, "context": context})
    return report, data

def split_report(report):
    marker = "💻 CORRECTED CODE:"
    if marker in report:
        parts = report.split(marker, 1)
        return parts[0].strip(), parts[1].strip()
    return report.strip(), None

def short(text, n=60):
    t = text.strip().replace("\n", " ")
    return (t[:n] + "...") if len(t) > n else t

# ══════════════════════════════════════════════════════════════════════════════
# TOPBAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="topbar">
    <div class="brand">
        <div class="brand-dot"></div>
        <span class="brand-name">BUGIQ</span>
        <span class="rrf-badge">✦ CE</span>
    </div>
    <div class="pipe">
        <span class="pc pc-bm25">BM25</span>
        <span class="pc-arr">+</span>
        <span class="pc pc-faiss">FAISS</span>
        <span class="pc-arr">→</span>
        <span class="pc pc-rrf">✦ CROSS-ENCODER RERANKED</span>
        <span class="pc-arr">→</span>
        <span class="pc pc-llm">LLAMA 3.3</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-heading">
    <h1>Intelligent Bug Analyzer<br>powered by <span>Cross-Encoder Reranking</span></h1>
    <p>BM25 keyword search · FAISS semantic search · Cross-Encoder reranks pairs · Llama 3.3 diagnoses</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="lbl">Stack Trace / Error Input</div>', unsafe_allow_html=True)
error_input = st.text_area(
    "error", label_visibility="collapsed", height=200,
    placeholder="Traceback (most recent call last):\n  File \"app.py\", line 42, in get_user\n    return user['name']\nKeyError: 'name'"
)
clicked = st.button("✦  Analyze with Cross-Encoder", type="primary")

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if clicked:
    if not error_input.strip():
        st.warning("Paste a stack trace or error above first.")
    else:
        with st.spinner("BM25 retrieving · FAISS retrieving · Cross-Encoder reranking · Diagnosing..."):
            try:
                report, data    = run_analysis(error_input)
                diagnosis, fixed_code = split_report(report)
                rrf_table       = data["rrf_table"]
                rrf_docs        = data["rrf_results"]

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                # ── Stats ─────────────────────────────────────────────────────
                st.markdown(f"""
                <div class="stats">
                    <div class="stat">
                        <span class="stat-val">{len(rrf_docs)}</span>
                        <span class="stat-key">Chunks · CE Reranked</span>
                    </div>
                    <div class="stat">
                        <span class="stat-val">BM25 + FAISS</span>
                        <span class="stat-key">Retrieval Methods</span>
                    </div>
                    <div class="stat">
                        <span class="stat-val">Llama 3.3</span>
                        <span class="stat-key">Diagnosis Model</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── RRF Dominant Block ─────────────────────────────────────────
                import streamlit.components.v1 as components

                boost_cls = {"Both ⚡": "b-both", "BM25 only": "b-bm25", "FAISS only": "b-faiss"}
                rrf_rows = ""
                for r in rrf_table:
                    bc = boost_cls.get(r["boost"], "b-faiss")
                    preview = short(r['doc'].page_content).replace("'", "&#39;").replace('"', '&quot;')
                    rrf_rows += f"""
                    <div class="rrf-row {'rrf-top' if r['final_rank']==1 else ''}">
                        <span class="rrf-num">#{r['final_rank']}</span>
                        <span class="rrf-preview">{preview}</span>
                        <span class="rrf-bm25">BM25 {r['bm25_rank']}</span>
                        <span class="rrf-faiss">FAISS {r['faiss_rank']}</span>
                        <span class="rrf-score">CE: {r['rrf_score']}</span>
                        <span class="boost-tag {bc}">{r['boost']}</span>
                    </div>"""

                rrf_component = f"""
                <style>
                @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
                * {{ box-sizing: border-box; margin: 0; padding: 0; }}
                body {{ background: transparent; font-family: 'IBM Plex Mono', monospace; }}
                .rrf-wrap {{
                    background: linear-gradient(135deg, rgba(243,156,18,0.07) 0%, rgba(13,17,23,0.0) 60%);
                    border: 1px solid rgba(243,156,18,0.25);
                    border-radius: 10px; padding: 20px;
                }}
                .rrf-title {{
                    font-size: 9px; letter-spacing: 3px; text-transform: uppercase;
                    color: #F39C12; margin-bottom: 16px;
                    display: flex; align-items: center; gap: 8px;
                }}
                .rrf-title::after {{ content: ''; flex: 1; height: 1px; background: rgba(243,156,18,0.2); }}
                .rrf-row {{
                    display: flex; align-items: center; gap: 10px;
                    padding: 10px 14px; border-radius: 6px; margin-bottom: 6px;
                    background: #161B22; border: 1px solid #21262D;
                    font-size: 11px; color: #8B949E;
                }}
                .rrf-top {{ border-color: rgba(243,156,18,0.35) !important; background: rgba(243,156,18,0.06) !important; }}
                .rrf-num {{ font-size: 13px; font-weight: 700; color: #F39C12; min-width: 28px; }}
                .rrf-preview {{ flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #8B949E; }}
                .rrf-top .rrf-preview {{ color: #C9D1D9; }}
                .rrf-bm25  {{ color: #58A6FF; white-space: nowrap; font-size: 10px; }}
                .rrf-faiss {{ color: #3FB950; white-space: nowrap; font-size: 10px; }}
                .rrf-score {{ color: #F39C12; font-weight: 600; white-space: nowrap; font-size: 10px; }}
                .boost-tag {{ font-size: 9px; padding: 2px 8px; border-radius: 10px; white-space: nowrap; }}
                .b-both  {{ background: rgba(243,156,18,0.15); color: #F39C12; border: 1px solid rgba(243,156,18,0.3); }}
                .b-bm25  {{ background: rgba(88,166,255,0.1);  color: #58A6FF; border: 1px solid rgba(88,166,255,0.2); }}
                .b-faiss {{ background: rgba(63,185,80,0.1);   color: #3FB950; border: 1px solid rgba(63,185,80,0.2); }}
                </style>
                <div class="rrf-wrap">
                    <div class="rrf-title">✦ Cross-Encoder — Live Reranking</div>
                    {rrf_rows}
                </div>
                """
                components.html(rrf_component, height=360, scrolling=False)

                # ── Diagnosis ─────────────────────────────────────────────────
                st.markdown('<div class="sec-lbl">Diagnosis Report</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="diag-box">{diagnosis}</div>', unsafe_allow_html=True)

                # ── Corrected Code ────────────────────────────────────────────
                if fixed_code:
                    clean = re.sub(r'^```[a-zA-Z]*\n?', '', fixed_code.strip())
                    clean = re.sub(r'```$', '', clean.strip()).strip()
                    st.markdown('<div class="sec-lbl">Corrected Code</div>', unsafe_allow_html=True)
                    st.code(clean, language="python")

                # ── Raw Chunks ────────────────────────────────────────────────
                st.markdown(f'<div class="sec-lbl">Retrieved Context — {len(rrf_docs)} chunks after Cross-Encoder</div>', unsafe_allow_html=True)
                for i, doc in enumerate(rrf_docs, 1):
                    src = doc.metadata.get("source", "unknown")
                    with st.expander(f"#{i}  {src}"):
                        st.code(doc.page_content.strip(), language="text")

            except Exception as e:
                st.error(f"Error: {str(e)}")