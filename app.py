import streamlit as st
import os
import re
import random
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RFP Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body, .main { background-color: #0f1117; color: #e0e0e0; }
    .answer-box {
        background: #1e2130;
        border-left: 4px solid #4f8ef7;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 0.6rem 0 1rem 0;
        font-size: 0.97rem;
        line-height: 1.75;
        white-space: pre-wrap;
    }
    .summary-box {
        background: #1a2535;
        border-left: 4px solid #38d9a9;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 0.6rem 0 1rem 0;
        font-size: 0.95rem;
        line-height: 1.75;
    }
    .source-box {
        background: #181c27;
        border: 1px solid #2a2f45;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin: 0.35rem 0;
        font-size: 0.83rem;
        color: #a0a8c0;
        white-space: pre-wrap;
    }
    .qa-card {
        background: #1e2130;
        border: 1px solid #2a2f45;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.8rem;
    }
    .qa-q { color: #7ecfff; font-weight: 700; font-size: 0.96rem; margin-bottom: 0.4rem; }
    .qa-a { color: #c8d0e8; font-size: 0.92rem; line-height: 1.65; }
    .q-label {
        background: #252a3d; border-radius: 6px;
        padding: 0.45rem 1rem; font-size: 0.86rem;
        color: #7ecfff; font-style: italic;
        display: inline-block; margin-bottom: 0.5rem;
    }
    .step-badge {
        background: #2a2f45; border-radius: 20px;
        padding: 0.28rem 0.9rem; font-size: 0.78rem;
        color: #8892b0; margin-bottom: 0.5rem; display: inline-block;
    }
    .hist-card {
        background: #1e2130; border: 1px solid #2a2f45;
        border-radius: 8px; padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem; font-size: 0.82rem;
    }
    .hist-q { color: #7ecfff; font-weight: 600; margin-bottom: 0.25rem; }
    .hist-a { color: #b0bcd4; font-size: 0.80rem; }
    .src-badge {
        display: inline-block; background: #1f3a2a;
        border: 1px solid #38d9a9; border-radius: 20px;
        padding: 0.2rem 0.7rem; font-size: 0.75rem;
        color: #38d9a9; margin-bottom: 0.6rem;
    }
    h1, h2, h3 { color: #e8eaf6; }
    .stTextInput > div > div > input {
        background: #1e2130; border: 1px solid #2a2f45;
        color: #e0e0e0; border-radius: 8px;
    }
    .stButton > button { border-radius: 8px; font-weight: 600; padding: 0.45rem 1.2rem; }
    hr { border-color: #2a2f45; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "processed":        False,
    "current_q":        "",
    "current_answer":   None,
    "current_sources":  [],
    "current_summary":  None,
    "history":          [],
    "view_hist_idx":    None,
    "auto_qa":          None,
    "qa_seed":          42,
    "all_chunks":       [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def run_llm_query(question: str):
    """Fresh chain every call — zero answer bleed."""
    from bot import load_chain_approach1, query_chain
    chain  = load_chain_approach1()
    result = query_chain(chain, question, 1)
    answer  = (result.get("answer") or "").strip()
    sources = result.get("sources", [])
    if not answer:
        answer = "No relevant information found in the document for this question."
    return answer, sources


def groq_call(prompt: str, max_tokens: int = 300) -> str | None:
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        return None
    try:
        import requests as _r
        r = _r.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
            timeout=25,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY — Wikipedia REST API + DuckDuckGo Instant Answer as fallback
# Extracts MULTIPLE specific phrases from the answer, tries all of them,
# picks the best matching external content, condenses to 5-6 lines.
# ─────────────────────────────────────────────────────────────────────────────
def generate_summary(question: str, answer: str) -> str:
    import requests as _r

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        return "⚠️ Add GROQ_API_KEY to your .env to enable summaries."

    HEADERS = {"User-Agent": "RFP-Intelligence/1.0 (educational project)"}

    # ── Step 1: extract 3 specific search phrases FROM THE ANSWER TEXT ────────
    # We ask for multiple candidates so we can try each one and pick the best hit
    raw_topics = groq_call(
        f'Here is a document answer:\n"""\n{answer[:600]}\n"""\n\n'
        "List the 3 most specific technical or business concepts mentioned in this answer. "
        "Order them from most specific to most general. "
        "Reply with ONLY a numbered list, one phrase per line, 2-5 words each. "
        "No explanation. No punctuation at end of lines.\n"
        "Example format:\n1. IT infrastructure management\n2. Request for proposal\n3. Vendor selection",
        max_tokens=60,
    ) or ""

    # Parse out the phrases
    topic_candidates = []
    for line in raw_topics.strip().splitlines():
        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip().strip('"').strip("'")
        if 2 <= len(line.split()) <= 6 and line:
            topic_candidates.append(line)

    # Always add a fallback using first meaningful words of the answer
    answer_words = [w for w in re.sub(r"[^\w\s]", " ", answer).split() if len(w) > 4]
    if answer_words:
        topic_candidates.append(" ".join(answer_words[:4]))

    # Deduplicate
    seen_t = set()
    unique_candidates = []
    for t in topic_candidates:
        key = t.lower()
        if key not in seen_t:
            seen_t.add(key)
            unique_candidates.append(t)

    # ── Step 2 & 3: search Wikipedia then DuckDuckGo for EACH candidate ───────
    best_text  = ""
    best_url   = ""
    best_title = ""
    best_src   = ""

    def fetch_wikipedia(term: str):
        """Returns (text, url, title) or ('','','')."""
        slug = term.strip().replace(" ", "_")
        # Direct lookup
        try:
            r = _r.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
                timeout=7, headers=HEADERS,
            )
            if r.status_code == 200:
                d = r.json()
                ext = d.get("extract", "")
                if len(ext) > 120:
                    return (
                        ext[:1600],
                        d.get("content_urls", {}).get("desktop", {}).get("page", ""),
                        d.get("title", term),
                    )
        except Exception:
            pass
        # Wikipedia search fallback
        try:
            sr = _r.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search",
                        "srsearch": term, "srlimit": 1, "format": "json"},
                timeout=7, headers=HEADERS,
            )
            if sr.status_code == 200:
                hits = sr.json().get("query", {}).get("search", [])
                if hits:
                    slug2 = hits[0]["title"].replace(" ", "_")
                    r2 = _r.get(
                        f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug2}",
                        timeout=7, headers=HEADERS,
                    )
                    if r2.status_code == 200:
                        d = r2.json()
                        ext = d.get("extract", "")
                        if len(ext) > 120:
                            return (
                                ext[:1600],
                                d.get("content_urls", {}).get("desktop", {}).get("page", ""),
                                d.get("title", hits[0]["title"]),
                            )
        except Exception:
            pass
        return ("", "", "")

    def fetch_duckduckgo(term: str):
        """Returns (text, url, title) or ('','','')."""
        try:
            r = _r.get(
                "https://api.duckduckgo.com/",
                params={"q": term, "format": "json",
                        "no_html": "1", "skip_disambig": "1"},
                timeout=7, headers=HEADERS,
            )
            if r.status_code == 200:
                d = r.json()
                abstract = d.get("AbstractText", "")
                src_name = d.get("AbstractSource", "DuckDuckGo")
                if len(abstract) > 120:
                    return (
                        abstract[:1600],
                        d.get("AbstractURL", ""),
                        f"{d.get('Heading', term)} ({src_name})",
                    )
                related = d.get("RelatedTopics", [])
                texts = [t.get("Text", "") for t in related
                         if isinstance(t, dict) and len(t.get("Text", "")) > 80]
                if texts:
                    return (
                        " ".join(texts[:5])[:1400],
                        d.get("AbstractURL", ""),
                        f"{d.get('Heading', term)} (DuckDuckGo)",
                    )
        except Exception:
            pass
        return ("", "", "")

    # Try every candidate — Wikipedia first, then DuckDuckGo
    # Stop as soon as we find a good result (>200 chars)
    for candidate in unique_candidates:
        txt, url, title = fetch_wikipedia(candidate)
        if len(txt) > 200:
            best_text, best_url, best_title, best_src = txt, url, title, "Wikipedia"
            break
        txt, url, title = fetch_duckduckgo(candidate)
        if len(txt) > 200:
            best_text, best_url, best_title, best_src = txt, url, title, "DuckDuckGo"
            break

    # ── Step 4: condense the best found content into 5-6 lines with Groq ──────
    if best_text:
        condensed = groq_call(
            f'The document answer said:\n"""\n{answer[:500]}\n"""\n\n'
            f'External source ({best_src}) provides this background on "{best_title}":\n'
            f'"""\n{best_text[:1400]}\n"""\n\n'
            "Task: Write a 5-6 line plain paragraph that summarises the external source content "
            "in a way that is DIRECTLY relevant to the document answer above. "
            "Every sentence must connect to something mentioned in the document answer. "
            "Do NOT repeat or copy the document answer. No heading. No bullet points. "
            "Plain paragraph only. Output strictly 5-6 lines — no more, no less.",
            max_tokens=280,
        )
        if condensed:
            link = f"[{best_title}]({best_url})" if best_url else best_title
            return condensed + f"\n\n✅ *Source: {best_src} — {link}*"

    # ── Step 5: Groq knowledge fallback (clearly labelled) ────────────────────
    best_candidate = unique_candidates[0] if unique_candidates else question[:50]
    fallback = groq_call(
        f'The document answer said:\n"""\n{answer[:500]}\n"""\n\n'
        f'Write a 5-6 line factual encyclopedia-style paragraph about "{best_candidate}" '
        "that provides real-world background context directly relevant to the answer above. "
        "No heading. No bullets. Plain paragraph only. Strictly 5-6 lines.",
        max_tokens=280,
    )
    if fallback:
        return (
            fallback
            + "\n\n⚠️ *External source unavailable — summary from general knowledge.*"
        )

    return "⚠️ Could not generate summary. Check GROQ_API_KEY in .env."


# ─────────────────────────────────────────────────────────────────────────────
# AUTO Q&A
# ─────────────────────────────────────────────────────────────────────────────
def generate_auto_qa(sources: list, seed: int = 42, previous_pairs: list = None) -> list:
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        return [{"q": "⚠️ GROQ_API_KEY missing", "a": "Add GROQ_API_KEY to your .env file."}]
    if not sources:
        return []

    shuffled = sources[:]
    random.seed(seed)
    random.shuffle(shuffled)

    context = "\n\n".join(
        (s.get("text", "") if isinstance(s, dict) else str(s))[:500]
        for s in shuffled[:8]
    )

    avoid_str = ""
    if previous_pairs:
        prev_qs = "\n".join(f"- {p['q']}" for p in previous_pairs)
        avoid_str = (
            f"\n\nDo NOT ask any of these previously generated questions:\n{prev_qs}\n"
            "Generate 5 completely DIFFERENT questions not listed above."
        )

    try:
        import requests as _r
        prompt = (
            f"Based strictly on this document content:\n\n{context}"
            f"{avoid_str}\n\n"
            "Generate exactly 5 important questions a reader would want answered, "
            "and answer each using ONLY the content above. "
            "Cover different aspects: goals, requirements, timeline, contacts, technical details.\n\n"
            "Format EXACTLY like this — no extra text:\n"
            "Q1: <question>\nA1: <2-3 sentence answer>\n"
            "Q2: <question>\nA2: <2-3 sentence answer>\n"
            "Q3: <question>\nA3: <2-3 sentence answer>\n"
            "Q4: <question>\nA4: <2-3 sentence answer>\n"
            "Q5: <question>\nA5: <2-3 sentence answer>\n"
        )
        resp = _r.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.7,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return [{"q": f"API error {resp.status_code}", "a": resp.text[:200]}]

        text = resp.json()["choices"][0]["message"]["content"]
        qs   = re.findall(r"Q\d+:\s*(.+)", text)
        ans  = re.findall(r"A\d+:\s*(.+)", text)
        return [{"q": q.strip(), "a": a.strip()} for q, a in zip(qs, ans)]
    except Exception as e:
        return [{"q": "Error", "a": str(e)}]


def export_history_txt() -> str:
    lines = ["RFP INTELLIGENCE — CHAT HISTORY\n", "=" * 60 + "\n"]
    for i, item in enumerate(st.session_state.history, 1):
        lines.append(f"\n[{i}] Q: {item['q']}\n    A: {item['answer']}\n")
        if item.get("summary"):
            lines.append(f"\n    Summary:\n    {item['summary']}\n")
        lines.append("-" * 60 + "\n")
    return "".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📄 RFP Intelligence")
    st.markdown("---")

    st.markdown('<div class="step-badge">STEP 1 — Upload PDF</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    st.markdown('<div class="step-badge">STEP 2 — Process</div>', unsafe_allow_html=True)
    if st.button("⚙️ Process Document", use_container_width=True):
        if not uploaded_file:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Processing PDF…"):
                try:
                    tmp      = Path(tempfile.mkdtemp())
                    pdf_path = tmp / uploaded_file.name
                    pdf_path.write_bytes(uploaded_file.read())
                    import subprocess, sys
                    res = subprocess.run(
                        [sys.executable, "ingest.py", str(pdf_path)],
                        capture_output=True, text=True, cwd=os.getcwd(),
                    )
                    if res.returncode == 0:
                        st.session_state.processed       = True
                        st.session_state.current_q       = ""
                        st.session_state.current_answer  = None
                        st.session_state.current_sources = []
                        st.session_state.current_summary = None
                        st.session_state.auto_qa         = None
                        st.session_state.all_chunks      = []
                        st.session_state.qa_seed         = 42
                        st.success("✅ Document processed!")
                    else:
                        st.error(f"Ingest error:\n{res.stderr[-600:]}")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    if st.session_state.processed:
        st.success("✅ Ready to query")
    else:
        st.info("Upload & process a PDF to begin.")

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🕘 Chat History")

        col_exp, col_clr = st.columns(2)
        with col_exp:
            st.download_button(
                "📥 Export",
                data=export_history_txt(),
                file_name="rfp_chat_history.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_clr:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.history       = []
                st.session_state.view_hist_idx = None
                st.rerun()

        for idx, item in enumerate(reversed(st.session_state.history)):
            real_idx = len(st.session_state.history) - 1 - idx
            short_q  = item["q"][:50] + ("…" if len(item["q"]) > 50 else "")
            short_a  = item["answer"][:70] + ("…" if len(item["answer"]) > 70 else "")
            border   = "#4f8ef7" if st.session_state.view_hist_idx == real_idx else "#2a2f45"
            st.markdown(
                f'<div class="hist-card" style="border-color:{border};border-width:1px;border-style:solid;">'
                f'<div class="hist-q">Q: {short_q}</div>'
                f'<div class="hist-a">{short_a}</div></div>',
                unsafe_allow_html=True,
            )
            if st.button("👁 View", key=f"view_{real_idx}", use_container_width=True):
                st.session_state.view_hist_idx = real_idx
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY VIEW
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.view_hist_idx is not None:
    item = st.session_state.history[st.session_state.view_hist_idx]
    st.markdown("## 🕘 Viewing Past Answer")
    if st.button("← Back"):
        st.session_state.view_hist_idx = None
        st.rerun()

    st.markdown(f'<div class="q-label">💬 {item["q"]}</div>', unsafe_allow_html=True)
    st.markdown("### 🟢 Answer")
    st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)

    if item.get("sources"):
        with st.expander("📚 Source Chunks", expanded=False):
            for i, src in enumerate(item["sources"], 1):
                text = src.get("text", str(src)) if isinstance(src, dict) else str(src)
                page = src.get("page", "?") if isinstance(src, dict) else "?"
                st.markdown(
                    f'<div class="source-box"><strong>Chunk {i} · Page {page}</strong><br>{text}</div>',
                    unsafe_allow_html=True,
                )

    if item.get("summary"):
        st.markdown("### 📋 External Summary")
        st.markdown(f'<div class="summary-box">🌐 {item["summary"]}</div>', unsafe_allow_html=True)

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_ask, tab_autoqa = st.tabs([
    "🔍 Ask a Question",
    "🤖 Auto Q&A Generator",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ask a Question
# ══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    st.markdown("### 🔍 Ask a Question About Your Document")
    st.markdown(
        '<div class="step-badge">STEP 3 — Type your question and click Get Answer</div>',
        unsafe_allow_html=True,
    )

    col_q, col_btn = st.columns([4, 1])
    with col_q:
        question = st.text_input(
            "", placeholder="e.g. What are the goals of this RFP?",
            label_visibility="collapsed", key="question_input",
        )
    with col_btn:
        ask_btn = st.button(
            "🟢 Get Answer", use_container_width=True,
            disabled=not st.session_state.processed,
        )

    if ask_btn:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            st.session_state.current_answer  = None
            st.session_state.current_sources = []
            st.session_state.current_summary = None
            st.session_state.current_q       = question.strip()
            st.session_state.view_hist_idx   = None

            with st.spinner("Fetching answer from your document…"):
                try:
                    answer, sources = run_llm_query(question.strip())
                    st.session_state.current_answer  = answer
                    st.session_state.current_sources = sources
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            st.session_state.history.append({
                "q":       st.session_state.current_q,
                "answer":  st.session_state.current_answer,
                "sources": st.session_state.current_sources,
                "summary": None,
            })

    if st.session_state.current_answer:
        st.markdown(
            f'<div class="q-label">💬 Answering: {st.session_state.current_q}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("### 🟢 LLM Answer")
        st.markdown(
            f'<div class="answer-box">{st.session_state.current_answer}</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.current_sources:
            with st.expander("📚 Source Chunks from PDF", expanded=False):
                for i, src in enumerate(st.session_state.current_sources, 1):
                    text = src.get("text", str(src)) if isinstance(src, dict) else str(src)
                    page = src.get("page", "?") if isinstance(src, dict) else "?"
                    st.markdown(
                        f'<div class="source-box"><strong>Chunk {i} · Page {page}</strong><br>{text}</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("---")
        st.markdown(
            '<div class="step-badge">STEP 4 — External source summary (5-6 lines)</div>',
            unsafe_allow_html=True,
        )
        st.caption("Fetches real content from Wikipedia or DuckDuckGo — no hallucination.")

        if st.button("📋 Generate External Summary"):
            with st.spinner("Fetching from Wikipedia / DuckDuckGo…"):
                summary = generate_summary(
                    st.session_state.current_q,
                    st.session_state.current_answer,
                )
                st.session_state.current_summary = summary
                if st.session_state.history:
                    st.session_state.history[-1]["summary"] = summary

        if st.session_state.current_summary:
            st.markdown("### 📋 External Summary")
            st.markdown(
                f'<div class="summary-box">'
                f'🌐 <strong>External context for your answer:</strong>'
                f'<br><br>{st.session_state.current_summary}'
                f'</div>',
                unsafe_allow_html=True,
            )

    elif not st.session_state.processed:
        st.info("👈 Upload and process your PDF from the sidebar to get started.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Auto Q&A Generator
# ══════════════════════════════════════════════════════════════════════════════
with tab_autoqa:
    st.markdown("### 🤖 Auto Q&A Generator")
    st.markdown(
        "Generates **5 questions** a reader would ask about this document "
        "— with answers from the PDF. Each **Regenerate** gives completely different questions."
    )

    if not st.session_state.processed:
        st.info("👈 Process a PDF first to use this feature.")
    else:
        if st.button("✨ Generate Top 5 Q&As from PDF", use_container_width=False):
            with st.spinner("Reading document and generating Q&As…"):
                try:
                    from bot import load_chain_approach1, query_chain
                    chain   = load_chain_approach1()
                    result  = query_chain(chain, "overview goals requirements deadlines contacts", 1)
                    sources = result.get("sources", [])
                    st.session_state.all_chunks = sources
                    st.session_state.qa_seed    = 42
                    st.session_state.auto_qa    = generate_auto_qa(sources, seed=42)
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.auto_qa is not None:
            pairs = st.session_state.auto_qa
            if not pairs:
                st.warning("Could not generate Q&As. Check GROQ_API_KEY in .env.")
            else:
                st.success(f"✅ Generated {len(pairs)} Q&A pairs")
                for i, p in enumerate(pairs, 1):
                    st.markdown(
                        f'<div class="qa-card">'
                        f'<div class="qa-q">Q{i}: {p["q"]}</div>'
                        f'<div class="qa-a">{p["a"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("---")
                col_save, col_regen = st.columns(2)

                with col_save:
                    if st.button("💾 Save All to Chat History", use_container_width=True):
                        for p in pairs:
                            st.session_state.history.append({
                                "q": p["q"], "answer": p["a"],
                                "sources": [], "summary": None,
                            })
                        st.success("✅ Saved to chat history!")

                with col_regen:
                    if st.button("🔄 Regenerate Q&As", use_container_width=True):
                        with st.spinner("Generating different questions…"):
                            try:
                                st.session_state.qa_seed += random.randint(10, 99)
                                sources = st.session_state.all_chunks
                                if not sources:
                                    from bot import load_chain_approach1, query_chain
                                    chain   = load_chain_approach1()
                                    result  = query_chain(
                                        chain,
                                        "overview goals requirements deadlines contacts", 1,
                                    )
                                    sources = result.get("sources", [])
                                    st.session_state.all_chunks = sources

                                st.session_state.auto_qa = generate_auto_qa(
                                    sources,
                                    seed=st.session_state.qa_seed,
                                    previous_pairs=st.session_state.auto_qa,
                                )
                                st.rerun()
                            except Exception as e:
                                st.error(f"Regenerate error: {e}")