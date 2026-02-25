import streamlit as st
import requests

API = "http://localhost:8000/api"

st.set_page_config(page_title="RAG Explorer v2", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0a0a0f; color: #e8e8f0; }
    section[data-testid="stSidebar"] { background-color: #111118; border-right: 1px solid #2a2a3a; }

    .stat-box {
        background: #1a1a24; border: 1px solid #2a2a3a;
        border-radius: 10px; padding: 14px; text-align: center; margin-bottom: 8px;
    }
    .stat-box small { color: #888; font-size: 0.72rem; display: block; margin-bottom: 4px; }

    .info-card {
        background: #111118; border: 1px solid #2a2a3a;
        border-radius: 10px; padding: 16px 20px; margin-bottom: 10px;
    }
    .info-label { font-size: 0.72rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
    .info-value { font-size: 0.9rem; color: #ccc; line-height: 1.7; }

    .chunk-card {
        background: #111118; border-left: 3px solid #6366f1;
        border-radius: 0 10px 10px 0; padding: 16px 20px;
        margin-bottom: 10px; line-height: 1.75; font-size: 0.88rem; color: #ccc;
    }
    .chunk-meta { font-size: 0.73rem; color: #555; margin-bottom: 8px; }

    .answer-wrapper {
        background: #111118; border: 1px solid #2a2a3a;
        border-radius: 12px; padding: 26px 28px; margin-bottom: 16px;
    }
    .answer-header { font-size: 1.1rem; font-weight: 700; margin-bottom: 4px; }
    .answer-format { font-size: 0.78rem; color: #888; margin-bottom: 16px; padding-bottom: 14px; border-bottom: 1px solid #1e1e2e; }
    .answer-text { font-size: 0.93rem; color: #e8e8f0; line-height: 1.95; }

    .confidence-bar-bg {
        background: #1a1a24; border-radius: 999px; height: 10px;
        width: 100%; margin: 8px 0 4px 0; overflow: hidden;
    }
    .confidence-bar-fill {
        height: 10px; border-radius: 999px; transition: width 0.4s ease;
    }
    .confidence-card {
        background: #111118; border: 1px solid #2a2a3a;
        border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
    }
    .confidence-title { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px; }
    .confidence-score { font-size: 2rem; font-weight: 800; line-height: 1; }
    .confidence-label { font-size: 0.85rem; font-weight: 600; margin-top: 4px; }
    .confidence-explain { font-size: 0.8rem; color: #888; margin-top: 8px; line-height: 1.5; }

    .doc-tag {
        display: inline-block; background: #1a1a24; border: 1px solid #2a2a3a;
        border-radius: 999px; padding: 4px 12px; font-size: 0.75rem;
        color: #aaa; margin: 3px;
    }
    .doc-active-tag {
        display: inline-block; border-radius: 999px; padding: 4px 12px;
        font-size: 0.75rem; font-weight: 600; margin: 3px; color: white;
    }
    .source-badge {
        display: inline-block; font-size: 0.7rem; font-weight: 600;
        padding: 2px 8px; border-radius: 999px; margin-bottom: 6px;
    }

    h1, h2, h3, h4 { color: #e8e8f0 !important; }
    .stButton > button {
        background: #6366f1 !important; color: white !important;
        border: none !important; border-radius: 8px !important; font-weight: 600 !important;
    }
    .stButton > button:hover { background: #5457e8 !important; }
    .stTabs [data-baseweb="tab"] { color: #888; font-size: 1rem; }
    .stTabs [aria-selected="true"] { color: #6366f1 !important; border-bottom-color: #6366f1 !important; }
    .stSelectbox > div > div {
        background: #1a1a24 !important; border: 1px solid #2a2a3a !important;
        color: #e8e8f0 !important; border-radius: 8px !important;
    }
    .stTextInput > div > input {
        background: #1a1a24 !important; border: 1px solid #2a2a3a !important;
        color: #e8e8f0 !important; border-radius: 8px !important;
    }
    div[data-testid="stExpander"] {
        background: #111118 !important; border: 1px solid #2a2a3a !important; border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_COLORS = {
    "fixed_size": "#6366f1", "sentence_based": "#10b981",
    "semantic": "#f59e0b", "document_structure": "#ef4444",
}
PROMPT_COLORS = {
    "zero_shot": "#8b5cf6", "few_shot": "#0ea5e9",
    "chain_of_thought": "#f97316", "role_based": "#ec4899",
}
CHUNK_ORDER  = ["fixed_size", "sentence_based", "semantic", "document_structure"]
PROMPT_ORDER = ["zero_shot", "few_shot", "chain_of_thought", "role_based"]

CHUNK_LABELS = {
    "fixed_size": "⊞ Fixed-Size Chunking",
    "sentence_based": "≡ Sentence-Based Chunking",
    "semantic": "◈ Semantic Chunking",
    "document_structure": "⊡ Document-Structure Chunking",
}
PROMPT_LABELS = {
    "zero_shot": "⚡ Zero-Shot",
    "few_shot": "🎯 Few-Shot",
    "chain_of_thought": "🧠 Chain-of-Thought",
    "role_based": "🎭 Role-Based",
}

CHUNK_INFO = {
    "fixed_size":         {"what": "Splits into equal-sized token pieces, ignoring sentence or topic boundaries.", "pros": "Simple, fast, predictable.", "cons": "May cut sentences mid-way.", "best": "Raw data, logs, code files."},
    "sentence_based":     {"what": "Groups full sentences together, keeping each sentence intact.", "pros": "No broken sentences, preserves meaning.", "cons": "Chunk sizes vary widely.", "best": "Articles, reports, general documents."},
    "semantic":           {"what": "Uses AI embeddings to detect topic changes and splits at those boundaries.", "pros": "Each chunk stays on one topic.", "cons": "Slower — needs embedding model.", "best": "Mixed-topic documents, research papers."},
    "document_structure": {"what": "Splits at headings, paragraphs and sections — respects the document's layout.", "pros": "Respects author's intended structure.", "cons": "Only works on structured documents.", "best": "Reports, manuals, structured PDFs."},
}

PROMPT_INFO = {
    "zero_shot":        {"what": "Sends your question directly with no examples.", "output": "Short 2-3 sentence plain prose answer.", "pros": "Fast, low cost.", "cons": "Less format control.", "best": "Simple factual questions."},
    "few_shot":         {"what": "Gives 2 example Q&As before your question so the AI learns the format.", "output": "Structured bullet-point list with 4-6 points.", "pros": "Consistent structured output.", "cons": "Uses more tokens.", "best": "When you want organized bullet answers."},
    "chain_of_thought": {"what": "Forces the AI to reason step-by-step before answering.", "output": "Step 1 → Step 2 → Step 3 → Final Answer.", "pros": "Most accurate, shows reasoning.", "cons": "Longer output.", "best": "Complex multi-part questions."},
    "role_based":       {"what": "Assigns the AI an expert identity before answering.", "output": "Expert Overview → Key Details → Recommendation.", "pros": "Authoritative, domain-specific.", "cons": "May over-generalize.", "best": "Technical, medical, legal documents."},
}

SUGGESTED = [
    "What is the main topic of this document?",
    "Summarize the key points in simple terms.",
    "What are the most important conclusions?",
    "What problems does this document address?",
]

DOC_COLORS = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#ec4899", "#0ea5e9"]

# ── Session State ─────────────────────────────────────────────────────────────
for k, v in {
    "uploaded_docs":  [],           # list of {filename, collection_name, data}
    "active_doc_idx": 0,            # which doc is active in chunking tab
    "all_rag_results": None,
    "llm_configured": False,
    "current_q": "",
    "trigger_rag": False,
    "expert_role": "domain expert",
    "selected_prompt_strategy": "zero_shot",
}.items():
    if k not in st.session_state: st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
def check_status():
    try:
        r = requests.get(f"{API}/rag/status", timeout=3)
        st.session_state.llm_configured = r.json().get("llm_configured", False)
        return True
    except: return False

def call_all_rag(question, collection_names, role):
    combined = ",".join(collection_names)
    try:
        r = requests.post(
            f"{API}/rag/query",
            json={"question": question, "collection_name": combined,
                  "prompting_strategy": "zero_shot", "role": role,
                  "top_k": 3, "run_all_strategies": True},
            timeout=180
        )
        if r.status_code == 200:
            st.session_state.all_rag_results = r.json()
        else:
            st.error(f"Error: {r.json().get('detail', 'Unknown')}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Backend not reachable. Run: `python main.py`")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def upload_doc(file):
    """Upload a file to backend and return the response data."""
    r = requests.post(
        f"{API}/chunking/analyze",
        files={"file": (file.name, file.getvalue(), "application/octet-stream")},
        data={"chunk_size": 500, "auto_index": True},
        timeout=120
    )
    if r.status_code == 200:
        return r.json()
    else:
        raise Exception(r.json().get("detail", "Unknown error"))

def render_confidence(conf):
    """Render a visual confidence score card."""
    score = conf.get("score", 0)
    label = conf.get("label", "Unknown")
    color = conf.get("color", "#888")
    explanation = conf.get("explanation", "")
    top   = conf.get("top_score", 0)
    avg   = conf.get("avg_score", 0)

    st.markdown(
        f"""<div class='confidence-card'>
        <div class='confidence-title'>🎯 Answer Confidence Score</div>
        <div style='display:flex;align-items:flex-end;gap:12px'>
            <div class='confidence-score' style='color:{color}'>{score}%</div>
            <div class='confidence-label' style='color:{color};padding-bottom:4px'>{label} Confidence</div>
        </div>
        <div class='confidence-bar-bg'>
            <div class='confidence-bar-fill' style='width:{score}%;background:{color}'></div>
        </div>
        <div class='confidence-explain'>{explanation}</div>
        <div style='display:flex;gap:16px;margin-top:10px'>
            <span style='font-size:0.75rem;color:#666'>Best chunk match: <b style='color:{color}'>{top}%</b></span>
            <span style='font-size:0.75rem;color:#666'>Avg chunk match: <b style='color:#aaa'>{avg}%</b></span>
        </div>
        </div>""",
        unsafe_allow_html=True
    )

backend_online = check_status()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ◈ RAG Explorer v2")
    st.markdown("---")
    if backend_online: st.success("✅ Backend Online")
    else: st.error("❌ Backend Offline\nRun: `python main.py`")
    if st.session_state.llm_configured: st.success("✅ Groq Connected")
    else: st.warning("⚠️ Groq Not Configured")

    st.markdown("---")

    # Show uploaded docs in sidebar
    if st.session_state.uploaded_docs:
        st.markdown(f"### 📂 Uploaded Documents ({len(st.session_state.uploaded_docs)})")
        for i, doc in enumerate(st.session_state.uploaded_docs):
            color = DOC_COLORS[i % len(DOC_COLORS)]
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"
                f"<span style='color:{color};font-size:1.1rem'>●</span>"
                f"<span style='font-size:0.82rem;color:#ccc'>{doc['filename']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown("---")
        if st.button("↺ Clear All & Upload New", use_container_width=True):
            st.session_state.uploaded_docs    = []
            st.session_state.all_rag_results  = None
            st.session_state.current_q        = ""
            st.session_state.trigger_rag      = False
            st.session_state.active_doc_idx   = 0
            st.rerun()
    else:
        st.markdown("**📦 Chunking** — how docs are split")
        st.markdown("**💬 Prompting** — compare 4 AI answers")
        st.markdown("**🎯 Confidence** — how well AI can answer")
        st.markdown("**📂 Multi-Doc** — upload multiple files")

# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD PAGE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.uploaded_docs:
    st.markdown(
        "<h1 style='text-align:center'>Upload. Chunk. Query.<br>"
        "<span style='color:#6366f1'>Compare Everything.</span></h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;color:#888;font-size:1rem'>"
        "Upload one or more documents. Explore chunking. Ask questions and compare 4 AI answers with confidence scores.</p>",
        unsafe_allow_html=True
    )
    st.markdown("")

    # Feature highlights
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.markdown("<div class='stat-box'><small>NEW</small><b style='color:#6366f1'>📂 Multi-Doc</b><small style='color:#666'>Upload multiple files</small></div>", unsafe_allow_html=True)
    with fc2:
        st.markdown("<div class='stat-box'><small>NEW</small><b style='color:#10b981'>🎯 Confidence</b><small style='color:#666'>Know how reliable answers are</small></div>", unsafe_allow_html=True)
    with fc3:
        st.markdown("<div class='stat-box'><small></small><b style='color:#f59e0b'>📦 4 Chunking</b><small style='color:#666'>Compare split strategies</small></div>", unsafe_allow_html=True)
    with fc4:
        st.markdown("<div class='stat-box'><small></small><b style='color:#ec4899'>💬 4 Prompting</b><small style='color:#666'>Different answer styles</small></div>", unsafe_allow_html=True)

    st.markdown("")
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("#### Upload Your Document(s)")
        st.caption("You can upload multiple files — questions will be answered across all of them.")

        uploaded_files = st.file_uploader(
            "Drop documents here (PDF, DOCX, TXT, and more)",
            type=None,
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button(f"▶  Process {len(uploaded_files)} Document{'s' if len(uploaded_files) > 1 else ''}", use_container_width=True):
                all_docs = []
                errors   = []
                prog = st.progress(0, text="Processing documents...")

                for i, f in enumerate(uploaded_files):
                    prog.progress((i) / len(uploaded_files), text=f"Processing: {f.name}...")
                    try:
                        data = upload_doc(f)
                        all_docs.append({
                            "filename":        f.name,
                            "collection_name": data["collection_name"],
                            "data":            data,
                        })
                    except Exception as e:
                        errors.append(f"{f.name}: {str(e)}")

                prog.progress(1.0, text="Done!")

                if all_docs:
                    st.session_state.uploaded_docs = all_docs
                    if errors:
                        for e in errors:
                            st.warning(f"⚠️ {e}")
                    st.success(f"✅ {len(all_docs)} document{'s' if len(all_docs) > 1 else ''} ready!")
                    st.rerun()
                else:
                    for e in errors:
                        st.error(f"❌ {e}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
else:
    docs = st.session_state.uploaded_docs
    is_multi = len(docs) > 1

    # ── Top bar ───────────────────────────────────────────────────────────────
    if is_multi:
        st.markdown(f"**📂 {len(docs)} Documents loaded** — questions will search across all of them")
        doc_tags = ""
        for i, doc in enumerate(docs):
            color = DOC_COLORS[i % len(DOC_COLORS)]
            doc_tags += f"<span class='doc-active-tag' style='background:{color}20;border:1px solid {color};color:{color}'>● {doc['filename']}</span>"
        st.markdown(doc_tags, unsafe_allow_html=True)

        # Add another doc
        with st.expander("➕ Add another document"):
            extra_file = st.file_uploader("Upload another document", type=None, key="extra_upload")
            if extra_file:
                if st.button("Add Document", key="add_doc_btn"):
                    with st.spinner(f"Processing {extra_file.name}..."):
                        try:
                            data = upload_doc(extra_file)
                            st.session_state.uploaded_docs.append({
                                "filename": extra_file.name,
                                "collection_name": data["collection_name"],
                                "data": data,
                            })
                            st.session_state.all_rag_results = None
                            st.success(f"✅ {extra_file.name} added!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ {str(e)}")
    else:
        data = docs[0]["data"]
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1:
            st.markdown(f"**📄 {docs[0]['filename']}**")
            st.caption(f"{data['total_words']:,} words · {data['total_chars']:,} characters · {data.get('file_format','').upper()}")
        with c2:
            clr = "#10b981" if data.get("indexed") else "#888"
            st.markdown(f"<div class='stat-box'><small>ChromaDB</small><b style='color:{clr}'>{'Indexed ✓' if data.get('indexed') else 'Not Indexed'}</b></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='stat-box'><small>Semantic Chunks</small><b style='color:#f59e0b'>{data['strategies']['semantic']['chunk_count']}</b></div>", unsafe_allow_html=True)
        with c4:
            if st.button("➕ Add Doc", use_container_width=True):
                st.session_state._show_add = True

        if st.session_state.get("_show_add"):
            extra_file = st.file_uploader("Upload another document", type=None, key="extra_upload")
            if extra_file:
                with st.spinner(f"Processing {extra_file.name}..."):
                    try:
                        extra_data = upload_doc(extra_file)
                        st.session_state.uploaded_docs.append({
                            "filename": extra_file.name,
                            "collection_name": extra_data["collection_name"],
                            "data": extra_data,
                        })
                        st.session_state._show_add = False
                        st.session_state.all_rag_results = None
                        st.success(f"✅ {extra_file.name} added!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

    st.markdown("---")
    tab1, tab2 = st.tabs(["📦  Chunking Strategies", "💬  Prompting + AI Answers"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — CHUNKING
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### Chunking Strategies")
        st.markdown("Chunking splits your document into smaller pieces for the AI to read. Each strategy does this differently.")
        st.markdown("")

        # If multi-doc, pick which document to explore
        if is_multi:
            doc_names = [d["filename"] for d in docs]
            active_idx = st.selectbox(
                "🔽 Select document to explore:",
                options=range(len(docs)),
                format_func=lambda i: doc_names[i],
                key="active_doc_select"
            )
            st.session_state.active_doc_idx = active_idx
            active_data = docs[active_idx]["data"]
            st.markdown("")
        else:
            active_data = docs[0]["data"]

        # Chunking strategy dropdown
        selected_chunk = st.selectbox(
            "🔽 Select a chunking strategy to explore:",
            options=CHUNK_ORDER,
            format_func=lambda x: CHUNK_LABELS[x],
            key="chunk_select"
        )

        cur   = active_data["strategies"][selected_chunk]
        info  = CHUNK_INFO[selected_chunk]
        color = CHUNK_COLORS[selected_chunk]
        st.markdown("")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"<div class='info-card'><div class='info-label'>What this strategy does</div><div class='info-value'>{info['what']}</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='stat-box'><small>Chunks created</small><b style='color:{color};font-size:1.8rem'>{cur['chunk_count']}</b><small style='color:#555'>avg {cur['avg_tokens']} tokens each</small></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='info-card'><div class='info-label'>✅ Pros</div><div class='info-value'>{info['pros']}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='info-card'><div class='info-label'>❌ Cons</div><div class='info-value'>{info['cons']}</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='info-card'><div class='info-label'>📌 Best For</div><div class='info-value'>{info['best']}</div></div>", unsafe_allow_html=True)

        # Compare all 4
        st.markdown("---")
        st.markdown("#### All 4 Strategies at a Glance")
        compare_cols = st.columns(4)
        for i, key in enumerate(CHUNK_ORDER):
            c  = active_data["strategies"][key]
            clr = CHUNK_COLORS[key]
            marker = " ◀" if key == selected_chunk else ""
            with compare_cols[i]:
                st.markdown(
                    f"<div class='stat-box' style='border-color:{'#333' if key != selected_chunk else clr}'>"
                    f"<small style='color:{clr}'>{CHUNK_LABELS[key].split(' ',1)[1]}{marker}</small>"
                    f"<b style='color:{clr};font-size:1.5rem'>{c['chunk_count']}</b>"
                    f"<small style='color:#555'>chunks · avg {c['avg_tokens']} tokens</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # Chunk viewer
        st.markdown("---")
        st.markdown(f"#### View Chunks — {CHUNK_LABELS[selected_chunk]}")
        chunks = cur["chunks"]
        chunk_labels_dd = [f"Chunk #{ch['id']+1}  —  {ch['token_count']} tokens  —  \"{ch['text'][:55]}...\"" for ch in chunks]

        sel_idx = st.selectbox(
            "🔽 Select a chunk to read:",
            options=range(len(chunks)),
            format_func=lambda i: chunk_labels_dd[i],
            key="chunk_detail"
        )
        sel_ch = chunks[sel_idx]
        st.markdown(
            f"<div class='chunk-card' style='border-left-color:{color}'>"
            f"<div class='chunk-meta'>Chunk #{sel_ch['id']+1} &nbsp;·&nbsp; {sel_ch['token_count']} tokens &nbsp;·&nbsp; {len(sel_ch['text'])} characters</div>"
            f"{sel_ch['text']}"
            f"</div>",
            unsafe_allow_html=True
        )

        with st.expander(f"📋 View all {len(chunks)} chunks at once"):
            for ch in chunks:
                st.markdown(
                    f"<div class='chunk-card' style='border-left-color:{color}'>"
                    f"<div class='chunk-meta'>Chunk #{ch['id']+1} · {ch['token_count']} tokens</div>"
                    f"{ch['text']}"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — PROMPTING + AI ANSWERS
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### Prompting Strategies + AI Answers")
        if is_multi:
            sources_str = " + ".join([f"**{d['filename']}**" for d in docs])
            st.markdown(f"Searching across: {sources_str}")
        st.markdown("Ask a question — all 4 strategies answer it. Select the dropdown to compare answers.")
        st.markdown("")

        if not st.session_state.llm_configured:
            st.error("🔑 Groq API Key not configured. Add `GROQ_API_KEY=gsk_...` to `/backend/.env` and restart backend.")
        else:
            # Suggested questions
            st.markdown("**💡 Click a suggested question or type your own:**")
            rq_cols = st.columns(4)
            for i, q in enumerate(SUGGESTED):
                with rq_cols[i]:
                    if st.button(q, key=f"rq_{i}", use_container_width=True):
                        st.session_state.current_q       = q
                        st.session_state.trigger_rag     = True
                        st.session_state.all_rag_results = None
                        st.rerun()

            st.markdown("")
            col1, col2 = st.columns([3, 1])
            with col1:
                typed_q = st.text_input(
                    "Or type your own question:",
                    value=st.session_state.current_q,
                    placeholder="e.g. What causes air pollution?",
                    key="rag_typed"
                )
            with col2:
                role = st.text_input("Expert role (Role-Based):", value=st.session_state.expert_role)
                st.session_state.expert_role = role

            st.markdown("")
            btn_label = f"◈  Get Answers from All 4 Strategies {'(across all docs)' if is_multi else ''}"
            if st.button(btn_label, use_container_width=True):
                if typed_q.strip():
                    st.session_state.current_q       = typed_q
                    st.session_state.trigger_rag     = True
                    st.session_state.all_rag_results = None
                else:
                    st.warning("Please enter or select a question first.")

            # Auto-trigger
            if st.session_state.trigger_rag and st.session_state.current_q.strip():
                st.session_state.trigger_rag = False
                collection_names = [d["collection_name"] for d in docs]
                with st.spinner("Retrieving relevant chunks → Running 4 prompting strategies → Getting answers..."):
                    call_all_rag(st.session_state.current_q, collection_names, st.session_state.expert_role)

            # ── Results ───────────────────────────────────────────────────────
            if st.session_state.all_rag_results:
                result = st.session_state.all_rag_results
                st.markdown("---")
                st.markdown(f"**Question:** *\"{result['question']}\"*")
                st.markdown("")

                # ── Confidence Score ──────────────────────────────────────────
                conf = result.get("confidence", {})
                if conf:
                    render_confidence(conf)

                # ── Multi-doc source indicator ────────────────────────────────
                if result.get("multi_doc") and is_multi:
                    chunk_sources = {}
                    for ch in result.get("retrieved_chunks", []):
                        src = ch.get("source_collection", "")
                        chunk_sources[src] = chunk_sources.get(src, 0) + 1

                    st.markdown("**📂 Chunks pulled from:**")
                    src_html = ""
                    for i, doc in enumerate(docs):
                        col_name = doc["collection_name"]
                        count    = chunk_sources.get(col_name, 0)
                        color    = DOC_COLORS[i % len(DOC_COLORS)]
                        if count > 0:
                            src_html += f"<span class='doc-active-tag' style='background:{color}20;border:1px solid {color};color:{color}'>● {doc['filename']} ({count} chunk{'s' if count>1 else ''})</span>"
                        else:
                            src_html += f"<span class='doc-tag'>{doc['filename']} (0 chunks)</span>"
                    st.markdown(src_html, unsafe_allow_html=True)
                    st.markdown("")

                # ── Strategy dropdown ─────────────────────────────────────────
                st.markdown("#### Select a Prompting Strategy to View its Answer")

                def strategy_option_label(key):
                    r = result["results"].get(key, {})
                    meta = r.get("meta", {})
                    icon = meta.get("icon", "")
                    name = meta.get("name", PROMPT_LABELS[key])
                    fmt  = PROMPT_INFO[key]["output"]
                    return f"{icon} {name}  —  {fmt}"

                selected_ps = st.selectbox(
                    "🔽 Select strategy:",
                    options=PROMPT_ORDER,
                    format_func=strategy_option_label,
                    index=PROMPT_ORDER.index(st.session_state.selected_prompt_strategy),
                    key="prompt_view_select"
                )
                st.session_state.selected_prompt_strategy = selected_ps

                info  = PROMPT_INFO[selected_ps]
                color = PROMPT_COLORS[selected_ps]

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"<div class='info-card'><div class='info-label'>How this works</div><div class='info-value'>{info['what']}</div></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='info-card'><div class='info-label'>📋 Output Format</div><div class='info-value' style='color:#f59e0b'>{info['output']}</div></div>", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"<div class='info-card'><div class='info-label'>📌 Best For</div><div class='info-value'>{info['best']}</div></div>", unsafe_allow_html=True)

                # Selected answer
                r     = result["results"].get(selected_ps, {})
                meta  = r.get("meta", {})
                icon  = meta.get("icon", "")
                name  = meta.get("name", PROMPT_LABELS[selected_ps])
                tokens = r.get("tokens_used", "—")

                if "error" in r:
                    st.error(f"Error: {r['error']}")
                elif "answer" in r:
                    answer_html = r["answer"].replace("\n", "<br>")
                    st.markdown(
                        f"<div class='answer-wrapper' style='border-left:4px solid {color}'>"
                        f"<div class='answer-header' style='color:{color}'>{icon} {name} — Answer</div>"
                        f"<div class='answer-format'>Format: {info['output']} &nbsp;·&nbsp; {tokens} tokens</div>"
                        f"<div class='answer-text'>{answer_html}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                # Compare all 4 collapsed
                with st.expander("👁 Compare all 4 answers at once"):
                    for key in PROMPT_ORDER:
                        r2   = result["results"].get(key, {})
                        c2   = PROMPT_COLORS[key]
                        meta2 = r2.get("meta", {})
                        icon2 = meta2.get("icon", "")
                        name2 = meta2.get("name", PROMPT_LABELS[key])
                        if "answer" in r2:
                            ans_html = r2["answer"].replace("\n", "<br>")
                            st.markdown(
                                f"<div class='answer-wrapper' style='border-left:4px solid {c2};margin-bottom:12px'>"
                                f"<div class='answer-header' style='color:{c2}'>{icon2} {name2}</div>"
                                f"<div class='answer-format'>{PROMPT_INFO[key]['output']}</div>"
                                f"<div class='answer-text'>{ans_html}</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                # Source chunks
                with st.expander("📎 Source chunks used to generate answers"):
                    for i, chunk in enumerate(result.get("retrieved_chunks", [])):
                        sim = chunk["similarity_score"] * 100
                        src = chunk.get("source_collection", "")
                        # Find doc name for this collection
                        doc_name = next((d["filename"] for d in docs if d["collection_name"] == src), src)
                        color_idx = next((i for i, d in enumerate(docs) if d["collection_name"] == src), 0)
                        src_color = DOC_COLORS[color_idx % len(DOC_COLORS)]

                        st.markdown(
                            f"<span class='source-badge' style='background:{src_color}20;color:{src_color};border:1px solid {src_color}'>📄 {doc_name}</span>",
                            unsafe_allow_html=True
                        )
                        st.markdown(f"**Chunk #{i+1}** — Relevance: `{sim:.1f}%`")
                        st.markdown(f"<div class='chunk-card'>{chunk['text']}</div>", unsafe_allow_html=True)
                        st.markdown("")