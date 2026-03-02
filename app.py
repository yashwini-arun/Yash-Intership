import streamlit as st
from search_engine import keyword_search, semantic_search, hybrid_search

st.set_page_config(
    page_title="JobFind - Smart Search",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0d0f14; color: #e8eaf0; }
.stApp { background: #0d0f14; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 3rem 2.5rem; max-width: 1050px; }

.hero { text-align:center; padding:2rem 0 1.2rem 0; }
.hero h1 {
    font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800;
    background:linear-gradient(135deg,#a78bfa,#60a5fa,#34d399);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin-bottom:0.2rem; letter-spacing:-1px;
}
.hero p { color:#4b5563; font-size:0.9rem; letter-spacing:0.06em; }

.stButton > button {
    font-family:'Syne',sans-serif !important; font-weight:700 !important;
    font-size:0.88rem !important; border-radius:12px !important;
    padding:0.65rem 0.5rem !important; width:100% !important;
    transition:all 0.2s ease !important;
}
div[data-testid="column"]:nth-child(1) .stButton > button {
    background:rgba(245,158,11,0.08) !important; color:#f59e0b !important;
    border:2px solid rgba(245,158,11,0.35) !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button {
    background:rgba(52,211,153,0.08) !important; color:#34d399 !important;
    border:2px solid rgba(52,211,153,0.35) !important;
}
div[data-testid="column"]:nth-child(3) .stButton > button {
    background:rgba(167,139,250,0.08) !important; color:#a78bfa !important;
    border:2px solid rgba(167,139,250,0.35) !important;
}

.stTextInput > div > div > input {
    background:#161b27 !important; border:1.5px solid #2d3748 !important;
    border-radius:12px !important; color:#e8eaf0 !important;
    font-family:'DM Sans',sans-serif !important; font-size:1rem !important;
    padding:0.85rem 1.2rem !important;
}
.stTextInput > div > div > input:focus {
    border-color:#a78bfa !important;
    box-shadow:0 0 0 3px rgba(167,139,250,0.12) !important;
}
.stTextInput > div > div > input::placeholder { color:#374151 !important; }
.stSelectbox > div > div {
    background:#161b27 !important; border:1.5px solid #2d3748 !important;
    border-radius:12px !important; color:#e8eaf0 !important;
}

.mode-banner {
    border-radius:14px; padding:1rem 1.4rem;
    margin:0.8rem 0 1.2rem 0;
    display:flex; align-items:center; gap:14px;
}
.mode-banner-kw  { background:rgba(245,158,11,0.07);  border:1.5px solid rgba(245,158,11,0.3); }
.mode-banner-sem { background:rgba(52,211,153,0.07);  border:1.5px solid rgba(52,211,153,0.3); }
.mode-banner-hyb { background:rgba(167,139,250,0.07); border:1.5px solid rgba(167,139,250,0.3); }
.mode-icon  { font-size:2rem; line-height:1; }
.mode-info  { flex:1; }
.mode-title { font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; margin-bottom:2px; }
.mode-desc  { font-size:0.82rem; color:#6b7280; }
.mode-tag   {
    font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700;
    letter-spacing:0.12em; padding:3px 10px; border-radius:20px; white-space:nowrap;
}
.tag-kw  { background:rgba(245,158,11,0.15);  color:#f59e0b; }
.tag-sem { background:rgba(52,211,153,0.15);  color:#34d399; }
.tag-hyb { background:rgba(167,139,250,0.15); color:#a78bfa; }

.card-kw {
    background:linear-gradient(135deg,rgba(26,21,0,0.5) 0%,#161b27 60%);
    border:1px solid rgba(245,158,11,0.2); border-left:4px solid #f59e0b;
    border-radius:14px; padding:1.3rem 1.5rem; margin-bottom:0.9rem;
}
.card-sem {
    background:linear-gradient(135deg,rgba(0,26,16,0.5) 0%,#161b27 60%);
    border:1px solid rgba(52,211,153,0.15); border-left:4px solid #34d399;
    border-radius:14px; padding:1.3rem 1.5rem; margin-bottom:0.9rem;
}
.card-hyb {
    background:linear-gradient(135deg,rgba(13,8,32,0.5) 0%,#161b27 60%);
    border:1px solid rgba(167,139,250,0.15); border-left:4px solid #a78bfa;
    border-radius:14px; padding:1.3rem 1.5rem; margin-bottom:0.9rem;
}

.jc-top   { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.5rem; }
.jc-rank-kw  { font-family:'Syne',sans-serif; font-size:0.68rem; font-weight:700; letter-spacing:0.1em; color:rgba(245,158,11,0.7); }
.jc-rank-sem { font-family:'Syne',sans-serif; font-size:0.68rem; font-weight:700; letter-spacing:0.1em; color:rgba(52,211,153,0.7); }
.jc-rank-hyb { font-family:'Syne',sans-serif; font-size:0.68rem; font-weight:700; letter-spacing:0.1em; color:rgba(167,139,250,0.7); }
.jc-title { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#e8eaf0; }
.badge    { padding:3px 11px; border-radius:20px; font-size:0.72rem; font-weight:700; font-family:'Syne',sans-serif; }
.badge-r  { background:rgba(52,211,153,0.12);  color:#34d399; border:1px solid rgba(52,211,153,0.25); }
.badge-o  { background:rgba(239,68,68,0.12);   color:#f87171; border:1px solid rgba(239,68,68,0.25); }
.badge-h  { background:rgba(251,146,60,0.12);  color:#fb923c; border:1px solid rgba(251,146,60,0.25); }
.jc-meta   { color:#6b7280; font-size:0.84rem; margin-bottom:0.4rem; }
.jc-skills { font-size:0.81rem; color:#4b5563; margin-bottom:0.5rem; }
.jc-desc   { font-size:0.86rem; color:#9ca3af; line-height:1.6; margin-bottom:0.8rem; }

.score-box-kw  { background:rgba(245,158,11,0.05);  border:1px solid rgba(245,158,11,0.15);  border-radius:8px; padding:0.65rem 1rem; }
.score-box-sem { background:rgba(52,211,153,0.05);  border:1px solid rgba(52,211,153,0.15);  border-radius:8px; padding:0.65rem 1rem; }
.score-box-hyb { background:rgba(167,139,250,0.05); border:1px solid rgba(167,139,250,0.15); border-radius:8px; padding:0.65rem 1rem; }
.score-bar-wrap { height:4px; background:#1e2535; border-radius:4px; margin-bottom:0.55rem; overflow:hidden; }
.sb-fill-kw  { height:100%; background:linear-gradient(90deg,#f59e0b,#fcd34d); border-radius:4px; }
.sb-fill-sem { height:100%; background:linear-gradient(90deg,#34d399,#6ee7b7); border-radius:4px; }
.sb-fill-hyb { height:100%; background:linear-gradient(90deg,#a78bfa,#60a5fa); border-radius:4px; }
.score-pills { display:flex; gap:16px; flex-wrap:wrap; }
.score-pill  { display:flex; flex-direction:column; gap:1px; }
.sp-label    { font-family:'Syne',sans-serif; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em; color:#374151; }
.sp-val-kw   { font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:700; color:#f59e0b; }
.sp-val-sem  { font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:700; color:#34d399; }
.sp-val-hyb  { font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:700; color:#a78bfa; }
.sp-val-sub  { font-family:'Syne',sans-serif; font-size:0.82rem; font-weight:600; color:#6b7280; }

.results-hdr { font-family:'Syne',sans-serif; font-size:1.25rem; font-weight:700; color:#e8eaf0; margin-bottom:1rem; }
.empty-state { text-align:center; padding:3.5rem 2rem; }
.empty-icon  { font-size:2.5rem; margin-bottom:0.8rem; }
.empty-h3    { font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:#374151; }
.empty-p     { font-size:0.83rem; color:#2d3748; margin-top:0.3rem; }

[data-testid="stSidebar"] { background:#0a0c10 !important; border-right:1px solid #1e2535 !important; }
[data-testid="stSidebar"] .block-container { padding:1.5rem !important; }
.sb-title { font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:#e8eaf0; margin-bottom:0.8rem; }
.sb-sec   { font-family:'Syne',sans-serif; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.12em; margin:1rem 0 0.4rem 0; }
.tip-chip { display:inline-block; background:#161b27; border:1px solid #2d3748; border-radius:8px; padding:3px 9px; font-size:0.79rem; color:#6b7280; margin:2px 2px; }
.how-row  { font-size:0.82rem; line-height:2.2; color:#4b5563; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Session state ──
if "search_type" not in st.session_state:
    st.session_state.search_type = "hybrid"

# ── Hero ──
st.markdown(
    '<div class="hero"><h1>🎯 JobFind</h1>'
    '<p>KEYWORD &nbsp;·&nbsp; SEMANTIC &nbsp;·&nbsp; HYBRID — three ways to find your next role</p></div>',
    unsafe_allow_html=True
)

# ── Mode Buttons ──
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("⚡  Keyword Search"):
        st.session_state.search_type = "keyword"
with c2:
    if st.button("🧠  Semantic Search"):
        st.session_state.search_type = "semantic"
with c3:
    if st.button("🔀  Hybrid Search"):
        st.session_state.search_type = "hybrid"

stype = st.session_state.search_type

# ── Mode Banner ──
if stype == "keyword":
    st.markdown(
        '<div class="mode-banner mode-banner-kw">'
        '<div class="mode-icon">⚡</div>'
        '<div class="mode-info">'
        '<div class="mode-title" style="color:#f59e0b;">Keyword Search — BM25</div>'
        '<div class="mode-desc">Finds exact word matches in job titles, skills, and descriptions. Best for specific role names or technologies.</div>'
        '</div>'
        '<span class="mode-tag tag-kw">EXACT MATCH</span>'
        '</div>',
        unsafe_allow_html=True
    )
elif stype == "semantic":
    st.markdown(
        '<div class="mode-banner mode-banner-sem">'
        '<div class="mode-icon">🧠</div>'
        '<div class="mode-info">'
        '<div class="mode-title" style="color:#34d399;">Semantic Search — FAISS</div>'
        '<div class="mode-desc">Understands the meaning behind your query using AI embeddings. Best for natural language like "I want a remote data job".</div>'
        '</div>'
        '<span class="mode-tag tag-sem">MEANING-BASED</span>'
        '</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="mode-banner mode-banner-hyb">'
        '<div class="mode-icon">🔀</div>'
        '<div class="mode-info">'
        '<div class="mode-title" style="color:#a78bfa;">Hybrid Search — BM25 + FAISS</div>'
        '<div class="mode-desc">Combines exact keyword matching with semantic understanding. Use the alpha slider to tune the balance.</div>'
        '</div>'
        '<span class="mode-tag tag-hyb">RECOMMENDED</span>'
        '</div>',
        unsafe_allow_html=True
    )

# ── Search Bar ──
placeholders = {
    "keyword":  "e.g. Python Developer  or  SQL Analyst  or  React JavaScript",
    "semantic": "e.g. remote data job  or  work involving AI  or  build mobile apps",
    "hybrid":   "e.g. ML engineer Bangalore  or  backend cloud dev  or  remote NLP role",
}
q1, q2 = st.columns([7, 1])
with q1:
    query = st.text_input("q", placeholder=placeholders[stype], label_visibility="collapsed")
with q2:
    top_k = st.selectbox("k", [5, 10, 15], label_visibility="collapsed")

alpha = 0.6
if stype == "hybrid":
    alpha = st.slider("Semantic weight alpha  —  left = more keyword  |  right = more semantic", 0.0, 1.0, 0.6, 0.1)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ── Card renderer — uses string concatenation, NO f-strings with HTML ──
def render_card(job, rank, stype, alpha=0.6):
    badge_map = {"Remote": "badge-r", "Onsite": "badge-o", "Hybrid": "badge-h"}
    badge_cls = badge_map.get(job["type"], "badge-o")
    score_val = str(job["score"])
    bar_pct   = str(min(int(job["score"] * 100), 100) if job["score"] <= 1 else min(int(job["score"] * 8), 100))

    if stype == "keyword":
        card_cls  = "card-kw"
        rank_cls  = "jc-rank-kw"
        box_cls   = "score-box-kw"
        fill_cls  = "sb-fill-kw"
        val_cls   = "sp-val-kw"
        score_html = (
            '<div class="score-pill">'
            '<span class="sp-label">BM25 Score</span>'
            '<span class="' + val_cls + '">' + score_val + '</span>'
            '</div>'
            '<div class="score-pill">'
            '<span class="sp-label">Method</span>'
            '<span class="sp-val-sub">Exact Token Match</span>'
            '</div>'
        )
    elif stype == "semantic":
        card_cls  = "card-sem"
        rank_cls  = "jc-rank-sem"
        box_cls   = "score-box-sem"
        fill_cls  = "sb-fill-sem"
        val_cls   = "sp-val-sem"
        score_html = (
            '<div class="score-pill">'
            '<span class="sp-label">Similarity Score</span>'
            '<span class="' + val_cls + '">' + score_val + '</span>'
            '</div>'
            '<div class="score-pill">'
            '<span class="sp-label">Method</span>'
            '<span class="sp-val-sub">Cosine Similarity</span>'
            '</div>'
        )
    else:
        card_cls  = "card-hyb"
        rank_cls  = "jc-rank-hyb"
        box_cls   = "score-box-hyb"
        fill_cls  = "sb-fill-hyb"
        val_cls   = "sp-val-hyb"
        bm25_s    = str(job.get("bm25_score", "-"))
        sem_s     = str(job.get("semantic_score", "-"))
        score_html = (
            '<div class="score-pill">'
            '<span class="sp-label">Hybrid Score</span>'
            '<span class="' + val_cls + '">' + score_val + '</span>'
            '</div>'
            '<div class="score-pill">'
            '<span class="sp-label">BM25</span>'
            '<span class="sp-val-sub">' + bm25_s + '</span>'
            '</div>'
            '<div class="score-pill">'
            '<span class="sp-label">Semantic</span>'
            '<span class="sp-val-sub">' + sem_s + '</span>'
            '</div>'
            '<div class="score-pill">'
            '<span class="sp-label">Alpha</span>'
            '<span class="sp-val-sub">' + str(alpha) + '</span>'
            '</div>'
        )

    html = (
        '<div class="' + card_cls + '">'
          '<div class="jc-top">'
            '<div>'
              '<div class="' + rank_cls + '">RANK #' + str(rank) + '</div>'
              '<div class="jc-title">' + job["job_title"] + '</div>'
            '</div>'
            '<span class="badge ' + badge_cls + '">' + job["type"] + '</span>'
          '</div>'
          '<div class="jc-meta">'
            '🏢 <b style="color:#9ca3af">' + job["company"] + '</b>'
            ' &nbsp;·&nbsp; 📍 ' + job["location"] +
            ' &nbsp;·&nbsp; 🕒 ' + job["experience"] +
          '</div>'
          '<div class="jc-skills">🛠️ ' + job["skills"] + '</div>'
          '<div class="jc-desc">' + job["description"] + '</div>'
          '<div class="' + box_cls + '">'
            '<div class="score-bar-wrap">'
              '<div class="' + fill_cls + '" style="width:' + bar_pct + '%"></div>'
            '</div>'
            '<div class="score-pills">' + score_html + '</div>'
          '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ── Run Search ──
if query.strip():
    with st.spinner("Searching..."):
        if stype == "keyword":
            results = keyword_search(query, top_k=top_k)
        elif stype == "semantic":
            results = semantic_search(query, top_k=top_k)
        else:
            results = hybrid_search(query, top_k=top_k, alpha=alpha)

    if not results:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">🔍</div>'
            '<div class="empty-h3">No results found</div>'
            '<div class="empty-p">Try a different query or switch modes</div>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        hdr_colors = {"keyword": "#f59e0b", "semantic": "#34d399", "hybrid": "#a78bfa"}
        hdr_color  = hdr_colors[stype]
        st.markdown(
            '<div class="results-hdr">' + str(len(results)) + ' results for '
            '<span style="color:' + hdr_color + '">"' + query + '"</span></div>',
            unsafe_allow_html=True
        )
        for i, job in enumerate(results, 1):
            render_card(job, i, stype, alpha)

else:
    empty_cfg = {
        "keyword":  ("⚡", "Type an exact skill or job title",      'e.g. "Python Developer" or "React JavaScript"'),
        "semantic": ("🧠", "Describe the job you want naturally",   'e.g. "remote job involving AI and data"'),
        "hybrid":   ("🎯", "Search anything — hybrid handles it",   'e.g. "ML engineer Bangalore" or "backend cloud dev"'),
    }
    icon, h3, p = empty_cfg[stype]
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-icon">' + icon + '</div>'
        '<div class="empty-h3">' + h3 + '</div>'
        '<div class="empty-p">' + p + '</div>'
        '</div>',
        unsafe_allow_html=True
    )

# ── Sidebar ──
with st.sidebar:
    st.markdown('<div class="sb-title">💡 Sample Queries</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec" style="color:#f59e0b;">⚡ Keyword Mode</div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="tip-chip">Python Developer</span>'
        '<span class="tip-chip">SQL Analyst</span>'
        '<span class="tip-chip">React JS</span>'
        '<span class="tip-chip">DevOps AWS</span>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sb-sec" style="color:#34d399;">🧠 Semantic Mode</div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="tip-chip">remote data job</span>'
        '<span class="tip-chip">job involving AI</span>'
        '<span class="tip-chip">build mobile apps</span>'
        '<span class="tip-chip">cloud work</span>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sb-sec" style="color:#a78bfa;">🔀 Hybrid Mode</div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="tip-chip">ML engineer Bangalore</span>'
        '<span class="tip-chip">backend cloud dev</span>'
        '<span class="tip-chip">remote NLP role</span>'
        '<span class="tip-chip">data Chennai</span>',
        unsafe_allow_html=True
    )

    st.markdown("<hr style='border-color:#1e2535;margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown('<div class="sb-sec" style="color:#374151;">HOW EACH MODE WORKS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="how-row">'
        '<span style="color:#f59e0b;font-weight:700;">⚡ Keyword</span> — exact BM25 token match<br>'
        '&nbsp;&nbsp;Best for: titles, skills, tech names<br><br>'
        '<span style="color:#34d399;font-weight:700;">🧠 Semantic</span> — AI embedding similarity<br>'
        '&nbsp;&nbsp;Best for: intent, natural language<br><br>'
        '<span style="color:#a78bfa;font-weight:700;">🔀 Hybrid</span> — BM25 + semantic combined<br>'
        '&nbsp;&nbsp;Best for: everything'
        '</div>',
        unsafe_allow_html=True
    )