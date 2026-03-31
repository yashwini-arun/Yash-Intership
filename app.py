import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from pipeline.orchestrator import run_pipeline
from guardrails.input_guardrail import InputGuardrailError
from guardrails.output_guardrail import OutputGuardrailError

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0a0a0f; color: #e2e8f0; }

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero p { color: #94a3b8; font-size: 1rem; margin: 0; }

/* Search bar */
.stTextInput > div > div > input {
    background: #13131a !important;
    border: 1.5px solid #2d2d3d !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    padding: 0.8rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.15) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #13131a !important;
    border: 1.5px solid #2d2d3d !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.65rem 1rem !important;
    width: 100% !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    transition: opacity 0.2s !important;
    height: 3.2rem !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Agent cards */
.agent-card {
    background: #13131a;
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.agent-card-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
}
.agent-badge {
    background: linear-gradient(135deg, #667eea22, #764ba222);
    border: 1px solid #667eea44;
    color: #a78bfa;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.agent-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e2e8f0;
}

/* Metric boxes */
.metric-box {
    background: #0d0d14;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem; }
.metric-value { font-size: 0.95rem; font-weight: 600; color: #a78bfa; word-break: break-word; }

/* Source tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d0d14 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: #64748b !important;
    font-weight: 500 !important;
    padding: 0.4rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background: #1e1e2e !important;
    color: #a78bfa !important;
}

/* Source paper card */
.paper-card {
    background: #0d0d14;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.paper-title { font-weight: 600; color: #e2e8f0; font-size: 0.95rem; margin-bottom: 0.3rem; }
.paper-meta { font-size: 0.78rem; color: #64748b; margin-bottom: 0.5rem; }
.paper-summary { font-size: 0.88rem; color: #94a3b8; line-height: 1.6; margin-bottom: 0.6rem; }
.paper-link a { color: #667eea; font-size: 0.82rem; text-decoration: none; }

/* Progress steps */
.step-done {
    display: flex; align-items: center; gap: 0.5rem;
    background: #0d1f0d; border: 1px solid #1a3a1a;
    border-radius: 10px; padding: 0.55rem 1rem;
    margin-bottom: 0.4rem; font-size: 0.9rem; color: #4ade80;
}
.step-active {
    display: flex; align-items: center; gap: 0.5rem;
    background: #13131a; border: 1px solid #2d2d3d;
    border-radius: 10px; padding: 0.55rem 1rem;
    margin-bottom: 0.4rem; font-size: 0.9rem; color: #a78bfa;
    animation: pulse 1.5s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }

/* Final report */
.report-container {
    background: #13131a;
    border: 1px solid #1e1e2e;
    border-left: 4px solid #667eea;
    border-radius: 16px;
    padding: 2rem;
    line-height: 1.8;
    color: #cbd5e1;
}
.report-container h1, .report-container h2, .report-container h3 {
    color: #e2e8f0 !important;
}

/* Divider */
hr { border-color: #1e1e2e !important; margin: 1.5rem 0 !important; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 960px !important; margin: auto; }
</style>
""", unsafe_allow_html=True)

# ── LANGUAGE CONFIG ───────────────────────────────────────────────────────────
LANGUAGES = {
    "🇬🇧 English": "English",
    "🇮🇳 Hindi": "Hindi",
    "🇮🇳 Tamil": "Tamil",
    "🇮🇳 Telugu": "Telugu",
    "🇰🇷 Korean": "Korean",
    "🇫🇷 French": "French",
}

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔬 Research Assistant</h1>
    <p>Multi-agent AI research system · arXiv · Wikipedia · Semantic Scholar</p>
</div>
""", unsafe_allow_html=True)

# ── INPUT ROW ─────────────────────────────────────────────────────────────────
col_q, col_lang, col_btn = st.columns([4, 2, 1.5])

with col_q:
    query = st.text_input(
        "query",
        placeholder="Ask anything — e.g. Impact of AI on healthcare...",
        label_visibility="collapsed"
    )

with col_lang:
    lang_display = st.selectbox(
        "language",
        options=list(LANGUAGES.keys()),
        index=0,
        label_visibility="collapsed"
    )
    selected_language = LANGUAGES[lang_display]

with col_btn:
    search_btn = st.button("🔍  Research Now", use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── PIPELINE ──────────────────────────────────────────────────────────────────
if search_btn and query.strip():

    steps_placeholder = st.empty()
    steps_done = []

    def update_status(msg):
        steps_done.append(msg)
        html = '<div style="margin-bottom:1rem;">'
        for i, s in enumerate(steps_done):
            cls = "step-active" if i == len(steps_done) - 1 else "step-done"
            icon = "⟳" if i == len(steps_done) - 1 else "✓"
            html += f'<div class="{cls}"><span>{icon}</span>{s.replace("...", "")}</div>'
        html += "</div>"
        steps_placeholder.markdown(html, unsafe_allow_html=True)

    with st.spinner(""):
        try:
            result = run_pipeline(query, status_callback=update_status, output_language=selected_language)
        except InputGuardrailError as e:
            st.markdown(f"""
            <div style="background:#1f0d0d;border:1px solid #3a1a1a;border-left:4px solid #f87171;border-radius:12px;padding:1rem 1.4rem;margin-top:1rem;">
                <div style="font-size:0.95rem;font-weight:600;color:#f87171;margin-bottom:4px;">🚫 Input Blocked by Guardrail</div>
                <div style="font-size:0.85rem;color:#94a3b8;">{e}</div>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        except OutputGuardrailError as e:
            st.markdown(f"""
            <div style="background:#1f0d0d;border:1px solid #3a1a1a;border-left:4px solid #fbbf24;border-radius:12px;padding:1rem 1.4rem;margin-top:1rem;">
                <div style="font-size:0.95rem;font-weight:600;color:#fbbf24;margin-bottom:4px;">⚠️ Output Blocked by Guardrail</div>
                <div style="font-size:0.85rem;color:#94a3b8;">{e}</div>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    # Final done state
    html = '<div style="margin-bottom:1rem;">'
    for s in steps_done:
        html += f'<div class="step-done"><span>✓</span>{s.replace("...", "")}</div>'
    html += "</div>"
    steps_placeholder.markdown(html, unsafe_allow_html=True)


    # ── GUARDRAILS PANEL ──────────────────────────────────────────────────────
    guardrail_log = result.get("guardrail_log", [])
    with st.expander("🛡️ Guardrails — All Checks", expanded=True):
        if not guardrail_log:
            st.info("No guardrail data available.")
        else:
            for item in guardrail_log:
                status  = item.get("status", "pass")
                check   = item.get("check", "")
                detail  = item.get("detail", "")
                if status == "pass":
                    color  = "#4ade80"
                    bg     = "#0d1f0d"
                    border = "#1a3a1a"
                    icon   = "✅"
                    label  = "PASSED"
                else:
                    color  = "#f87171"
                    bg     = "#1f0d0d"
                    border = "#3a1a1a"
                    icon   = "❌"
                    label  = "FAILED"

                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:1rem;background:{bg};border:1px solid {border};border-radius:10px;padding:0.75rem 1rem;margin-bottom:0.5rem;">
                    <div style="font-size:1.1rem;">{icon}</div>
                    <div style="flex:1;">
                        <div style="font-size:0.88rem;font-weight:600;color:{color};">{check}</div>
                        <div style="font-size:0.78rem;color:#64748b;margin-top:2px;">{detail}</div>
                    </div>
                    <div style="font-size:0.7rem;font-weight:700;color:{color};background:{border};padding:2px 8px;border-radius:20px;">{label}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── AGENT 1 — MULTILINGUAL ────────────────────────────────────────────
    a1 = result["agent1_multilingual"]
    st.markdown(f"""
    <div class="agent-card">
        <div class="agent-card-header">
            <span class="agent-badge">Agent 1</span>
            <span class="agent-title">🌍 Multilingual Agent</span>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.8rem;">
            <div class="metric-box">
                <div class="metric-label">Original Query</div>
                <div class="metric-value">{a1['original_query'][:55]}{'...' if len(a1['original_query'])>55 else ''}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Detected Language</div>
                <div class="metric-value">{a1['detected_language']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Output Language</div>
                <div class="metric-value">{selected_language}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── AGENT 2 — SEARCH ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class="agent-card">
        <div class="agent-card-header">
            <span class="agent-badge">Agent 2</span>
            <span class="agent-title">🔍 Search Agent</span>
        </div>
        <div style="color:#64748b; font-size:0.82rem; margin-top:0.3rem;">
            📌 Raw source papers are fetched from external APIs (arXiv, Wikipedia, CrossRef) and are published in English. 
            Agent 3 onwards processes and outputs everything in <strong style="color:#a78bfa">{selected_language}</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    raw = result["agent2_search"]

    # ── SEARCH VERIFICATION RESULT ────────────────────────────────────────
    sq = result.get("search_quality", {})
    retries = result.get("search_retry_count", 0)
    if sq:
        sq_score  = sq.get("quality_score", 0)
        sq_pass   = sq.get("is_sufficient", True)
        sq_reason = sq.get("reason", "")

        if sq_score >= 90:
            sq_color  = "#4ade80"
            sq_bg     = "#0d1f0d"
            sq_border = "#1a3a1a"
            sq_label  = "Quality Passed"
            sq_icon   = "✅"
        elif sq_score >= 70:
            sq_color  = "#fbbf24"
            sq_bg     = "#1a1500"
            sq_border = "#3a2f00"
            sq_label  = f"Retrying for better quality (attempt {retries})"
            sq_icon   = "🔄"
        else:
            sq_color  = "#f87171"
            sq_bg     = "#1f0d0d"
            sq_border = "#3a1a1a"
            sq_label  = f"Low Quality — Retried {retries}x"
            sq_icon   = "❌"

        col1, col2 = st.columns([1, 6])
        with col1:
            st.markdown(
                f'<div style="background:{sq_bg};border:1px solid {sq_border};border-radius:12px;padding:1rem;text-align:center;">' +
                f'<div style="font-size:2rem;font-weight:700;color:{sq_color};line-height:1;">{sq_score}</div>' +
                f'<div style="font-size:0.7rem;color:{sq_color};opacity:0.7;">/ 100</div></div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f'<div style="background:{sq_bg};border:1px solid {sq_border};border-radius:12px;padding:1rem;">' +
                f'<div style="font-size:0.9rem;font-weight:600;color:{sq_color};">{sq_icon} Search Verification Agent — {sq_label}</div>' +
                f'<div style="font-size:0.8rem;color:#94a3b8;margin-top:6px;">{sq_reason}</div></div>',
                unsafe_allow_html=True
            )
        if retries > 0:
            st.info(f"🔄 Search was retried {retries} time(s) with a refined query to improve quality.")

    tabs = st.tabs(["📰 arXiv", "📖 Wikipedia", "🎓 CrossRef"])

    with tabs[0]:
        items = raw.get("arxiv", [])
        if not items:
            st.info("No results found.")
        for item in items:
            if "error" in item:
                st.error(f"Error: {item['error']}")
            else:
                authors = ", ".join(item.get("authors", []))
                url = item.get("url", "")
                link_html = f'<a href="{url}" target="_blank">🔗 Open Paper</a>' if url else ""
                st.markdown(f"""
                <div class="paper-card">
                    <div class="paper-title">{item.get('title','N/A')}</div>
                    <div class="paper-meta">👤 {authors}</div>
                    <div class="paper-summary">{item.get('summary','')}</div>
                    <div class="paper-link">{link_html}</div>
                </div>
                """, unsafe_allow_html=True)

    with tabs[1]:
        items = raw.get("wikipedia", [])
        if not items:
            st.info("No results found.")
        for item in items:
            if "error" in item:
                st.error(f"Error: {item['error']}")
            else:
                url = item.get("url", "")
                link_html = f'<a href="{url}" target="_blank">🔗 Open Article</a>' if url else ""
                st.markdown(f"""
                <div class="paper-card">
                    <div class="paper-title">{item.get('title','N/A')}</div>
                    <div class="paper-summary">{item.get('summary','')}</div>
                    <div class="paper-link">{link_html}</div>
                </div>
                """, unsafe_allow_html=True)

    with tabs[2]:
        items = raw.get("semantic_scholar", [])
        if not items:
            st.info("No results found.")
        for item in items:
            if "error" in item:
                st.error(f"⚠️ {item['error']}")
            else:
                authors = ", ".join(item.get("authors", []))
                year = item.get("year", "N/A")
                journal = item.get("journal", "")
                url = item.get("url", "")
                link_html = f'<a href="{url}" target="_blank">🔗 Open Paper</a>' if url else ""
                journal_html = f'<span style="color:#64748b"> · {journal}</span>' if journal else ""
                st.markdown(f"""
                <div class="paper-card">
                    <div class="paper-title">{item.get('title','N/A')} <span style="color:#64748b;font-weight:400">({year})</span></div>
                    <div class="paper-meta">👤 {authors}{journal_html}</div>
                    <div class="paper-summary">{item.get('summary','')}</div>
                    <div class="paper-link">{link_html}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AGENT 3 — SUMMARIZATION ───────────────────────────────────────────
    with st.expander("📝 Agent 3 — Summarization Agent", expanded=False):
        st.markdown(f"""
        <div style="background:#0d0d14;border-radius:12px;padding:1.2rem;color:#94a3b8;line-height:1.8;font-size:0.92rem;">
        {result['agent3_summary']}
        </div>
        """, unsafe_allow_html=True)

    # ── AGENT 4 — VERIFICATION ────────────────────────────────────────────
    with st.expander("✅ Agent 4 — Verification Agent", expanded=True):
        import re
        verified_text = str(result['agent4_verified'])

        # Extract score — always Arabic numerals e.g. 84/100
        score_match = re.search(r'\b(\d{1,3})\s*/\s*100\b', verified_text)
        score = int(score_match.group(1)) if score_match else None

        # Extract REASON — 2 sentences after "REASON:"
        reason_match = re.search(
            r'REASON\s*:\s*(.+?)(?=\nWELL|$)',
            verified_text, re.DOTALL | re.IGNORECASE
        )
        reason_text = reason_match.group(1).strip().replace('\n', ' ') if reason_match else ""

        # Show score card only if score found
        if score is not None:
            if score >= 80:
                score_color = "#4ade80"
                score_bg = "#0d1f0d"
                score_border = "#1a3a1a"
                score_label = "High Accuracy"
            elif score >= 60:
                score_color = "#fbbf24"
                score_bg = "#1a1500"
                score_border = "#3a2f00"
                score_label = "Moderate Accuracy"
            else:
                score_color = "#f87171"
                score_bg = "#1f0d0d"
                score_border = "#3a1a1a"
                score_label = "Low Accuracy"

            display_reason = reason_text if reason_text else "Score based on source support, claim strength, and cross-source consistency."

            st.markdown(f"""
            <div style="background:{score_bg};border:1px solid {score_border};border-radius:14px;padding:1.4rem 1.6rem;margin-bottom:1rem;">
                <div style="display:flex;align-items:flex-start;gap:1.5rem;">
                    <div style="text-align:center;min-width:80px;flex-shrink:0;">
                        <div style="font-size:2.8rem;font-weight:700;color:{score_color};line-height:1;">{score}</div>
                        <div style="font-size:0.72rem;color:{score_color};opacity:0.7;margin-top:2px;">/ 100</div>
                    </div>
                    <div style="width:1px;min-height:60px;background:{score_border};flex-shrink:0;"></div>
                    <div>
                        <div style="font-size:1rem;font-weight:600;color:{score_color};margin-bottom:6px;">{score_label}</div>
                        <div style="font-size:0.82rem;color:#94a3b8;line-height:1.7;">{display_reason}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)



    st.markdown("<hr>", unsafe_allow_html=True)

    # ── AGENT 5 — FINAL REPORT ────────────────────────────────────────────
    lang_badge = f'<span style="background:#1e1e2e;border:1px solid #2d2d3d;color:#a78bfa;font-size:0.78rem;padding:0.2rem 0.7rem;border-radius:20px;font-weight:600;">📌 {selected_language}</span>'
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:1rem;">
        <span class="agent-badge">Agent 5</span>
        <span style="font-size:1.15rem;font-weight:700;color:#e2e8f0;">📄 Final Research Report</span>
        {lang_badge}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="report-container">
    {result['agent5_final_report']}
    </div>
    """, unsafe_allow_html=True)

elif search_btn and not query.strip():
    st.warning("⚠️ Please enter a research question first.")

else:
    # ── LANDING ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:2rem 0;">
        <div style="display:inline-flex;gap:1.5rem;flex-wrap:wrap;justify-content:center;margin-bottom:2rem;">
            <div style="background:#13131a;border:1px solid #1e1e2e;border-radius:14px;padding:1.2rem 1.5rem;min-width:160px;">
                <div style="font-size:1.8rem;margin-bottom:0.4rem;">🌍</div>
                <div style="font-weight:600;color:#e2e8f0;font-size:0.9rem;">Multilingual</div>
                <div style="color:#64748b;font-size:0.78rem;">6 languages</div>
            </div>
            <div style="background:#13131a;border:1px solid #1e1e2e;border-radius:14px;padding:1.2rem 1.5rem;min-width:160px;">
                <div style="font-size:1.8rem;margin-bottom:0.4rem;">🔍</div>
                <div style="font-weight:600;color:#e2e8f0;font-size:0.9rem;">3 Data Sources</div>
                <div style="color:#64748b;font-size:0.78rem;">arXiv · Wiki · Scholar</div>
            </div>
            <div style="background:#13131a;border:1px solid #1e1e2e;border-radius:14px;padding:1.2rem 1.5rem;min-width:160px;">
                <div style="font-size:1.8rem;margin-bottom:0.4rem;">🤖</div>
                <div style="font-weight:600;color:#e2e8f0;font-size:0.9rem;">5 AI Agents</div>
                <div style="color:#64748b;font-size:0.78rem;">Llama 3.3 70B</div>
            </div>
            <div style="background:#13131a;border:1px solid #1e1e2e;border-radius:14px;padding:1.2rem 1.5rem;min-width:160px;">
                <div style="font-size:1.8rem;margin-bottom:0.4rem;">✅</div>
                <div style="font-weight:600;color:#e2e8f0;font-size:0.9rem;">Verified</div>
                <div style="color:#64748b;font-size:0.78rem;">Fact-checked output</div>
            </div>
        </div>
        <div style="color:#64748b;font-size:0.9rem;">
            Type any question above and select your output language · Press 🔍 Research
        </div>
    </div>
    """, unsafe_allow_html=True)