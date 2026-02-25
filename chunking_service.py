import re
from typing import List, Dict, Any
from dataclasses import dataclass, field
from services.embedding_service import EmbeddingService


@dataclass
class Chunk:
    id: int
    text: str
    token_count: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ══════════════════════════════════════════════
# 1. FIXED-SIZE CHUNKING
# ══════════════════════════════════════════════
def fixed_size_chunking(text: str, chunk_size: int = 500) -> List[Chunk]:
    chars = chunk_size * 4
    chunks, start, idx = [], 0, 0
    while start < len(text):
        end = min(start + chars, len(text))
        t = text[start:end]
        chunks.append(Chunk(idx, t, estimate_tokens(t), start, end,
                            {"strategy": "fixed_size", "chunk_size_target": chunk_size}))
        start = end
        idx += 1
    return chunks


# ══════════════════════════════════════════════
# 2. SENTENCE-BASED CHUNKING
# ══════════════════════════════════════════════
def sentence_based_chunking(text: str, max_sentences: int = 5) -> List[Chunk]:
    splitter = re.compile(r'(?<=[.!?])\s+')
    sentences = [s.strip() for s in splitter.split(text.strip()) if s.strip()]
    chunks, idx = [], 0
    for i in range(0, len(sentences), max_sentences):
        group = sentences[i:i + max_sentences]
        t = " ".join(group)
        start = text.find(group[0]) if group else 0
        chunks.append(Chunk(idx, t, estimate_tokens(t), start, start + len(t),
                            {"strategy": "sentence_based", "sentence_count": len(group)}))
        idx += 1
    return chunks


# ══════════════════════════════════════════════
# 3. SEMANTIC CHUNKING  
# ══════════════════════════════════════════════
def semantic_chunking(text: str, threshold: float = 0.35, min_chunk_sentences: int = 2) -> List[Chunk]:
    """
    Split text where topic/meaning changes using embedding similarity.
    When cosine similarity between consecutive sentences drops below threshold → new chunk.
    """
    splitter = re.compile(r'(?<=[.!?])\s+')
    sentences = [s.strip() for s in splitter.split(text.strip()) if s.strip()]

    if len(sentences) <= min_chunk_sentences:
        t = " ".join(sentences)
        return [Chunk(0, t, estimate_tokens(t), 0, len(t),
                      {"strategy": "semantic", "similarity_threshold": threshold})]

    # Embed all sentences at once (efficient batch)
    embeddings = EmbeddingService.embed(sentences)

    # Find split points where similarity drops
    groups, current_group = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = EmbeddingService.cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim < threshold and len(current_group) >= min_chunk_sentences:
            groups.append(current_group)
            current_group = [sentences[i]]
        else:
            current_group.append(sentences[i])
    if current_group:
        groups.append(current_group)

    chunks = []
    for idx, group in enumerate(groups):
        t = " ".join(group)
        start = text.find(group[0]) if group else 0
        # Compute avg intra-chunk similarity as quality score
        if len(group) > 1:
            embs = EmbeddingService.embed(group)
            sims = [EmbeddingService.cosine_similarity(embs[j], embs[j+1])
                    for j in range(len(embs)-1)]
            avg_sim = round(float(sum(sims) / len(sims)), 3)
        else:
            avg_sim = 1.0
        chunks.append(Chunk(idx, t, estimate_tokens(t), start, start + len(t), {
            "strategy": "semantic",
            "sentence_count": len(group),
            "avg_intra_similarity": avg_sim,
            "similarity_threshold": threshold
        }))
    return chunks


# ══════════════════════════════════════════════
# 4. DOCUMENT-STRUCTURE CHUNKING
# ══════════════════════════════════════════════
def document_structure_chunking(text: str) -> List[Chunk]:
    heading_re = re.compile(r'^(#{1,6}\s.+|[A-Z][A-Z\s]{3,}:?)$', re.MULTILINE)
    sections, last_end, current_heading = [], 0, "Introduction"

    for match in heading_re.finditer(text):
        section_text = text[last_end:match.start()].strip()
        if section_text:
            sections.append((current_heading, section_text, last_end, match.start()))
        current_heading = match.group().strip()
        last_end = match.end()

    remaining = text[last_end:].strip()
    if remaining:
        sections.append((current_heading, remaining, last_end, len(text)))

    # Fallback: paragraph split
    if not sections:
        for idx, para in enumerate(re.split(r'\n\s*\n', text)):
            para = para.strip()
            if para:
                start = text.find(para)
                sections.append((f"Paragraph {idx+1}", para, start, start + len(para)))

    chunks = []
    for idx, (heading, content, start, end) in enumerate(sections):
        t = f"{heading}\n{content}" if heading != content else content
        chunks.append(Chunk(idx, t, estimate_tokens(t), start, end, {
            "strategy": "document_structure",
            "heading": heading,
            "section_index": idx
        }))
    return chunks


# ══════════════════════════════════════════════
# METADATA & UNIFIED RUNNER
# ══════════════════════════════════════════════
STRATEGY_META = {
    "fixed_size": {
        "name": "Fixed-Size Chunking",
        "description": "Splits text into equal token-sized chunks regardless of content.",
        "pros": ["Simple & fast", "Predictable sizes", "Zero compute overhead"],
        "cons": ["May break mid-sentence", "Ignores meaning"],
        "use_when": "Large documents, quick baseline RAG",
        "color": "#6366f1",
        "icon": "⊞"
    },
    "sentence_based": {
        "name": "Sentence-Based Chunking",
        "description": "Groups sentences together, respecting natural language boundaries.",
        "pros": ["Meaning preserved", "Clean readable chunks", "No broken sentences"],
        "cons": ["Uneven sizes", "No semantic awareness"],
        "use_when": "Articles, policies, clean prose documents",
        "color": "#10b981",
        "icon": "≡"
    },
    "semantic": {
        "name": "Semantic Chunking",
        "description": "Splits where topic changes using embedding similarity between sentences.",
        "pros": ["Best semantic relevance", "Topic-coherent chunks", "Production-grade"],
        "cons": ["Slower (needs embeddings)", "Threshold tuning needed"],
        "use_when": "High-accuracy RAG, enterprise search, mixed-topic documents",
        "color": "#f59e0b",
        "icon": "◈"
    },
    "document_structure": {
        "name": "Document-Structure Chunking",
        "description": "Splits by headings, sections, and paragraphs using document hierarchy.",
        "pros": ["Very natural splits", "Preserves hierarchy", "High precision"],
        "cons": ["Needs structured documents", "Fails on unformatted text"],
        "use_when": "PDFs, manuals, legal docs, structured reports",
        "color": "#ef4444",
        "icon": "⊡"
    },
}

STRATEGY_ORDER = ["fixed_size", "sentence_based", "semantic", "document_structure"]


def run_all_strategies(text: str) -> Dict[str, Any]:
    runners = {
        "fixed_size": fixed_size_chunking,
        "sentence_based": sentence_based_chunking,
        "semantic": semantic_chunking,
        "document_structure": document_structure_chunking,
    }
    results = {}
    for key in STRATEGY_ORDER:
        chunks = runners[key](text)
        results[key] = {
            "meta": STRATEGY_META[key],
            "chunk_count": len(chunks),
            "avg_tokens": round(sum(c.token_count for c in chunks) / len(chunks), 1) if chunks else 0,
            "chunks": [{
                "id": c.id,
                "text": c.text,
                "token_count": c.token_count,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "metadata": c.metadata
            } for c in chunks]
        }
    return results