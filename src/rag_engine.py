"""
rag_engine.py
=============
Single RAG module (replaces the old duplicated rag_engine.py + rag_retriever.py,
which held two different, overlapping rule lists — only one of which was
actually wired into the app). This is now the only source of expert rules.

MEMORY NOTE (Aug 2026): this used to try loading the real sentence-transformers
MiniLM model ('all-MiniLM-L6-v2') at import time, falling back to the
deterministic OfflineHashEmbedder only if that failed (e.g. no network to
huggingface.co). That eager MiniLM load ran as a side effect of
`import rag_engine` — before FastAPI's own startup hook even fires, i.e.
before any of main.py's "lazy load the heavy stuff" logic gets a chance to
run — and was one of several transformer models competing for the 512MB
ceiling on Render's free tier, contributing directly to the /analyze
OOM crash-loop (process silently dying mid-request, no traceback, then
auto-restarting).

Since this RAG step only compares a fixed, short list of expert-rule
strings for retrieval (not open-ended semantic search over arbitrary
text), the hashing-based OfflineHashEmbedder's lower semantic fidelity is
an acceptable trade for a large, guaranteed memory savings — no torch
model load at all for this module. Set USE_REAL_EMBEDDER=1 in the
environment to opt back into MiniLM (e.g. after upgrading off the free
tier); everything else in this file is unchanged either way.
"""

import os
import faiss
import numpy as np

if os.environ.get("USE_REAL_EMBEDDER") == "1":
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("[rag_engine] USE_REAL_EMBEDDER=1 — using real MiniLM embedder.")
    except Exception as e:
        print(f"[rag_engine] Could not load real MiniLM embedder ({e}); using offline fallback.")
        try:
            from .offline_embedder import OfflineHashEmbedder
        except ImportError:
            from offline_embedder import OfflineHashEmbedder
        embedder = OfflineHashEmbedder()
else:
    try:
        from .offline_embedder import OfflineHashEmbedder
    except ImportError:
        from offline_embedder import OfflineHashEmbedder
    embedder = OfflineHashEmbedder()
    print("[rag_engine] Using OfflineHashEmbedder (default, low-memory). "
          "Set USE_REAL_EMBEDDER=1 to use MiniLM instead.")

# Combined + de-duplicated expert knowledge base from both old files.
# Expand this list over time as you add more technical-analysis heuristics.
EXPERT_RULES = [
    "High RSI (>70) suggests overbought conditions; expect price reversal.",
    "Low RSI (<30) indicates oversold conditions; potential for bullish bounce.",
    "MACD bullish crossover suggests rising upward momentum.",
    "MACD bearish crossover suggests weakening upward momentum.",
    "Prices touching the Upper Bollinger Band often lead to mean-reversion pullbacks.",
    "Prices touching the Lower Bollinger Band often indicate oversold conditions.",
    "Breakouts on high volume confirm the strength of a new trend.",
    "Price above both the 50-day and 200-day SMA suggests a bullish trend regime.",
    "Price below both the 50-day and 200-day SMA suggests a bearish trend regime.",
    "Rising ATR alongside a directional move indicates increasing volatility/conviction.",
]

_index = None  # lazily built, cached module-level singleton


def build_research_index():
    """Builds (or returns the cached) FAISS index of expert rules."""
    global _index
    if _index is not None:
        return _index
    vectors = embedder.encode(EXPERT_RULES)
    _index = faiss.IndexFlatL2(vectors.shape[1])
    _index.add(np.array(vectors).astype("float32"))
    return _index


def retrieve_alpha_context(query: str, index=None, k: int = 3):
    """Retrieves the top-k rules most relevant to a headline/news string."""
    if index is None:
        index = build_research_index()
    query_vec = embedder.encode([query])
    _, indices = index.search(np.array(query_vec).astype("float32"), k)
    return [EXPERT_RULES[i] for i in indices[0]]