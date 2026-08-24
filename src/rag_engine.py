"""
rag_engine.py
=============
Single RAG module (replaces the old duplicated rag_engine.py + rag_retriever.py,
which held two different, overlapping rule lists — only one of which was
actually wired into the app). This is now the only source of expert rules.
"""

import faiss
import numpy as np

# Prefer the real sentence-transformers MiniLM model (used in production,
# where huggingface.co is reachable). This sandbox has no route to
# huggingface.co, so fall back to a deterministic offline embedder purely
# so the training/eval pipeline can be built and validated end-to-end.
# See offline_embedder.py for details — re-run training with real network
# access (train_real_data.py) before treating results as final.
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"[rag_engine] Could not load real MiniLM embedder ({e}); using offline fallback.")
    try:
        from .offline_embedder import OfflineHashEmbedder
    except ImportError:
        from offline_embedder import OfflineHashEmbedder
    embedder = OfflineHashEmbedder()

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