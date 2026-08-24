"""
offline_embedder.py
====================
Fallback for when the real sentence-transformers model can't be downloaded
(no network path to huggingface.co — e.g. this sandbox). NOT used in
production: rag_engine.py always tries the real MiniLM embedder first and
only falls back to this if that download fails.

This is a deterministic, seeded, hashing-based text embedder: each text is
tokenized, each token deterministically maps (via a stable hash) to a
random unit vector, and the text embedding is the normalized average of its
token vectors. It's not semantically meaningful the way a trained
transformer is — it's here purely so architecture/pipeline code can be
built, run, and validated end-to-end without internet access. Weights
trained against it are a proxy, not the final artifact — see the big
warning this module prints on import, and re-run training in an
environment with real internet access (train_real_data.py) before treating
results as final.
"""

import hashlib
import re
import numpy as np

EMBED_DIM = 384


def _token_vector(token: str) -> np.ndarray:
    seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=EMBED_DIM)
    return v / (np.linalg.norm(v) + 1e-8)


class OfflineHashEmbedder:
    """Drop-in stand-in for sentence_transformers.SentenceTransformer,
    exposing the same `.encode(list_of_str) -> np.ndarray` interface."""

    def __init__(self):
        print(
            "\n"
            "⚠️  OFFLINE FALLBACK EMBEDDER ACTIVE — could not reach huggingface.co "
            "to download the real MiniLM model.\n"
            "    Rule/query embeddings are a deterministic hash-based proxy, NOT "
            "real semantic embeddings.\n"
            "    This is fine for validating the pipeline end-to-end, but re-run "
            "training with real internet access\n"
            "    (train_real_data.py) before treating results as final.\n"
        )

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            if not tokens:
                tokens = ["<empty>"]
            vecs = np.stack([_token_vector(t) for t in tokens])
            v = vecs.mean(axis=0)
            out[i] = v / (np.linalg.norm(v) + 1e-8)
        return out
