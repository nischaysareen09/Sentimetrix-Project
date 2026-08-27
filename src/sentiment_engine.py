"""
sentiment_engine.py
====================
Calls Hugging Face's hosted Inference API for FinBERT sentiment scoring
instead of loading the model locally via `transformers.pipeline`.

Why this exists: FinBERT (ProsusAI/finbert, ~440MB in memory once loaded)
was a major contributor to the OOM crashes on Render's free 512MB
instance -- alongside torch, sentence-transformers, faiss, and the TCN
model all coexisting in the same process. Moving inference to HF's
servers means FinBERT's weights never load into this process's memory at
all; only a small HTTP request/response crosses the wire.

No huggingface_hub SDK dependency here -- plain `requests` avoids
version-mismatch issues with that library's `provider=` argument across
versions, and this API is simple enough not to need an SDK wrapper.
"""

import os
import time
import requests

HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # optional but recommended -- see .env.example

# HF's free Inference API can "cold start" a model that hasn't been
# called recently: the first request gets a 503 with an estimated wait
# time instead of a result. Retry a couple of times with that wait
# baked in rather than failing immediately.
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 20


class SentimentEngine:
    def __init__(self):
        # Nothing to load -- inference happens remotely. This constructor
        # exists so main.py's lazy-singleton pattern (get_sentiment_engine())
        # still works unchanged; there's just no heavy model to warm up.
        if not HF_TOKEN:
            print(
                "[sentiment_engine] Warning: HF_TOKEN is not set. The public "
                "Inference API works without a token for light use, but is "
                "aggressively rate-limited and models may take longer to "
                "cold-start. Set HF_TOKEN in your environment for reliable "
                "use (see .env.example)."
            )
        print("Sentiment Engine ready (remote HF Inference API, FinBERT).")

    def analyze(self, text: str) -> dict:
        """Returns {'label': 'POSITIVE'|'NEGATIVE'|'NEUTRAL', 'score': float}.

        Always returns this shape, even on total failure -- callers should
        never need to handle an exception from this method.
        """
        if not text or not text.strip():
            return {"label": "NEUTRAL", "score": 0.5}

        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        # FinBERT's HF widget truncates long inputs anyway; keep the
        # payload small and fast rather than sending an unbounded string.
        payload = {"inputs": text[:1000]}

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    HF_API_URL, headers=headers, json=payload,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )

                if resp.status_code == 503:
                    # Model is cold-starting on HF's side. The response
                    # body tells us roughly how long to wait.
                    try:
                        wait_s = float(resp.json().get("estimated_time", 3))
                    except Exception:
                        wait_s = 3
                    wait_s = min(wait_s, 15)  # don't block a request forever
                    print(f"[sentiment_engine] Model cold-starting on HF, "
                          f"waiting {wait_s:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait_s)
                    continue

                resp.raise_for_status()
                data = resp.json()
                return self._parse_response(data)

            except Exception as e:
                last_error = e
                print(f"[sentiment_engine] Request failed (attempt "
                      f"{attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1.5 * (attempt + 1))

        print(f"[sentiment_engine] All retries exhausted, falling back to "
              f"NEUTRAL. Last error: {last_error}")
        return {"label": "NEUTRAL", "score": 0.5}

    @staticmethod
    def _parse_response(data) -> dict:
        """HF's text-classification Inference API returns a nested list:
        [[{"label": "positive", "score": 0.87}, {"label": "neutral", ...}, ...]]
        -- a list of predictions for each input, each itself a list of all
        class scores sorted descending. We sent one input, so we want
        data[0][0] (the top-scoring class for that single input). Some HF
        API versions/errors return a flat list instead
        ([{"label":...}, ...]) or a dict with an 'error' key -- handle all
        three shapes defensively rather than assuming one."""
        try:
            if isinstance(data, dict) and "error" in data:
                raise ValueError(f"HF API error: {data['error']}")

            if isinstance(data, list) and len(data) > 0:
                first = data[0]
                if isinstance(first, list) and len(first) > 0:
                    top = first[0]  # nested shape: [[{...}, {...}]]
                elif isinstance(first, dict):
                    top = first  # flat shape: [{...}, {...}]
                else:
                    raise ValueError(f"Unexpected response shape: {data!r}")

                return {
                    "label": str(top.get("label", "neutral")).upper(),
                    "score": float(top.get("score", 0.5)),
                }

            raise ValueError(f"Empty or unrecognized response: {data!r}")

        except Exception as e:
            print(f"[sentiment_engine] Failed to parse HF response ({e}); "
                  f"raw response: {data!r}")
            return {"label": "NEUTRAL", "score": 0.5}