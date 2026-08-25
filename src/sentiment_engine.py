import os
import time
import requests


class SentimentEngine:
    def __init__(self):
        """
        Finance-domain sentiment via Hugging Face's free Inference API,
        instead of loading FinBERT locally.

        CHANGED (again): the first version of this fix used
        huggingface_hub's InferenceClient(provider="hf-inference", ...),
        but the huggingface_hub version actually resolved/installed by pip
        (pinned indirectly by sentence-transformers/transformers'
        dependency ranges) predates the `provider` argument, causing:
        TypeError: InferenceClient.__init__() got an unexpected keyword
        argument 'provider' — crashing every /analyze call.
        Sidestepped entirely by calling the HTTP inference endpoint
        directly with `requests` (a dependency with no version coupling
        to huggingface_hub at all), instead of going through the SDK.

        This still removes FinBERT's ~440MB local memory footprint (the
        original problem — see main.py history) since inference runs on
        Hugging Face's servers, not this process.

        Requires an HF_TOKEN environment variable (a free Hugging Face
        access token, "read" scope — created at
        https://huggingface.co/settings/tokens, set as a Render
        environment variable). This does NOT require a paid HF plan;
        that restriction only applies to hosting Spaces (Docker/Gradio),
        not to calling the Inference API.
        """
        print("Initializing Sentiment Engine (FinBERT via HF Inference API)...")
        self.token = os.environ.get("HF_TOKEN")
        if not self.token:
            print(
                "WARNING: HF_TOKEN environment variable not set. "
                "Sentiment analysis will fail and fall back to NEUTRAL for "
                "every request until this is set (see Render env vars)."
            )
        self.api_url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
        print("Sentiment Engine ready (remote).")

    def analyze(self, text):
        """Returns {'label': 'POSITIVE'|'NEGATIVE'|'NEUTRAL', 'score': float}.

        BUG FIXED (carried over from the local-FinBERT version): FinBERT
        natively returns lowercase labels ('positive'/'negative'/
        'neutral'), but the frontend (SentimentPanel.jsx) only ever
        matched uppercase. Normalizing to uppercase here keeps that fix
        intact regardless of where the model actually runs.
        """
        if not self.token:
            return {"label": "NEUTRAL", "score": 0.5}

        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {"inputs": text}

        for attempt in range(2):
            try:
                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=15)
                data = resp.json()

                # A cold model on HF's side returns {"error": "...", "estimated_time": N}
                # with a 503 while it loads — worth one short retry.
                if isinstance(data, dict) and "error" in data:
                    wait = data.get("estimated_time", 3)
                    if attempt == 0:
                        print(f"[sentiment] Model loading on HF's side, retrying in {wait:.1f}s...")
                        time.sleep(min(wait, 10))
                        continue
                    raise RuntimeError(data["error"])

                # Successful response shape: [[{"label": ..., "score": ...}, ...]]
                # (a batch of 1 input -> list of per-class scores, sorted
                # highest first).
                scores = data[0] if isinstance(data, list) else data
                top = scores[0]
                return {
                    "label": str(top["label"]).upper(),
                    "score": float(top["score"]),
                }
            except Exception as e:
                print(f"Sentiment Error (HF Inference API, attempt {attempt + 1}/2): {e}")

        return {"label": "NEUTRAL", "score": 0.5}