import os
from huggingface_hub import InferenceClient


class SentimentEngine:
    def __init__(self):
        """
        Finance-domain sentiment via Hugging Face's free Inference API,
        instead of loading FinBERT locally.

        CHANGED: this used to load ProsusAI/finbert in-process via
        transformers.pipeline(), which needs ~440MB+ of RAM on top of an
        already-loaded torch/transformers/sentence-transformers/faiss
        stack — enough to OOM-crash a 512MB Render free-tier instance on
        the first /analyze call (see main.py history). Hugging Face's
        Inference API runs the same FinBERT model on THEIR servers; we
        just send text over HTTP and get a label/score back. This removes
        FinBERT's memory footprint from our process entirely, at the cost
        of a network call per request (and the free tier's own rate
        limits — fine for a demo/portfolio app, not for high production
        traffic).

        Requires an HF_TOKEN environment variable (a free Hugging Face
        access token, "read" scope is enough — created at
        https://huggingface.co/settings/tokens, set as a Render
        environment variable / secret). This does NOT require a paid
        plan; that restriction only applies to hosting Spaces (Docker/
        Gradio), not to calling the Inference API.

        NOTE on model choice (unchanged from the original local-FinBERT
        version): financial headlines are full of words that read as
        negative in everyday English ("stumble", "misses", "drop",
        "volatility") even when describing routine market activity — a
        general-purpose sentiment model tends to misclassify almost
        everything as NEGATIVE regardless of actual financial meaning.
        FinBERT is trained specifically on financial text and doesn't
        have that bias, so we still use it here — just remotely instead
        of locally.
        """
        print("Initializing Sentiment Engine (FinBERT via HF Inference API)...")
        token = os.environ.get("HF_TOKEN")
        if not token:
            print(
                "WARNING: HF_TOKEN environment variable not set. "
                "Sentiment analysis will fail and fall back to NEUTRAL for "
                "every request until this is set (see Render/host env vars)."
            )
        self.client = InferenceClient(provider="hf-inference", api_key=token)
        self.model = "ProsusAI/finbert"
        print("Sentiment Engine ready (remote).")

    def analyze(self, text):
        """Returns {'label': 'POSITIVE'|'NEGATIVE'|'NEUTRAL', 'score': float}.

        BUG FIXED (carried over from the local-FinBERT version): FinBERT
        natively returns lowercase labels ('positive'/'negative'/
        'neutral'), but the frontend (SentimentPanel.jsx) only ever
        matched uppercase. Normalizing to uppercase here keeps that fix
        intact regardless of where the model actually runs.
        """
        try:
            results = self.client.text_classification(text, model=self.model)
            top = results[0]  # highest-scoring label
            return {
                "label": str(top.label).upper(),
                "score": float(top.score),
            }
        except Exception as e:
            print(f"Sentiment Error (HF Inference API): {e}")
            return {"label": "NEUTRAL", "score": 0.5}