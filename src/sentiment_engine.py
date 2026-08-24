from transformers import pipeline
import torch

class SentimentEngine:
    def __init__(self):
        """
        Initializes a finance-domain sentiment model.

        NOTE: the original version used
        'distilbert-base-uncased-finetuned-sst-2-english', which is trained
        on movie reviews (SST-2). Financial headlines are full of words that
        read as negative in everyday English ("stumble", "misses", "drop",
        "volatility") even when describing routine market activity — so a
        general-purpose model tends to classify almost everything as
        NEGATIVE with high confidence, regardless of which ticker it's
        actually about. FinBERT is trained specifically on financial text
        (analyst reports, earnings calls) and doesn't have that bias.
        """
        print("Loading Sentiment Engine (FinBERT)...")
        device = 0 if torch.cuda.is_available() else -1
        self.analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=device,
            truncation=True,   # avoid crashes on longer concatenated headlines
        )
        print("Sentiment Engine Loaded.")

    def analyze(self, text):
        """Returns {'label': 'POSITIVE'|'NEGATIVE'|'NEUTRAL', 'score': float}.

        BUG FIXED: FinBERT (ProsusAI/finbert) natively returns lowercase
        labels ('positive'/'negative'/'neutral'), but the frontend
        (SentimentPanel.jsx) only ever matched uppercase ('POSITIVE'/
        'NEGATIVE'). That mismatch meant every real sentiment result
        silently fell through to the neutral/gray UI branch regardless of
        what the model actually predicted — always showing "NEUTRAL"
        no matter the news. Normalizing to uppercase here makes the API
        contract unambiguous for any consumer, not just a frontend patch.
        """
        try:
            result = self.analyzer(text)[0]
            return {
                "label": str(result.get("label", "neutral")).upper(),
                "score": float(result.get("score", 0.5)),
            }
        except Exception as e:
            print(f"Sentiment Error: {e}")
            return {"label": "NEUTRAL", "score": 0.5}