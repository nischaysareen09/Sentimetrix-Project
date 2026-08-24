import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentEngine:
    def __init__(self):
        """
        Initializes a finance-domain sentiment model (FinBERT), loaded via
        direct AutoTokenizer/AutoModelForSequenceClassification classes
        instead of transformers' `pipeline("sentiment-analysis", ...)`
        wrapper.

        WHY NOT `pipeline()`: it's a convenience wrapper that keeps extra
        Python-level bookkeeping and preprocessing/postprocessing machinery
        alive for the object's lifetime on top of the model weights
        themselves. On a memory-constrained host (Render free tier,
        512MB) that overhead — stacked on top of the TCN model, FAISS
        index, and everything else already resident — was enough to push
        the process over the limit and get it OOM-killed mid-request (see:
        /analyze crash-loop, Aug 2026). Loading the tokenizer + model
        directly and running inference in a plain torch.no_grad() block
        does the identical computation with a smaller resident footprint.

        NOTE: FinBERT is trained specifically on financial text (analyst
        reports, earnings calls), unlike a generic SST-2 sentiment model
        which tends to misread routine financial vocabulary ("stumble",
        "misses", "drop") as broadly negative. That reasoning is unchanged
        from the previous version of this file — only the loading
        mechanism changed, not the model choice.
        """
        print("Loading Sentiment Engine (FinBERT, direct model load — no pipeline)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "ProsusAI/finbert"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # fp16 weights: FinBERT is ~440MB in fp32, ~220MB in fp16. CPU
        # inference in fp16 is slightly slower per-call than fp32, but on
        # a 512MB instance the memory headroom matters far more than a few
        # extra milliseconds per request.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        self.model.to(self.device)
        self.model.eval()

        # FinBERT's own config already maps id -> label
        # ('positive'/'negative'/'neutral'), so no need to hardcode it here.
        self.id2label = self.model.config.id2label
        print("Sentiment Engine Loaded.")

    def analyze(self, text):
        """Returns {'label': 'POSITIVE'|'NEGATIVE'|'NEUTRAL', 'score': float}.

        BUG FIX CARRIED OVER: FinBERT natively returns lowercase labels
        ('positive'/'negative'/'neutral'), but the frontend
        (SentimentPanel.jsx) only ever matches uppercase. Normalizing to
        uppercase here keeps that contract unambiguous for any consumer.
        """
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            # Softmax in fp32 regardless of model dtype — fp16 softmax can
            # lose precision right at the decision boundary between classes.
            probs = torch.softmax(logits.float(), dim=-1)[0]
            pred_id = int(torch.argmax(probs).item())
            label = self.id2label[pred_id]
            score = float(probs[pred_id].item())

            return {"label": str(label).upper(), "score": score}
        except Exception as e:
            print(f"Sentiment Error: {e}")
            return {"label": "NEUTRAL", "score": 0.5}