"""
intelligence_engine.py
=======================
Template-based response generator — replaces the previous distilgpt2-based
IntelligenceEngine.

WHY: distilgpt2, loaded via transformers' `pipeline()`, was one of several
transformer models resident in memory at once on a Render free-tier
instance with a 512MB ceiling (alongside FinBERT, the MiniLM RAG embedder,
and the TCN model). Loading it — even lazily, on first /chat call — was
part of what pushed the process over the memory limit and got it
OOM-killed mid-request.

This isn't purely a memory trade-off, either: distilgpt2's actual output
quality here was already weak (an 82M-parameter base model with no
fine-tuning, wrapped in enough prompt-engineering to keep it on topic —
see the old generate_analysis()'s "Robust Extraction" and "Final Safety
Check" fallback logic, which existed specifically because raw generation
frequently produced unusable text). A deterministic template generator is
faster, uses no extra memory, and always returns a coherent, on-topic
sentence referencing the real signal/ticker/rule — which is what the old
fallback path was already doing whenever generation misbehaved anyway.

The public interface (constructor + generate_analysis()) is unchanged
from the distilgpt2 version, so main.py needs no changes. If you upgrade
off the free tier and want real generative text again, this class is the
only thing that needs to change back.
"""

import random


class IntelligenceEngine:
    def __init__(self):
        print("Intelligence Engine ready (template-based, no model load).")

    def generate_analysis(self, context_rules, technical_signal, ticker, user_query):
        """
        Builds a natural-language response from the RAG context + technical
        signal, without any model inference.
        """
        signal_map = {0: "Sell", 1: "Hold", 2: "Buy"}
        signal_text = signal_map.get(technical_signal, "Hold")

        rules = context_rules if isinstance(context_rules, list) else [context_rules]
        rules = [r for r in rules if r]
        rule_line = rules[0] if rules else None

        openers = {
            "Sell": [
                f"The current technical signal for {ticker} is SELL.",
                f"{ticker}'s indicators are leaning bearish right now.",
            ],
            "Buy": [
                f"The current technical signal for {ticker} is BUY.",
                f"{ticker}'s indicators are leaning bullish right now.",
            ],
            "Hold": [
                f"The current technical signal for {ticker} is HOLD.",
                f"{ticker}'s indicators are mixed / range-bound right now.",
            ],
        }
        opener = random.choice(openers[signal_text])

        parts = [opener]
        if rule_line:
            parts.append(f'This lines up with the model\'s top-weighted rule: "{rule_line}"')

        query = (user_query or "").strip()
        if query:
            parts.append(
                f'On your question — "{query}" — treat this signal as one input among '
                f"several; confirm against the RSI/MACD panels before acting on it."
            )
        else:
            parts.append(
                "Treat this signal as one input among several; confirm against the "
                "RSI/MACD panels before acting on it."
            )

        return " ".join(parts)