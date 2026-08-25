from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import torch
import joblib
import json
import time
import traceback
import numpy as np

# backend/main.py lives inside backend/, but src/, models/, and metrics.json
# all live at the PROJECT ROOT (one level up from backend/) — not inside
# backend/. Add the project root to sys.path so `from src...` imports work
# regardless of which directory uvicorn is launched from.
#
# CORRECTION: an earlier version of this comment/fix incorrectly assumed
# src/models/metrics.json lived INSIDE backend/ and changed PROJECT_ROOT to
# CURRENT_DIR — that broke `from src.market_data import ...` entirely
# (ModuleNotFoundError: No module named 'src'), because src/ actually does
# live at the project root, one level above backend/, as originally
# written. Reverted to the correct, original logic below.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../backend
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                       # project root (one level up)
sys.path.append(PROJECT_ROOT)

from src.market_data import get_alpha_live_data, search_ticker
from src.rag_engine import build_research_index, retrieve_alpha_context, embedder
from src.model_engine import SentimetrixTCN, predict_signal
from src.sentiment_engine import SentimentEngine
from src.news_engine import NewsEngine

app = FastAPI(title="Sentimetrix-TCN API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ---------------------------------------------------------------------------
# Result cache — keeps repeated requests for the SAME ticker deterministic.
# ---------------------------------------------------------------------------
# BUG FIXED: previously /news hit live RSS feeds and /analyze recomputed
# sentiment + the model's signal on every single click. Both news and
# sentiment models are technically deterministic given identical input text,
# but the INPUT (live news headlines) changes underneath you between calls,
# so the same ticker could show a different signal/sentiment/news 2 seconds
# apart. From a user's perspective that reads as "this app is broken."
#
# Fix: cache the full response per ticker for a TTL window. Within that
# window, clicking "Run Analysis" N times returns the exact same JSON —
# same news, same sentiment, same signal, same confidence. After the TTL
# expires, the next click fetches a fresh snapshot (so it still feels live,
# just not jittery on every click).
RESULT_CACHE_TTL_SECONDS = 10 * 60  # 10 minutes
_news_cache: dict[str, tuple[float, dict]] = {}
_analyze_cache: dict[str, tuple[float, dict]] = {}


def _cache_get(cache: dict, key: str):
    entry = cache.get(key)
    if entry and (time.time() - entry[0]) < RESULT_CACHE_TTL_SECONDS:
        return entry[1]
    return None


def _cache_set(cache: dict, key: str, value: dict):
    cache[key] = (time.time(), value)
    return value

# ---------------------------------------------------------------------------
# Lightweight assets: load at startup (cheap, needed for almost every request)
# ---------------------------------------------------------------------------
model = None
scaler = None
faiss_index = None


@app.on_event("startup")
def load_lightweight_assets():
    """
    Only the cheap assets load at startup: the TCN weights, the scaler,
    and the FAISS rule index. The heavy Transformer models (LLM + sentiment)
    are loaded lazily on first use (see get_sentiment_engine()
    below) so the server can start accepting requests immediately instead
    of stalling (or OOM-ing on low-memory hosts like Render's free tier)
    while every model loads up front.
    """
    global model, scaler, faiss_index
    try:
        print("Loading lightweight Sentimetrix assets...")

        # BUG FIXED: this used to hardcode SentimetrixTCN(input_size=19), a
        # leftover from before the real training pipeline existed. The
        # actual trained checkpoint (from src/train.py) has 17 input
        # features -- whatever compute_indicators() genuinely produces,
        # minus target/returns -- not a fixed 19. That mismatch caused a
        # RuntimeError on every startup, silently swallowed by the except
        # block below, leaving `model` as None -> every /analyze call then
        # failed. Fixed by reading the real feature count from
        # metrics.json (written by train.py) so the model always matches
        # whatever checkpoint is actually on disk.
        feature_dim = 19  # fallback only, used if metrics.json is missing/stale
        metrics_path = os.path.join(PROJECT_ROOT, "metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    saved_dim = json.load(f).get("feature_dim")
                if saved_dim:
                    feature_dim = int(saved_dim)
                    print(f"Using feature_dim={feature_dim} from metrics.json")
            except Exception as e:
                print(f"Could not read feature_dim from metrics.json ({e}); falling back to {feature_dim}")

        model = SentimetrixTCN(input_size=feature_dim)
        weights_path = os.path.join(MODELS_DIR, "alpha_weights.pth")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")

            # FIX: the old code unconditionally dropped tcn.0.weight/bias
            # before loading, which silently left the model's FIRST conv
            # layer at random initialization on every single run — even
            # when the checkpoint's shape matched perfectly. That's almost
            # certainly why confidence was hovering near chance level
            # (~33-40% on a 3-class problem). Now we only drop it if the
            # shape genuinely doesn't match the current model definition.
            own_state = model.state_dict()
            for key in ("tcn.0.weight", "tcn.0.bias"):
                if key in state_dict and key in own_state and state_dict[key].shape != own_state[key].shape:
                    print(f"Shape mismatch on '{key}': checkpoint {state_dict[key].shape} "
                          f"vs model {own_state[key].shape} — dropping this key only.")
                    state_dict.pop(key)

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Warning: missing keys not loaded from checkpoint: {missing}")
            print("Model weights loaded.")
        else:
            print("No trained weights found. Using randomly initialized model.")
        model.eval()

        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("Scaler loaded.")
        else:
            print("No scaler found. Predictions will be less reliable.")

        faiss_index = build_research_index()
        print("RAG index ready.")
        print("Lightweight assets loaded. Server is ready to accept requests.")

    except Exception as e:
        print("STARTUP ERROR:", e)
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Heavy assets: lazy singletons, loaded on first use
# ---------------------------------------------------------------------------
# AI ANALYST REMOVED: distilgpt2 (via IntelligenceEngine) added real memory
# weight on Render's free 512MB instance for a feature that wasn't
# essential -- cutting it gives FinBERT (still used by /analyze) more
# headroom. If /analyze still OOMs after this, FinBERT itself is the next
# thing to address (e.g. swap for a smaller sentiment model).
_sentiment_engine = None
_news_engine = None


def get_sentiment_engine() -> SentimentEngine:
    global _sentiment_engine
    if _sentiment_engine is None:
        _sentiment_engine = SentimentEngine()
    return _sentiment_engine


def get_news_engine() -> NewsEngine:
    global _news_engine
    if _news_engine is None:
        _news_engine = NewsEngine()
    return _news_engine


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class AnalysisRequest(BaseModel):
    ticker: str
    news_context: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "online", "system": "Sentimetrix-TCN", "docs": "/health for component status"}


@app.get("/health")
def health_check():
    """Reports the actual load state of every component, not just 'the
    process is running.' The old health_check always said {'status':
    'online'} even if the model failed to load, the scaler was missing, or
    the RAG index never built — completely useless for diagnosing a broken
    deploy. This tells you exactly what's wired up and what isn't."""
    components = {
        "tcn_model": model is not None,
        "scaler": scaler is not None,
        "rag_index": faiss_index is not None,
        "sentiment_loaded": _sentiment_engine is not None,  # lazy — False until first /analyze call
        "news_engine_loaded": _news_engine is not None,
    }
    core_ready = components["tcn_model"] and components["rag_index"]
    return {
        "status": "healthy" if core_ready else "degraded",
        "components": components,
        "cache_entries": {"news": len(_news_cache), "analyze": len(_analyze_cache)},
    }


@app.get("/market-data/{ticker}")
def get_market_data(ticker: str):
    try:
        df, kpis, resolved_ticker = get_alpha_live_data(ticker)

        if df.empty:
            print(f"Ticker '{ticker}' failed, falling back to AAPL")
            df, kpis, resolved_ticker = get_alpha_live_data("AAPL")

        if df.empty:
            raise HTTPException(status_code=404, detail="Market data not available.")

        df_history = df.tail(100).reset_index()
        df_history.rename(columns={"index": "date", "Date": "date"}, inplace=True)
        # BUG FIXED: `date` was a raw pandas Timestamp, which serializes to
        # a full ISO string with time-of-day and timezone offset
        # ("2026-04-21T00:00:00-04:00") — unreadable as an X-axis label on
        # a daily chart where the time component is always midnight anyway.
        # Format as a clean date string once, here, so every consumer
        # (chart axis, tooltip) gets the same readable value.
        df_history["date"] = df_history["date"].dt.strftime("%Y-%m-%d")
        history = df_history.to_dict(orient="records")

        return {
            "ticker": resolved_ticker,
            "original_ticker": ticker,
            "kpis": kpis,
            "history": history,
        }
    except HTTPException:
        raise
    except Exception as e:
        print("MARKET DATA ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/news/{ticker}")
def get_news(ticker: str):
    cache_key = ticker.upper().strip()

    cached = _cache_get(_news_cache, cache_key)
    if cached is not None:
        return cached

    try:
        resolved = search_ticker(ticker)
        query_ticker = resolved if resolved else ticker

        context, articles = get_news_engine().fetch_company_news(query_ticker)

        if not context:
            context = "Market volatility observed in the sector."
            articles = []

        result = {"context": context, "articles": articles}
        return _cache_set(_news_cache, cache_key, result)
    except Exception as e:
        print("NEWS ERROR:", e)
        fallback = {"context": "Market volatility observed in the sector.", "articles": []}
        return _cache_set(_news_cache, cache_key, fallback)


@app.post("/analyze")
def analyze_stock(req: AnalysisRequest):
    # Cache key includes the news context too: if the user manually edits
    # the "Analysis Context" box and re-runs, that's a deliberate new input
    # and SHOULD produce a new result. If they didn't touch it, the context
    # they send is itself already cache-stable (see /news above), so the
    # key collapses to the same value on repeated clicks -> same result.
    cache_key = f"{req.ticker.upper().strip()}::{(req.news_context or '').strip()}"

    cached = _cache_get(_analyze_cache, cache_key)
    if cached is not None:
        return cached

    try:
        print("Analyze request:", req.ticker)

        df, _, resolved_ticker = get_alpha_live_data(req.ticker)

        if df.empty:
            print(f"Ticker '{req.ticker}' failed, falling back to AAPL")
            df, _, resolved_ticker = get_alpha_live_data("AAPL")

        if df.empty:
            raise HTTPException(status_code=404, detail="Ticker data not found.")

        news_context = req.news_context
        if not news_context:
            news_context, _ = get_news_engine().fetch_company_news(resolved_ticker)
        if not news_context:
            news_context = "Market volatility observed."

        rules = retrieve_alpha_context(news_context, faiss_index)

        if scaler is None:
            raise HTTPException(status_code=500, detail="Scaler not loaded")

        pred_class, conf, weights = predict_signal(model, df, rules, embedder, scaler)
        sentiment = get_sentiment_engine().analyze(news_context)

        # BUG FIXED: the model's attention weights over the retrieved rules
        # were computed but never included in the response, so the
        # frontend had no way to show *which* rules the model actually
        # weighted most heavily — the single most useful piece of
        # explainability this architecture produces was being thrown away.
        weights_list = np.atleast_1d(weights).tolist() if weights is not None else []

        result = {
            "ticker": resolved_ticker,
            "signal_class": pred_class,
            "confidence": conf,
            "rules_triggered": rules,
            "attention_weights": weights_list,
            "sentiment": sentiment,
            "news_context_used": news_context,
        }
        return _cache_set(_analyze_cache, cache_key, result)
    except HTTPException:
        raise
    except Exception as e:
        print("ANALYZE ERROR:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    metrics = {"accuracy": None, "precision": None, "recall": None, "f1": None}
    metrics_path = os.path.join(PROJECT_ROOT, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    return metrics