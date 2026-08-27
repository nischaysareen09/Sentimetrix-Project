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
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../backend
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                       # project root (one level up)
sys.path.append(PROJECT_ROOT)

from src.market_data import get_alpha_live_data, search_ticker, get_ticker_tape
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

model = None
scaler = None
faiss_index = None


@app.on_event("startup")
def load_lightweight_assets():
    """
    Only the cheap assets load at startup: the TCN weights, the scaler,
    and the FAISS rule index. Sentiment analysis now happens via a remote
    HF Inference API call (see src/sentiment_engine.py) instead of loading
    FinBERT locally, so there's no heavy local sentiment model to warm up
    at all anymore -- get_sentiment_engine() below is effectively free.
    """
    global model, scaler, faiss_index
    try:
        print("Loading lightweight Sentimetrix assets...")

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


class AnalysisRequest(BaseModel):
    ticker: str
    news_context: str | None = None


@app.get("/")
def root():
    return {"status": "online", "system": "Sentimetrix-TCN", "docs": "/health for component status"}


@app.get("/health")
def health_check():
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


@app.get("/tape")
def tape():
    """Lightweight batch quotes for the scrolling ticker tape. Cheap and
    cached independently of /market-data -- never blocks or fails the
    rest of the app since get_ticker_tape() never raises."""
    return {"tape": get_ticker_tape()}


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