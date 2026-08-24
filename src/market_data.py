"""
market_data.py
==============
Single source of truth for price data + technical indicators.

Why this file exists:
The old repo computed indicators in TWO places (data_engine.py for training,
data_fetcher.py for serving) with slightly different code. That's a bug
waiting to happen: change one, forget the other, and your live model sees
different features than it was trained on. Everything now goes through
`compute_indicators()`.
"""

import time
import functools
import pandas as pd
import numpy as np
import ta
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Shared HTTP session for yfinance
# ---------------------------------------------------------------------------
# BUG FIXED: yfinance's default requests session sends a generic/no
# User-Agent, and Yahoo aggressively rate-limits traffic from cloud hosting
# IP ranges (Render, AWS, GCP, etc.) that looks non-browser-like — every
# call was failing with "Too Many Requests. Rate limited." A shared session
# with a real browser User-Agent (and keep-alive reuse instead of a fresh
# TCP/TLS handshake per call) is the standard yfinance workaround and
# meaningfully cuts down how often Yahoo blocks these requests. Not a
# guarantee — Yahoo can still rate-limit a session that calls it too
# often — but combined with the TTL cache below it's a big improvement.
_YF_SESSION = requests.Session()
_YF_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
})

# ---------------------------------------------------------------------------
# Indicator computation (shared by training + serving)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS_HINT = 19  # kept for reference; model pads/truncates to this


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds RSI, MACD, Bollinger Bands, ATR, and SMAs to a raw OHLCV dataframe.
    Used identically by both the training pipeline and the live API so the
    model always sees the same feature definitions it was trained on.
    """
    if df.empty:
        return df

    # Standardize column names (yfinance sometimes returns MultiIndex columns)
    df = df.copy()
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]

    # RSI (14)
    df["RSI_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    # MACD (12, 26, 9)
    macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD_12_26_9"] = macd.macd()
    df["MACDh_12_26_9"] = macd.macd_diff()
    df["MACDs_12_26_9"] = macd.macd_signal()

    # Bollinger Bands (20, 2)
    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["BBL_20_2.0"] = bb.bollinger_lband()
    df["BBM_20_2.0"] = bb.bollinger_mavg()
    df["BBU_20_2.0"] = bb.bollinger_hband()
    df["BBP_20_2.0"] = bb.bollinger_pband()
    df["BBD_20_2.0"] = bb.bollinger_wband()

    # ATR (14)
    df["ATRr_14"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()

    # SMA (50, 200)
    df["SMA_50"] = ta.trend.SMAIndicator(close=df["close"], window=50).sma_indicator()
    df["SMA_200"] = ta.trend.SMAIndicator(close=df["close"], window=200).sma_indicator()

    return df


def add_training_labels(df: pd.DataFrame, up_thresh: float = 0.025, down_thresh: float = -0.025) -> pd.DataFrame:
    """Adds next-day return + a 3-class Buy/Hold/Sell label. Training only."""
    df = df.copy()
    df["returns"] = df["close"].pct_change().shift(-1)
    df["target"] = df["returns"].apply(
        lambda r: 2 if r > up_thresh else (0 if r < down_thresh else 1)
    )
    return df


# ---------------------------------------------------------------------------
# Ticker search / resolution ("Bitcoin" -> "BTC-USD")
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def search_ticker(query: str):
    """Resolve a free-text query to a Yahoo Finance ticker symbol."""
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    try:
        r = _YF_SESSION.get(url, timeout=5)
        data = r.json()
        if data.get("quotes"):
            return data["quotes"][0]["symbol"]
    except Exception as e:
        print(f"[market_data] search_ticker error: {e}")
    return None


# ---------------------------------------------------------------------------
# TTL cache for live data
# ---------------------------------------------------------------------------
# The original code used @functools.lru_cache with no expiry, meaning a
# ticker's price data would never refresh for the lifetime of the process.
# This is a manual TTL cache instead: cheap, no extra dependency.

_LIVE_CACHE: dict[str, tuple[float, tuple]] = {}
_LIVE_CACHE_TTL_SECONDS = 15 * 60  # widened from 5 to 15 min — fewer Yahoo calls
                                    # overall means fewer opportunities to get
                                    # rate-limited in the first place.


def get_alpha_live_data(ticker: str):
    """
    Fetches real-time price data, computes indicators, and returns
    (df, kpis, resolved_ticker). Cached for _LIVE_CACHE_TTL_SECONDS so
    repeated requests don't hammer Yahoo Finance, but data still refreshes.
    """
    # Normalize input before it ever hits the cache or yfinance: raw API
    # callers (curl, Postman, a recruiter poking the API directly) won't
    # necessarily send clean uppercase symbols the way the frontend does.
    # Without this, " aapl " and "AAPL" would be treated as different cache
    # keys and yfinance can behave inconsistently on lowercase/padded input.
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return pd.DataFrame(), {}, ticker

    now = time.time()
    cached = _LIVE_CACHE.get(ticker)
    if cached and (now - cached[0]) < _LIVE_CACHE_TTL_SECONDS:
        return cached[1]

    result = _fetch_live_uncached(ticker)
    _LIVE_CACHE[ticker] = (now, result)
    return result


def _fetch_live_uncached(ticker: str):
    """
    Fetches one ticker via yfinance, using the shared browser-UA session
    and a short retry-with-backoff for transient rate limits — Yahoo's
    "Too Many Requests" is often momentary, and a single retry a couple
    seconds later frequently succeeds where the first call didn't.
    """
    resolved_ticker = ticker

    def _history(tk_obj, retries=2, base_delay=1.5):
        last_err = None
        for attempt in range(retries + 1):
            try:
                return tk_obj.history(period="max", interval="1d", auto_adjust=True)
            except Exception as e:
                last_err = e
                msg = str(e)
                if "Too Many Requests" in msg or "Rate limited" in msg:
                    if attempt < retries:
                        delay = base_delay * (2 ** attempt)
                        print(f"[market_data] Rate limited, retrying in {delay:.1f}s "
                              f"(attempt {attempt + 1}/{retries})...")
                        time.sleep(delay)
                        continue
                raise
        raise last_err

    try:
        ticker_obj = yf.Ticker(resolved_ticker, session=_YF_SESSION)
        df = _history(ticker_obj)

        if df.empty:
            found = search_ticker(ticker)
            if found and found != ticker:
                print(f"[market_data] Resolving '{ticker}' -> '{found}'")
                resolved_ticker = found
                ticker_obj = yf.Ticker(resolved_ticker, session=_YF_SESSION)
                df = _history(ticker_obj)
    except Exception as e:
        print(f"[market_data] yfinance error for '{ticker}': {e}")
        return pd.DataFrame(), {}, ticker

    if df.empty:
        return pd.DataFrame(), {}, ticker

    try:
        df = compute_indicators(df)
        info = ticker_obj.info
        kpis = {
            "Market Cap": info.get("marketCap", "N/A"),
            "52W High": info.get("fiftyTwoWeekHigh", "N/A"),
        }
    except Exception as e:
        print(f"[market_data] feature engineering error for '{resolved_ticker}': {e}")
        return pd.DataFrame(), {}, ticker

    return df.dropna(), kpis, resolved_ticker


# ---------------------------------------------------------------------------
# Training data fetch (deep history, with labels, no KPI lookup needed)
# ---------------------------------------------------------------------------

def fetch_training_data(ticker: str, start_date: str = "2020-01-01") -> pd.DataFrame:
    """Fetches historical data + indicators + labels for model training."""
    try:
        ticker_obj = yf.Ticker(ticker, session=_YF_SESSION)
        df = ticker_obj.history(start=start_date, auto_adjust=True)
    except Exception as e:
        print(f"[market_data] training fetch error for '{ticker}': {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = compute_indicators(df)
    df = add_training_labels(df)
    return df.dropna()