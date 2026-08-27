"""
market_data.py
==============
Single source of truth for price data + technical indicators.
"""

import time
import functools
import pandas as pd
import numpy as np
import ta
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# NOTE on sessions: do NOT pass a custom requests.Session() into yf.Ticker().
# ---------------------------------------------------------------------------
# Newer yfinance versions manage their own internal session using curl_cffi
# (real browser TLS/JA3 fingerprint impersonation, not just a User-Agent
# header) and explicitly reject any plain requests.Session passed in:
#   "Yahoo API requires curl_cffi session not <class
#    'requests.sessions.Session'>. Solution: stop setting session, let YF
#    handle."
# So: don't set a session at all for yf.Ticker/yf.download -- yfinance
# handles it internally. search_ticker() below still uses its own plain
# requests.Session for a totally different (non-yfinance) Yahoo search
# endpoint, unaffected by this constraint.
_SEARCH_SESSION = requests.Session()
_SEARCH_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
})

FEATURE_COLUMNS_HINT = 19  # kept for reference; model pads/truncates as needed


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds RSI, MACD, Bollinger Bands, ATR, and SMAs to a raw OHLCV
    dataframe. Used identically by training and serving so the model
    always sees the same feature definitions it was trained on."""
    if df.empty:
        return df

    df = df.copy()
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]

    df["RSI_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD_12_26_9"] = macd.macd()
    df["MACDh_12_26_9"] = macd.macd_diff()
    df["MACDs_12_26_9"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["BBL_20_2.0"] = bb.bollinger_lband()
    df["BBM_20_2.0"] = bb.bollinger_mavg()
    df["BBU_20_2.0"] = bb.bollinger_hband()
    df["BBP_20_2.0"] = bb.bollinger_pband()
    df["BBD_20_2.0"] = bb.bollinger_wband()

    df["ATRr_14"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()

    df["SMA_50"] = ta.trend.SMAIndicator(close=df["close"], window=50).sma_indicator()
    df["SMA_200"] = ta.trend.SMAIndicator(close=df["close"], window=200).sma_indicator()

    return df


def add_training_labels(df: pd.DataFrame, up_thresh: float = 0.025, down_thresh: float = -0.025) -> pd.DataFrame:
    df = df.copy()
    df["returns"] = df["close"].pct_change().shift(-1)
    df["target"] = df["returns"].apply(
        lambda r: 2 if r > up_thresh else (0 if r < down_thresh else 1)
    )
    return df


@functools.lru_cache(maxsize=64)
def search_ticker(query: str):
    """Resolve a free-text query to a Yahoo Finance ticker symbol."""
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    try:
        r = _SEARCH_SESSION.get(url, timeout=5)
        data = r.json()
        if data.get("quotes"):
            return data["quotes"][0]["symbol"]
    except Exception as e:
        print(f"[market_data] search_ticker error: {e}")
    return None


_LIVE_CACHE: dict[str, tuple[float, tuple]] = {}
_LIVE_CACHE_TTL_SECONDS = 15 * 60


def get_alpha_live_data(ticker: str):
    """Fetches real-time price data, computes indicators, and returns
    (df, kpis, resolved_ticker). Cached so repeated requests don't hammer
    Yahoo Finance, but data still refreshes periodically."""
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
        ticker_obj = yf.Ticker(resolved_ticker)
        df = _history(ticker_obj)

        if df.empty:
            found = search_ticker(ticker)
            if found and found != ticker:
                print(f"[market_data] Resolving '{ticker}' -> '{found}'")
                resolved_ticker = found
                ticker_obj = yf.Ticker(resolved_ticker)
                df = _history(ticker_obj)
    except Exception as e:
        print(f"[market_data] yfinance error for '{ticker}': {e}")
        return pd.DataFrame(), {}, ticker

    if df.empty:
        return pd.DataFrame(), {}, ticker

    try:
        df = compute_indicators(df)
    except Exception as e:
        print(f"[market_data] indicator computation error for '{resolved_ticker}': {e}")
        return pd.DataFrame(), {}, ticker

    kpis = {"Market Cap": "N/A", "52W High": "N/A"}
    for attempt in range(2):
        try:
            info = ticker_obj.info
            kpis = {
                "Market Cap": info.get("marketCap", "N/A"),
                "52W High": info.get("fiftyTwoWeekHigh", "N/A"),
            }
            break
        except Exception as e:
            print(f"[market_data] KPI lookup error for '{resolved_ticker}' "
                  f"(attempt {attempt + 1}/2): {e}")
            if attempt == 0:
                time.sleep(1.5)

    return df.dropna(), kpis, resolved_ticker


def fetch_training_data(ticker: str, start_date: str = "2020-01-01") -> pd.DataFrame:
    """Fetches historical data + indicators + labels for model training."""
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=start_date, auto_adjust=True)
    except Exception as e:
        print(f"[market_data] training fetch error for '{ticker}': {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = compute_indicators(df)
    df = add_training_labels(df)
    return df.dropna()


# ---------------------------------------------------------------------------
# Scrolling ticker tape: batch quotes for a fixed watchlist
# ---------------------------------------------------------------------------
TAPE_TICKERS = [
    "^GSPC", "^DJI", "^IXIC",           # major indices
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "V", "WMT", "DIS", "NFLX",
]

TAPE_DISPLAY_NAMES = {
    "^GSPC": "S&P 500",
    "^DJI": "DOW",
    "^IXIC": "NASDAQ",
}

_TAPE_CACHE: dict = {"data": None, "ts": 0}
_TAPE_CACHE_TTL_SECONDS = 5 * 60


def get_ticker_tape():
    """Returns a list of {symbol, display, price, change_pct} for the
    scrolling tape, batch-fetched in ONE yf.download() call (not N
    separate requests) and cached. Never raises -- on any failure it
    returns whatever was last cached (even if stale) or an empty list,
    since the tape is decorative and shouldn't be able to break the app."""
    now = time.time()
    if _TAPE_CACHE["data"] is not None and (now - _TAPE_CACHE["ts"]) < _TAPE_CACHE_TTL_SECONDS:
        return _TAPE_CACHE["data"]

    try:
        df = yf.download(
            TAPE_TICKERS, period="5d", interval="1d",
            group_by="ticker", auto_adjust=True, progress=False, threads=True,
        )

        results = []
        for sym in TAPE_TICKERS:
            try:
                closes = df[sym]["Close"].dropna()
                if len(closes) < 2:
                    continue
                last, prev = closes.iloc[-1], closes.iloc[-2]
                change_pct = ((last - prev) / prev) * 100 if prev else 0.0
                results.append({
                    "symbol": sym.lstrip("^"),
                    "display": TAPE_DISPLAY_NAMES.get(sym, sym),
                    "price": round(float(last), 2),
                    "change_pct": round(float(change_pct), 2),
                })
            except Exception as e:
                print(f"[market_data] tape: skipping '{sym}': {e}")
                continue

        if results:
            _TAPE_CACHE["data"] = results
            _TAPE_CACHE["ts"] = now
        return results if results else (_TAPE_CACHE["data"] or [])

    except Exception as e:
        print(f"[market_data] get_ticker_tape error: {e}")
        return _TAPE_CACHE["data"] or []