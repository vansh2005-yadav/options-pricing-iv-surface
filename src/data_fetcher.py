"""
data_fetcher.py
---------------
Fetch real market data using yfinance:
  - spot price
  - options chain (calls + puts) for the nearest expiries
  - approximate risk-free rate (^IRX = 13-week T-bill yield)
"""

import pandas as pd
import yfinance as yf
from datetime import datetime


def get_spot_price(ticker: str) -> float:
    """
    Return the latest closing price for the given ticker.

    Parameters
    ----------
    ticker : str  e.g. 'SPY', 'AAPL'

    Returns
    -------
    float
    """
    tk = yf.Ticker(ticker)
    hist = tk.history(period="2d")
    if hist.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'. Check the symbol.")
    return float(hist["Close"].iloc[-1])


def get_risk_free_rate() -> float:
    """
    Fetch the 13-week US T-bill yield (^IRX) as a proxy for the risk-free rate.
    Returns rate as a decimal (e.g. 0.053 for 5.3 %).
    Falls back to 0.05 if the fetch fails.
    """
    try:
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        if hist.empty:
            print("[WARNING] Could not fetch ^IRX. Using fallback rate 0.05.")
            return 0.05
        rate_pct = float(hist["Close"].iloc[-1])
        return rate_pct / 100.0
    except Exception as e:
        print(f"[WARNING] Risk-free rate fetch failed ({e}). Using fallback 0.05.")
        return 0.05


def get_options_chain(ticker: str, num_expiries: int = 3) -> pd.DataFrame:
    """
    Download the options chain for the nearest `num_expiries` expiration dates.

    Returns a cleaned DataFrame with columns:
        expiry, strike, option_type, lastPrice, bid, ask,
        impliedVolatility (yfinance native), volume, openInterest,
        mid_price, T (years to expiry)

    Parameters
    ----------
    ticker       : str  – underlying ticker
    num_expiries : int  – how many expiry dates to pull

    Returns
    -------
    pd.DataFrame
    """
    tk = yf.Ticker(ticker)
    expirations = tk.options          # tuple of date strings 'YYYY-MM-DD'

    if not expirations:
        raise ValueError(f"No options data available for '{ticker}'.")

    today = datetime.today().date()
    selected = expirations[:num_expiries]

    frames = []
    for exp_str in selected:
        try:
            chain = tk.option_chain(exp_str)

            calls = chain.calls.copy()
            calls["option_type"] = "call"

            puts  = chain.puts.copy()
            puts["option_type"]  = "put"

            combined = pd.concat([calls, puts], ignore_index=True)
            combined["expiry"] = exp_str

            # Time to expiry in years (calendar days / 365)
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            days_to_exp = (exp_date - today).days
            combined["T"] = days_to_exp / 365.0

            frames.append(combined)

        except Exception as e:
            print(f"[WARNING] Could not fetch chain for {exp_str}: {e}")

    if not frames:
        raise RuntimeError("Failed to fetch any options chain data.")

    df = pd.concat(frames, ignore_index=True)

    # Mid price = average of bid and ask (more reliable than lastPrice)
    df["mid_price"] = (df["bid"] + df["ask"]) / 2.0

    # Keep only the columns we need
    keep_cols = [
        "expiry", "strike", "option_type",
        "lastPrice", "bid", "ask", "mid_price",
        "impliedVolatility", "volume", "openInterest", "T"
    ]
    # Only keep columns that actually exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Basic cleaning
    df = df[df["T"] > 0]                      # drop expired
    df = df[df["mid_price"] > 0]              # drop zero-price rows
    df = df[df["strike"] > 0]
    df.dropna(subset=["strike", "mid_price", "T"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df