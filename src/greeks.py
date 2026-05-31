"""
greeks.py  (NEW in v2)
----------------------
Analytical Black-Scholes Greeks for European options:

    Delta  – sensitivity to spot price
    Gamma  – rate of change of delta
    Vega   – sensitivity to volatility  (per 1% move in vol)
    Theta  – time decay                 (per calendar day)
    Rho    – sensitivity to interest rate (per 1% move in rate)

All Greeks are computed analytically (closed-form), not numerically.
The greeks_summary() function returns a clean dict for printing or DataFrames.
"""

import numpy as np
from scipy.stats import norm
from src.black_scholes import d1, d2, _validate_inputs


# ──────────────────────────────────────────────────────────
# Individual Greek functions
# ──────────────────────────────────────────────────────────

def delta(S, K, T, r, sigma, option_type="call"):
    """
    Delta: ∂V/∂S
    Call delta ∈ (0, 1)   |   Put delta ∈ (-1, 0)
    """
    _validate_inputs(S, K, T, r, sigma)
    _d1 = d1(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return float(norm.cdf(_d1))
    else:
        return float(norm.cdf(_d1) - 1.0)


def gamma(S, K, T, r, sigma):
    """
    Gamma: ∂²V/∂S²  (same for calls and puts)
    Always positive. Highest near ATM, near expiry.
    """
    _validate_inputs(S, K, T, r, sigma)
    _d1 = d1(S, K, T, r, sigma)
    return float(norm.pdf(_d1) / (S * sigma * np.sqrt(T)))


def vega(S, K, T, r, sigma):
    """
    Vega: ∂V/∂σ  (same for calls and puts)
    Returned per 1% change in volatility (divide raw by 100).
    """
    _validate_inputs(S, K, T, r, sigma)
    _d1 = d1(S, K, T, r, sigma)
    raw_vega = S * norm.pdf(_d1) * np.sqrt(T)
    return float(raw_vega / 100.0)   # per 1% vol move


def theta(S, K, T, r, sigma, option_type="call"):
    """
    Theta: ∂V/∂T  (time decay per calendar day)
    Almost always negative — the option loses value as time passes.
    Divided by 365 to express as per-day decay.
    """
    _validate_inputs(S, K, T, r, sigma)
    _d1 = d1(S, K, T, r, sigma)
    _d2 = _d1 - sigma * np.sqrt(T)

    common_term = -(S * norm.pdf(_d1) * sigma) / (2.0 * np.sqrt(T))

    if option_type.lower() == "call":
        raw = common_term - r * K * np.exp(-r * T) * norm.cdf(_d2)
    else:
        raw = common_term + r * K * np.exp(-r * T) * norm.cdf(-_d2)

    return float(raw / 365.0)   # per calendar day


def rho(S, K, T, r, sigma, option_type="call"):
    """
    Rho: ∂V/∂r  (per 1% change in interest rate)
    Calls have positive rho; puts have negative rho.
    """
    _validate_inputs(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)

    if option_type.lower() == "call":
        raw = K * T * np.exp(-r * T) * norm.cdf(_d2)
    else:
        raw = -K * T * np.exp(-r * T) * norm.cdf(-_d2)

    return float(raw / 100.0)   # per 1% rate move


# ──────────────────────────────────────────────────────────
# Convenience: compute all Greeks at once
# ──────────────────────────────────────────────────────────

def greeks_summary(S, K, T, r, sigma, option_type="call"):
    """
    Return all five Greeks in a single dict.

    Returns
    -------
    dict with keys: delta, gamma, vega, theta, rho
    """
    return {
        "delta": delta(S, K, T, r, sigma, option_type),
        "gamma": gamma(S, K, T, r, sigma),
        "vega" : vega(S, K, T, r, sigma),
        "theta": theta(S, K, T, r, sigma, option_type),
        "rho"  : rho(S, K, T, r, sigma, option_type),
    }


def greeks_for_dataframe(df, spot, r):
    """
    Vectorised Greek computation for every row in the options DataFrame.
    Adds columns: delta, gamma, vega, theta, rho.

    Parameters
    ----------
    df   : pd.DataFrame  – must have columns: strike, T, iv, option_type
    spot : float
    r    : float

    Returns
    -------
    pd.DataFrame with Greek columns appended
    """
    import pandas as pd

    df = df.copy()
    df_valid = df[df["iv"].notna() & (df["iv"] > 0)].copy()

    cols = ["delta", "gamma", "vega", "theta", "rho"]
    for col in cols:
        df[col] = float("nan")

    for idx, row in df_valid.iterrows():
        try:
            g = greeks_summary(
                S           = spot,
                K           = row["strike"],
                T           = row["T"],
                r           = r,
                sigma       = row["iv"],
                option_type = row["option_type"],
            )
            for col in cols:
                df.at[idx, col] = g[col]
        except Exception:
            pass   # leave as NaN if computation fails

    return df