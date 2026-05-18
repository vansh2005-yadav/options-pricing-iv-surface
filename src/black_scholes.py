"""
black_scholes.py
----------------
Core Black-Scholes pricing functions for European call and put options.
All inputs are validated before computation.
"""

import numpy as np
from scipy.stats import norm


def _validate_inputs(S, K, T, r, sigma):
    """Raise ValueError if any input is out of valid range."""
    if S <= 0:
        raise ValueError(f"Spot price S must be > 0, got {S}")
    if K <= 0:
        raise ValueError(f"Strike K must be > 0, got {K}")
    if T <= 0:
        raise ValueError(f"Time to expiry T must be > 0, got {T}")
    if sigma <= 0:
        raise ValueError(f"Volatility sigma must be > 0, got {sigma}")


def d1(S, K, T, r, sigma):
    """
    Compute d1 term in Black-Scholes formula.

    Parameters
    ----------
    S     : float  – current spot price
    K     : float  – option strike price
    T     : float  – time to expiry in years
    r     : float  – risk-free interest rate (annualised, continuous)
    sigma : float  – volatility (annualised)

    Returns
    -------
    float
    """
    _validate_inputs(S, K, T, r, sigma)
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma):
    """Compute d2 = d1 - sigma*sqrt(T)."""
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call_price(S, K, T, r, sigma):
    """
    Black-Scholes price for a European call option.

    Returns
    -------
    float : call option price
    """
    _validate_inputs(S, K, T, r, sigma)
    _d1 = d1(S, K, T, r, sigma)
    _d2 = _d1 - sigma * np.sqrt(T)
    return S * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)


def bs_put_price(S, K, T, r, sigma):
    """
    Black-Scholes price for a European put option.

    Returns
    -------
    float : put option price
    """
    _validate_inputs(S, K, T, r, sigma)
    _d1 = d1(S, K, T, r, sigma)
    _d2 = _d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-_d2) - S * norm.cdf(-_d1)


def bs_price(S, K, T, r, sigma, option_type="call"):
    """
    Unified pricing function.

    Parameters
    ----------
    option_type : str  – 'call' or 'put' (case-insensitive)

    Returns
    -------
    float : option price
    """
    ot = option_type.lower()
    if ot == "call":
        return bs_call_price(S, K, T, r, sigma)
    elif ot == "put":
        return bs_put_price(S, K, T, r, sigma)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")