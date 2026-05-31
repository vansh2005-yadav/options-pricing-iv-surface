"""
implied_volatility.py  (v2 — UPGRADED)
---------------------------------------
v1 used bisection only.
v2 uses Newton-Raphson as the primary solver (much faster convergence)
with automatic fallback to bisection if Newton diverges or overshoots.

Why Newton-Raphson?
    Bisection converges at ~0.5 per iteration (linear).
    Newton-Raphson converges quadratically — typically 4-6 iterations
    vs 30-50 for bisection to reach the same tolerance.
    The derivative used is Vega (∂Price/∂σ), computed analytically.

Fallback logic:
    If Newton produces a negative sigma, overshoots the bounds,
    or hasn't converged after max_iter, bisection takes over.
    This makes the solver robust on deep ITM/OTM options where
    Newton can become unstable.
"""

import numpy as np
from scipy.stats import norm
from src.black_scholes import bs_price, d1


# ──────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────

def _bs_vega(S, K, T, r, sigma):
    """Raw vega (∂Price/∂σ) — used as the Newton-Raphson derivative."""
    _d1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(_d1) * np.sqrt(T)


def _intrinsic(S, K, T, r, option_type):
    """Lower arbitrage bound for the option price."""
    disc = K * np.exp(-r * T)
    if option_type == "call":
        return max(S - disc, 0.0)
    return max(disc - S, 0.0)


# ──────────────────────────────────────────────────────────
# Newton-Raphson solver
# ──────────────────────────────────────────────────────────

def _newton_raphson(market_price, S, K, T, r, option_type,
                    tol=1e-6, max_iter=50):
    """
    Newton-Raphson IV solver.
    Initial guess: Brenner-Subrahmanyam approximation.
    Returns float sigma if converged, else None.
    """
    # Brenner-Subrahmanyam initial guess (works well near ATM)
    sigma = np.sqrt(2.0 * np.pi / T) * (market_price / S)
    sigma = max(min(sigma, 5.0), 0.001)   # clamp to sane range

    for _ in range(max_iter):
        try:
            price = bs_price(S, K, T, r, sigma, option_type)
            v     = _bs_vega(S, K, T, r, sigma)
        except Exception:
            return None

        if v < 1e-10:          # vega near zero → Newton unstable
            return None

        diff  = price - market_price
        sigma_new = sigma - diff / v

        if sigma_new <= 0 or sigma_new > 10.0:   # out of bounds
            return None

        if abs(sigma_new - sigma) < tol:
            return sigma_new

        sigma = sigma_new

    return None   # did not converge


# ──────────────────────────────────────────────────────────
# Bisection solver (fallback — same as v1 but cleaner)
# ──────────────────────────────────────────────────────────

def _bisection(market_price, S, K, T, r, option_type,
               tol=1e-6, max_iter=500,
               sigma_low=1e-4, sigma_high=10.0):
    """Bisection fallback — guaranteed to converge within bounds."""
    price_low  = bs_price(S, K, T, r, sigma_low,  option_type)
    price_high = bs_price(S, K, T, r, sigma_high, option_type)

    if market_price < price_low or market_price > price_high:
        return None

    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2.0
        price_mid = bs_price(S, K, T, r, sigma_mid, option_type)
        diff      = price_mid - market_price

        if abs(diff) < tol:
            return sigma_mid

        if diff > 0:
            sigma_high = sigma_mid
        else:
            sigma_low  = sigma_mid

    return (sigma_low + sigma_high) / 2.0


# ──────────────────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────────────────

def implied_volatility(
    market_price,
    S,
    K,
    T,
    r,
    option_type="call",
    tol=1e-6,
    max_iter=50,
):
    """
    Compute implied volatility using Newton-Raphson with bisection fallback.

    Parameters
    ----------
    market_price : float  – observed mid price of the option
    S            : float  – spot price
    K            : float  – strike
    T            : float  – time to expiry in years
    r            : float  – risk-free rate
    option_type  : str    – 'call' or 'put'
    tol          : float  – convergence tolerance
    max_iter     : int    – max Newton-Raphson iterations

    Returns
    -------
    float or None
    """
    if market_price is None or market_price <= 0:
        return None
    if T is None or T <= 0:
        return None

    ot = option_type.lower()

    # Arbitrage bound check
    lb = _intrinsic(S, K, T, r, ot)
    if market_price < lb * 0.999:
        return None

    # Try Newton-Raphson first
    iv = _newton_raphson(market_price, S, K, T, r, ot, tol=tol, max_iter=max_iter)

    # Fall back to bisection if Newton failed
    if iv is None or iv <= 0 or iv > 10.0:
        iv = _bisection(market_price, S, K, T, r, ot, tol=tol)

    return iv