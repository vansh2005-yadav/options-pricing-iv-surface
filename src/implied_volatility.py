"""
implied_volatility.py
---------------------
Compute implied volatility from market option prices using the
bisection method. Robust and dependency-light for v1.
"""

import numpy as np
try:
    from src.black_scholes import bs_price
except ModuleNotFoundError:
    from black_scholes import bs_price


def implied_volatility(
    market_price,
    S,
    K,
    T,
    r,
    option_type="call",
    tol=1e-6,
    max_iter=500,
    sigma_low=1e-4,
    sigma_high=10.0,
):
    """
    Compute implied volatility via bisection.

    Parameters
    ----------
    market_price : float  – observed market price of the option
    S            : float  – spot price
    K            : float  – strike price
    T            : float  – time to expiry in years
    r            : float  – risk-free rate
    option_type  : str    – 'call' or 'put'
    tol          : float  – convergence tolerance on price difference
    max_iter     : int    – maximum bisection iterations
    sigma_low    : float  – lower bound for vol search
    sigma_high   : float  – upper bound for vol search

    Returns
    -------
    float or None
        Implied volatility if converged, else None.
    """
    if market_price <= 0:
        return None
    if T <= 0:
        return None

    # Boundary check: market price must be within arbitrage bounds
    intrinsic = max(S - K * np.exp(-r * T), 0) if option_type == "call" else max(K * np.exp(-r * T) - S, 0)
    if market_price < intrinsic * 0.999:   # small tolerance for rounding
        return None

    price_low  = bs_price(S, K, T, r, sigma_low,  option_type)
    price_high = bs_price(S, K, T, r, sigma_high, option_type)

    # Market price must be between the two boundary prices
    if market_price < price_low:
        return None
    if market_price > price_high:
        return None

    for _ in range(max_iter):
        sigma_mid  = (sigma_low + sigma_high) / 2.0
        price_mid  = bs_price(S, K, T, r, sigma_mid, option_type)
        diff       = price_mid - market_price

        if abs(diff) < tol:
            return sigma_mid

        if diff > 0:
            sigma_high = sigma_mid
        else:
            sigma_low  = sigma_mid

    # Return best estimate even if tolerance not fully met
    return (sigma_low + sigma_high) / 2.0