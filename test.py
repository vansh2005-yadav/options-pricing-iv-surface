"""
test_bs.py
----------
Quick sanity-check for Black-Scholes and IV solver.
No external data needed — runs offline instantly.

Run with:
    python test_bs.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.black_scholes      import bs_call_price, bs_put_price, bs_price
from src.implied_volatility  import implied_volatility


def test_put_call_parity():
    """C - P  =  S - K·e^(-rT)  (put-call parity)"""
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    C = bs_call_price(S, K, T, r, sigma)
    P = bs_put_price(S,  K, T, r, sigma)
    import math
    lhs = C - P
    rhs = S - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 1e-8, f"Put-call parity FAILED: {lhs:.6f} != {rhs:.6f}"
    print(f"[PASS] Put-call parity  C={C:.4f}  P={P:.4f}")


def test_iv_round_trip():
    """Price an option with known sigma, then recover sigma via IV solver."""
    S, K, T, r, sigma = 100, 105, 0.5, 0.04, 0.25
    price = bs_call_price(S, K, T, r, sigma)
    iv    = implied_volatility(price, S, K, T, r, option_type="call")
    assert iv is not None, "IV solver returned None"
    assert abs(iv - sigma) < 1e-4, f"IV round-trip FAILED: got {iv:.6f}, expected {sigma:.6f}"
    print(f"[PASS] IV round-trip  sigma={sigma:.4f}  iv_recovered={iv:.4f}  price={price:.4f}")


def test_deep_itm_otm():
    """Deep ITM call should approach intrinsic; deep OTM should approach zero."""
    S, K_itm, K_otm, T, r, sigma = 100, 50, 200, 1.0, 0.05, 0.20
    import math
    intrinsic = S - K_itm * math.exp(-r * T)
    call_itm  = bs_call_price(S, K_itm, T, r, sigma)
    call_otm  = bs_call_price(S, K_otm, T, r, sigma)
    assert call_itm > intrinsic * 0.99, "Deep ITM call below intrinsic"
    assert call_otm < 0.5,              f"Deep OTM call too large: {call_otm:.4f}"
    print(f"[PASS] Deep ITM call={call_itm:.4f}  Deep OTM call={call_otm:.6f}")


def test_zero_vol_edge():
    """At near-zero volatility, call price ≈ intrinsic (discounted)."""
    import math
    S, K, T, r, sigma = 110, 100, 1.0, 0.05, 1e-6
    call = bs_call_price(S, K, T, r, sigma)
    expected = S - K * math.exp(-r * T)
    assert abs(call - expected) < 0.01, f"Near-zero vol call={call:.4f} expected≈{expected:.4f}"
    print(f"[PASS] Near-zero vol  call={call:.4f}  expected≈{expected:.4f}")


if __name__ == "__main__":
    print("Running Black-Scholes sanity checks...\n")
    test_put_call_parity()
    test_iv_round_trip()
    test_deep_itm_otm()
    test_zero_vol_edge()
    print("\nAll tests passed ✓")