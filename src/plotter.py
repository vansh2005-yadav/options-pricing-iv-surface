"""
plotter.py
----------
Visualisation functions for the IV surface project (v1).

Plots produced:
  1. BS Price vs Spot          – call & put prices across spot range
  2. IV Smile                  – IV vs Strike for one expiry
  3. IV Surface (3-D)          – Strike × Expiry × IV
  4. IV Heatmap                – 2-D colour grid of IV
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe for all environments
import matplotlib.pyplot as plt
from matplotlib import cm
import os


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1.  BS Price vs Spot
# ─────────────────────────────────────────────
def plot_bs_price_vs_spot(K, T, r, sigma, output_dir=OUTPUT_DIR):
    """
    Plot call and put BS prices as spot price varies from 50 % to 150 % of K.
    """
    from src.black_scholes import bs_call_price, bs_put_price

    spots  = np.linspace(K * 0.5, K * 1.5, 300)
    calls  = [bs_call_price(s, K, T, r, sigma) for s in spots]
    puts   = [bs_put_price(s,  K, T, r, sigma) for s in spots]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(spots, calls, label="Call Price", color="#2196F3", linewidth=2)
    ax.plot(spots, puts,  label="Put Price",  color="#F44336", linewidth=2)
    ax.axvline(K, color="grey", linestyle="--", linewidth=1, label=f"Strike = {K}")
    ax.set_title(f"Black-Scholes Option Prices  (K={K}, T={T:.2f}y, σ={sigma:.0%}, r={r:.1%})",
                 fontsize=12)
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Option Price")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "bs_price_vs_spot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved → {path}")
    return path


# ─────────────────────────────────────────────
# 2.  IV Smile for one expiry
# ─────────────────────────────────────────────
def plot_iv_smile(df_expiry, spot, expiry_label, output_dir=OUTPUT_DIR):
    """
    Parameters
    ----------
    df_expiry    : pd.DataFrame with columns ['strike', 'option_type', 'iv']
                   already filtered to ONE expiry date.
    spot         : float  – underlying spot price (for ATM line)
    expiry_label : str    – used in the chart title and filename
    """
    calls = df_expiry[df_expiry["option_type"] == "call"].sort_values("strike")
    puts  = df_expiry[df_expiry["option_type"] == "put"].sort_values("strike")

    calls = calls[calls["iv"].notna() & (calls["iv"] > 0)]
    puts  = puts[puts["iv"].notna()  & (puts["iv"]  > 0)]

    if calls.empty and puts.empty:
        print(f"[WARNING] No valid IV data for {expiry_label}. Skipping smile plot.")
        return None

    fig, ax = plt.subplots(figsize=(9, 5))

    if not calls.empty:
        ax.scatter(calls["strike"], calls["iv"], label="Call IV",
                   color="#2196F3", s=25, alpha=0.8)
        ax.plot(calls["strike"], calls["iv"],
                color="#2196F3", linewidth=1.2, alpha=0.6)

    if not puts.empty:
        ax.scatter(puts["strike"], puts["iv"], label="Put IV",
                   color="#F44336", s=25, alpha=0.8)
        ax.plot(puts["strike"], puts["iv"],
                color="#F44336", linewidth=1.2, alpha=0.6)

    ax.axvline(spot, color="green", linestyle="--", linewidth=1.2,
               label=f"Spot = {spot:.2f}")
    ax.set_title(f"Implied Volatility Smile  –  Expiry: {expiry_label}", fontsize=12)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    safe_label = expiry_label.replace("/", "-").replace(" ", "_")
    path = os.path.join(output_dir, f"iv_smile_{safe_label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved → {path}")
    return path


# ─────────────────────────────────────────────
# 3.  IV Surface – 3-D plot
# ─────────────────────────────────────────────
def plot_iv_surface_3d(df, output_dir=OUTPUT_DIR):
    """
    Parameters
    ----------
    df : pd.DataFrame with columns ['strike', 'T', 'iv', 'option_type']
         covering multiple expiries.
    """
    df_plot = df[df["iv"].notna() & (df["iv"] > 0)].copy()

    if df_plot.empty:
        print("[WARNING] No valid IV data for 3-D surface. Skipping.")
        return None

    # Use calls only for a cleaner surface (calls have better liquidity OTM)
    df_calls = df_plot[df_plot["option_type"] == "call"]
    if df_calls.empty:
        df_calls = df_plot    # fallback to all

    strikes    = df_calls["strike"].values
    maturities = df_calls["T"].values
    ivs        = df_calls["iv"].values

    fig = plt.figure(figsize=(11, 7))
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.scatter(strikes, maturities, ivs,
                      c=ivs, cmap=cm.RdYlGn_r, s=18, alpha=0.85)

    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label="Implied Volatility")
    ax.set_xlabel("Strike",            labelpad=10)
    ax.set_ylabel("Time to Expiry (Y)", labelpad=10)
    ax.set_zlabel("Implied Volatility", labelpad=10)
    ax.set_title("Implied Volatility Surface", fontsize=13, pad=15)

    # Format z-axis as percentage
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda z, _: f"{z:.0%}"))

    path = os.path.join(output_dir, "iv_surface_3d.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved → {path}")
    return path


# ─────────────────────────────────────────────
# 4.  IV Heatmap
# ─────────────────────────────────────────────
def plot_iv_heatmap(df, output_dir=OUTPUT_DIR):
    """
    2-D heatmap: rows = expiry dates, columns = strike bins.

    Parameters
    ----------
    df : pd.DataFrame with columns ['expiry', 'strike', 'iv', 'option_type']
    """
    df_plot = df[(df["iv"].notna()) & (df["iv"] > 0)].copy()
    df_plot = df_plot[df_plot["option_type"] == "call"]   # calls only

    if df_plot.empty:
        print("[WARNING] No valid call IV data for heatmap. Skipping.")
        return None

    pivot = df_plot.pivot_table(
        index="expiry", columns="strike", values="iv", aggfunc="mean"
    )

    if pivot.empty:
        print("[WARNING] Pivot table is empty. Skipping heatmap.")
        return None

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.8 + 1)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r",
                   vmin=pivot.values[~np.isnan(pivot.values)].min(),
                   vmax=pivot.values[~np.isnan(pivot.values)].max())

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Implied Volatility")
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{k:.0f}" for k in pivot.columns],
                       rotation=90, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Expiry Date")
    ax.set_title("Implied Volatility Heatmap  (Calls)", fontsize=12)

    fig.tight_layout()
    path = os.path.join(output_dir, "iv_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved → {path}")
    return path