"""
main.py
-------
Option Pricing & Implied Volatility Surface  –  v1 (Baseline)

What this script does:
  1. Fetches real SPY options data and spot price via yfinance
  2. Fetches current risk-free rate from ^IRX (13-week T-bill)
  3. Calculates implied volatility for every option row using bisection
  4. Produces 4 publication-ready charts saved to /outputs:
       • bs_price_vs_spot.png
       • iv_smile_<expiry>.png   (one per expiry)
       • iv_surface_3d.png
       • iv_heatmap.png

Run from the project root:
    python main.py
"""

import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")          # suppress yfinance noise

# ── Project imports ───────────────────────────────────────────────
from src.data_fetcher      import get_spot_price, get_risk_free_rate, get_options_chain
from src.implied_volatility import implied_volatility
from src.plotter            import (
    plot_bs_price_vs_spot,
    plot_iv_smile,
    plot_iv_surface_3d,
    plot_iv_heatmap,
)

# ── Configuration ─────────────────────────────────────────────────
TICKER       = "SPY"          # underlying ticker
NUM_EXPIRIES = 4              # number of nearest expiry dates to pull
OUTPUT_DIR   = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_iv_for_dataframe(df: pd.DataFrame, spot: float, r: float) -> pd.DataFrame:
    """
    Iterate over every row in the options dataframe and compute IV.
    Adds a column 'iv' (float, or NaN if computation failed).
    """
    ivs = []
    total = len(df)

    for i, row in df.iterrows():
        iv = implied_volatility(
            market_price = row["mid_price"],
            S            = spot,
            K            = row["strike"],
            T            = row["T"],
            r            = r,
            option_type  = row["option_type"],
        )
        ivs.append(iv)

        # Simple progress indicator every 100 rows
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  IV computed: {i + 1}/{total}", end="\r")

    print()   # newline after progress
    df = df.copy()
    df["iv"] = ivs
    return df


def main():
    print("=" * 60)
    print("  Option Pricing & IV Surface  –  v1 Baseline")
    print("=" * 60)

    # ── 1. Fetch market data ──────────────────────────────────────
    print(f"\n[1/4] Fetching spot price for {TICKER} ...")
    spot = get_spot_price(TICKER)
    print(f"      Spot price: ${spot:.2f}")

    print("\n[2/4] Fetching risk-free rate ...")
    r = get_risk_free_rate()
    print(f"      Risk-free rate: {r:.2%}")

    print(f"\n[3/4] Fetching options chain ({NUM_EXPIRIES} expiries) ...")
    df = get_options_chain(TICKER, num_expiries=NUM_EXPIRIES)
    print(f"      Rows fetched (after cleaning): {len(df)}")
    print(f"      Expiries: {sorted(df['expiry'].unique())}")

    # ── 2. Compute implied volatility ────────────────────────────
    print("\n[4/4] Computing implied volatilities ...")
    df = compute_iv_for_dataframe(df, spot=spot, r=r)

    valid_iv  = df["iv"].notna().sum()
    total_rows = len(df)
    print(f"      IV computed successfully: {valid_iv}/{total_rows} rows "
          f"({valid_iv / total_rows:.1%})")

    # Save the processed data for reference
    csv_path = os.path.join(OUTPUT_DIR, "options_with_iv.csv")
    df.to_csv(csv_path, index=False)
    print(f"      Data saved → {csv_path}")

    # ── 3. Plot 1: BS price vs spot (theory, using ATM parameters) ─
    print("\n[PLOT 1] Black-Scholes price vs spot ...")
    first_expiry = sorted(df["expiry"].unique())[0]
    T_first      = df[df["expiry"] == first_expiry]["T"].iloc[0]

    # Use median ATM IV as sigma reference
    atm_rows = df[(df["option_type"] == "call") &
                  (abs(df["strike"] - spot) / spot < 0.03) &
                  (df["iv"].notna())]
    sigma_ref = float(atm_rows["iv"].median()) if not atm_rows.empty else 0.20
    print(f"      Reference sigma (ATM IV): {sigma_ref:.2%}")

    plot_bs_price_vs_spot(K=round(spot), T=T_first, r=r,
                          sigma=sigma_ref, output_dir=OUTPUT_DIR)

    # ── 4. Plot 2: IV Smile per expiry ────────────────────────────
    print("\n[PLOT 2] IV Smile per expiry ...")
    for expiry in sorted(df["expiry"].unique()):
        df_exp = df[df["expiry"] == expiry]
        plot_iv_smile(df_exp, spot=spot,
                      expiry_label=expiry, output_dir=OUTPUT_DIR)

    # ── 5. Plot 3: 3-D IV Surface ─────────────────────────────────
    print("\n[PLOT 3] 3-D IV Surface ...")
    plot_iv_surface_3d(df, output_dir=OUTPUT_DIR)

    # ── 6. Plot 4: IV Heatmap ─────────────────────────────────────
    print("\n[PLOT 4] IV Heatmap ...")
    plot_iv_heatmap(df, output_dir=OUTPUT_DIR)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✓  All done!  Charts saved to /outputs")
    print("=" * 60)

    print(f"""
Quick Summary
─────────────
  Ticker           : {TICKER}
  Spot Price       : ${spot:.2f}
  Risk-Free Rate   : {r:.2%}
  Expiries Fetched : {len(df['expiry'].unique())}
  Total Options    : {total_rows}
  IV Solved        : {valid_iv} ({valid_iv / total_rows:.1%})
  Reference IV     : {sigma_ref:.2%}
""")


if __name__ == "__main__":
    main()