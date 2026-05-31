# Option Pricing & Implied Volatility Surface — v1 (Baseline)

> *Series*:This is Version 1 of a 4-part project building from a clean baseline to industry-grade derivatives tooling.  
> *Goal of v1:* Solid fundamentals — real market data, correct Black-Scholes implementation, and a working IV solver.

---

## What This Project Does

| Step | Description |
|------|-------------|
| *Data* | Pulls real SPY options chain (calls + puts) via `yfinance` |
| **Pricing** | Implements Black-Scholes analytical formula for European options |
| **IV Solver** | Computes Implied Volatility from market prices using bisection method |
| **Visualisation** | Produces 4 charts: price curve, IV smile, 3-D surface, heatmap |

---

## Project Structure

```
option_pricing_v1/
│
├── main.py                  ← Entry point. Run this.
├── requirements.txt         ← All dependencies (pinned)
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── black_scholes.py     ← BS pricing (call + put)
│   ├── implied_volatility.py← IV solver (bisection method)
│   ├── data_fetcher.py      ← yfinance data fetching
│   └── plotter.py           ← All chart generation
│
└── outputs/                 ← Auto-created. Charts + CSV saved here.
```

---

## Setup Instructions (Python 3.10.10)

### Step 1 — Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> If `py_vollib` fails to install on Windows, run:
> ```bash
> pip install py_vollib --no-build-isolation
> ```
> It is only used as an optional reference — the project works without it.

### Step 3 — Run

```bash
python main.py
```

You need an internet connection. The script fetches live market data.

---

## Output Files

All files are written to the `outputs/` folder:

| File | Description |
|------|-------------|
| `bs_price_vs_spot.png` | Call & put prices across a range of spot prices |
| `iv_smile_<YYYY-MM-DD>.png` | IV smile for each expiry date |
| `iv_surface_3d.png` | 3-D scatter of Strike × Maturity × IV |
| `iv_heatmap.png` | 2-D colour heatmap of IV (calls only) |
| `options_with_iv.csv` | Full dataset with computed IV column |

---

## Key Concepts Demonstrated

### Black-Scholes Formula
```
Call = S·N(d1) − K·e^(−rT)·N(d2)
Put  = K·e^(−rT)·N(−d2) − S·N(−d1)

d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
d2 = d1 − σ·√T
```

### Implied Volatility
IV is the volatility σ that makes the BS formula equal the observed market price.  
Solved numerically via **bisection** — reliable and guaranteed to converge within bounds.

### IV Smile / Skew
Real markets show that IV is not constant across strikes — this is the **volatility smile**.  
BS assumes flat vol; the smile reveals that assumption breaks down in practice.

---

## Data Source

- **Options Chain:** Yahoo Finance via `yfinance` (real market data)
- **Risk-Free Rate:** 13-week US T-bill yield (`^IRX`) from Yahoo Finance
- **Underlying:** SPY (S&P 500 ETF) — liquid, well-priced options

---

## Roadmap

| Version | Focus |
|---------|-------|
| **v1 (this)** | Black-Scholes, bisection IV, IV smile + surface |
| v2 | Option Greeks, Newton-Raphson IV solver, skew analysis |
| v3 | Local volatility (Dupire), SABR model, surface interpolation |
| v4 | Calibration pipelines, performance optimisation, clean API |

---

## Requirements

```
numpy==1.24.3
pandas==2.0.3
scipy==1.11.1
matplotlib==3.7.2
yfinance==0.2.31
py_vollib==1.0.1
py_lets_be_rational==1.0.1
requests==2.31.0
```

Python version: **3.10.10**

