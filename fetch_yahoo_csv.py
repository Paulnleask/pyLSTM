# =========================
# fetch_yahoo_csv.py
# =========================

"""
Fetch stock prices using Yahoo! Finance's yfinance package
Converts stock data to CSV format
Requires stock ticker, e.g. AAPL, NVDA, etc.

To run (for example NVDA stock):
    python fetch_yahoo_csv.py --ticker NVDA --start 2015-01-01 --out NVDA.csv --outdir data
"""

# ---------------- Import required libraries and dependencies ----------------
import argparse
import yfinance as yf
import pandas as pd
from pathlib import Path

# ---------------- Argument parser ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--ticker", required=True)                                                  # e.g. AAPL, MSFT, TSLA
parser.add_argument("--start",  default="2015-01-01")                                           # YYYY-MM-DD
parser.add_argument("--end",    default=None)                                                   # None = up to today
parser.add_argument("--out",    required=True)                                                  # output CSV path
parser.add_argument("--outdir", default="data", type=str)                                       # output directory
parser.add_argument("--adjust", action="store_true", help="Use Adj Close instead of Close")     # adjusted close
args = parser.parse_args()

# ---------------- Create output directory ----------------
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

# ---------------- Download stock data with yfinance ----------------
data = yf.download(args.ticker, start=args.start, end=args.end, progress=False, auto_adjust=False)
if data.empty:
    raise SystemExit(f"No data returned for {args.ticker}. Check the symbol or date range.")

# ---------------- Drop ticker level multi-index ----------------
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# ---------------- Ensure required columns exist ----------------
cols = ["Open","High","Low","Close","Volume"]
missing = [c for c in cols if c not in data.columns]
if missing:
    raise SystemExit(f"Missing columns in download: {missing}")
df = data.reset_index()  # has a Date column now

# ---------------- Optionally replace Close with Adjusted Close ----------------
if args.adjust:
    if "Adj Close" not in df.columns:
        raise SystemExit("Adj Close not available in download.")
    df["Close"] = df["Adj Close"]

# ---------------- Reorder & trim columns for loader ----------------
out = df[["Date","Open","High","Low","Close","Volume"]].copy()
# Cast types to safe defaults
out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")

# ---------------- Output stock data to csv ----------------
out.to_csv(outdir / args.out, index=False)
print(f"Wrote {len(out)} rows to {args.out}")