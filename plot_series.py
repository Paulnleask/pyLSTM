# =========================
# plot_series.py
# =========================

"""
Plot the original PRICE series, the in-sample fitted PRICES over the training region,
and the out-of-sample forecast PRICES produced by predict_lstm.py.

To run (for example PLTR stock):
    python plot_series.py --csv data/PLTR.csv --artifacts artifacts
"""

# ---------------- Import required libraries and dependencies ----------------
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str)
    ap.add_argument("--artifacts", default="artifacts", type=str)
    ap.add_argument("--title", default="LSTM Fit and Forecast", type=str)
    args = ap.parse_args()

    artifacts = Path(args.artifacts)
    cfg = json.loads(Path(artifacts / "config.json").read_text())

    df = pd.read_csv(args.csv)
    if cfg.get("date_column_present", False) and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    feature = cfg["feature"]
    cutoff_idx_price = int(cfg["cutoff_idx_price"])

    # Original PRICE series
    df_all = df[["Date", feature]].rename(columns={feature: "y"})

    # In-sample fitted PRICES
    df_train = pd.read_csv(artifacts / "train_preds.csv")
    if "Date" in df_train.columns:
        df_train["Date"] = pd.to_datetime(df_train["Date"])

    # Future forecast PRICES
    df_future = pd.read_csv(artifacts / "future_preds.csv")
    if "Date" in df_future.columns:
        df_future["Date"] = pd.to_datetime(df_future["Date"])

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_all["Date"], df_all["y"], label="Historical stock price", linewidth=1.5)
    plt.plot(df_train["Date"], df_train["y_hat"], label="Training data fit", linewidth=1.5)
    plt.plot(df_future["Date"], df_future["y_pred"], label="LSTM forecast", linewidth=2, linestyle="--")

    # vertical cutoff line
    if "Date" in df_all.columns:
        cutoff_date = df_all["Date"].iloc[cutoff_idx_price - 1]
        plt.axvline(cutoff_date, color="grey", alpha=0.5, linestyle=":")
        plt.text(cutoff_date, plt.ylim()[1], " cutoff", va="top", ha="left", fontsize=9, color="grey")

    csv_title = Path(args.csv).stem
    plt.title(args.title)
    plt.xlabel("Date")
    plt.ylabel(f"{csv_title} Stock Price ($) ({feature})")
    plt.legend()
    plt.tight_layout()
    out_png = artifacts / "fit_and_forecast.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    main()