# =========================
# predict_lstm.py
# =========================

"""
Predict the next H steps directly (no recursion) using the trained multi-horizon LSTM.
For logreturn/delta targets, reconstruct the PRICE path.

To run (for example PLTR stock):
    python plot_series.py --csv data/PLTR.csv --artifacts artifacts
"""

# ---------------- Import required libraries and dependencies ----------------
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ────────────────────────────────────────────────────────────────────────────────
# LSTM Model
# ────────────────────────────────────────────────────────────────────────────────

# ---------------- Long short-term memory network using pyTorch ----------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_len: int, num_layers: int = 1, dropout: float = 0.0, horizon: int = 1):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM( input_size=input_size, hidden_size=hidden_len, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_len, horizon)

    # ---------------- Linear regression forward pass ----------------
    def forward(self, x):
        # x: (batch_size, seq_len, 1)
        out, _ = self.lstm(x)       # (batch_size, seq_len, hidden_len)
        h_last = out[:, -1, :]      # (batch_size, hidden_len)
        yhat = self.head(h_last)    # (batch_size, horizon)
        return yhat

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str)
    ap.add_argument("--artifacts", default="artifacts", type=str)
    args = ap.parse_args()

    artifacts = Path(args.artifacts)
    cfg = json.loads(Path(artifacts / "config.json").read_text())
    model_state = torch.load(artifacts / "model_state.pt", map_location="cpu", weights_only=True)

    # Load CSV
    df = pd.read_csv(args.csv)
    if cfg.get("date_column_present", False) and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    df[cfg["feature"]] = pd.to_numeric(df[cfg["feature"]], errors="coerce")
    df = df.dropna(subset=[cfg["feature"]]).reset_index(drop=True)

    prices = df[cfg["feature"]].astype(float).values
    dates  = df["Date"].values if "Date" in df.columns else np.arange(len(prices))

    # Build target series for the input window
    tgt = cfg["target"]
    if tgt == "price":
        work = prices.astype(np.float32)
        dates_work = dates
    elif tgt == "logreturn":
        work = np.diff(np.log(prices + 1e-12)).astype(np.float32)
        dates_work = dates[1:]
    else:
        work = np.diff(prices).astype(np.float32)
        dates_work = dates[1:]

    # Normalization (from training)
    norm = cfg["normalize"]
    if norm["type"] == "minmax":
        w_min, w_max = norm["min"], norm["max"]
        def norm_f(v): return (v - w_min) / (w_max - w_min)
        def denorm_f(v): return v * (w_max - w_min) + w_min
    else:
        mu, sigma = norm["mean"], norm["std"]
        def norm_f(v): return (v - mu) / sigma
        def denorm_f(v): return v * sigma + mu

    work_norm = norm_f(work)

    seq_len = int(cfg["seq_len"])
    horizon = int(cfg["horizon"])
    cutoff_idx_price = int(cfg["cutoff_idx_price"])
    cutoff_idx_work  = int(cfg["cutoff_idx_work"])

    # Build the single input window ending at the cutoff index in target space
    if cutoff_idx_work < seq_len:
        raise ValueError("Not enough target points before cutoff to build the input window.")
    window = work_norm[cutoff_idx_work - seq_len : cutoff_idx_work].astype(np.float32)  # (seq_len,)

    # Model
    model = LSTMRegressor(1, int(cfg["hidden_len"]), int(cfg["num_layers"]), float(cfg["dropout"]), horizon)
    model.load_state_dict(model_state)
    model.eval()

    # One forward pass → next H normalized targets
    x = torch.from_numpy(window[None, :, None])  # (1, seq_len, 1)
    with torch.no_grad():
        yhat_norm = model(x).cpu().numpy().squeeze(0)  # (H,)

    preds_target = denorm_f(yhat_norm)  # to target space (price / logreturn / delta)

    # Reconstruct PRICE path
    if tgt == "price":
        preds_price = preds_target
    else:
        p = float(cfg["last_observed_price"])  # price at (cutoff_idx_price - 1)
        preds_price = []
        for v in preds_target:
            if tgt == "logreturn":
                p = p * float(np.exp(v))
            else:  # delta
                p = p + float(v)
            preds_price.append(p)
        preds_price = np.array(preds_price, dtype=float)

    # Future business-day dates
    if "Date" in df.columns:
        last_date = pd.to_datetime(df["Date"].iloc[cutoff_idx_price - 1])
        future_dates = pd.bdate_range(last_date + pd.tseries.offsets.BDay(1), periods=horizon)
    else:
        future_dates = np.arange(cutoff_idx_price, cutoff_idx_price + horizon)

    out = pd.DataFrame({"Date": future_dates, "y_pred": preds_price})
    out_path = artifacts / "future_preds.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved forecast (next {horizon} steps) to {out_path}")


if __name__ == "__main__":
    main()
