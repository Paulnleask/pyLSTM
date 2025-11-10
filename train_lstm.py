# =========================
# train_lstm.py
# =========================

"""
Train a multi-horizon LSTM forecaster on a single 1D series from CSV
(Date, Open, High, Low, Close, Volume).

Target options:
    * price      : raw price y_t
    * logreturn  : r_t = log(y_t / y_{t-1})      (default, recommended)
    * delta      : d_t = y_t - y_{t-1}

- The model predicts the next H steps directly (no recursion)
- Normalization is fit on the TRAIN split of the TARGET series (no leakage)
- Optimizer: Adaptive momentum estimate (Adam)
- Const function: Mean-Squared Error (MSE)

Saves:
  - artifacts/model_state.pt
  - artifacts/config.json
  - artifacts/train_preds.csv    (in-sample fitted PRICES for the first horizon step)

To run (for example PLTR stock closing price with logreturn):
    python train_lstm.py --csv data/PLTR.csv --feature Close --target logreturn --normalize standard --outdir artifacts
"""

# ---------------- Import required libraries and dependencies ----------------
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys

# ---------------- Hyper-parameters ----------------
cutoff = 0.9            # Fraction of data (from start) used for training
horizon = 128           # Number of future week days for forecasting (weekends ignored)
seq_len = 128           # Lookback window length, e.g. 1 (daily), 30 (monthly), 90 (quarterly)
hidden_len = 64         # Dimension of the hidden state
epochs = 1000           # Number of iterations 
batch_size = 64         # Number of training samples processed per optimizer step
lr = 1e-3               # Learning rate
num_layers = 2          # Number of deep layers
dropout = 0.4           # Dropout probability

# ────────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ────────────────────────────────────────────────────────────────────────────────

# ---------------- Save to json file ----------------
def save_json(path, obj):
    """
    path: a string representing the file path where to save the data (e.g. "config.json")
    obj: the Python object you want to save (e.g. a dictionary containing training parameters)
    Open the file at path in write mode
    Dump (serialize) this object into a JSON file
    """
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ---------------- Device information ----------------
def print_device_info():
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (no CUDA device available)")

# ---------------- Sequence builder ----------------
def make_sequences_multi(series_1d: np.ndarray, seq_len: int, horizon: int):
    """
    Build (X, Y) pairs from a 1D series in target space.
        X[i] = series[i - seq_len : i], e.g. [1,2,3,4,5], [2,3,4,5,6]
        Y[i] = series[i : i + horizon], e.g. [6,7,8], [7,8,9]
    Shapes:
        X: (N, seq_len, 1),  Y: (N, horizon)
    """
    X, Y = [], []
    for i in range(seq_len, len(series_1d) - horizon + 1):
        X.append(series_1d[i - seq_len:i])
        Y.append(series_1d[i:i + horizon])
    X = np.array(X, dtype=np.float32)[..., None]
    Y = np.array(Y, dtype=np.float32)
    return X, Y

# ────────────────────────────────────────────────────────────────────────────────
# LSTM Model
# ────────────────────────────────────────────────────────────────────────────────

# ---------------- Long short-term memory network using PyTorch ----------------
class LSTMRegressor(nn.Module):

    """
    Predict multiple future steps simultaneously (e.g. horizon-step forecast) using linear regression
    """

    def __init__(self, input_size: int, hidden_len: int, num_layers: int = 1, dropout: float = 0.0, horizon: int = 1):
        
        """
        Parameters:
            - input_size: number of inputs at each LSTM, e.g. N if considering N time series
            - hidden_len: dimension of the hidden state
            - num_layers: number of stacked LSTMs for deeper learning
            - dropout: dropout probability between layers for regularization and preventing overfitting
            - horizon: window size 
        """

        # Call nn.Module.__init__()
        super().__init__()

        # Set the window size (horizon)
        self.horizon = horizon
        
        # Define the LSTM neuron
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_len, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        
        # Compute the linear regression output
        self.head = nn.Linear(hidden_len, horizon) 

    # ---------------- Linear regression forward pass ----------------
    def forward(self, x):

        """
        Input x: (batch_size, seq_len, 1)
        """

        # Compute the output from the LSTM cell on input x
        out, _ = self.lstm(x)       # (batch_size, seq_len, hidden_len)

        # Determine the last hidden state (h)
        h_last = out[:, -1, :]      # (batch_size, hidden_len)

        # Predict output with linear regression on the hidden state (y=W*h+b)
        yhat = self.head(h_last)    # (batch_size, horizon)

        # Return the final output
        return yhat

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():

    # ---------------- Argument parser ----------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str, help="CSV with Date,Open,High,Low,Close,Volume")
    ap.add_argument("--feature", default="Close", type=str)
    ap.add_argument("--target", default="logretrun", choices=["pirce", "logreturn", "delta".replace(",", "")], help="Prediction target (default: logretrun)")
    ap.add_argument("--normalize", default="standard", choices=["standard", "minmax"], help="Normalization on target series (default: standard)")
    ap.add_argument("--outdir", default="artifacts", type=str)
    args = ap.parse_args()

    # ---------------- Create output directory ----------------
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- Prepare data from CSV file ----------------
    """
    Read CSV file (like AAPL.csv) into a DataFrame
    Check if the CSV actually has a "Date" column
    Converts the "Date" column from strings (like "2025-01-01") to actual datetime objects
    Sort data by time and reset indexes after sorting, also remove any rows that failed date conversion 
    Check requested feature actually exists in the CSV, if not show available features
    """
    # Import stock csv file
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        sys.exit("Error: data file was not found")

    if "Date" in df.columns: 
        df["Data"] = pd.to_datetime(df["Date"], errors="coerce") 
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if args.feature not in df.columns:
        raise ValueError(f"Feature '{args.feature}' not in CSV. Columns: {list(df.columns)}")

    # ---------------- Ensure numeric feature values ----------------
    """
    Converts the column (say, "Close") from strings to floating-point numbers
    Drops all rows where the feature column is invalid, and renumbers the rows
    """
    df[args.feature] = pd.to_numeric(df[args.feature], errors="coerce") # Converts the column (say, "Close") from strings to floating-point numbers
    df = df.dropna(subset=[args.feature]).reset_index(drop=True) # Drops all rows where the feature column is invalid, and renumbers the rows

    # ---------------- Extract numpy arrays for modelling ----------------
    """
    Converts the feature column (e.g., Close) into a NumPy array of floats, where .values extracts the underlying array
    If a Date column exists -> use those actual datetimes, otherwise use simple integer indices
    """
    prices = df[args.feature].astype(float).values
    dates  = df["Date"].values if "Date" in df.columns else np.arange(len(prices))
    Np = len(prices)

    # ---------------- Build target series (work) & its dates (aligned) ----------------
    tgt = args.target
    if tgt == "price":
        work = prices.astype(np.float32)
        dates_work = dates
    elif tgt == "logreturn":
        work = np.diff(np.log(prices + 1e-12)).astype(np.float32) # returns log of differences log(p_t/p_{t-1})
        dates_work = dates[1:]
    else:  # delta
        work = np.diff(prices).astype(np.float32) # returns differences p_t - p_{t-1}
        dates_work = dates[1:]
    Nw = len(work)

    # ---------------- Compute cutoff in work space and price space ----------------
    cutoff_idx_price = int(math.floor(cutoff * Np))
    cutoff_idx_work  = int(math.floor(cutoff * Nw))
    if cutoff_idx_work <= seq_len + horizon:
        raise ValueError("cutoff*len(target_series) must exceed seq_len + horizon to form training samples.")

    # ---------------- Normalization on training only data in target space ----------------
    work_train = work[:cutoff_idx_work]

    # Minmax normalization
    if args.normalize == "minmax":
        w_min, w_max = float(np.min(work_train)), float(np.max(work_train))
        if w_max == w_min:
            w_max = w_min + 1.0
        def norm(v):   return (v - w_min) / (w_max - w_min)
        def denorm(v): return v * (w_max - w_min) + w_min
        norm_params = {"type": "minmax", "min": w_min, "max": w_max}
    # Standardization (Z-Score normalization)
    else:
        mu = float(np.mean(work_train)) # Mean
        sigma = float(np.std(work_train) + 1e-12) # Standard deviation
        def norm(v):
            return (v - mu) / sigma
        def denorm(v):
            return v * sigma + mu
        norm_params = {"type": "standard", "mean": mu, "std": sigma}
    work_norm = norm(work)

    # Make sequences (multi-horizon) & training split by label index
    X_all, Y_all = make_sequences_multi(work_norm, seq_len, horizon)

    # Each label corresponds to the first horizon step at work index i
    label_indices = np.arange(seq_len, Nw - horizon + 1) # e.g. [270, 271, ... , 2460]
    train_mask = label_indices < cutoff_idx_work
    X_train, Y_train = X_all[train_mask], Y_all[train_mask]

    # Dates aligned 1:1 with labels (for clean masking)
    label_dates = dates_work[seq_len : seq_len + len(label_indices)]

    # ---------------- 1. The model ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info()
    model = LSTMRegressor(input_size=1, hidden_len=hidden_len, num_layers=num_layers, dropout=dropout, horizon=horizon).to(device)

    # ---------------- 2. Loss and optimizer ----------------
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---------------- 3. Data loader ----------------
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)), batch_size=batch_size, shuffle=True, drop_last=False)

    # ---------------- 4. Training ----------------
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            yhat = model(bx)                  # (B, H)
            loss = loss_fn(yhat, by)          # MSE over (B,H)
            loss.backward()
            optimizer.step()
            total += loss.item() * bx.size(0)

        mse = total / len(loader.dataset)
        rmse_norm = math.sqrt(mse)

        # Report RMSE in target (data) space and as basis-points / % of σ
        if norm_params["type"] == "standard":
            sigma = norm_params["std"]
            rmse_data = rmse_norm * sigma
        else:
            rmse_data = rmse_norm * (norm_params["max"] - norm_params["min"])
            sigma = float(np.std(work_train) + 1e-12)

        rmse_bp = 1e4 * rmse_data
        pct_of_sigma = 100.0 * rmse_data / sigma

        print(f"Epoch {epoch:4d}/{epochs} | MSE(norm): {mse:.6f} | "
              f"RMSE(target)= {rmse_data:.6f}  (~{rmse_bp:.1f} bp) | "
              f"RMSE ≈ {pct_of_sigma:.1f}% of σ(returns)")

    # ---------------- 5. Save the data ----------------
    torch.save(model.state_dict(), outdir / "model_state.pt")
    config = {
        "feature": args.feature,
        "target": tgt,
        "horizon": horizon,
        "seq_len": seq_len,
        "hidden_len": hidden_len,
        "num_layers": num_layers,
        "dropout": dropout,
        "cutoff": cutoff,
        "normalize": norm_params,
        "optimizer": "adam",
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "csv_path": str(Path(args.csv).resolve()),
        "n_price": Np,
        "n_work": Nw,
        "cutoff_idx_price": cutoff_idx_price,
        "cutoff_idx_work": cutoff_idx_work,
        "date_column_present": "Date" in df.columns,
        "last_observed_price": float(prices[cutoff_idx_price - 1]),
    }
    save_json(outdir / "config.json", config)

    # ---------------- 6. Prepare data for first horizon step ----------------
    model.eval()
    with torch.no_grad():
        Yhat_train_norm = model(torch.from_numpy(X_train).to(device)).cpu().numpy()  # (N_train, H)

    first_step_norm = Yhat_train_norm[:, 0]     # (N_train,)
    first_step = denorm(first_step_norm)        # back to target space

    # Map to PRICES at next date for each label (reconstruct +1 step)
    idxs = label_indices[train_mask]            # aligned with X_train/Y_train
    if tgt == "price":
        yhat_price = first_step
        plot_dates = label_dates[train_mask]    # aligned dates (fix)
    else:
        yhat_price = []
        for i, v in zip(idxs, first_step):
            base_price = prices[i]              # price at time i (step BEFORE first horizon)
            if tgt == "logreturn":
                p = base_price * np.exp(v)
            else:  # delta
                p = base_price + v
            yhat_price.append(p)
        yhat_price = np.array(yhat_price)
        plot_dates = label_dates[train_mask]    # aligned dates (fix)

    pd.DataFrame({"Date": plot_dates, "y_hat": yhat_price}).to_csv(outdir / "train_preds.csv", index=False)

    print("\nSaved:")
    print(f"  - {outdir/'model_state.pt'}")
    print(f"  - {outdir/'config.json'}")
    print(f"  - {outdir/'train_preds.csv'}  (first-horizon fitted PRICES over training span)")

if __name__ == "__main__":
    main()