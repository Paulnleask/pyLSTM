# 📈 pyLSTM

A lightweight **Python + PyTorch** framework for **time-series forecasting** using **Long Short-Term Memory (LSTM)** neural networks.  
It enables end-to-end stock prediction — from fetching historical data to model training, forecasting, and visualization.

---

## 🧩 Features
- Fetch financial time series from **Yahoo! Finance**  
- Train multi-horizon **LSTM regressors** on stock data  
- Forecast future prices directly (no recursive predictions)  
- Configurable normalization (standard or min–max)  
- Visualize in-sample fits and out-of-sample forecasts  
- Generates clean CSV artifacts and plots for reproducibility  

---

## 📦 Requirements

| Dependency | Minimum Version | Notes |
|-------------|----------------|--------|
| **Python** | 3.9+ | Required |
| **PyTorch** | 1.12+ | Deep learning backend |
| **pandas** | — | Data handling |
| **numpy** | — | Numerical operations |
| **matplotlib** | — | Plotting and visualization |
| **yfinance** | — | Yahoo! Finance data source |

---

## 🛠️ Building and running the Project

```bash
# 1️⃣ Clone the repository
git clone https://github.com/Paulnleask/pyLSTM.git
cd pyLSTM

# 2️⃣ Fetch stock from yfinance (e.g. PLTR)
python fetch_yahoo_csv.py --ticker PLTR --start 2015-01-01 --out PLTR.csv --outdir data

# 3️⃣ Train model on (e.g. Close) prices from stock
python train_lstm.py --csv data/PLTR.csv --feature Close --target logreturn --normalize standard --outdir artifacts

# 4️⃣ Predict future stock prices with model parameters from training
python predict_lstm.py --csv data/PLTR.csv --artifacts artifacts

# 5️⃣ Plot the historical data, training prediction and testin prediction
python plot_series.py --csv data/PLTR.csv --artifacts artifacts
