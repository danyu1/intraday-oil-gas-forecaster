# scripts/backtest.py
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"
OUT  = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)

SYMBOL = "CL"
H = 30  # 5, 15, 30

KELLY = 0.03
COMMISSION_PER_CONTRACT = 2.0
SLIPPAGE_TICKS = 1
TICK_VALUE = {"CL": 10.0, "NG": 12.5}[SYMBOL]
TICK_SIZE  = {"CL": 0.01, "NG": 0.001}[SYMBOL]
COST_RET = (SLIPPAGE_TICKS * TICK_VALUE + COMMISSION_PER_CONTRACT) / 10000.0

def size_positions(sig, kelly=KELLY):
    s = np.asarray(sig, float)
    vol = np.nanstd(s); vol = 1.0 if (not np.isfinite(vol) or vol == 0) else vol
    return np.clip(kelly * (s / vol), -1.0, 1.0)

def next_h_log_returns(prices: np.ndarray, h: int):
    logp = np.log(prices.astype(float)); r = np.zeros_like(logp)
    if 0 < h < len(logp): r[:-h] = logp[h:] - logp[:-h]
    return r

def main():
    feat_path = FEAT / f"{SYMBOL}.parquet"
    ens_path  = OUT / f"ens_{SYMBOL}_{H}m.parquet"
    if not feat_path.exists() or not ens_path.exists():
        print(f"[err] missing: {feat_path} or {ens_path}"); return

    feat = pd.read_parquet(feat_path); feat["timestamp"] = pd.to_datetime(feat["timestamp"], utc=True)
    ens  = pd.read_parquet(ens_path);  ens["ds"]        = pd.to_datetime(ens["ds"], utc=True)

    # Merge on timestamp
    df = pd.merge(feat, ens, left_on="timestamp", right_on="ds", how="inner").sort_values("timestamp").reset_index(drop=True)
    if len(df) < 50:
        print("[err] not enough aligned rows"); return

    prices = df["close"].to_numpy()
    preds  = df["yhat"].to_numpy()     # prediction of r_t(H)
    pos = size_positions(preds)
    realized_h = next_h_log_returns(prices, H)

    pnl_gross = pos * realized_h
    pos_change = np.abs(np.diff(np.concatenate([[0.0], pos])))
    pnl_net = pnl_gross - pos_change * COST_RET

    equity = np.cumprod(1 + np.nan_to_num(pnl_net, nan=0.0))
    out = pd.DataFrame({"timestamp": df["timestamp"], "pos": pos, "ret_gross": pnl_gross, "ret_net": pnl_net, "equity": equity})
    op = OUT / f"bt_{SYMBOL}_{H}m.parquet"
    out.to_parquet(op, index=False)
    sr = float(np.nanmean(pnl_net) / (np.nanstd(pnl_net, ddof=1) + 1e-9))
    print(f"[ok] saved backtest -> {op} | rows={len(out)} | Sharpe≈{sr:.3f} (net)")

if __name__ == "__main__":
    main()
