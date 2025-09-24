# scripts/quick_report.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SYMBOL = "CL"  # or "NG"
H = 30         # 5, 15, or 30 to match your backtest file

def main():
    bt_path = OUT / f"bt_{SYMBOL}_{H}m.parquet"
    if not bt_path.exists():
        print(f"[err] missing {bt_path}. Run backtest first.")
        return

    bt = pd.read_parquet(bt_path)
    bt["timestamp"] = pd.to_datetime(bt["timestamp"], utc=True)
    bt = bt.sort_values("timestamp").reset_index(drop=True)
    equity = bt["equity"].astype(float)

    # --- Time context & CAGR ---
    t0, t1 = bt["timestamp"].iloc[0], bt["timestamp"].iloc[-1]
    days = max((t1 - t0).days, 1)
    years = days / 365.25
    cagr = float(equity.iloc[-1] ** (1/years) - 1) if years > 0 else np.nan

    # median bar interval in minutes (for your info only)
    if len(bt) > 2:
        step_min = np.median(np.diff(bt["timestamp"]).astype("timedelta64[m]").astype(float))
    else:
        step_min = np.nan

    # --- Daily returns from equity (robust) ---
    eq_daily = equity.copy()
    eq_daily.index = bt["timestamp"]
    eq_daily = eq_daily.resample("1D").last().dropna()
    daily_ret = eq_daily.pct_change().dropna()

    # Daily Sharpe (252d)
    if len(daily_ret) > 2 and np.nanstd(daily_ret) > 0:
        d_sharpe = float(np.nanmean(daily_ret) / (np.nanstd(daily_ret, ddof=1) + 1e-9) * np.sqrt(252))
    else:
        d_sharpe = np.nan

    # Max drawdown from equity
    roll_max = equity.cummax()
    dd = (equity / roll_max - 1).min()

    print(f"Summary for {SYMBOL} {H}m")
    print(f"  Daily Sharpe (net): {d_sharpe:.3f}")
    print(f"  CAGR (from equity): {cagr:.2%}")
    print(f"  Max drawdown:       {dd:.2%}")
    print(f"  Bars:               {len(bt)} | step≈{step_min:.1f} min")

    # Plot equity
    fig, ax = plt.subplots()
    ax.plot(bt["timestamp"], equity)
    ax.set_title(f"Equity Curve — {SYMBOL} {H}m")
    ax.set_xlabel("Time"); ax.set_ylabel("Equity (normalized)")
    fig.autofmt_xdate()
    png = OUT / f"equity_{SYMBOL}_{H}m.png"
    plt.savefig(png, bbox_inches="tight", dpi=120)
    print(f"[ok] saved {png}")

if __name__ == "__main__":
    main()
