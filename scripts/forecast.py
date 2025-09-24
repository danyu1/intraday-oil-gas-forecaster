import argparse, sys, traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .data_loader import load_series, add_returns_and_targets
from .constants import SYMBOL_TITLES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="CL_SPREAD", choices=["CL_SPREAD","NG_SPREAD"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--days", type=int, default=100)
    ap.add_argument("--out", default="outputs/plot_spread.png")
    args = ap.parse_args()

    df = load_series(args.symbol, args.path)
    df = add_returns_and_targets(df)

    last = df.tail(args.days * (24*12))  # approx bars in N days (5m bars ≈ 12 per hour * 24)

    # Toy forecast: moving average delta just for visualization
    pred = last["close"].rolling(12).mean().shift(1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(last["timestamp"], last["close"], label="Spread Close")
    ax.plot(last["timestamp"], pred, label="Toy Forecast")
    ax.set_title(SYMBOL_TITLES.get(args.symbol, args.symbol))
    ax.legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("Saved", args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
