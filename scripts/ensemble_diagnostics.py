import argparse, json, traceback, sys
import numpy as np
import pandas as pd

from .data_loader import load_series, add_returns_and_targets


def safe_get_feature_names(scaler_bundle) -> list:
    try:
        names = getattr(scaler_bundle.get("scaler", None), "feature_names_in_", None)
        if names is None:
            return []
        # Convert numpy array -> list explicitly (fixes ambiguous truth value error)
        return list(names.tolist() if hasattr(names, "tolist") else list(names))
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="CL_SPREAD", choices=["CL_SPREAD","NG_SPREAD"], help="virtual symbol")
    ap.add_argument("--path", default=None)
    args = ap.parse_args()

    df = load_series(args.symbol, args.path)
    df = add_returns_and_targets(df)

    # Toy example: two model pseudo-preds to illustrate correlation/sharpe calc
    y = df["fwd_ret_3"].values
    x1 = np.sign(df["ret_5m"].values) * 0.0001   # dummy signal 1
    x2 = -x1                                      # dummy signal 2

    corr = np.corrcoef(x1, x2)[0,1]
    def sharpe(x):
        mu, sd = x.mean(), x.std() + 1e-9
        return mu/sd

    print(f"=== {args.symbol} sample | rows={len(df)} ===")
    print(f"  Model1 Sharpe: {sharpe(x1):7.3f}")
    print(f"  Model2 Sharpe: {sharpe(x2):7.3f}")
    print("  Corr matrix:\n", pd.DataFrame(np.corrcoef(np.vstack([x1,x2])), index=["M1","M2"], columns=["M1","M2"]))

    # Save a small JSON so downstream scripts can read
    out = {
        "rows": int(len(df)),
        "corr": float(corr),
    }
    with open("outputs/ensemble_diag.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved outputs/ensemble_diag.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
