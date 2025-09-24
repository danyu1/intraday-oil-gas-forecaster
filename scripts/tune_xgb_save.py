from pathlib import Path
import json, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"
OUT  = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)

SYMBOL = "CL"
HORIZONS = [5, 15, 30]
N_SPLITS = 5
N_TRIALS = 30
SEED = 42
rng = np.random.default_rng(SEED)

def sharpe(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    if a.size < 2: return -1e9
    return float(np.nanmean(a) / (np.nanstd(a, ddof=1) + 1e-9))

def load_features(sym):
    df = pd.read_parquet(FEAT / f"{sym}.parquet").sort_values("timestamp").reset_index(drop=True)
    return df

def build_xy(df, h):
    drop = {"timestamp","symbol","open","high","low","close","volume"}
    drop |= {c for c in df.columns if c.startswith("tgt_")}
    X = df[[c for c in df.columns if c not in drop]]
    y = df[f"tgt_ret_{h}m"]
    m = y.notna()
    ds = pd.to_datetime(df.loc[m, "timestamp"]).values
    return X[m].to_numpy(), y[m].to_numpy(), ds

def sample_params():
    return {
        "max_depth": int(rng.integers(3, 9)),
        "eta": float(rng.uniform(0.01, 0.15)),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "min_child_weight": float(rng.uniform(1.0, 10.0)),
        "lambda": float(rng.uniform(0.0, 10.0)),  # L2
        "alpha": float(rng.uniform(0.0, 5.0)),    # L1
    }

def oof_with_params(X, y, params, seed=SEED):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof = np.full_like(y, np.nan, dtype=float)
    for tr, te in tscv.split(X):
        dtr = xgb.DMatrix(X[tr], label=y[tr]); dte = xgb.DMatrix(X[te])
        full = {"objective":"reg:squarederror", "eval_metric":"rmse", "nthread":0, "seed": seed, **params}
        bst = xgb.train(full, dtr, num_boost_round=1000, verbose_eval=False)
        oof[te] = bst.predict(dte)
    return oof

def main():
    df = load_features(SYMBOL)
    best_all = {}
    for h in HORIZONS:
        X, y, ds = build_xy(df, h)
        if len(y) < N_SPLITS + 1:
            print(f"[warn] not enough rows for {h}m"); continue
        best = (-1e9, None, None)
        for t in range(N_TRIALS):
            p = sample_params()
            oof = oof_with_params(X, y, p)
            sr = sharpe(oof)
            if sr > best[0]: best = (sr, p, oof)
            if (t+1) % 5 == 0:
                print(f"[{SYMBOL} {h}m] trial {t+1}/{N_TRIALS} best Sharpe={best[0]:.3f}")
        sr, pbest, oof = best
        best_all[str(h)] = pbest
        outp = OUT / f"oof_xgb_{SYMBOL}_{h}m.parquet"
        pd.DataFrame({"ds": ds, "y": y, "pred": oof}).to_parquet(outp, index=False)
        print(f"[ok] saved tuned OOF -> {outp} | best Sharpe={sr:.3f} | params={pbest}")

    # save params JSON
    params_path = OUT / f"xgb_best_params_{SYMBOL}.json"
    params_path.write_text(json.dumps(best_all, indent=2))
    print(f"[ok] wrote {params_path}")

if __name__ == "__main__":
    main()
