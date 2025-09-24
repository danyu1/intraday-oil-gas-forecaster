from pathlib import Path
import json, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"
OUT  = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)

SYMBOL = "CL"
HORIZONS = [5, 15, 30]
N_SPLITS = 5
SEED = 42

def load_best_params(symbol, h):
    p = OUT / f"xgb_best_params_{symbol}.json"
    if p.exists():
        d = json.loads(p.read_text())
        if str(h) in d: return d[str(h)]
    # defaults if no cache yet
    return {"max_depth":6, "eta":0.05, "subsample":0.8, "colsample_bytree":0.8,
            "min_child_weight":1.0, "lambda":0.0, "alpha":0.0}

def load_features(sym: str):
    df = pd.read_parquet(FEAT / f"{sym}.parquet").sort_values("timestamp").reset_index(drop=True)
    return df

def build_xy(df: pd.DataFrame, h: int):
    drop = {"timestamp","symbol","open","high","low","close","volume"}
    drop |= {c for c in df.columns if c.startswith("tgt_")}
    X = df[[c for c in df.columns if c not in drop]]
    y = df[f"tgt_ret_{h}m"]; m = y.notna()
    ds = pd.to_datetime(df.loc[m, "timestamp"]).values
    return X[m].to_numpy(), y[m].to_numpy(), ds

def main():
    np.random.seed(SEED)
    df = load_features(SYMBOL)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    for h in HORIZONS:
        X, y, ds = build_xy(df, h)
        if len(y) < N_SPLITS + 1:
            print(f"[warn] insufficient rows for {h}m"); continue

        params = load_best_params(SYMBOL, h)
        oof = np.full(y.shape, np.nan, dtype=float)
        for tr, te in tscv.split(X):
            dtr = xgb.DMatrix(X[tr], label=y[tr]); dte = xgb.DMatrix(X[te])
            full = {"objective":"reg:squarederror","eval_metric":"rmse","nthread":0,"seed":SEED, **params}
            bst = xgb.train(full, dtr, num_boost_round=800)
            oof[te] = bst.predict(dte)
        outp = OUT / f"oof_xgb_{SYMBOL}_{h}m.parquet"
        pd.DataFrame({"ds": ds, "y": y, "pred": oof}).to_parquet(outp, index=False)
        print(f"[ok] OOF -> {outp} (using params for {h}m: {params})")

if __name__ == "__main__":
    main()
