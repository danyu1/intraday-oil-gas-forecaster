# scripts/train_tcn_oof.py
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import torch, torch.nn as nn, torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"
OUT  = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)

SYMBOL = "CL"             # set "NG" for NG
HORIZONS = [5, 15, 30]
N_SPLITS = 5
CTX = 512
MAX_EPOCHS = 80
PATIENCE = 5
BATCH = 256
LR = 3e-3
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

def load_features(sym: str):
    p = FEAT / f"{sym}.parquet"
    if not p.exists(): raise FileNotFoundError(f"Missing {p}. Run make_features.py first.")
    df = pd.read_parquet(p).sort_values("timestamp").reset_index(drop=True)
    # features matrix
    drop = {"timestamp","symbol","open","high","low","close","volume"}
    drop |= {c for c in df.columns if c.startswith("tgt_")}
    X = df[[c for c in df.columns if c not in drop]].astype("float32")
    ts = pd.to_datetime(df["timestamp"])
    return df, X, ts

def make_xy_windows(X: np.ndarray, y: np.ndarray, ts: np.ndarray, ctx: int):
    n = len(y)
    idx = np.arange(ctx, n)
    Xw = np.stack([X[i-ctx:i] for i in idx], axis=0)
    yw = y[idx]
    dsw = ts[idx]  # timestamps aligned to the target at time t
    return Xw, yw, dsw, idx

class Chomp1d(nn.Module):
    def __init__(self, c): super().__init__(); self.c = c
    def forward(self, x): return x[:, :, :-self.c] if self.c > 0 else x

class TCNBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, d=1, drop=0.1):
        super().__init__()
        pad = (k-1)*d
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, k, padding=pad, dilation=d), Chomp1d(pad), nn.ReLU(), nn.Dropout(drop),
            nn.Conv1d(out_c, out_c, k, padding=pad, dilation=d), Chomp1d(pad), nn.ReLU(), nn.Dropout(drop),
        )
        self.down = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x): return nn.functional.relu(self.net(x) + self.down(x))

class TCN(nn.Module):
    def __init__(self, in_feats, chans=(32,32,32), k=3, drop=0.1):
        super().__init__()
        layers, c_in = [], in_feats
        for i, c_out in enumerate(chans):
            layers.append(TCNBlock(c_in, c_out, k=k, d=2**i, drop=drop)); c_in = c_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Conv1d(c_in, 1, 1), nn.Flatten())
        self.pool = nn.AdaptiveAvgPool1d(1); self.final = nn.Linear(1,1)
    def forward(self, x):
        x = x.transpose(1,2); z = self.tcn(x); z = self.head(z)
        z = z.unsqueeze(1); z = self.pool(z).squeeze(-1)
        return self.final(z).squeeze(-1)

def train_epoch(model, opt, crit, X, y):
    model.train(); idx = np.random.permutation(len(y)); total = 0.0
    for s in range(0, len(y), BATCH):
        b = idx[s:s+BATCH]; xb = torch.from_numpy(X[b]); yb = torch.from_numpy(y[b])
        opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        total += float(loss.detach()) * len(b)
    return total / len(y)

@torch.no_grad()
def eval_loss(model, crit, X, y):
    model.eval(); losses = []
    for s in range(0, len(y), BATCH):
        xb = torch.from_numpy(X[s:s+BATCH]); yb = torch.from_numpy(y[s:s+BATCH])
        losses.append(float(crit(model(xb), yb)) * len(yb))
    return sum(losses) / max(1, len(y))

@torch.no_grad()
def predict(model, X):
    model.eval(); out = []
    for s in range(0, len(X), BATCH):
        xb = torch.from_numpy(X[s:s+BATCH]); out.append(model(xb).numpy())
    return np.concatenate(out, 0)

def standardize_train_test(Xtr, Xte):
    mu = np.nanmean(Xtr, 0, keepdims=True); sd = np.nanstd(Xtr, 0, keepdims=True); sd[sd==0]=1.0
    return (Xtr-mu)/sd, (Xte-mu)/sd

def main():
    df, Xdf, ts_all = load_features(SYMBOL)

    for h in HORIZONS:
        y_full = df[f"tgt_ret_{h}m"].to_numpy().astype("float32")
        mask_valid = np.isfinite(y_full)
        for col in Xdf.columns:
            mask_valid &= np.isfinite(Xdf[col].to_numpy())
        X = Xdf.to_numpy().astype("float32")[mask_valid]
        y = y_full[mask_valid]
        ts = ts_all[mask_valid].to_numpy()

        if len(y) <= CTX + 100:
            print(f"[warn] too few rows for {SYMBOL} {h}m"); continue

        Xw, yw, dsw, idx = make_xy_windows(X, y, ts, CTX)

        oof = np.full_like(yw, np.nan, dtype="float32")
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        fold = 0
        for tr_idx, te_idx in tscv.split(Xw):
            fold += 1
            Xtr, ytr = Xw[tr_idx], yw[tr_idx]
            Xte, yte = Xw[te_idx], yw[te_idx]

            Xtr_s, Xte_s = standardize_train_test(
                Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
            )
            nfeat = Xtr.shape[2]
            Xtr_s = Xtr_s.reshape(len(Xtr), CTX, nfeat)
            Xte_s = Xte_s.reshape(len(Xte), CTX, nfeat)

            model = TCN(in_feats=nfeat, chans=(32,32,32), k=3, drop=0.1)
            opt = optim.Adam(model.parameters(), lr=LR)
            crit = nn.SmoothL1Loss()

            best = (np.inf, None); patience = 0
            for ep in range(1, MAX_EPOCHS+1):
                _ = train_epoch(model, opt, crit, Xtr_s, ytr)
                va = eval_loss(model, crit, Xte_s, yte)
                if va < best[0] - 1e-6:
                    best = (va, {k: v.cpu().detach().clone() for k, v in model.state_dict().items()})
                    patience = 0
                else:
                    patience += 1
                if patience >= PATIENCE: break
            if best[1] is not None:
                model.load_state_dict(best[1])

            oof[te_idx] = predict(model, Xte_s)
            print(f"[{SYMBOL} {h}m] fold {fold} train={len(tr_idx)} test={len(te_idx)} best_val={best[0]:.6f}")

        # Save WITH ds so ensemble can align by time
        mask = np.isfinite(oof)
        out = pd.DataFrame({"ds": dsw[mask], "y": yw[mask], "pred": oof[mask]})
        outp = OUT / f"oof_dl_tcn_{SYMBOL}_{h}m.parquet"
        out.to_parquet(outp, index=False)
        print(f"[ok] wrote {outp} rows={len(out)}")

if __name__ == "__main__":
    main()
