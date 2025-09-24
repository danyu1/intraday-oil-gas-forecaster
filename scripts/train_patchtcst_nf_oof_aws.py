from __future__ import annotations
import os, sys, argparse, math, time, json, traceback
from pathlib import Path
import re
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Torch + stability guards (helps avoid OOM / "Killed" on CPU boxes)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch, torch.nn as nn, torch.optim as optim
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

try:
    import boto3
except Exception:
    boto3 = None

# --------- Paths (support SageMaker + local) ---------
ROOT = Path(__file__).resolve().parents[1]

# If running in SageMaker with a channel named "features", SM will mount it here:
SM_CHANNEL_FEATURES = os.getenv("SM_CHANNEL_FEATURES")  # e.g. /opt/ml/input/data/features
# Local fallback (repo)
FEAT_LOCAL = ROOT / "data" / "features"

def feat_dir() -> Path:
    p = Path(SM_CHANNEL_FEATURES) if SM_CHANNEL_FEATURES else FEAT_LOCAL
    p.mkdir(parents=True, exist_ok=True)
    return p

def sm_out_dir() -> Path:
    d = os.getenv("SM_MODEL_DIR") or str(ROOT / "outputs")
    Path(d).mkdir(parents=True, exist_ok=True)
    return Path(d)

OUT_LOCAL = ROOT / "outputs"  # kept for prints
N_SPLITS = 5

# --------- Helpers ---------
def parse_args():
    p = argparse.ArgumentParser()
    # data / task
    p.add_argument("--symbol", type=str, default="CL")  # keep same default
    p.add_argument("--horizons", type=str, default="5,15,30")
    p.add_argument("--ctx", type=int, default=512, help="lookback context length")
    p.add_argument("--features_s3", type=str, default="", help="s3://bucket/data/features/CL.parquet")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits", type=int, default=N_SPLITS)
    p.add_argument("--features_channel", type=str, default=os.getenv("SM_CHANNEL_FEATURES", ""), help="explicit path to mounted features channel")

    # optimization
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=8)

    # PatchTST-ish model hparams
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patch_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)

    # misc
    p.add_argument("--save_ckpt", action="store_true", help="save best checkpoint per horizon")
    p.add_argument("--resume", action="store_true", help="(reserved) resume if ckpt exists")
    return p.parse_args()

def set_seed(seed: int):
    torch.manual_seed(seed); np.random.seed(seed)

def parse_s3_uri(uri: str):
    """
    Parse s3://bucket/key -> (bucket, key)
    """
    m = re.match(r"^s3://([^/]+)/(.+)$", uri.strip())
    if not m:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return m.group(1), m.group(2)

def s3_download_file(uri: str, local_path: Path):
    """
    Download a single file from S3 via boto3.
    """
    if not uri:
        return
    if boto3 is None:
        raise RuntimeError("boto3 is not available in this environment; cannot use --features_s3.")
    bucket, key = parse_s3_uri(uri)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(local_path))

def maybe_pull_features_from_s3(symbol: str, s3_uri: str, target_dir: Path):
    if not s3_uri:
        return
    local = target_dir / f"{symbol}.parquet"
    print(f"[S3] downloading {s3_uri} -> {local}", flush=True)
    s3_download_file(s3_uri, local)

def load_features(sym: str, channel_path: str = ""):
    base = Path(channel_path) if channel_path else feat_dir()
    p = base / f"{sym}.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. If data is in S3, pass --features_s3 s3://.../{sym}.parquet "
            f"or mount a SageMaker channel named 'features' with {sym}.parquet inside."
        )
    df = pd.read_parquet(p).sort_values("timestamp").reset_index(drop=True)
    drop = {"timestamp","symbol","open","high","low","close","volume"}
    drop |= {c for c in df.columns if c.startswith("tgt_")}
    X = df[[c for c in df.columns if c not in drop]].astype("float32")
    ts = pd.to_datetime(df["timestamp"])
    return df, X, ts

def standardize_train_test(Xtr: np.ndarray, Xte: np.ndarray):
    mu = np.nanmean(Xtr, 0, keepdims=True); sd = np.nanstd(Xtr, 0, keepdims=True)
    sd[sd==0] = 1.0
    return (Xtr-mu)/sd, (Xte-mu)/sd

def make_xy_windows(X: np.ndarray, y: np.ndarray, ts: np.ndarray, ctx: int):
    n = len(y); idx = np.arange(ctx, n)
    Xw = np.stack([X[i-ctx:i] for i in idx], axis=0)     # [N, ctx, feats]
    yw = y[idx]                                          # [N]
    dsw = ts[idx]                                        # timestamps aligned to target
    return Xw, yw, dsw, idx

# --------- Minimal PatchTST-style model ---------
class PatchEmbedding(nn.Module):
    def __init__(self, in_feats: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(in_feats * patch_len, d_model)

    def forward(self, x):  # x: [B, L, F]
        B, L, F = x.shape
        # unfold along time dimension (simple strided patching)
        num_patches = 1 + (L - self.patch_len) // self.stride
        patches = []
        for i in range(num_patches):
            s = i * self.stride
            e = s + self.patch_len
            patches.append(x[:, s:e, :].reshape(B, -1))  # [B, F*patch_len]
        P = torch.stack(patches, dim=1)  # [B, P, F*patch_len]
        return self.proj(P)              # [B, P, d_model]

class PatchTSTNF(nn.Module):
    def __init__(self, in_feats, d_model=128, n_heads=8, n_layers=4, dropout=0.1, patch_len=16, stride=8):
        super().__init__()
        self.embed = PatchEmbedding(in_feats, patch_len, stride, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
                                               dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1))  # will apply to [B, d_model, P] later
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):  # x: [B, L, F]
        z = self.embed(x)             # [B, P, d_model]
        z = self.encoder(z)           # [B, P, d_model]
        z = z.transpose(1, 2)         # [B, d_model, P]
        z = self.head(z).squeeze(-1)  # [B, d_model]
        return self.fc(z).squeeze(-1) # [B]

# --------- Train / Eval ---------
def train_epoch(model, opt, crit, X, y, batch, device):
    model.train(); total = 0.0
    idx = np.random.permutation(len(y))
    for s in range(0, len(y), batch):
        b = idx[s:s+batch]
        xb = torch.from_numpy(X[b]).to(device, non_blocking=True)
        yb = torch.from_numpy(y[b]).to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        total += float(loss.detach()) * len(b)
    return total / max(1, len(y))

@torch.no_grad()
def eval_loss(model, crit, X, y, batch, device):
    model.eval(); total = 0.0
    for s in range(0, len(y), batch):
        xb = torch.from_numpy(X[s:s+batch]).to(device, non_blocking=True)
        yb = torch.from_numpy(y[s:s+batch]).to(device, non_blocking=True)
        total += float(crit(model(xb), yb)) * len(yb)
    return total / max(1, len(y))

@torch.no_grad()
def predict(model, X, batch, device):
    model.eval(); out = []
    for s in range(0, len(X), batch):
        xb = torch.from_numpy(X[s:s+batch]).to(device, non_blocking=True)
        out.append(model(xb).detach().cpu().numpy())
    return np.concatenate(out, 0)

# --------- Main ---------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT = sm_out_dir()
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Prefer an explicitly provided channel path, then SM env, then local
    feat_base = Path(args.features_channel) if args.features_channel else feat_dir()

    print(f"[env] device={device} SM_MODEL_DIR={OUT} CWD={os.getcwd()}", flush=True)
    print(f"[env] features_dir={feat_base}", flush=True)

    # Optionally pull a single parquet from S3 (helpful if not using channels)
    if args.features_s3:
        try:
            maybe_pull_features_from_s3(args.symbol, args.features_s3, feat_base)
        except Exception as e:
            print(f"[warn] S3 download failed: {e}", flush=True)

    df, Xdf, ts_all = load_features(args.symbol, channel_path=str(feat_base))

    # ensure finite rows across features + targets per horizon
    all_cols_ok = np.ones(len(df), dtype=bool)
    for col in Xdf.columns:
        all_cols_ok &= np.isfinite(Xdf[col].to_numpy())

    # Adjust default batch size on tiny CPU instances (avoid OOM)
    if device == "cpu" and args.batch > 1024:
        args.batch = 1024

    for h in [int(x) for x in args.horizons.split(",") if x.strip()]:
        # Targets named tgt_ret_{h}m (unchanged)
        y_full = df.get(f"tgt_ret_{h}m", pd.Series(np.nan, index=df.index)).to_numpy().astype("float32")
        mask = all_cols_ok & np.isfinite(y_full)
        X_np = Xdf.to_numpy().astype("float32")[mask]
        y = y_full[mask]
        ts = ts_all[mask].to_numpy()

        if len(y) <= args.ctx + 100:
            print(f"[warn] too few rows for {args.symbol} {h}m (rows={len(y)})"); continue

        # windows
        Xw, yw, dsw, idx = make_xy_windows(X_np, y, ts, args.ctx)
        # OOF
        oof = np.full_like(yw, np.nan, dtype="float32")
        tscv = TimeSeriesSplit(n_splits=args.splits)
        fold = 0
        best_ckpt = None
        best_val = math.inf

        for tr_idx, te_idx in tscv.split(Xw):
            fold += 1
            Xtr, ytr = Xw[tr_idx], yw[tr_idx]
            Xte, yte = Xw[te_idx], yw[te_idx]

            # standardize on flattened features, then reshape back
            nfeat = Xtr.shape[2]
            Xtr_s, Xte_s = standardize_train_test(
                Xtr.reshape(len(Xtr), -1),
                Xte.reshape(len(Xte), -1)
            )
            Xtr_s = Xtr_s.reshape(len(Xtr), args.ctx, nfeat)
            Xte_s = Xte_s.reshape(len(Xte), args.ctx, nfeat)

            model = PatchTSTNF(
                in_feats=nfeat,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                dropout=args.dropout,
                patch_len=args.patch_len,
                stride=args.stride,
            ).to(device)

            opt = optim.Adam(model.parameters(), lr=args.lr)
            crit = nn.SmoothL1Loss()

            best = (math.inf, None); patience = 0
            for ep in range(1, args.max_epochs + 1):
                tr_loss = train_epoch(model, opt, crit, Xtr_s, ytr, args.batch, device)
                va_loss = eval_loss(model, crit, Xte_s, yte, args.batch, device)
                improved = va_loss < (best[0] - 1e-6)
                if improved:
                    best = (va_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                    patience = 0
                else:
                    patience += 1
                if ep == 1 or ep % 5 == 0 or improved:
                    print(f"[{args.symbol} {h}m] fold {fold} ep {ep} tr={tr_loss:.6f} va={va_loss:.6f} best={best[0]:.6f}", flush=True)
                if patience >= args.patience:
                    break

            if best[1] is not None:
                model.load_state_dict(best[1])

            if best[0] < best_val:
                best_val = best[0]
                best_ckpt = best[1]

            oof[te_idx] = predict(model, Xte_s, args.batch, device)

        # Save OOF with timestamps (for ensemble alignment) -> model artifacts dir
        mask_o = np.isfinite(oof)
        out = pd.DataFrame({"ds": dsw[mask_o], "y": yw[mask_o], "pred": oof[mask_o]})
        outp = sm_out_dir() / f"oof_dl_patchtst_{args.symbol}_{h}m.parquet"
        out.to_parquet(outp, index=False)
        print(f"[ok] wrote {outp} rows={len(out)}", flush=True)

        # Optional: save best checkpoint across folds (also into SM_MODEL_DIR)
        if args.save_ckpt and best_ckpt is not None:
            ckpt_path = sm_out_dir() / f"patchtst_best_{args.symbol}_{h}m.pt"
            torch.save(best_ckpt, ckpt_path)
            print(f"[ok] saved checkpoint {ckpt_path}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[FATAL ERROR]")
        traceback.print_exc()
        sys.exit(1)
