from pathlib import Path


# Base directories (adjust if needed)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"


# Default spread parquet files produced by iq_spreads_ingest.py
SPREAD_FILES = {
"CL_SPREAD": PROCESSED_DIR / "CL1-CL2_SPREAD_300s.parquet",
"NG_SPREAD": PROCESSED_DIR / "NG1-NG2_SPREAD_300s.parquet",
}


# Virtual symbols → human-friendly names (for plots, logs)
SYMBOL_TITLES = {
"CL_SPREAD": "CL1–CL2 Calendar Spread",
"NG_SPREAD": "NG1–NG2 Calendar Spread",
}


# Model horizons used across scripts (in 5-minute bars)
FORWARD_HORIZONS = (1, 3, 6)