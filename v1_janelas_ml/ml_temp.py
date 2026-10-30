import re

import os
import sys
import json
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return default if v is None or str(v).strip()=="" else int(float(v))

def env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return default if v is None or str(v).strip()=="" else float(v)

def clean_path(s: str) -> str:
    if s is None:
        return None
    return str(s).strip().strip('"').strip("'").strip()

def load_gran(path: str):
    path = clean_path(path)
    if not path:
        raise RuntimeError("DF_INPUT_PATH not set.")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DF_INPUT_PATH not found: {p}")

    df = pd.read_csv(p)
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least 2 columns (signals + timestamp).")

    tcol = df.columns[-1]
    df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol).reset_index(drop=True)

    ts = df[tcol].to_numpy(dtype=float)
    X = df.drop(columns=[tcol]).copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    return X, ts, tcol

def infer_step(ts: np.ndarray) -> float:
    if len(ts) < 2:
        return 1.0
    diffs = np.diff(ts)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 1.0
    # use median for robustness
    return float(np.median(diffs))

def make_windows(arr: np.ndarray, L: int):
    # arr: (T, D)
    T, D = arr.shape
    n = T - L
    if n <= 0:
        return None, None
    Xw = np.zeros((n, L*D), dtype=np.float32)
    Yw = np.zeros((n, D), dtype=np.float32)
    for i in range(n):
        Xw[i] = arr[i:i+L].reshape(-1)
        Yw[i] = arr[i+L]
    return Xw, Yw

def recursive_forecast(model, arr: np.ndarray, L: int, H: int):
    T, D = arr.shape
    hist = arr.copy()
    preds = []
    for _ in range(H):
        x = hist[-L:].reshape(1, -1)
        y = model.predict(x)
        y = np.asarray(y).reshape(1, D)
        preds.append(y[0])
        hist = np.vstack([hist, y])
    return np.asarray(preds, dtype=np.float32)

def naive_forecast(arr: np.ndarray, H: int):
    last = arr[-1]
    return np.repeat(last.reshape(1,-1), repeats=H, axis=0).astype(np.float32)

def save_forecast(out_dir: Path, stem: str, cols, ts_hist, ts_future, y_hist, y_pred, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    # CSV
    df_out = pd.DataFrame(y_pred, columns=cols)
    df_out["timestamp"] = ts_future
    df_out.to_csv(out_dir / f"{prefix}__{stem}__forecast.csv", index=False)

    # plots (optional and limited)
    max_plots = env_int("MAX_PLOTS", 30)
    make_plots = env_int("MAKE_PLOTS", 1)
    if make_plots <= 0:
        return

    for j, c in enumerate(cols):
        if j >= max_plots:
            break
        fig, ax = plt.subplots()
        ax.plot(ts_hist, y_hist[:, j], label="hist")
        ax.plot(ts_future, y_pred[:, j], label="pred")
        ax.set_title(f"{prefix} - {stem} - {c}")
        ax.set_xlabel("timestamp")
        ax.set_ylabel(c)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}__{stem}__{j:04d}__{safe_name(c)}.png", dpi=150)
        plt.close(fig)

def safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", s)
    s = s.strip().strip(".")
    return s[:120] if len(s) > 120 else s

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

def build_models(n_train: int):
    k = max(1, min(env_int("KNN_K", 7), n_train))
    rf_estimators = env_int("RF_N", 20)
    rf_depth = os.environ.get("RF_DEPTH")
    rf_depth = None if rf_depth in (None, "", "None") else int(float(rf_depth))
    models = {
        "KNN": KNeighborsRegressor(n_neighbors=k, weights=os.environ.get("KNN_W", "distance")),
        "RF": RandomForestRegressor(
            n_estimators=rf_estimators,
            max_depth=rf_depth,
            random_state=42,
            n_jobs=env_int("RF_JOBS", -1),
        ),
        "RIDGE": Ridge(alpha=env_float("RIDGE_A", 1.0)),
    }
    return models

def main():
    DF_INPUT_PATH = os.environ.get("DF_INPUT_PATH")
    H = env_int("FORECAST_HORIZON", 50)
    L = env_int("ML_CONTEXT", 200)
    out_root = Path(os.environ.get("OUTPUT_ROOT", str(Path.cwd() / "outputs_next50")))
    out_dir = out_root / "ml"

    Xdf, ts, _ = load_gran(DF_INPUT_PATH)
    arr = Xdf.to_numpy(dtype=np.float32)
    cols = list(Xdf.columns)
    stem = Path(clean_path(DF_INPUT_PATH)).stem

    step = infer_step(ts)
    ts_future = ts[-1] + step * np.arange(1, H+1)

    Xw, Yw = make_windows(arr, L)
    if Xw is None or Xw.shape[0] < 2:
        logger.warning(f"Too few windows for L={L}. Falling back to naive.")
        for name in ["NAIVE"]:
            pred = naive_forecast(arr, H)
            save_forecast(out_dir, stem, cols, ts, ts_future, arr, pred, name)
        return

    n_train = Xw.shape[0]
    models = build_models(n_train)

    for name, model in models.items():
        logger.info(f"Training {name} with n_train={n_train} L={L} H={H}")
        model.fit(Xw, Yw)
        pred = recursive_forecast(model, arr, L, H)
        save_forecast(out_dir, stem, cols, ts, ts_future, arr, pred, name)
        logger.info(f"Done {name}")

if __name__ == "__main__":
    main()
