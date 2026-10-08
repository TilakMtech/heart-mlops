"""Download the UCI Heart Disease (Cleveland) dataset into data/raw/heart_raw.csv.

Sources are tried in order, and the output always has the same 14 UCI columns:

1. UCI Machine Learning Repository (official) - processed.cleveland.data
2. GitHub mirror of the same 303 Cleveland records (ISLR "Heart.csv"),
   used when the UCI host is unreachable (e.g. corporate proxy).

Usage:
    python scripts/download_data.py            # skip if file already exists
    python scripts/download_data.py --force    # re-download
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heart.config import RAW_COLUMNS, RAW_CSV  # noqa: E402

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
MIRROR_URL = "https://raw.githubusercontent.com/rashida048/Datasets/master/Heart.csv"

log = logging.getLogger("download")


def _get(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def from_uci() -> pd.DataFrame:
    text = _get(UCI_URL)
    return pd.read_csv(io.StringIO(text), header=None, names=RAW_COLUMNS, na_values="?")


def from_mirror() -> pd.DataFrame:
    """ISLR copy of the Cleveland file; text labels are mapped back to UCI codes."""
    df = pd.read_csv(io.StringIO(_get(MIRROR_URL)), index_col=0)
    df = df.rename(columns={
        "Age": "age", "Sex": "sex", "ChestPain": "cp", "RestBP": "trestbps",
        "Chol": "chol", "Fbs": "fbs", "RestECG": "restecg", "MaxHR": "thalach",
        "ExAng": "exang", "Oldpeak": "oldpeak", "Slope": "slope", "Ca": "ca",
        "Thal": "thal", "AHD": "num",
    })
    df["cp"] = df["cp"].map({"typical": 1, "nontypical": 2, "nonanginal": 3, "asymptomatic": 4})
    df["thal"] = df["thal"].map({"normal": 3, "fixed": 6, "reversable": 7})
    df["num"] = df["num"].map({"No": 0, "Yes": 1})
    return df[RAW_COLUMNS]


def download(force: bool = False, out: Path = RAW_CSV) -> Path:
    if out.exists() and not force:
        log.info("Raw data already present at %s (use --force to refresh)", out)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, fn in (("UCI", from_uci), ("GitHub mirror", from_mirror)):
        try:
            df = fn()
            if len(df) < 300 or list(df.columns) != RAW_COLUMNS:
                raise ValueError(f"unexpected shape {df.shape}")
            df.to_csv(out, index=False)
            log.info("Downloaded %d rows from %s -> %s", len(df), name, out)
            return out
        except Exception as exc:  # noqa: BLE001 - try next source
            log.warning("Source %s failed: %s", name, exc)
    raise SystemExit("All data sources failed - see README 'Data' section for manual steps.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    download(force=p.parse_args().force)
