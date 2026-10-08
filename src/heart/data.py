"""Loading, validation and cleaning of the raw UCI file.

Cleaning is deliberately *stateless* (no statistics learned from data):
- coerce types, map '?' to NaN
- binarise the target (num 1-4 -> 1)
- drop exact duplicates
- set physiologically impossible values to NaN (e.g. chol = 0)

Statistical imputation (median / mode) happens inside the sklearn pipeline so
that it is learned on training folds only and shipped with the model.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from heart.config import (
    CLEAN_CSV, INPUT_FEATURES, RAW_COLUMNS, RAW_CSV, TARGET, VALID_CODES, VALID_RANGES,
)

log = logging.getLogger(__name__)


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - run `python scripts/download_data.py` first")
    df = pd.read_csv(path, na_values=["?", ""])
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw data missing columns: {sorted(missing)}")
    return df[RAW_COLUMNS]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy with INPUT_FEATURES + binary TARGET."""
    out = df.copy()
    for col in RAW_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["num"])
    out[TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns="num")

    # out-of-domain values -> NaN (imputed later)
    for col, (lo, hi) in VALID_RANGES.items():
        bad = ~out[col].between(lo, hi) & out[col].notna()
        if bad.any():
            log.info("%s: %d out-of-range values set to NaN", col, int(bad.sum()))
        out.loc[bad, col] = np.nan
    for col, codes in VALID_CODES.items():
        bad = ~out[col].isin(codes) & out[col].notna()
        out.loc[bad, col] = np.nan

    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    if len(out) != before:
        log.info("Dropped %d duplicate rows", before - len(out))
    return out[INPUT_FEATURES + [TARGET]]


def missing_report(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()[lambda s: s > 0]


def load_clean(path: Path = CLEAN_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def build_clean_dataset(raw: Path = RAW_CSV, out: Path = CLEAN_CSV) -> pd.DataFrame:
    df = clean(load_raw(raw))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Clean dataset: %s rows -> %s (missing: %s)", len(df), out,
             missing_report(df).to_dict())
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[INPUT_FEATURES], df[TARGET]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_clean_dataset()
