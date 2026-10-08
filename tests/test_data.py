import numpy as np
import pandas as pd
import pytest

from heart.config import INPUT_FEATURES, TARGET
from heart.data import build_clean_dataset, clean, load_raw, split_xy


def test_clean_binarises_target(raw_df):
    out = clean(raw_df)
    assert set(out[TARGET].unique()) <= {0, 1}
    assert out[TARGET].sum() == (raw_df["num"] > 0).sum()


def test_clean_keeps_expected_columns(raw_df):
    out = clean(raw_df)
    assert list(out.columns) == INPUT_FEATURES + [TARGET]
    assert "num" not in out.columns


def test_question_marks_become_nan():
    raw = pd.DataFrame([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, "?", "?", 0]],
                       columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"])
    out = clean(raw)
    assert np.isnan(out.loc[0, "ca"]) and np.isnan(out.loc[0, "thal"])


def test_out_of_range_values_are_nulled(raw_df):
    raw_df.loc[5, "chol"] = 0       # impossible cholesterol
    raw_df.loc[6, "cp"] = 9         # unknown code
    out = clean(raw_df)
    assert out["chol"].isna().sum() >= 1
    assert out["cp"].isna().sum() >= 1


def test_duplicates_dropped(raw_df):
    dup = pd.concat([raw_df, raw_df.head(5)], ignore_index=True)
    assert len(clean(dup)) == len(clean(raw_df))


def test_clean_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    clean(raw_df)
    pd.testing.assert_frame_equal(before, raw_df)


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.csv")


def test_load_raw_rejects_bad_schema(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_raw(p)


def test_build_clean_dataset_roundtrip(raw_csv, tmp_path):
    out = tmp_path / "clean.csv"
    df = build_clean_dataset(raw_csv, out)
    assert out.exists()
    X, y = split_xy(df)
    assert X.shape[1] == len(INPUT_FEATURES)
    assert len(X) == len(y)
