import json

import numpy as np
import pytest

from heart import config as C
from heart.predict import HeartRiskModel
from heart.quality import check_quality
from heart.train import Candidate, _holdout_metrics, select_winner, threshold_scores


class FixedProbabilities:
    def predict_proba(self, X):
        p = np.array([0.39, 0.40, 0.49, 0.80])
        return np.column_stack([1 - p, p])


def test_threshold_scoring_and_holdout_match():
    pipe = FixedProbabilities()
    y = [0, 1, 1, 0]
    cv = threshold_scores(pipe, None, y)
    test = _holdout_metrics(pipe, None, y)
    assert cv['recall'] == test['test_recall'] == 1.0
    assert cv['precision'] == test['test_precision'] == 2 / 3
    assert cv['recall_at_05'] == 0.0


def test_serving_uses_metadata_threshold(trained_model_files, sample_record):
    model_path, meta_path = trained_model_files
    meta = json.loads(meta_path.read_text())
    meta['threshold'] = C.DECISION_THRESHOLD
    meta_path.write_text(json.dumps(meta))
    model = HeartRiskModel(model_path, meta_path)
    model.pipeline = FixedProbabilities()
    result = model.predict([sample_record] * 4)
    assert [r['prediction'] for r in result] == [0, 1, 1, 1]
    assert result[1]['confidence'] == 0.4


def test_selection_never_uses_holdout_tie_break():
    a = Candidate('first', None, {}, {}, result={'cv_roc_auc_mean': 0.9, 'test_roc_auc': 0.8})
    b = Candidate('second', None, {}, {}, result={'cv_roc_auc_mean': 0.9, 'test_roc_auc': 1.0})
    assert select_winner([a, b]) is a


def test_quality_gate_accepts_boundary():
    check_quality({'threshold': C.DECISION_THRESHOLD,
                   'metrics': {'cv_roc_auc_mean': C.MIN_CV_ROC_AUC, 'cv_recall_mean': C.MIN_CV_RECALL}})


@pytest.mark.parametrize('metric,value', [
    ('cv_roc_auc_mean', 0.8499), ('cv_recall_mean', 0.7999),
    ('cv_recall_mean', float('nan')), ('cv_roc_auc_mean', float('inf')),
])
def test_quality_gate_rejects_bad_scores(metric, value):
    meta = {'threshold': C.DECISION_THRESHOLD, 'metrics': {'cv_roc_auc_mean': 0.9, 'cv_recall_mean': 0.9}}
    meta['metrics'][metric] = value
    with pytest.raises(ValueError):
        check_quality(meta)


def test_quality_gate_rejects_threshold_mismatch():
    with pytest.raises(ValueError):
        check_quality({'threshold': 0.5, 'metrics': {'cv_roc_auc_mean': 0.9, 'cv_recall_mean': 0.9}})


def test_quality_gate_rejects_missing_recall():
    with pytest.raises(KeyError):
        check_quality({'threshold': C.DECISION_THRESHOLD, 'metrics': {'cv_roc_auc_mean': 0.9}})
