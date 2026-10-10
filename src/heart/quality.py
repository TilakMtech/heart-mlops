"""CI promotion gate evaluated at the exported decision threshold."""
import json
import math

from heart import config as C


def check_quality(meta):
    if meta.get("threshold") != C.DECISION_THRESHOLD:
        raise ValueError("Model threshold differs from configured evaluation threshold")
    for metric, minimum in [("cv_roc_auc_mean", C.MIN_CV_ROC_AUC),
                            ("cv_recall_mean", C.MIN_CV_RECALL)]:
        value = float(meta["metrics"][metric])
        if not math.isfinite(value) or value < minimum:
            raise ValueError(f"{metric}={value} must be >= {minimum}")


if __name__ == "__main__":
    meta = json.loads(C.MODEL_META_PATH.read_text())
    check_quality(meta)
    print(f"Quality gate passed at threshold {meta['threshold']}: {meta['metrics']}")
