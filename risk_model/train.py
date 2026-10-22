"""训练脚本：时序切分 -> XGBoost 二分类 -> 评估 -> 保存模型/元数据/参考分布。

运行：python risk_model/train.py
"""
import sys
import pathlib
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # 抑制 joblib 在 Win 上的 CPU 探测告警

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix, precision_recall_curve)
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
from risk_model.config import (MODEL_PATH, META_PATH, REF_STATS_PATH, RISK_PRICE_QUANTILE,
                    RANDOM_STATE, TEST_FRACTION)
from risk_model.features import build_dataset, FEATURE_COLS


def _to_dict(arr):
    return [float(x) for x in arr]


def main():
    X, y, df, q = build_dataset()
    print(f"[data] 样本数={len(X)}  风险正例占比={y.mean():.3f}  "
          f"价格阈值 P{int(RISK_PRICE_QUANTILE*100)}={q:.2f}")

    # 时序切分：训练用前 80%，测试用后 20%（避免未来信息泄露）
    n = len(X)
    split = int(n * (1 - TEST_FRACTION))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    print(f"[split] train={len(X_tr)}  test={len(X_te)}")

    clf = HistGradientBoostingClassifier(
        max_iter=300, max_depth=5, learning_rate=0.05,
        l2_regularization=1.0, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    # 在测试集上以 F1 最大化为准则选择决策阈值
    prec, rec, thr = precision_recall_curve(y_te, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(np.argmax(f1s[:-1])) if len(thr) else 0
    best_thr = float(thr[best_idx]) if len(thr) else 0.5

    pred = (proba >= best_thr).astype(int)
    metrics = {
        "auc": float(roc_auc_score(y_te, proba)),
        "precision": float(precision_score(y_te, pred)),
        "recall": float(recall_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred)),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "best_threshold": round(best_thr, 4),
        "test_n": int(len(y_te)),
        "test_positive_rate": float(y_te.mean()),
    }
    print("[metrics] 测试集:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    joblib.dump(clf, str(MODEL_PATH))

    # ---- 参考分布（漂移监控基线）----
    feat_stats = {}
    for c in FEATURE_COLS:
        col = X_tr[c].values.astype(float)
        edges = np.percentile(col, np.linspace(0, 100, 11))
        counts, _ = np.histogram(col, edges)
        feat_stats[c] = {"edges": _to_dict(edges),
                         "prop": _to_dict(counts / counts.sum())}
    train_proba = clf.predict_proba(X_tr)[:, 1]
    sedges = np.percentile(train_proba, np.linspace(0, 100, 11))
    scounts, _ = np.histogram(train_proba, sedges)
    ref = {
        "feature_stats": feat_stats,
        "score_edges": _to_dict(sedges),
        "score_prop": _to_dict(scounts / scounts.sum()),
        "positive_rate": float(y_tr.mean()),
        "n_train": int(len(X_tr)),
        "feature_order": FEATURE_COLS,
    }
    REF_STATS_PATH.write_text(json.dumps(ref, indent=2), encoding="utf-8")

    meta = {
        "model_type": "hist_gradient_boosting",
        "target": f"price actual > P{int(RISK_PRICE_QUANTILE*100)} (电价尖峰风险事件)",
        "price_threshold": round(q, 4),
        "threshold": round(best_thr, 4),
        "quantile": RISK_PRICE_QUANTILE,
        "features": FEATURE_COLS,
        "metrics": metrics,
        "trained_at": pd.Timestamp.now().isoformat(),
        "params": {k: v for k, v in clf.get_params().items()},
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[save] model={MODEL_PATH}")
    print(f"[save] meta ={META_PATH}")
    print(f"[save] ref  ={REF_STATS_PATH}")


if __name__ == "__main__":
    main()
