"""指标监控：预测日志落库 + 特征/分数漂移（PSI）+ 告警判定。"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json
import sqlite3
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from risk_model.config import PRED_DB_PATH, REF_STATS_PATH, PSI_WARN, PSI_ALERT

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    score REAL,
    label INTEGER,
    features TEXT,
    input_hash TEXT
)
"""


def _conn():
    conn = sqlite3.connect(str(PRED_DB_PATH))
    conn.execute(_SCHEMA)
    return conn


def log_prediction(score, label, features: dict, input_hash: str):
    """将一次预测结果写入 SQLite，供漂移分析使用。"""
    conn = _conn()
    conn.execute(
        "INSERT INTO predictions(ts,score,label,features,input_hash) VALUES(?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), float(score), int(label),
         json.dumps(features, default=float), input_hash))
    conn.commit()
    conn.close()


def recent_features_df(n: int = 500) -> pd.DataFrame:
    conn = _conn()
    rows = conn.execute(
        "SELECT features FROM predictions ORDER BY id DESC LIMIT ?", (n,)).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([json.loads(r[0]) for r in rows])


def recent_scores(n: int = 500):
    conn = _conn()
    rows = conn.execute(
        "SELECT score,label FROM predictions ORDER BY id DESC LIMIT ?", (n,)).fetchall()
    conn.close()
    return rows


def _psi(edges, exp_prop, actual) -> float:
    """计算 PSI：actual 相对参考分布(edges+prop) 的偏移。"""
    cut = np.asarray(edges, dtype=float).copy()
    cut[0], cut[-1] = -np.inf, np.inf
    act_counts, _ = np.histogram(actual, cut)
    if act_counts.sum() == 0:
        return 0.0
    act_p = np.clip(act_counts / act_counts.sum(), 1e-6, None)
    exp_p = np.clip(np.asarray(exp_prop, dtype=float), 1e-6, None)
    return float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))


def compute_drift():
    """对比最近服务样本与训练参考分布，返回漂移报告。"""
    if not REF_STATS_PATH.exists():
        return None
    ref = json.loads(REF_STATS_PATH.read_text(encoding="utf-8"))
    fs = ref["feature_stats"]
    df = recent_features_df(n=500)
    if df.empty:
        return {"max_psi": 0.0, "status": "ok", "n_samples": 0, "items": []}

    items, max_psi = [], 0.0
    for feat, st in fs.items():
        if feat not in df.columns:
            continue
        actual = df[feat].dropna().values.astype(float)
        if len(actual) == 0:
            continue
        psi = _psi(st["edges"], st["prop"], actual)
        items.append({"feature": feat, "psi": round(psi, 4)})
        max_psi = max(max_psi, psi)

    status = ("ok" if max_psi < PSI_WARN else
              "warn" if max_psi < PSI_ALERT else "alert")
    return {"max_psi": round(max_psi, 4), "status": status,
            "n_samples": int(len(df)), "items": items}


def score_drift():
    """服务侧正例（风险）率，用于与训练基准对比。"""
    rows = recent_scores()
    if not rows:
        return None
    labels = [r[1] for r in rows]
    return sum(labels) / len(labels)
