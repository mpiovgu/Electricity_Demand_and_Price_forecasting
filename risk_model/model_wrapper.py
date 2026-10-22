"""模型加载与预测封装（服务与离线评估共用）。"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import joblib
from risk_model.config import MODEL_PATH, META_PATH
from risk_model.features import FEATURE_COLS


class RiskModel:
    def __init__(self):
        if not META_PATH.exists() or not MODEL_PATH.exists():
            raise FileNotFoundError(
                "模型未训练，请先运行 risk_model/train.py")
        self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        self.model = joblib.load(str(MODEL_PATH))
        self.threshold = float(self.meta["threshold"])
        self.feature_cols = list(self.meta.get("features", FEATURE_COLS))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_cols])[:, 1]

    def predict(self, X: pd.DataFrame):
        """返回 (scores, labels)。"""
        scores = self.predict_proba(X)
        labels = (scores >= self.threshold).astype(int)
        return scores, labels
