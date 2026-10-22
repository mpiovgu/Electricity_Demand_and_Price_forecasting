"""FastAPI 风控服务：鉴权、输入校验、预测、Prometheus 指标、漂移监控端点。

启动：python risk_model/app.py   (或 uvicorn risk_model.app:app --host 0.0.0.0 --port 8000)
"""
import sys
import time
import hashlib
import json
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
from fastapi import FastAPI, Request, Header, HTTPException, Depends
from prometheus_client import make_asgi_app

from risk_model.config import API_KEY, SERVICE_HOST, SERVICE_PORT, PSI_WARN, PSI_ALERT
from risk_model.schemas import (RiskRawInput, RiskFeatures, PredictResponse,
                    HealthResponse, DriftResponse, DriftItem)
from risk_model.model_wrapper import RiskModel
from risk_model.features import build_features_from_raw, FEATURE_COLS
from risk_model.monitor import log_prediction, compute_drift, score_drift
from risk_model import prom_metrics as pm

app = FastAPI(title="电力价格风控模型服务", version="1.0")
app.mount("/metrics", make_asgi_app())   # 原生 Prometheus 指标端点

model = None


@app.on_event("startup")
def startup():
    global model
    model = RiskModel()
    pm.MODEL_INFO.labels(
        model="histgb-risk",
        threshold=str(model.threshold),
        quantile=str(model.meta["quantile"]),
        trained_at=model.meta.get("trained_at", "")).set(1)


def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="无效或缺失 API Key")


def _dump(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()


@app.get("/health", response_model=HealthResponse)
def health():
    m = model.meta["metrics"]
    return HealthResponse(
        status="ok", model_loaded=model is not None,
        training_positive_rate=m["test_positive_rate"],
        threshold=model.threshold, test_auc=m["auc"])


@app.post("/predict", response_model=PredictResponse,
          dependencies=[Depends(verify_api_key)])
def predict(raw: RiskRawInput, request: Request):
    start = time.time()
    X = build_features_from_raw(_dump(raw))
    scores, labels = model.predict(X)
    score, label = float(scores[0]), int(labels[0])

    pm.REQUEST_COUNT.labels("/predict", "POST", "200").inc()
    pm.REQUEST_LATENCY.labels("/predict").observe(time.time() - start)
    pm.PREDICTION_COUNT.labels(str(label)).inc()
    pm.PREDICTION_SCORE.observe(score)

    feats = X.iloc[0].to_dict()
    ih = hashlib.md5(json.dumps(feats, sort_keys=True, default=float).encode()).hexdigest()
    log_prediction(score, label, feats, ih)
    sr = score_drift()
    if sr is not None:
        pm.POSITIVE_RATE.set(sr)

    return PredictResponse(
        risk_score=round(score, 4), risk_label=label,
        threshold=model.threshold, is_risk=bool(label),
        features_used=FEATURE_COLS)


@app.post("/predict/features", response_model=PredictResponse,
          dependencies=[Depends(verify_api_key)])
def predict_features(rf: RiskFeatures, request: Request):
    start = time.time()
    if len(rf.features) != len(FEATURE_COLS):
        raise HTTPException(
            status_code=400,
            detail=f"特征数量应为 {len(FEATURE_COLS)}，实际 {len(rf.features)}")
    X = pd.DataFrame([rf.features], columns=FEATURE_COLS)
    scores, labels = model.predict(X)
    score, label = float(scores[0]), int(labels[0])

    pm.REQUEST_COUNT.labels("/predict/features", "POST", "200").inc()
    pm.REQUEST_LATENCY.labels("/predict/features").observe(time.time() - start)
    pm.PREDICTION_COUNT.labels(str(label)).inc()
    pm.PREDICTION_SCORE.observe(score)

    feats = dict(zip(FEATURE_COLS, rf.features))
    log_prediction(score, label, feats,
                   hashlib.md5(str(rf.features).encode()).hexdigest())
    sr = score_drift()
    if sr is not None:
        pm.POSITIVE_RATE.set(sr)

    return PredictResponse(
        risk_score=round(score, 4), risk_label=label,
        threshold=model.threshold, is_risk=bool(label))


@app.get("/monitor/drift", response_model=DriftResponse,
         dependencies=[Depends(verify_api_key)])
def drift():
    rep = compute_drift()
    if rep is None:
        raise HTTPException(status_code=503, detail="参考分布未生成，请先训练")
    for it in rep["items"]:
        pm.FEATURE_DRIFT.labels(it["feature"]).set(it["psi"])
    pm.MAX_DRIFT.set(rep["max_psi"])
    return DriftResponse(
        max_psi=rep["max_psi"], status=rep["status"],
        n_samples=rep["n_samples"],
        items=[DriftItem(**i) for i in rep["items"]])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)
