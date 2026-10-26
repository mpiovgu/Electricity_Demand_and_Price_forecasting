"""FastAPI 风控服务 + Langfuse 可观测性 + Prometheus 指标 + 漂移监控。

相比 app.py，额外能力：
  1. 每个请求按 x-user-id / x-session-id 绑定到 Langfuse trace（UI 可按用户/会话筛选、回放、聚合）；
  2. 模型推理记为 generation，可自动算成本、可打分；
  3. 保留 /metrics（Prometheus）与 /monitor/drift（PSI 漂移）；
  4. /traces 端点演示如何按 user_id / session_id 程序化查询历史。

启动：
    pip install langfuse
    export LANGFUSE_PUBLIC_KEY=pk-...
    export LANGFUSE_SECRET_KEY=sk-...
    export LANGFUSE_HOST=https://cloud.langfuse.com
    uvicorn risk_model.app_langfuse:app --host 0.0.0.0 --port 8000
"""
import sys
import time
import hashlib
import json
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from typing import Optional
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Depends
from prometheus_client import make_asgi_app

from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe

from risk_model.config import API_KEY, SERVICE_HOST, SERVICE_PORT
from risk_model.schemas import (RiskRawInput, RiskFeatures, PredictResponse,
                                HealthResponse, DriftResponse, DriftItem)
from risk_model.model_wrapper import RiskModel
from risk_model.features import build_features_from_raw, FEATURE_COLS
from risk_model.monitor import log_prediction, compute_drift, score_drift
from risk_model import prom_metrics as pm

# ---- Langfuse 客户端（全局单例，底层走 OTLP）----
# 不传参时自动读环境变量 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
langfuse = Langfuse()

app = FastAPI(title="电力价格风控模型服务 (Langfuse + Prometheus)", version="1.0")
app.mount("/metrics", make_asgi_app())   # 原生 Prometheus 指标端点

model = None


def _dump(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()


@app.on_event("startup")
def startup():
    global model
    model = RiskModel()
    pm.MODEL_INFO.labels(
        model="histgb-risk",
        threshold=str(model.threshold),
        quantile=str(model.meta["quantile"]),
        trained_at=model.meta.get("trained_at", "")).set(1)


@app.on_event("shutdown")
def shutdown():
    langfuse.flush()   # 进程退出前把缓冲的 trace 发出去


# ---- 依赖：从请求头取身份（user / session）----
def get_identity(
    x_user_id: Optional[str] = Header(None, alias="x-user-id"),
    x_session_id: Optional[str] = Header(None, alias="x-session-id"),
):
    # 没有就给默认值，保证每一条 trace 都能被归因
    return {
        "user_id": x_user_id or "anonymous",
        "session_id": x_session_id or "default-session",
    }


def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="无效或缺失 API Key")


@app.get("/health", response_model=HealthResponse)
def health():
    m = model.meta["metrics"]
    return HealthResponse(
        status="ok", model_loaded=model is not None,
        training_positive_rate=m["test_positive_rate"],
        threshold=model.threshold, test_auc=m["auc"])


@app.post("/predict", response_model=PredictResponse,
          dependencies=[Depends(verify_api_key)])
@observe(name="predict")                 # 自动为本次请求创建 trace + 根 span
def predict(raw: RiskRawInput,
            identity: dict = Depends(get_identity)):
    start = time.time()

    # ① 把 user / session 绑到当前 trace（Langfuse 一等公民字段）
    langfuse_context.update_current_trace(
        user_id=identity["user_id"],
        session_id=identity["session_id"],
        input=_dump(raw),
        tags=["predict", "risk"])

    X = build_features_from_raw(_dump(raw))
    scores, labels = model.predict(X)
    score, label = float(scores[0]), int(labels[0])

    # ② Prometheus 指标（与 app.py 一致）
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

    # ③ 把模型推理记为 generation：自动算成本、可后续打分
    trace_id = langfuse_context.get_current_trace_id()
    langfuse.generation(
        trace_id=trace_id, name="model_predict", model="histgb-risk",
        input=_dump(raw),
        output={"risk_score": round(score, 4), "risk_label": label},
        usage={"input": 1, "output": 1, "total": 2})   # 示例用量，真实场景填实际 token/样本数

    return PredictResponse(
        risk_score=round(score, 4), risk_label=label,
        threshold=model.threshold, is_risk=bool(label),
        features_used=FEATURE_COLS)


@app.post("/predict/features", response_model=PredictResponse,
          dependencies=[Depends(verify_api_key)])
@observe(name="predict_features")
def predict_features(rf: RiskFeatures,
                     identity: dict = Depends(get_identity)):
    start = time.time()
    langfuse_context.update_current_trace(
        user_id=identity["user_id"],
        session_id=identity["session_id"],
        input={"features": rf.features},
        tags=["predict", "features"])

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
    log_prediction(score, label, dict(zip(FEATURE_COLS, rf.features)),
                   hashlib.md5(str(rf.features).encode()).hexdigest())
    sr = score_drift()
    if sr is not None:
        pm.POSITIVE_RATE.set(sr)

    trace_id = langfuse_context.get_current_trace_id()
    langfuse.generation(
        trace_id=trace_id, name="model_predict", model="histgb-risk",
        input=rf.features,
        output={"risk_score": round(score, 4), "risk_label": label},
        usage={"input": 1, "output": 1, "total": 2})

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


@app.get("/traces")
def query_traces(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    page: int = 1,
    _: dict = Depends(verify_api_key),
):
    """④ 按 user_id / session_id 程序化查询历史 trace（演示"怎么调"）。"""
    res = langfuse.fetch_traces(user_id=user_id, session_id=session_id, page=page)
    return {
        "total_pages": res.total_pages,
        "page": res.page,
        "traces": [
            {"id": t.id, "name": t.name,
             "user_id": t.user_id, "session_id": t.session_id,
             "timestamp": str(t.timestamp)}
            for t in res.data
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)
