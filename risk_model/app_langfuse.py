"""生产可用版：Langfuse 可观测 + Prometheus 指标 + 漂移监控 + JWT 身份 + lifespan。

相比 Demo 版的关键改动（上线必备）：
  1. 身份从 JWT 解析（user_id 不可伪造，session_id 由服务端签发），不再信客户端 header；
  2. 用 lifespan 替代已弃用的 on_event，进程退出前 langfuse.flush()；
  3. 风险模型不是 LLM，模型推理改记为 span + metadata（真实算力/特征），不用占位的 token usage；
  4. 每次 trace 打 release / environment 标签，便于分环境对比；
  5. 预留 sampling_rate（高流量采样）与环境变量读取密钥。

启动：
    pip install langfuse pyjwt
    export LANGFUSE_PUBLIC_KEY=pk-...
    export LANGFUSE_SECRET_KEY=sk-...
    export LANGFUSE_HOST=https://cloud.langfuse.com
    export JWT_SECRET=<你的签名密钥>
    uvicorn risk_model.app_langfuse:app --host 0.0.0.0 --port 8000
"""
import sys
import os
import time
import hashlib
import json
import pathlib
import uuid
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from typing import Optional
from contextlib import asynccontextmanager
import pandas as pd
import jwt
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import make_asgi_app

from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe

from risk_model.config import API_KEY, SERVICE_HOST, SERVICE_PORT
from risk_model.schemas import (RiskRawInput, RiskFeatures, PredictResponse,
                                HealthResponse, DriftResponse, DriftItem,
                                SessionResponse)
from risk_model.model_wrapper import RiskModel
from risk_model.features import build_features_from_raw, FEATURE_COLS
from risk_model.monitor import log_prediction, compute_drift, score_drift
from risk_model import prom_metrics as pm

# ---- 运行时配置（上线从环境变量 / 密钥管理读取，切勿硬编码）----
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
RELEASE = os.getenv("RELEASE", "v1.2.0")
ENV = os.getenv("ENVIRONMENT", "production")

# ---- Langfuse 客户端（全局单例，底层走 OTLP）----
# 高流量时开启采样，避免成本与性能爆炸：
# langfuse = Langfuse(sampling_rate=0.1)
langfuse = Langfuse()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：加载模型并上报元信息
    global model
    model = RiskModel()
    pm.MODEL_INFO.labels(
        model="histgb-risk",
        threshold=str(model.threshold),
        quantile=str(model.meta["quantile"]),
        trained_at=model.meta.get("trained_at", "")).set(1)
    yield
    langfuse.flush()   # 优雅停机：把缓冲的 trace 全发出去


app = FastAPI(title="电力价格风控模型服务 (生产版)", version=RELEASE, lifespan=lifespan)
app.mount("/metrics", make_asgi_app())   # 原生 Prometheus 指标端点

model = None


def _dump(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()


# ---- 身份：从 JWT 解析（user_id 恒定、session_id 服务端可控）----
bearer = HTTPBearer(auto_error=True)


def decode_token(creds: HTTPAuthorizationCredentials) -> dict:
    """生产：用你的鉴权公钥/密钥验证并解析；这里用 HS256 示例。"""
    try:
        return jwt.decode(creds.credentials, JWT_SECRET, algorithms=["HS256"])
    except Exception:
        raise HTTPException(status_code=401, detail="无效或过期 token")


def get_identity(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    x_session_id: Optional[str] = Header(None, alias="x-session-id"),
):
    claims = decode_token(creds)
    # user_id 恒定来自 JWT（不可伪造）；
    # session_id 优先用 JWT 的 sid，若服务端刚新建会话则取客户端回传的 x-session-id
    return {
        "user_id": claims["sub"],
        "session_id": claims.get("sid") or x_session_id or "no-session",
    }


def verify_api_key(x_api_key: str = Header(None)):
    # 服务间调用鉴权（与用户 JWT 分离）
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="无效或缺失 API Key")


@app.get("/health", response_model=HealthResponse)
def health():
    m = model.meta["metrics"]
    return HealthResponse(
        status="ok", model_loaded=model is not None,
        training_positive_rate=m["test_positive_rate"],
        threshold=model.threshold, test_auc=m["auc"])


@app.post("/session", response_model=SessionResponse,
          dependencies=[Depends(verify_api_key)])
@observe(name="session_start")
def create_session(identity: dict = Depends(get_identity)):
    # 服务端生成 session_id —— 这是客户端手上还没有的值，必须返回给它，
    # 否则后续 /predict 无法带上同一个 session_id，会话就断了。
    new_sid = f"sess_{uuid.uuid4().hex}"
    langfuse_context.update_current_trace(
        user_id=identity["user_id"],
        session_id=new_sid,                 # 本次会话的起点 trace
        tags=[ENV], release=RELEASE, metadata={"environment": ENV})
    return SessionResponse(session_id=new_sid, user_id=identity["user_id"])


@app.post("/predict", response_model=PredictResponse,
          dependencies=[Depends(verify_api_key)])
@observe(name="predict")
def predict(raw: RiskRawInput,
            identity: dict = Depends(get_identity)):
    start = time.time()

    # ① 绑定用户/会话 + 发布标签（Langfuse 一等公民字段）
    langfuse_context.update_current_trace(
        user_id=identity["user_id"],
        session_id=identity["session_id"],
        input=_dump(raw),
        tags=[ENV],
        release=RELEASE,
        metadata={"environment": ENV})

    X = build_features_from_raw(_dump(raw))
    scores, labels = model.predict(X)
    score, label = float(scores[0]), int(labels[0])

    # ② Prometheus 指标
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

    # ③ 风险模型不是 LLM：记 span + metadata（真实算力/特征），不用占位 token
    trace_id = langfuse_context.get_current_trace_id()
    langfuse.span(
        trace_id=trace_id, name="model_predict",
        input=_dump(raw),
        output={"risk_score": round(score, 4), "risk_label": label},
        metadata={"features": feats, "inference_ms": round((time.time() - start) * 1000, 2)})

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
        tags=[ENV], release=RELEASE, metadata={"environment": ENV})

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
    log_prediction(score, label, feats, hashlib.md5(str(rf.features).encode()).hexdigest())
    sr = score_drift()
    if sr is not None:
        pm.POSITIVE_RATE.set(sr)

    trace_id = langfuse_context.get_current_trace_id()
    langfuse.span(
        trace_id=trace_id, name="model_predict",
        input=rf.features,
        output={"risk_score": round(score, 4), "risk_label": label},
        metadata={"features": feats, "inference_ms": round((time.time() - start) * 1000, 2)})

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
    """按 user_id / session_id 查询历史 trace。

    上线建议：此接口返回用户数据，需再加 RBAC（仅管理员/本人）与限流。
    """
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
