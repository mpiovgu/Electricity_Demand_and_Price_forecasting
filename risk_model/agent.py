"""基于 LangGraph 的 ReAct 智能体：把既有电力价格风控能力封装为工具，
用 create_react_agent 构建一个能与用户自然语言对话、并调用模型/监控工具的 Agent。

依赖（新增）：
    pip install langgraph langchain langchain-openai

配置（环境变量）：
    OPENAI_API_KEY  必填
    OPENAI_BASE_URL 可选，兼容 OpenAI 协议的中转/私有部署
    AGENT_MODEL     可选，默认 gpt-4o-mini
    AGENT_SYSTEM    可选，覆盖系统提示词

运行（交互式对话）：
    python run_agent.py
或作为库：
    from risk_model.agent import build_agent, get_tools
    app = build_agent()
    print(app.invoke({"messages": [("user", "现在电价尖峰风险高吗？")]})["messages"][-1].content)
"""
import os
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from typing import List
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from risk_model.config import MODEL_PATH, META_PATH
from risk_model.features import FEATURE_COLS
from risk_model.model_wrapper import RiskModel
from risk_model.monitor import compute_drift
from risk_model.schemas import RiskRawInput

# 进程级共享的模型实例（按需懒加载）
_MODEL: RiskModel | None = None


def _get_model() -> RiskModel:
    """懒加载风控模型；模型未训练时给出明确指引。"""
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                "模型尚未训练，请先运行 `python -m risk_model.train` 生成 "
                f"{MODEL_PATH.name} 与 {META_PATH.name}")
        _MODEL = RiskModel()
    return _MODEL


# ----------------------------- 工具定义 -----------------------------
@tool
def predict_risk(raw: dict) -> dict:
    """根据一次原始观测（t 时刻已知量与历史滞后值）预测电价尖峰风险。

    参数 raw 需包含字段：
    timestamp(可选), total_load_forecast, price_day_ahead, forecast_solar,
    forecast_wind_onshore, total_load_actual_prev, price_actual_prev,
    price_actual_prev2, temp, humidity, wind_speed, pressure, clouds_all, rain_1h(可选)。
    """
    m = _get_model()
    from risk_model.features import build_features_from_raw
    X = build_features_from_raw(raw)
    scores, labels = m.predict(X)
    score, label = float(scores[0]), int(labels[0])
    return {
        "risk_score": round(score, 4),
        "risk_label": label,
        "is_risk": bool(label),
        "threshold": m.threshold,
    }


@tool
def predict_risk_from_features(features: List[float]) -> dict:
    """直接用已构造的特征向量打分（顺序需与 FEATURE_COLS 一致，共 %d 维）。""" % len(FEATURE_COLS)
    if len(features) != len(FEATURE_COLS):
        return {"error": f"特征数应为 {len(FEATURE_COLS)}，实际 {len(features)}"}
    import pandas as pd
    m = _get_model()
    X = pd.DataFrame([features], columns=FEATURE_COLS)
    scores, labels = m.predict(X)
    score, label = float(scores[0]), int(labels[0])
    return {
        "risk_score": round(score, 4),
        "risk_label": label,
        "is_risk": bool(label),
        "threshold": m.threshold,
    }


@tool
def model_health() -> dict:
    """返回模型服务健康度：是否加载、训练正样本率、阈值、测试集 AUC。"""
    m = _get_model()
    metrics = m.meta["metrics"]
    return {
        "status": "ok",
        "model_loaded": True,
        "training_positive_rate": metrics["test_positive_rate"],
        "threshold": m.threshold,
        "test_auc": metrics["auc"],
    }


@tool
def check_drift() -> dict:
    """计算特征漂移（PSI）：返回最大 PSI、状态（ok/warn/alert）及各特征明细。"""
    rep = compute_drift()
    if rep is None:
        return {"error": "参考分布未生成，请先训练模型"}
    return {
        "max_psi": rep["max_psi"],
        "status": rep["status"],
        "n_samples": rep["n_samples"],
        "items": rep["items"],
    }


def get_tools():
    """返回供 Agent 使用的工具列表（统一入口，便于复用/测试）。"""
    return [predict_risk, predict_risk_from_features, model_health, check_drift]


# ----------------------------- 模型与 Agent -----------------------------
DEFAULT_SYSTEM = (
    "你是「电力价格风控助手」，服务于西班牙电力市场的电价尖峰风险预警系统。"
    "你可以使用工具调用风控模型预测电价尖峰风险、查询模型健康度、检测特征漂移。"
    "请基于工具返回的真实结果作答，不要编造数据；若用户缺少必要参数，请明确询问。"
    "风险定义为下一小时实际电价处于历史 P90 高位分位。"
)


def build_model() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少环境变量 OPENAI_API_KEY，无法初始化 LLM。")
    return ChatOpenAI(
        model=os.getenv("AGENT_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),  # 兼容中转/私有部署
    )


def build_agent(checkpointer=None):
    """构建 ReAct Agent。

    checkpointer 用于多轮记忆（如 langgraph.checkpoint.memory.InMemorySaver()）。
    返回可直接调用 .invoke({"messages": [...]}) 的 compiled graph。
    """
    model = build_model()
    agent = create_react_agent(
        model,
        tools=get_tools(),
        prompt=os.getenv("AGENT_SYSTEM", DEFAULT_SYSTEM),
        checkpointer=checkpointer,
    )
    return agent
