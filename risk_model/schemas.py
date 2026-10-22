"""FastAPI 请求/响应数据模型。"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from typing import List, Optional
from pydantic import BaseModel, Field
from risk_model.features import FEATURE_COLS


class RiskRawInput(BaseModel):
    """在线打分原始输入：t 时刻已知量 + 前序历史值 + 时间。"""
    timestamp: Optional[str] = Field(None, description="ISO 时间，缺省用当前时间")
    total_load_forecast: float = Field(..., description="t 时刻负荷日前预测")
    price_day_ahead: float = Field(..., description="t 时刻电价日前预测")
    forecast_solar: float = Field(..., description="光伏日前预测")
    forecast_wind_onshore: float = Field(..., description="陆风日前预测")
    total_load_actual_prev: float = Field(..., description="t-1 实际负荷")
    price_actual_prev: float = Field(..., description="t-1 实际电价")
    price_actual_prev2: float = Field(..., description="t-2 实际电价")
    temp: float
    humidity: float
    wind_speed: float
    pressure: float
    clouds_all: float
    rain_1h: float = 0.0


class RiskFeatures(BaseModel):
    """直接传入已构造的特征向量（与 FEATURE_COLS 顺序一致）。"""
    features: List[float] = Field(
        ..., description=f"长度需等于 {len(FEATURE_COLS)} 的特征向量")


class PredictResponse(BaseModel):
    risk_score: float
    risk_label: int
    threshold: float
    is_risk: bool
    features_used: Optional[List[str]] = None


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    training_positive_rate: float
    threshold: float
    test_auc: float


class DriftItem(BaseModel):
    feature: str
    psi: float


class DriftResponse(BaseModel):
    max_psi: float
    status: str          # ok | warn | alert
    n_samples: int
    items: List[DriftItem]
