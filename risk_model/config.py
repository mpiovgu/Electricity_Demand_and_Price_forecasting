"""全局配置：路径、风控定义、服务与监控参数。"""
from pathlib import Path
import os

# ---- 路径 ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ENERGY_CSV = DATA_DIR / "energy_dataset.csv"
WEATHER_CSV = DATA_DIR / "weather_features.csv"

ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "risk_model.joblib"      # 训练好的模型（sklearn）
META_PATH = ARTIFACT_DIR / "model_metadata.json"     # 指标 / 阈值 / 特征
REF_STATS_PATH = ARTIFACT_DIR / "reference_stats.json"  # 训练参考分布（漂移基线）
PRED_DB_PATH = ARTIFACT_DIR / "predictions.db"       # 预测日志（SQLite）

# ---- 风控定义 ----
# 风险事件 = 下一小时实际电价处于历史高位分位（尖峰风险）
RISK_PRICE_QUANTILE = 0.90
RANDOM_STATE = 42
TEST_FRACTION = 0.20   # 时序切分：后 20% 作测试集

# ---- 服务 ----
API_KEY = os.getenv("RISK_API_KEY", "risk-demo-secret-key")
SERVICE_HOST = os.getenv("RISK_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("RISK_PORT", "8000"))

# ---- 漂移告警阈值 (PSI, Population Stability Index) ----
PSI_WARN = 0.25    # >0.25 提示关注
PSI_ALERT = 0.50   # >0.50 严重漂移，需排查
