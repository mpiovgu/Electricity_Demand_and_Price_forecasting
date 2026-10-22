"""Prometheus 指标定义（请求量、延迟、预测分布、漂移、正例率）。"""
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "risk_api_requests_total", "Total API requests",
    ["endpoint", "method", "status"])

REQUEST_LATENCY = Histogram(
    "risk_api_request_latency_seconds", "Request latency in seconds",
    ["endpoint"])

PREDICTION_COUNT = Counter(
    "risk_model_predictions_total", "Total scored predictions",
    ["risk_label"])

PREDICTION_SCORE = Histogram(
    "risk_model_score", "Risk score distribution",
    buckets=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

FEATURE_DRIFT = Gauge(
    "risk_model_feature_drift_psi", "PSI drift per feature", ["feature"])

MAX_DRIFT = Gauge(
    "risk_model_max_feature_drift_psi", "Max feature drift across features")

POSITIVE_RATE = Gauge(
    "risk_model_serving_positive_rate", "Serving positive (risk) rate")

MODEL_INFO = Gauge(
    "risk_model_info", "Model metadata",
    ["model", "threshold", "quantile", "trained_at"])
