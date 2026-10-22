"""临时冒烟测试：用 TestClient 验证健康检查、鉴权、预测、漂移、指标端点。"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from risk_model.app import app
from risk_model.features import build_dataset, FEATURE_COLS


def _sample_raw():
    X, y, df, q = build_dataset()
    row, prev, prev2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    return {
        "timestamp": str(row["time"]),
        "total_load_forecast": float(row["total load forecast"]),
        "price_day_ahead": float(row["price day ahead"]),
        "forecast_solar": float(row["forecast solar day ahead"]),
        "forecast_wind_onshore": float(row["forecast wind onshore day ahead"]),
        "total_load_actual_prev": float(prev["total load actual"]),
        "price_actual_prev": float(prev["price actual"]),
        "price_actual_prev2": float(prev2["price actual"]),
        "temp": float(row["temp"]), "humidity": float(row["humidity"]),
        "wind_speed": float(row["wind_speed"]), "pressure": float(row["pressure"]),
        "clouds_all": float(row["clouds_all"]), "rain_1h": 0.0,
    }


def main():
    H = {"X-API-Key": "risk-demo-secret-key"}
    with TestClient(app) as client:
        # 1) 健康检查（无需鉴权）
        r = client.get("/health")
        print("GET /health ->", r.status_code, r.json())

        # 2) 无 Key 应 401
        r = client.post("/predict", json=_sample_raw())
        print("POST /predict (no key) ->", r.status_code)

        # 3) 带 Key 预测，多发几条积累漂移样本
        for _ in range(6):
            r = client.post("/predict", json=_sample_raw(), headers=H)
            assert r.status_code == 200, r.text
        print("POST /predict (auth) ->", r.status_code, r.json())

        # 4) 直接传特征向量
        X, y, df, q = build_dataset()
        feats = X.iloc[-1].tolist()
        r = client.post("/predict/features", json={"features": feats}, headers=H)
        print("POST /predict/features ->", r.status_code, r.json())

        # 5) 漂移监控
        r = client.get("/monitor/drift", headers=H)
        print("GET /monitor/drift ->", r.status_code, r.json())

        # 6) Prometheus 指标
        r = client.get("/metrics")
        lines = [l for l in r.text.splitlines() if l.startswith("risk_")]
        print("GET /metrics ->", r.status_code, "risk_ 指标行数:", len(lines))
        print("  示例:", lines[0] if lines else None)
    print("\nALL OK")


if __name__ == "__main__":
    main()
