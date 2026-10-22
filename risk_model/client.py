"""示例客户端：从训练集抽取真实样本构造原始输入，调用风控服务并查看漂移。

运行前先启动服务：python risk_model/app.py
然后：python risk_model/client.py
"""
import sys
import time
import pathlib
import requests
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from risk_model.features import build_dataset

BASE = "http://localhost:8000"
KEY = "risk-demo-secret-key"
HEADERS = {"X-API-Key": KEY}


def _sample_raw():
    """取训练集最后一个可用样本，回推其原始输入字段。"""
    X, y, df, q = build_dataset()
    row = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    return {
        "timestamp": str(row["time"]),
        "total_load_forecast": float(row["total load forecast"]),
        "price_day_ahead": float(row["price day ahead"]),
        "forecast_solar": float(row["forecast solar day ahead"]),
        "forecast_wind_onshore": float(row["forecast wind onshore day ahead"]),
        # 注意：t-1 / t-2 实际值来自历史行，真实服务中由特征存储提供
        "total_load_actual_prev": float(prev["total load actual"]),
        "price_actual_prev": float(prev["price actual"]),
        "price_actual_prev2": float(prev2["price actual"]),
        "temp": float(row["temp"]),
        "humidity": float(row["humidity"]),
        "wind_speed": float(row["wind_speed"]),
        "pressure": float(row["pressure"]),
        "clouds_all": float(row["clouds_all"]),
        "rain_1h": float(row.get("rain_1h", 0.0)),
    }


def main():
    sample = _sample_raw()
    print(">>> /health")
    print(requests.get(f"{BASE}/health").json())

    print("\n>>> /predict (原始输入)")
    for _ in range(5):   # 多发几条以积累漂移样本
        r = requests.post(f"{BASE}/predict", json=sample, headers=HEADERS)
        print(r.json())
        time.sleep(0.1)

    print("\n>>> /monitor/drift")
    d = requests.get(f"{BASE}/monitor/drift", headers=HEADERS).json()
    print(d)

    print("\n>>> /metrics (前若干行)")
    m = requests.get(f"{BASE}/metrics").text
    print("\n".join(line for line in m.splitlines()
                    if line.startswith(("risk_", "# HELP risk_", "# TYPE risk_")))[:1200])


if __name__ == "__main__":
    main()
