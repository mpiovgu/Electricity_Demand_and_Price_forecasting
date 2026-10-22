"""特征工程：合并能源/天气数据，构造无泄露的训练特征与原始输入→特征转换。

风控目标：用 t 时刻「已知信息」（日前预测、天气、历史实际值）预测
t 时刻实际电价是否进入高位分位（风险事件）。特征严格不包含
t 时刻的 price actual / total load actual，避免标签泄露。
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import math
import numpy as np
import pandas as pd
from risk_model.config import ENERGY_CSV, WEATHER_CSV, RISK_PRICE_QUANTILE

# 模型特征契约（训练与服务必须保持一致的顺序）
FEATURE_COLS = [
    "total_load_forecast",                # t 时刻负荷日前预测（已知）
    "price_day_ahead",                    # t 时刻电价日前预测（已知）
    "forecast_solar",                     # 光伏日前预测
    "forecast_wind_onshore",              # 陆风日前预测（海风日前预测整库为空，已剔除）
    "total_load_actual_prev",             # t-1 实际负荷（历史已知）
    "price_actual_prev",                  # t-1 实际电价
    "price_actual_prev2",                 # t-2 实际电价
    "load_forecast_error",                # 预测负荷 - 上小时实际负荷
    "price_actual_diff",                  # t-1 与 t-2 电价差
    "renewable_forecast_share",           # 可再生预测出力占比
    "price_forecast_minus_actual_prev",   # 日前电价 - 上小时实际电价
    "temp", "humidity", "wind_speed", "pressure", "clouds_all", "rain_1h",
    "temp_sq",                            # 温度非线性（取暖/制冷）
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month", "is_weekend",
]

WEATHER_NUM = ["temp", "temp_min", "temp_max", "pressure", "humidity",
               "wind_speed", "wind_deg", "rain_1h", "rain_3h", "snow_3h",
               "clouds_all", "weather_id"]


def load_merged() -> pd.DataFrame:
    """读取能源表，并将 5 城市天气按时间聚合为全国统一特征后合并。"""
    # 两表均含 +01:00 时区且存在混合偏移，统一转 UTC 后去时区，保证可合并
    e = pd.read_csv(ENERGY_CSV)
    e["time"] = pd.to_datetime(e["time"], utc=True).dt.tz_convert(None)
    w = pd.read_csv(WEATHER_CSV)
    w["dt_iso"] = pd.to_datetime(w["dt_iso"], utc=True).dt.tz_convert(None)
    wg = (w.groupby("dt_iso")[WEATHER_NUM].mean().reset_index()
            .rename(columns={"dt_iso": "time"}))
    df = e.merge(wg, on="time", how="inner")
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _add_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_sin"] = np.sin(2 * np.pi * df["time"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["time"].dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["time"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["time"].dt.dayofweek / 7)
    df["month"] = df["time"].dt.month
    df["is_weekend"] = (df["time"].dt.dayofweek >= 5).astype(int)
    return df


def build_dataset():
    """构建训练数据集，返回 (X, y, df_eng, price_threshold)。"""
    df = load_merged()
    df = _add_cyclical(df)

    # 将带空格的原始能源列映射为无空格的特征名（与 FEATURE_COLS / 服务一致）
    df["total_load_forecast"] = df["total load forecast"]
    df["price_day_ahead"] = df["price day ahead"]
    df["forecast_solar"] = df["forecast solar day ahead"]
    df["forecast_wind_onshore"] = df["forecast wind onshore day ahead"]

    # 滞后项（t-1, t-2，均为历史已知）
    df["total_load_actual_prev"] = df["total load actual"].shift(1)
    df["price_actual_prev"] = df["price actual"].shift(1)
    df["price_actual_prev2"] = df["price actual"].shift(2)

    # 派生特征（不泄露 t 时刻真实值）
    df["load_forecast_error"] = df["total_load_forecast"] - df["total_load_actual_prev"]
    df["price_actual_diff"] = df["price_actual_prev"] - df["price_actual_prev2"]
    df["renewable_forecast"] = df["forecast_solar"] + df["forecast_wind_onshore"]
    df["renewable_forecast_share"] = df["renewable_forecast"] / df["total_load_forecast"].replace(0, np.nan)
    df["price_forecast_minus_actual_prev"] = df["price_day_ahead"] - df["price_actual_prev"]
    df["temp_sq"] = df["temp"] ** 2

    # 目标：t 时刻实际电价处于历史 P90 以上 -> 风险事件
    q = df["price actual"].quantile(RISK_PRICE_QUANTILE)
    df["risk"] = (df["price actual"] > q).astype(int)

    df = df.dropna(subset=FEATURE_COLS + ["risk"]).reset_index(drop=True)
    X = df[FEATURE_COLS].copy()
    y = df["risk"].copy()
    return X, y, df, float(q)


def build_features_from_raw(raw: dict) -> pd.DataFrame:
    """由单次原始输入构造与服务端一致的特征向量（用于在线打分）。"""
    ts = raw.get("timestamp")
    t = pd.to_datetime(ts) if ts else pd.Timestamp.now()
    hour, dow, month = t.hour, t.dayofweek, t.month

    tlf = float(raw["total_load_forecast"])
    pap = float(raw["price_actual_prev"])
    pap2 = float(raw["price_actual_prev2"])
    fsol = float(raw["forecast_solar"])
    fwon = float(raw["forecast_wind_onshore"])
    pda = float(raw["price_day_ahead"])

    rec = {
        "total_load_forecast": tlf,
        "price_day_ahead": pda,
        "forecast_solar": fsol,
        "forecast_wind_onshore": fwon,
        "total_load_actual_prev": float(raw["total_load_actual_prev"]),
        "price_actual_prev": pap,
        "price_actual_prev2": pap2,
        "load_forecast_error": tlf - float(raw["total_load_actual_prev"]),
        "price_actual_diff": pap - pap2,
        "renewable_forecast_share": (fsol + fwon) / tlf if tlf else 0.0,
        "price_forecast_minus_actual_prev": pda - pap,
        "temp": float(raw["temp"]),
        "humidity": float(raw["humidity"]),
        "wind_speed": float(raw["wind_speed"]),
        "pressure": float(raw["pressure"]),
        "clouds_all": float(raw["clouds_all"]),
        "rain_1h": float(raw.get("rain_1h", 0.0)),
        "temp_sq": float(raw["temp"]) ** 2,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "dow_sin": math.sin(2 * math.pi * dow / 7),
        "dow_cos": math.cos(2 * math.pi * dow / 7),
        "month": month,
        "is_weekend": int(dow >= 5),
    }
    return pd.DataFrame([rec])[FEATURE_COLS]
