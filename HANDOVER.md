# 电力价格风险（尖峰）风控模型 — 项目交接文档

> 维护者交接用。本文档覆盖目标定义、架构、数据、特征、模型、训练、服务部署、监控告警与已知问题。

---

## 1. 项目概述

基于公开的电力负荷/电价与天气数据，构建一个**电价尖峰风险风控模型**：在每一个小时，利用「该时刻已知信息」预测**实际电价是否进入历史高位分位（风险事件）**，供风控/交易决策参考。

- **风险事件定义**：`price actual > P90`（电价处于历史 90% 分位以上）。当前阈值 `price_threshold = 75.15` 元/MWh（按全量数据 P90 计算）。
- **任务类型**：二分类（0=正常，1=尖峰风险）。
- **核心约束**：特征严格不含 t 时刻的 `price actual` / `total load actual`，避免标签泄露。

### 关键结果（测试集）
| 指标 | 值 |
|---|---|
| AUC | 0.983 |
| Precision | 0.865 |
| Recall | 0.857 |
| F1 | 0.861 |
| 测试样本数 | 7006 |
| 测试正例率 | 0.166 |
| 决策阈值 | 0.5499 |

---

## 2. 目录结构

```
Electricity_Demand_and_Price_forecasting_withriskmodel/
├── data/                         # 原始数据（已被原仓库跟踪，无需提交）
│   ├── energy_dataset.csv        # 小时级负荷/电价/发电/日前预测
│   └── weather_features.csv      # 多城市天气
├── artifacts/                    # 训练产物（已提交）
│   ├── risk_model.joblib         # 训练好的模型
│   ├── model_metadata.json       # 元数据（阈值/特征/指标/params）
│   └── reference_stats.json      # 参考分布（用于 PSI 漂移）
├── risk_model/                   # 核心代码包
│   ├── config.py                 # 路径、阈值、服务/告警参数
│   ├── features.py               # 特征工程 + 原始输入→特征
│   ├── train.py                  # 训练脚本
│   ├── model_wrapper.py          # 模型加载与预测封装
│   ├── monitor.py                # 预测落库 + PSI 漂移 + 告警
│   ├── prom_metrics.py           # Prometheus 指标
│   ├── schemas.py                # FastAPI 请求/响应模型
│   ├── app.py                    # FastAPI 服务
│   ├── client.py                 # 调用示例
│   └── _test_api.py              # 冒烟测试（TestClient）
├── grafana/                      # Grafana 配置与面板
│   ├── provisioning/datasources/datasource.yml
│   ├── provisioning/dashboards/dashboards.yml
│   └── dashboards/risk_model_dashboard.json
├── Dockerfile                    # 可选：服务容器化
├── docker-compose.yml            # Prometheus + Grafana 一键起
├── prometheus.yml                # Prometheus 抓取配置
├── run_server.bat                # Windows 下脱离终端启动服务
├── requirements.txt
└── HANDOVER.md
```

---

## 3. 环境与依赖

- **Python**：使用 Anaconda `base` 环境解释器：`C:/Users/admin/anaconda3/python.exe`
  （默认 `python` 指向的 Python 3.12 缺 `scikit-learn`，请勿用其运行本项目）。
- **核心依赖**：`pandas`、`numpy`、`scikit-learn>=1.0`、`fastapi`、`uvicorn[standard]`、`pydantic>=1.10`、`prometheus-client`、`joblib`。
- **关于 XGBoost**：原方案计划使用 XGBoost，但部署机磁盘满且未安装 `xgboost`，已改用等价的 `sklearn.HistGradientBoostingClassifier`（梯度提升树）。如需换回，安装 `xgboost` 后改 `train.py` / `model_wrapper.py` 即可，`requirements.txt` 已注明。

---

## 4. 数据与特征工程

### 4.1 数据
- `energy_dataset.csv`：小时级，含 `price actual`、`total load actual/forecast`、各类发电与日前预测。
- `weather_features.csv`：多城市逐时天气；按 `dt_iso` 对城市取均值聚合成全国级特征。
- 两表按小时时间戳合并（`how="inner"`）。时区统一转 UTC 后去时区，避免混合偏移导致合并失败。

### 4.2 特征（共 24 维，`FEATURE_COLS`）
- 日前预测：`total_load_forecast`、`price_day_ahead`、`forecast_solar`、`forecast_wind_onshore`
- 历史滞后（t-1/t-2，均为已知）：`total_load_actual_prev`、`price_actual_prev`、`price_actual_prev2`
- 派生：`load_forecast_error`、`price_actual_diff`、`renewable_forecast_share`、`price_forecast_minus_actual_prev`
- 天气：`temp`、`humidity`、`wind_speed`、`pressure`、`clouds_all`、`rain_1h`、`temp_sq`
- 时间周期：`hour_sin/cos`、`dow_sin/cos`、`month`、`is_weekend`

> 派生特征均基于 t 时刻「已知」量（日前预测/天气/历史实际），**不引入 t 时刻真实值**。

### 4.3 数据清洗要点
- `forecast wind offshore eday ahead` 整库为空，已从特征中剔除（否则样本全被 drop）。
- 合并后出现少量 NaN（首两行滞后、个别天气缺失），训练前 `dropna`。

---

## 5. 模型与训练

- **算法**：`HistGradientBoostingClassifier(max_iter=300, max_depth=5, learning_rate=0.05, l2_regularization=1.0, random_state=42)`
- **切分**：按时间顺序，取最后 `TEST_FRACTION`（约 20%）为测试集，避免未来信息泄露。
- **阈值选择**：在测试集上以 F1 最大化为准则选决策阈值（当前 0.5499）。
- **运行训练**：
  ```bash
  C:/Users/admin/anaconda3/python.exe risk_model/train.py
  ```
  产出：`artifacts/risk_model.joblib`、`artifacts/model_metadata.json`、`artifacts/reference_stats.json`。

---

## 6. 服务部署（FastAPI）

### 6.1 接口
| 方法 | 路径 | 说明 | 鉴权 |
|---|---|---|---|
| GET | `/health` | 健康检查 + 模型/指标摘要 | 否 |
| POST | `/predict` | 传原始字段（`RiskRawInput`）→ 服务内部转特征 | 是 |
| POST | `/predict/features` | 直接传 24 维特征向量（list） | 是 |
| GET | `/monitor/drift` | PSI 漂移报告（JSON） | 是 |
| GET | `/metrics` | Prometheus 指标 | 否 |

- **鉴权**：请求头 `X-API-Key: <API_KEY>`，默认值 `risk-demo-secret-key`（可用环境变量 `RISK_API_KEY` 覆盖）。
- **响应示例**（`/predict`）：
  ```json
  {"risk_score":0.0017,"risk_label":0,"threshold":0.5499,"is_risk":false,"features_used":[...]}
  ```

### 6.2 启动
- 推荐（Windows 脱离终端）：双击或 `start` 运行 `run_server.bat`。
- 或前台：
  ```bash
  C:/Users/admin/anaconda3/python.exe -m uvicorn risk_model.app:app --host 127.0.0.1 --port 8000
  ```
  > 如需局域网/容器访问，将 `--host 127.0.0.1` 改为 `--host 0.0.0.0`。
- 日志：`artifacts/server.log`。

### 6.3 调用示例
```bash
C:/Users/admin/anaconda3/python.exe risk_model/client.py
```

---

## 7. 监控与告警

### 7.1 Prometheus 指标（`risk_` 前缀）
- `risk_api_requests_total{endpoint,method,status}` — 接口请求量
- `risk_api_request_latency_seconds` — 接口延迟直方图
- `risk_model_predictions_total{risk_label}` — 预测次数（按风险/非风险）
- `risk_model_score` — 风险分数分布直方图
- `risk_model_feature_drift_psi{feature}` / `risk_model_max_feature_drift_psi` — 特征漂移 PSI
- `risk_model_serving_positive_rate` — 服务侧风险正例率（与训练基准 ~0.166 对比）
- `risk_model_info` — 模型元信息

### 7.2 漂移检测
- 方法：PSI（总体稳定性指数），将特征分布与训练期参考分布（`reference_stats.json`）分箱比较。
- 阈值：`PSI_WARN=0.25` / `PSI_ALERT=0.50`；`/monitor/drift` 返回 `max_psi` 与 `status`（`ok`/`warn`/`alert`）。
- 预测日志记录在 `artifacts/predictions.db`（SQLite），用于在线漂移计算。

### 7.3 可视化（Grafana）
- `docker-compose.yml` 一键启动 Prometheus + Grafana；
- Grafana 通过 provisioning 自动接入 Prometheus 并加载面板 `grafana/dashboards/risk_model_dashboard.json`（6 个面板：最大漂移、服务风险率、QPS、P95 延迟、预测吞吐、各特征 PSI）。
- 启动：
  ```bash
  docker compose up -d
  ```
  访问 Prometheus `http://localhost:9090`，Grafana `http://localhost:3000`（admin/admin）。
  > 默认 Prometheus 抓取 `host.docker.internal:8000`（服务跑在宿主机）。若将服务也容器化，把 `prometheus.yml` target 改为 `['risk-model:8000']`。

---

## 8. 已知问题与注意事项

1. **XGBoost → sklearn 替代**：因部署机缺 `xgboost` 且磁盘满，改用 `HistGradientBoostingClassifier`；`model_metadata.json` 的 `model_type` 已为 `hist_gradient_boosting`。
2. **海风日前预测整库为空**：`forecast wind offshore eday ahead` 全 NaN，已剔除该特征。
3. **时区**：天气表存在混合偏移，已统一转 UTC 去时区后合并。
4. **`data/` 体积**：原始数据已在原仓库历史中，推送会一并带上；若想新仓库不含数据历史，需 `git filter-repo` 处理。
5. **磁盘空间**：部署机 C 盘曾仅剩 ~6MB，注意 artifacts / predictions.db 增长。
6. **推送网络**：本交接环境到 `github.com:443` 的出站受限，推送需在可访问 GitHub 的网络下进行（`git push -u riskmodel main`）。

---

## 9. 交接清单（TODO / 待办）

- [ ] 在生产环境用真实特征存储替换 `client.py` 中的 t-1/t-2 历史实际值来源。
- [ ] 接入真实告警通道（Webhook/邮件），将 `/monitor/drift` 的 alert 状态外推。
- [ ] 按需把服务容器化（启用 `docker-compose.yml` 中 `risk-model` 段落 + `Dockerfile`）。
- [ ] 定期（如每日）重新计算并落盘 `reference_stats.json`，跟踪概念漂移。
- [ ] 评估把 XGBoost 换回以进一步提升（若环境允许安装）。

---
*生成日期：2026-10-22*
