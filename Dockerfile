# 风控模型服务镜像（可选：将服务也容器化，配合 docker-compose 中 risk-model 段落）
# 说明：仅复制代码与已训练产物 artifacts/，无需 350MB 原始数据。
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY risk_model ./risk_model
COPY artifacts ./artifacts

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "risk_model.app:app", "--host", "0.0.0.0", "--port", "8000"]
