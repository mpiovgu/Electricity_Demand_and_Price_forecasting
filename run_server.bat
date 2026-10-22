@echo off
REM 启动电力价格风控模型服务（脱离终端，可双击或 start 调用）
cd /d "c:/Users/admin/Desktop/fsdownload/Electricity_Demand_and_Price_forecasting"
C:/Users/admin/anaconda3/python.exe -m uvicorn risk_model.app:app --host 127.0.0.1 --port 8000 >> "c:/Users/admin/Desktop/fsdownload/Electricity_Demand_and_Price_forecasting/artifacts/server.log" 2>&1
