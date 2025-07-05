# Predictive-Modeling-and-Efficiency-Optimization-of-Solar-Panel-Systems
# Predictive Maintenance for Solar PV Systems 🌞

This repository contains the implementation of our undergraduate final year project at NED University of Engineering & Technology, titled:

**"Predictive Modeling and Efficiency Optimization of Solar Panel Systems"**

## 🚀 Project Summary

We designed a real-time, AI-based predictive maintenance system for solar PV installations using LSTM neural networks. The system forecasts inverter output and flags anomalies due to soiling, temperature, or inverter failure using multivariate SCADA data and environmental inputs (PM2.5, humidity, etc.).

### 🔧 Core Technologies

- Python, Pandas, NumPy
- TensorFlow, Keras (LSTM modeling)
- Streamlit (dashboard)
- Huawei SmartLogger 3000A (data acquisition)
- SCADA logs from Reon Energy Ltd

### 📊 Features

- Multivariate LSTM forecasting for energy output
- Anomaly classification into 4 actionable maintenance levels
- Real-time interactive Streamlit dashboard
- Retraining mechanism for adaptive seasonal learning
- Performance metrics: MAE, RMSE, R², Precision, Recall

### 🧪 Results

- **MAE:** 4.6%
- **RMSE:** 6.9%
- **Recall:** 91%
- **Precision:** 83%
- Anomalies detected 2–5 days in advance



### 📸 Dashboard Preview

![image](https://github.com/user-attachments/assets/1c9a68d6-d2c8-49d9-ae0c-11dbd35d4b8c)





### 📜 License

MIT License – see [LICENSE](./LICENSE) for details.

