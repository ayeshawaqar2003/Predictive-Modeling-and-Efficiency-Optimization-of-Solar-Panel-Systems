# Predictive-Modeling-and-Efficiency-Optimization-of-Solar-Panel-Systems
# Predictive Maintenance for Solar PV Systems 🌞

This repository contains the complete implementation of our undergraduate final year project at NED University of Engineering & Technology, titled:

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

### 📁 Repository Structure

- `docs/`: Final thesis PDF
- `data/`: Cleaned datasets (sample only)
- `notebooks/`: Jupyter notebooks for EDA, training, and evaluation
- `src/`: Core Python scripts for preprocessing, model, and utils
- `streamlit_app/`: Streamlit web UI for visualization and real-time insights

### 📸 Dashboard Preview

![Streamlit UI Screenshot](docs/dashboard_screenshot.png)

### 👩‍💻 Contributors

- **Ayesha Waqar** – AI Modeling, Dashboard Development  
- **Syeda Michelle Sajjad** – Data Engineering, SCADA Logs  
- **Sabrina Khan** – Preprocessing, Literature Review

Supervised by: Ms. Arham Iqbal and Dr. Ghous Bakhsh  
Industry Partner: Reon Energy Ltd

### 📜 License

MIT License – see [LICENSE](./LICENSE) for details.

