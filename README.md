# Solar Panel Predictive Maintenance System 🌞

A machine learning pipeline that predicts which solar panel blocks need cleaning — before efficiency drops, not after.

Built as my bachelor's thesis. The real problem: managing acres of solar panels in an environment where sandstorms and dust are constant. The old solution was washing everything every week regardless. That wastes water, labor, and time. This system learns from three years of performance and weather data to predict *which* blocks actually need maintenance, and *when*.

---

## The Problem

Large-scale solar arrays lose efficiency gradually due to dust and sandstorm accumulation. Traditional maintenance schedules are fixed — wash everything, every week. This is:

- **Wasteful** — most panels don't need cleaning that frequently
- **Expensive** — labor and water costs at scale are significant
- **Inefficient** — fixed schedules miss panels that actually need urgent attention

This system replaces guesswork with data-driven predictions.

---

## What It Does

- Ingests historical solar panel performance data (Expected kWh vs Actual kWh) alongside weather and environmental variables
- Engineers a performance gap feature and classifies panel blocks into 4 maintenance urgency categories (0 = optimal, 3 = critical)
- Trains and compares three deep learning architectures — **Bidirectional LSTM, GRU, and RNN** — across multiple sequence window lengths (7, 15, 30, 60 days)
- Uses **TimeSeriesSplit cross-validation** to ensure no future data leaks into training
- Outputs a ranked comparison table of all model/timestep combinations by F1 score, accuracy, precision, and recall

---

## Model Architectures

| Architecture | Details |
|---|---|
| **Bidirectional LSTM** | Stacked 2-layer BiLSTM with batch normalization and dropout |
| **GRU** | Stacked 2-layer GRU with batch normalization and dropout |
| **SimpleRNN** | Baseline with batch normalization and dropout |

All models use:
- **RobustScaler** — handles outliers from sandstorm events better than MinMaxScaler
- **Softmax classification head** — 4-class output (maintenance urgency levels)
- **Early stopping + learning rate reduction** — prevents overfitting automatically
- **TimeSeriesSplit** — strict chronological cross-validation, no data leakage

---

## Results

The pipeline outputs a full benchmark comparison table across all architecture and timestep combinations. The top-performing configuration was the 30-day window using a GRU network:

| Architecture | Timesteps | Mean Accuracy | Mean F1-Score | Mean Precision | Mean Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GRU** | 30 | 0.8421 | 0.8214 | 0.8250 | 0.8190 |
| **LSTM** | 30 | 0.8355 | 0.8108 | 0.8145 | 0.8085 |
| **RNN** | 15 | 0.6850 | 0.6412 | 0.6530 | 0.6385 |



## Tech Stack

* **Language:** Python 3.9+
* **Deep Learning Frameworks:** TensorFlow / Keras
* **Machine Learning & Data Engineering:** scikit-learn, NumPy, pandas
* **Data Visualization:** Matplotlib, Seaborn


 
---
## Key Design Decisions
 
**Why RobustScaler instead of MinMaxScaler?**
Solar panel data contains extreme outliers during sandstorm events. RobustScaler uses median and interquartile range instead of min/max, making it significantly more stable under these conditions.
 
**Why TimeSeriesSplit instead of random train/test split?**
Random splitting on time-series data leaks future information into training — the model learns from data that wouldn't exist yet at prediction time. TimeSeriesSplit enforces strict chronological order, making performance estimates realistic.
 
**Why 4 classification categories instead of regression?**
Maintenance scheduling is a decision problem, not a continuous prediction problem. A maintenance crew needs to know *how urgently* a block needs attention — not a raw efficiency number. Four quartile-based categories map directly to actionable priority levels.
 
---
 
## Future Work
 
- Extend predictions to non-weather maintenance issues (inverter faults, wiring degradation)
- Add real-time data ingestion pipeline
- Incorporate block-level geospatial mapping for crew routing optimization

## Author

**Ayesha Waqar** — Electronics Engineering graduate, Master's student specializing in AI


📧 [ayeshagul2003@hotmail.com · 💼 www.linkedin.com/in/ayeshagulwaqar · 🐙github.com/ayeshawaqar2003




### 📜 License

MIT License – see [LICENSE](./LICENSE) for details.

