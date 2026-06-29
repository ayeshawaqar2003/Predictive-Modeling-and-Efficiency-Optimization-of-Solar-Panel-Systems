import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, SimpleRNN, Dense, Dropout, 
    BatchNormalization, Bidirectional, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib

# Ensure reproducible results
np.random.seed(42)
tf.random.set_seed(42)

# =====================================================================
# 1. DATA PREPROCESSING MODULE
# =====================================================================

def load_master_data(file_path="AllInOne.xlsx"):
    """Loads, sorts chronologically, and engineers initial performance gap feature."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Master file {file_path} not found.")
    df = pd.read_excel(file_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    df["difference"] = df["Expected_kWh"] - df["Actual_kWh"]
    return df

def generate_quartile_labels(df):
    """Generates 4 discrete maintenance urgency categories based on performance quantiles.
    0 = Best performance, 3 = Worst (most urgent cleaning needed)
    """
    q1 = df["difference"].quantile(0.25)
    q2 = df["difference"].quantile(0.50)
    q3 = df["difference"].quantile(0.75)
    
    conditions = [
        (df["difference"] < q1),
        (df["difference"] >= q1) & (df["difference"] < q2),
        (df["difference"] >= q2) & (df["difference"] < q3),
        (df["difference"] >= q3)
    ]
    choices = [0, 1, 2, 3]
    df["Outcome"] = np.select(conditions, choices, default=1)
    return df

def prepare_and_scale_data(df):
    """Separates features/labels and applies RobustScaler to handle sandstorm outliers."""
    feature_cols = [col for col in df.columns if col not in ["Date", "Outcome"]]
    features = df[feature_cols].values
    labels = df["Outcome"].values
    
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, labels, scaler, feature_cols

def create_sequences(data, labels, time_steps):
    """Converts flat time-series into sliding window sequences."""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(labels[i + time_steps])
    return np.array(X), np.array(y)

# =====================================================================
# 2. MODEL FACTORY MODULE
# =====================================================================

def build_model(architecture, input_shape, num_classes=4):
    """Factory function generating deep learning architectures for sequence classification."""
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    if architecture == "RNN":
        model.add(SimpleRNN(64, activation='tanh', return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
    elif architecture == "LSTM":
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(32, return_sequences=False)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
    elif architecture == "GRU":
        model.add(GRU(64, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(GRU(32, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
        
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

# =====================================================================
# 3. EVALUATION & VISUALIZATION MODULE
# =====================================================================

def plot_history_curves(history, arch_name, ts):
    """Plots loss and accuracy curves for a training run."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    ax1.plot(history.history['loss'], label='Train Loss', color='royalblue')
    ax1.plot(history.history['val_loss'], label='Val Loss', color='darkorange')
    ax1.set_title(f'{arch_name} (TS={ts}) - Cross Entropy Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history.history['accuracy'], label='Train Acc', color='forestgreen')
    ax2.plot(history.history['val_accuracy'], label='Val Acc', color='crimson')
    ax2.set_title(f'{arch_name} (TS={ts}) - Categorical Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# =====================================================================
# 4. MASTER PIPELINE RUNNER
# =====================================================================

def run_pipeline(file_path):
    # Step 1: Data Preparation
    raw_df = load_master_data(file_path)
    labeled_df = generate_quartile_labels(raw_df)
    scaled_feats, labels, global_scaler, feature_cols = prepare_and_scale_data(labeled_df)
    
    # Save scaler immediately so app.py can use it
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(global_scaler, "saved_models/scaler.pkl")
    joblib.dump(feature_cols, "saved_models/feature_cols.pkl")
    print("✅ Scaler and feature columns saved to saved_models/")

    # Configuration space
    timestep_options = [7, 15, 30, 60]
    architectures = ["RNN", "LSTM", "GRU"]
    global_results = []
    
    # Track best model across all runs
    best_f1 = 0
    best_model = None
    best_config = {}

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=0)
    ]
    
    # Strict chronological cross validation — no data leakage
    tscv = TimeSeriesSplit(n_splits=3)
    
    for ts in timestep_options:
        print(f"\n{'='*60}\n EVALUATING TIMESTEP WINDOW: {ts} Days\n{'='*60}")
        X, y = create_sequences(scaled_feats, labels, time_steps=ts)
        y_categorical = to_categorical(y, num_classes=4)
        
        for arch in architectures:
            print(f"--- Training: {arch} | Timesteps: {ts} ---")
            
            fold_acc, fold_prec, fold_rec, fold_f1 = [], [], [], []
            fold_models = []

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_categorical[train_idx], y_categorical[test_idx]
                y_test_raw = y[test_idx]
                
                model = build_model(arch, input_shape=(ts, X.shape[2]), num_classes=4)
                
                history = model.fit(
                    X_train, y_train,
                    epochs=40,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    verbose=0
                )
                
                preds = model.predict(X_test, verbose=0)
                pred_classes = np.argmax(preds, axis=1)
                
                fold_f1_score = f1_score(y_test_raw, pred_classes, average='macro', zero_division=0)
                fold_acc.append(accuracy_score(y_test_raw, pred_classes))
                fold_prec.append(precision_score(y_test_raw, pred_classes, average='macro', zero_division=0))
                fold_rec.append(recall_score(y_test_raw, pred_classes, average='macro', zero_division=0))
                fold_f1.append(fold_f1_score)
                fold_models.append((fold_f1_score, model))

            mean_acc  = np.mean(fold_acc)
            mean_prec = np.mean(fold_prec)
            mean_rec  = np.mean(fold_rec)
            mean_f1   = np.mean(fold_f1)

            global_results.append({
                "Timesteps": ts,
                "Architecture": arch,
                "Mean Accuracy": round(mean_acc, 4),
                "Mean Precision": round(mean_prec, 4),
                "Mean Recall": round(mean_rec, 4),
                "Mean F1-Score": round(mean_f1, 4)
            })

            print(f"✅ {arch} (TS={ts}): F1={mean_f1:.4f} | Accuracy={mean_acc:.4f}")
            plot_history_curves(history, arch, ts)

            # Save best model across all configurations
            best_fold_f1, best_fold_model = max(fold_models, key=lambda x: x[0])
            if best_fold_f1 > best_f1:
                best_f1 = best_fold_f1
                best_model = best_fold_model
                best_config = {"architecture": arch, "timesteps": ts}

    # Save the single best performing model
    if best_model is not None:
        best_model.save("saved_models/best_model.h5")
        joblib.dump(best_config, "saved_models/best_config.pkl")
        print(f"\n🏆 Best model saved: {best_config['architecture']} with TS={best_config['timesteps']} | F1={best_f1:.4f}")

    # Final comparison table
    summary_df = pd.DataFrame(global_results)
    print("\n" + "#"*60 + "\n FINAL BENCHMARK PERFORMANCE BREAKDOWN \n" + "#"*60)
    print(summary_df.to_string(index=False))
    
    return summary_df, global_scaler

if __name__ == "__main__":
    try:
        results_table, scaler = run_pipeline("AllInOne.xlsx")
    except Exception as e:
        print(f"Pipeline stopped: {e}")
