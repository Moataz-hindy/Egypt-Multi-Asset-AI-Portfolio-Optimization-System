# =============================================================================
# lstm_regression.py
# LSTM-based regression model
# Target  : return column
# Scaler  : MinMaxScaler (fit on features; return scaled separately)
# Sequence: config.LSTM_SEQ_LEN (default = 30)
# =============================================================================

import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from config import (
    REGRESSION_TARGET, TRAIN_RATIO, RANDOM_STATE,
    LSTM_SEQ_LEN, LSTM_EPOCHS, LSTM_BATCH, LSTM_UNITS,
    LSTM_DROPOUT, LSTM_LR, LSTM_REG_PLOTS_DIR,
)
from utils import (
    get_feature_columns, select_features, time_series_split,
    minmax_scale, build_sequences, regression_metrics,
    save_predictions_regression, save_metrics_table,
    plot_actual_vs_predicted, plot_lstm_training_curves,
)


# =============================================================================
# Model builder
# =============================================================================

def _build_lstm_regressor(seq_len: int, n_features: int) -> tf.keras.Model:
    """Construct a two-layer LSTM regression model."""
    tf.random.set_seed(RANDOM_STATE)

    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True,
             input_shape=(seq_len, n_features)),
        Dropout(LSTM_DROPOUT),
        BatchNormalization(),

        LSTM(LSTM_UNITS // 2, return_sequences=False),
        Dropout(LSTM_DROPOUT),
        BatchNormalization(),

        Dense(32, activation="relu"),
        Dropout(LSTM_DROPOUT / 2),

        Dense(1, activation="linear"),   # single continuous output
    ])

    model.compile(
        optimizer=Adam(learning_rate=LSTM_LR),
        loss="mse",
        metrics=["mae"],
    )
    return model


# =============================================================================
# Main runner
# =============================================================================

def run_lstm_regressor(df: pd.DataFrame, dataset_name: str) -> dict:
    """Train and evaluate the LSTM regressor on one dataset."""
    print(f"\n[LSTM Regressor] Dataset: {dataset_name}")

    feature_cols = get_feature_columns(dataset_name, model_type="standard")
    X, y, idx, used_features = select_features(df, feature_cols, REGRESSION_TARGET)

    if len(X) < LSTM_SEQ_LEN + 10:
        print(f"  [SKIP] Not enough samples for seq_len={LSTM_SEQ_LEN} ({len(X)} rows).")
        return {}

    # Chronological split
    X_train, X_test, y_train, y_test = time_series_split(X, y, TRAIN_RATIO)

    # -----------------------------------------------------------------------
    # Scale features with MinMaxScaler (fit on train only)
    # -----------------------------------------------------------------------
    X_train_s, X_test_s, feat_scaler = minmax_scale(X_train, X_test)

    # -----------------------------------------------------------------------
    # Scale the TARGET independently so we can invert predictions later
    # -----------------------------------------------------------------------
    y_scaler    = MinMaxScaler()
    y_train_s   = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_s    = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # Build sequences
    X_train_seq, y_train_seq = build_sequences(X_train_s, y_train_s, LSTM_SEQ_LEN)
    X_test_seq,  y_test_seq  = build_sequences(X_test_s,  y_test_s,  LSTM_SEQ_LEN)

    if len(X_train_seq) < 10 or len(X_test_seq) < 5:
        print("  [SKIP] Too few sequences after windowing.")
        return {}

    n_features = X_train_seq.shape[2]

    model = _build_lstm_regressor(LSTM_SEQ_LEN, n_features)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=0),
    ]

    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
        validation_split=0.15,
        callbacks=callbacks,
        shuffle=False,         # preserve time order
        verbose=1,
    )

    # Predictions (scaled space)
    y_pred_s = model.predict(X_test_seq).flatten()

    # Invert scaling to get real-world return values
    y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    y_true = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    metrics = regression_metrics(y_true, y_pred, "LSTM_Regressor", dataset_name)

    # Indices for the test sequences
    train_end    = len(X_train)
    test_seq_idx = idx[train_end + LSTM_SEQ_LEN: train_end + LSTM_SEQ_LEN + len(y_test_seq)]

    save_predictions_regression(
        dataset_name, "LSTM_Regressor", y_true, y_pred, test_seq_idx
    )
    plot_actual_vs_predicted(
        y_true, y_pred, "LSTM_Regressor", dataset_name, LSTM_REG_PLOTS_DIR
    )
    plot_lstm_training_curves(
        history, "LSTM_Regressor", dataset_name, LSTM_REG_PLOTS_DIR, task="regression"
    )

    return metrics


# =============================================================================
# Master runner
# =============================================================================

def run_all_lstm_regressors(df: pd.DataFrame, dataset_name: str):
    """Wrapper to run LSTM regressor and persist results."""
    result = run_lstm_regressor(df, dataset_name)
    if result:
        save_metrics_table([result], "lstm_regression", dataset_name)
    return [result] if result else []
