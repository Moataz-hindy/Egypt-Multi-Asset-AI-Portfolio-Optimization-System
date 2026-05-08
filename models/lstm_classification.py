# =============================================================================
# lstm_classification.py
# LSTM-based binary / multi-class classifier
# Target  : label column
# Scaler  : MinMaxScaler
# Sequence: config.LSTM_SEQ_LEN (default = 30)
# =============================================================================

import os
import numpy as np
import pandas as pd

# Suppress TF info logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

from config import (
    CLASSIFICATION_TARGET, TRAIN_RATIO, RANDOM_STATE,
    LSTM_SEQ_LEN, LSTM_EPOCHS, LSTM_BATCH, LSTM_UNITS,
    LSTM_DROPOUT, LSTM_LR, LSTM_CLASS_PLOTS_DIR,
)
from utils import (
    get_feature_columns, select_features, time_series_split,
    minmax_scale, build_sequences, classification_metrics,
    save_predictions_classification, save_metrics_table,
    plot_confusion_matrix, plot_lstm_training_curves,
)


# =============================================================================
# Model builder
# =============================================================================

def _build_lstm_classifier(seq_len: int, n_features: int, n_classes: int) -> tf.keras.Model:
    """Construct a two-layer LSTM classification model."""
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

        # Output: softmax for multi-class, sigmoid for binary
        Dense(n_classes, activation="softmax" if n_classes > 2 else "sigmoid"),
    ])

    loss = "categorical_crossentropy" if n_classes > 2 else "binary_crossentropy"
    model.compile(
        optimizer=Adam(learning_rate=LSTM_LR),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


# =============================================================================
# Label encoding helpers
# =============================================================================

def _encode_labels(y_train, y_test):
    """Map arbitrary integer labels → 0-based ints; return mappings."""
    classes      = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())
    label_to_int = {c: i for i, c in enumerate(classes)}
    int_to_label = {i: c for c, i in label_to_int.items()}
    y_train_enc  = np.array([label_to_int[v] for v in y_train])
    y_test_enc   = np.array([label_to_int[v] for v in y_test])
    return y_train_enc, y_test_enc, classes, int_to_label


# =============================================================================
# Main runner
# =============================================================================

def run_lstm_classifier(df: pd.DataFrame, dataset_name: str) -> dict:
    """Train and evaluate the LSTM classifier on one dataset."""
    print(f"\n[LSTM Classifier] Dataset: {dataset_name}")

    # Use full standard feature set (LSTM can handle all features)
    feature_cols = get_feature_columns(dataset_name, model_type="standard")
    X, y, idx, used_features = select_features(df, feature_cols, CLASSIFICATION_TARGET)

    if len(X) < LSTM_SEQ_LEN + 10:
        print(f"  [SKIP] Not enough samples for seq_len={LSTM_SEQ_LEN} ({len(X)} rows).")
        return {}

    # Chronological split BEFORE building sequences
    X_train, X_test, y_train, y_test = time_series_split(X, y, TRAIN_RATIO)

    # MinMax scale
    X_train_s, X_test_s, _ = minmax_scale(X_train, X_test)

    # Build sequences
    X_train_seq, y_train_seq = build_sequences(X_train_s, y_train, LSTM_SEQ_LEN)
    X_test_seq,  y_test_seq  = build_sequences(X_test_s,  y_test,  LSTM_SEQ_LEN)

    if len(X_train_seq) < 10 or len(X_test_seq) < 5:
        print("  [SKIP] Too few sequences after windowing.")
        return {}

    # Encode labels
    y_train_enc, y_test_enc, classes, int_to_label = _encode_labels(y_train_seq, y_test_seq)
    n_classes  = len(classes)
    n_features = X_train_seq.shape[2]

    # One-hot encode for categorical_crossentropy
    y_train_oh = to_categorical(y_train_enc, num_classes=n_classes)
    y_test_oh  = to_categorical(y_test_enc,  num_classes=n_classes)

    # Build and train
    model = _build_lstm_classifier(LSTM_SEQ_LEN, n_features, n_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=0),
    ]

    history = model.fit(
        X_train_seq, y_train_oh,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
        validation_split=0.15,       # last 15 % of training set as validation
        callbacks=callbacks,
        shuffle=False,               # CRITICAL: keep time order
        verbose=1,
    )

    # Predictions
    y_prob_oh   = model.predict(X_test_seq)
    y_pred_enc  = np.argmax(y_prob_oh, axis=1)
    y_pred      = np.array([int_to_label[v] for v in y_pred_enc])
    y_true      = np.array([int_to_label[v] for v in y_test_enc])

    metrics = classification_metrics(y_true, y_pred, "LSTM_Classifier", dataset_name)

    # Indices for test set (offset by seq_len because sequences consume the first seq_len rows)
    train_end    = len(X_train)
    test_seq_idx = idx[train_end + LSTM_SEQ_LEN: train_end + LSTM_SEQ_LEN + len(y_test_seq)]

    save_predictions_classification(
        dataset_name, "LSTM_Classifier", y_true, y_pred, y_prob_oh, test_seq_idx
    )

    # Confusion matrix
    mapping  = {-1: "Sell", 0: "Hold", 1: "Buy"}
    cm       = confusion_matrix(y_true, y_pred, labels=classes)
    cls_names = [mapping.get(c, str(c)) for c in classes]
    plot_confusion_matrix(cm, cls_names, "LSTM_Classifier", dataset_name, LSTM_CLASS_PLOTS_DIR)

    # Training curves
    plot_lstm_training_curves(
        history, "LSTM_Classifier", dataset_name, LSTM_CLASS_PLOTS_DIR, task="classification"
    )

    return metrics


# =============================================================================
# Master runner
# =============================================================================

def run_all_lstm_classifiers(df: pd.DataFrame, dataset_name: str):
    """Wrapper to run LSTM classifier and persist results."""
    result = run_lstm_classifier(df, dataset_name)
    if result:
        save_metrics_table([result], "lstm_classification", dataset_name)
    return [result] if result else []
