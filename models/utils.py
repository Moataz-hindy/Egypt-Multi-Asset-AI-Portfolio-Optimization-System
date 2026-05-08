# =============================================================================
# utils.py — Shared utility functions for the ML pipeline
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score,
)

from config import (
    STANDARD_FEATURES, LR_FEATURES,
    TBILLS_FEATURES, TBILLS_LR_FEATURES,
    CLASSIFICATION_TARGET, REGRESSION_TARGET,
    TRAIN_RATIO, LSTM_SEQ_LEN, ALL_OUTPUT_DIRS,
    PREDICTIONS_DIR, METRICS_DIR,
)


# =============================================================================
# 1. Directory setup
# =============================================================================

def create_output_dirs():
    """Create all required output directories if they don't exist."""
    for d in ALL_OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)
    print("[INFO] Output directories ready.")


# =============================================================================
# 2. Data loading & feature selection
# =============================================================================

def load_dataset(path: str, dataset_name: str) -> pd.DataFrame:
    """
    Load a CSV, parse dates, drop NaN rows, and return a clean DataFrame.
    'return' column is kept in the frame (as target) but NEVER used as a feature.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)   # ensure chronological order
    df = df.dropna()
    print(f"[INFO] Loaded {dataset_name}: {len(df)} rows after dropping NaN.")
    return df


def get_feature_columns(dataset_name: str, model_type: str = "standard") -> list:
    """
    Return the appropriate feature list for a given dataset and model type.

    model_type options:
        'standard' — all features (tree / XGBoost / LSTM)
        'lr'       — features allowed for Logistic / Linear Regression
    """
    is_tbills = (dataset_name == "TBills")

    if model_type == "lr":
        return TBILLS_LR_FEATURES if is_tbills else LR_FEATURES
    else:
        return TBILLS_FEATURES if is_tbills else STANDARD_FEATURES


def select_features(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Drop rows where any feature or the target is missing, then return X and y.
    """
    cols_needed = feature_cols + [target_col]
    # Keep only columns that actually exist in the dataframe
    cols_needed = [c for c in cols_needed if c in df.columns]
    sub = df[cols_needed].dropna()

    # Filter feature_cols to only those present
    feature_cols = [c for c in feature_cols if c in sub.columns]

    X = sub[feature_cols].values
    y = sub[target_col].values
    return X, y, sub.index.tolist(), feature_cols


# =============================================================================
# 3. Train / test split (no shuffling)
# =============================================================================

def time_series_split(X, y, train_ratio: float = TRAIN_RATIO):
    """Chronological split — never shuffle."""
    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]


# =============================================================================
# 4. Scalers
# =============================================================================

def standard_scale(X_train, X_test):
    """Fit StandardScaler on training data; transform both sets."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


def minmax_scale(X_train, X_test):
    """Fit MinMaxScaler on training data; transform both sets."""
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# =============================================================================
# 5. LSTM sequence builder
# =============================================================================

def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = LSTM_SEQ_LEN):
    """
    Convert a 2-D feature array into overlapping sequences for LSTM input.

    Returns
    -------
    X_seq : shape (n_samples, seq_len, n_features)
    y_seq : shape (n_samples,)              — label/target for the last step
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len: i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


# =============================================================================
# 6. Classification metrics
# =============================================================================

def classification_metrics(y_true, y_pred, model_name: str, dataset_name: str) -> dict:
    """Compute and print classification metrics; return as dict."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"  [{dataset_name}] {model_name} — Classification Metrics")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    return {
        "model": model_name,
        "dataset": dataset_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
    }


# =============================================================================
# 7. Regression metrics
# =============================================================================

def regression_metrics(y_true, y_pred, model_name: str, dataset_name: str) -> dict:
    """Compute and print regression metrics; return as dict."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"  [{dataset_name}] {model_name} — Regression Metrics")
    print(f"{'='*60}")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  MAE  : {mae:.6f}")
    print(f"  R²   : {r2:.4f}")

    return {
        "model": model_name,
        "dataset": dataset_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


# =============================================================================
# 8. Saving predictions
# =============================================================================

def save_predictions_classification(
    dataset_name: str, model_name: str,
    y_true, y_pred, y_prob=None, indices=None
):
    """Save classification predictions (with optional probability columns) to CSV."""
    df = pd.DataFrame({"actual": y_true, "predicted": y_pred}, index=indices)
    if y_prob is not None:
        if y_prob.ndim == 1:
            df["prob_class1"] = y_prob
        else:
            for i in range(y_prob.shape[1]):
                df[f"prob_class{i}"] = y_prob[:, i]

    # Derive buy/sell signal: 1 → Buy, 0 → Hold, -1 → Sell (if 3-class)
    df["signal"] = df["predicted"].map(
        lambda x: "Buy" if x == 1 else ("Sell" if x == -1 else "Hold")
    )

    fname = f"{dataset_name}_{model_name}_classification_predictions.csv"
    path  = os.path.join(PREDICTIONS_DIR, fname)
    df.to_csv(path)
    print(f"[SAVED] Predictions → {path}")
    return df


def save_predictions_regression(
    dataset_name: str, model_name: str,
    y_true, y_pred, indices=None
):
    """Save regression predictions with actual vs predicted to CSV."""
    df = pd.DataFrame({"actual": y_true, "predicted": y_pred}, index=indices)
    df["error"] = df["actual"] - df["predicted"]

    fname = f"{dataset_name}_{model_name}_regression_predictions.csv"
    path  = os.path.join(PREDICTIONS_DIR, fname)
    df.to_csv(path)
    print(f"[SAVED] Predictions → {path}")
    return df


# =============================================================================
# 9. Saving metric tables
# =============================================================================

def save_metrics_table(records: list, task: str, dataset_name: str):
    """Append metric rows to a per-dataset CSV summary table."""
    df   = pd.DataFrame(records)
    fname = f"{dataset_name}_{task}_metrics.csv"
    path  = os.path.join(METRICS_DIR, fname)
    df.to_csv(path, index=False)
    print(f"[SAVED] Metrics table → {path}")


# =============================================================================
# 10. Plotting helpers
# =============================================================================

def plot_confusion_matrix(cm, class_names, model_name: str, dataset_name: str,
                          save_dir: str):
    """Plot and save a heatmap of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix\n{dataset_name} — {model_name}")
    plt.tight_layout()
    fname = f"{dataset_name}_{model_name}_confusion_matrix.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)


def plot_model_comparison_classification(metrics_list: list, dataset_name: str,
                                          save_dir: str):
    """Bar chart comparing accuracy / F1 across classification models."""
    df     = pd.DataFrame(metrics_list)
    models = df["model"].tolist()
    x      = np.arange(len(models))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, df["accuracy"], width, label="Accuracy", color="steelblue")
    ax.bar(x + width/2, df["f1_score"], width, label="F1-Score",  color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Classification Model Comparison — {dataset_name}")
    ax.legend()
    plt.tight_layout()
    fname = f"{dataset_name}_classification_model_comparison.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)


def plot_model_comparison_regression(metrics_list: list, dataset_name: str,
                                      save_dir: str):
    """Bar chart comparing RMSE / MAE / R² across regression models."""
    df     = pd.DataFrame(metrics_list)
    models = df["model"].tolist()
    x      = np.arange(len(models))
    width  = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, col, color, label in zip(
        axes,
        ["RMSE", "MAE", "R2"],
        ["steelblue", "darkorange", "seagreen"],
        ["RMSE", "MAE", "R²"],
    ):
        ax.bar(x, df[col], width * 2.5, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_title(label)
    fig.suptitle(f"Regression Model Comparison — {dataset_name}")
    plt.tight_layout()
    fname = f"{dataset_name}_regression_model_comparison.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted(y_true, y_pred, model_name: str, dataset_name: str,
                              save_dir: str):
    """Line plot of actual vs predicted values (regression)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true,  label="Actual",    linewidth=1.2, color="steelblue")
    ax.plot(y_pred,  label="Predicted", linewidth=1.0, color="darkorange", alpha=0.8)
    ax.set_title(f"Actual vs Predicted — {dataset_name} | {model_name}")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Return")
    ax.legend()
    plt.tight_layout()
    fname = f"{dataset_name}_{model_name}_actual_vs_predicted.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)


def plot_feature_importance(importances: np.ndarray, feature_names: list,
                             model_name: str, dataset_name: str, save_dir: str):
    """Horizontal bar chart of feature importances."""
    idx    = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.5)))
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="steelblue"
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {dataset_name} | {model_name}")
    plt.tight_layout()
    fname = f"{dataset_name}_{model_name}_feature_importance.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)

    # Also save importance values to CSV
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)
    csv_path = os.path.join(METRICS_DIR,
                            f"{dataset_name}_{model_name}_feature_importance.csv")
    imp_df.to_csv(csv_path, index=False)
    print(f"[SAVED] Feature importance → {csv_path}")


def plot_lstm_training_curves(history, model_name: str, dataset_name: str,
                               save_dir: str, task: str = "classification"):
    """Plot loss (and accuracy for classification) over training epochs."""
    fig, axes = plt.subplots(1, 2 if task == "classification" else 1,
                              figsize=(12 if task == "classification" else 6, 4))

    if task == "classification":
        ax_loss, ax_acc = axes
    else:
        ax_loss = axes if not isinstance(axes, np.ndarray) else axes[0]

    ax_loss.plot(history.history["loss"],     label="Train Loss", color="steelblue")
    ax_loss.plot(history.history["val_loss"], label="Val Loss",   color="darkorange")
    ax_loss.set_title("Loss Curve")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    if task == "classification" and "accuracy" in history.history:
        ax_acc.plot(history.history["accuracy"],     label="Train Acc", color="steelblue")
        ax_acc.plot(history.history["val_accuracy"], label="Val Acc",   color="darkorange")
        ax_acc.set_title("Accuracy Curve")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()

    fig.suptitle(f"LSTM Training Curves — {dataset_name} | {model_name}")
    plt.tight_layout()
    fname = f"{dataset_name}_{model_name}_training_curves.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)
