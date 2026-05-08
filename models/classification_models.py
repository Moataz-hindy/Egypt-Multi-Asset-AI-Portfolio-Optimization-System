# =============================================================================
# classification_models.py
# Classical classification models:
#   1. Logistic Regression
#   2. Decision Tree Classifier
#   3. XGBoost Classifier
# Target: label column
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import (
    CLASSIFICATION_TARGET, TRAIN_RATIO, RANDOM_STATE,
    CLASS_PLOTS_DIR,
)
from utils import (
    get_feature_columns, select_features, time_series_split,
    standard_scale, classification_metrics,
    save_predictions_classification, save_metrics_table,
    plot_confusion_matrix, plot_model_comparison_classification,
    plot_feature_importance,
)


# =============================================================================
# Helper — resolve integer class labels to human-readable names
# =============================================================================

def _label_names(y):
    unique = sorted(np.unique(y).astype(int).tolist())
    mapping = {-1: "Sell", 0: "Hold", 1: "Buy"}
    return [mapping.get(u, str(u)) for u in unique], unique


# =============================================================================
# 1. Logistic Regression
# =============================================================================

def run_logistic_regression(df: pd.DataFrame, dataset_name: str) -> dict:
    """Train and evaluate Logistic Regression on the given dataset."""
    print(f"\n[Logistic Regression] Dataset: {dataset_name}")

    # lr feature set (no bb_pb)
    feature_cols = get_feature_columns(dataset_name, model_type="lr")
    X, y, idx, used_features = select_features(df, feature_cols, CLASSIFICATION_TARGET)

    if len(X) < 50:
        print(f"  [SKIP] Not enough samples after preprocessing ({len(X)}).")
        return {}

    X_train, X_test, y_train, y_test = time_series_split(X, y, TRAIN_RATIO)
    X_train_s, X_test_s, _           = standard_scale(X_train, X_test)

    # Split indices for saving
    test_idx = idx[len(X_train):]

    # Encode labels to 0-based integers for sklearn (handles -1 / 0 / 1)
    classes      = sorted(np.unique(y).tolist())
    label_to_int = {c: i for i, c in enumerate(classes)}
    int_to_label = {i: c for c, i in label_to_int.items()}
    y_train_enc  = np.array([label_to_int[v] for v in y_train])
    y_test_enc   = np.array([label_to_int[v] for v in y_test])

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_train_s, y_train_enc)

    y_pred_enc  = model.predict(X_test_s)
    y_prob      = model.predict_proba(X_test_s)

    # Decode back to original labels
    y_pred = np.array([int_to_label[v] for v in y_pred_enc])
    y_true = np.array([int_to_label[v] for v in y_test_enc])

    metrics = classification_metrics(y_true, y_pred, "LogisticRegression", dataset_name)

    # Save predictions
    save_predictions_classification(
        dataset_name, "LogisticRegression", y_true, y_pred, y_prob, test_idx
    )

    # Confusion matrix plot
    class_names, _ = _label_names(classes)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plot_confusion_matrix(cm, class_names, "LogisticRegression", dataset_name, CLASS_PLOTS_DIR)

    return metrics


# =============================================================================
# 2. Decision Tree Classifier
# =============================================================================

def run_decision_tree(df: pd.DataFrame, dataset_name: str) -> dict:
    """Train and evaluate Decision Tree Classifier."""
    print(f"\n[Decision Tree] Dataset: {dataset_name}")

    feature_cols = get_feature_columns(dataset_name, model_type="standard")
    X, y, idx, used_features = select_features(df, feature_cols, CLASSIFICATION_TARGET)

    if len(X) < 50:
        print(f"  [SKIP] Not enough samples ({len(X)}).")
        return {}

    X_train, X_test, y_train, y_test = time_series_split(X, y, TRAIN_RATIO)
    X_train_s, X_test_s, _           = standard_scale(X_train, X_test)
    test_idx = idx[len(X_train):]

    model = DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)

    metrics = classification_metrics(y_test, y_pred, "DecisionTree", dataset_name)

    save_predictions_classification(
        dataset_name, "DecisionTree", y_test, y_pred, y_prob, test_idx
    )

    # Confusion matrix
    classes     = sorted(np.unique(y).tolist())
    class_names, _ = _label_names(classes)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plot_confusion_matrix(cm, class_names, "DecisionTree", dataset_name, CLASS_PLOTS_DIR)

    # Feature importance
    plot_feature_importance(
        model.feature_importances_, used_features,
        "DecisionTree", dataset_name, CLASS_PLOTS_DIR
    )

    return metrics


# =============================================================================
# 3. XGBoost Classifier
# =============================================================================

def run_xgboost_classifier(df: pd.DataFrame, dataset_name: str) -> dict:
    """Train and evaluate XGBoost Classifier."""
    print(f"\n[XGBoost Classifier] Dataset: {dataset_name}")

    feature_cols = get_feature_columns(dataset_name, model_type="standard")
    X, y, idx, used_features = select_features(df, feature_cols, CLASSIFICATION_TARGET)

    if len(X) < 50:
        print(f"  [SKIP] Not enough samples ({len(X)}).")
        return {}

    X_train, X_test, y_train, y_test = time_series_split(X, y, TRAIN_RATIO)
    X_train_s, X_test_s, _           = standard_scale(X_train, X_test)
    test_idx = idx[len(X_train):]

    # XGBoost requires 0-based integer labels
    classes      = sorted(np.unique(y).tolist())
    label_to_int = {c: i for i, c in enumerate(classes)}
    int_to_label = {i: c for c, i in label_to_int.items()}
    y_train_enc  = np.array([label_to_int[v] for v in y_train])
    y_test_enc   = np.array([label_to_int[v] for v in y_test])

    n_classes = len(classes)
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective=objective,
        num_class=n_classes if n_classes > 2 else None,
        use_label_encoder=False,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(
        X_train_s, y_train_enc,
        eval_set=[(X_test_s, y_test_enc)],
        verbose=False,
    )

    y_pred_enc = model.predict(X_test_s)
    y_prob     = model.predict_proba(X_test_s)

    y_pred = np.array([int_to_label[v] for v in y_pred_enc])
    y_true = np.array([int_to_label[v] for v in y_test_enc])

    metrics = classification_metrics(y_true, y_pred, "XGBoostClassifier", dataset_name)

    save_predictions_classification(
        dataset_name, "XGBoostClassifier", y_true, y_pred, y_prob, test_idx
    )

    # Confusion matrix
    class_names, _ = _label_names(classes)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plot_confusion_matrix(cm, class_names, "XGBoostClassifier", dataset_name, CLASS_PLOTS_DIR)

    # Feature importance
    plot_feature_importance(
        model.feature_importances_, used_features,
        "XGBoostClassifier", dataset_name, CLASS_PLOTS_DIR
    )

    return metrics


# =============================================================================
# Master runner — execute all classical classifiers on one dataset
# =============================================================================

def run_all_classifiers(df: pd.DataFrame, dataset_name: str):
    """Run all classical classification models and save a comparison table."""
    all_metrics = []

    for runner in [run_logistic_regression, run_decision_tree, run_xgboost_classifier]:
        result = runner(df, dataset_name)
        if result:
            all_metrics.append(result)

    if all_metrics:
        # Save summary CSV
        save_metrics_table(all_metrics, "classification", dataset_name)
        # Comparison plot
        plot_model_comparison_classification(all_metrics, dataset_name, CLASS_PLOTS_DIR)

    return all_metrics
