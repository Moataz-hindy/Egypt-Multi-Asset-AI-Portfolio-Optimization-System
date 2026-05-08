# =============================================================================
# regression_models.py
# Classical regression models:
#   1. Linear Regression
#   2. Random Forest Regressor
#   3. SVR (Support Vector Regressor)
# Target: return column
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from config import (
    REGRESSION_TARGET, TRAIN_RATIO, RANDOM_STATE,
    REG_PLOTS_DIR,
)
from utils import (
    get_feature_columns, select_features, time_series_split,
    standard_scale, regression_metrics,
    save_predictions_regression, save_metrics_table,
    plot_actual_vs_predicted, plot_model_comparison_regression,
    plot_feature_importance,
)


# =============================================================================
# 1. Linear Regression
# =============================================================================

def run_linear_regression(df: pd.DataFrame, dataset_name: str) -> dict:
    """Train and evaluate Linear Regression on the given dataset."""
    print(f"\n[Linear Regression] Dataset: {dataset_name}")

    # lr feature set (no bb_pb)
    feature_cols = get_feature_columns(dataset_name, model_type="lr")
    X, y, idx, used_features = select_features(df, feature_cols, REGRESSION_TARGET)

    if len(X) < 50:
        print(f"  [SKIP] Not enough samples ({len(X)}).")
        return {}

    X_train, X_test, y_train, y_test = time_series_split(X, y, TRAIN_RATIO)
    X_train_s, X_test_s, _           = standard_scale(X_train, X_test)
    test_idx = idx[len(X_train):]

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = regression_metrics(y_test, y_pred, "LinearRegression", dataset_name)

    save_predictions_regression(
        dataset_name, "LinearRegression", y_test, y_pred, test_idx
    )
    plot_actual_vs_predicted(
        y_test, y_pred, "LinearRegression", dataset_name, REG_PLOTS_DIR
    )

    # Coefficient magnitudes as "importance"
    importances = np.abs(model.coef_)
    plot_feature_importance(
        importances, used_features, "LinearRegression", dataset_name, REG_PLOTS_DIR
    )

    return metrics


# =============================================================================
# 2. Random Forest Regressor
# =============================================================================

def run_random_forest_regressor(df: pd.DataFrame, dataset_name: str) -> dict:
    """Train and evaluate Random Forest Regressor."""
    print(f"\n[Random Forest Regressor] Dataset: {dataset_name}")

    feature_cols = get_feature_columns(dataset_name, model_type="standard")
    X, y, idx, used_features = select_features(df, feature_cols, REGRESSION_TARGET)

    if len(X) < 50:
        print(f"  [SKIP] Not enough samples ({len(X)}).")
        return {}

    X_train, X_test, y_train, y_test = time_series_split(X, y, TRAIN_RATIO)
    X_train_s, X_test_s, _           = standard_scale(X_train, X_test)
    test_idx = idx[len(X_train):]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = regression_metrics(y_test, y_pred, "RandomForest", dataset_name)

    save_predictions_regression(
        dataset_name, "RandomForest", y_test, y_pred, test_idx
    )
    plot_actual_vs_predicted(
        y_test, y_pred, "RandomForest", dataset_name, REG_PLOTS_DIR
    )
    plot_feature_importance(
        model.feature_importances_, used_features,
        "RandomForest", dataset_name, REG_PLOTS_DIR
    )

    return metrics


# =============================================================================
# 3. SVR (Support Vector Regressor)
# =============================================================================

def run_svr(df: pd.DataFrame, dataset_name: str) -> dict:
    """Train and evaluate Support Vector Regressor."""
    print(f"\n[SVR] Dataset: {dataset_name}")

    feature_cols = get_feature_columns(dataset_name, model_type="standard")
    X, y, idx, used_features = select_features(df, feature_cols, REGRESSION_TARGET)

    if len(X) < 50:
        print(f"  [SKIP] Not enough samples ({len(X)}).")
        return {}

    X_train, X_test, y_train, y_test = time_series_split(X, y, TRAIN_RATIO)
    X_train_s, X_test_s, _           = standard_scale(X_train, X_test)
    test_idx = idx[len(X_train):]

    model = SVR(kernel="rbf", C=1.0, epsilon=0.01, gamma="scale")
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = regression_metrics(y_test, y_pred, "SVR", dataset_name)

    save_predictions_regression(
        dataset_name, "SVR", y_test, y_pred, test_idx
    )
    plot_actual_vs_predicted(
        y_test, y_pred, "SVR", dataset_name, REG_PLOTS_DIR
    )

    return metrics


# =============================================================================
# Master runner — execute all classical regressors on one dataset
# =============================================================================

def run_all_regressors(df: pd.DataFrame, dataset_name: str):
    """Run all classical regression models and save a comparison table."""
    all_metrics = []

    for runner in [run_linear_regression, run_random_forest_regressor, run_svr]:
        result = runner(df, dataset_name)
        if result:
            all_metrics.append(result)

    if all_metrics:
        save_metrics_table(all_metrics, "regression", dataset_name)
        plot_model_comparison_regression(all_metrics, dataset_name, REG_PLOTS_DIR)

    return all_metrics
