# =============================================================================
# config.py — Central configuration for the AI Investment Recommendation System
# =============================================================================

import os
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Dataset paths (place CSV files in the data/ folder)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(MODELS_DIR), "final_data")

DATASETS = {
    "EGX100":     os.path.join(DATA_DIR, "final_EGX100.csv"),
    "EGX30":      os.path.join(DATA_DIR, "final_EGX30.csv"),
    "Gold":       os.path.join(DATA_DIR, "final_gold.csv"),
    "PalmHills":  os.path.join(DATA_DIR, "final_palm_hills.csv"),
    "SODIC":      os.path.join(DATA_DIR, "final_sodic.csv"),
    "TBills":     os.path.join(DATA_DIR, "final_tbills.csv"),
    "TMG":        os.path.join(DATA_DIR, "final_tmg.csv"),
}

# ---------------------------------------------------------------------------
# T-Bills has a completely different feature set
# ---------------------------------------------------------------------------
TBILLS_FEATURES = ["return_lag1", "return_lag2", "return_lag3", "annual_yield"]

# ---------------------------------------------------------------------------
# Standard features shared by all other assets
# ---------------------------------------------------------------------------
STANDARD_FEATURES = [
    "return_lag1",
    "return_lag2",
    "dist_to_ma5",
    "macd_hist",
    "bb_pb",
    "bb_bandwidth",
    "rsi",
    "rolling_volatility",
]

# Features used by Logistic Regression and Linear Regression
# (bb_pb is dropped; rsi is kept)
LR_FEATURES = [
    "return_lag1",
    "return_lag2",
    "dist_to_ma5",
    "macd_hist",
    "bb_bandwidth",
    "rsi",
    "rolling_volatility",
]

# LR features for T-Bills (subset that exists)
TBILLS_LR_FEATURES = ["return_lag1", "return_lag2", "return_lag3", "annual_yield"]

# ---------------------------------------------------------------------------
# Target columns
# ---------------------------------------------------------------------------
CLASSIFICATION_TARGET = "label"
REGRESSION_TARGET     = "return"   # predict the raw return

# ---------------------------------------------------------------------------
# Train / test split  (no shuffling — time-series order is preserved)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.80   # 80 % train, 20 % test

# ---------------------------------------------------------------------------
# LSTM hyper-parameters
# ---------------------------------------------------------------------------
LSTM_SEQ_LEN   = 30
LSTM_EPOCHS    = 50
LSTM_BATCH     = 32
LSTM_UNITS     = 64        # units in the first LSTM layer
LSTM_DROPOUT   = 0.2
LSTM_LR        = 1e-3      # Adam learning rate

# ---------------------------------------------------------------------------
# Output directory structure
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(MODELS_DIR), "outputs")
PREDICTIONS_DIR       = os.path.join(OUTPUT_DIR, "predictions")
METRICS_DIR           = os.path.join(OUTPUT_DIR, "metrics")
PLOTS_DIR             = os.path.join(OUTPUT_DIR, "plots")

# Sub-folders per task
CLASS_PLOTS_DIR       = os.path.join(PLOTS_DIR, "classification")
REG_PLOTS_DIR         = os.path.join(PLOTS_DIR, "regression")
LSTM_CLASS_PLOTS_DIR  = os.path.join(PLOTS_DIR, "lstm_classification")
LSTM_REG_PLOTS_DIR    = os.path.join(PLOTS_DIR, "lstm_regression")

ALL_OUTPUT_DIRS = [
    PREDICTIONS_DIR,
    METRICS_DIR,
    CLASS_PLOTS_DIR,
    REG_PLOTS_DIR,
    LSTM_CLASS_PLOTS_DIR,
    LSTM_REG_PLOTS_DIR,
]

# ---------------------------------------------------------------------------
# Random seed (used where applicable — trees, weight init, etc.)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
