"""
model_utils.py
--------------
Helpers shared by train_models.py, evaluate.py and the Streamlit app:
data loading, the train/test split, the preprocessing pipeline, metrics
and feature-importance aggregation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402


def load_processed_data() -> pd.DataFrame:
    """Load the cleaned dataset created by data_preprocessing.py."""
    if not cfg.PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            "Cleaned data not found. Run `python src/data_preprocessing.py` first."
        )
    return pd.read_csv(cfg.PROCESSED_DATA_PATH)


def split_data(df: pd.DataFrame):
    """Single, reproducible 80/20 split (stratified on the classification label)."""
    X = df[cfg.MODEL_FEATURES]
    y_cls = df[cfg.CLASSIFICATION_TARGET]
    y_reg = df[cfg.REGRESSION_TARGET]
    return train_test_split(
        X, y_cls, y_reg,
        test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y_cls,
    )  # -> X_train, X_test, yc_train, yc_test, yr_train, yr_test


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features, pass binary flags through, one-hot encode categories."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cfg.NUMERIC_FEATURES),
            ("bin", "passthrough", cfg.BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cfg.CATEGORICAL_FEATURES),
        ]
    )


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def confusion(y_true, y_pred) -> list:
    return confusion_matrix(y_true, y_pred).tolist()


def aggregated_feature_importance(pipeline) -> dict:
    """
    Feature importance per ORIGINAL column (one-hot columns are summed back
    to their parent, e.g. all City_* columns -> City). Normalised to sum to 1.
    """
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = pre.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        raw = np.abs(np.ravel(model.coef_)).astype(float)
    else:
        return {}

    agg = {}
    for name, val in zip(names, raw):
        block, col = name.split("__", 1)
        if block == "cat":
            parent = next((c for c in cfg.CATEGORICAL_FEATURES if col.startswith(c + "_")), col)
        else:
            parent = col
        agg[parent] = agg.get(parent, 0.0) + float(val)

    total = sum(agg.values()) or 1.0
    return dict(sorted(((k, round(v / total, 5)) for k, v in agg.items()),
                       key=lambda kv: kv[1], reverse=True))
