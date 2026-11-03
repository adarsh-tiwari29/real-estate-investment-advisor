"""
train_models.py
---------------
Step 2 of the pipeline: train, compare and register the two models.

    python src/train_models.py

* Classification -> "Is this a Good Investment?"   (Logistic Regression,
                                                     Random Forest, XGBoost)
* Regression     -> "Estimated price after 5 years" (Linear Regression,
                                                     Random Forest, XGBoost)

Every candidate is logged to MLflow (params, metrics, plots, model). The best
model of each task is registered in the MLflow Model Registry with the alias
"champion" and also saved to models/ for the Streamlit app.

View the experiments afterwards with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no GUI needed while training
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import seaborn as sns
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

sys.path.append(str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402
import model_utils as mu  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("training")
logging.getLogger("mlflow").setLevel(logging.WARNING)


# ----------------------------------------------------------------------------
# Candidate models
# ----------------------------------------------------------------------------
def classification_candidates() -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100, max_depth=14, min_samples_leaf=20,
            n_jobs=-1, random_state=cfg.RANDOM_STATE),
        "XGBClassifier": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", n_jobs=-1,
            random_state=cfg.RANDOM_STATE),
    }


def regression_candidates() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100, max_depth=14, min_samples_leaf=20,
            n_jobs=-1, random_state=cfg.RANDOM_STATE),
        "XGBRegressor": XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, n_jobs=-1, random_state=cfg.RANDOM_STATE),
    }


# ----------------------------------------------------------------------------
# MLflow helpers
# ----------------------------------------------------------------------------
def setup_mlflow() -> None:
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(cfg.MLFLOW_EXPERIMENT) is None:
        cfg.MLFLOW_ARTIFACT_DIR.mkdir(exist_ok=True)
        mlflow.create_experiment(cfg.MLFLOW_EXPERIMENT,
                                 artifact_location=cfg.MLFLOW_ARTIFACT_DIR.as_uri())
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT)


def log_sklearn_model(pipeline, X_sample, preds) -> None:
    """Log the model; works on both MLflow 2.x (artifact_path) and 3.x (name)."""
    signature = infer_signature(X_sample, preds)
    kwargs = dict(signature=signature, input_example=X_sample.head(3),
                  serialization_format="cloudpickle")  # we trust our own models
    try:
        mlflow.sklearn.log_model(pipeline, name="model", **kwargs)          # MLflow 3.x
    except TypeError:
        mlflow.sklearn.log_model(pipeline, artifact_path="model", **kwargs)  # MLflow 2.x


def register_champion(run_id: str, model_name: str) -> str:
    """Register the best run's model and point the 'champion' alias at it."""
    result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client = MlflowClient()
    try:
        client.set_registered_model_alias(model_name, "champion", result.version)
    except Exception:  # very old MLflow without aliases -> ignore
        pass
    return str(result.version)


def save_confusion_plot(cm, title, path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Not Good", "Good"], yticklabels=["Not Good", "Good"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def save_pred_plot(y_true, y_pred, title, path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    idx = y_true.sample(min(5000, len(y_true)), random_state=1).index
    ax.scatter(y_true.loc[idx], y_pred[y_true.index.get_indexer(idx)], s=4, alpha=0.4)
    lims = [y_true.min(), y_true.max()]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Actual (Lakhs)"); ax.set_ylabel("Predicted (Lakhs)"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------------
# Training loops
# ----------------------------------------------------------------------------
def train_task(task, candidates, X_train, X_test, y_train, y_test):
    """Train every candidate for one task, log to MLflow, return results."""
    results = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", mu.build_preprocessor()), ("model", estimator)])
        with mlflow.start_run(run_name=f"{task}_{name}") as run:
            t0 = time.time()
            pipe.fit(X_train, y_train)
            train_time = time.time() - t0
            preds = pipe.predict(X_test)

            plot_path = cfg.FIGURES_DIR / f"{task}_{name}.png"
            if task == "classification":
                proba = pipe.predict_proba(X_test)[:, 1]
                metrics = mu.classification_metrics(y_test, preds, proba)
                cm = mu.confusion(y_test, preds)
                save_confusion_plot(cm, name, plot_path)
            else:
                metrics = mu.regression_metrics(y_test, preds)
                cm = None
                save_pred_plot(y_test, preds, name, plot_path)

            mlflow.set_tags({"task": task, "model_family": name})
            mlflow.log_param("model", name)
            mlflow.log_params({k: v for k, v in estimator.get_params().items()
                               if isinstance(v, (int, float, str, bool)) and v is not None})
            mlflow.log_param("n_train_rows", len(X_train))
            mlflow.log_param("n_features_raw", X_train.shape[1])
            mlflow.log_metrics({**metrics, "train_time_sec": round(train_time, 2)})
            mlflow.log_artifact(str(plot_path), artifact_path="plots")
            log_sklearn_model(pipe, X_train, pipe.predict(X_train.head(50)))

            results[name] = {"pipeline": pipe, "metrics": metrics, "run_id": run.info.run_id,
                             "confusion_matrix": cm, "train_time_sec": round(train_time, 2)}
            log.info("%-15s %-24s %s  (%.1fs)", task, name, metrics, train_time)
    return results


def main() -> None:
    cfg.MODELS_DIR.mkdir(exist_ok=True)
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = mu.load_processed_data()
    X_train, X_test, yc_train, yc_test, yr_train, yr_test = mu.split_data(df)
    log.info("Train: %s rows | Test: %s rows", len(X_train), len(X_test))

    setup_mlflow()

    cls_res = train_task("classification", classification_candidates(),
                         X_train, X_test, yc_train, yc_test)
    reg_res = train_task("regression", regression_candidates(),
                         X_train, X_test, yr_train, yr_test)

    # Pick champions: best F1 for classification, lowest RMSE for regression
    best_cls = max(cls_res, key=lambda k: cls_res[k]["metrics"]["f1_score"])
    best_reg = min(reg_res, key=lambda k: reg_res[k]["metrics"]["rmse"])
    cls_version = register_champion(cls_res[best_cls]["run_id"], cfg.REGISTERED_CLASSIFIER)
    reg_version = register_champion(reg_res[best_reg]["run_id"], cfg.REGISTERED_REGRESSOR)
    log.info("Champion classifier: %s (registry v%s)", best_cls, cls_version)
    log.info("Champion regressor : %s (registry v%s)", best_reg, reg_version)

    # Save champions for the Streamlit app
    joblib.dump(cls_res[best_cls]["pipeline"], cfg.CLASSIFIER_PATH, compress=3)
    joblib.dump(reg_res[best_reg]["pipeline"], cfg.REGRESSOR_PATH, compress=3)

    metadata = {
        "classifier": {
            "model": best_cls, "registry_name": cfg.REGISTERED_CLASSIFIER,
            "registry_version": cls_version, "metrics": cls_res[best_cls]["metrics"],
            "confusion_matrix": cls_res[best_cls]["confusion_matrix"],
            "feature_importance": mu.aggregated_feature_importance(cls_res[best_cls]["pipeline"]),
        },
        "regressor": {
            "model": best_reg, "registry_name": cfg.REGISTERED_REGRESSOR,
            "registry_version": reg_version, "metrics": reg_res[best_reg]["metrics"],
            "feature_importance": mu.aggregated_feature_importance(reg_res[best_reg]["pipeline"]),
        },
        "features": cfg.MODEL_FEATURES,
        "train_rows": len(X_train), "test_rows": len(X_test),
    }
    with open(cfg.MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    all_metrics = {
        "classification": {k: {**v["metrics"], "train_time_sec": v["train_time_sec"]}
                           for k, v in cls_res.items()},
        "regression": {k: {**v["metrics"], "train_time_sec": v["train_time_sec"]}
                       for k, v in reg_res.items()},
    }
    with open(cfg.METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    log.info("Models saved to %s", cfg.MODELS_DIR.relative_to(cfg.ROOT_DIR))
    log.info("Open MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
