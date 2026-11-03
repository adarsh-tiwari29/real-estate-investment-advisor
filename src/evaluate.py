"""
evaluate.py
-----------
Step 3 of the pipeline: independently evaluate the two saved champion models
on the held-out 20 % test set and write a human-readable report.

    python src/evaluate.py

Outputs (in reports/)
    evaluation_report.txt           -> metrics + classification report
    figures/champion_confusion_matrix.png
    figures/champion_actual_vs_predicted.png
    figures/champion_residuals.png
    figures/champion_feature_importance.png
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report

sys.path.append(str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402
import model_utils as mu  # noqa: E402


def main() -> None:
    for p in (cfg.CLASSIFIER_PATH, cfg.REGRESSOR_PATH):
        if not p.exists():
            raise FileNotFoundError(f"{p.name} missing. Run `python src/train_models.py` first.")

    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    clf = joblib.load(cfg.CLASSIFIER_PATH)
    reg = joblib.load(cfg.REGRESSOR_PATH)
    meta = json.loads(cfg.MODEL_METADATA_PATH.read_text())

    df = mu.load_processed_data()
    _, X_test, _, yc_test, _, yr_test = mu.split_data(df)

    # ---------------- Classification -----------------------------------------
    yc_pred = clf.predict(X_test)
    yc_proba = clf.predict_proba(X_test)[:, 1]
    cls_m = mu.classification_metrics(yc_test, yc_pred, yc_proba)
    cm = np.array(mu.confusion(yc_test, yc_pred))

    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Not Good", "Good"], yticklabels=["Not Good", "Good"])
    ax.set(title=f"Confusion Matrix - {meta['classifier']['model']}",
           xlabel="Predicted", ylabel="Actual")
    fig.tight_layout(); fig.savefig(cfg.FIGURES_DIR / "champion_confusion_matrix.png", dpi=120)
    plt.close(fig)

    # ---------------- Regression ---------------------------------------------
    yr_pred = reg.predict(X_test)
    reg_m = mu.regression_metrics(yr_test, yr_pred)
    resid = yr_test.to_numpy() - yr_pred
    mape = float(np.mean(np.abs(resid) / yr_test.to_numpy()) * 100)

    sample = np.random.default_rng(0).choice(len(yr_test), 6000, replace=False)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.scatter(yr_test.to_numpy()[sample], yr_pred[sample], s=4, alpha=0.4)
    lims = [yr_test.min(), yr_test.max()]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set(title=f"Actual vs Predicted - {meta['regressor']['model']}",
           xlabel="Actual 5Y price (Lakhs)", ylabel="Predicted 5Y price (Lakhs)")
    fig.tight_layout(); fig.savefig(cfg.FIGURES_DIR / "champion_actual_vs_predicted.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.histplot(resid, bins=60, kde=True, ax=ax)
    ax.set(title="Residual distribution", xlabel="Actual - Predicted (Lakhs)")
    fig.tight_layout(); fig.savefig(cfg.FIGURES_DIR / "champion_residuals.png", dpi=120)
    plt.close(fig)

    # ---------------- Feature importance ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in [(axes[0], "classifier", "Good Investment"),
                           (axes[1], "regressor", "Future Price")]:
        fi = pd.Series(meta[key]["feature_importance"]).head(12)[::-1]
        ax.barh(fi.index, fi.values, color="#2a6f97")
        ax.set_title(f"Top features - {title}")
    fig.tight_layout(); fig.savefig(cfg.FIGURES_DIR / "champion_feature_importance.png", dpi=120)
    plt.close(fig)

    # ---------------- Report -------------------------------------------------
    report = [
        "REAL ESTATE INVESTMENT ADVISOR - MODEL EVALUATION",
        "=" * 52,
        f"Test rows: {len(X_test):,}",
        "",
        f"CLASSIFICATION  (champion: {meta['classifier']['model']})",
        *(f"  {k:<10}: {v}" for k, v in cls_m.items()),
        "  Confusion matrix [[TN, FP], [FN, TP]]:",
        f"  {cm.tolist()}",
        "",
        classification_report(yc_test, yc_pred, target_names=["Not Good", "Good"]),
        f"REGRESSION  (champion: {meta['regressor']['model']})",
        *(f"  {k:<10}: {v}" for k, v in reg_m.items()),
        f"  MAPE      : {mape:.2f} %",
        "",
        "All candidate models (from training run):",
        json.dumps(json.loads(cfg.METRICS_PATH.read_text()), indent=2),
    ]
    out = cfg.REPORTS_DIR / "evaluation_report.txt"
    out.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[:22]))
    print(f"\nReport saved -> {out.relative_to(cfg.ROOT_DIR)}")


if __name__ == "__main__":
    main()
