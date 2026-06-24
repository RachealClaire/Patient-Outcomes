"""
test_model.py
-------------
Loads a saved .joblib regression model and evaluates it on a test set.
Call from your Jupyter notebook with:
    %run test_model.py
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from pathlib import Path

# ── 2. Configuration — edit these paths before running ───────────────────────
MODEL_PATH   = "/home/kyomukama/ACE/camonet/ML_Rerun/Outputs/ML_outputs/model_RF.joblib"
DATA   = data_clean.copy()
TARGET_COL   = "length_hosp_stay"
RESULTS_DIR  = "/home/kyomukama/ACE/camonet/ML_Rerun/Outputs/ml_test_results"

# ── 3. Create results directory ───────────────────────────────────────────────
results_path = Path(RESULTS_DIR)
results_path.mkdir(parents=True, exist_ok=True)
print(f"Results will be saved to: {results_path.resolve()}")

# ── 4. Load model ─────────────────────────────────────────────────────────────
print("\nLoading model …")
model = joblib.load(MODEL_PATH)
print(f"  Model type : {type(model).__name__}")

scaler = model.steps[0][1]
print("Features scaler was trained on:", scaler.n_features_in_)

if hasattr(scaler, "feature_names_in_"):
    print("Feature names:", list(scaler.feature_names_in_))

# ── 5. Load test data ─────────────────────────────────────────────────────────
print(f"\nLoading test data from '{DATA}' …")
df = DATA
print(f"  Shape      : {df.shape}")
print(f"  Columns    : {list(df.columns)}")

# Split features and target
if TARGET_COL not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COL}' not found in data. "
        f"Available columns: {list(df.columns)}"
    )

X_test = df.drop(columns=[TARGET_COL])
y_test = df[TARGET_COL]

# ── 6. Predict ────────────────────────────────────────────────────────────────
print("\nRunning predictions …")

# Model predictions are on log scale
y_pred_log = model.predict(X_test)

# Convert predictions back to original LOS scale
y_pred = np.expm1(y_pred_log)

# Actual values remain on original scale
y_actual = y_test.copy()

# Ensure no negative predictions
y_pred = np.maximum(y_pred, 0)

# ── 7. Evaluation metrics ─────────────────────────────────────────────────────
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

mape = np.mean(
    np.abs(
        (y_actual - y_pred) /
        np.where(y_actual != 0, y_actual, 1)
    )
) * 100

print("\n── Regression Metrics ──────────────────────────────")
print(f"  MAE   (Mean Absolute Error)          : {mae:.3f} days")
print(f"  RMSE  (Root Mean Squared Error)      : {rmse:.3f} days")
print(f"  R²    (Coefficient of Determination) : {r2:.4f}")
print(f"  MAPE  (Mean Abs. Percentage Error)   : {mape:.2f} %")
print("────────────────────────────────────────────────────\n")

# Save metrics
metrics_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", "MAPE (%)"],
    "Value": [round(mae, 4), round(rmse, 4), round(r2, 4), round(mape, 4)]
})

metrics_path = results_path / "metrics_summary.csv"
metrics_df.to_csv(metrics_path, index=False)

print(f"  Metrics saved → {metrics_path}")


# ── 7. Evaluation metrics ─────────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100

print("\n── Regression Metrics ──────────────────────────────")
print(f"  MAE   (Mean Absolute Error)          : {mae:.3f} days")
print(f"  RMSE  (Root Mean Squared Error)      : {rmse:.3f} days")
print(f"  R²    (Coefficient of Determination) : {r2:.4f}")
print(f"  MAPE  (Mean Abs. Percentage Error)   : {mape:.2f} %")
print("────────────────────────────────────────────────────\n")

# Save metrics to CSV
metrics_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", "MAPE (%)"],
    "Value":  [round(mae, 4), round(rmse, 4), round(r2, 4), round(mape, 4)],
})
metrics_path = results_path / "metrics_summary.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"  Metrics saved → {metrics_path}")

# ── 8. Feature importances (if the model supports it) ─────────────────────────

rf_model = None

if hasattr(model, "feature_importances_"):
    rf_model = model

elif hasattr(model, "named_steps"):
    for name, step in model.named_steps.items():
        if hasattr(step, "feature_importances_"):
            rf_model = step
            break

if rf_model is not None:

    importances = pd.Series(
        rf_model.feature_importances_,
        index=X_test.columns
    )

    top10 = importances.sort_values(
        ascending=False
    ).head(10)

    print("\nTop 10 Most Important Features")
    print(top10)

    feat_path = results_path / "feature_importances.csv"

    importances.sort_values(
        ascending=False
    ).to_csv(
        feat_path,
        header=["importance"]
    )

    print(f"Feature importances saved → {feat_path}")

if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    top10 = importances.sort_values(ascending=False).head(10)
    print("Top 10 feature importances:")
    print(top10.to_string())
    print()
    feat_path = results_path / "feature_importances.csv"
    importances.sort_values(ascending=False).to_csv(feat_path, header=["importance"])
    print(f"  Feature importances saved → {feat_path}")

elif hasattr(model, "coef_"):
    coef = pd.Series(np.abs(model.coef_), index=X_test.columns)
    top10 = coef.sort_values(ascending=False).head(10)
    print("Top 10 feature coefficients (absolute):")
    print(top10.to_string())
    print()
    coef_path = results_path / "feature_coefficients.csv"
    coef.sort_values(ascending=False).to_csv(coef_path, header=["abs_coefficient"])
    print(f"  Coefficients saved → {coef_path}")

# ── 9. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Actual vs Predicted
axes[0].scatter(
    y_actual,
    y_pred,
    alpha=0.6
)

lims = [
    min(y_actual.min(), y_pred.min()),
    max(y_actual.max(), y_pred.max())
]

axes[0].plot(
    lims,
    lims,
    "r--",
    linewidth=2,
    label="Perfect Prediction"
)

axes[0].set_xlabel("Actual Length of Stay (days)")
axes[0].set_ylabel("Predicted Length of Stay (days)")
axes[0].set_title(f"Actual vs Predicted (R² = {r2:.3f})")
axes[0].legend()

# Residuals
residuals = y_actual - y_pred

axes[1].hist(
    residuals,
    bins=30
)

axes[1].axvline(
    0,
    linestyle="--",
    linewidth=2
)

axes[1].set_xlabel("Residual (Actual − Predicted)")
axes[1].set_ylabel("Frequency")
axes[1].set_title(f"Residual Distribution (MAE = {mae:.2f})")

plt.tight_layout()

plot_path = results_path / "model_evaluation_plots.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nPlots saved → {plot_path}")

# 9a. Actual vs Predicted scatter
axes[0].scatter(y_test, y_pred, alpha=0.5, edgecolors="none", color="steelblue")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
axes[0].set_xlabel("Actual Length of Stay (days)")
axes[0].set_ylabel("Predicted Length of Stay (days)")
axes[0].set_title(f"Actual vs Predicted  |  R² = {r2:.3f}")
axes[0].legend()

# 9b. Residuals distribution
residuals = y_test - y_pred
axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Residual (Actual − Predicted)")
axes[1].set_ylabel("Frequency")
axes[1].set_title(f"Residuals Distribution  |  MAE = {mae:.3f}")

plt.tight_layout()
plot_path = results_path / "model_evaluation_plots.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  Plots saved → {plot_path}")

# ── 10. Full Predictions Table ────────────────────────────────────────────────
results_df = pd.DataFrame({
    "Actual": y_actual.values,
    "Predicted": np.round(y_pred, 2),
    "Residual": np.round(y_actual.values - y_pred, 2)
})

preds_path = results_path / "predictions.csv"

results_df.to_csv(
    preds_path,
    index=False
)

print(f"Predictions saved → {preds_path}")

print("\nSample predictions:")
print(results_df.head(10).to_string(index=False))

print(f"\n✓ All results saved to: {results_path.resolve()}")