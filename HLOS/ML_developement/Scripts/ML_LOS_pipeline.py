import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, GridSearchCV,
    RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import shap

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

OUTPUT_DIR = "/home/kyomukama/ACE/Patient outcomes/HLOS/ML_developement/Outputs/ML_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(filename):
    """Helper: returns full path inside Outputs/"""
    return os.path.join(OUTPUT_DIR, filename)

# =============================================================================
# PLOT STYLE
# =============================================================================

plt.rcParams.update({
    "figure.dpi":        150,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "font.family":       "sans-serif",
})

PALETTE = {
    "RF":          "#4C72B0",
    "XGB":         "#DD8452",
    "LGBM":        "#55A868",
    "GB":          "#8172B2",
    "ElasticNet":  "#937860",
    "SVR":         "#64B5CD",
    "CatBoost":    "#CCB974",
    "Stacked":     "#C44E52",
}

# =============================================================================
# CONFIG
# =============================================================================

RANDOM_SEED    = 1
TEST_SIZE      = 0.2
CV_SPLITS      = 5
CV_REPEATS     = 3
BOOTSTRAP_ITER = 1000

np.random.seed(RANDOM_SEED)

# =============================================================================
# DATA
# =============================================================================

df = ml_data.copy()
TARGET = "Length of hospital stay (days)"

df = df[df[TARGET] > 0].reset_index(drop=True)

p99 = df[TARGET].quantile(0.99)
df[TARGET] = df[TARGET].clip(upper=p99)

X = df.drop(columns=[TARGET])
y = df[TARGET]

feature_names = X.columns.tolist()

for col in X.columns:
    if X[col].isna().mean() > 0.05:
        X[f"{col}_missing"] = X[col].isna().astype(int)
        feature_names = X.columns.tolist()

y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y_log, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

y_train_orig = np.expm1(y_train)
y_test_orig  = np.expm1(y_test)

los_bins_train = pd.qcut(y_train, q=5, labels=False, duplicates="drop")

cv_tuning = RepeatedStratifiedKFold(
    n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_SEED
)
cv_splits_tuning = list(cv_tuning.split(X_train, los_bins_train))

cv_oof = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
cv_splits_oof = list(cv_oof.split(X_train, los_bins_train))

# =============================================================================
# MODELS
# =============================================================================
# Tree/boosting models go in a plain Pipeline (no scaling needed).
# Scale-sensitive models (ElasticNet, SVR) get a StandardScaler step first.
# `needs_scaling` flags which models require scaling — used later for SHAP
# explainer selection too.

needs_scaling = {
    "RF": False, "XGB": False, "LGBM": False, "GB": False, "CatBoost": False,
    "ElasticNet": True, "SVR": True,
}

models = {
    "RF": {
        "model": RandomForestRegressor(random_state=RANDOM_SEED),
        "params": {
            "model__n_estimators":     [300, 600],
            "model__max_depth":        [10, 20, None],
            "model__min_samples_leaf": [2, 4],
        },
    },
    "XGB": {
        "model": XGBRegressor(
            random_state=RANDOM_SEED, eval_metric="mae",
            tree_method="hist", verbosity=0
        ),
        "params": {
            "model__n_estimators":     [500, 1000],
            "model__max_depth":        [3, 5],
            "model__learning_rate":    [0.03, 0.05],
            "model__subsample":        [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__min_child_weight": [5, 10],
        },
    },
    "LGBM": {
        "model": LGBMRegressor(random_state=RANDOM_SEED, verbose=-1),
        "params": {
            "model__n_estimators":  [400, 800],
            "model__learning_rate": [0.01, 0.05],
            "model__num_leaves":    [31, 63],
        },
    },
    "GB": {
        "model": GradientBoostingRegressor(random_state=RANDOM_SEED),
        "params": {
            "model__n_estimators":     [300, 500],
            "model__max_depth":        [2, 3],
            "model__learning_rate":    [0.03, 0.05],
            "model__subsample":        [0.8, 1.0],
        },
    },
    "ElasticNet": {
        "model": ElasticNet(random_state=RANDOM_SEED, max_iter=10000),
        "params": {
            "model__alpha":    [0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.2, 0.5, 0.8],
        },
    },
    "SVR": {
        "model": SVR(),
        "params": {
            "model__C":       [0.5, 1.0, 5.0],
            "model__epsilon": [0.05, 0.1, 0.2],
            "model__kernel":  ["rbf"],
        },
    },
    "CatBoost": {
        "model": CatBoostRegressor(
            random_state=RANDOM_SEED, verbose=0, loss_function="MAE"
        ),
        "params": {
            "model__iterations":     [400, 800],
            "model__depth":          [4, 6],
            "model__learning_rate":  [0.03, 0.05],
        },
    },
}

# =============================================================================
# METRICS
# =============================================================================

def evaluate(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "R2":   r2_score(y_true, y_pred),
    }

# =============================================================================
# TRAIN + TUNE
# =============================================================================

best_models = {}

for name, config in models.items():
    steps = []
    if needs_scaling.get(name, False):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", config["model"]))
    pipe = Pipeline(steps)

    grid = GridSearchCV(
        pipe, config["params"],
        cv=cv_splits_tuning,
        scoring="neg_mean_absolute_error",
        n_jobs=-1, verbose=0, error_score="raise",
    )
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f"{name} best params: {grid.best_params_}")

# =============================================================================
# SAVE BASE MODELS
# =============================================================================

for name, model in best_models.items():
    path = out(f"model_{name}.joblib")
    joblib.dump(model, path)
    print(f"Saved base model: {path}")

# =============================================================================
# STACKING
# =============================================================================

oof_preds = []
for name, model in best_models.items():
    preds = cross_val_predict(model, X_train, y_train, cv=cv_splits_oof)
    oof_preds.append(preds)

meta_X_train = np.column_stack(oof_preds)
meta_X_test  = np.column_stack([m.predict(X_test) for m in best_models.values()])

meta_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge",  RidgeCV(alphas=[0.1, 1.0, 10.0])),
])
meta_model.fit(meta_X_train, y_train)
stacked_pred = np.expm1(meta_model.predict(meta_X_test))

# Save stacked (meta) model
joblib.dump(meta_model, out("model_Stacked_meta.joblib"))
print(f"Saved stacked meta-model: {out('model_Stacked_meta.joblib')}")

# Save stacking config (base model names, for reconstruction)
stacking_config = {
    "base_model_names": list(best_models.keys()),
    "feature_names":    feature_names,
    "random_seed":      RANDOM_SEED,
}
joblib.dump(stacking_config, out("stacking_config.joblib"))
print(f"Saved stacking config: {out('stacking_config.joblib')}")

# =============================================================================
# EVALUATION
# =============================================================================

results = {}
all_preds = {}

for name, model in best_models.items():
    pred = np.expm1(model.predict(X_test))
    all_preds[name] = pred
    results[name]   = evaluate(y_test_orig, pred)

all_preds["Stacked"] = stacked_pred
results["Stacked"]   = evaluate(y_test_orig, stacked_pred)

results_df = pd.DataFrame(results).T.round(4)
print("\nMODEL PERFORMANCE")
print(results_df)

# Save results table
results_df.to_csv(out("model_performance_metrics.csv"))
print(f"Saved: {out('model_performance_metrics.csv')}")

# =============================================================================
# SAVE BEST MODEL (lowest MAE across base + stacked)
# =============================================================================

best_overall_name = min(
    (n for n in results if n != "Stacked"),   # Excluding stacked model
    key=lambda n: results[n]["MAE"]
)

# Determine the overall best model by MAE
best_overall_mae  = results[best_overall_name]["MAE"]
best_overall_r2   = results[best_overall_name]["R2"]

best_overall_obj = best_models[best_overall_name]

# Save the best model
joblib.dump(best_overall_obj, out("best_model.joblib"))

# Save a human-readable summary of the best model
best_model_info = {
    "name":          best_overall_name,
    "MAE":           best_overall_mae,
    "RMSE":          results[best_overall_name]["RMSE"],
    "R2":            best_overall_r2,
    "feature_names": feature_names,
    "random_seed":   RANDOM_SEED
}
joblib.dump(best_model_info, out("best_model_info.joblib"))

# Also write a plain-text card for quick inspection
with open(out("best_model_summary.txt"), "w") as f:
    f.write("=" * 50 + "\n")
    f.write("BEST MODEL SUMMARY\n")
    f.write("=" * 50 + "\n")
    f.write(f"  Model        : {best_overall_name}\n")
    f.write(f"  MAE (days)   : {best_overall_mae:.4f}\n")
    f.write(f"  RMSE (days)  : {results[best_overall_name]['RMSE']:.4f}\n")
    f.write(f"  R²           : {best_overall_r2:.4f}\n")
    f.write(f"  Features     : {len(feature_names)}\n")
    f.write("-" * 50 + "\n")
    f.write("All model MAEs (for reference):\n")
    for n, r in sorted(results.items(), key=lambda x: x[1]["MAE"]):
        marker = " ← best" if n == best_overall_name else ""
        f.write(f"  {n:<10} MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  R²={r['R2']:.4f}{marker}\n")
    f.write("=" * 50 + "\n")
    # f.write(best_model_info["note"] + "\n")

print(f"\n{'='*55}")
print(f"BEST MODEL: {best_overall_name}  |  MAE={best_overall_mae:.4f}  |  R²={best_overall_r2:.4f}")
print(f"  → Saved as: best_model.joblib")
print(f"  → Summary : best_model_summary.txt")
print(f"{'='*55}\n")

# =============================================================================
# BOOTSTRAP 95% CI
# =============================================================================

def bootstrap_ci(y_true, y_pred, n_iter=BOOTSTRAP_ITER, seed=RANDOM_SEED):
    rng  = np.random.RandomState(seed)
    n    = len(y_true)
    maes = [
        mean_absolute_error(y_true[idx], y_pred[idx])
        for idx in (rng.choice(n, n, replace=True) for _ in range(n_iter))
    ]
    return np.percentile(maes, [2.5, 97.5])

ci_results = {}
print("\nBOOTSTRAP 95% CI (MAE)")
for name, pred in all_preds.items():
    lo, hi = bootstrap_ci(y_test_orig.values, pred)
    ci_results[name] = (lo, hi)
    print(f"  {name}: [{lo:.3f}, {hi:.3f}]")

# =============================================================================
# HELPER: save & show
# =============================================================================

def save_fig(filename):
    path = out(filename)
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")

# =============================================================================
# VISUALIZATION 1 — Model Performance Dashboard (combined)
# =============================================================================

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

model_names = list(results.keys())
colors      = [PALETTE[n] for n in model_names]

for i, metric in enumerate(["RMSE", "MAE", "R2"]):
    ax = fig.add_subplot(gs[0, i])
    vals = [results[n][metric] for n in model_names]
    bars = ax.bar(model_names, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals) * 0.01,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold"
        )
    best_idx = int(np.argmin(vals)) if metric != "R2" else int(np.argmax(vals))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2)

ax = fig.add_subplot(gs[1, 0])
mae_vals = [results[n]["MAE"] for n in model_names]
lo_vals  = [ci_results[n][0]  for n in model_names]
hi_vals  = [ci_results[n][1]  for n in model_names]
yerr     = [
    [mae_vals[i] - lo_vals[i] for i in range(len(model_names))],
    [hi_vals[i]  - mae_vals[i] for i in range(len(model_names))],
]
ax.bar(model_names, mae_vals, color=colors, edgecolor="white", linewidth=0.8, zorder=2)
ax.errorbar(model_names, mae_vals, yerr=yerr,
            fmt="none", color="black", capsize=5, linewidth=1.5, zorder=3)
ax.set_title("MAE with 95% Bootstrap CI")
ax.set_ylabel("MAE (days)")
ax.tick_params(axis="x", rotation=30)

ax = fig.add_subplot(gs[1, 1])
best_name = min(results, key=lambda n: results[n]["MAE"])
pred_best = all_preds[best_name]
ax.scatter(y_test_orig, pred_best, alpha=0.4, s=15,
           color=PALETTE[best_name], edgecolors="none")
lims = [min(y_test_orig.min(), pred_best.min()), max(y_test_orig.max(), pred_best.max())]
ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
ax.set_xlabel("Actual LOS (days)")
ax.set_ylabel("Predicted LOS (days)")
ax.set_title(f"Predicted vs Actual — {best_name}")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 2])
residuals = pred_best - y_test_orig.values
ax.scatter(pred_best, residuals, alpha=0.4, s=15,
           color=PALETTE[best_name], edgecolors="none")
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.set_xlabel("Predicted LOS (days)")
ax.set_ylabel("Residual (days)")
ax.set_title(f"Residuals — {best_name}")

fig.suptitle("Hospital Length-of-Stay — Model Performance Summary",
             fontsize=13, fontweight="bold", y=1.01)
save_fig("model_performance.png")

# =============================================================================
# VISUALIZATION 1 (individual) — Per-model metric bar charts
# =============================================================================

for name in model_names:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        val = results[name][metric]
        bar = ax.bar([name], [val], color=PALETTE[name], edgecolor="black", linewidth=1.5)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.text(0, val + val * 0.02, f"{val:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    fig.suptitle(f"{name} — Performance Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(f"metrics_{name}.png")

# =============================================================================
# VISUALIZATION 2 — Predicted vs Actual (combined all models)
# =============================================================================

n_models = len(all_preds)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
if n_models == 1:
    axes = [axes]

for ax, (name, pred) in zip(axes, all_preds.items()):
    ax.scatter(y_test_orig, pred, alpha=0.35, s=12,
               color=PALETTE[name], edgecolors="none")
    lims = [min(y_test_orig.min(), pred.min()), max(y_test_orig.max(), pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    r2  = results[name]["R2"]
    mae = results[name]["MAE"]
    ax.set_title(f"{name}\nR²={r2:.3f}  MAE={mae:.3f}", fontsize=10)
    ax.set_xlabel("Actual LOS (days)")
    ax.set_ylabel("Predicted LOS (days)")

fig.suptitle("Predicted vs Actual — All Models", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("predicted_vs_actual_all.png")

# =============================================================================
# VISUALIZATION 2 (individual) — Predicted vs Actual per model
# =============================================================================

for name, pred in all_preds.items():
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test_orig, pred, alpha=0.4, s=15,
               color=PALETTE[name], edgecolors="none")
    lims = [min(y_test_orig.min(), pred.min()), max(y_test_orig.max(), pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    r2  = results[name]["R2"]
    mae = results[name]["MAE"]
    ax.set_title(f"{name} — Predicted vs Actual\nR²={r2:.3f}  MAE={mae:.3f}", fontsize=11)
    ax.set_xlabel("Actual LOS (days)")
    ax.set_ylabel("Predicted LOS (days)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_fig(f"predicted_vs_actual_{name}.png")

# =============================================================================
# VISUALIZATION 3 — Residual distributions (combined)
# =============================================================================

fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
if n_models == 1:
    axes = [axes]

for ax, (name, pred) in zip(axes, all_preds.items()):
    resid = pred - y_test_orig.values
    ax.hist(resid, bins=40, color=PALETTE[name], edgecolor="white", linewidth=0.5)
    ax.axvline(0,            color="black", linewidth=1.2, linestyle="--", label="Zero")
    ax.axvline(resid.mean(), color="red",   linewidth=1.2, linestyle="-",
               label=f"Mean={resid.mean():.2f}")
    ax.set_title(f"{name} — Residuals")
    ax.set_xlabel("Residual (days)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

fig.suptitle("Residual Distributions", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("residual_distributions.png")

# =============================================================================
# VISUALIZATION 3 (individual) — Residual distribution per model
# =============================================================================

for name, pred in all_preds.items():
    fig, ax = plt.subplots(figsize=(6, 4))
    resid = pred - y_test_orig.values
    ax.hist(resid, bins=40, color=PALETTE[name], edgecolor="white", linewidth=0.5)
    ax.axvline(0,            color="black", linewidth=1.2, linestyle="--", label="Zero")
    ax.axvline(resid.mean(), color="red",   linewidth=1.2, linestyle="-",
               label=f"Mean={resid.mean():.2f}")
    ax.set_title(f"{name} — Residual Distribution", fontsize=11)
    ax.set_xlabel("Residual (days)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig(f"residuals_{name}.png")

# =============================================================================
# VISUALIZATION 4 (individual) — MAE with CI per model
# =============================================================================

for name in model_names:
    fig, ax = plt.subplots(figsize=(4, 4))
    mae_val = results[name]["MAE"]
    lo, hi  = ci_results[name]
    ax.bar([name], [mae_val], color=PALETTE[name], edgecolor="white", linewidth=0.8, zorder=2)
    ax.errorbar([name], [mae_val], yerr=[[mae_val - lo], [hi - mae_val]],
                fmt="none", color="black", capsize=8, linewidth=2, zorder=3)
    ax.set_title(f"{name} — MAE with 95% Bootstrap CI")
    ax.set_ylabel("MAE (days)")
    ax.text(0, hi + (hi - lo) * 0.1, f"[{lo:.3f}, {hi:.3f}]",
            ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save_fig(f"mae_ci_{name}.png")

# =============================================================================
# SHAP — with original feature names
# Tree-based models (RF, XGB, LGBM, GB, CatBoost) use TreeExplainer.
# Linear models (ElasticNet) use LinearExplainer.
# Kernel-based models (SVR) use KernelExplainer on a small background sample
# since it's model-agnostic but much slower.
# =============================================================================

rng        = np.random.RandomState(RANDOM_SEED)
sample_idx = rng.choice(len(X_test), size=min(300, len(X_test)), replace=False)
X_sample   = X_test[sample_idx]
X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

TREE_MODELS   = {"RF", "XGB", "LGBM", "GB", "CatBoost"}
LINEAR_MODELS = {"ElasticNet"}
KERNEL_MODELS = {"SVR"}

for name, model in best_models.items():
    try:
        inner = model.named_steps["model"]

        if "scaler" in model.named_steps:
            X_sample_for_shap = pd.DataFrame(
                model.named_steps["scaler"].transform(X_sample_df),
                columns=feature_names,
            )
        else:
            X_sample_for_shap = X_sample_df

        if name in TREE_MODELS:
            explainer   = shap.TreeExplainer(inner)
            shap_values = explainer.shap_values(X_sample_for_shap)
        elif name in LINEAR_MODELS:
            explainer   = shap.LinearExplainer(inner, X_sample_for_shap)
            shap_values = explainer.shap_values(X_sample_for_shap)
        elif name in KERNEL_MODELS:
            background  = shap.sample(X_sample_for_shap, min(50, len(X_sample_for_shap)))
            explainer   = shap.KernelExplainer(inner.predict, background)
            shap_values = explainer.shap_values(X_sample_for_shap, nsamples=100)
        else:
            explainer   = shap.Explainer(inner, X_sample_for_shap)
            shap_values = explainer(X_sample_for_shap).values

        # --- SHAP Summary beeswarm ---
        fig, ax = plt.subplots(figsize=(9, max(5, len(feature_names) * 0.3)))
        shap.summary_plot(
            shap_values, X_sample_df,
            plot_type="dot", max_display=20, show=False,
        )
        plt.title(f"SHAP Summary — {name}", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        save_fig(f"shap_summary_{name}.png")

        # --- SHAP Bar importance (with value labels) ---
        # Use the same absolute-mean SHAP metric as SHAP's default importance plot,
        # and keep the same descending ranking so the chart does not reorder when
        # you add the value labels.
        sv = shap_values
        if isinstance(sv, list):
            if len(sv) == 2:
                sv = np.abs(sv[1])
            else:
                sv = np.mean([np.abs(s) for s in sv], axis=0)
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = np.abs(sv[:, :, 1])
        else:
            sv = np.abs(sv)

        mean_abs = sv.mean(axis=0)
        order    = np.argsort(mean_abs)[::-1]
        feats    = np.array(feature_names)[order]
        vals     = mean_abs[order]

        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
        ax.barh(feats, vals, color="#1f77b4")
        ax.invert_yaxis()

        pad = vals.max() * 0.01
        for i, v in enumerate(vals):
            ax.text(v + pad, i, f"+{v:.3f}", va="center", ha="left",
                    color="red", fontsize=9)

        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_xlim(0, vals.max() * 1.18)
        ax.spines[["top", "right"]].set_visible(False)
        plt.title(f"SHAP Feature Importance — {name}", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        save_fig(f"shap_importance_{name}.png")

    except Exception as e:
        print(f"SHAP failed for {name}: {e}")

# =============================================================================
# SUMMARY — List all saved outputs
# =============================================================================

print("\n" + "=" * 60)
print(f"ALL OUTPUTS SAVED TO: {os.path.abspath(OUTPUT_DIR)}/")
print("=" * 60)
saved_files = sorted(os.listdir(OUTPUT_DIR))
for f in saved_files:
    size = os.path.getsize(out(f)) / 1024
    print(f"  {f:<50} {size:>8.1f} KB")
print("=" * 60)
