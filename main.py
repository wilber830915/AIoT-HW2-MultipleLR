
# ============================================================
# Flood Prediction with Multiple Linear Regression
# CRISP-DM: Complete End-to-End Implementation
#
# Dataset: Kaggle Playground Series S4E5
# Source: https://www.kaggle.com/datasets/johnwicknice/playground-series
# Target: FloodProbability
#
# Feature Selection:
#   1. SelectKBest
#   2. RFE
#   3. Sequential Feature Selection
#   4. Lasso
#   5. Permutation Importance
#
# Feature counts: 14, 20, 25, 30
# Model: Multiple Linear Regression
# Evaluation: RMSE, MAE, R2, Prediction Interval Coverage
# ============================================================

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    RFE,
    SequentialFeatureSelector,
)
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
TARGET = "FloodProbability"
FEATURE_COUNTS = [14, 20, 25, 30]
ALPHA = 0.05

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"


# ============================================================
# CRISP-DM 1: BUSINESS UNDERSTANDING
# ============================================================

print("\n" + "=" * 70)
print("CRISP-DM 1: BUSINESS UNDERSTANDING")
print("=" * 70)

print("Objective: Predict FloodProbability.")
print("Model: Multiple Linear Regression.")
print("Official competition metric: R2.")
print("Additional metrics: RMSE and MAE.")
print("Feature selection: 5 methods x 4 feature counts.")
print("Prediction intervals: 95% split conformal intervals.")


# ============================================================
# CRISP-DM 2: DATA UNDERSTANDING
# ============================================================

print("\n" + "=" * 70)
print("CRISP-DM 2: DATA UNDERSTANDING")
print("=" * 70)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

if TARGET not in train_df.columns:
    raise ValueError(
        f"Target '{TARGET}' not found in train.csv."
    )

ID_COL = "id" if "id" in train_df.columns else None

base_features = [
    col for col in train_df.columns
    if col != TARGET and col != ID_COL
]

missing_test_features = [
    col for col in base_features
    if col not in test_df.columns
]

if missing_test_features:
    raise ValueError(
        f"Missing test features: {missing_test_features}"
    )

non_numeric = train_df[base_features].select_dtypes(
    exclude=np.number
).columns.tolist()

if non_numeric:
    raise ValueError(
        "This implementation expects numerical features. "
        f"Encode categorical columns first: {non_numeric}"
    )

X_raw = train_df[base_features].copy()
y = train_df[TARGET].copy()

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Original feature count:", len(base_features))
print("Duplicate training rows:", train_df.duplicated().sum())

print("\nMissing values:")
print(
    train_df[base_features + [TARGET]]
    .isna()
    .sum()
    .sort_values(ascending=False)
    .head(15)
)

print("\nTarget summary:")
print(y.describe())

# Target distribution plot
plt.figure(figsize=(8, 5))
sns.histplot(y, kde=True)
plt.title("Distribution of FloodProbability")
plt.xlabel(TARGET)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "target_distribution.png",
    dpi=160,
)
plt.close()


# ============================================================
# CRISP-DM 3: DATA PREPARATION
# ============================================================

print("\n" + "=" * 70)
print("CRISP-DM 3: DATA PREPARATION")
print("=" * 70)

# Ten predefined interaction features.
# Together with the 20 original predictors, these produce
# 30 candidate features for the requested comparison.

INTERACTIONS = [
    ("MonsoonIntensity", "ClimateChange"),
    ("MonsoonIntensity", "DrainageSystems"),
    ("MonsoonIntensity", "RiverManagement"),
    ("ClimateChange", "Deforestation"),
    ("ClimateChange", "Urbanization"),
    ("Deforestation", "Urbanization"),
    ("DamsQuality", "DrainageSystems"),
    ("WetlandLoss", "CoastalVulnerability"),
    ("Watersheds", "Landslides"),
    ("PopulationScore", "InadequatePlanning"),
]


def add_interactions(X):
    """Create predefined pairwise interaction features."""

    X = X.copy()

    for left, right in INTERACTIONS:
        if left not in X.columns:
            raise ValueError(
                f"Required interaction column missing: {left}"
            )

        if right not in X.columns:
            raise ValueError(
                f"Required interaction column missing: {right}"
            )

        X[f"{left}_x_{right}"] = X[left] * X[right]

    return X


X_all = add_interactions(X_raw)

if X_all.shape[1] < max(FEATURE_COUNTS):
    raise ValueError(
        f"Need at least {max(FEATURE_COUNTS)} candidate features, "
        f"but only {X_all.shape[1]} were created."
    )

print("Original predictors:", X_raw.shape[1])
print("All candidate predictors:", X_all.shape[1])

# Split 15% for final testing.
X_dev, X_test, y_dev, y_test = train_test_split(
    X_all,
    y,
    test_size=0.15,
    random_state=RANDOM_STATE,
)

# Remaining data: 60% fit, 15% validation, 10% calibration
# of the original dataset.
X_fit, X_temp, y_fit, y_temp = train_test_split(
    X_dev,
    y_dev,
    test_size=25 / 85,
    random_state=RANDOM_STATE,
)

X_val, X_cal, y_val, y_cal = train_test_split(
    X_temp,
    y_temp,
    test_size=0.40,
    random_state=RANDOM_STATE,
)

print("\nData split:")
print(f"Fit:         {len(X_fit):,} rows")
print(f"Validation:  {len(X_val):,} rows")
print(f"Calibration: {len(X_cal):,} rows")
print(f"Test:        {len(X_test):,} rows")


def make_preprocessor():
    """Impute missing values and standardize features."""

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


# ============================================================
# CRISP-DM 4: MODELING
# ============================================================

print("\n" + "=" * 70)
print("CRISP-DM 4: MODELING")
print("=" * 70)

METHODS = [
    "SelectKBest",
    "RFE",
    "Sequential",
    "Lasso",
    "Permutation",
]


def fit_feature_selector(method, k, X_train, y_train):
    """
    Fit feature selection using training data only.

    Returns:
        List of selected original/engineered feature names.
    """

    n_features = X_train.shape[1]
    k = min(k, n_features)

    preprocessor = make_preprocessor()
    X_scaled = preprocessor.fit_transform(X_train)
    y_array = np.asarray(y_train)

    # --------------------------------------------------------
    # Method 1: SelectKBest
    # --------------------------------------------------------
    if method == "SelectKBest":

        selector = SelectKBest(
            score_func=f_regression,
            k=k,
        )

        selector.fit(X_scaled, y_array)
        mask = selector.get_support()

    # --------------------------------------------------------
    # Method 2: Recursive Feature Elimination
    # --------------------------------------------------------
    elif method == "RFE":

        selector = RFE(
            estimator=LinearRegression(),
            n_features_to_select=k,
            step=1,
        )

        selector.fit(X_scaled, y_array)
        mask = selector.get_support()

    # --------------------------------------------------------
    # Method 3: Sequential Feature Selection
    # Forward selection with a training-only subsample.
    # --------------------------------------------------------
    elif method == "Sequential":

        # SequentialFeatureSelector requires k < n_features.
        if k >= n_features:
            mask = np.ones(n_features, dtype=bool)

        else:
            rng = np.random.RandomState(RANDOM_STATE)
            n_sample = min(len(X_scaled), 3000)

            idx = rng.choice(
                len(X_scaled),
                size=n_sample,
                replace=False,
            )

            selector = SequentialFeatureSelector(
                estimator=LinearRegression(),
                n_features_to_select=k,
                direction="forward",
                scoring="neg_root_mean_squared_error",
                cv=2,
                n_jobs=-1,
            )

            selector.fit(X_scaled[idx], y_array[idx])
            mask = selector.get_support()

    # --------------------------------------------------------
    # Method 4: Lasso
    # Rank coefficients by absolute magnitude and select top k.
    # --------------------------------------------------------
    elif method == "Lasso":

        selector = LassoCV(
            cv=3,
            random_state=RANDOM_STATE,
            max_iter=5000,
            n_jobs=-1,
        )

        selector.fit(X_scaled, y_array)

        importance = np.abs(selector.coef_)
        indices = np.argsort(importance)[::-1][:k]

        mask = np.zeros(n_features, dtype=bool)
        mask[indices] = True

    # --------------------------------------------------------
    # Method 5: Permutation Importance
    # --------------------------------------------------------
    elif method == "Permutation":

        baseline = LinearRegression()
        baseline.fit(X_scaled, y_array)

        rng = np.random.RandomState(RANDOM_STATE)
        n_sample = min(len(X_scaled), 10000)

        idx = rng.choice(
            len(X_scaled),
            size=n_sample,
            replace=False,
        )

        perm = permutation_importance(
            baseline,
            X_scaled[idx],
            y_array[idx],
            scoring="neg_root_mean_squared_error",
            n_repeats=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        indices = np.argsort(
            perm.importances_mean
        )[::-1][:k]

        mask = np.zeros(n_features, dtype=bool)
        mask[indices] = True

    else:
        raise ValueError(f"Unknown method: {method}")

    selected = X_train.columns[mask].tolist()

    if len(selected) != k:
        raise RuntimeError(
            f"{method} selected {len(selected)} features, "
            f"but {k} were requested."
        )

    return selected


# ------------------------------------------------------------
# Run all 20 feature-selection experiments
# ------------------------------------------------------------

results = []
selected_features = {}
fitted_models = {}

for method in METHODS:
    for k in FEATURE_COUNTS:

        print(f"\nTraining {method}, k={k}")
        start_time = time.time()

        features = fit_feature_selector(
            method,
            k,
            X_fit,
            y_fit,
        )

        model = Pipeline([
            ("preprocess", make_preprocessor()),
            ("regressor", LinearRegression()),
        ])

        model.fit(
            X_fit[features],
            y_fit,
        )

        val_pred = model.predict(
            X_val[features]
        )

        val_residuals = y_val.to_numpy() - val_pred

        val_rmse = np.sqrt(
            mean_squared_error(y_val, val_pred)
        )

        val_se = (
            np.std(val_residuals ** 2)
            / (2.0 * val_rmse * np.sqrt(len(val_residuals)))
        )

        val_mae = mean_absolute_error(
            y_val, val_pred
        )

        val_r2 = r2_score(
            y_val, val_pred
        )

        key = (method, k)

        selected_features[key] = features
        fitted_models[key] = model

        results.append({
            "method": method,
            "requested_k": k,
            "n_features": len(features),
            "validation_RMSE": val_rmse,
            "validation_SE": val_se,
            "validation_MAE": val_mae,
            "validation_R2": val_r2,
            "runtime_seconds": time.time() - start_time,
        })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "validation_RMSE",
    ascending=True,
).reset_index(drop=True)

results_df.to_csv(
    OUTPUT_DIR / "feature_selection_comparison.csv",
    index=False,
)

print("\nFeature selection results:")
print(results_df.to_string(index=False))

# Plot comparison of all feature-selection methods.
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.lineplot(
    data=results_df,
    x="n_features",
    y="validation_RMSE",
    hue="method",
    marker="o",
    ax=axes[0],
)

axes[0].set_title("Feature Selection vs Validation RMSE")
axes[0].set_xlabel("Number of Features")
axes[0].set_ylabel("Validation RMSE")

sns.lineplot(
    data=results_df,
    x="n_features",
    y="validation_R2",
    hue="method",
    marker="o",
    ax=axes[1],
)

axes[1].set_title("Feature Selection vs Validation R2")
axes[1].set_xlabel("Number of Features")
axes[1].set_ylabel("Validation R2")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "feature_selection_comparison.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# CRISP-DM 5: MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("CRISP-DM 5: MODEL EVALUATION")
print("=" * 70)

# One-standard-error rule: among models within one SE of the minimum
# validation RMSE, prefer the model with the fewest features.
min_row = results_df.loc[
    results_df["validation_RMSE"].idxmin()
]
selection_threshold = (
    min_row["validation_RMSE"] + min_row["validation_SE"]
)
candidates = results_df[
    results_df["validation_RMSE"] <= selection_threshold
].copy()
candidates["method_order"] = candidates["method"].map(
    {method: index for index, method in enumerate(METHODS)}
)
candidates = candidates.sort_values(
    ["n_features", "validation_RMSE", "method_order"],
    ascending=True,
)
best_row = candidates.iloc[0]

best_method = best_row["method"]
best_k = int(best_row["requested_k"])
best_key = (best_method, best_k)

best_features = selected_features[best_key]
best_model = fitted_models[best_key]

print("\nMinimum validation RMSE:")
print("Method:", min_row["method"])
print("Feature count:", int(min_row["requested_k"]))
print("Validation RMSE:", min_row["validation_RMSE"])
print("Validation SE:", min_row["validation_SE"])

print("\nBest configuration (1-SE rule):")
print("Method:", best_method)
print("Feature count:", best_k)
print("Features:", best_features)

# ------------------------------------------------------------
# Final test-set evaluation
# ------------------------------------------------------------

test_pred = best_model.predict(
    X_test[best_features]
)

test_rmse = np.sqrt(
    mean_squared_error(y_test, test_pred)
)

test_mae = mean_absolute_error(
    y_test, test_pred
)

test_r2 = r2_score(
    y_test, test_pred
)

print("\nIndependent test metrics:")
print(f"RMSE: {test_rmse:.6f}")
print(f"MAE : {test_mae:.6f}")
print(f"R2  : {test_r2:.6f}")

test_metrics = pd.DataFrame([{
    "best_method": best_method,
    "n_features": best_k,
    "RMSE": test_rmse,
    "MAE": test_mae,
    "R2": test_r2,
}])

test_metrics.to_csv(
    OUTPUT_DIR / "test_metrics.csv",
    index=False,
)

# ------------------------------------------------------------
# 95% split conformal prediction intervals
# Calibration data is not used to fit the model.
# ------------------------------------------------------------

cal_pred = best_model.predict(
    X_cal[best_features]
)

cal_residuals = np.abs(
    np.asarray(y_cal) - cal_pred
)

n_cal = len(cal_residuals)

rank = int(
    np.ceil((n_cal + 1) * (1 - ALPHA))
)

if rank > n_cal:
    q_hat = np.inf
else:
    q_hat = np.sort(cal_residuals)[rank - 1]

lower = test_pred - q_hat
upper = test_pred + q_hat

test_array = np.asarray(y_test)

coverage = np.mean(
    (test_array >= lower) &
    (test_array <= upper)
)

mean_width = np.mean(upper - lower)

print("\n95% Prediction Interval:")
print(f"Conformal quantile: {q_hat:.6f}")
print(f"Observed coverage:  {coverage:.4%}")
print(f"Mean interval width: {mean_width:.6f}")

interval_metrics = pd.DataFrame([{
    "nominal_coverage": 1 - ALPHA,
    "observed_test_coverage": coverage,
    "mean_interval_width": mean_width,
    "conformal_quantile": q_hat,
}])

interval_metrics.to_csv(
    OUTPUT_DIR / "prediction_interval_metrics.csv",
    index=False,
)

# ------------------------------------------------------------
# Plot 1: Actual vs predicted with prediction interval
# ------------------------------------------------------------

plot_df = pd.DataFrame({
    "actual": test_array,
    "predicted": test_pred,
    "lower": lower,
    "upper": upper,
})

plot_df = plot_df.sample(
    n=min(300, len(plot_df)),
    random_state=RANDOM_STATE,
)

plot_df = plot_df.sort_values(
    "predicted"
).reset_index(drop=True)

x = np.arange(len(plot_df))

fig, ax = plt.subplots(figsize=(14, 6))

ax.fill_between(
    x,
    plot_df["lower"].to_numpy(),
    plot_df["upper"].to_numpy(),
    alpha=0.25,
    label="95% Prediction Interval",
)

ax.plot(
    x,
    plot_df["actual"].to_numpy(),
    ".",
    markersize=4,
    label="Actual",
)

ax.plot(
    x,
    plot_df["predicted"].to_numpy(),
    "-",
    linewidth=1.5,
    label="Predicted",
)

ax.set_title(
    f"Actual vs Predicted with 95% Prediction Interval\n"
    f"{best_method}, k={best_k}"
)

ax.set_xlabel("Test samples sorted by predicted value")
ax.set_ylabel("FloodProbability")
ax.legend()

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "prediction_interval.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# Plot 2: Parity plot and residual plot
# ------------------------------------------------------------

residuals = test_array - test_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(
    test_array,
    test_pred,
    alpha=0.2,
    s=8,
)

vmin = min(
    float(np.min(test_array)),
    float(np.min(test_pred)),
)

vmax = max(
    float(np.max(test_array)),
    float(np.max(test_pred)),
)

axes[0].plot(
    [vmin, vmax],
    [vmin, vmax],
    "k--",
    label="Perfect Prediction",
)

axes[0].set_title(
    f"Actual vs Predicted (R2={test_r2:.4f})"
)

axes[0].set_xlabel("Actual FloodProbability")
axes[0].set_ylabel("Predicted FloodProbability")
axes[0].legend()

axes[1].scatter(
    test_pred,
    residuals,
    alpha=0.2,
    s=8,
)

axes[1].axhline(
    0,
    linestyle="--",
    color="black",
)

axes[1].set_title("Residual Analysis")
axes[1].set_xlabel("Predicted FloodProbability")
axes[1].set_ylabel("Residual")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "regression_diagnostics.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# CRISP-DM 6: DEPLOYMENT
# ============================================================

print("\n" + "=" * 70)
print("CRISP-DM 6: DEPLOYMENT")
print("=" * 70)

# Save the selected features.
with open(
    OUTPUT_DIR / "best_features.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "method": best_method,
            "n_features": best_k,
            "features": best_features,
        },
        file,
        ensure_ascii=False,
        indent=2,
    )

# Save the model evaluated on the held-out test set.
joblib.dump(
    best_model,
    OUTPUT_DIR / "best_model.joblib",
)

# Refit the selected configuration on all labeled training data
# for the final Kaggle submission.
X_submission = add_interactions(
    train_df[base_features].copy()
)

X_kaggle_test = add_interactions(
    test_df[base_features].copy()
)

submission_model = Pipeline([
    ("preprocess", make_preprocessor()),
    ("regressor", LinearRegression()),
])

submission_model.fit(
    X_submission[best_features],
    train_df[TARGET],
)

kaggle_pred = submission_model.predict(
    X_kaggle_test[best_features]
)

if SUBMISSION_PATH.exists():

    submission = pd.read_csv(SUBMISSION_PATH)

    if len(submission) != len(test_df):
        raise ValueError(
            "sample_submission.csv row count does not match test.csv."
        )

    submission[TARGET] = kaggle_pred

else:

    submission = pd.DataFrame({
        TARGET: kaggle_pred,
    })

    if ID_COL is not None and ID_COL in test_df.columns:
        submission.insert(
            0,
            ID_COL,
            test_df[ID_COL].to_numpy(),
        )

submission.to_csv(
    OUTPUT_DIR / "submission.csv",
    index=False,
)

joblib.dump(
    submission_model,
    OUTPUT_DIR / "submission_model.joblib",
)

# Save experiment configuration.
config = {
    "target": TARGET,
    "candidate_feature_count": int(X_all.shape[1]),
    "feature_counts": FEATURE_COUNTS,
    "methods": METHODS,
    "best_method": best_method,
    "best_k": best_k,
    "random_state": RANDOM_STATE,
    "prediction_interval_alpha": ALPHA,
}

with open(
    OUTPUT_DIR / "experiment_config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        config,
        file,
        ensure_ascii=False,
        indent=2,
    )

print("\nAll tasks completed.")
print("Output directory:", OUTPUT_DIR.resolve())
print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.iterdir()):
    print(" -", path.name)

print("\nFinal Kaggle submission preview:")
print(submission.head())

print("\nFinal test metrics:")
print(test_metrics.to_string(index=False))

print("\nPrediction interval metrics:")
print(interval_metrics.to_string(index=False))