"""
Export every experiment from main.py into web/model_data.js so that the
interactive page can run entirely in the browser (GitHub Pages friendly).

Run:  python3 export_web_data.py
"""

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

from scipy import stats

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
TARGET = "FloodProbability"
FEATURE_COUNTS = [14, 20, 25, 30]
ALPHA = 0.05

DATA_DIR = Path("data")
WEB_DIR = Path(".")
SCATTER_SIZE = 220
OUTLIER_COUNT = 12

METHODS = ["SelectKBest", "RFE", "Sequential", "Lasso", "Permutation"]

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
    X = X.copy()
    for left, right in INTERACTIONS:
        X[f"{left}_x_{right}"] = X[left] * X[right]
    return X


def make_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def fit_feature_selector(method, k, X_train, y_train):
    n_features = X_train.shape[1]
    k = min(k, n_features)

    preprocessor = make_preprocessor()
    X_scaled = preprocessor.fit_transform(X_train)
    y_array = np.asarray(y_train)

    if method == "SelectKBest":
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X_scaled, y_array)
        mask = selector.get_support()

    elif method == "RFE":
        selector = RFE(
            estimator=LinearRegression(),
            n_features_to_select=k,
            step=1,
        )
        selector.fit(X_scaled, y_array)
        mask = selector.get_support()

    elif method == "Sequential":
        if k >= n_features:
            mask = np.ones(n_features, dtype=bool)
        else:
            rng = np.random.RandomState(RANDOM_STATE)
            n_sample = min(len(X_scaled), 3000)
            idx = rng.choice(len(X_scaled), size=n_sample, replace=False)
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

    elif method == "Permutation":
        baseline = LinearRegression()
        baseline.fit(X_scaled, y_array)
        rng = np.random.RandomState(RANDOM_STATE)
        n_sample = min(len(X_scaled), 10000)
        idx = rng.choice(len(X_scaled), size=n_sample, replace=False)
        perm = permutation_importance(
            baseline,
            X_scaled[idx],
            y_array[idx],
            scoring="neg_root_mean_squared_error",
            n_repeats=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        indices = np.argsort(perm.importances_mean)[::-1][:k]
        mask = np.zeros(n_features, dtype=bool)
        mask[indices] = True

    else:
        raise ValueError(f"Unknown method: {method}")

    selected = X_train.columns[mask].tolist()
    if len(selected) != k:
        raise RuntimeError(f"{method} selected {len(selected)} features, wanted {k}.")
    return selected


def ols_coefficients(estimator, X_fit_features, y_fit):
    """Recover original-scale OLS coefficients + inference for a fitted pipeline."""
    X_scaled = estimator.named_steps["preprocess"].transform(X_fit_features)
    scaler = estimator.named_steps["preprocess"].named_steps["scaler"]
    reg = estimator.named_steps["regressor"]

    X_scaled = np.asarray(X_scaled, dtype=float)
    y_array = np.asarray(y_fit, dtype=float)
    n, p = X_scaled.shape

    design = np.column_stack([np.ones(n), X_scaled])
    beta, *_ = np.linalg.lstsq(design, y_array, rcond=None)
    resid = y_array - design @ beta
    df = n - design.shape[1]
    sigma2 = float(resid @ resid / df)

    xtx_inv = np.linalg.pinv(design.T @ design)
    cov_scaled = sigma2 * xtx_inv
    se_scaled = np.sqrt(np.clip(np.diag(cov_scaled), 0, None))

    means = float(0) if not hasattr(scaler, "mean_") else np.asarray(scaler.mean_, dtype=float)
    scales = np.asarray(scaler.scale_, dtype=float)

    # theta_orig = T @ theta_scaled
    T = np.zeros((p + 1, p + 1))
    T[0, 0] = 1.0
    T[0, 1:] = -means / scales
    for i in range(p):
        T[i + 1, i + 1] = 1.0 / scales[i]

    theta_orig = T @ beta
    cov_orig = T @ cov_scaled @ T.T
    se_orig = np.sqrt(np.clip(np.diag(cov_orig), 0, None))
    tstat = theta_orig / np.where(se_orig > 0, se_orig, np.nan)
    pvalue = 2 * stats.t.sf(np.abs(tstat), df)
    tcrit = stats.t.ppf(1 - ALPHA / 2, df)
    ci_lower = theta_orig - tcrit * se_orig
    ci_upper = theta_orig + tcrit * se_orig

    # VIF computed in the standardized feature space.
    corr = np.corrcoef(X_scaled, rowvar=False)
    try:
        vif_all = np.diag(np.linalg.inv(corr))
    except np.linalg.LinAlgError:
        vif_all = np.diag(np.linalg.pinv(corr))
    vif_all = np.clip(vif_all, 0, None)

    names = ["Intercept"] + list(X_fit_features.columns)
    out = []
    for i, name in enumerate(names):
        out.append({
            "feature": name,
            "coef": float(theta_orig[i]),
            "stderr": float(se_orig[i]) if np.isfinite(se_orig[i]) else None,
            "tstat": float(tstat[i]) if np.isfinite(tstat[i]) else None,
            "pvalue": float(pvalue[i]) if np.isfinite(pvalue[i]) else None,
            "ci_lower": float(ci_lower[i]) if np.isfinite(ci_lower[i]) else None,
            "ci_upper": float(ci_upper[i]) if np.isfinite(ci_upper[i]) else None,
            "vif": None if i == 0 else float(vif_all[i - 1]),
        })
    return out


def main():
    print("Loading data ...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    ID_COL = "id" if "id" in train_df.columns else None
    base_features = [
        col for col in train_df.columns
        if col != TARGET and col != ID_COL
    ]

    X_raw = train_df[base_features].copy()
    y = train_df[TARGET].copy()
    X_all = add_interactions(X_raw)

    X_dev, X_test, y_dev, y_test = train_test_split(
        X_all, y, test_size=0.15, random_state=RANDOM_STATE,
    )
    X_fit, X_temp, y_fit, y_temp = train_test_split(
        X_dev, y_dev, test_size=25 / 85, random_state=RANDOM_STATE,
    )
    X_val, X_cal, y_val, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.40, random_state=RANDOM_STATE,
    )

    print(f"fit={len(X_fit):,} val={len(X_val):,} cal={len(X_cal):,} test={len(X_test):,}")

    # Input statistics for the browser sliders.
    feature_stats = {}
    for col in base_features:
        s = X_raw[col]
        feature_stats[col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "mean": float(s.mean()),
            "std": float(s.std()),
        }

    target_hist, target_edges = np.histogram(y.to_numpy(), bins=45)
    target_centers = (target_edges[:-1] + target_edges[1:]) / 2
    target_histogram = {
        "centers": [float(v) for v in target_centers],
        "counts": [int(v) for v in target_hist],
    }

    models = {m: [] for m in METHODS}
    experiments = []
    scatter_rng = np.random.RandomState(RANDOM_STATE)
    n_test = len(X_test)
    scatter_idx = scatter_rng.choice(n_test, size=min(SCATTER_SIZE, n_test), replace=False)

    for method in METHODS:
        for k in FEATURE_COUNTS:
            print(f"Training {method} k={k} ...", flush=True)
            start = time.time()

            features = fit_feature_selector(method, k, X_fit, y_fit)

            model = Pipeline([
                ("preprocess", make_preprocessor()),
                ("regressor", LinearRegression()),
            ])
            model.fit(X_fit[features], y_fit)

            val_pred = model.predict(X_val[features])
            val_array = y_val.to_numpy()
            val_residuals = val_array - val_pred
            val_rmse = float(np.sqrt(mean_squared_error(val_array, val_pred)))
            val_mae = float(mean_absolute_error(val_array, val_pred))
            val_r2 = float(r2_score(val_array, val_pred))
            val_se = float(
                np.std(val_residuals ** 2)
                / (2.0 * val_rmse * np.sqrt(len(val_residuals)))
            )

            test_pred = model.predict(X_test[features])
            test_array = y_test.to_numpy()
            test_rmse = float(np.sqrt(mean_squared_error(test_array, test_pred)))
            test_mae = float(mean_absolute_error(test_array, test_pred))
            test_r2 = float(r2_score(test_array, test_pred))

            cal_pred = model.predict(X_cal[features])
            cal_residuals = np.abs(y_cal.to_numpy() - cal_pred)
            n_cal = len(cal_residuals)
            rank = int(np.ceil((n_cal + 1) * (1 - ALPHA)))
            q_hat = float(np.inf) if rank > n_cal else float(np.sort(cal_residuals)[rank - 1])

            lower = test_pred - q_hat
            upper = test_pred + q_hat
            coverage = float(np.mean((test_array >= lower) & (test_array <= upper)))
            width = float(np.mean(upper - lower))

            coefficients = ols_coefficients(model, X_fit[features], y_fit)

            order = np.argsort(test_pred[scatter_idx])
            sidx = scatter_idx[order]
            scatter = {
                "actual": [float(v) for v in test_array[sidx]],
                "predicted": [float(v) for v in test_pred[sidx]],
                "lower": [float(v) for v in (test_pred[sidx] - q_hat)],
                "upper": [float(v) for v in (test_pred[sidx] + q_hat)],
                "residual": [float(v) for v in (test_array[sidx] - test_pred[sidx])],
            }

            abs_res = np.abs(test_array - test_pred)
            top = np.argsort(abs_res)[::-1][:OUTLIER_COUNT]
            outliers = []
            for rank_i, idx in enumerate(top, start=1):
                actual = float(test_array[idx])
                predicted = float(test_pred[idx])
                residual = actual - predicted
                outliers.append({
                    "rank": rank_i,
                    "id": int(train_df.iloc[X_test.index[idx]][ID_COL]) if ID_COL else int(idx),
                    "actual": actual,
                    "predicted": predicted,
                    "residual": residual,
                    "abs_error": abs(float(residual)),
                    "pct_error": float(abs(residual) / actual * 100) if actual != 0 else None,
                })

            runtime = time.time() - start

            models[method].append({
                "k": k,
                "n_features": len(features),
                "features": features,
                "validation": {"rmse": val_rmse, "mae": val_mae, "r2": val_r2, "se": val_se},
                "test": {"rmse": test_rmse, "mae": test_mae, "r2": test_r2},
                "interval": {
                    "q_hat": q_hat,
                    "coverage": coverage,
                    "width": width,
                    "nominal": 1 - ALPHA,
                },
                "runtime_seconds": runtime,
                "coefficients": coefficients,
                "scatter": scatter,
                "outliers": outliers,
            })

            experiments.append({
                "method": method,
                "k": k,
                "n_features": len(features),
                "validation_RMSE": val_rmse,
                "validation_SE": val_se,
                "validation_MAE": val_mae,
                "validation_R2": val_r2,
                "test_RMSE": test_rmse,
                "test_MAE": test_mae,
                "test_R2": test_r2,
                "interval_coverage": coverage,
                "runtime_seconds": runtime,
            })

    experiments.sort(key=lambda r: r["validation_RMSE"])

    # One-standard-error rule: among models within one SE of the minimum
    # validation RMSE, prefer the model with the fewest features.
    min_row = min(
        experiments,
        key=lambda r: (
            r["validation_RMSE"],
            r["n_features"],
            METHODS.index(r["method"]),
        ),
    )
    threshold = min_row["validation_RMSE"] + min_row["validation_SE"]
    candidates = [
        r for r in experiments
        if r["validation_RMSE"] <= threshold
    ]
    best_row = min(
        candidates,
        key=lambda r: (
            r["n_features"],
            r["validation_RMSE"],
            METHODS.index(r["method"]),
        ),
    )
    best_method = best_row["method"]
    best_k = int(best_row["k"])

    duplicates = int(train_df.duplicated().sum())

    data = {
        "meta": {
            "target": TARGET,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "n_base_features": len(base_features),
            "n_candidate_features": int(X_all.shape[1]),
            "duplicate_rows": duplicates,
            "feature_counts": FEATURE_COUNTS,
            "methods": METHODS,
            "random_state": RANDOM_STATE,
            "alpha": ALPHA,
            "selection_rule": "one_standard_error",
            "best_method": best_method,
            "best_k": best_k,
            "min_validation": min_row,
            "split": {
                "fit": int(len(X_fit)),
                "validation": int(len(X_val)),
                "calibration": int(len(X_cal)),
                "test": int(len(X_test)),
            },
            "best_validation": best_row,
        },
        "baseFeatures": base_features,
        "interactions": [[a, b] for a, b in INTERACTIONS],
        "featureStats": feature_stats,
        "targetHistogram": target_histogram,
        "methods": METHODS,
        "models": models,
        "experiments": experiments,
    }

    out_path = WEB_DIR / "model_data.js"
    out_path.write_text(
        "window.FLOOD_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    print(
        f"Min validation: {min_row['method']} k={min_row['k']} "
        f"val RMSE={min_row['validation_RMSE']:.9f} "
        f"(SE={min_row['validation_SE']:.9f})"
    )
    print(
        f"1-SE best: {best_method} k={best_k} | "
        f"val RMSE={best_row['validation_RMSE']:.9f} "
        f"test RMSE={best_row['test_RMSE']:.9f} "
        f"test R2={best_row['test_R2']:.4f}"
    )


if __name__ == "__main__":
    main()
