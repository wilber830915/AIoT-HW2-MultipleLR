"""
Export the Automobile dataset experiments to automobile_model_data.js.

Dataset: https://www.kaggle.com/datasets/toramky/automobile-dataset
Target: price
"""

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
TARGET = "price"
FEATURE_COUNTS = [8, 16, 24, 32]
ALPHA = 0.05

DATA_PATH = Path("data") / "Automobile_data.csv"
OUT_PATH = Path("automobile_model_data.js")
SCATTER_SIZE = 220
OUTLIER_COUNT = 12

METHODS = ["SelectKBest", "RFE", "Sequential", "Lasso", "Permutation"]

NUMERIC_FEATURES = [
    "symboling",
    "normalized-losses",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-size",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
]

CATEGORICAL_FEATURES = [
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "engine-type",
    "num-of-cylinders",
    "fuel-system",
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.replace("?", np.nan)
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df


def make_preprocessor():
    return Pipeline([
        ("columns", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                )),
            ]), CATEGORICAL_FEATURES),
        ])),
        ("scaler", StandardScaler()),
    ])


def feature_sources(preprocessor):
    names = preprocessor.named_steps["columns"].get_feature_names_out()
    sources = []
    for name in names:
        if name.startswith("num__"):
            sources.append({
                "feature": name[5:],
                "source": {"type": "numeric", "name": name[5:]},
            })
        else:
            rest = name[5:]
            matched = None
            for col in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
                if rest.startswith(col + "_"):
                    matched = (col, rest[len(col) + 1:])
                    break
            if matched is None:
                raise RuntimeError(f"Cannot decode dummy feature: {name}")
            col, value = matched
            sources.append({
                "feature": f"{col} = {value}",
                "source": {"type": "dummy", "name": col, "value": value},
            })
    return sources


def fit_feature_selector(method, k, X_train, y_train, names):
    n_features = X_train.shape[1]
    k = min(k, n_features)
    y_array = np.asarray(y_train)

    if method == "SelectKBest":
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X_train, y_array)
        mask = selector.get_support()

    elif method == "RFE":
        selector = RFE(
            estimator=LinearRegression(),
            n_features_to_select=k,
            step=1,
        )
        selector.fit(X_train, y_array)
        mask = selector.get_support()

    elif method == "Sequential":
        if k >= n_features:
            mask = np.ones(n_features, dtype=bool)
        else:
            rng = np.random.RandomState(RANDOM_STATE)
            n_sample = min(len(X_train), 3000)
            idx = rng.choice(len(X_train), size=n_sample, replace=False)
            selector = SequentialFeatureSelector(
                estimator=LinearRegression(),
                n_features_to_select=k,
                direction="forward",
                scoring="neg_root_mean_squared_error",
                cv=2,
                n_jobs=-1,
            )
            selector.fit(X_train[idx], y_array[idx])
            mask = selector.get_support()

    elif method == "Lasso":
        selector = LassoCV(
            cv=3,
            random_state=RANDOM_STATE,
            max_iter=5000,
            n_jobs=-1,
        )
        selector.fit(X_train, y_array)
        indices = np.argsort(np.abs(selector.coef_))[::-1][:k]
        mask = np.zeros(n_features, dtype=bool)
        mask[indices] = True

    elif method == "Permutation":
        baseline = LinearRegression()
        baseline.fit(X_train, y_array)
        rng = np.random.RandomState(RANDOM_STATE)
        n_sample = min(len(X_train), 10000)
        idx = rng.choice(len(X_train), size=n_sample, replace=False)
        perm = permutation_importance(
            baseline,
            X_train[idx],
            y_array[idx],
            scoring="neg_root_mean_squared_error",
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        indices = np.argsort(perm.importances_mean)[::-1][:k]
        mask = np.zeros(n_features, dtype=bool)
        mask[indices] = True

    else:
        raise ValueError(f"Unknown method: {method}")

    selected = [names[i] for i in np.where(mask)[0]]
    if len(selected) != k:
        raise RuntimeError(f"{method} selected {len(selected)}, wanted {k}.")
    return selected, np.where(mask)[0]


def ols_coefficients(preprocessor, selected_idx, sources, y_fit):
    X_scaled = preprocessor.transform(selected_idx["X"])
    X_scaled = np.asarray(X_scaled, dtype=float)[:, selected_idx["indices"]]
    scaler = preprocessor.named_steps["scaler"]
    y_array = np.asarray(y_fit, dtype=float)
    n, p = X_scaled.shape

    design = np.column_stack([np.ones(n), X_scaled])
    beta, *_ = np.linalg.lstsq(design, y_array, rcond=None)
    resid = y_array - design @ beta
    df = n - design.shape[1]
    sigma2 = float(resid @ resid / df)
    cov_scaled = sigma2 * np.linalg.pinv(design.T @ design)
    means = np.asarray(scaler.mean_, dtype=float)[selected_idx["indices"]]
    scales = np.asarray(scaler.scale_, dtype=float)[selected_idx["indices"]]

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

    corr = np.corrcoef(X_scaled, rowvar=False)
    try:
        vif_all = np.diag(np.linalg.inv(corr))
    except np.linalg.LinAlgError:
        vif_all = np.diag(np.linalg.pinv(corr))
    vif_all = np.clip(vif_all, 0, None)

    selected_sources = [sources[i] for i in selected_idx["indices"]]
    out = [{
        "feature": "Intercept",
        "coef": float(theta_orig[0]),
        "stderr": float(se_orig[0]) if np.isfinite(se_orig[0]) else None,
        "tstat": float(tstat[0]) if np.isfinite(tstat[0]) else None,
        "pvalue": float(pvalue[0]) if np.isfinite(pvalue[0]) else None,
        "ci_lower": float(ci_lower[0]) if np.isfinite(ci_lower[0]) else None,
        "ci_upper": float(ci_upper[0]) if np.isfinite(ci_upper[0]) else None,
        "vif": None,
        "source": None,
    }]
    for i, meta in enumerate(selected_sources):
        out.append({
            "feature": meta["feature"],
            "coef": float(theta_orig[i + 1]),
            "stderr": float(se_orig[i + 1]) if np.isfinite(se_orig[i + 1]) else None,
            "tstat": float(tstat[i + 1]) if np.isfinite(tstat[i + 1]) else None,
            "pvalue": float(pvalue[i + 1]) if np.isfinite(pvalue[i + 1]) else None,
            "ci_lower": float(ci_lower[i + 1]) if np.isfinite(ci_lower[i + 1]) else None,
            "ci_upper": float(ci_upper[i + 1]) if np.isfinite(ci_upper[i + 1]) else None,
            "vif": float(vif_all[i]),
            "source": meta["source"],
        })
    return out


def choose_step(span):
    if span > 1500:
        return 10
    if span > 100:
        return 1
    if span > 10:
        return 0.1
    return 0.01


def main():
    df = load_data()
    print("Automobile rows after dropping missing target:", len(df))

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()

    preprocessor = make_preprocessor()
    X_all = preprocessor.fit_transform(X)
    names = list(preprocessor.named_steps["columns"].get_feature_names_out())
    sources = feature_sources(preprocessor)
    print("Candidate transformed features:", len(names))

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE,
    )
    X_fit, X_temp, y_fit, y_temp = train_test_split(
        X_dev, y_dev, test_size=25 / 85, random_state=RANDOM_STATE,
    )
    X_val, X_cal, y_val, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.40, random_state=RANDOM_STATE,
    )
    print(f"fit={len(X_fit)} val={len(X_val)} cal={len(X_cal)} test={len(X_test)}")

    preprocessor = make_preprocessor()
    Xt_fit = preprocessor.fit_transform(X_fit)
    Xt_val = preprocessor.transform(X_val)
    Xt_test = preprocessor.transform(X_test)
    Xt_cal = preprocessor.transform(X_cal)
    names = list(preprocessor.named_steps["columns"].get_feature_names_out())
    sources = feature_sources(preprocessor)

    inputs = {"numeric": [], "categorical": []}
    for col in NUMERIC_FEATURES:
        values = pd.to_numeric(df[col], errors="coerce")
        span = float(values.max() - values.min())
        step = choose_step(span)
        inputs["numeric"].append({
            "name": col,
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75)),
            "step": step,
        })
    for col in CATEGORICAL_FEATURES:
        values = df[col].dropna().astype(str)
        options = sorted(values.unique().tolist())
        default = str(values.mode().iloc[0])
        inputs["categorical"].append({
            "name": col,
            "options": options,
            "default": default,
        })

    target_hist, target_edges = np.histogram(y.to_numpy(), bins=40)
    target_centers = (target_edges[:-1] + target_edges[1:]) / 2
    target_histogram = {
        "centers": [float(v) for v in target_centers],
        "counts": [int(v) for v in target_hist],
    }

    models = {m: [] for m in METHODS}
    experiments = []
    scatter_rng = np.random.RandomState(RANDOM_STATE)
    scatter_idx = scatter_rng.choice(
        len(X_test), size=min(SCATTER_SIZE, len(X_test)), replace=False,
    )

    for method in METHODS:
        for k in FEATURE_COUNTS:
            print(f"Training {method} k={k} ...", flush=True)
            start = time.time()

            selected, indices = fit_feature_selector(
                method, k, Xt_fit, y_fit, names,
            )
            reg = LinearRegression()
            reg.fit(Xt_fit[:, indices], y_fit)

            def predict(Xt):
                return reg.predict(Xt[:, indices])

            val_pred = predict(Xt_val)
            val_array = y_val.to_numpy()
            val_residuals = val_array - val_pred
            val_rmse = float(np.sqrt(mean_squared_error(val_array, val_pred)))
            val_mae = float(mean_absolute_error(val_array, val_pred))
            val_r2 = float(r2_score(val_array, val_pred))
            val_se = float(
                np.std(val_residuals ** 2)
                / (2.0 * val_rmse * np.sqrt(len(val_residuals)))
            )

            test_pred = predict(Xt_test)
            test_array = y_test.to_numpy()
            test_rmse = float(np.sqrt(mean_squared_error(test_array, test_pred)))
            test_mae = float(mean_absolute_error(test_array, test_pred))
            test_r2 = float(r2_score(test_array, test_pred))

            cal_pred = predict(Xt_cal)
            cal_residuals = np.abs(y_cal.to_numpy() - cal_pred)
            n_cal = len(cal_residuals)
            rank = int(np.ceil((n_cal + 1) * (1 - ALPHA)))
            q_hat = float(np.inf) if rank > n_cal else float(np.sort(cal_residuals)[rank - 1])

            lower = test_pred - q_hat
            upper = test_pred + q_hat
            coverage = float(np.mean((test_array >= lower) & (test_array <= upper)))
            width = float(np.mean(upper - lower))

            coefficients = ols_coefficients(
                preprocessor,
                {"X": X_fit, "indices": indices},
                sources,
                y_fit,
            )

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
                    "id": int(X_test.index[idx]),
                    "actual": actual,
                    "predicted": predicted,
                    "residual": residual,
                    "abs_error": abs(float(residual)),
                    "pct_error": float(abs(residual) / actual * 100) if actual != 0 else None,
                })

            runtime = time.time() - start
            models[method].append({
                "k": k,
                "n_features": len(selected),
                "features": selected,
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
                "n_features": len(selected),
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

    # One-standard-error rule: prefer the simplest model within one SE of
    # the minimum validation RMSE.
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
    best = min(
        candidates,
        key=lambda r: (
            r["n_features"],
            r["validation_RMSE"],
            METHODS.index(r["method"]),
        ),
    )

    data = {
        "meta": {
            "target": TARGET,
            "target_display": "Price (USD)",
            "task_type": "price",
            "currency": True,
            "probability": False,
            "metric_digits": 0,
            "train_rows": int(len(df)),
            "test_rows": 0,
            "n_base_features": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
            "n_candidate_features": len(names),
            "duplicate_rows": int(df.duplicated().sum()),
            "feature_counts": FEATURE_COUNTS,
            "methods": METHODS,
            "random_state": RANDOM_STATE,
            "alpha": ALPHA,
            "selection_rule": "one_standard_error",
            "best_method": best["method"],
            "best_k": best["k"],
            "min_validation": min_row,
            "split": {
                "fit": int(len(X_fit)),
                "validation": int(len(X_val)),
                "calibration": int(len(X_cal)),
                "test": int(len(X_test)),
            },
            "best_validation": best,
        },
        "baseFeatures": NUMERIC_FEATURES,
        "numericFeatures": NUMERIC_FEATURES,
        "categoricalFeatures": CATEGORICAL_FEATURES,
        "interactions": [],
        "featureStats": {
            row["name"]: {
                "min": row["min"],
                "max": row["max"],
                "median": row["median"],
                "q25": row["q25"],
                "q75": row["q75"],
                "step": row["step"],
            }
            for row in inputs["numeric"]
        },
        "inputs": inputs,
        "targetHistogram": target_histogram,
        "methods": METHODS,
        "models": models,
        "experiments": experiments,
    }

    OUT_PATH.write_text(
        "window.AUTOMOBILE_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")
    print(
        f"Min validation: {min_row['method']} k={min_row['k']} "
        f"val RMSE={min_row['validation_RMSE']:.2f} "
        f"(SE={min_row['validation_SE']:.2f})"
    )
    print(
        f"1-SE best: {best['method']} k={best['k']} | "
        f"val RMSE={best['validation_RMSE']:.2f} "
        f"test RMSE={best['test_RMSE']:.2f} "
        f"test R2={best['test_R2']:.4f}"
    )


if __name__ == "__main__":
    main()
