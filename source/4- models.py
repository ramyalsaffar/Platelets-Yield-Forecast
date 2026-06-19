"""
File 4 of 7  -  models.py
=========================
Model benchmark for the SBC platelet YIELD project.

Runs after 3- features.py in the shared namespace (cell style); the files run in
order with no cross-imports between them.

What this file does:
    - Defines the candidate model families, from simple to strong:
        Linear, ElasticNet (penalized), CART (a single tree), Random Forest,
        and HistGradientBoosting, plus XGBoost and LightGBM (required; they match
        the blood-demand machine-learning literature).
    - Benchmarks them on the de-capped target_rate using DONOR-GROUPED
      cross-validation (the same donor is never in train and test), for BOTH
      prediction moments: advance (history only) and day-of (adds the measured
      precount).
    - Compares against honest baselines:
        * mean         : predict the average rate (the null model).
        * precount-only: a one-feature linear model (the "precount drives yield"
                         insight).
        * naive rule   : the current threshold rule (precount >= 250 and
                         weight >= 60 -> expect a double-or-better). This is the
                         real incumbent for the recruitment decision.
    - Adds a short double-or-better (units >= 2) classification benchmark, which
      is what recruitment actually decides, reported with AUC and accuracy.

Honesty note on the machine count: it is an at-the-chair value, not available
when booking, and it predicts the TOTAL yield, not the per-liter rate. So it is
NOT a fair head-to-head for the advance model. We report it only as a reference
(how well it predicts the realized total), to motivate the work, not as a rival
we "beat". The fair targets to beat are the mean, precount-only, and naive rule.

The metric helpers (grouped_cv_regression, grouped_cv_classification) are reused
by file 5 for tuning.

Output in the namespace:
    bench_reg_advance, bench_reg_dayof   : regression result tables.
    bench_clf_advance, bench_clf_dayof   : classification result tables.
    make_regressors, make_classifiers, grouped_cv_* : reused by file 5.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, accuracy_score, recall_score, precision_score)

# XGBoost and LightGBM are required (install them in your environment). They are
# part of the benchmark, matching the blood-demand machine-learning literature.
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# features_df and the FEATURES_*/TARGET/TARGET_CLASS/GROUP names come from the
# previous file (shared namespace; no cross-imports).

N_SPLITS = 4
RS = 42
SINGLE, DOUBLE = 3.0, 6.2


def make_regressors():
    """Return the candidate regression models (linear ones are scaled).

    XGBoost and LightGBM are always included. XGBoost is extreme gradient
    boosting; LightGBM is light gradient boosting machine; both build many small
    trees in sequence and match the blood-demand machine-learning literature.
    """
    return {
        "Linear": Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
        "ElasticNet": Pipeline([("s", StandardScaler()),
                                ("m", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RS))]),
        "CART": DecisionTreeRegressor(max_depth=6, random_state=RS),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=RS),
        "HistGB": HistGradientBoostingRegressor(random_state=RS),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                n_jobs=-1, random_state=RS, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  n_jobs=-1, random_state=RS, verbose=-1),
    }


def make_classifiers():
    """Return the candidate classification models (linear ones are scaled)."""
    return {
        "Logistic": Pipeline([("s", StandardScaler()),
                              ("m", LogisticRegression(max_iter=1000))]),
        "CART": DecisionTreeClassifier(max_depth=6, random_state=RS),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=RS),
        "HistGB": HistGradientBoostingClassifier(random_state=RS),
    }


def grouped_cv_regression(X, y, groups, precount_col):
    """Donor-grouped CV for the rate models plus mean and precount-only baselines.

    Returns a DataFrame of MAE, RMSE, and R2 (out-of-fold).
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    rows = []

    def score(name, pred):
        """Append one row of regression metrics (MAE, RMSE, R2) for a model."""
        rows.append({"model": name,
                     "MAE": round(mean_absolute_error(y, pred), 3),
                     "RMSE": round(np.sqrt(mean_squared_error(y, pred)), 3),
                     "R2": round(r2_score(y, pred), 3)})

    # Baselines first (the bar to beat).
    score("baseline_mean",
          cross_val_predict(DummyRegressor(strategy="mean"), X, y, cv=gkf, groups=groups))
    score("baseline_precount_only",
          cross_val_predict(Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
                            X[[precount_col]], y, cv=gkf, groups=groups))
    for name, model in make_regressors().items():
        score(name, cross_val_predict(model, X, y, cv=gkf, groups=groups))
    return pd.DataFrame(rows).set_index("model")


def grouped_cv_classification(X, y, groups, precount_series, weight_series):
    """Donor-grouped CV for double-or-better, plus the naive-rule baseline.

    Returns a DataFrame of AUC, accuracy, and recall (out-of-fold).
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    rows = []

    def score(name, proba, pred):
        """Append one row of classification metrics (AUC, accuracy, recall)."""
        rows.append({"model": name,
                     "AUC": round(roc_auc_score(y, proba), 3) if proba is not None else np.nan,
                     "accuracy": round(accuracy_score(y, pred), 3),
                     "recall": round(recall_score(y, pred, zero_division=0), 3)})

    # Naive current rule: precount >= 250 and weight >= 60 -> predict double-or-better.
    naive_pred = ((precount_series >= 250) & (weight_series >= 60)).astype(int).values
    score("naive_rule", None, naive_pred)

    for name, model in make_classifiers().items():
        proba = cross_val_predict(model, X, y, cv=gkf, groups=groups, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        score(name, proba, pred)
    return pd.DataFrame(rows).set_index("model")


# ----------------------------------------------------------------------------
# Run (cell style). Heavy benchmarks live under __main__ so importing this file
# (e.g. from file 5 for the helper functions) stays cheap.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    pd.set_option("display.width", 100)
    groups = features_df[GROUP]
    y_rate = features_df[TARGET]
    y_dbl = (features_df[TARGET_CLASS] >= 2).astype(int)        # double-or-better
    weight = features_df["weight_kg"]

    # PRIMARY: advance model WITH the precount (drawn a few days ahead of the appointment).
    bench_reg_advance = grouped_cv_regression(features_df[FEATURES_ADVANCE], y_rate, groups,
                                              precount_col="platelet_precount")
    # FALLBACK: donors with no count on file (history + demographics only).
    bench_reg_fallback = grouped_cv_regression(features_df[FEATURES_NOPRECOUNT], y_rate, groups,
                                               precount_col="avg_precount_prior")
    bench_clf_advance = grouped_cv_classification(features_df[FEATURES_ADVANCE], y_dbl, groups,
                                                 features_df["platelet_precount"], weight)
    bench_clf_fallback = grouped_cv_classification(features_df[FEATURES_NOPRECOUNT], y_dbl, groups,
                                                  features_df["avg_precount_prior"], weight)

    print("=== REGRESSION on target_rate (donor-grouped CV) ===")
    print("\n[ADVANCE (primary): precount available at booking]")
    print(bench_reg_advance)
    print("\n[FALLBACK: no count on file, history + demographics only]")
    print(bench_reg_fallback)

    print("\n=== CLASSIFICATION: double-or-better (units >= 2) ===")
    print("\n[ADVANCE (primary)]")
    print(bench_clf_advance)
    print("\n[FALLBACK]")
    print(bench_clf_fallback)

    mae_machine = mean_absolute_error(features_df["accurate_count"], features_df["machine_count"])
    print(f"\n[reference only] machine count vs accurate TOTAL: MAE {mae_machine:.3f} x10^11")
    print("(not comparable to the rate models above; shown to motivate, not as a rival)")

    best = bench_reg_advance.drop(index=["baseline_mean", "baseline_precount_only"])["MAE"].idxmin()
    print(f"\nLowest advance MAE among models: {best}. "
          "If simple Linear/CART is within ~0.01, ship the simple one (file 5 confirms).")
