"""
File 5 of 7  -  tuning.py
=========================
Hyperparameter tuning for the SBC platelet YIELD project, done the leakage-safe
way, on the PRIMARY advance model (precount available at booking).

Runs after 4- models.py in the shared namespace (cell style); the files run in
order with no cross-imports between them.

Why this file exists separately from file 4:
    File 4 compared models with default settings. Tuning can quietly cheat in
    two ways, which this file avoids:
      1. Donor leakage: the same donor in train and test inflates the score.
         -> every split is BY DONOR (GroupShuffleSplit for the held-out test,
            GroupKFold inside the search).
      2. Selection-on-test leakage: choosing hyperparameters by looking at the
         test score. -> hyperparameters are chosen by cross-validation on the
         TRAIN donors only; the held-out test donors are scored exactly once,
         at the end.

Approach:
    - Split donors 70/30 into train/test (no donor in both).
    - Grid-search each candidate with GroupKFold(3) on the train donors,
      scoring by mean absolute error.
    - Refit the best on all train donors, score once on the held-out test.
    - Report tuned models next to the simple fixed baselines (plain Linear and
      precount-only), so we can see whether tuning or complexity actually helps.

Expectation from file 4: the simple model is already competitive, so tuning the
ensembles should not beat it by much. If so, ship the simple model.

Output in the namespace:
    tuning_results : test-set metrics for every candidate.
    BEST_MODEL     : the refit estimator with the lowest test MAE.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# features_df, FEATURES_ADVANCE, TARGET, GROUP come from earlier files
# (shared namespace; no cross-imports).

RS = 42
SEARCH_FOLDS = 3

# Candidate models and their grids. Every strong model in the file 4 benchmark
# is tuned here, including XGBoost and LightGBM.
_CANDIDATES = {
    "ElasticNet": (
        Pipeline([("s", StandardScaler()), ("m", ElasticNet(random_state=RS))]),
        {"m__alpha": [0.001, 0.01, 0.1], "m__l1_ratio": [0.2, 0.5, 0.8]},
    ),
    "CART": (
        DecisionTreeRegressor(random_state=RS),
        {"max_depth": [4, 6, 8, 12], "min_samples_leaf": [20, 50, 100]},
    ),
    "HistGB": (
        HistGradientBoostingRegressor(random_state=RS),
        {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]},
    ),
    "XGBoost": (
        XGBRegressor(subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                     random_state=RS, verbosity=0),
        {"n_estimators": [200, 400], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]},
    ),
    "LightGBM": (
        LGBMRegressor(subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                      random_state=RS, verbose=-1),
        {"n_estimators": [200, 400], "num_leaves": [15, 31], "learning_rate": [0.05, 0.1]},
    ),
}


def _metrics(y, pred):
    """Return MAE, RMSE, R2 as a rounded dict."""
    return {"MAE": round(mean_absolute_error(y, pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y, pred)), 3),
            "R2": round(r2_score(y, pred), 3)}


def tune_advance_model(df):
    """Tune candidates on train donors, score once on held-out test donors.

    Returns
    -------
    (pandas.DataFrame, estimator)
        A results table (one row per candidate plus fixed baselines) and the
        refit estimator with the lowest test MAE.
    """
    X, y, groups = df[FEATURES_ADVANCE], df[TARGET], df[GROUP]

    # 70/30 split BY DONOR (held-out test donors are never seen during search).
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.30,
                                    random_state=RS).split(X, y, groups))
    Xtr, ytr, gtr = X.iloc[tr], y.iloc[tr], groups.iloc[tr]
    Xte, yte = X.iloc[te], y.iloc[te]

    rows, fitted = [], {}

    # Fixed baselines (no tuning), scored on the same held-out test.
    lin = Pipeline([("s", StandardScaler()), ("m", LinearRegression())]).fit(Xtr, ytr)
    rows.append({"model": "Linear (fixed)", "best_params": "-", **_metrics(yte, lin.predict(Xte))})
    fitted["Linear (fixed)"] = lin

    pre = Pipeline([("s", StandardScaler()), ("m", LinearRegression())]).fit(
        Xtr[["platelet_precount"]], ytr)
    rows.append({"model": "precount-only (fixed)", "best_params": "-",
                 **_metrics(yte, pre.predict(Xte[["platelet_precount"]]))})

    # Tuned candidates: search on TRAIN donors only, then score test once.
    for name, (est, grid) in _CANDIDATES.items():
        gs = GridSearchCV(est, grid, cv=GroupKFold(SEARCH_FOLDS),
                          scoring="neg_mean_absolute_error", n_jobs=1)
        gs.fit(Xtr, ytr, groups=gtr)
        rows.append({"model": name, "best_params": gs.best_params_,
                     **_metrics(yte, gs.best_estimator_.predict(Xte))})
        fitted[name] = gs.best_estimator_

    res = pd.DataFrame(rows).set_index("model")
    best_name = res["MAE"].idxmin()
    return res, fitted.get(best_name, lin), best_name


# ----------------------------------------------------------------------------
# Run (cell style)
# ----------------------------------------------------------------------------
tuning_results, BEST_MODEL, _best_name = tune_advance_model(features_df)

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("=== TUNING on the ADVANCE model (held-out test donors, MAE in x10^11/L) ===\n")
    print(tuning_results[["MAE", "RMSE", "R2", "best_params"]].to_string())
    simple = tuning_results.loc["Linear (fixed)", "MAE"]
    best_mae = tuning_results["MAE"].min()
    print(f"\nlowest test MAE         : {best_mae} ({_best_name})")
    print(f"simple Linear test MAE  : {simple}")
    print(f"gain from tuning/complexity: {round(simple - best_mae, 3)} x10^11/L")
    if simple - best_mae <= 0.01:
        print("=> Tuning/complexity does not meaningfully help. Ship the simple Linear model.")
    else:
        print(f"=> {_best_name} is meaningfully better; consider it.")
