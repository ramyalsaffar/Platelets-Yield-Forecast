"""
File 7 of 7  -  plotting.py
===========================
Charts for the SBC platelet YIELD project. Five figures:

    1. Feature importance      : standardized Linear coefficients, to show the
                                 precount dominates and body size barely moves
                                 the per-liter rate.
    2. Predicted vs actual rate: on held-out test donors, to show the fit.
    3. Machine drift           : the apheresis machine under-reads more at high
                                 yields (motivation for the project).
    4. Scheduling payoff        : cumulative products vs appointments, model
                                 (book highest-predicted first) vs naive
                                 (random), showing the target is reached with
                                 fewer appointments.
    5. Target sensitivity      : the appointment saving versus the demand-target
                                 level, to show the payoff holds across targets.

Runs after 6- scheduling_policy.py in the shared namespace (cell style); the files
run in order with no cross-imports. Figures default to dpi=200 and take a savepath, the
same convention as the inventory project. simulate_policy / schedule_candidates
must already be defined (file 6).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # no display in the sandbox
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error

# features_df, FEATURES_ADVANCE, make_regressors, TARGET, GROUP, schedule_candidates
# come from earlier files (shared namespace; no cross-imports).

RS = 42
DPI = 200

# Plain-English labels for the raw feature names (used on chart axes).
LABELS = {
    "platelet_precount": "Pre-donation platelet count",
    "avg_yield_rate_prior": "Past average richness",
    "avg_precount_prior": "Past average precount",
    "last_precount": "Last-visit precount",
    "last_yield_rate": "Last-visit richness",
    "tbv_l": "Total blood volume",
    "height_cm": "Height",
    "weight_kg": "Weight",
    "bmi": "Body mass index",
    "is_male": "Male",
    "is_first_visit": "First-time donor",
    "n_prior_donations": "Number of past donations",
    "days_since_last_donation": "Days since last donation",
    "age": "Age",
}


def _fit_holdout():
    """Train Linear on train donors; return held-out (actual, predicted) rate."""
    X, y, g = features_df[FEATURES_ADVANCE], features_df[TARGET], features_df[GROUP]
    tr, te = next(GroupShuffleSplit(1, test_size=0.30, random_state=RS).split(X, y, g))
    model = make_regressors()["Linear"].fit(X.iloc[tr], y.iloc[tr])
    return y.iloc[te].values, model.predict(X.iloc[te]), model


def plot_feature_importance(savepath=None):
    """Standardized Linear coefficients (absolute), largest at top."""
    _, _, model = _fit_holdout()
    coefs = np.abs(model.named_steps["m"].coef_)
    imp = pd.Series(coefs, index=FEATURES_ADVANCE).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([LABELS.get(c, c) for c in imp.index], imp.values, color="#b03a4b")
    ax.set_title("Feature importance (standardized Linear coefficients)")
    ax.set_xlabel("Effect on predicted rate (absolute, standardized)")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=DPI)
    return fig


def plot_pred_vs_actual(savepath=None):
    """Predicted vs actual rate on held-out test donors."""
    actual, pred, _ = _fit_holdout()
    r2, mae = r2_score(actual, pred), mean_absolute_error(actual, pred)
    rng = np.random.default_rng(RS)
    idx = rng.choice(len(actual), size=min(1500, len(actual)), replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual[idx], pred[idx], s=8, alpha=0.3, color="#3a6ea5")
    lim = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lim, lim, "k--", lw=1, label="perfect")
    ax.set_xlabel("Actual rate (x10^11 / liter)")
    ax.set_ylabel("Predicted rate")
    ax.set_title(f"Predicted vs actual, held-out donors (R2={r2:.2f}, MAE={mae:.3f})")
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=DPI)
    return fig


def plot_machine_drift(savepath=None):
    """Machine under-read (accurate minus machine) binned by accurate yield."""
    d = features_df.dropna(subset=["machine_count", "accurate_count"]).copy()
    d["underread"] = d["accurate_count"] - d["machine_count"]
    bins = np.arange(2, 12, 1.0)
    d["bin"] = pd.cut(d["accurate_count"], bins)
    g = d.groupby("bin", observed=True)["underread"].mean()
    centers = [iv.mid for iv in g.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="grey", lw=0.8)
    ax.plot(centers, g.values, "-o", color="#b03a4b")
    ax.set_xlabel("Accurate (true) yield (x10^11)")
    ax.set_ylabel("Machine under-read: accurate minus machine")
    ax.set_title("The machine under-reads more at high yields")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=DPI)
    return fig


def plot_scheduling_curve(savepath=None):
    """Cumulative products vs appointments: model order vs naive (expected)."""
    c = schedule_candidates
    actual = c["actual_products"].values.astype(float)
    pred = c["pred_products"].values.astype(float)
    order = np.argsort(-pred)                       # model books highest-predicted first
    model_cum = np.cumsum(actual[order])
    k = np.arange(1, len(actual) + 1)
    naive_cum = k * actual.mean()                   # expected random-order curve
    target = int(round(actual.sum() * 0.40))
    appts_model = int(np.searchsorted(model_cum, target) + 1)
    appts_naive = int(np.ceil(target / actual.mean()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k, model_cum, color="#2a9d5c", label="model (book best first)")
    ax.plot(k, naive_cum, color="#888888", ls="--", label="naive (random)")
    ax.axhline(target, color="#b03a4b", lw=1, ls=":", label=f"demand target ({target})")
    ax.scatter([appts_model, appts_naive], [target, target], color="black", zorder=5)
    ax.annotate(f"{appts_model}", (appts_model, target), textcoords="offset points",
                xytext=(0, 8), ha="center")
    ax.annotate(f"{appts_naive}", (appts_naive, target), textcoords="offset points",
                xytext=(0, 8), ha="center")
    ax.set_xlabel("Donor appointments booked")
    ax.set_ylabel("Cumulative products collected")
    ax.set_title("Scheduling payoff: target reached with fewer appointments")
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=DPI)
    return fig


def plot_target_sensitivity(savepath=None):
    """Sweep the demand-target level; show the appointment saving holds across it."""
    total = int(schedule_candidates["actual_products"].sum())
    fracs = [0.20, 0.30, 0.40, 0.50, 0.60]
    reductions = [simulate_policy(schedule_candidates, int(round(total * fr)))["reduction_pct"]
                  for fr in fracs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([int(fr * 100) for fr in fracs], reductions, "-o", color="#2a9d5c")
    ax.set_xlabel("Demand target (percent of available products)")
    ax.set_ylabel("Appointment reduction vs naive (percent)")
    ax.set_title("Scheduling payoff is robust across demand targets")
    ax.set_ylim(0, max(reductions) * 1.3 if reductions else 1)
    for x, yv in zip([int(fr * 100) for fr in fracs], reductions):
        ax.annotate(f"{yv:.0f}", (x, yv), textcoords="offset points", xytext=(0, 8), ha="center")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=DPI)
    return fig


def plot_all(outdir="/mnt/user-data/outputs"):
    """Make and save all five figures; return the list of paths."""
    jobs = [("feature_importance.png", plot_feature_importance),
            ("pred_vs_actual.png", plot_pred_vs_actual),
            ("machine_drift.png", plot_machine_drift),
            ("scheduling_payoff.png", plot_scheduling_curve),
            ("target_sensitivity.png", plot_target_sensitivity)]
    paths = []
    for fname, fn in jobs:
        p = f"{outdir}/{fname}"
        fn(savepath=p)
        plt.close("all")
        paths.append(p)
    return paths


# ----------------------------------------------------------------------------
# Run (cell style)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    out = plot_all()
    print("Saved figures:")
    for p in out:
        print(" ", p)
