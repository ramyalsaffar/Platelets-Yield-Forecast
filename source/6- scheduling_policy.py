"""
File 6 of 7  -  scheduling_policy.py
====================================
The payoff layer for the SBC platelet YIELD project: turn the advance yield
prediction into a booking plan and show the gain over the naive schedule.

Runs after 5- tuning.py in the shared namespace (cell style); the files run in
order with no cross-imports between them.

Pipeline of this file:
    1. Train the simple model (Linear) on TRAIN donors only, predict the rate
       for held-out TEST donors (so the numbers are out-of-sample and honest).
    2. Turn each donor's predicted rate into expected WHOLE products:
         capacity   = max safe liters from TBV and weight (citrate-limited for
                      small donors), the same capacity rule used to build the data
         potential  = predicted_rate * capacity
         products   = whole units, rounded DOWN (single >=3.0, double >=6.2,
                      triple >=9.3). This is where body size matters: it sets
                      how much can be safely collected, not the per-liter rate.
    3. Simulate booking against a product target and compare:
         naive : book donors in arbitrary (random) order, as the current system
                 does, with no idea who is high-yield.
         model : book the highest-predicted donors first.
       Metric: how many donor appointments are needed to reach the target. Fewer
       appointments = less chair time and staff, and it stops the failure the
       team described (booking a day full of single-product donors).
    4. Recruitment view: among the donors the model flags as highest-yield, what
       fraction truly give a double-or-better, versus the base rate.

The ground-truth "actual products" a donor can give is derived from their
observed rate (target_rate) times the same capacity, so the only thing the model
adds error to is the rate it predicts.

Output in the namespace:
    schedule_candidates : per-donor table (predicted vs actual products) for plotting.
    scheduling_summary  : the headline before/after numbers.
    simulate_policy      : the booking simulation (used by file 7).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# features_df, FEATURES_ADVANCE, make_regressors, TARGET, GROUP come from earlier
# files (shared namespace; no cross-imports).

RS = 42
SINGLE, DOUBLE, TRIPLE = 3.0, 6.2, 9.3


def citrate_factor(weight_kg):
    """Capacity de-rating for small donors (citrate tolerance), as in file 1."""
    return np.where(weight_kg >= 70, 1.0, np.where(weight_kg >= 55, 0.85, 0.72))


def max_safe_liters(tbv_l, weight_kg):
    """Maximum blood volume that can be safely processed (capacity)."""
    return tbv_l * 0.80 * citrate_factor(weight_kg)


def to_products(potential_yield):
    """Whole transfusable units from a yield (round DOWN at the tier cutoffs)."""
    p = np.asarray(potential_yield)
    return np.where(p >= TRIPLE, 3, np.where(p >= DOUBLE, 2, np.where(p >= SINGLE, 1, 0)))


def build_candidates(df):
    """Train Linear on train donors, predict test donors, build a per-donor table.

    One row per held-out test donor (their most recent donation), with predicted
    and actual whole products.
    """
    X, y, groups = df[FEATURES_ADVANCE], df[TARGET], df[GROUP]
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.30,
                                    random_state=RS).split(X, y, groups))

    model = make_regressors()["Linear"].fit(X.iloc[tr], y.iloc[tr])

    test = df.iloc[te].sort_values("donation_date").groupby("donor_index").tail(1).copy()
    cap = max_safe_liters(test["tbv_l"].values, test["weight_kg"].values)
    test["pred_rate"] = model.predict(test[FEATURES_ADVANCE])
    test["pred_products"] = to_products(test["pred_rate"].values * cap)
    test["actual_products"] = to_products(test[TARGET].values * cap)   # observed rate * capacity
    return test[["donor_index", "pred_rate", "pred_products", "actual_products",
                 "tbv_l", "weight_kg", "platelet_precount"]].reset_index(drop=True)


def simulate_policy(candidates, target_products, n_trials=300, seed=RS):
    """Appointments needed to reach a product target: model order vs naive order.

    Returns a dict with the mean appointments for each and the reduction.
    """
    actual = candidates["actual_products"].values.astype(float)
    pred = candidates["pred_products"].values.astype(float)
    n = len(actual)
    rng = np.random.default_rng(seed)

    def appts_to_target(order):
        """Appointments needed to reach the target when donors are booked in `order`."""
        cum = np.cumsum(actual[order])
        k = int(np.searchsorted(cum, target_products) + 1)
        return min(k, n)

    # Model books highest-predicted first (ties broken randomly, averaged).
    model_appts = []
    for _ in range(n_trials):
        jitter = rng.random(n) * 1e-6
        model_appts.append(appts_to_target(np.argsort(-(pred + jitter))))
    # Naive books in random order.
    naive_appts = [appts_to_target(rng.permutation(n)) for _ in range(n_trials)]

    m, na = float(np.mean(model_appts)), float(np.mean(naive_appts))
    return {"target_products": target_products,
            "appts_model": round(m, 1), "appts_naive": round(na, 1),
            "reduction_pct": round(100 * (na - m) / na, 1)}


def recruitment_precision(candidates, top_frac=0.20):
    """Among the top-predicted donors, the share that truly give a double-or-better."""
    k = max(1, int(len(candidates) * top_frac))
    top = candidates.nlargest(k, "pred_products")
    base = (candidates["actual_products"] >= 2).mean()
    hit = (top["actual_products"] >= 2).mean()
    return {"top_frac": top_frac, "base_rate_double_plus": round(base, 3),
            "top_precision_double_plus": round(hit, 3),
            "lift": round(hit / base, 2) if base else float("nan")}


# ----------------------------------------------------------------------------
# Run (cell style)
# ----------------------------------------------------------------------------
schedule_candidates = build_candidates(features_df)
_total_products = int(schedule_candidates["actual_products"].sum())
_target = int(round(_total_products * 0.40))               # a high-demand day target
scheduling_summary = simulate_policy(schedule_candidates, _target)
_recruit = recruitment_precision(schedule_candidates)

if __name__ == "__main__":
    c = schedule_candidates
    print("=== SCHEDULING / RECRUITMENT (held-out test donors) ===")
    print(f"candidate donors        : {len(c):,}")
    print(f"products available (sum): {_total_products:,}")
    print(f"avg products per donor  : {c['actual_products'].mean():.2f}")
    print(f"predicted vs actual products match: "
          f"{(c['pred_products']==c['actual_products']).mean():.0%} exact")

    print("\n--- Booking to a demand target (fewer appointments is better) ---")
    s = scheduling_summary
    print(f"target products          : {s['target_products']}")
    print(f"appointments, naive      : {s['appts_naive']}")
    print(f"appointments, model      : {s['appts_model']}")
    print(f"reduction                : {s['reduction_pct']}%  "
          "(same products collected, fewer donors booked)")

    print("\n--- Recruitment targeting (top 20% predicted) ---")
    r = _recruit
    print(f"base rate double-or-better : {r['base_rate_double_plus']:.0%}")
    print(f"among top-predicted        : {r['top_precision_double_plus']:.0%}  "
          f"(lift {r['lift']}x)")
    print("\nThis is the payoff: rank by predicted products, send high-yield donors to "
          "high-demand days, and recruit the likely doubles/triples first.")
