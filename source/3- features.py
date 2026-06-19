"""
File 3 of 7  -  features.py
===========================
Feature engineering for the SBC platelet YIELD project.

Runs after 2- cleaning.py in the shared namespace (cell style); the files run
in order with no cross-imports between them.

It defines build_features (needed by files 4 and 5) and produces two feature
sets, matching the two prediction moments we identified:

    FEATURES_ADVANCE   : the primary model. The precount is usually drawn a few
                         days before the appointment and platelet counts are
                         stable, so it is available at booking and is included
                         here, with demographics and history.
    FEATURES_NOPRECOUNT: a fallback for donors with no count on file, using only
                         demographics and history.

LEAKAGE RULE (critical): every history feature is built from STRICTLY PRIOR
donations only, per donor, in date order. We use shift(1) and prior-only
cumulative sums, never the current row, so the model cannot peek at the future.
First-visit gaps are filled with fixed domain constants (not data-derived
statistics), which keeps the fill itself leakage-free; the is_first_visit flag
lets the model treat those rows differently.

Note on machine type: real SBC data should add the apheresis machine and
software version as a categorical feature (it shifts yield). It is not in the
synthetic table, so it is intentionally omitted here rather than fabricated.

Output in the namespace:
    features_df : clean table plus engineered features.
    FEATURES_ADVANCE, FEATURES_NOPRECOUNT, TARGET, TARGET_CLASS, GROUP
"""

import numpy as np
import pandas as pd

# donations_clean comes from the previous file (shared namespace; no cross-imports).

# Leakage-free fallback values for a donor's first visit (no history yet).
FILL_PRECOUNT = 255.0      # typical population precount (x10^3/uL)
FILL_RATE = 1.6            # typical richness (x10^11 per liter processed)
FILL_DAYS = 365.0          # "no recent prior donation"


def build_features(df):
    """Engineer demographic and leakage-safe lagged-history features.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned donation table from file 2 (must include target_rate).

    Returns
    -------
    pandas.DataFrame
        The input rows plus engineered feature columns, sorted by donor and
        date. History features use only prior donations (no future leakage).
    """
    f = df.sort_values(["donor_index", "donation_date", "procedure_number"]).copy()
    g = f.groupby("donor_index", sort=False)

    # --- Demographics (stable, always known) ---
    f["is_male"] = (f["gender"] == "M").astype(int)
    f["bmi"] = (f["weight_kg"] / (f["height_cm"] / 100.0) ** 2).round(1)

    # --- Experience and lagged history (PRIOR donations only) ---
    f["n_prior_donations"] = g.cumcount()                      # 0 on first visit
    f["is_first_visit"] = (f["n_prior_donations"] == 0).astype(int)

    # Days since the donor's last COMPLETED donation. Platelet recovery depends
    # on the last collection, not on a deferred visit, so recompute it here from
    # the clean donation dates. This also keeps it consistent with is_first_visit
    # (a first completed donation has no prior date, so it is filled below).
    f["days_since_last_donation"] = g["donation_date"].diff().dt.days

    f["last_precount"] = g["platelet_precount"].shift(1)
    f["last_yield_rate"] = g["target_rate"].shift(1)

    # Prior-only running averages: cumulative sum minus the current row,
    # divided by the count of prior rows (NaN on the first visit).
    n_prior = f["n_prior_donations"].replace(0, np.nan)
    f["avg_precount_prior"] = ((g["platelet_precount"].cumsum() - f["platelet_precount"])
                               / n_prior).round(1)
    f["avg_yield_rate_prior"] = ((g["target_rate"].cumsum() - f["target_rate"])
                                 / n_prior).round(3)

    # --- Fill first-visit gaps with fixed constants (no data leakage) ---
    f["last_precount"] = f["last_precount"].fillna(FILL_PRECOUNT)
    f["avg_precount_prior"] = f["avg_precount_prior"].fillna(FILL_PRECOUNT)
    f["last_yield_rate"] = f["last_yield_rate"].fillna(FILL_RATE)
    f["avg_yield_rate_prior"] = f["avg_yield_rate_prior"].fillna(FILL_RATE)
    f["days_since_last_donation"] = f["days_since_last_donation"].fillna(FILL_DAYS)

    return f.reset_index(drop=True)


# Column roles for files 4 and 5.
_DEMOGRAPHIC = ["is_male", "age", "height_cm", "weight_kg", "tbv_l", "bmi"]
_HISTORY = ["n_prior_donations", "is_first_visit", "last_precount", "last_yield_rate",
            "avg_precount_prior", "avg_yield_rate_prior", "days_since_last_donation"]

# The precount is usually drawn a few days BEFORE the booked appointment, and
# platelet counts are stable over that gap, so it is available at booking time.
# It is therefore the primary ADVANCE feature. A precount-free fallback covers
# donors who have no count on file at all.
FEATURES_ADVANCE = _DEMOGRAPHIC + _HISTORY + ["platelet_precount"]  # advance precount available
FEATURES_NOPRECOUNT = _DEMOGRAPHIC + _HISTORY                       # fallback: no count on file
TARGET = "target_rate"          # regression target (de-capped richness)
TARGET_CLASS = "units"          # 1/2/3 tier target for the classification version
GROUP = "donor_index"           # for donor-grouped cross-validation in file 5


# ----------------------------------------------------------------------------
# Run (cell style)
# ----------------------------------------------------------------------------
features_df = build_features(donations_clean)

if __name__ == "__main__":
    f = features_df
    print("=== FEATURES BUILT ===")
    print(f"rows                    : {len(f):,}")
    print(f"advance features  ({len(FEATURES_ADVANCE):>2}) : {FEATURES_ADVANCE}")
    print(f"fallback features ({len(FEATURES_NOPRECOUNT):>2}) : same minus 'platelet_precount'")
    print(f"first visits            : {int(f['is_first_visit'].sum()):,} "
          f"({f['is_first_visit'].mean():.0%})")

    print("\n=== NO MISSING VALUES IN FEATURES ===")
    print(f"advance set NaN count   : {int(f[FEATURES_ADVANCE].isna().sum().sum())}")
    print(f"fallback set NaN count  : {int(f[FEATURES_NOPRECOUNT].isna().sum().sum())}")
    print(f"precount IS in advance set : {'platelet_precount' in FEATURES_ADVANCE}")

    print("\n=== LEAKAGE SPOT CHECK (a repeat donor) ===")
    rid = f[f["n_prior_donations"] >= 3]["donor_index"].iloc[0]
    sub = f[f["donor_index"] == rid].sort_values("donation_date")
    expect = sub["platelet_precount"].shift(1).iloc[1:4].round(0).tolist()
    got = sub["last_precount"].iloc[1:4].round(0).tolist()
    print(f"donor {rid}: last_precount equals previous row's precount -> {expect == got}")

    print("\n=== SIGNAL CHECK (corr with target_rate) ===")
    print(f"day-of precount         : {f['platelet_precount'].corr(f[TARGET]):.2f}")
    print(f"last precount (history) : {f['last_precount'].corr(f[TARGET]):.2f}")
    print(f"avg precount (history)  : {f['avg_precount_prior'].corr(f[TARGET]):.2f}")
    print(f"tbv                     : {f['tbv_l'].corr(f[TARGET]):.2f}")
