"""
File 2 of 7  -  cleaning.py
===========================
Turn the messy, as-recorded donations table (donations_raw, from file 1) into a
clean, analysis-ready table (donations_clean), and report exactly what was fixed.

The cleaning is organized around the two recognized standards:
  - Kahn harmonized framework for electronic health record data, with three
    dimensions: CONFORMANCE (does the format fit the model), COMPLETENESS (are
    values present where expected), and PLAUSIBILITY (do the values make sense).
  - DAMA six dimensions: completeness, validity, consistency, uniqueness,
    timeliness, accuracy. DAMA is the Data Management Association.
Each step is tagged VERIFICATION (check against internal rules) or VALIDATION
(check against an external gold standard: the FDA and AABB donation limits).
AABB is the blood-bank accreditation body; FDA is the Food and Drug Administration.

Design (mirrors the inventory pipeline's cleaner):
  - a column-role dictionary (DATA_DICTIONARY) drives the per-column treatment,
  - the function returns (clean_df, report) with a count for every check,
  - it adds flag columns (row_was_imputed, unit_slip_fixed, units_mismatch) so
    later files know what was touched,
  - it splits DATA ERRORS (fixed or dropped) from REAL SIGNAL (kept), so genuine
    low-yield donations stay in.

Runs after 1- synthetic_yield.py in the shared namespace (cell style); the files
run in order with no cross-imports. It reuses nadler_tbv_liters from file 1.

Inputs (from the namespace):
    donations_raw   : messy table to clean
    donations_truth : clean ground truth, used only to verify recovery
Outputs:
    donations_clean, FEATURE_BLOCKLIST, TARGET_COLS, BENCHMARK_COLS, clean_donations
"""

import numpy as np
import pandas as pd

# --- product sizes and external (validation) limits ---
SINGLE, DOUBLE, TRIPLE = 3.0, 6.2, 9.3      # x10^11 platelets per single/double/triple
PRECOUNT_FLOOR = 150.0                        # FDA: minimum pre-donation count (x10^3/uL)
YIELD_SAFE_MAX = 13.0                         # implausibly high yield ceiling (x10^11)
MIN_GAP_DAYS = 2                              # FDA: at least 48 hours between collections

# Data-sanity ranges: values outside these are entry ERRORS, set to missing.
# These are NOT eligibility rules (the precount floor is applied separately).
RANGES = {"weight_kg": (35.0, 200.0), "height_cm": (120.0, 215.0),
          "platelet_precount": (50.0, 900.0)}

# Data dictionary: for every column, its (meaning, unit, role). The role drives
# how the cleaner treats the column; the meaning and unit make the inputs
# reproducible and map cleanly onto real warehouse (EDW) tables. EDW means
# enterprise data warehouse. Roles: id, context, datetime, categorical, feature,
# derived (recomputed), measured, target_source, benchmark, reconcile, leak,
# status, blocked (kept but never a model feature).
DATA_DICTIONARY = {
    "procedure_index":          ("Unique id for one donation event",                 "id",            "id"),
    "donor_index":              ("Unique id for one donor",                          "id",            "id"),
    "procedure_number":         ("Which visit this is for the donor (1 = first)",    "count",         "context"),
    "donation_date":            ("Calendar date of the donation",                    "date",          "datetime"),
    "days_since_last_donation": ("Days since the donor's previous donation",         "days",          "derived"),
    "gender":                   ("Donor sex, M or F",                                "category",      "categorical"),
    "birthdate":                ("Donor date of birth",                              "date",          "datetime"),
    "age":                      ("Donor age on the donation date",                   "years",         "derived"),
    "height_cm":                ("Donor height",                                     "centimeters",   "feature"),
    "weight_kg":                ("Donor weight",                                     "kilograms",     "feature"),
    "tbv_l":                    ("Total blood volume (Nadler formula)",              "liters",        "derived"),
    "platelet_precount":        ("Pre-donation platelet count",                      "x10^3/uL",      "feature"),
    "donor_type":               ("first_time or regular",                           "category",      "categorical"),
    "show_rate":                ("Donor's historical show-up probability",           "fraction 0-1",  "blocked"),
    "demand_level":             ("That day's demand tier (drove the cap)",           "category",      "leak"),
    "status":                   ("completed, low_yield, or deferred",                "category",      "status"),
    "target_tier":              ("Tier targeted that day (1/2/3)",                   "units",         "leak"),
    "liters_processed":         ("Blood volume processed during apheresis",          "liters",        "measured"),
    "accurate_count":           ("Gold-standard post-count of platelets collected",  "x10^11",        "target_source"),
    "machine_count":            ("Apheresis machine's own yield estimate",           "x10^11",        "benchmark"),
    "units":                    ("Whole transfusable products (0-3)",                "count",         "reconcile"),
    "true_richness":            ("Generator-only true per-liter richness",           "x10^11/L",      "leak"),
}

# Column roles for leakage control (used to pick model features later).
TARGET_COLS = ["target_rate", "accurate_count", "units"]
BENCHMARK_COLS = ["machine_count"]
FLAG_COLS = ["row_was_imputed", "unit_slip_fixed", "units_mismatch"]
ID_CONTEXT_COLS = ["donor_index", "procedure_index", "procedure_number",
                   "donation_date", "donor_type"] + FLAG_COLS
LEAK_COLS = ["liters_processed", "target_tier", "demand_level", "true_richness",
             "status", "birthdate"]
FEATURE_BLOCKLIST = TARGET_COLS + BENCHMARK_COLS + ID_CONTEXT_COLS + ["show_rate"]


# ----------------------------------------------------------------------------
# Small value-level helpers (conformance fixes)
# ----------------------------------------------------------------------------
def _to_num(x):
    """Parse a possibly-text number ('72 kg', '1,250', '~250 ') to float, else NaN."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = (str(x).replace(",", "").replace("kg", "").replace("KG", "")
         .replace("~", "").strip())
    try:
        return float(s)
    except ValueError:
        return np.nan


def _norm_gender(x):
    """Map any gender spelling/case/whitespace to 'M' or 'F', else NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    return "M" if s.startswith("M") else ("F" if s.startswith("F") else np.nan)


def _norm_donor_type(x):
    """Map donor_type spellings to 'first_time' or 'regular', else NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    return "first_time" if "first" in s else ("regular" if "regular" in s else np.nan)


_DATE_FORMATS = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%m-%d-%Y",
                 "%d-%b-%Y", "%B %d, %Y"]


def _parse_date(x):
    """Parse a date in any of several recorded formats; only truly bad text -> NaT."""
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x
    s = str(x).strip()
    if not s or s.lower() in {"n/a", "unknown", "none", "nan"}:
        return pd.NaT
    for fmt in _DATE_FORMATS:
        try:
            return pd.Timestamp(pd.to_datetime(s, format=fmt))
        except (ValueError, TypeError):
            continue
    try:
        return pd.Timestamp(pd.to_datetime(s, errors="raise"))
    except Exception:
        return pd.NaT


# ----------------------------------------------------------------------------
# Production quality gate (data-validation): assert the cleaned table meets
# every invariant the downstream files rely on. Raises on any violation.
# ----------------------------------------------------------------------------
def _quality_gate(out):
    """Check the cleaned table against hard invariants; raise if any fail."""
    checks = {
        "no missing in numeric features":
            int(out[["weight_kg", "height_cm", "platelet_precount", "tbv_l", "age"]]
                .isna().sum().sum()) == 0,
        "gender only M/F": set(out["gender"].unique()).issubset({"M", "F"}),
        "precount at or above FDA floor": bool((out["platelet_precount"] >= PRECOUNT_FLOOR).all()),
        "accurate_count positive": bool((out["accurate_count"] > 0).all()),
        "target_rate positive and finite":
            bool(np.isfinite(out["target_rate"]).all() and (out["target_rate"] > 0).all()),
        "units in 0..3": set(np.unique(out["units"]).tolist()).issubset({0, 1, 2, 3}),
        "no leaky columns": not any(c in out.columns for c in LEAK_COLS),
    }
    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, f"QUALITY GATE FAILED: {failed}"
    return list(checks.keys())


# ----------------------------------------------------------------------------
# Main cleaner
# ----------------------------------------------------------------------------
def clean_donations(df, cap_unsafe=False, verbose=True):
    """Clean the raw donations table; return (clean_df, report)."""
    out = df.copy()
    rep = {"rows_in": len(out)}
    out["row_was_imputed"] = False
    out["unit_slip_fixed"] = False
    out["units_mismatch"] = False

    # === Step 1  CONFORMANCE (verification): fix types, dates, categories ===
    out["gender"] = out["gender"].map(_norm_gender)
    out["donor_type"] = out["donor_type"].map(_norm_donor_type)
    out["donation_date"] = out["donation_date"].map(_parse_date)
    out["birthdate"] = pd.to_datetime(out["birthdate"], errors="coerce")
    for col in ["weight_kg", "height_cm", "platelet_precount", "age",
                "liters_processed", "accurate_count", "machine_count"]:
        out[col] = out[col].map(_to_num)

    # === Step 2  UNIQUENESS (verification): drop duplicates and overlays ===
    rep["exact_duplicate_rows"] = int(out.duplicated().sum())
    before = len(out)
    out = out.drop_duplicates(subset="procedure_index", keep="first").reset_index(drop=True)
    rep["duplicate_or_overlay_dropped"] = before - len(out)

    # === Step 3  ACCURACY (validation): repair clear unit slips ===
    # Height: 48-90 "cm" is impossible for an adult, so it was recorded in inches.
    h = out["height_cm"]
    inch = h.notna() & (h >= 48) & (h <= 90)
    out.loc[inch, "height_cm"] = (out.loc[inch, "height_cm"] * 2.54).round(1)
    # Weight: treat as pounds and convert to kilograms ONLY when the kilogram
    # reading is clearly impossible -- body-mass-index above 55 for the recorded
    # height, or, when the height is not usable, above a 180 kg donor ceiling --
    # AND the pounds reading is plausible. The ambiguous overlap (a value that is
    # plausible as either unit) cannot be resolved without a recorded unit, so it
    # is left as-is; extreme values are still caught by the range check below.
    w = out["weight_kg"]
    h_ok = out["height_cm"].between(120, 230)          # plausible height to trust BMI
    hm = out["height_cm"] / 100.0
    bmi_kg = w / (hm ** 2)
    bmi_lbs = (w / 2.20462) / (hm ** 2)
    lbs = w.notna() & (
        (h_ok & (bmi_kg > 55) & bmi_lbs.between(16, 45)) |
        (~h_ok & (w > 180))
    )
    out.loc[lbs, "weight_kg"] = (out.loc[lbs, "weight_kg"] / 2.20462).round(1)
    out["unit_slip_fixed"] = lbs | inch
    rep["unit_slips_fixed"] = int((lbs | inch).sum())

    # === Step 4  PLAUSIBILITY atemporal (verification): impossible -> missing ===
    rep["impossible_to_missing"] = {}
    for col, (lo, hi) in RANGES.items():
        bad = out[col].notna() & ((out[col] < lo) | (out[col] > hi))
        out.loc[bad, col] = np.nan
        rep["impossible_to_missing"][col] = int(bad.sum())

    # === Step 5  PLAUSIBILITY temporal (validation): bad timing -> drop ===
    today = pd.Timestamp.today().normalize()
    bad_date = out["donation_date"].isna()
    before_birth = out["donation_date"].notna() & (out["donation_date"] < out["birthdate"])
    future = out["donation_date"] > today
    drop_t = bad_date | before_birth | future
    rep["dropped_missing_or_unparseable_date"] = int(bad_date.sum())
    rep["dropped_impossible_date"] = int((before_birth | future).sum())
    out = out[~drop_t].reset_index(drop=True)
    # enforce the 48-hour minimum gap (drop the closer repeat within a donor)
    out = out.sort_values(["donor_index", "donation_date"]).reset_index(drop=True)
    gap = out.groupby("donor_index")["donation_date"].diff().dt.days
    too_close = gap.notna() & (gap < MIN_GAP_DAYS)
    rep["too_close_under_48h_dropped"] = int(too_close.sum())
    out = out[~too_close].reset_index(drop=True)

    # === Step 6  COMPLETENESS (verification): impute safe gaps, flag, recompute ===
    imp = pd.Series(False, index=out.index)
    for col in ["weight_kg", "height_cm", "platelet_precount"]:
        miss = out[col].isna()
        donor_med = out.groupby("donor_index")[col].transform("median")
        fill = donor_med.fillna(out[col].median())
        out.loc[miss, col] = fill[miss]
        imp |= miss
    gmiss = out["gender"].isna()
    if gmiss.any():
        out.loc[gmiss, "gender"] = out["gender"].mode().iloc[0]
        imp |= gmiss
    dtmiss = out["donor_type"].isna()
    if dtmiss.any():
        out.loc[dtmiss, "donor_type"] = "regular"
        imp |= dtmiss
    out["row_was_imputed"] = imp
    rep["rows_imputed"] = int(imp.sum())
    # recompute derived fields from the cleaned inputs (computational conformance)
    out["age"] = ((out["donation_date"] - out["birthdate"]).dt.days // 365).astype(int)
    out["tbv_l"] = [round(nadler_tbv_liters(hh, ww, gg), 2)
                    for hh, ww, gg in zip(out["height_cm"], out["weight_kg"], out["gender"])]

    # === Step 7  ACCURACY cross-field (verification): reconcile units vs count ===
    recomputed = np.where(out["accurate_count"] >= TRIPLE, 3,
                  np.where(out["accurate_count"] >= DOUBLE, 2,
                  np.where(out["accurate_count"] >= SINGLE, 1, 0)))
    out["units_mismatch"] = out["units"].astype(float).fillna(-1).astype(int) != recomputed
    rep["units_recorded_vs_count_mismatch"] = int(out["units_mismatch"].sum())
    out["units"] = recomputed                       # trust the measured count

    # === Step 8  PROJECT cleaning: keep real donations, build target, de-leak ===
    before = len(out)
    out = out[out["status"].isin(["completed", "low_yield"])].copy()
    rep["dropped_deferred"] = before - len(out)
    # safety/sanity (validation against FDA limits)
    safe = (out["platelet_precount"] >= PRECOUNT_FLOOR) & (out["liters_processed"] > 0) \
        & (out["accurate_count"] > 0)
    if cap_unsafe:
        rep["capped_high_yield"] = int((out["accurate_count"] > YIELD_SAFE_MAX).sum())
        out.loc[out["accurate_count"] > YIELD_SAFE_MAX, "accurate_count"] = YIELD_SAFE_MAX
    else:
        safe &= out["accurate_count"] <= YIELD_SAFE_MAX
    rep["dropped_unsafe_or_invalid"] = int((~safe).sum())
    out = out[safe].reset_index(drop=True)
    # de-capped richness target, then drop leakage columns
    out["target_rate"] = (out["accurate_count"] / out["liters_processed"]).round(3)
    out["units"] = np.where(out["accurate_count"] >= TRIPLE, 3,
                    np.where(out["accurate_count"] >= DOUBLE, 2,
                    np.where(out["accurate_count"] >= SINGLE, 1, 0)))
    out = out.drop(columns=[c for c in LEAK_COLS if c in out.columns])
    rep["rows_out"] = len(out)
    rep["leaky_cols_remaining"] = [c for c in LEAK_COLS if c in out.columns]
    rep["quality_gate"] = f"passed ({len(_quality_gate(out))} checks)"

    if verbose:
        print("=== CLEANING REPORT (Kahn + DAMA) ===")
        for k, v in rep.items():
            print(f"{k:34}: {v}")
    return out, rep


# ----------------------------------------------------------------------------
# Run (notebook-cell style)
# ----------------------------------------------------------------------------
donations_clean, _clean_report = clean_donations(donations_raw, verbose=False)

if __name__ == "__main__":
    donations_clean, rep = clean_donations(donations_raw, verbose=True)
    d = donations_clean
    print("\n=== RESULT ===")
    print(f"clean rows                        : {len(d):,}")
    print(f"gender values                     : {sorted(d['gender'].unique())}")
    print(f"all numerics numeric              : "
          f"{all(pd.api.types.is_numeric_dtype(d[c]) for c in ['weight_kg','height_cm','platelet_precount','age','tbv_l'])}")
    print(f"any missing in features           : "
          f"{int(d[['weight_kg','height_cm','platelet_precount','tbv_l','age']].isna().sum().sum())}")
    print(f"leaky cols remaining              : {rep['leaky_cols_remaining'] or 'none'}")
    feats = [c for c in d.columns if c not in FEATURE_BLOCKLIST]
    print(f"feature-eligible cols             : {feats}")

    # --- Ground-truth recovery proof: compare recovered fields to donations_truth ---
    truth = donations_truth[donations_truth["status"].isin(["completed", "low_yield"])]
    m = d.merge(truth[["procedure_index", "gender", "platelet_precount", "weight_kg"]],
                on="procedure_index", suffixes=("", "_true"))
    g_ok = (m["gender"] == m["gender_true"]).mean()
    # compare only rows that were not imputed (imputation changes the value by design)
    keep = ~m["row_was_imputed"]
    p_err = (m.loc[keep, "platelet_precount"] - m.loc[keep, "platelet_precount_true"]).abs().mean()
    w_err = (m.loc[keep, "weight_kg"] - m.loc[keep, "weight_kg_true"]).abs().mean()
    print("\n=== GROUND-TRUTH RECOVERY ===")
    print(f"rows matched to truth             : {len(m):,}")
    print(f"gender recovered exactly          : {g_ok:.1%}")
    print(f"precount mean abs error (non-imp) : {p_err:.3f}")
    print(f"weight  mean abs error (non-imp)  : {w_err:.3f}")
