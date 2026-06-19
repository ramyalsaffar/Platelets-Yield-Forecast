"""
File 1 of 7  -  synthetic_yield.py
==================================
Synthetic donor and donation generator for the Stanford Blood Center (SBC)
platelet YIELD prediction project.

Run order (these files act as notebook cells sharing ONE namespace, like the
inventory project; there are no cross-imports):
    1. synthetic_yield.py   <-- this file
    2. cleaning.py
    3. features.py
    4. models.py
    5. tuning.py
    6. scheduling_policy.py
    7. plotting.py

Design (all approved):
    - ~3,000 unique donors over a 3-year window, ~30k-40k donations.
    - ~40% one-time/occasional donors, ~60% regulars (up to 24 donations/year).
    - ~70% male; age mean ~50, SD ~16, range 18-75 (platelet donors skew older).
    - Height/weight drawn by sex; TBV (Total Blood Volume) from Nadler's formula.
    - Per-donor stable baseline precount (mean ~255, range ~150-500).

Biology is generated FIRST, then yield is built from it, so pre-donation
platelet count ("precount") and body size are the genuine drivers:
    richness (platelets collected per liter processed) = precount_per_L * CE * recruitment
    yield = richness * liters_processed
This makes the rate (accurate_count / liters) recover a donor's true richness
even when the recorded total was CAPPED by the tier that was asked for.

Real-world noise/scenarios baked in:
    - Day-to-day precount wobble around the donor's baseline (small; counts are stable),
      occasional illness dips, and a transient dip if the donor returns very soon.
    - Capping: on low-demand days only a single is asked, even from strong donors.
    - Machine count is biased and drifts MORE at high yields (under-reads); the
      separate accurate count is the true label.
    - Deferrals when the day-of precount falls below the 150 (x10^3/uL) floor.
    - Small, high-count donors citrate-limited out of a triple.
    - Seasonal demand and per-donor show-rate (carried for the scheduling file).

Column map to the EDW (Enterprise Data Warehouse) tables you saw on-site:
    donor_index, procedure_index, procedure_number  -> PROCEDURERUN
    height_cm, weight_kg, platelet_precount          -> DONORVITALS
    birthdate, gender                                -> DONOR
    accurate_count, machine_count, liters_processed  -> the second database
Note: liters_processed, machine_count and the during-procedure values are
LABEL-only; they must never be used as model inputs (handled in cleaning.py).
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Config (tweak here; everything is seeded for reproducibility)
# ----------------------------------------------------------------------------
SEED = 42
DEFECT_RATE = 1.0   # master knob: scales every injected raw-data defect rate below.
                    # File 2 detects and repairs or drops these. Set 0.0 for clean data.
N_DONORS = 3000
START_DATE = pd.Timestamp("2022-01-01")
N_DAYS = 3 * 365                      # 3-year window
FRAC_MALE = 0.70
FRAC_FIRST_TIME = 0.40               # one-time / occasional donors

# Product tiers in units of x10^11 platelets (US apheresis cutoffs)
SINGLE, DOUBLE, TRIPLE = 3.0, 6.2, 9.3
PRECOUNT_FLOOR = 150.0               # x10^3/uL; below this the donor is deferred


def nadler_tbv_liters(height_cm, weight_kg, gender):
    """Total Blood Volume in liters via Nadler's formula.

    Parameters
    ----------
    height_cm : float
        Donor height in centimeters.
    weight_kg : float
        Donor weight in kilograms.
    gender : str
        'M' or 'F'.

    Returns
    -------
    float
        Estimated total blood volume in liters.
    """
    h = height_cm / 100.0
    if gender == "M":
        return 0.3669 * h ** 3 + 0.03219 * weight_kg + 0.6041
    return 0.3561 * h ** 3 + 0.03308 * weight_kg + 0.1833


def _make_donors(rng):
    """Create the per-donor table (demographics, baseline biology, behavior).

    Returns
    -------
    pandas.DataFrame
        One row per donor with stable attributes used to generate visits.
    """
    n = N_DONORS
    gender = np.where(rng.random(n) < FRAC_MALE, "M", "F")

    # Age (skews older), clipped to the eligible range.
    age = np.clip(rng.normal(50, 16, n), 18, 75).round().astype(int)
    birthdate = START_DATE - pd.to_timedelta(age * 365.25, unit="D")

    # Height / weight by sex, then clip to plausible, eligible values.
    male = gender == "M"
    height = np.where(male, rng.normal(176, 7, n), rng.normal(162, 6.5, n))
    height = np.clip(height, 150, 200)
    weight = np.where(male, rng.normal(85, 14, n), rng.normal(71, 14, n))
    weight = np.clip(weight, 50, 150)               # 50 kg eligibility floor

    tbv = np.array([nadler_tbv_liters(h, w, g)
                    for h, w, g in zip(height, weight, gender)])

    # Stable personal baseline precount (x10^3/uL), mild spread, clipped.
    baseline_precount = np.clip(rng.normal(258, 55, n), 150, 520)

    # Donor-level recruitment offset. The recruitment factor (platelet mobilization
    # during apheresis) is modeled per-visit as precount-dependent in the loop below;
    # this is each donor's small, stable individual deviation from that curve.
    recruitment_offset = np.clip(rng.normal(0.0, 0.03, n), -0.08, 0.08)

    # Behavior: type, annual frequency (regulars), and show-rate.
    is_first_time = rng.random(n) < FRAC_FIRST_TIME
    annual_freq = np.where(
        is_first_time,
        0.0,
        np.clip(rng.gamma(shape=2.0, scale=3.0, size=n), 1.0, 24.0),
    )
    show_rate = np.clip(rng.normal(0.90, 0.07, n), 0.60, 0.99)

    return pd.DataFrame({
        "donor_index": np.arange(n),
        "gender": gender,
        "birthdate": birthdate,
        "age_at_start": age,
        "height_cm": height.round(1),
        "weight_kg": weight.round(1),
        "tbv_l": tbv.round(2),
        "baseline_precount": baseline_precount.round(0),
        "recruitment_offset": recruitment_offset,
        "is_first_time": is_first_time,
        "annual_freq": annual_freq,
        "show_rate": show_rate.round(3),
    })


def _daily_demand(rng):
    """Daily platelet demand level over the window, split into low/med/high.

    Returns
    -------
    numpy.ndarray
        Array of length N_DAYS with values 'low', 'med', or 'high'.
    """
    t = np.arange(N_DAYS)
    seasonal = 12 * np.sin(2 * np.pi * t / 365.25)          # annual swing
    weekday = (pd.to_datetime(START_DATE) + pd.to_timedelta(t, unit="D")).weekday
    weekend = np.where(weekday >= 5, -8, 0)                 # quieter weekends
    demand = 100 + seasonal + weekend + rng.normal(0, 6, N_DAYS)
    lo, hi = np.quantile(demand, [1 / 3, 2 / 3])
    return np.where(demand <= lo, "low", np.where(demand <= hi, "med", "high"))


def _visit_days(donor, rng):
    """Day-offsets (0-based) of a donor's visits within the window.

    First-time/occasional donors get 1-2 visits; regulars get visits spaced by
    gaps whose mean matches their annual frequency (so the 24/year cap holds in
    expectation, and gaps stay >= 2 days per FDA rules).
    """
    if donor.is_first_time:
        k = 1 if rng.random() < 0.7 else 2
        days = sorted(rng.integers(0, N_DAYS, size=k).tolist())
        if k == 2 and days[1] - days[0] < 2:
            days[1] = min(days[0] + 2, N_DAYS - 1)
        return days

    mean_gap = 365.0 / donor.annual_freq
    day = int(rng.integers(0, 60))                          # first visit early-ish
    out = []
    while day < N_DAYS:
        # Enforce the FDA cap of <= 24 donations per rolling 365 days.
        while sum(1 for d in out if d > day - 365) >= 24:
            day += 1
            if day >= N_DAYS:
                return out
        out.append(day)
        gap = max(2, int(rng.gamma(shape=4.0, scale=mean_gap / 4.0)))
        day += gap
    return out


def generate_sbc_yield_data(seed=SEED):
    """Generate the full synthetic donation-level table.

    Parameters
    ----------
    seed : int
        Seed for the random generator (reproducible output).

    Returns
    -------
    pandas.DataFrame
        One row per donation attempt (completed or deferred), with donor
        demographics, day-of biology, the recorded counts, and bookkeeping
        columns used by later files.
    """
    rng = np.random.default_rng(seed)
    donors = _make_donors(rng)
    demand = _daily_demand(rng)

    rows = []
    proc_id = 0
    for donor in donors.itertuples(index=False):
        day_offsets = _visit_days(donor, rng)
        last_day = None
        for visit_no, day in enumerate(day_offsets, start=1):
            proc_id += 1
            date = START_DATE + pd.Timedelta(days=int(day))
            days_since = np.nan if last_day is None else int(day - last_day)
            last_day = day

            # ---- Day-of precount: baseline + small wobble + dips ----
            precount = donor.baseline_precount * (1 + rng.normal(0, 0.06))
            if not np.isnan(days_since) and days_since < 14:      # incomplete recovery
                precount *= 0.90 + 0.10 * (days_since / 14.0)
            if rng.random() < 0.03:                                # occasional illness
                precount *= rng.uniform(0.55, 0.80)
            precount = round(precount, 0)

            base = dict(
                procedure_index=proc_id,
                donor_index=donor.donor_index,
                procedure_number=visit_no,
                donation_date=date,
                days_since_last_donation=days_since,
                gender=donor.gender,
                birthdate=donor.birthdate,
                age=int((date - donor.birthdate).days // 365),
                height_cm=donor.height_cm,
                weight_kg=donor.weight_kg,
                tbv_l=donor.tbv_l,
                platelet_precount=precount,
                donor_type="first_time" if donor.is_first_time else "regular",
                show_rate=donor.show_rate,
                demand_level=demand[day],
            )

            # ---- Deferral if below the regulatory floor ----
            if precount < PRECOUNT_FLOOR:
                rows.append({**base, "status": "deferred", "target_tier": 0,
                             "liters_processed": np.nan, "accurate_count": np.nan,
                             "machine_count": np.nan, "units": 0,
                             "true_richness": np.nan})
                continue

            # ---- Biology: richness (per-liter) and safe capacity ----
            ce = float(np.clip(rng.normal(0.55, 0.04), 0.45, 0.65))
            # Recruitment factor >1, and HIGHER at low precounts (it declines as the
            # pre-donation count rises). Centered on the population mean precount
            # (~258) so the average factor stays ~1.18, plus donor offset and noise.
            rf_curve = 1.18 - 0.06 * (precount - 258.0) / 100.0
            recruitment = float(np.clip(rf_curve + donor.recruitment_offset
                                        + rng.normal(0, 0.04), 1.0, 1.45))
            precount_per_L = precount / 100.0                      # -> x10^11 per liter
            richness = precount_per_L * ce * recruitment                # x10^11 per liter processed

            citrate_factor = 1.0 if donor.weight_kg >= 70 else (
                0.85 if donor.weight_kg >= 55 else 0.72)           # small donors limited
            max_safe_liters = donor.tbv_l * 0.80 * citrate_factor
            potential = richness * max_safe_liters                 # max safe yield this visit

            cap_tier = 3 if potential >= TRIPLE else (2 if potential >= DOUBLE
                                                      else (1 if potential >= SINGLE else 0))

            # ---- Naive current rule, then capped by the day's demand ----
            could = 3 if (precount >= 300 and donor.weight_kg >= 70) else (
                2 if (precount >= 250 and donor.weight_kg >= 60) else 1)
            demand_cap = {"low": 1, "med": 2, "high": 3}[demand[day]]
            tier = max(0, min(could, demand_cap, cap_tier))
            if tier == 0:                                          # can't make a single
                rows.append({**base, "status": "low_yield", "target_tier": 0,
                             "liters_processed": round(potential / richness, 2),
                             "accurate_count": round(potential, 2),
                             "machine_count": round(potential * 0.99, 2),
                             "units": 0, "true_richness": round(richness, 3)})
                continue

            # Aim a little above the tier floor so the product reliably qualifies,
            # then never exceed the donor's safe maximum.
            target_yield = min({1: SINGLE, 2: DOUBLE, 3: TRIPLE}[tier] * 1.08, potential)

            # ---- Run the machine to target; record both counts ----
            liters = target_yield / richness
            accurate = round(max(richness * liters * (1 + rng.normal(0, 0.035)), 0.1), 2)
            drift = float(np.clip(0.03 * max(0.0, accurate - 6.0), 0, 0.15))
            machine = round(accurate * (1 - drift) * (1 + rng.normal(0, 0.02)), 2)

            # Tier comes from the RECORDED (rounded) count, so units always matches.
            units = 3 if accurate >= TRIPLE else (2 if accurate >= DOUBLE
                                                  else (1 if accurate >= SINGLE else 0))

            rows.append({**base, "status": "completed", "target_tier": tier,
                         "liters_processed": round(liters, 2),
                         "accurate_count": accurate,
                         "machine_count": machine,
                         "units": units, "true_richness": round(richness, 3)})

    df = pd.DataFrame(rows).sort_values(["donation_date", "donor_index"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Defect injection: realistic raw-intake data-quality problems on TOP of the
# clean biology, mapped to the Kahn (conformance, completeness, plausibility)
# and DAMA dimensions. Rates are grounded in the healthcare data-quality
# literature and all scale with DEFECT_RATE. The biological signal in the kept
# fields is untouched; defects sit on the RECORDED values so file 2 can detect
# and repair or drop them. A clean copy is kept separately as the ground truth.
# ----------------------------------------------------------------------------
_GENDER_VARIANTS = {"M": ["M", "m", "Male", "MALE", " M ", "male"],
                    "F": ["F", "f", "Female", "FEMALE", " F ", "female"]}


def _pick(rng, n, rate):
    """Choose distinct row positions for a defect at the given (scaled) rate."""
    k = int(round(n * rate * DEFECT_RATE))
    if k <= 0:
        return np.array([], dtype=int)
    return rng.choice(n, size=min(k, n), replace=False)


def inject_defects(clean_df, seed=SEED + 1):
    """Return a messy, as-recorded copy of the clean donations table."""
    rng = np.random.default_rng(seed)
    df = clean_df.copy()
    n = len(df)
    # Recorded fields become free-form (object) so text, typos, and blanks can live in them.
    for col in ["gender", "donation_date", "weight_kg", "platelet_precount",
                "height_cm", "age", "donor_type"]:
        df[col] = df[col].astype(object)

    # Conformance: gender coded inconsistently (~25%).
    for i in _pick(rng, n, 0.25):
        canon = "M" if str(df.at[i, "gender"]).strip().upper().startswith("M") else "F"
        df.at[i, "gender"] = rng.choice(_GENDER_VARIANTS[canon])

    # Conformance: donor_type whitespace/case noise (~10%).
    for i in _pick(rng, n, 0.10):
        v = str(df.at[i, "donor_type"])
        df.at[i, "donor_type"] = rng.choice([v.upper(), " " + v, v + " ", v.title()])

    # Conformance: dates as mixed-format text (~15%) and a few unparseable (~1%).
    for i in _pick(rng, n, 0.15):
        d = pd.Timestamp(df.at[i, "donation_date"])
        df.at[i, "donation_date"] = d.strftime(
            rng.choice(["%m/%d/%Y", "%d-%b-%Y", "%m-%d-%Y", "%B %d, %Y"]))
    for i in _pick(rng, n, 0.01):
        df.at[i, "donation_date"] = rng.choice(["", "n/a", "2024-13-45", "unknown"])

    # Conformance: numbers stored as text with stray characters/units (~3%).
    for i in _pick(rng, n, 0.03):
        w = df.at[i, "weight_kg"]
        if pd.notna(w):
            df.at[i, "weight_kg"] = rng.choice([f"{w} kg", f"{w} ", f"{int(float(w))} KG"])
    for i in _pick(rng, n, 0.03):
        p = df.at[i, "platelet_precount"]
        if pd.notna(p):
            df.at[i, "platelet_precount"] = rng.choice([f"{int(float(p))} ", f"{int(float(p)):,}", f"~{int(float(p))}"])

    # Completeness: blank/missing values (~3% each).
    for col, rate in [("platelet_precount", 0.03), ("weight_kg", 0.03),
                      ("height_cm", 0.03), ("gender", 0.03), ("donation_date", 0.03)]:
        for i in _pick(rng, n, rate):
            df.at[i, col] = np.nan

    # Plausibility (atemporal): impossible values from entry error (~2% of rows).
    impossible = {"age": [0, 200, -5], "weight_kg": [7.0, 700.0, -10.0],
                  "height_cm": [12.0, 999.0], "platelet_precount": [5.0, 9999.0, -100.0]}
    for i in _pick(rng, n, 0.02):
        col = rng.choice(list(impossible))
        df.at[i, col] = float(rng.choice(impossible[col]))

    # Plausibility (temporal): donation before birthdate or in the future (~1%).
    for i in _pick(rng, n, 0.01):
        bd = pd.Timestamp(df.at[i, "birthdate"])
        if rng.integers(0, 2) == 0:
            df.at[i, "donation_date"] = (bd - pd.Timedelta(days=int(rng.integers(1, 400)))).strftime("%Y-%m-%d")
        else:
            df.at[i, "donation_date"] = "2031-0%d-15" % int(rng.integers(1, 9))

    # Accuracy (cross-field): recorded units disagree with the recorded count (~2%).
    for i in _pick(rng, n, 0.02):
        u = int(df.at[i, "units"])
        df.at[i, "units"] = int(np.clip(u + rng.choice([-1, 1, 2]), 0, 3))

    # Accuracy: unit slips (weight in pounds ~2%, height in inches ~1%).
    for i in _pick(rng, n, 0.02):
        w = df.at[i, "weight_kg"]
        if isinstance(w, (int, float)) and pd.notna(w) and w > 0:
            df.at[i, "weight_kg"] = round(float(w) * 2.20462, 1)
    for i in _pick(rng, n, 0.01):
        h = df.at[i, "height_cm"]
        if isinstance(h, (int, float)) and pd.notna(h) and h > 0:
            df.at[i, "height_cm"] = round(float(h) / 2.54, 1)

    # Uniqueness: exact duplicates (~3%) and near-duplicate overlays (~2%).
    exact = df.iloc[_pick(rng, n, 0.03)].copy()
    overlay = df.iloc[_pick(rng, n, 0.02)].copy()
    wcol = overlay.columns.get_loc("weight_kg")
    for j in range(len(overlay)):
        w = overlay.iat[j, wcol]
        if isinstance(w, (int, float)) and pd.notna(w):
            overlay.iat[j, wcol] = round(float(w) + float(rng.choice([-1, 1])), 1)
    df = pd.concat([df, exact, overlay], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Run (notebook-cell style) and report realism checks
# ----------------------------------------------------------------------------
donations_truth = generate_sbc_yield_data(SEED)            # clean biological ground truth
donations_raw = inject_defects(donations_truth, seed=SEED + 1)   # messy, as recorded

if __name__ == "__main__":
    df = donations_truth
    print("=== RAW DEFECTS (file 2 will clean these) ===")
    print(f"truth rows {len(donations_truth):,} -> raw rows {len(donations_raw):,} "
          f"(+{len(donations_raw) - len(donations_truth):,} duplicate/overlay)")
    print(f"raw exact duplicate rows : {int(donations_raw.duplicated().sum()):,}")
    print(f"raw gender labels         : {sorted(set(map(str, donations_raw['gender'].dropna().unique())))[:8]}")
    print()
    comp = df[df["status"] == "completed"]
    per_donor = df.groupby("donor_index").size()
    ft = df.drop_duplicates("donor_index")["donor_type"].eq("first_time").mean()

    print("=== SIZE ===")
    print(f"donors            : {df['donor_index'].nunique():,}")
    print(f"donation rows      : {len(df):,}  (completed {len(comp):,}, "
          f"deferred {int((df['status']=='deferred').sum()):,}, "
          f"low-yield {int((df['status']=='low_yield').sum()):,})")
    print(f"first-time donors  : {ft:6.1%}")
    print(f"donations/donor    : mean {per_donor.mean():.1f}  median {per_donor.median():.0f}"
          f"  max {per_donor.max()}")

    print("\n=== DEMOGRAPHICS ===")
    d1 = df.drop_duplicates("donor_index")
    print(f"male share         : {d1['gender'].eq('M').mean():6.1%}")
    print(f"age                : mean {d1['age'].mean():.1f}  sd {d1['age'].std():.1f}"
          f"  range {d1['age'].min()}-{d1['age'].max()}")
    print(f"weight kg (M/F)    : {d1[d1.gender=='M'].weight_kg.mean():.0f} / "
          f"{d1[d1.gender=='F'].weight_kg.mean():.0f}")
    print(f"precount (day-of)  : mean {d1['platelet_precount'].mean():.0f}  "
          f"range {d1['platelet_precount'].min():.0f}-{d1['platelet_precount'].max():.0f}")

    print("\n=== YIELD / TIERS (completed) ===")
    print(f"accurate count     : mean {comp['accurate_count'].mean():.2f}  "
          f"sd {comp['accurate_count'].std():.2f} (x10^11)")
    print(f"tier share 1/2/3   : "
          f"{comp['units'].eq(1).mean():.0%} / {comp['units'].eq(2).mean():.0%} / "
          f"{comp['units'].eq(3).mean():.0%}")
    print(f"machine under-read : mean {(comp['accurate_count']-comp['machine_count']).mean():.3f} x10^11; "
          f"at high yield (>=8): {(comp[comp.accurate_count>=8].eval('accurate_count-machine_count')).mean():.3f}")

    print("\n=== CAPPING CHECK (this is the key signal) ===")
    rate = comp["accurate_count"] / comp["liters_processed"]
    print(f"corr(precount, RAW count)  : {comp['platelet_precount'].corr(comp['accurate_count']):.2f}  "
          "(weaker, because totals are capped)")
    print(f"corr(precount, RATE)       : {comp['platelet_precount'].corr(rate):.2f}  "
          "(strong, capping removed)")
    print(f"corr(rate, true_richness)  : {rate.corr(comp['true_richness']):.2f}  "
          "(rate recovers true biology)")
