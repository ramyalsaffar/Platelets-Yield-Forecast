# Platelet Yield Prediction Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1A7A3B)
![LightGBM](https://img.shields.io/badge/LightGBM-7CB342)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Data: synthetic](https://img.shields.io/badge/data-synthetic-lightgrey)

A machine-learning pipeline that predicts, before a donor's appointment, how rich
their platelet donation will be, and turns that prediction into a better booking
and recruitment plan. It runs end to end on synthetic data that mimics real
Stanford Blood Center (SBC) intake records, including their data-quality defects.

SBC is the blood center. "Yield" is the number of platelets a donation produces,
measured in units of 10 to the 11th power. A donation is graded as a single,
double, or triple product.

## Results

Measured on held-out donors the model never saw during training.

| What | Result |
| --- | --- |
| Per-donor yield-rate prediction | mean absolute error 0.123, R-squared 0.80 |
| Whole-product prediction (single / double / triple) | exact 85% of the time |
| Appointments to reach a high-demand target | about 24% fewer than random booking |
| Recruitment precision (top 20% of donors flagged) | 92% give a double-or-better, vs 55% at the base rate (1.67x lift) |

Mean absolute error is the average size of the prediction error. R-squared is the
share of the variation the model explains (1.0 is perfect).

## Approach

- Predict the per-liter yield rate (richness), not the raw total. The rate is
  invariant to the demand cap that limits what historical records show, so it is
  the honest learning target. Body size sets how much can be safely collected
  (capacity), which is applied afterward to turn the rate into whole products.
- The pre-donation platelet count (precount) is the dominant predictor and is
  known at booking, because a donor's counts are stable between visits. A fallback
  feature set drops it for donors with no count on file.
- Every history feature uses prior donations only, and cross-validation groups by
  donor so the same donor is never in both train and test.

### Models compared

Seven regressors span the standard families for tabular data, from simple and
interpretable to strong, so the simplest model that is good enough wins: Linear
regression, ElasticNet (penalized linear), CART (a single decision tree), Random
Forest (averaged trees), HistGradientBoosting, XGBoost, and LightGBM (gradient
boosting, which builds many small trees in sequence). They are defined as plain
scikit-learn estimators in one place (`make_regressors`), not as custom classes,
so adding or swapping a model is a one-line change. A simple Linear model wins and
ships; this holds even after tuning the stronger models, which do not beat it.

### Automation

Cleaning, the model benchmark, hyperparameter tuning, and final model selection
run automatically from raw data to the chosen model, with no manual steps in
between. Tuning selects the lowest-error model on its own and compares it against
fixed baselines, so complexity is only kept when it clearly earns its place.

### How the tuning works, and why

- The held-out test set is split by donor with GroupShuffleSplit (70 percent
  train, 30 percent test), so no donor is in both. GroupShuffleSplit keeps whole
  donors on one side of the split.
- Each candidate is grid-searched with GroupKFold (3 folds) inside the training
  donors only, so tuning never sees the test donors. GroupKFold is k-fold
  cross-validation that keeps each donor within a single fold.
- The model with the lowest test mean absolute error is selected, then compared
  against two fixed baselines (plain Linear, and a precount-only model). The
  pipeline keeps the simple model unless the tuned one beats it by a clear margin.
- Grid search over small, sensible grids is used on purpose: the grids are tiny,
  and the donor-grouped folds are the part that protects against leakage. The goal
  is an honest, leakage-free estimate, not chasing tiny validation gains.

## Repository structure

```
1- synthetic_yield.py      Generate clean + defect-injected synthetic data
2- cleaning.py             Clean the messy data; data dictionary + quality gate
3- features.py             Leakage-safe features and the model target
4- models.py               Model benchmark vs honest baselines
5- tuning.py               Leakage-safe hyperparameter tuning
6- scheduling_policy.py    Predicted rate -> products -> booking simulation
7- plotting.py             Generate the five result charts
```

## Requirements

```
pip install numpy pandas scikit-learn matplotlib xgboost lightgbm
```

## Running the pipeline

The seven files are meant to run as notebook cells, in number order, sharing one
workspace. They do not import each other, so the number prefixes set the run
order, and no separate runner script is needed.

- In Jupyter or JupyterHub: paste each file into its own cell, in order 1 to 7,
  and run top to bottom.
- In a plain Python session: run them in order in the same process so they share
  one namespace, for example `exec(open("1- synthetic_yield.py").read())` through
  file 7.

File 1 exposes one knob, `DEFECT_RATE` (default 1.0), that scales the injected
defects; set it to 0 for clean data. Everything is seeded (`SEED = 42`), so runs
are reproducible.

## Data quality

The synthetic data is deliberately messy, then cleaned to recognized standards.

- `1- synthetic_yield.py` keeps a clean ground-truth table and injects realistic
  intake defects on top of it: duplicates and overlays, missing values, mixed and
  unparseable dates, inconsistent gender coding, numbers stored as text, unit
  slips (pounds, inches), impossible values, impossible timing, and recorded-units
  versus measured-count mismatches.
- `2- cleaning.py` detects and repairs or drops each defect class. Missing values
  are imputed and flagged, and records that are deferrals or fail the safety limits
  are dropped. Steps are tagged verification (against internal rules) or validation (against the FDA and
  AABB donation limits). A data dictionary gives every column a meaning, unit, and
  role, and a final quality gate asserts the cleaned table's invariants (no missing
  features, gender only M or F, precount at or above 150, units 0 to 3, no leaky
  columns) before release.

Defects and checks map to the Kahn framework for electronic health record data
(conformance, completeness, plausibility) and the DAMA six dimensions
(completeness, validity, consistency, uniqueness, timeliness, accuracy). FDA is
the Food and Drug Administration; AABB is the blood-bank accreditation body; DAMA
is the Data Management Association.

## Limitations

- Synthetic data only. Connecting to real SBC records (extraction) is out of scope
  for this demo; the cleaning and modeling are built to accept real data in the
  same shape.
- Weight unit slips can only be resolved in the clear cases (a value impossible as
  kilograms for the recorded height). Truly ambiguous values, plausible as either
  pounds or kilograms, are left as recorded, since they cannot be resolved without
  a recorded unit. Weight is a weak predictor, so the effect on the model is small.
- The apheresis machine's own count is reported only as a reference, not as a head
  to head rival, because it is an at-the-chair value that is not available at
  booking and predicts the total, not the per-liter rate.

## License

Released under the MIT License. See `LICENSE`.
