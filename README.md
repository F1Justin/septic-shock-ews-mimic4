# Heart Rate Lag-1 Autocorrelation in Pre-shock Dynamics: A MIMIC-IV Exploratory Analysis

Replication code for:

> **Heart Rate Lag-1 Autocorrelation Is Associated with Pre-shock Dynamics in Late-Onset Septic Shock: A Single-Cohort Exploratory Analysis of MIMIC-IV**

---

## Overview

This repository contains all analysis scripts, output tables, and figures for an exploratory study examining early-warning signals (EWS) — specifically HR lag-1 autocorrelation (AC1-HR) and MAP variance — in the 48 hours preceding vasopressor initiation in septic shock patients.

**Key finding:** Late-centred rolling-window HR AC1 is associated with shock in a matched conditional logistic model (OR = 1.60, 95% CI 1.15–2.22, p = 0.005), but the association attenuates after adjustment for concurrent haemodynamic values, suggesting co-evolution rather than independent early-warning capability.

---

## Data Access

This study uses **MIMIC-IV v3.1**, which requires credentialed access via PhysioNet:

1. Complete CITI training and apply at: https://physionet.org/content/mimiciv/
2. Download MIMIC-IV v3.1 and place under `mimiciv/3.1/`
3. Expected structure:
   ```
   mimiciv/3.1/
   ├── hosp/
   └── icu/
   ```

`data/` and `mimiciv/` are excluded from this repository per the PhysioNet Data Use Agreement.

---

## Reproduction

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize database (one-time, optional)

```bash
python scripts/01_setup_db.py
```

Builds DuckDB views and derived tables (sepsis3, SOFA, vasopressor equivalents) from the MIMIC-IV source files.
You can still run this step manually, but `runall.sh` now performs a DuckDB preflight check and will automatically run `scripts/01_setup_db.py` if the database is missing or if copied views still point to stale absolute paths from another machine.

### 3. Run full pipeline

```bash
bash runall.sh
```

This runs a DuckDB preflight check, auto-runs Step 1 when needed, then executes Steps 2–8 sequentially and recompiles `report.pdf`. See `SETUP.md` for step-by-step details.

Useful options:

```bash
bash runall.sh --with-setup   # force Step 1 before the pipeline
bash runall.sh --skip-setup   # skip Step 1 auto-checks
```

---

## Repository Structure

```
├── scripts/          # Analysis pipeline (01–08)
├── output/           # Figures and result tables
├── report.tex        # Manuscript source
├── report.pdf        # Compiled manuscript
├── SETUP.md          # Detailed pipeline documentation
├── runall.sh         # DuckDB preflight + optional Step 1 + Steps 2–8 + compile PDF
└── requirements.txt  # Python dependencies
```

---

## Cohort Summary

| Stage | N |
|-------|---|
| MIMIC-IV ICU stays | 94,458 |
| Sepsis-3 eligible | 41,296 |
| Septic shock (operational definition) | 10,310 |
| Analysable (72 h data quality) | 725 cases / 1,447 controls |

Matching: 1:2 risk-set, admission SOFA ±2 calipers.

---

## License

Code: [MIT License](LICENSE)

MIMIC-IV data is governed by the [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciv/view-license/3.1/). Derived patient-level data is not included in this repository.
