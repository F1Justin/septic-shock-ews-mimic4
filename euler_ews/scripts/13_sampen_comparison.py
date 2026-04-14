"""
SampEn vs Topological Metrics: head-to-head robustness comparison

Research question: Does Sample Entropy (SampEn) survive mean-level adjustment
in the same way as n_extrema? If SampEn loses significance at S2 (like AC1),
while n_extrema retains it, this strengthens the claim that topological
oscillatory complexity captures a mean-independent structural signal.

Metrics computed per 12-hour rolling window of HR residuals:
    sampen_m1  : Sample Entropy, m=1, r=0.2*local_std   (primary)
    sampen_m2  : Sample Entropy, m=2, r=0.2*local_std   (sensitivity)
    euler_hr   : Euler χ(0)                              (from 11, reproduced)
    n_extrema_hr: local extrema count                    (from 11, reproduced)

Models (matched on SampEn complete-case):
    M1  : base model
    S1  : + late-window raw mean HR
    S2  : + late-window raw mean HR + MAP   ← key robustness test
    S3  : early-window SampEn, + early HR/MAP means
    S5  : no-sedation subgroup

Output:
    euler_ews/output/tbl_sampen_logistic.csv
    euler_ews/output/fig_sampen_4way_forest.png   ← main comparison figure
    euler_ews/output/fig_sampen_m1_vs_m2.png      ← m sensitivity
"""

from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import matplotlib
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "DejaVu Sans"],
    "axes.unicode_minus": False,
})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from statsmodels.discrete.conditional_models import ConditionalLogit

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent          # euler_ews/
NAAS_ROOT    = PROJECT_ROOT.parent                   # NaaS/

COHORT_PATH  = NAAS_ROOT / "data" / "cohort.parquet"
VITALS_PATH  = NAAS_ROOT / "data" / "vitals_cleaned.parquet"
DIAG_PATH    = NAAS_ROOT / "data" / "cleaning_diagnostics.parquet"
DB_PATH      = NAAS_ROOT / "mimiciv" / "mimiciv.db"
OUTPUT_DIR   = PROJECT_ROOT / "output"

WINDOW_SIZE        = 12
MIN_ACTUAL_TOPO    = 6   # Euler / n_extrema threshold (matches 09/11)
MIN_ACTUAL_SAMPEN   = 8   # SampEn minimum non-interpolated points per window
SAMPEN_R_STD        = 0.2 # "standard" r factor  → reported NaN rate ~55-60%
SAMPEN_R_RELAXED    = 0.5 # "relaxed"  r factor  → NaN rate ~3-5%, used for regression
# Both r values use patient-level 48-h HR residual std (not window-local std),
# following Richman & Moorman (2000).  The high NaN at r=0.2 is itself reported
# as a key limitation of SampEn for sparse EHR data.

EARLY_LO, EARLY_HI = -48, -24
LATE_LO,  LATE_HI  = -12,   0

SEDATIVE_ITEMIDS    = (221385, 221623, 221668, 222168, 225150, 229420)
BETABLOCKER_ITEMIDS = (221429, 225153, 225974)


# ── Metric functions ──────────────────────────────────────────────────────────

def euler_at_zero(vals: np.ndarray, is_interp: np.ndarray) -> float:
    actual = vals[~is_interp & ~np.isnan(vals)]
    if len(actual) < MIN_ACTUAL_TOPO:
        return np.nan
    below = actual <= 0
    if not below.any():
        return 0.0
    n_comp = int(below[0]) + int((np.diff(below.astype(np.int8)) == 1).sum())
    return float(n_comp)


def n_extrema_fn(vals: np.ndarray, is_interp: np.ndarray) -> float:
    actual = vals[~is_interp & ~np.isnan(vals)]
    if len(actual) < MIN_ACTUAL_TOPO:
        return np.nan
    n_min = len(argrelextrema(actual, np.less,    order=1)[0])
    n_max = len(argrelextrema(actual, np.greater, order=1)[0])
    return float(n_min + n_max)


def sample_entropy(
    vals: np.ndarray,
    is_interp: np.ndarray,
    m: int = 1,
    r: float | None = None,
    r_factor: float = SAMPEN_R_STD,
    min_actual: int = MIN_ACTUAL_SAMPEN,
) -> float:
    """
    Sample Entropy (SampEn) of the non-interpolated HR residual values
    in a single 12-hour window.

    Parameters
    ----------
    vals, is_interp : window arrays from the vitals time series
    m               : template length (1 = primary, 2 = sensitivity)
    r               : absolute tolerance; if provided, overrides r_factor
    r_factor        : used only when r is None: tolerance = r_factor * std(actual)
    min_actual      : minimum non-interpolated points; returns NaN if not met

    Design note on r:
        Using r = r_factor * local_window_std produces >70% NaN at hourly
        EHR resolution (12 points), because the local std estimate is unstable
        and the resulting r is too tight for A (length-m+1 template) counts.
        Richman & Moorman (2000) recommend computing std from the full time
        series. Therefore, callers should pre-compute r = 0.2 * patient_48h_std
        and pass it as `r` (absolute). This is done in build_features().

    Algorithm (Richman & Moorman 2000):
        B = number of template pairs of length m     that match within r
        A = number of template pairs of length m+1   that match within r
        SampEn = -ln(A / B)   (NaN when B=0 or A=0)
    """
    actual = vals[~is_interp & ~np.isnan(vals)]
    if len(actual) < min_actual:
        return np.nan
    if r is None:
        sd = actual.std(ddof=1)
        if sd == 0:
            return np.nan
        r = r_factor * sd
    if r <= 0:
        return np.nan
    N = len(actual)

    def _count(template_len: int) -> int:
        """Count self-excluding template-match pairs (Chebyshev distance < r)."""
        count = 0
        for i in range(N - template_len):
            tmpl = actual[i : i + template_len]
            for j in range(i + 1, N - template_len):
                if np.max(np.abs(actual[j : j + template_len] - tmpl)) < r:
                    count += 1
        return count

    B = _count(m)
    if B == 0:
        return np.nan          # undefined; mark as missing rather than inf/0
    A = _count(m + 1)
    if A == 0:
        # Technically SampEn → +∞; we return NaN to avoid inflating the mean
        return np.nan
    return float(-np.log(A / B))


# ── Window-level feature computation (single pass) ───────────────────────────

def window_features(
    ts: pd.DataFrame,
    T0: pd.Timestamp,
    global_r_m1: float | None = None,    # relaxed r  (m=1, used for regression)
    global_r_m2: float | None = None,    # relaxed r  (m=2)
    r_std_m1: float | None = None,       # standard r (m=1, reported for NaN diagnostic)
    r_std_m2: float | None = None,       # standard r (m=2)
) -> pd.DataFrame:
    """
    Compute per-window metrics in a single pass.
    Returns one row per rolling 12-hour window centre.

    SampEn is computed twice:
      - sampen_std_m*: r = 0.2 * patient_48h_std  (standard; high NaN, not used in regression)
      - sampen_rel_m*: r = 0.5 * patient_48h_std  (relaxed;  low NaN,  used in regression)
    """
    ts = ts.sort_values("charttime").reset_index(drop=True)
    if len(ts) < WINDOW_SIZE:
        return pd.DataFrame()

    hr_vals   = ts["hr_residual"].to_numpy(float)
    hr_interp = ts["hr_is_interpolated"].to_numpy(bool)

    rows = []
    for i in range(len(ts) - WINDOW_SIZE + 1):
        sl      = slice(i, i + WINDOW_SIZE)
        hv, hi  = hr_vals[sl], hr_interp[sl]
        n_act   = int((~hi & ~np.isnan(hv)).sum())
        center  = ts["charttime"].iloc[i + WINDOW_SIZE // 2]
        h_before = (center - T0).total_seconds() / 3600

        rows.append({
            "hours_before_T0":   round(h_before, 1),
            "euler_hr":          euler_at_zero(hv, hi),
            "n_extrema_hr":      n_extrema_fn(hv, hi),
            # standard r (for NaN coverage report)
            "sampen_std_m1":     sample_entropy(hv, hi, m=1, r=r_std_m1),
            "sampen_std_m2":     sample_entropy(hv, hi, m=2, r=r_std_m2),
            # relaxed r (for regression)
            "sampen_rel_m1":     sample_entropy(hv, hi, m=1, r=global_r_m1),
            "sampen_rel_m2":     sample_entropy(hv, hi, m=2, r=global_r_m2),
            "n_actual_hr":       n_act,
            "low_conf_topo":     n_act < MIN_ACTUAL_TOPO,
            "low_conf_sampen":   n_act < MIN_ACTUAL_SAMPEN,
        })
    return pd.DataFrame(rows)


def build_features(vitals: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate over (stay_id, T0) pairs and compute early/late window means
    for all four metrics. One row per (stay_id, T0).

    SampEn tolerance r is set to SAMPEN_R_FACTOR * std(all actual HR residuals
    in the 48-h observation window) — following Richman & Moorman (2000) who
    recommend computing r from the full time series, not from sub-windows.
    This avoids the >70% NaN rate that arises when r is based on the noisy
    local 12-point window std.
    """
    print("── Computing window features (euler + n_extrema + SampEn) ──────")
    pairs   = vitals.groupby(["stay_id", "T0"], sort=False)
    n_total = pairs.ngroups
    rows = []
    for i, ((sid, t0), grp) in enumerate(pairs, 1):
        # Pre-compute patient-level r from full 48-h actual HR residuals
        actual_48h = grp.loc[
            ~grp["hr_is_interpolated"] & grp["hr_residual"].notna(),
            "hr_residual"
        ].to_numpy(float)
        if len(actual_48h) >= 2:
            sd_48h = actual_48h.std(ddof=1)
            if sd_48h > 0:
                # standard (r=0.2): reported for NaN diagnostic only
                r_std_m1 = SAMPEN_R_STD    * sd_48h
                r_std_m2 = SAMPEN_R_STD    * sd_48h
                # relaxed (r=0.5): used for regression
                r_rel_m1 = SAMPEN_R_RELAXED * sd_48h
                r_rel_m2 = SAMPEN_R_RELAXED * sd_48h
            else:
                r_std_m1 = r_std_m2 = r_rel_m1 = r_rel_m2 = None
        else:
            r_std_m1 = r_std_m2 = r_rel_m1 = r_rel_m2 = None

        wins = window_features(grp, pd.Timestamp(t0),
                               global_r_m1=r_rel_m1, global_r_m2=r_rel_m2,
                               r_std_m1=r_std_m1, r_std_m2=r_std_m2)
        if wins.empty:
            continue

        def wmean_topo(col, lo, hi):
            mask = (
                (wins["hours_before_T0"] >= lo) &
                (wins["hours_before_T0"] < hi) &
                (~wins["low_conf_topo"])
            )
            return wins.loc[mask, col].dropna().mean()

        def wmean_sampen(col, lo, hi):
            mask = (
                (wins["hours_before_T0"] >= lo) &
                (wins["hours_before_T0"] < hi) &
                (~wins["low_conf_sampen"])
            )
            return wins.loc[mask, col].dropna().mean()

        rows.append({
            "stay_id": sid,
            "T0":      pd.Timestamp(t0),
            # topological
            "euler_hr_early_mean":            wmean_topo("euler_hr",      EARLY_LO, EARLY_HI),
            "euler_hr_late_mean":             wmean_topo("euler_hr",      LATE_LO,  LATE_HI),
            "n_extrema_hr_early_mean":        wmean_topo("n_extrema_hr",  EARLY_LO, EARLY_HI),
            "n_extrema_hr_late_mean":         wmean_topo("n_extrema_hr",  LATE_LO,  LATE_HI),
            # SampEn standard r=0.2 (for NaN diagnostic only)
            "sampen_std_m1_early_mean":       wmean_sampen("sampen_std_m1", EARLY_LO, EARLY_HI),
            "sampen_std_m1_late_mean":        wmean_sampen("sampen_std_m1", LATE_LO,  LATE_HI),
            # SampEn relaxed r=0.5, m=1 (primary regression predictor)
            "sampen_rel_m1_early_mean":       wmean_sampen("sampen_rel_m1", EARLY_LO, EARLY_HI),
            "sampen_rel_m1_late_mean":        wmean_sampen("sampen_rel_m1", LATE_LO,  LATE_HI),
            # SampEn relaxed r=0.5, m=2 (sensitivity)
            "sampen_rel_m2_early_mean":       wmean_sampen("sampen_rel_m2", EARLY_LO, EARLY_HI),
            "sampen_rel_m2_late_mean":        wmean_sampen("sampen_rel_m2", LATE_LO,  LATE_HI),
        })
        if i % 200 == 0:
            print(f"  {i:,}/{n_total:,}...", flush=True)
    return pd.DataFrame(rows)


# ── Clinical covariates (identical to script 11) ─────────────────────────────

def simplify_careunit(cu: str) -> str:
    if pd.isna(cu):
        return "Other"
    cu = str(cu).upper()
    if "MICU" in cu or ("MEDICAL" in cu and "SICU" not in cu):
        return "MICU"
    if "SICU" in cu or "SURGICAL" in cu or "TSICU" in cu:
        return "SICU"
    if "CCU" in cu or "CARDIAC" in cu or "CSRU" in cu or "CVICU" in cu:
        return "CCU_CSRU"
    if "NICU" in cu or "NEURO" in cu:
        return "Neuro_NICU"
    return "Other"


def pull_covariates(df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        con.register("_ids", df[["stay_id", "subject_id"]].drop_duplicates())
        demog = con.execute("""
            SELECT ie.stay_id, ie.first_careunit
            FROM _ids i
            INNER JOIN mimiciv_icu.icustays ie ON ie.stay_id = i.stay_id
        """).df()

        con.register("_st0", df[["stay_id", "T0"]].drop_duplicates())
        vent = con.execute("""
            SELECT DISTINCT s.stay_id, s.T0, 1 AS vent_before_window
            FROM mimiciv_derived.ventilation v
            INNER JOIN _st0 s ON v.stay_id = s.stay_id
            WHERE v.ventilation_status = 'InvasiveVent'
              AND v.starttime <= s.T0 - INTERVAL '12 hours'
              AND (v.endtime IS NULL OR v.endtime >= s.T0 - INTERVAL '12 hours')
        """).df()

        itemids = ",".join(map(str, SEDATIVE_ITEMIDS))
        con.register("_st0_sed", df[["stay_id", "T0"]].drop_duplicates())
        sed = con.execute(f"""
            SELECT DISTINCT s.stay_id, s.T0, 1 AS sedation_before_window
            FROM mimiciv_icu.inputevents ie
            INNER JOIN _st0_sed s ON ie.stay_id = s.stay_id
            WHERE ie.itemid IN ({itemids})
              AND COALESCE(ie.amount, 0) > 0
              AND ie.starttime < s.T0
              AND COALESCE(ie.endtime, ie.starttime) >= s.T0 - INTERVAL '12 hours'
        """).df()

        bb_ids = ",".join(map(str, BETABLOCKER_ITEMIDS))
        con.register("_st0_bb", df[["stay_id", "T0"]].drop_duplicates())
        bb = con.execute(f"""
            SELECT DISTINCT s.stay_id, s.T0, 1 AS betablocker_before_window
            FROM mimiciv_icu.inputevents ie
            INNER JOIN _st0_bb s ON ie.stay_id = s.stay_id
            WHERE ie.itemid IN ({bb_ids})
              AND COALESCE(ie.amount, 0) > 0
              AND ie.starttime < s.T0
              AND COALESCE(ie.endtime, ie.starttime) >= s.T0 - INTERVAL '12 hours'
        """).df()
    finally:
        con.close()

    mdf = (
        df
        .merge(demog, on="stay_id", how="left")
        .merge(vent,  on=["stay_id", "T0"], how="left")
        .merge(sed,   on=["stay_id", "T0"], how="left")
        .merge(bb,    on=["stay_id", "T0"], how="left")
    )
    mdf["vent_before_window"]        = mdf["vent_before_window"].fillna(0).astype(int)
    mdf["sedation_before_window"]    = mdf["sedation_before_window"].fillna(0).astype(int)
    mdf["betablocker_before_window"] = mdf["betablocker_before_window"].fillna(0).astype(int)
    mdf["icu_type"]     = mdf["first_careunit"].apply(simplify_careunit)
    mdf["icu_SICU"]     = (mdf["icu_type"] == "SICU").astype(float)
    mdf["icu_CCU_CSRU"] = (mdf["icu_type"] == "CCU_CSRU").astype(float)
    mdf["icu_Neuro"]    = (mdf["icu_type"] == "Neuro_NICU").astype(float)
    mdf["icu_Other"]    = (mdf["icu_type"] == "Other").astype(float)
    return mdf


# ── Conditional logistic regression ──────────────────────────────────────────

BASE_COVARS = [
    "vent_before_window", "sedation_before_window", "betablocker_before_window",
    "mon_abp", "mon_mixed",
    "icu_SICU", "icu_CCU_CSRU", "icu_Neuro", "icu_Other",
]


def fit_clogit(
    data: pd.DataFrame,
    predictor: str,
    extra_covars: list[str] | None = None,
    include_sedation: bool = True,
    label: str = "",
) -> pd.DataFrame:
    extra  = extra_covars or []
    covars = list(extra) + [c for c in BASE_COVARS
                            if include_sedation or c != "sedation_before_window"]
    all_vars = [predictor] + covars

    pair_df = pd.read_parquet(COHORT_PATH)[["stay_id", "T0", "matched_pair_id"]].drop_duplicates()
    df2 = data.merge(pair_df, on=["stay_id", "T0"], how="left")
    df2 = df2.dropna(subset=["matched_pair_id", *all_vars])

    pair_bal = df2.groupby("matched_pair_id")["binary_group"].agg(shock="sum", total="size")
    valid    = pair_bal[(pair_bal["shock"] >= 1) & (pair_bal["shock"] < pair_bal["total"])].index
    df2      = df2[df2["matched_pair_id"].isin(valid)].copy()

    X = df2[all_vars].astype(float).values
    y = df2["binary_group"].astype(int).values
    g = df2["matched_pair_id"].values

    model  = ConditionalLogit(y, X, groups=g)
    result = model.fit(method="bfgs", disp=False, maxiter=500)
    cis    = result.conf_int()

    rows = []
    for i, var in enumerate(all_vars):
        rows.append({
            "Variable": var,
            "OR":       np.exp(result.params[i]),
            "CI_lo":    np.exp(cis[i, 0]),
            "CI_hi":    np.exp(cis[i, 1]),
            "P":        float(result.pvalues[i]),
            "N_obs":    len(df2),
            "N_strata": len(valid),
            "model":    label,
        })
    return pd.DataFrame(rows)


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_or(or_: float, lo: float, hi: float) -> str:
    return f"{or_:.2f} ({lo:.2f}–{hi:.2f})"


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def print_result(df: pd.DataFrame, predictor: str, title: str) -> None:
    row = df[df["Variable"] == predictor].iloc[0]
    print(f"  {title:45s}  OR={row['OR']:.2f} ({row['CI_lo']:.2f}–{row['CI_hi']:.2f})"
          f"  p={fmt_p(row['P'])}  N={row['N_obs']:,}  strata={row['N_strata']:,}")


# ── 4-way comparison forest plot ─────────────────────────────────────────────

METRIC_LABELS = {
    "euler":    "Euler χ(0)",
    "n_ext":    "n_extrema",
    "sampen_m1":"SampEn (m=1)",
    "ac1":      "AC1",
}
MODEL_LABELS = ["M1 base", "S1 +HR mean", "S2 +HR+MAP mean"]


METRIC_DISPLAY = {
    "euler":     r"HR Euler $\chi(0)$",
    "n_ext":     r"HR $N_\mathrm{ext}$",
    "sampen_m1": "SampEn (m=1)",
    "ac1":       "HR AC1",
}


def fourway_forest(
    results: dict[str, list[pd.DataFrame]],    # metric → [m1_df, s1_df, s2_df]
    predictors: dict[str, str],                # metric → predictor column name
    path: Path,
) -> None:
    """
    3-column figure (one column per model: M1 / S1 / S2).
    Within each column, 4 horizontal rows for the 4 metrics.
    OR text annotations are placed in a fixed right-side column outside the
    axes so that wide CIs (e.g. AC1) never push the annotation off-canvas.
    """
    metrics = list(results.keys())
    n_metrics = len(metrics)
    n_models  = len(MODEL_LABELS)

    # Collect all data first to find global x range across all columns
    all_data: list[list[tuple]] = []   # all_data[col][row] = (y, or, lo, hi, p, label)
    for col_idx in range(n_models):
        col_rows = []
        for row_idx, metric in enumerate(metrics):
            df   = results[metric][col_idx]
            pred = predictors[metric]
            row  = df[df["Variable"] == pred]
            if row.empty:
                continue
            row = row.iloc[0]
            n_obs = int(row["N_obs"])
            col_rows.append((
                row_idx,
                row["OR"], row["CI_lo"], row["CI_hi"],
                row["P"],
                f"{METRIC_DISPLAY.get(metric, metric)}\n(N={n_obs:,})",
            ))
        all_data.append(col_rows)

    # Global x bounds (shared across all columns for comparability)
    all_los = [r[2] for col in all_data for r in col]
    all_his = [r[3] for col in all_data for r in col]
    x_lo = max(0.0, min(all_los + [0.5]) * 0.92)
    x_hi = max(all_his + [2.5]) * 1.02   # point plot stops here
    # Text annotation goes into figure-level space via ax.annotate transform,
    # but simpler: we widen each axes by a fixed fraction for the text column.
    x_text_start = x_hi * 1.03           # left edge of text in data coords

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(5.5 * n_models, 0.85 * n_metrics + 2.2),
        sharey=True,
    )
    fig.suptitle(
        "Mean-adjustment robustness: OR (95% CI) across metrics",
        fontsize=12, y=1.01,
    )

    for col_idx, (ax, model_label) in enumerate(zip(axes, MODEL_LABELS)):
        col_rows = all_data[col_idx]
        ys       = [r[0] for r in col_rows]
        row_lbls = [r[5] for r in col_rows]

        for y, o, lo, hi, p, _ in col_rows:
            sig    = p < 0.05
            color  = "#c0392b" if sig else "#7f8c8d"
            marker = "o" if sig else "s"
            ax.plot([lo, hi], [y, y], color=color, lw=2.0, zorder=2,
                    solid_capstyle="round")
            ax.plot(o, y, marker, color=color, ms=8, zorder=3,
                    markeredgewidth=0.5, markeredgecolor="white")
            ax.text(
                x_text_start, y,
                f"OR {fmt_or(o, lo, hi)}\np={fmt_p(p)}",
                va="center", ha="left", fontsize=7.5,
                color=color, linespacing=1.4,
            )

        ax.axvline(1.0, color="#95a5a6", ls="--", lw=1.0, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels(
            row_lbls if col_idx == 0 else [""] * len(row_lbls),
            fontsize=9,
        )
        ax.set_xlabel("Odds Ratio", fontsize=9)
        ax.set_title(model_label, fontsize=10, fontweight="bold", pad=6)
        # Fixed x range across all panels; extra space on right for annotations
        ax.set_xlim(left=x_lo, right=x_text_start + 1.1)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def m_sensitivity_forest(
    m1_models: list[tuple[str, pd.DataFrame]],  # (model_label, df) for m=1
    m2_models: list[tuple[str, pd.DataFrame]],  # (model_label, df) for m=2
    path: Path,
) -> None:
    """Side-by-side: SampEn m=1 (left) vs m=2 (right) across M1/S1/S2."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    fig.suptitle("SampEn: template length sensitivity (m=1 vs m=2)", fontsize=11)

    for ax, models, title, pred_col in [
        (axes[0], m1_models, r"SampEn $m=1$, $r=0.5\sigma_{48}$ (primary)",     "sampen_rel_m1_late_mean"),
        (axes[1], m2_models, r"SampEn $m=2$, $r=0.5\sigma_{48}$ (sensitivity)", "sampen_rel_m2_late_mean"),
    ]:
        ys, ors, los, his, pvals, labels = [], [], [], [], [], []
        for y_idx, (lbl, df) in enumerate(models):
            pred = pred_col
            row  = df[df["Variable"] == pred]
            if row.empty:
                continue
            row = row.iloc[0]
            ys.append(y_idx)
            ors.append(row["OR"])
            los.append(row["CI_lo"])
            his.append(row["CI_hi"])
            pvals.append(row["P"])
            labels.append(f"{lbl}  (N={int(row['N_obs']):,})")

        x_hi   = max(his + [2.0])
        x_text = x_hi * 1.05
        for y, o, lo, hi, p in zip(ys, ors, los, his, pvals):
            c = "#c0392b" if p < 0.05 else "#7f8c8d"
            ax.plot([lo, hi], [y, y], color=c, lw=2.0, zorder=2, solid_capstyle="round")
            ax.plot(o, y, "o", color=c, ms=8, zorder=3,
                    markeredgewidth=0.5, markeredgecolor="white")
            ax.text(x_text, y,
                    f"OR {fmt_or(o, lo, hi)}  p={fmt_p(p)}",
                    va="center", ha="left", fontsize=8.5, color=c)
        ax.axvline(1.0, color="#95a5a6", ls="--", lw=1.0)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels if ax is axes[0] else [""] * len(labels), fontsize=9)
        ax.set_xlabel("Odds Ratio", fontsize=9.5)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlim(left=0, right=x_text + 0.5)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("── Loading data ─────────────────────────────────────────────────")
    cohort = pd.read_parquet(COHORT_PATH).drop_duplicates(["stay_id", "T0"])
    cohort["T0"] = pd.to_datetime(cohort["T0"])

    diag = (
        pd.read_parquet(DIAG_PATH)
        .drop_duplicates(["stay_id", "T0"])[["stay_id", "T0", "dominant_source", "excluded"]]
    )
    diag["T0"] = pd.to_datetime(diag["T0"])

    vitals = pd.read_parquet(VITALS_PATH)
    vitals["charttime"] = pd.to_datetime(vitals["charttime"])
    vitals["T0"]        = pd.to_datetime(vitals["T0"])

    passed = diag[~diag["excluded"]][["stay_id", "T0"]]
    vitals = vitals.merge(passed, on=["stay_id", "T0"])

    # ── 2. Feature computation ────────────────────────────────────────────────
    feat = build_features(vitals, cohort)
    print(f"  Feature rows: {len(feat):,}")

    # NaN rate summary — quantifies the coverage cost of SampEn vs topo metrics
    print("\n── NaN rate comparison (late window) ────────────────────────────")
    for col in [
        "euler_hr_late_mean",
        "n_extrema_hr_late_mean",
        "sampen_std_m1_late_mean",    # standard r=0.2 — expected high NaN
        "sampen_rel_m1_late_mean",    # relaxed  r=0.5 — used for regression
        "sampen_rel_m2_late_mean",    # m=2 sensitivity
    ]:
        nan_n = feat[col].isna().sum()
        print(f"  {col:42s}  NaN={nan_n:,} / {len(feat):,}"
              f"  ({100*nan_n/len(feat):.1f}%)")

    # ── 3. Build modelling dataframe ─────────────────────────────────────────
    print("\n── Assembling modelling dataframe ──────────────────────────────")
    late_raw = (
        vitals.loc[
            (vitals["charttime"] > vitals["T0"] - pd.Timedelta(hours=12)) &
            (vitals["charttime"] <= vitals["T0"])
        ]
        .groupby(["stay_id", "T0"], as_index=False)
        .agg(late_hr_mean_raw=("hr_raw", "mean"), late_map_mean_raw=("map_raw", "mean"))
    )
    early_raw = (
        vitals.loc[
            (vitals["charttime"] > vitals["T0"] - pd.Timedelta(hours=48)) &
            (vitals["charttime"] <= vitals["T0"] - pd.Timedelta(hours=24))
        ]
        .groupby(["stay_id", "T0"], as_index=False)
        .agg(early_hr_mean_raw=("hr_raw", "mean"), early_map_mean_raw=("map_raw", "mean"))
    )

    mdf = (
        cohort[["stay_id", "T0", "subject_id", "group"]]
        .merge(feat,      on=["stay_id", "T0"], how="inner")
        .merge(diag[["stay_id", "T0", "dominant_source"]], on=["stay_id", "T0"], how="left")
        .merge(late_raw,  on=["stay_id", "T0"], how="left")
        .merge(early_raw, on=["stay_id", "T0"], how="left")
    )
    mdf["binary_group"] = (mdf["group"] == "shock").astype(int)
    mdf["monitoring"]   = mdf["dominant_source"].str.lower().fillna("nbp")
    mdf["mon_abp"]      = (mdf["monitoring"] == "abp").astype(float)
    mdf["mon_mixed"]    = (mdf["monitoring"] == "mixed").astype(float)
    mdf = pull_covariates(mdf)

    print(f"  Total rows: {len(mdf):,}  "
          f"shock={mdf['binary_group'].sum():,}  "
          f"control={(~mdf['binary_group'].astype(bool)).sum():,}")

    # ── 4. Fit models ─────────────────────────────────────────────────────────
    print("\n── Fitting conditional logistic models ─────────────────────────")

    # SampEn relaxed r=0.5, m=1  (primary regression predictor)
    print("\n  [SampEn r=0.5*global_std, m=1  — primary]")
    sp1_m1 = fit_clogit(mdf, "sampen_rel_m1_late_mean", label="SampEn-r0.5 M1")
    print_result(sp1_m1, "sampen_rel_m1_late_mean", "M1 base")
    sp1_s1 = fit_clogit(mdf, "sampen_rel_m1_late_mean",
                        extra_covars=["late_hr_mean_raw"], label="SampEn-r0.5 S1")
    print_result(sp1_s1, "sampen_rel_m1_late_mean", "S1 +late HR mean")
    sp1_s2 = fit_clogit(mdf, "sampen_rel_m1_late_mean",
                        extra_covars=["late_hr_mean_raw", "late_map_mean_raw"],
                        label="SampEn-r0.5 S2")
    print_result(sp1_s2, "sampen_rel_m1_late_mean", "S2 +late HR+MAP mean")
    sp1_s3 = fit_clogit(mdf, "sampen_rel_m1_early_mean",
                        extra_covars=["early_hr_mean_raw", "early_map_mean_raw"],
                        label="SampEn-r0.5 S3 early")
    print_result(sp1_s3, "sampen_rel_m1_early_mean", "S3 early window")
    mdf_no_sed = mdf[mdf["sedation_before_window"] == 0].copy()
    sp1_s5 = fit_clogit(mdf_no_sed, "sampen_rel_m1_late_mean",
                        include_sedation=False, label="SampEn-r0.5 S5 no-sed")
    print_result(sp1_s5, "sampen_rel_m1_late_mean", "S5 no-sedation")

    # SampEn relaxed r=0.5, m=2  (sensitivity)
    print("\n  [SampEn r=0.5*global_std, m=2  — sensitivity]")
    sp2_m1 = fit_clogit(mdf, "sampen_rel_m2_late_mean", label="SampEn-r0.5-m2 M1")
    print_result(sp2_m1, "sampen_rel_m2_late_mean", "M1 base")
    sp2_s1 = fit_clogit(mdf, "sampen_rel_m2_late_mean",
                        extra_covars=["late_hr_mean_raw"], label="SampEn-r0.5-m2 S1")
    print_result(sp2_s1, "sampen_rel_m2_late_mean", "S1 +late HR mean")
    sp2_s2 = fit_clogit(mdf, "sampen_rel_m2_late_mean",
                        extra_covars=["late_hr_mean_raw", "late_map_mean_raw"],
                        label="SampEn-r0.5-m2 S2")
    print_result(sp2_s2, "sampen_rel_m2_late_mean", "S2 +late HR+MAP mean")

    # Topological metrics (reproduced on the same cohort for direct comparison)
    print("\n  [Euler χ(0) — reproduced]")
    eu_m1 = fit_clogit(mdf, "euler_hr_late_mean", label="Euler M1")
    print_result(eu_m1, "euler_hr_late_mean", "M1 base")
    eu_s1 = fit_clogit(mdf, "euler_hr_late_mean",
                       extra_covars=["late_hr_mean_raw"], label="Euler S1")
    print_result(eu_s1, "euler_hr_late_mean", "S1 +late HR mean")
    eu_s2 = fit_clogit(mdf, "euler_hr_late_mean",
                       extra_covars=["late_hr_mean_raw", "late_map_mean_raw"],
                       label="Euler S2")
    print_result(eu_s2, "euler_hr_late_mean", "S2 +late HR+MAP mean")

    print("\n  [n_extrema — reproduced]")
    nx_m1 = fit_clogit(mdf, "n_extrema_hr_late_mean", label="n_ext M1")
    print_result(nx_m1, "n_extrema_hr_late_mean", "M1 base")
    nx_s1 = fit_clogit(mdf, "n_extrema_hr_late_mean",
                       extra_covars=["late_hr_mean_raw"], label="n_ext S1")
    print_result(nx_s1, "n_extrema_hr_late_mean", "S1 +late HR mean")
    nx_s2 = fit_clogit(mdf, "n_extrema_hr_late_mean",
                       extra_covars=["late_hr_mean_raw", "late_map_mean_raw"],
                       label="n_ext S2")
    print_result(nx_s2, "n_extrema_hr_late_mean", "S2 +late HR+MAP mean")

    # AC1 (loaded from shared ews_windows, same as script 11)
    print("\n  [AC1 — comparator]")
    ews_wins = pd.read_parquet(NAAS_ROOT / "data" / "ews_windows.parquet")
    ews_wins["T0"] = pd.to_datetime(ews_wins["T0"])
    ac1_late = (
        ews_wins[
            (ews_wins["hours_before_T0"] >= LATE_LO) &
            (ews_wins["hours_before_T0"] < LATE_HI) &
            (~ews_wins["low_conf_hr"])
        ]
        .groupby(["stay_id", "T0"])["ac1_hr"].mean()
        .reset_index()
        .rename(columns={"ac1_hr": "ac1_hr_late_mean"})
    )
    mdf_ac1 = mdf.merge(ac1_late, on=["stay_id", "T0"], how="left")
    ac1_m1 = fit_clogit(mdf_ac1, "ac1_hr_late_mean", label="AC1 M1")
    print_result(ac1_m1, "ac1_hr_late_mean", "M1 base")
    ac1_s1 = fit_clogit(mdf_ac1, "ac1_hr_late_mean",
                        extra_covars=["late_hr_mean_raw"], label="AC1 S1")
    print_result(ac1_s1, "ac1_hr_late_mean", "S1 +late HR mean")
    ac1_s2 = fit_clogit(mdf_ac1, "ac1_hr_late_mean",
                        extra_covars=["late_hr_mean_raw", "late_map_mean_raw"],
                        label="AC1 S2")
    print_result(ac1_s2, "ac1_hr_late_mean", "S2 +late HR+MAP mean")

    # ── 5. Summary table ──────────────────────────────────────────────────────
    print("\n── Summary table ────────────────────────────────────────────────")
    summary_rows = []
    entries = [
        # (metric_label, predictor, m1_df, s1_df, s2_df)
        ("Euler χ(0)",    "euler_hr_late_mean",      eu_m1,  eu_s1,  eu_s2),
        ("n_extrema",     "n_extrema_hr_late_mean",  nx_m1,  nx_s1,  nx_s2),
        ("SampEn m=1 (r=0.5)", "sampen_rel_m1_late_mean", sp1_m1, sp1_s1, sp1_s2),
        ("SampEn m=2 (r=0.5)", "sampen_rel_m2_late_mean", sp2_m1, sp2_s1, sp2_s2),
        ("AC1",           "ac1_hr_late_mean",        ac1_m1, ac1_s1, ac1_s2),
    ]
    for metric, pred, m1_df, s1_df, s2_df in entries:
        for model_label, df in [("M1", m1_df), ("S1", s1_df), ("S2", s2_df)]:
            row = df[df["Variable"] == pred].iloc[0]
            summary_rows.append({
                "Metric":    metric,
                "Model":     model_label,
                "Predictor": pred,
                "N_obs":     row["N_obs"],
                "N_strata":  row["N_strata"],
                "OR":        round(row["OR"], 3),
                "CI_lo":     round(row["CI_lo"], 3),
                "CI_hi":     round(row["CI_hi"], 3),
                "P":         row["P"],
                "OR_fmt":    fmt_or(row["OR"], row["CI_lo"], row["CI_hi"]),
                "P_fmt":     fmt_p(row["P"]),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "tbl_sampen_logistic.csv", index=False, encoding="utf-8-sig")
    print(summary_df[["Metric", "Model", "N_obs", "OR_fmt", "P_fmt"]].to_string(index=False))

    # ── 6. Figures ────────────────────────────────────────────────────────────
    print("\n── Generating figures ───────────────────────────────────────────")

    # 4-way forest: chi(0) / n_extrema / SampEn(m=1) / AC1  ×  M1 / S1 / S2
    fourway_forest(
        results={
            "euler":     [eu_m1,  eu_s1,  eu_s2],
            "n_ext":     [nx_m1,  nx_s1,  nx_s2],
            "sampen_m1": [sp1_m1, sp1_s1, sp1_s2],
            "ac1":       [ac1_m1, ac1_s1, ac1_s2],
        },
        predictors={
            "euler":     "euler_hr_late_mean",
            "n_ext":     "n_extrema_hr_late_mean",
            "sampen_m1": "sampen_rel_m1_late_mean",
            "ac1":       "ac1_hr_late_mean",
        },
        path=OUTPUT_DIR / "fig_sampen_4way_forest.png",
    )

    # m=1 vs m=2 sensitivity
    m_sensitivity_forest(
        m1_models=[
            ("M1 base",         sp1_m1),
            ("S1 +HR mean",     sp1_s1),
            ("S2 +HR+MAP mean", sp1_s2),
        ],
        m2_models=[
            ("M1 base",         sp2_m1),
            ("S1 +HR mean",     sp2_s1),
            ("S2 +HR+MAP mean", sp2_s2),
        ],
        path=OUTPUT_DIR / "fig_sampen_m1_vs_m2.png",
    )

    # ── 7. GEE between-group comparison (SampEn vs topo metrics) ─────────────
    print("\n── GEE between-group comparison ────────────────────────────────")

    # Build patient-level stats dataframe (early / late / delta per metric)
    gee_base = (
        cohort[["stay_id", "T0", "group"]]
        .merge(feat, on=["stay_id", "T0"], how="inner")
    )
    gee_base["binary_group"] = (gee_base["group"] == "shock").astype(int)

    gee_metrics = [
        ("euler_hr",      "Euler χ(0)",        "euler_hr_early_mean",          "euler_hr_late_mean"),
        ("n_extrema_hr",  "n_extrema",          "n_extrema_hr_early_mean",      "n_extrema_hr_late_mean"),
        ("sampen_rel_m1", "SampEn(r=0.5,m=1)", "sampen_rel_m1_early_mean",     "sampen_rel_m1_late_mean"),
        ("sampen_std_m1", "SampEn(r=0.2,m=1)", "sampen_std_m1_early_mean",     "sampen_std_m1_late_mean"),
    ]

    def gee_p(data: pd.DataFrame, outcome: str) -> tuple[float, float]:
        tmp = data[["stay_id", "binary_group", outcome]].dropna().copy()
        if tmp.empty or tmp["binary_group"].nunique() < 2:
            return np.nan, np.nan
        try:
            res = GEE.from_formula(
                f"{outcome} ~ binary_group",
                groups="stay_id",
                data=tmp,
                family=Gaussian(),
                cov_struct=Exchangeable(),
            ).fit(maxiter=100)
            return float(res.params["binary_group"]), float(res.pvalues["binary_group"])
        except Exception:
            return np.nan, np.nan

    gee_rows = []
    raw_ps   = []
    for key, label, ecol, lcol in gee_metrics:
        tmp = gee_base.copy()
        tmp["delta"] = tmp[lcol] - tmp[ecol]

        shock   = tmp[tmp["group"] == "shock"]
        control = tmp[tmp["group"] == "control"]
        _, p_e = gee_p(tmp, ecol)
        _, p_l = gee_p(tmp, lcol)
        _, p_d = gee_p(tmp, "delta")

        row = {
            "Metric":          label,
            "Early_shock":     shock[ecol].dropna().mean(),
            "Early_control":   control[ecol].dropna().mean(),
            "Early_p_raw":     p_e,
            "Late_shock":      shock[lcol].dropna().mean(),
            "Late_control":    control[lcol].dropna().mean(),
            "Late_p_raw":      p_l,
            "Delta_shock":     shock["delta"].dropna().mean(),
            "Delta_control":   control["delta"].dropna().mean(),
            "Delta_p_raw":     p_d,
        }
        gee_rows.append(row)
        raw_ps.extend([p_e, p_l, p_d])

    gee_tbl = pd.DataFrame(gee_rows)

    # Holm correction across all 4 metrics × 3 windows = 12 tests
    all_p = np.array(raw_ps)
    corr  = np.full(len(all_p), np.nan)
    valid = ~np.isnan(all_p)
    if valid.any():
        corr[valid] = multipletests(all_p[valid], method="holm")[1]
    for j, suffix in enumerate(["Early", "Late", "Delta"]):
        gee_tbl[f"{suffix}_holm"] = corr[j::3]
        gee_tbl[f"{suffix}_holm_fmt"] = gee_tbl[f"{suffix}_holm"].map(
            lambda p: "<0.001" if not pd.isna(p) and p < 0.001 else (f"{p:.3f}" if not pd.isna(p) else "—")
        )

    gee_tbl.to_csv(OUTPUT_DIR / "tbl_sampen_gee.csv", index=False, encoding="utf-8-sig")

    # Print compact comparison
    print(f"\n  {'Metric':25s} {'Early shock':>12} {'Early ctrl':>11} {'Holm p':>8}"
          f"  {'Late shock':>11} {'Late ctrl':>10} {'Holm p':>8}"
          f"  {'Δ shock':>8} {'Δ ctrl':>7} {'Holm p':>8}")
    print("  " + "-" * 105)
    for _, r in gee_tbl.iterrows():
        print(f"  {r['Metric']:25s} "
              f"{r['Early_shock']:12.3f} {r['Early_control']:11.3f} {r['Early_holm_fmt']:>8}  "
              f"{r['Late_shock']:11.3f} {r['Late_control']:10.3f} {r['Late_holm_fmt']:>8}  "
              f"{r['Delta_shock']:8.3f} {r['Delta_control']:7.3f} {r['Delta_holm_fmt']:>8}")

    print(f"\n[Done] Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
