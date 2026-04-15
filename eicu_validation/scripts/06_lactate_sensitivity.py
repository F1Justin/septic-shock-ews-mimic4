"""
eICU 外部验证 — Step 6: Lactate-restricted sensitivity analysis

策略：在现有 587 matched pairs 中，仅保留 shock 病例在 T0±24h 内
有 lactate > 2 mmol/L 的 pair（保留原始匹配关系不变），
重跑 M1 和 T0 分层模型，验证 surrogate definition 的结果稳健性。

输出：
  output/tbl_lactate_sensitivity.csv
  output/tbl_lactate_cohort_summary.csv
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent
NAAS_ROOT    = PROJECT_ROOT.parent
EICU_DIR     = NAAS_ROOT / "eicu-crd" / "2.0"
OUTPUT_DIR   = PROJECT_ROOT / "output"

LACTATE_THRESH = 2.0
COEXIST_H      = 24

BASE_COVARS = [
    "vent_before_window", "sedation_before_window",
    "betablocker_before_window", "mon_abp",
]


def fmt_or(o, lo, hi):
    return f"{o:.2f} ({lo:.2f}\u2013{hi:.2f})"

def fmt_p(p):
    if pd.isna(p):
        return "\u2014"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fit_clogit(data, predictor, extra_covars=None, label=""):
    extra  = extra_covars or []
    covars = list(extra) + list(BASE_COVARS)
    all_vars = [predictor] + covars

    df2 = data.dropna(subset=["matched_pair_id", *all_vars]).copy()
    pair_bal = df2.groupby("matched_pair_id")["binary_group"].agg(
        shock="sum", total="size"
    )
    valid = pair_bal[
        (pair_bal["shock"] >= 1) & (pair_bal["shock"] < pair_bal["total"])
    ].index
    df2 = df2[df2["matched_pair_id"].isin(valid)].copy()

    if len(df2) < 20 or df2["binary_group"].sum() < 5:
        return pd.DataFrame()

    X = df2[all_vars].astype(float).values
    y = df2["binary_group"].astype(int).values
    g = df2["matched_pair_id"].values

    try:
        model  = ConditionalLogit(y, X, groups=g)
        result = model.fit(method="bfgs", disp=False, maxiter=500)
        cis    = result.conf_int()
    except Exception as e:
        print(f"  [warn] {label}: {e}")
        return pd.DataFrame()

    rows = []
    for i, var in enumerate(all_vars):
        rows.append({
            "Variable": var,
            "OR": np.exp(result.params[i]),
            "CI_lo": np.exp(cis[i, 0]),
            "CI_hi": np.exp(cis[i, 1]),
            "P": float(result.pvalues[i]),
            "N_obs": len(df2),
            "N_strata": len(valid),
            "model": label,
        })
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load original cohort & modelling data (same as 04) ─────────────
    print("── 加载数据 ──────────────────────────────────────────────────────")
    cohort  = pd.read_parquet(OUTPUT_DIR / "cohort_eicu.parquet")
    stats   = pd.read_parquet(OUTPUT_DIR / "ews_patient_eicu.parquet")
    diag    = pd.read_parquet(OUTPUT_DIR / "cleaning_diagnostics_eicu.parquet")
    vitals  = pd.read_parquet(OUTPUT_DIR / "vitals_eicu.parquet")

    passed = diag[~diag["excluded"]][["patientunitstayid", "T0_min"]]
    cohort = cohort.merge(passed, on=["patientunitstayid", "T0_min"])
    stats  = stats.merge(passed, on=["patientunitstayid", "T0_min"])

    orig_shock = cohort[cohort["group"] == "shock"]
    print(f"  Original analysable shock: {len(orig_shock)}")
    print(f"  Original analysable pairs: {cohort['matched_pair_id'].nunique()}")

    # ── 2. Load lactate, identify qualified shock cases ───────────────────
    print(f"\n── Lactate 筛选 (>{LACTATE_THRESH}, ±{COEXIST_H}h of T0) ──────")
    lab = pd.read_csv(
        EICU_DIR / "lab.csv.gz",
        usecols=["patientunitstayid", "labname", "labresult", "labresultoffset"],
    )
    lac = lab[lab["labname"] == "lactate"].copy()
    lac["labresult"] = pd.to_numeric(lac["labresult"], errors="coerce")
    lac = lac.dropna(subset=["labresult"])

    lac_shock = lac.merge(
        orig_shock[["patientunitstayid", "T0_min"]].drop_duplicates(),
        on="patientunitstayid", how="inner",
    )
    lac_shock["dt_h"] = (lac_shock["labresultoffset"] - lac_shock["T0_min"]) / 60.0

    qualified_ids = set(
        lac_shock[
            (lac_shock["dt_h"].abs() <= COEXIST_H) &
            (lac_shock["labresult"] > LACTATE_THRESH)
        ]["patientunitstayid"]
    )
    has_any_lac = set(lac[lac["patientunitstayid"].isin(
        set(orig_shock["patientunitstayid"]))]["patientunitstayid"])

    n_orig = len(orig_shock)
    n_no_lac = n_orig - len(has_any_lac)
    n_lac_low = len(has_any_lac) - len(qualified_ids & has_any_lac)
    n_qual = len(qualified_ids)
    print(f"  Has lactate > {LACTATE_THRESH} in window: {n_qual} / {n_orig} ({n_qual/n_orig*100:.1f}%)")
    print(f"  No lactate data at all: {n_no_lac}")
    print(f"  Has lactate but ≤ {LACTATE_THRESH} in window: {n_lac_low}")

    # ── 3. Subset existing pairs (keep both shock + matched control) ──────
    print("\n── 子集筛选（保留原始匹配关系）──────────────────────────────────")
    qualified_pairs = set(
        cohort[
            (cohort["group"] == "shock") &
            (cohort["patientunitstayid"].isin(qualified_ids))
        ]["matched_pair_id"]
    )
    cohort_lac = cohort[cohort["matched_pair_id"].isin(qualified_pairs)].copy()
    print(f"  Retained pairs: {len(qualified_pairs)}")
    print(f"  Retained observations: {len(cohort_lac)} "
          f"(shock {(cohort_lac['group']=='shock').sum()}, "
          f"control {(cohort_lac['group']=='control').sum()})")

    # ── 4. Build modelling dataframe (same pipeline as 04) ────────────────
    print("\n── 组装建模数据 ─────────────────────────────────────────────────")
    sampen_cols = [c for c in ["sampen_rel_hr_late_mean"] if c in stats.columns]

    late_raw = (
        vitals[
            (vitals["hours_before_T0"] >= -12) &
            (vitals["hours_before_T0"] < 0)
        ]
        .groupby(["patientunitstayid", "T0_min"], as_index=False)
        .agg(late_hr_mean_raw=("hr_raw", "mean"), late_map_mean_raw=("map_raw", "mean"))
    )

    mdf = (
        cohort_lac
        .merge(stats[["patientunitstayid", "T0_min",
                       "ac1_hr_late_mean", "ac1_hr_early_mean",
                       "euler_hr_late_mean", "euler_hr_early_mean",
                       "n_extrema_hr_late_mean"] + sampen_cols],
               on=["patientunitstayid", "T0_min"], how="inner")
        .merge(late_raw, on=["patientunitstayid", "T0_min"], how="left")
        .merge(diag[["patientunitstayid", "T0_min", "dominant_map_source"]],
               on=["patientunitstayid", "T0_min"], how="left")
    )
    mdf["binary_group"] = (mdf["group"] == "shock").astype(int)
    mdf["mon_abp"] = (mdf["dominant_map_source"] == "abp").astype(float)
    for col in BASE_COVARS:
        if col in mdf.columns:
            mdf[col] = mdf[col].fillna(0).astype(float)

    n_shock = mdf["binary_group"].sum()
    n_ctrl  = len(mdf) - n_shock
    n_pairs_final = mdf["matched_pair_id"].nunique()
    print(f"  Analysable: {len(mdf)} (shock {n_shock}, control {n_ctrl}), "
          f"pairs {n_pairs_final}")

    # ── 5. Run models ────────────────────────────────────────────────────
    print("\n── 条件 logistic 模型 ───────────────────────────────────────────")

    METRICS = [
        ("ac1_hr_late_mean",       "AC1"),
        ("n_extrema_hr_late_mean", "n_extrema"),
        ("euler_hr_late_mean",     "Euler"),
    ]
    if "sampen_rel_hr_late_mean" in mdf.columns:
        METRICS.append(("sampen_rel_hr_late_mean", "SampEn(r=0.5)"))

    T0_THRESHOLDS = [
        ("全样本", None),
        ("T0≥24h", 24),
        ("T0≥48h", 48),
        ("T0≥72h", 72),
    ]

    MIMIC_REF = {"AC1": 1.60, "n_extrema": 0.88, "Euler": 0.75, "SampEn(r=0.5)": 0.86}

    orig_t0 = pd.read_csv(OUTPUT_DIR / "tbl_t0_sensitivity.csv")

    all_rows = []
    for strat_label, th in T0_THRESHOLDS:
        sub = mdf if th is None else mdf[mdf["T0_h"] >= th].copy()
        n_sub_pairs = sub["matched_pair_id"].nunique()
        n_sub_shock = sub["binary_group"].sum()
        print(f"\n  [{strat_label}] obs={len(sub)}, shock={n_sub_shock}, pairs≤{n_sub_pairs}")

        for metric, mname in METRICS:
            r_m1 = fit_clogit(sub, metric, label=f"{mname} {strat_label} M1")
            r_s1 = fit_clogit(sub, metric, extra_covars=["late_hr_mean_raw"],
                              label=f"{mname} {strat_label} S1")

            for model_tag, r_df in [("M1", r_m1), ("S1", r_s1)]:
                # Look up original result
                orig_match = orig_t0[
                    (orig_t0["metric"] == mname) &
                    (orig_t0["T0_strat"] == strat_label)
                ]
                orig_or_str = orig_match.iloc[0]["OR_fmt"] if not orig_match.empty else "—"
                orig_p_str  = orig_match.iloc[0]["P_fmt"]  if not orig_match.empty else "—"

                if not r_df.empty:
                    r = r_df[r_df["Variable"] == metric].iloc[0]
                    sig = " *" if r["P"] < 0.05 else ""
                    print(f"    {mname:12s} {model_tag}: OR={r['OR']:.2f} "
                          f"({r['CI_lo']:.2f}\u2013{r['CI_hi']:.2f}) "
                          f"p={fmt_p(r['P'])}{sig}  "
                          f"N={int(r['N_obs'])}, strata={int(r['N_strata'])}  "
                          f"[orig M1: {orig_or_str}]")
                    all_rows.append({
                        "T0_strat": strat_label, "metric": mname, "model": model_tag,
                        "N_pairs": n_sub_pairs,
                        "N_informative_strata": int(r["N_strata"]),
                        "N_obs": int(r["N_obs"]),
                        "OR": r["OR"], "CI_lo": r["CI_lo"], "CI_hi": r["CI_hi"],
                        "P": r["P"],
                        "OR_fmt": fmt_or(r["OR"], r["CI_lo"], r["CI_hi"]),
                        "P_fmt": fmt_p(r["P"]),
                        "orig_OR_fmt": str(orig_or_str),
                        "orig_P_fmt": str(orig_p_str),
                        "MIMIC_ref_OR": MIMIC_REF.get(mname, np.nan),
                    })
                else:
                    print(f"    {mname:12s} {model_tag}: — (model failed / too few)  "
                          f"[orig M1: {orig_or_str}]")
                    all_rows.append({
                        "T0_strat": strat_label, "metric": mname, "model": model_tag,
                        "N_pairs": n_sub_pairs,
                        "N_informative_strata": np.nan, "N_obs": np.nan,
                        "OR": np.nan, "CI_lo": np.nan, "CI_hi": np.nan, "P": np.nan,
                        "OR_fmt": "—", "P_fmt": "—",
                        "orig_OR_fmt": str(orig_or_str), "orig_P_fmt": str(orig_p_str),
                        "MIMIC_ref_OR": MIMIC_REF.get(mname, np.nan),
                    })

    result_tbl = pd.DataFrame(all_rows)
    result_tbl.to_csv(OUTPUT_DIR / "tbl_lactate_sensitivity.csv",
                      index=False, encoding="utf-8-sig")

    # Cohort summary
    summary = pd.DataFrame([{
        "original_analysable_pairs": cohort["matched_pair_id"].nunique(),
        "shock_with_any_lactate": len(has_any_lac & set(orig_shock["patientunitstayid"])),
        "shock_with_lactate_gt2": n_qual,
        "retained_pairs": len(qualified_pairs),
        "analysable_pairs": n_pairs_final,
        "pct_retained": f"{n_qual/n_orig*100:.1f}%",
    }])
    summary.to_csv(OUTPUT_DIR / "tbl_lactate_cohort_summary.csv",
                   index=False, encoding="utf-8-sig")

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("SUMMARY: Lactate-restricted sensitivity (M1 only)")
    print("=" * 75)
    m1_only = result_tbl[result_tbl["model"] == "M1"].copy()
    for _, r in m1_only.iterrows():
        lac_str = f"{r['OR_fmt']:24s} p={r['P_fmt']:6s}" if r["OR_fmt"] != "—" else "— (convergence failure)"
        print(f"  {r['T0_strat']:8s} {r['metric']:12s}: "
              f"LAC {lac_str}  |  ORIG {r['orig_OR_fmt']}  p={r['orig_P_fmt']}")

    print(f"\n  Cohort: {n_orig} → lactate-qualified {n_qual} ({n_qual/n_orig*100:.1f}%) → "
          f"analysable pairs {n_pairs_final}")
    print(f"\n[完成] → tbl_lactate_sensitivity.csv, tbl_lactate_cohort_summary.csv")


if __name__ == "__main__":
    main()
