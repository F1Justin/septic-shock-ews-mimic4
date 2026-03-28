"""
Step 8: 非独立观测敏感性分析

与新版主分析保持一致:
  - 主要关注 late-window AC1-HR mean 与 early-to-late delta
  - GEE / 标准 logistic 比较
  - dedup subset
  - stay-level cluster bootstrap
  - conditional logistic（按 matched_pair_id）
"""

import logging
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "mimiciv" / "mimiciv.db"
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_FILE = LOG_DIR / "08_cluster_sensitivity.log"

BOOTSTRAP_N = 1000
RANDOM_SEED = 42
LATE_WINDOW_HOURS = 12
SEDATIVE_ITEMIDS = (221385, 221623, 221668, 222168, 225150, 229420)
BETABLOCKER_ITEMIDS = (221429, 225153, 225974)

LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def setup_logging() -> logging.Logger:
    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    log = logging.getLogger("sensitivity")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(logging.Formatter(fmt))
        log.addHandler(h)
    return log


log = setup_logging()


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


def format_p(p: float) -> str:
    if pd.isna(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def df_to_html(tbl: pd.DataFrame, caption: str) -> str:
    css = """
<style>
body { font-family: Arial, sans-serif; font-size: 13px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }
th, td { border: 1px solid #ccc; padding: 6px 10px; }
th { background: #2c3e50; color: white; }
tr:nth-child(even) { background: #f8f8f8; }
td.sig  { color: #c0392b; font-weight: bold; }
td.good { color: #27ae60; font-weight: bold; }
caption { font-size: 14px; font-weight: bold; margin-bottom: 6px; text-align: left; }
</style>
"""
    lines = [css, f"<table><caption>{caption}</caption><thead><tr>"]
    for col in tbl.columns:
        lines.append(f"<th>{col}</th>")
    lines.append("</tr></thead><tbody>")
    for _, row in tbl.iterrows():
        lines.append("<tr>")
        for col in tbl.columns:
            val = str(row[col])
            cell_cls = ""
            if col == "P" or col.endswith("(p)") or col.endswith("P"):
                try:
                    pf = 0.0 if val == "<0.001" else float(val)
                    if val == "<0.001" or pf < 0.05:
                        cell_cls = ' class="sig"'
                except ValueError:
                    pass
            if col == "CI excludes 0":
                cell_cls = ' class="good"' if val == "Yes" else ' class="sig"'
            lines.append(f"<td{cell_cls}>{val}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def load_ews_features() -> pd.DataFrame:
    wins = pd.read_parquet(PROJECT_ROOT / "data" / "ews_windows.parquet")
    pstats = pd.read_parquet(PROJECT_ROOT / "data" / "ews_patient_stats.parquet")
    cohort = pd.read_parquet(PROJECT_ROOT / "data" / "cohort.parquet").drop_duplicates(["stay_id", "T0"])

    late = (
        wins[
            (wins["hours_before_T0"] >= -12)
            & (~wins["low_conf_hr"])
        ]
        .groupby(["stay_id", "T0"], as_index=False)
        .agg(ac1_hr_late_mean=("ac1_hr", "mean"))
    )

    df = (
        pstats[[
            "stay_id", "T0", "group", "dominant_source",
            "early_hr_mean", "late_hr_mean", "delta_hr",
        ]]
        .merge(late, on=["stay_id", "T0"], how="left")
        .merge(cohort[["stay_id", "T0", "subject_id"]], on=["stay_id", "T0"], how="left")
    )
    return df


def pull_covariates(df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        con.register("_ids", df[["stay_id", "subject_id"]].drop_duplicates())
        demog = con.execute("""
            SELECT
                ie.stay_id,
                ie.first_careunit
            FROM _ids i
            INNER JOIN mimiciv_icu.icustays ie
                ON ie.stay_id = i.stay_id
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

        sed_ids = ",".join(map(str, SEDATIVE_ITEMIDS))
        sedation = con.execute(f"""
            SELECT DISTINCT s.stay_id, s.T0, 1 AS sedation_before_window
            FROM mimiciv_icu.inputevents ie
            INNER JOIN _st0 s ON ie.stay_id = s.stay_id
            WHERE ie.itemid IN ({sed_ids})
              AND COALESCE(ie.amount, 0) > 0
              AND ie.starttime < s.T0
              AND COALESCE(ie.endtime, ie.starttime) >= s.T0 - INTERVAL '12 hours'
        """).df()

        bb_ids = ",".join(map(str, BETABLOCKER_ITEMIDS))
        betablocker = con.execute(f"""
            SELECT DISTINCT s.stay_id, s.T0, 1 AS betablocker_before_window
            FROM mimiciv_icu.inputevents ie
            INNER JOIN _st0 s ON ie.stay_id = s.stay_id
            WHERE ie.itemid IN ({bb_ids})
              AND COALESCE(ie.amount, 0) > 0
              AND ie.starttime < s.T0
              AND COALESCE(ie.endtime, ie.starttime) >= s.T0 - INTERVAL '12 hours'
        """).df()
    finally:
        con.close()

    out = (
        df
        .merge(demog, on="stay_id", how="left")
        .merge(vent, on=["stay_id", "T0"], how="left")
        .merge(sedation, on=["stay_id", "T0"], how="left")
        .merge(betablocker, on=["stay_id", "T0"], how="left")
    )
    out["vent_before_window"] = out["vent_before_window"].fillna(0).astype(int)
    out["sedation_before_window"] = out["sedation_before_window"].fillna(0).astype(int)
    out["betablocker_before_window"] = out["betablocker_before_window"].fillna(0).astype(int)
    out["binary_group"] = (out["group"] == "shock").astype(int)
    out["icu_type"] = out["first_careunit"].apply(simplify_careunit)
    out["monitoring"] = out["dominant_source"].str.lower().fillna("nbp")
    return out.dropna(subset=["ac1_hr_late_mean"])


FORMULA_ADJ = (
    "binary_group ~ ac1_hr_late_mean + vent_before_window "
    "+ sedation_before_window + betablocker_before_window "
    "+ C(icu_type, Treatment('MICU')) "
    "+ C(monitoring, Treatment('nbp'))"
)
FORMULA_UNADJ = "binary_group ~ ac1_hr_late_mean"


def run_dedup_tests(df: pd.DataFrame) -> tuple[dict, dict]:
    rng = np.random.default_rng(RANDOM_SEED)
    df2 = df.copy()
    df2["_rnd"] = rng.random(len(df2))
    dedup = (
        df2.sort_values("_rnd")
        .groupby("stay_id", as_index=False)
        .first()
        .drop(columns="_rnd")
        .reset_index(drop=True)
    )

    def summarize(data: pd.DataFrame, label: str) -> dict:
        s = data[data["binary_group"] == 1]
        c = data[data["binary_group"] == 0]
        _, p_late = stats.mannwhitneyu(s["ac1_hr_late_mean"], c["ac1_hr_late_mean"], alternative="two-sided")
        _, p_delta = stats.mannwhitneyu(s["delta_hr"].dropna(), c["delta_hr"].dropna(), alternative="two-sided")
        res = smf.logit(FORMULA_ADJ, data=data).fit(disp=False, maxiter=200)
        coef = res.params["ac1_hr_late_mean"]
        ci = res.conf_int().loc["ac1_hr_late_mean"]
        return {
            "label": label,
            "n_total": len(data),
            "late_shock": f"{s['ac1_hr_late_mean'].median():.3f}",
            "late_control": f"{c['ac1_hr_late_mean'].median():.3f}",
            "late_p": p_late,
            "delta_shock": f"{s['delta_hr'].median():.3f}",
            "delta_control": f"{c['delta_hr'].median():.3f}",
            "delta_p": p_delta,
            "logit_or": f"{np.exp(coef):.2f} ({np.exp(ci[0]):.2f}–{np.exp(ci[1]):.2f})",
            "logit_p": float(res.pvalues["ac1_hr_late_mean"]),
        }

    return summarize(df, "Full set"), summarize(dedup, "Dedup subset")


def build_dedup_table(full: dict, dedup: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Statistic": "Late-window AC1-HR, shock median",
            "Full": full["late_shock"],
            "Dedup": dedup["late_shock"],
            "Full P": format_p(full["late_p"]),
            "Dedup P": format_p(dedup["late_p"]),
        },
        {
            "Statistic": "Late-window AC1-HR, control median",
            "Full": full["late_control"],
            "Dedup": dedup["late_control"],
            "Full P": "—",
            "Dedup P": "—",
        },
        {
            "Statistic": "Early-to-late delta, shock median",
            "Full": full["delta_shock"],
            "Dedup": dedup["delta_shock"],
            "Full P": format_p(full["delta_p"]),
            "Dedup P": format_p(dedup["delta_p"]),
        },
        {
            "Statistic": "Early-to-late delta, control median",
            "Full": full["delta_control"],
            "Dedup": dedup["delta_control"],
            "Full P": "—",
            "Dedup P": "—",
        },
        {
            "Statistic": "Adjusted logistic OR for AC1 late mean",
            "Full": full["logit_or"],
            "Dedup": dedup["logit_or"],
            "Full P": format_p(full["logit_p"]),
            "Dedup P": format_p(dedup["logit_p"]),
        },
    ])


def analysis_gee(df: pd.DataFrame) -> pd.DataFrame:
    std_unadj = smf.logit(FORMULA_UNADJ, data=df).fit(disp=False, maxiter=200)
    std_adj = smf.logit(FORMULA_ADJ, data=df).fit(disp=False, maxiter=200)
    gee_unadj = GEE.from_formula(
        FORMULA_UNADJ,
        groups=df["stay_id"].values,
        data=df,
        family=Binomial(),
        cov_struct=Exchangeable(),
    ).fit(maxiter=100)
    gee_adj = GEE.from_formula(
        FORMULA_ADJ,
        groups=df["stay_id"].values,
        data=df,
        family=Binomial(),
        cov_struct=Exchangeable(),
    ).fit(maxiter=100)

    def row(name: str, res, var: str = "ac1_hr_late_mean") -> dict:
        ci = res.conf_int().loc[var]
        return {
            "Model": name,
            "OR (95% CI)": f"{np.exp(res.params[var]):.2f} ({np.exp(ci[0]):.2f}–{np.exp(ci[1]):.2f})",
            "P": format_p(res.pvalues[var]),
        }

    return pd.DataFrame([
        row("Standard logistic (unadjusted)", std_unadj),
        row("Standard logistic (adjusted)", std_adj),
        row("GEE logistic (unadjusted)", gee_unadj),
        row("GEE logistic (adjusted)", gee_adj),
    ])


def analysis_bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    shock_stays = np.sort(df.loc[df["binary_group"] == 1, "stay_id"].unique())
    control_stays = np.sort(df.loc[df["binary_group"] == 0, "stay_id"].unique())
    stay_blocks = {sid: grp for sid, grp in df.groupby("stay_id", sort=False)}
    rng = np.random.default_rng(RANDOM_SEED)

    delta_late = np.empty(BOOTSTRAP_N)
    delta_early = np.empty(BOOTSTRAP_N)
    delta_change = np.empty(BOOTSTRAP_N)

    for i in range(BOOTSTRAP_N):
        sampled_shock = rng.choice(shock_stays, size=len(shock_stays), replace=True)
        sampled_control = rng.choice(control_stays, size=len(control_stays), replace=True)
        bs = pd.concat(
            [*(stay_blocks[sid] for sid in sampled_shock), *(stay_blocks[sid] for sid in sampled_control)],
            ignore_index=True,
        )
        s = bs[bs["binary_group"] == 1]
        c = bs[bs["binary_group"] == 0]
        delta_late[i] = s["ac1_hr_late_mean"].mean() - c["ac1_hr_late_mean"].mean()
        delta_early[i] = s["early_hr_mean"].mean() - c["early_hr_mean"].mean()
        delta_change[i] = s["delta_hr"].mean() - c["delta_hr"].mean()

    s = df[df["binary_group"] == 1]
    c = df[df["binary_group"] == 0]

    def row(name: str, obs: float, arr: np.ndarray) -> dict:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return {
            "Statistic": name,
            "Observed Δ": f"{obs:.4f}",
            "Bootstrap 95% CI": f"[{lo:.4f}, {hi:.4f}]",
            "CI excludes 0": "Yes" if (lo > 0 or hi < 0) else "No",
        }

    return pd.DataFrame([
        row("Δ Late-window AC1-HR mean", s["ac1_hr_late_mean"].mean() - c["ac1_hr_late_mean"].mean(), delta_late),
        row("Δ Early-window AC1-HR mean", s["early_hr_mean"].mean() - c["early_hr_mean"].mean(), delta_early),
        row("Δ Early-to-late AC1-HR change", s["delta_hr"].mean() - c["delta_hr"].mean(), delta_change),
    ])


def analysis_conditional_logistic(df: pd.DataFrame) -> pd.DataFrame:
    cohort = pd.read_parquet(PROJECT_ROOT / "data" / "cohort.parquet")
    pair_df = cohort[["stay_id", "T0", "matched_pair_id"]].drop_duplicates(["stay_id", "T0"])
    df2 = df.merge(pair_df, on=["stay_id", "T0"], how="left").copy()
    df2 = df2.dropna(subset=["matched_pair_id", "ac1_hr_late_mean"])

    df2["icu_SICU"] = (df2["icu_type"] == "SICU").astype(float)
    df2["icu_CCU_CSRU"] = (df2["icu_type"] == "CCU_CSRU").astype(float)
    df2["icu_Neuro"] = (df2["icu_type"] == "Neuro_NICU").astype(float)
    df2["icu_Other"] = (df2["icu_type"] == "Other").astype(float)
    df2["mon_abp"] = (df2["monitoring"] == "abp").astype(float)
    df2["mon_mixed"] = (df2["monitoring"] == "mixed").astype(float)

    covars = [
        "ac1_hr_late_mean",
        "vent_before_window",
        "sedation_before_window",
        "betablocker_before_window",
        "icu_SICU", "icu_CCU_CSRU", "icu_Neuro", "icu_Other",
        "mon_abp", "mon_mixed",
    ]

    pair_balance = df2.groupby("matched_pair_id")["binary_group"].agg(shock="sum", total="size")
    valid_pairs = pair_balance[(pair_balance["shock"] >= 1) & (pair_balance["shock"] < pair_balance["total"])].index
    df2 = df2[df2["matched_pair_id"].isin(valid_pairs)].copy()

    result = ConditionalLogit(
        df2["binary_group"].astype(int).values,
        df2[covars].astype(float).values,
        groups=df2["matched_pair_id"].values,
    ).fit(method="bfgs", disp=False, maxiter=500)

    cis = result.conf_int()
    rows = []
    for i, var in enumerate(covars):
        rows.append(
            {
                "Variable": var,
                "OR": f"{np.exp(result.params[i]):.3f}",
                "95% CI": f"({np.exp(cis[i, 0]):.3f}–{np.exp(cis[i, 1]):.3f})",
                "P": format_p(float(result.pvalues[i])),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    log.info("=" * 70)
    log.info("Step 8: Cluster sensitivity analysis")
    log.info("=" * 70)

    df = pull_covariates(load_ews_features())
    log.info(f"Analysis dataset: {len(df):,} rows")

    full_res, dedup_res = run_dedup_tests(df)
    tbl_dedup = build_dedup_table(full_res, dedup_res)
    tbl_gee = analysis_gee(df)
    tbl_boot = analysis_bootstrap(df)
    tbl_clogit = analysis_conditional_logistic(df)

    outputs = {
        "tableS1_gee": tbl_gee,
        "tableS2_dedup": tbl_dedup,
        "tableS3_bootstrap": tbl_boot,
        "tableS4_conditional_logistic": tbl_clogit,
    }

    for name, tbl in outputs.items():
        csv_path = OUTPUT_DIR / f"{name}.csv"
        html_path = OUTPUT_DIR / f"{name}.html"
        tbl.to_csv(csv_path, index=False, encoding="utf-8-sig")
        html_path.write_text(df_to_html(tbl, name), encoding="utf-8")
        log.info(f"Saved → {csv_path.name}, {html_path.name}")

    log.info("\nTABLE S1\n" + tbl_gee.to_string(index=False))
    log.info("\nTABLE S2\n" + tbl_dedup.to_string(index=False))
    log.info("\nTABLE S3\n" + tbl_boot.to_string(index=False))
    log.info("\nTABLE S4\n" + tbl_clogit.to_string(index=False))
    log.info("Step 8 complete.")


if __name__ == "__main__":
    main()
