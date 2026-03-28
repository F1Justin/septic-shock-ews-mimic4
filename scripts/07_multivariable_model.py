"""
Step 7 (T1.2): HR AC1 主分析 — Conditional Logistic Regression

目的
----
  在 1:2 risk-set matching 设计下, 用 conditional logistic regression
  检验 late-centered-window HR AC1 与 shock 的关联是否仍然存在。

分析集
------
  仅纳入 excluded=False 的 analysable (stay_id, T0) 对。
  同一 stay_id 可在不同 T0 多次出现；主模型按 matched_pair_id 分层，
  因而显式尊重匹配设计。stay-level GEE 与标准 logistic 见 Step 8 (T1.3)。

主模型
------
  shock ~ ac1_hr_late_mean + vent_before_window + icu_type + monitoring
  strata = matched_pair_id

  说明:
    - 主自变量: late-centred rolling-window HR AC1 mean
      (windows indexed by midpoint time; selected where hours_before_T0 ∈ [-12, 0),
      with realised centers extending to about -5h)
    - vent_before_window: 晚期窗口起点 (-12h) 时是否已处于有创机械通气
    - icu_type / monitoring: 作为层内 care-setting 协变量
    - 年龄、性别、admission SOFA、time-to-event 匹配结构由条件似然吸收，
      不单独输出系数

输出
----
  output/table2_multivariable.csv   -- 主表
  output/table2_multivariable.html  -- HTML 预览
  output/table2_diagnostics.csv     -- 模型摘要
  logs/07_multivariable_model.log   -- 运行日志

用法: python scripts/07_multivariable_model.py
"""

import argparse
import sys
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit
from scipy import stats

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH      = PROJECT_ROOT / "mimiciv" / "mimiciv.db"
VITALS_PATH  = PROJECT_ROOT / "data" / "vitals_cleaned.parquet"
LOG_DIR      = PROJECT_ROOT / "logs"
OUTPUT_DIR   = PROJECT_ROOT / "output"
LATE_WINDOW_HOURS = 12
SEDATIVE_ITEMIDS = (221385, 221623, 221668, 222168, 225150, 229420)
BETABLOCKER_ITEMIDS = (221429, 225153, 225974)

LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "07_multivariable_model.log"

# ── 日志 ──────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    log = logging.getLogger("multivariable")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(logging.Formatter(fmt))
        log.addHandler(h)
    return log


log = setup_logging()


def tagged_path(path: Path, tag: str) -> Path:
    if not tag:
        return path
    return path.with_name(f"{path.stem}{tag}{path.suffix}")

# ── ICU type 简化 ──────────────────────────────────────────────────────────────
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


# ── 数据加载与特征构建 ────────────────────────────────────────────────────────

def load_ews_features(suffix: str = "") -> pd.DataFrame:
    """
    从 ews_windows 计算 early/late-centered-window mean AC1-HR，
    从 ews_patient_stats 获取 group / dominant_source / early-late 摘要，
    再与 cohort 合并得到 subject_id, sofa_admission。
    """
    wins   = pd.read_parquet(tagged_path(PROJECT_ROOT / "data" / "ews_windows.parquet", suffix))
    pstats = pd.read_parquet(tagged_path(PROJECT_ROOT / "data" / "ews_patient_stats.parquet", suffix))
    cohort = pd.read_parquet(PROJECT_ROOT / "data" / "cohort.parquet")
    cohort = cohort.drop_duplicates(["stay_id", "T0"])

    # early/late-centred windows: 仅保留 HR 置信度充足的窗口
    early = (
        wins[
            (wins["hours_before_T0"] >= -48) &
            (wins["hours_before_T0"] < -24) &
            (~wins["low_conf_hr"])
        ]
        .groupby(["stay_id", "T0"])
        .agg(
            ac1_hr_early_mean=("ac1_hr", "mean"),
            ac1_hr_early_n=("ac1_hr", "count"),
        )
        .reset_index()
    )
    late = (
        wins[
            (wins["hours_before_T0"] >= -12) &
            (wins["hours_before_T0"] < 0) &
            (~wins["low_conf_hr"])
        ]
        .groupby(["stay_id", "T0"])
        .agg(
            ac1_hr_late_mean=("ac1_hr", "mean"),
            ac1_hr_late_n=("ac1_hr", "count"),
        )
        .reset_index()
    )
    log.info(f"Early-window AC1-HR computed for {len(early):,} pairs "
             f"(mean={early['ac1_hr_early_mean'].mean():.3f})")
    log.info(f"Late-window AC1-HR computed for {len(late):,} pairs "
             f"(mean={late['ac1_hr_late_mean'].mean():.3f})")

    # 合并 pstats (group, dominant_source, early-late 摘要)
    df = pstats[[
        "stay_id", "T0", "group", "dominant_source",
        "early_hr_mean", "late_hr_mean", "delta_hr",
        "early_map_mean", "late_map_mean", "delta_map",
    ]].merge(
        early[["stay_id", "T0", "ac1_hr_early_mean", "ac1_hr_early_n"]],
        on=["stay_id", "T0"], how="left"
    ).merge(
        late[["stay_id", "T0", "ac1_hr_late_mean", "ac1_hr_late_n"]],
        on=["stay_id", "T0"], how="left"
    )

    # 合并 cohort → subject_id, sofa_admission
    df = df.merge(
        cohort[["stay_id", "T0", "subject_id", "sofa_admission"]],
        on=["stay_id", "T0"], how="left"
    )

    log.info(f"EWS feature table: {len(df):,} rows, "
             f"shock={int((df['group']=='shock').sum()):,}, "
             f"control={int((df['group']=='control').sum()):,}")
    return df


def pull_demographics(con: duckdb.DuckDBPyConnection,
                      df: pd.DataFrame) -> pd.DataFrame:
    """年龄、性别、ICU类型，按 stay_id 查询。"""
    con.register("_ids", df[["stay_id", "subject_id"]].drop_duplicates())
    result = con.execute("""
        SELECT
            ie.stay_id,
            p.anchor_age      AS age,
            p.gender,
            ie.first_careunit
        FROM _ids i
        INNER JOIN mimiciv_icu.icustays      ie ON ie.stay_id    = i.stay_id
        INNER JOIN mimiciv_hosp.patients      p ON p.subject_id  = i.subject_id
    """).df()
    log.info(f"Demographics: {len(result):,} rows")
    return result


def pull_vent_before_window(con: duckdb.DuckDBPyConnection,
                            df: pd.DataFrame) -> pd.DataFrame:
    """晚期窗口起点 (-12h) 时是否已处于有创机械通气状态。"""
    con.register("_st0", df[["stay_id", "T0"]])
    result = con.execute("""
        SELECT DISTINCT s.stay_id, s.T0, 1 AS vent_before_window
        FROM mimiciv_derived.ventilation v
        INNER JOIN _st0 s ON v.stay_id = s.stay_id
        WHERE v.ventilation_status = 'InvasiveVent'
          AND v.starttime <= s.T0 - INTERVAL '12 hours'
          AND (v.endtime IS NULL OR v.endtime >= s.T0 - INTERVAL '12 hours')
    """).df()
    log.info(f"InvasiveVent at -12h boundary: {len(result):,} stay-T0 pairs")
    return result


def pull_vaso_before_t0(con: duckdb.DuckDBPyConnection,
                        df: pd.DataFrame) -> pd.DataFrame:
    """T0 前是否有升压药记录 (NED > 0, starttime < T0)。"""
    con.register("_st0v", df[["stay_id", "T0"]])
    result = con.execute("""
        SELECT DISTINCT s.stay_id, s.T0, 1 AS vaso_before_t0
        FROM mimiciv_derived.norepinephrine_equivalent_dose n
        INNER JOIN _st0v s ON n.stay_id = s.stay_id
        WHERE n.norepinephrine_equivalent_dose > 0
          AND n.starttime < s.T0
    """).df()
    log.info(f"Vasopressor before T0: {len(result):,} stay-T0 pairs")
    return result


def pull_sedation_before_window(con: duckdb.DuckDBPyConnection,
                                df: pd.DataFrame) -> pd.DataFrame:
    """T0 前 12h 内是否有常见镇静剂输入。"""
    itemids = ",".join(map(str, SEDATIVE_ITEMIDS))
    con.register("_st0_sed", df[["stay_id", "T0"]].drop_duplicates())
    result = con.execute(f"""
        SELECT DISTINCT s.stay_id, s.T0, 1 AS sedation_before_window
        FROM mimiciv_icu.inputevents ie
        INNER JOIN _st0_sed s ON ie.stay_id = s.stay_id
        WHERE ie.itemid IN ({itemids})
          AND COALESCE(ie.amount, 0) > 0
          AND ie.starttime < s.T0
          AND COALESCE(ie.endtime, ie.starttime) >= s.T0 - INTERVAL '12 hours'
    """).df()
    log.info(f"Sedation before window: {len(result):,} stay-T0 pairs")
    return result


def pull_betablocker_before_window(con: duckdb.DuckDBPyConnection,
                                   df: pd.DataFrame) -> pd.DataFrame:
    """T0 前 12h 内是否有常见 beta-blocker 输入。"""
    itemids = ",".join(map(str, BETABLOCKER_ITEMIDS))
    con.register("_st0_bb", df[["stay_id", "T0"]].drop_duplicates())
    result = con.execute(f"""
        SELECT DISTINCT s.stay_id, s.T0, 1 AS betablocker_before_window
        FROM mimiciv_icu.inputevents ie
        INNER JOIN _st0_bb s ON ie.stay_id = s.stay_id
        WHERE ie.itemid IN ({itemids})
          AND COALESCE(ie.amount, 0) > 0
          AND ie.starttime < s.T0
          AND COALESCE(ie.endtime, ie.starttime) >= s.T0 - INTERVAL '12 hours'
    """).df()
    log.info(f"Beta-blocker before window: {len(result):,} stay-T0 pairs")
    return result


def build_model_data(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """合并所有协变量，构建建模用 DataFrame。"""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        demog = pull_demographics(con, df)
        vent  = pull_vent_before_window(con, df)
        sedation = pull_sedation_before_window(con, df)
        betablocker = pull_betablocker_before_window(con, df)
    finally:
        con.close()

    vitals = pd.read_parquet(
        tagged_path(VITALS_PATH, suffix),
        columns=["stay_id", "T0", "charttime", "map_raw", "hr_raw"],
    )
    vitals["charttime"] = pd.to_datetime(vitals["charttime"])
    vitals["T0"] = pd.to_datetime(vitals["T0"])
    early_vitals = vitals.loc[
        (vitals["charttime"] > vitals["T0"] - pd.Timedelta(hours=48)) &
        (vitals["charttime"] <= vitals["T0"] - pd.Timedelta(hours=24))
    ].groupby(["stay_id", "T0"], as_index=False).agg(
        early_map_mean_raw=("map_raw", "mean"),
        early_hr_mean_raw=("hr_raw", "mean"),
    )
    late_vitals = vitals.loc[
        (vitals["charttime"] > vitals["T0"] - pd.Timedelta(hours=LATE_WINDOW_HOURS)) &
        (vitals["charttime"] <= vitals["T0"])
    ].groupby(["stay_id", "T0"], as_index=False).agg(
        late_map_mean_raw=("map_raw", "mean"),
        late_hr_mean_raw=("hr_raw", "mean"),
    )

    mdf = (
        df
        .merge(demog, on="stay_id", how="left")
        .merge(vent,  on=["stay_id", "T0"], how="left")
        .merge(sedation, on=["stay_id", "T0"], how="left")
        .merge(betablocker, on=["stay_id", "T0"], how="left")
        .merge(early_vitals, on=["stay_id", "T0"], how="left")
        .merge(late_vitals, on=["stay_id", "T0"], how="left")
    )
    mdf["vent_before_window"] = mdf["vent_before_window"].fillna(0).astype(int)
    mdf["sedation_before_window"] = mdf["sedation_before_window"].fillna(0).astype(int)
    mdf["betablocker_before_window"] = mdf["betablocker_before_window"].fillna(0).astype(int)
    mdf["male"]           = (mdf["gender"] == "M").astype(int)
    mdf["binary_group"]   = (mdf["group"] == "shock").astype(int)
    mdf["age_10yr"]       = mdf["age"] / 10.0
    mdf["icu_type"]       = mdf["first_careunit"].apply(simplify_careunit)
    # monitoring: 统一为小写, 保持 abp/nbp/mixed
    mdf["monitoring"]     = mdf["dominant_source"].str.lower().fillna("nbp")

    log.info(f"Model data: {len(mdf):,} rows")
    log.info(f"  binary_group: shock={mdf['binary_group'].sum():,}, "
             f"control={(~mdf['binary_group'].astype(bool)).sum():,}")
    log.info(f"  vent_before_window: {mdf['vent_before_window'].mean()*100:.1f}%")
    log.info(f"  sedation_before_window: {mdf['sedation_before_window'].mean()*100:.1f}%")
    log.info(f"  betablocker_before_window: {mdf['betablocker_before_window'].mean()*100:.1f}%")
    log.info(f"  missing ac1_hr_early_mean:{mdf['ac1_hr_early_mean'].isna().sum()}")
    log.info(f"  missing ac1_hr_late_mean: {mdf['ac1_hr_late_mean'].isna().sum()}")
    log.info(f"  missing delta_hr:         {mdf['delta_hr'].isna().sum()}")

    # 仅按主模型真正需要的变量过滤缺失值。
    # delta_hr / age / sofa_admission 不在此列：delta_hr 仅作描述性摘要；
    # age 和 sofa_admission 在条件 logistic 中由匹配对似然吸收，不作为协变量。
    # fit_conditional_model() 会在自身内部再次 dropna(predictor, extra_covariates)。
    required = ["ac1_hr_late_mean"]
    before = len(mdf)
    mdf = mdf.dropna(subset=required)
    log.info(f"After dropping missing on primary predictor: {len(mdf):,} (dropped {before-len(mdf)})")

    return mdf


# ── 统计工具 ──────────────────────────────────────────────────────────────────

def hosmer_lemeshow(y_true: np.ndarray, y_pred: np.ndarray,
                    g: int = 10) -> tuple[float, float]:
    """
    Hosmer-Lemeshow 检验 (g 十分位组).
    返回 (H统计量, p值)。
    """
    df = pd.DataFrame({"y": y_true, "p": y_pred})
    df["decile"] = pd.qcut(df["p"], g, duplicates="drop", labels=False)
    grouped = df.groupby("decile").agg(
        obs1=("y", "sum"),
        n=("y", "count"),
        p_mean=("p", "mean"),
    ).reset_index()
    grouped["exp1"] = grouped["p_mean"] * grouped["n"]
    grouped["exp0"] = (1 - grouped["p_mean"]) * grouped["n"]
    grouped["obs0"] = grouped["n"] - grouped["obs1"]
    H = (
        ((grouped["obs1"] - grouped["exp1"]) ** 2 / grouped["exp1"]).sum()
        + ((grouped["obs0"] - grouped["exp0"]) ** 2 / grouped["exp0"]).sum()
    )
    dof = len(grouped) - 2
    p   = 1 - stats.chi2.cdf(H, dof) if dof > 0 else np.nan
    return H, p


def mcfadden_r2(result: sm.regression.linear_model.RegressionResultsWrapper) -> float:
    """McFadden 伪 R²."""
    return 1 - result.llf / result.llnull


# ── 模型拟合 ──────────────────────────────────────────────────────────────────

PRIMARY_COVARS = (
    "age_10yr + male + sofa_admission + vent_before_window "
    "+ C(icu_type, Treatment('MICU')) "
    "+ C(monitoring, Treatment('nbp'))"
)

FORMULA_PRIMARY   = f"binary_group ~ ac1_hr_late_mean + {PRIMARY_COVARS}"
FORMULA_SECONDARY = f"binary_group ~ tau_hr + {PRIMARY_COVARS}"
# 仅协变量（无主指标）用于计算增量 R²
FORMULA_COVAR_ONLY = f"binary_group ~ {PRIMARY_COVARS}"


def fit_model(formula: str, data: pd.DataFrame,
              label: str) -> object:
    log.info(f"Fitting [{label}]: {formula[:80]}...")
    model  = smf.logit(formula, data=data)
    result = model.fit(disp=False, maxiter=200)
    if not result.converged:
        log.warning(f"  [{label}] did NOT converge!")
    else:
        log.info(f"  [{label}] converged, AIC={result.aic:.1f}, "
                 f"BIC={result.bic:.1f}, "
                 f"McFadden R²={mcfadden_r2(result):.4f}")

    # Hosmer-Lemeshow
    y_pred = result.predict()
    y_true = data["binary_group"].values[result.model.endog_names != ""]
    # align length (dropna inside smf can reduce rows)
    idx = result.model.data.orig_endog.index
    y_true_aligned = data.loc[idx, "binary_group"].values
    H, p_hl = hosmer_lemeshow(y_true_aligned, np.asarray(y_pred))
    log.info(f"  [{label}] Hosmer-Lemeshow H={H:.2f}, p={p_hl:.3f} "
             f"({'good fit' if p_hl > 0.05 else 'poor fit'})")
    return result


def extract_coef_table(result, label: str) -> pd.DataFrame:
    """
    从 logit 结果提取 OR (95% CI) 表。
    返回列: Variable, OR, CI_lo, CI_hi, P, label
    """
    params = result.params
    conf   = result.conf_int()
    pvals  = result.pvalues

    rows = []
    for var in params.index:
        if var == "Intercept":
            continue
        rows.append({
            "Variable": var,
            "coef":     params[var],
            "OR":       np.exp(params[var]),
            "CI_lo":    np.exp(conf.loc[var, 0]),
            "CI_hi":    np.exp(conf.loc[var, 1]),
            "P":        pvals[var],
            "model":    label,
        })
    return pd.DataFrame(rows)


# ── 变量名美化 ────────────────────────────────────────────────────────────────

VAR_LABELS: dict[str, str] = {
    "ac1_hr_early_mean":                      "HR AC1, early-centred window mean †",
    "ac1_hr_late_mean":                       "HR AC1, late-centred window mean †",
    "tau_hr":                                 "HR AC1, Kendall-τ (48h trend slope) †",
    "age_10yr":                               "Age (per 10 years)",
    "male":                                   "Male sex",
    "sofa_admission":                         "Admission SOFA (per 1 point)",
    "vent_before_window":                     "Invasive MV at -12h",
    "sedation_before_window":                 "Sedation exposure in prior 12h",
    "betablocker_before_window":              "Beta-blocker exposure in prior 12h",
    "icu_SICU":                               "ICU type: SICU vs MICU",
    "icu_CCU_CSRU":                           "ICU type: CCU/CSRU vs MICU",
    "icu_Neuro":                              "ICU type: Neuro/NICU vs MICU",
    "icu_Other":                              "ICU type: Other vs MICU",
    "mon_abp":                                "Monitoring: ABP vs NBP",
    "mon_mixed":                              "Monitoring: Mixed vs NBP",
    "early_map_mean_raw":                     "Early-window MAP mean (raw)",
    "early_hr_mean_raw":                      "Early-window HR mean (raw)",
    "late_map_mean_raw":                      "Late-window MAP mean (raw)",
    "late_hr_mean_raw":                       "Late-window HR mean (raw)",
    "C(icu_type, Treatment('MICU'))[T.CCU_CSRU]":    "ICU type: CCU/CSRU vs MICU",
    "C(icu_type, Treatment('MICU'))[T.Neuro_NICU]":  "ICU type: Neuro/NICU vs MICU",
    "C(icu_type, Treatment('MICU'))[T.Other]":        "ICU type: Other vs MICU",
    "C(icu_type, Treatment('MICU'))[T.SICU]":         "ICU type: SICU vs MICU",
    "C(monitoring, Treatment('nbp'))[T.abp]":         "Monitoring: ABP vs NBP",
    "C(monitoring, Treatment('nbp'))[T.mixed]":       "Monitoring: Mixed vs NBP",
}

# 排序优先级（主指标排首位）
VAR_ORDER_PRIMARY   = ["ac1_hr_late_mean"]
VAR_ORDER_SECONDARY = ["tau_hr"]
COVAR_ORDER = [
    "age_10yr", "male", "sofa_admission",
    "vent_before_window",
    "sedation_before_window",
    "betablocker_before_window",
    "C(icu_type, Treatment('MICU'))[T.SICU]",
    "C(icu_type, Treatment('MICU'))[T.CCU_CSRU]",
    "C(icu_type, Treatment('MICU'))[T.Neuro_NICU]",
    "C(icu_type, Treatment('MICU'))[T.Other]",
    "C(monitoring, Treatment('nbp'))[T.abp]",
    "C(monitoring, Treatment('nbp'))[T.mixed]",
]


def format_or(or_: float, lo: float, hi: float) -> str:
    return f"{or_:.2f} ({lo:.2f}–{hi:.2f})"


def format_p(p: float) -> str:
    if np.isnan(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fit_conditional_model(data: pd.DataFrame, predictor: str,
                          extra_covariates: list[str] | None = None,
                          include_sedation: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    以 matched_pair_id 分层的 conditional logistic regression.
    匹配变量 (如 admission SOFA / time-to-event) 由条件似然吸收, 不单独估计系数。
    """
    extra_covariates = extra_covariates or []
    cohort = pd.read_parquet(PROJECT_ROOT / "data" / "cohort.parquet")
    pair_df = cohort[["stay_id", "T0", "matched_pair_id"]].drop_duplicates(["stay_id", "T0"])

    df2 = data.merge(pair_df, on=["stay_id", "T0"], how="left").copy()
    df2 = df2.dropna(subset=["matched_pair_id", predictor, *extra_covariates])

    df2["icu_SICU"]     = (df2["icu_type"] == "SICU").astype(float)
    df2["icu_CCU_CSRU"] = (df2["icu_type"] == "CCU_CSRU").astype(float)
    df2["icu_Neuro"]    = (df2["icu_type"] == "Neuro_NICU").astype(float)
    df2["icu_Other"]    = (df2["icu_type"] == "Other").astype(float)
    df2["mon_abp"]      = (df2["monitoring"] == "abp").astype(float)
    df2["mon_mixed"]    = (df2["monitoring"] == "mixed").astype(float)

    covariate_cols = [
        predictor,
        *extra_covariates,
        "vent_before_window",
        "betablocker_before_window",
        "icu_SICU", "icu_CCU_CSRU", "icu_Neuro", "icu_Other",
        "mon_abp", "mon_mixed",
    ]
    if include_sedation:
        covariate_cols.insert(len(extra_covariates) + 2, "sedation_before_window")

    pair_balance = df2.groupby("matched_pair_id")["binary_group"].agg(shock="sum", total="size")
    valid_pairs = pair_balance[
        (pair_balance["shock"] >= 1) & (pair_balance["shock"] < pair_balance["total"])
    ].index
    df2 = df2[df2["matched_pair_id"].isin(valid_pairs)].copy()

    X = df2[covariate_cols].astype(float).values
    y = df2["binary_group"].astype(int).values
    g = df2["matched_pair_id"].values

    log.info(f"Fitting conditional logistic: predictor={predictor}, "
             f"observations={len(df2):,}, informative strata={len(valid_pairs):,}")

    model = ConditionalLogit(y, X, groups=g)
    result = model.fit(method="bfgs", disp=False, maxiter=500)

    cis = result.conf_int()
    rows = []
    for i, var in enumerate(covariate_cols):
        rows.append({
            "Variable": var,
            "OR": np.exp(result.params[i]),
            "CI_lo": np.exp(cis[i, 0]),
            "CI_hi": np.exp(cis[i, 1]),
            "P": float(result.pvalues[i]),
        })

    summary = {
        "Predictor": predictor,
        "Extra covariates": ", ".join(extra_covariates) if extra_covariates else "None",
        "N observations": f"{len(df2):,}",
        "Informative strata": f"{len(valid_pairs):,}",
        "Log-likelihood": f"{result.llf:.1f}",
    }
    return pd.DataFrame(rows), summary


def build_primary_table(tbl_raw: pd.DataFrame) -> pd.DataFrame:
    order = [
        "ac1_hr_early_mean",
        "ac1_hr_late_mean",
        "tau_hr",
        "early_map_mean_raw",
        "early_hr_mean_raw",
        "late_map_mean_raw",
        "late_hr_mean_raw",
        "vent_before_window",
        "sedation_before_window",
        "betablocker_before_window",
        "icu_SICU",
        "icu_CCU_CSRU",
        "icu_Neuro",
        "icu_Other",
        "mon_abp",
        "mon_mixed",
    ]
    order_idx = {v: i for i, v in enumerate(order)}
    tbl_raw = tbl_raw.copy()
    tbl_raw["_sort"] = tbl_raw["Variable"].map(order_idx).fillna(99)
    tbl_raw = tbl_raw.sort_values("_sort").drop(columns="_sort")

    rows = []
    for _, row in tbl_raw.iterrows():
        rows.append({
            "Variable": VAR_LABELS.get(row["Variable"], row["Variable"]),
            "OR (95% CI)": format_or(row["OR"], row["CI_lo"], row["CI_hi"]),
            "P": format_p(row["P"]),
        })
    return pd.DataFrame(rows)


def build_combined_table(res_primary, res_secondary,
                         data: pd.DataFrame) -> pd.DataFrame:
    """
    合并 Primary / Secondary 模型结果为单张宽表。
    列: Variable_label, OR_primary, OR_secondary
    """
    tbl_p = extract_coef_table(res_primary,   "primary")
    tbl_s = extract_coef_table(res_secondary, "secondary")

    # 合并到宽格式
    wide = tbl_p.rename(columns={
        "OR":    "OR_P",
        "CI_lo": "CI_lo_P",
        "CI_hi": "CI_hi_P",
        "P":     "P_P",
        "coef":  "coef_P",
    })[["Variable", "OR_P", "CI_lo_P", "CI_hi_P", "P_P"]].merge(
        tbl_s.rename(columns={
            "OR":    "OR_S",
            "CI_lo": "CI_lo_S",
            "CI_hi": "CI_hi_S",
            "P":     "P_S",
            "coef":  "coef_S",
        })[["Variable", "OR_S", "CI_lo_S", "CI_hi_S", "P_S"]],
        on="Variable", how="outer"
    )

    # 排序
    order = VAR_ORDER_PRIMARY + VAR_ORDER_SECONDARY + COVAR_ORDER
    order_idx = {v: i for i, v in enumerate(order)}
    wide["_sort"] = wide["Variable"].map(order_idx).fillna(99)
    wide = wide.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    # 构建输出行
    rows = []
    for _, row in wide.iterrows():
        var = row["Variable"]
        label = VAR_LABELS.get(var, var)

        or_p = format_or(row["OR_P"], row["CI_lo_P"], row["CI_hi_P"]) \
               if pd.notna(row.get("OR_P")) else "—"
        p_p  = format_p(row.get("P_P", np.nan))
        or_s = format_or(row["OR_S"], row["CI_lo_S"], row["CI_hi_S"]) \
               if pd.notna(row.get("OR_S")) else "—"
        p_s  = format_p(row.get("P_S", np.nan))

        rows.append({
            "Variable": label,
            "Primary: OR (95% CI)": or_p,
            "Primary: P":           p_p,
            "Secondary: OR (95% CI)": or_s,
            "Secondary: P":           p_s,
        })
    return pd.DataFrame(rows)


def build_diagnostics_table(res_p, res_s, res_cov, data: pd.DataFrame) -> pd.DataFrame:
    """模型诊断摘要表。"""
    idx = res_p.model.data.orig_endog.index
    y_true = data.loc[idx, "binary_group"].values

    diag_rows = []
    for label, res in [("Primary", res_p), ("Secondary", res_s),
                       ("Covariates only", res_cov)]:
        y_pred = res.predict()
        idx2   = res.model.data.orig_endog.index
        yt     = data.loc[idx2, "binary_group"].values
        H, p_hl = hosmer_lemeshow(yt, np.asarray(y_pred))
        mfr2 = mcfadden_r2(res)

        # Nagelkerke pseudo-R²
        n = len(yt)
        r2_cox = 1 - np.exp(-2 * (res.llf - res.llnull) / n)
        r2_nag = r2_cox / (1 - np.exp(2 * res.llnull / n))

        diag_rows.append({
            "Model":              label,
            "N":                  f"{int(res.nobs):,}",
            "AIC":                f"{res.aic:.1f}",
            "BIC":                f"{res.bic:.1f}",
            "McFadden R²":        f"{mfr2:.4f}",
            "Nagelkerke R²":      f"{r2_nag:.4f}",
            "HL H (p)":           f"{H:.2f} (p={format_p(p_hl)})",
        })
    return pd.DataFrame(diag_rows)


# ── HTML 输出 ─────────────────────────────────────────────────────────────────

HTML_CSS = """
<style>
body { font-family: Arial, sans-serif; font-size: 13px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
th { background: #2c3e50; color: white; }
tr:nth-child(even) { background: #f8f8f8; }
tr.primary-var td { background: #fef9e7; font-weight: bold; }
td.sig { color: #c0392b; font-weight: bold; }
caption { font-size: 14px; font-weight: bold; margin-bottom: 6px; text-align: left; }
</style>
"""

PRIMARY_LABELS = {
    "HR AC1, early-centred window mean †",
    "HR AC1, late-centred window mean †",
    "HR AC1, Kendall-τ (48h trend slope) †",
}


def df_to_html(tbl: pd.DataFrame, caption: str,
               highlight_first: bool = False) -> str:
    lines = [f"<table>",
             f"<caption>{caption}</caption>",
             "<thead><tr>"]
    for col in tbl.columns:
        lines.append(f"<th>{col}</th>")
    lines.append("</tr></thead><tbody>")

    for i, (_, row) in enumerate(tbl.iterrows()):
        is_primary = row.get("Variable", "") in PRIMARY_LABELS
        row_cls = ' class="primary-var"' if is_primary else ""
        lines.append(f"<tr{row_cls}>")
        for col in tbl.columns:
            val = str(row[col])
            cell_cls = ""
            if col in ("Primary: P", "Secondary: P", "P"):
                try:
                    pf = float(val) if val not in ("<0.001", "—") else 0
                    if val == "<0.001" or pf < 0.05:
                        cell_cls = ' class="sig"'
                except ValueError:
                    pass
            lines.append(f"<td{cell_cls}>{val}</td>")
        lines.append("</tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suffix",
        default="",
        help="附加到输入/输出文件名的后缀，例如 _double_detrend",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log.info("=" * 70)
    log.info("Step 7 (T1.2): HR AC1 Primary Conditional Logistic Regression")
    log.info("=" * 70)

    # 1. 构建特征矩阵
    df_ews = load_ews_features(args.suffix)
    mdf    = build_model_data(df_ews, args.suffix)

    # 2. 描述性统计
    log.info("")
    log.info("=== Predictor summary by group ===")
    for col in ["ac1_hr_early_mean", "ac1_hr_late_mean", "delta_hr", "sofa_admission", "age"]:
        g = mdf.groupby("binary_group")[col]
        log.info(
            f"  {col}: shock {g.get_group(1).median():.3f} "
            f"[{g.get_group(1).quantile(0.25):.3f}–"
            f"{g.get_group(1).quantile(0.75):.3f}]  vs  "
            f"control {g.get_group(0).median():.3f} "
            f"[{g.get_group(0).quantile(0.25):.3f}–"
            f"{g.get_group(0).quantile(0.75):.3f}]"
        )

    # 3. 拟合主模型 + 核心敏感性模型
    log.info("")
    raw_tbl, summary = fit_conditional_model(mdf, "ac1_hr_late_mean")
    raw_tbl_late, summary_late = fit_conditional_model(
        mdf,
        "ac1_hr_late_mean",
        extra_covariates=["late_map_mean_raw", "late_hr_mean_raw"],
    )
    raw_tbl_early, summary_early = fit_conditional_model(
        mdf,
        "ac1_hr_early_mean",
        extra_covariates=["early_map_mean_raw", "early_hr_mean_raw"],
    )
    raw_tbl_late_early, summary_late_early = fit_conditional_model(
        mdf,
        "ac1_hr_late_mean",
        extra_covariates=["early_map_mean_raw", "early_hr_mean_raw"],
    )
    mdf_no_sed = mdf.loc[mdf["sedation_before_window"] == 0].copy()
    raw_tbl_no_sed, summary_no_sed = fit_conditional_model(
        mdf_no_sed,
        "ac1_hr_late_mean",
        include_sedation=False,
    )
    tbl_coef = build_primary_table(raw_tbl)
    tbl_late = build_primary_table(raw_tbl_late)
    tbl_early = build_primary_table(raw_tbl_early)
    tbl_late_early = build_primary_table(raw_tbl_late_early)
    tbl_no_sed = build_primary_table(raw_tbl_no_sed)
    tbl_early_csv = pd.concat(
        [
            tbl_early.assign(Model="Early-window AC1 adjusted for early raw MAP/HR"),
            tbl_late_early.assign(Model="Late-window AC1 adjusted for early raw MAP/HR"),
        ],
        ignore_index=True,
    )
    tbl_diag = pd.DataFrame([
        {
            "Model": "Primary conditional logistic",
            "Predictor": VAR_LABELS["ac1_hr_late_mean"],
            **summary,
            "Note": "Matched-pair strata condition out matching variables and baseline hazard.",
        },
        {
            "Model": "Late-vitals-adjusted sensitivity",
            "Predictor": VAR_LABELS["ac1_hr_late_mean"],
            **summary_late,
            "Note": "Additionally adjusts for raw final-12h mean MAP and HR.",
        },
        {
            "Model": "Early-window independence",
            "Predictor": VAR_LABELS["ac1_hr_early_mean"],
            **summary_early,
            "Note": "Early-window AC1 adjusted for raw early-window mean MAP and HR.",
        },
        {
            "Model": "Late AC1 adjusted for early vitals",
            "Predictor": VAR_LABELS["ac1_hr_late_mean"],
            **summary_late_early,
            "Note": "Tests whether the late-centred AC1 summary adds information beyond earlier raw MAP and HR.",
        },
        {
            "Model": "No-sedation subgroup",
            "Predictor": VAR_LABELS["ac1_hr_late_mean"],
            **summary_no_sed,
            "Note": "Restricted to stay-T0 pairs without sedative exposure in the prior 12h; sedation covariate omitted.",
        },
    ])

    # 4. 输出
    csv_path  = tagged_path(OUTPUT_DIR / "table2_multivariable.csv", args.suffix)
    html_path = tagged_path(OUTPUT_DIR / "table2_multivariable.html", args.suffix)
    diag_csv  = tagged_path(OUTPUT_DIR / "table2_diagnostics.csv", args.suffix)
    s4_csv    = tagged_path(OUTPUT_DIR / "tableS4_late_vitals_sensitivity.csv", args.suffix)
    s4_html   = tagged_path(OUTPUT_DIR / "tableS4_late_vitals_sensitivity.html", args.suffix)
    s5_csv    = tagged_path(OUTPUT_DIR / "tableS5_early_window_independence.csv", args.suffix)
    s5_html   = tagged_path(OUTPUT_DIR / "tableS5_early_window_independence.html", args.suffix)
    s6_csv    = tagged_path(OUTPUT_DIR / "tableS6_no_sedation_subgroup.csv", args.suffix)
    s6_html   = tagged_path(OUTPUT_DIR / "tableS6_no_sedation_subgroup.html", args.suffix)

    tbl_coef.to_csv(csv_path, index=False, encoding="utf-8-sig")
    tbl_diag.to_csv(diag_csv, index=False, encoding="utf-8-sig")
    tbl_late.to_csv(s4_csv, index=False, encoding="utf-8-sig")
    tbl_early_csv.to_csv(s5_csv, index=False, encoding="utf-8-sig")
    tbl_no_sed.to_csv(s6_csv, index=False, encoding="utf-8-sig")
    log.info(f"CSV saved → {csv_path}")
    log.info(f"CSV saved → {diag_csv}")
    log.info(f"CSV saved → {s4_csv}")
    log.info(f"CSV saved → {s5_csv}")
    log.info(f"CSV saved → {s6_csv}")

    n = summary["N observations"]
    n_pairs = summary["Informative strata"]
    html_body = HTML_CSS
    html_body += df_to_html(
        tbl_coef,
        f"Table 2. Primary Conditional Logistic Regression for Late-Onset Septic Shock "
        f"(N={n}; informative matched strata={n_pairs}). "
        f"† primary predictor (highlighted). Reference: ICU type=MICU, monitoring=NBP. "
        f"Matching variables are conditioned out by design; standard logistic and GEE are sensitivity analyses.",
        highlight_first=True,
    )
    html_body += df_to_html(
        tbl_diag,
        "Model Summary",
    )
    html_path.write_text(html_body, encoding="utf-8")
    log.info(f"HTML saved → {html_path}")

    html_s4 = HTML_CSS
    html_s4 += df_to_html(
        tbl_late,
        "Table S4. Conditional Logistic Sensitivity Adjusting for Late-Window Raw MAP and HR",
        highlight_first=True,
    )
    s4_html.write_text(html_s4, encoding="utf-8")
    log.info(f"HTML saved → {s4_html}")

    html_s5 = HTML_CSS
    html_s5 += df_to_html(
        tbl_early,
        "Table S5A. Early-window HR AC1 adjusted for early-window raw MAP and HR",
        highlight_first=True,
    )
    html_s5 += df_to_html(
        tbl_late_early,
        "Table S5B. Late-window HR AC1 adjusted for early-window raw MAP and HR",
        highlight_first=True,
    )
    s5_html.write_text(html_s5, encoding="utf-8")
    log.info(f"HTML saved → {s5_html}")

    html_s6 = HTML_CSS
    html_s6 += df_to_html(
        tbl_no_sed,
        "Table S6. No-sedation subgroup conditional logistic model",
        highlight_first=True,
    )
    s6_html.write_text(html_s6, encoding="utf-8")
    log.info(f"HTML saved → {s6_html}")

    # 5. 终端摘要
    log.info("")
    log.info("━" * 70)
    log.info("TABLE 2 PREVIEW")
    log.info("━" * 70)
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.width", 200)
    log.info("\n" + tbl_coef.to_string(index=False))
    log.info("")
    log.info("MODEL SUMMARY")
    log.info("\n" + tbl_diag.to_string(index=False))
    log.info("")
    log.info("TABLE S4 PREVIEW")
    log.info("\n" + tbl_late.to_string(index=False))
    log.info("")
    log.info("TABLE S5A PREVIEW")
    log.info("\n" + tbl_early.to_string(index=False))
    log.info("")
    log.info("TABLE S5B PREVIEW")
    log.info("\n" + tbl_late_early.to_string(index=False))
    log.info("")
    log.info("TABLE S6 PREVIEW")
    log.info("\n" + tbl_no_sed.to_string(index=False))
    log.info("━" * 70)

    # 6. 结论判断
    ac1_p = float(raw_tbl.loc[raw_tbl["Variable"] == "ac1_hr_late_mean", "P"].iloc[0])
    ac1_p_late = float(raw_tbl_late.loc[raw_tbl_late["Variable"] == "ac1_hr_late_mean", "P"].iloc[0])
    ac1_p_early = float(raw_tbl_early.loc[raw_tbl_early["Variable"] == "ac1_hr_early_mean", "P"].iloc[0])
    ac1_p_late_early = float(raw_tbl_late_early.loc[raw_tbl_late_early["Variable"] == "ac1_hr_late_mean", "P"].iloc[0])
    ac1_p_no_sed = float(raw_tbl_no_sed.loc[raw_tbl_no_sed["Variable"] == "ac1_hr_late_mean", "P"].iloc[0])
    log.info("")
    if ac1_p < 0.05:
        log.info(f"✓ PRIMARY: HR AC1 late-centred window mean remains significant after "
                 f"matched conditional adjustment (p={format_p(ac1_p)}). "
                 f"'Independent association' supported within matched strata.")
    else:
        log.info(f"✗ PRIMARY: HR AC1 late-centred window mean NOT significant after "
                 f"matched conditional adjustment (p={format_p(ac1_p)}).")
    if ac1_p_late < 0.05:
        log.info(f"✓ SENSITIVITY: HR AC1 remains significant after additional adjustment "
                 f"for raw final-12h MAP/HR means (p={format_p(ac1_p_late)}).")
    else:
        log.info(f"✗ SENSITIVITY: HR AC1 loses significance after additional adjustment "
                 f"for raw final-12h MAP/HR means (p={format_p(ac1_p_late)}).")
    if ac1_p_early < 0.05:
        log.info(f"✓ EARLY: Early-window HR AC1 remains significant after adjustment "
                 f"for early raw MAP/HR means (p={format_p(ac1_p_early)}).")
    else:
        log.info(f"✗ EARLY: Early-window HR AC1 is not independently associated after "
                 f"adjustment for early raw MAP/HR means (p={format_p(ac1_p_early)}).")
    if ac1_p_late_early < 0.05:
        log.info(f"✓ LATE|EARLY VITALS: Late-window HR AC1 remains significant even after "
                 f"adjustment for early raw MAP/HR means (p={format_p(ac1_p_late_early)}).")
    else:
        log.info(f"✗ LATE|EARLY VITALS: Late-window HR AC1 does not retain significance after "
                 f"adjustment for early raw MAP/HR means (p={format_p(ac1_p_late_early)}).")
    if ac1_p_no_sed < 0.05:
        log.info(f"✓ NO-SEDATION: Late-window HR AC1 remains significant in the no-sedation "
                 f"subgroup (p={format_p(ac1_p_no_sed)}).")
    else:
        log.info(f"✗ NO-SEDATION: Late-window HR AC1 is not significant in the no-sedation "
                 f"subgroup (p={format_p(ac1_p_no_sed)}).")

    log.info("Step 7 complete.")


if __name__ == "__main__":
    main()
