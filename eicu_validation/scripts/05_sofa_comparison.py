"""
eICU 外部验证 — Step 5: EWS vs SOFA/APACHE 对比分析

研究问题：
  EWS 动态指标（AC1、n_extrema、Euler χ(0)，取自 T0 前 12h）
  与临床严重度评分（SOFA、APACHE，取自 T0 时刻或 ICU 入院）相比，
  是否提供独立的、互补的预警信息？

分析内容：
  1. AUC 对比（shock vs control，非配对 ROC）
  2. Mann-Whitney U 检验（组间分布差异）
  3. 条件 logistic 独立性检验
       - EWS alone
       - EWS + SOFA non-cardio（凝血/肝/肾/神经，排除升压药循环项）
       - EWS + APACHE 入院评分
     × 子集：全样本 / T0≥72h

关于 SOFA 心血管分项的处理：
  eICU 的 case 定义包含启用升压药，因此 SOFA 心血管项（vasopressor=3分）
  在 shock 组几乎为 3、在 control 组几乎为 0，与结局近似同义。
  AUC 展示 SOFA full（含该项，供参考）和 SOFA non-cardio（更公平）两个版本；
  条件 logistic 仅使用 SOFA non-cardio 作为协变量。

SOFA 数据来源（eICU 无原生 SOFA 表，逐项计算）：
  - 凝血：lab.platelets x 1000，T0 前最近一次
  - 肝：lab.total bilirubin，T0 前最近一次
  - 肾：lab.creatinine，T0 前最近一次
  - 神经：apacheApsVar.eyes/motor/verbal → GCS
  - 心血管：升压药使用（= binary_group）
  - 呼吸：缺 PaO2/FiO2 数据，保守设为 0

输出文件：
  output/fig_roc_vs_sofa.png       ROC 曲线对比图
  output/tbl_auc_comparison.csv    AUC 汇总表
  output/tbl_clogit_vs_sofa.csv    条件 logistic 独立性检验结果
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.sans-serif": ["Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.discrete.conditional_models import ConditionalLogit

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
EICU_DIR     = PROJECT_ROOT.parent / "eicu-crd" / "2.0"
OUTPUT_DIR   = PROJECT_ROOT / "output"

COHORT_PATH = OUTPUT_DIR / "cohort_eicu.parquet"
STATS_PATH  = OUTPUT_DIR / "ews_patient_eicu.parquet"
DIAG_PATH   = OUTPUT_DIR / "cleaning_diagnostics_eicu.parquet"

# T0 分层阈值（小时）
T0_SUBSETS = [("全样本", 0), ("T0≥72h", 72)]

BASE_CANDIDATE = [
    "vent_before_window",
    "sedation_before_window",
    "betablocker_before_window",
]

# ── SOFA 分项评分函数 ─────────────────────────────────────────────────────────

def _sofa_coag(v: float) -> float:
    if pd.isna(v): return np.nan
    for threshold, score in [(20, 4), (50, 3), (100, 2), (150, 1)]:
        if v < threshold:
            return score
    return 0


def _sofa_liver(v: float) -> float:
    if pd.isna(v): return np.nan
    for threshold, score in [(1.2, 0), (2.0, 1), (6.0, 2), (12.0, 3)]:
        if v < threshold:
            return score
    return 4


def _sofa_renal(v: float) -> float:
    if pd.isna(v): return np.nan
    for threshold, score in [(1.2, 0), (2.0, 1), (3.5, 2), (5.0, 3)]:
        if v < threshold:
            return score
    return 4


def _sofa_neuro(eyes: float, motor: float, verbal: float) -> float:
    vals = [x for x in [eyes, motor, verbal] if not pd.isna(x)]
    if not vals:
        return np.nan
    gcs = sum(vals)
    for threshold, score in [(6, 4), (10, 3), (13, 2), (15, 1)]:
        if gcs < threshold:
            return score
    return 0


# ── 数据加载与 SOFA 计算 ──────────────────────────────────────────────────────

def load_and_build_mdf() -> pd.DataFrame:
    cohort = pd.read_parquet(COHORT_PATH)
    stats  = pd.read_parquet(STATS_PATH)
    diag   = pd.read_parquet(DIAG_PATH)

    ids    = set(cohort["patientunitstayid"])
    t0_map = cohort[["patientunitstayid", "T0_min"]].drop_duplicates()

    # ── lab 数据：取 T0 前最近一次检测值 ─────────────────────────
    lab = pd.read_csv(
        EICU_DIR / "lab.csv.gz",
        usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
    )
    lab = lab[lab["patientunitstayid"].isin(ids)].copy()
    lab["labresult"] = pd.to_numeric(lab["labresult"], errors="coerce")
    lab = lab.merge(t0_map, on="patientunitstayid")

    def last_before_t0(labname: str, colname: str) -> pd.DataFrame:
        return (
            lab[(lab["labname"] == labname) & (lab["labresultoffset"] <= lab["T0_min"])]
            .sort_values("labresultoffset")
            .groupby("patientunitstayid")["labresult"]
            .last()
            .reset_index()
            .rename(columns={"labresult": colname})
        )

    plat = last_before_t0("platelets x 1000", "platelets")
    bili = last_before_t0("total bilirubin",  "bilirubin")
    crea = last_before_t0("creatinine",        "creatinine")

    # ── GCS 分项（APACHE APS 变量）───────────────────────────────
    aps = (
        pd.read_csv(EICU_DIR / "apacheApsVar.csv.gz",
                    usecols=["patientunitstayid", "eyes", "motor", "verbal"])
        [lambda d: d["patientunitstayid"].isin(ids)]
        .groupby("patientunitstayid").first()
        .reset_index()
    )

    # ── APACHE 评分（优先取 IV 版本）────────────────────────────
    apr = pd.read_csv(
        EICU_DIR / "apachePatientResult.csv.gz",
        usecols=["patientunitstayid", "apachescore", "apacheversion"],
    )
    apr = apr[apr["patientunitstayid"].isin(ids)].copy()
    apache = (
        apr[apr["apacheversion"] == "IV"]
        .groupby("patientunitstayid")["apachescore"].first()
        .combine_first(apr.groupby("patientunitstayid")["apachescore"].first())
        .reset_index()
        .rename(columns={"apachescore": "apache_score"})
    )

    # ── 组装基础 DataFrame ───────────────────────────────────────
    sdf = (
        cohort[[
            "patientunitstayid", "T0_min", "T0_h", "group", "matched_pair_id",
            *BASE_CANDIDATE,
        ]]
        .drop_duplicates()
        .merge(plat,   on="patientunitstayid", how="left")
        .merge(bili,   on="patientunitstayid", how="left")
        .merge(crea,   on="patientunitstayid", how="left")
        .merge(aps,    on="patientunitstayid", how="left")
        .merge(apache, on="patientunitstayid", how="left")
    )

    sdf["binary_group"]     = (sdf["group"] == "shock").astype(int)
    sdf["sofa_coag"]        = sdf["platelets"].apply(_sofa_coag)
    sdf["sofa_liver"]       = sdf["bilirubin"].apply(_sofa_liver)
    sdf["sofa_renal"]       = sdf["creatinine"].apply(_sofa_renal)
    sdf["sofa_neuro"]       = sdf.apply(
        lambda r: _sofa_neuro(r["eyes"], r["motor"], r["verbal"]), axis=1
    )
    # 心血管分项：以升压药使用为代理（= shock 定义）
    sdf["sofa_cardio_vaso"] = sdf["binary_group"].apply(lambda x: 3.0 if x == 1 else 0.0)
    sdf["sofa_resp"]        = 0.0   # 缺 PaO2/FiO2，保守取 0

    # SOFA full：含心血管分项（仅用于 AUC 展示，有循环论证风险）
    sdf["sofa_full"] = sdf[[
        "sofa_coag", "sofa_liver", "sofa_renal",
        "sofa_neuro", "sofa_cardio_vaso", "sofa_resp",
    ]].sum(axis=1, min_count=3)

    # SOFA non-cardio：排除升压药循环项，用于独立性检验
    sdf["sofa_noncardio"] = sdf[[
        "sofa_coag", "sofa_liver", "sofa_renal", "sofa_neuro", "sofa_resp",
    ]].sum(axis=1, min_count=2)

    # ── 合并 EWS 特征，并把 sdf 限制到同一 passed 母集 ──────────
    # 先 inner join passed，确保所有指标在同一 EWS 质量筛查通过的母集内计算。
    # 注意：这并不保证各指标使用完全相同的分母——SOFA/APACHE 本身有缺失，
    # 导致实际分析 n 仍各不相同（SOFA≈1,140 / APACHE≈989 / AC1≈1,138 / EWS≈1,152）。
    # AUC 比较应描述为"同一 passed 母集内，按各指标可用样本分别计算"，
    # 而非"完全相同分母的 head-to-head 比较"。
    passed = diag[~diag["excluded"]][["patientunitstayid", "T0_min"]]
    ews = stats.merge(passed, on=["patientunitstayid", "T0_min"])

    # inner join：sdf 先限制到 passed 子集，再合并 EWS 列
    mdf = (
        sdf.merge(passed, on=["patientunitstayid", "T0_min"], how="inner")
        .merge(
            ews[["patientunitstayid", "T0_min",
                 "ac1_hr_late_mean", "n_extrema_hr_late_mean", "euler_hr_late_mean"]],
            on=["patientunitstayid", "T0_min"],
            how="left",
        )
    )

    for c in BASE_CANDIDATE:
        mdf[c] = pd.to_numeric(mdf[c], errors="coerce").fillna(0)

    return mdf


# ── 统计工具 ──────────────────────────────────────────────────────────────────

def auc_with_ci(y: np.ndarray, x: np.ndarray, flip: bool = False) -> tuple[float, float, float]:
    """返回 (AUC, CI_lo, CI_hi)，使用 Hanley-McNeil 近似。"""
    x2 = -x if flip else x.copy()
    a  = roc_auc_score(y, x2)
    n1 = int(y.sum()); n2 = len(y) - n1
    q1  = a / (2 - a)
    q2  = 2 * a**2 / (1 + a)
    var = (a*(1-a) + (n1-1)*(q1-a**2) + (n2-1)*(q2-a**2)) / (n1 * n2)
    se  = np.sqrt(var)
    return a, a - 1.96*se, a + 1.96*se


def clogit(
    data:      pd.DataFrame,
    predictor: str,
    extra:     list[str] | None = None,
) -> dict | None:
    """
    条件 logistic 回归。
    自动过滤当前子集中无变异的协变量（避免奇异 Hessian）。
    返回 {OR, lo, hi, P, N, strata} 或 None。
    """
    extra_vars = extra or []
    # 动态过滤零方差协变量
    base_ok = [c for c in BASE_CANDIDATE
               if c in data.columns and data[c].std() > 0.01]
    all_vars = [predictor] + extra_vars + base_ok

    df2 = data.dropna(subset=["matched_pair_id", "binary_group", *all_vars]).copy()
    df2["binary_group"] = df2["binary_group"].astype(int)

    pair_bal = df2.groupby("matched_pair_id")["binary_group"].agg(s="sum", t="size")
    valid    = pair_bal[(pair_bal["s"] >= 1) & (pair_bal["s"] < pair_bal["t"])].index
    df2      = df2[df2["matched_pair_id"].isin(valid)].copy()

    if len(df2) < 20:
        return None

    try:
        res  = ConditionalLogit(
            df2["binary_group"].astype(int).values,
            df2[all_vars].astype(float).values,
            groups=df2["matched_pair_id"].values,
        ).fit(method="bfgs", disp=False, maxiter=500)
        cis = np.asarray(res.conf_int())
        return {
            "OR":     float(np.exp(res.params[0])),
            "lo":     float(np.exp(cis[0, 0])),
            "hi":     float(np.exp(cis[0, 1])),
            "P":      float(res.pvalues[0]),
            "N":      len(df2),
            "strata": len(valid),
        }
    except Exception as e:
        print(f"  [warn] clogit {predictor}+{extra}: {e}")
        return None


def fmt_or(r: dict | None) -> str:
    if not r:
        return "—"
    sig = " *" if r["P"] < 0.05 else ""
    p   = "<0.001" if r["P"] < 0.001 else f"{r['P']:.3f}"
    return f"OR={r['OR']:.2f}({r['lo']:.2f}–{r['hi']:.2f}) p={p}{sig}  N={r['N']}"


# ── 主分析 ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("加载数据 …")
    mdf = load_and_build_mdf()
    print(f"  样本: {len(mdf):,}  (shock={mdf['binary_group'].sum()}, "
          f"control={(mdf['binary_group']==0).sum()})\n")

    # 指标定义：(展示名, 列名, 是否需翻转使高分=shock)
    METRICS = [
        ("SOFA full (T0)*",  "sofa_full",             False),
        ("SOFA non-cardio",  "sofa_noncardio",         False),
        ("APACHE",           "apache_score",           False),
        ("AC1",              "ac1_hr_late_mean",       False),
        ("n_extrema",        "n_extrema_hr_late_mean", True),
        ("Euler χ(0)",       "euler_hr_late_mean",      True),
    ]

    EWS_METRICS = [
        ("n_extrema", "n_extrema_hr_late_mean"),
        ("Euler χ(0)", "euler_hr_late_mean"),
        ("AC1",        "ac1_hr_late_mean"),
    ]

    # ── 1. AUC ───────────────────────────────────────────────────
    print("=" * 64)
    print("1. AUC 对比（shock vs control）")
    print("   * SOFA full 含升压药心血管分项 = case 定义，AUC 偏高")
    print("=" * 64)
    auc_rows: list[dict] = []
    for name, col, flip in METRICS:
        sub = mdf[["binary_group", col]].dropna()
        if len(sub) < 50:
            continue
        a, lo, hi = auc_with_ci(sub["binary_group"].values, sub[col].values, flip)
        tag = "临床评分" if any(k in name for k in ("SOFA","APACHE")) else "EWS(T0前12h)"
        print(f"  {name:24s}: AUC={a:.3f} ({lo:.3f}–{hi:.3f})  n={len(sub):,}  [{tag}]")
        auc_rows.append({"指标": name, "AUC": a, "CI_lo": lo, "CI_hi": hi,
                         "n": len(sub), "类型": tag})
    auc_tbl = pd.DataFrame(auc_rows)

    # ── 2. Mann-Whitney ──────────────────────────────────────────
    print("\n" + "=" * 64)
    print("2. Mann-Whitney U（组间分布，全样本）")
    print("=" * 64)
    for name, col, _ in METRICS:
        s = mdf[mdf["group"] == "shock"][col].dropna()
        c = mdf[mdf["group"] == "control"][col].dropna()
        if len(s) < 20 or len(c) < 20:
            continue
        _, p = mannwhitneyu(s, c, alternative="two-sided")
        print(f"  {name:24s}: shock={s.median():.2f}  ctrl={c.median():.2f}  p={p:.4f}")

    # ── 3. 条件 logistic 独立性检验 ─────────────────────────────
    print("\n" + "=" * 64)
    print("3. 条件 logistic：EWS 独立于 SOFA non-cardio / APACHE")
    print("   SOFA non-cardio = 凝血+肝+肾+神经（排除升压药循环项）")
    print("=" * 64)
    clog_rows: list[dict] = []
    for label, t0_min_h in T0_SUBSETS:
        subset = mdf[mdf["T0_h"] >= t0_min_h].copy()
        n_pairs = subset["matched_pair_id"].nunique()
        print(f"\n  [{label}  n_pairs≈{n_pairs}]")
        for mname, met in EWS_METRICS:
            r0 = clogit(subset, met)
            r1 = clogit(subset, met, ["sofa_noncardio"])
            r2 = clogit(subset, met, ["apache_score"])
            print(f"    {mname} alone:         {fmt_or(r0)}")
            print(f"    {mname} +SOFA(nc):     {fmt_or(r1)}")
            print(f"    {mname} +APACHE:       {fmt_or(r2)}")
            for tag, r in [("alone", r0), ("+SOFA(nc)", r1), ("+APACHE", r2)]:
                if r:
                    clog_rows.append({"指标": mname, "子集": label, "模型": tag, **r})
    clog_tbl = pd.DataFrame(clog_rows)

    # ── 4. ROC 图 ────────────────────────────────────────────────
    COLORS = {
        "SOFA full (T0)*": "#e6550d",
        "SOFA non-cardio": "#fc8d59",
        "APACHE":          "#fdae6b",
        "AC1":             "#1f77b4",
        "n_extrema":       "#2ca02c",
        "Euler χ(0)":      "#d62728",
    }
    LS = {k: ("--" if any(x in k for x in ("SOFA","APACHE")) else "-") for k in COLORS}
    LW = {k: (1.6  if any(x in k for x in ("SOFA","APACHE")) else 2.2) for k in COLORS}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "eICU: EWS Dynamics (12h before T0) vs Clinical Severity Scores (at T0)\n"
        "dashed = severity scores  |  solid = EWS dynamics",
        fontsize=10,
    )
    for ax, (lbl, t0_min_h) in zip(axes, T0_SUBSETS):
        data = mdf[mdf["T0_h"] >= t0_min_h]
        n_pairs = data["matched_pair_id"].nunique()
        for name, col, flip in METRICS:
            sub = data[["binary_group", col]].dropna()
            if len(sub) < 20:
                continue
            y   = sub["binary_group"].values
            x   = -sub[col].values if flip else sub[col].values
            a   = roc_auc_score(y, x)
            fpr, tpr, _ = roc_curve(y, x)
            label_str = name.replace(" (T0)*", "").replace(" non-cardio", "(nc)")
            ax.plot(fpr, tpr, color=COLORS[name], ls=LS[name], lw=LW[name],
                    label=f"{label_str} {a:.3f}")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlabel("1 − Specificity", fontsize=9)
        ax.set_ylabel("Sensitivity", fontsize=9)
        ax.set_title(f"{lbl} (n_pairs≈{n_pairs})", fontsize=10)
        ax.legend(fontsize=8.5, loc="lower right")

    plt.tight_layout()
    roc_path = OUTPUT_DIR / "fig_roc_vs_sofa.png"
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 5. 输出 ──────────────────────────────────────────────────
    auc_path  = OUTPUT_DIR / "tbl_auc_comparison.csv"
    clog_path = OUTPUT_DIR / "tbl_clogit_vs_sofa.csv"
    auc_tbl.to_csv(auc_path,  index=False, encoding="utf-8-sig")
    if not clog_tbl.empty:
        clog_tbl.to_csv(clog_path, index=False, encoding="utf-8-sig")

    print(f"\n输出文件:")
    print(f"  {roc_path}")
    print(f"  {auc_path}")
    if not clog_tbl.empty:
        print(f"  {clog_path}")


if __name__ == "__main__":
    main()
