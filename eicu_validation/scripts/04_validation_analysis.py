"""
eICU 外部验证 — Step 4: 统计分析与可视化

复现 MIMIC-IV AC1 研究（ac1/scripts/07）的核心模型结构，
同步验证 Euler χ(0) 及调试分析（n_extrema、T0 分层）。

模型规格（条件 logistic，strata=matched_pair_id）:
  M1  : shock ~ ac1_hr_late_mean + vent + sedation + betablocker + monitoring
  S1  : M1 + late_hr_mean_raw        (检验信号被均值调整消除 → MIMIC 核心发现)
  S2  : M1 + late_hr_mean_raw + late_map_mean_raw
  S3  : 早窗 ac1_hr_early_mean（仅 T0>=24h 子集）
  E_M1: Euler 版 M1
  E_S1: Euler 版 S1
  N_M1: n_extrema 版 M1（调试：阈值无关的振荡指标）
  N_S1: n_extrema S1 +HR均值调整

调试输出（Euler 失败根因分析）:
  output/tbl_validation_nExtrema.csv
  output/tbl_euler_t0strat.csv
  output/fig_euler_debug_*.png（已由诊断脚本生成）
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
from statsmodels.discrete.conditional_models import ConditionalLogit

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent
NAAS_ROOT    = PROJECT_ROOT.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"

COHORT_PATH  = OUTPUT_DIR / "cohort_eicu.parquet"
VITALS_PATH  = OUTPUT_DIR / "vitals_eicu.parquet"
STATS_PATH   = OUTPUT_DIR / "ews_patient_eicu.parquet"
WINDOWS_PATH = OUTPUT_DIR / "ews_windows_eicu.parquet"
DIAG_PATH    = OUTPUT_DIR / "cleaning_diagnostics_eicu.parquet"

LATE_LO,  LATE_HI  = -12,  0
EARLY_LO, EARLY_HI = -24, -12

COLORS = {"shock": "#d62728", "control": "#1f77b4"}

# ICU type 哑变量已被 1:1 匹配吸收（matched pair 内 concordant），不纳入条件似然模型
# 仅保留在 pairs 内部有变异的时间协变量
BASE_COVARS = [
    "vent_before_window", "sedation_before_window", "betablocker_before_window",
    "mon_abp",
]


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def fmt_or(o: float, lo: float, hi: float) -> str:
    return f"{o:.2f} ({lo:.2f}–{hi:.2f})"


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def simplify_unittype(ut: str) -> str:
    if pd.isna(ut):
        return "Other"
    ut = str(ut).upper()
    if "MED-SURG" in ut:
        return "Med-Surg"
    if "MICU" in ut or ("MEDICAL" in ut and "SURG" not in ut):
        return "MICU"
    if "SICU" in ut or "SURGICAL" in ut or "TRAUMA" in ut:
        return "SICU"
    if "CCU" in ut or "CARDIAC" in ut:
        return "CCU"
    if "NEURO" in ut:
        return "Neuro"
    return "Other"


# ── 条件 logistic 回归 ────────────────────────────────────────────────────────

def fit_clogit(
    data:             pd.DataFrame,
    predictor:        str,
    extra_covars:     list[str] | None = None,
    include_sedation: bool = True,
    label:            str = "",
) -> pd.DataFrame:
    extra  = extra_covars or []
    covars = list(extra) + [
        c for c in BASE_COVARS
        if include_sedation or c != "sedation_before_window"
    ]
    all_vars = [predictor] + covars

    df2 = data.dropna(subset=["matched_pair_id", *all_vars]).copy()
    pair_bal = df2.groupby("matched_pair_id")["binary_group"].agg(
        shock="sum", total="size"
    )
    valid  = pair_bal[(pair_bal["shock"] >= 1) & (pair_bal["shock"] < pair_bal["total"])].index
    df2    = df2[df2["matched_pair_id"].isin(valid)].copy()

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
            "OR":       np.exp(result.params[i]),
            "CI_lo":    np.exp(cis[i, 0]),
            "CI_hi":    np.exp(cis[i, 1]),
            "P":        float(result.pvalues[i]),
            "N_obs":    len(df2),
            "N_strata": len(valid),
            "model":    label,
        })
    return pd.DataFrame(rows)


# ── 图表 ──────────────────────────────────────────────────────────────────────

def forest_plot(
    models: list[tuple[str, str, pd.DataFrame]],
    path:   Path,
    title:  str,
    ref_or: float | None = None,
    ref_label: str = "",
) -> None:
    n = len(models)
    fig, ax = plt.subplots(figsize=(10, 0.6 * n + 1.8))

    ys, ors, los, his, labels, pvals = [], [], [], [], [], []
    for label, pred, df in models:
        row = df[df["Variable"] == pred] if not df.empty else pd.DataFrame()
        if row.empty:
            continue
        row = row.iloc[0]
        ys.append(len(ys)); ors.append(row["OR"]); los.append(row["CI_lo"])
        his.append(row["CI_hi"]); labels.append(label); pvals.append(row["P"])

    for y, o, lo, hi, p in zip(ys, ors, los, his, pvals):
        c = "#d62728" if p < 0.05 else "#888888"
        ax.plot([lo, hi], [y, y], color=c, lw=1.8, zorder=2)
        ax.plot(o, y, "o", color=c, ms=7, zorder=3)
        ax.text(max(his + [2.5]) * 1.06, y,
                f"OR {fmt_or(o, lo, hi)}\np={fmt_p(p)}",
                va="center", ha="left", fontsize=8.5, color=c)

    ax.axvline(1.0, color="grey", ls="--", lw=1.0)
    if ref_or is not None:
        ax.axvline(ref_or, color="steelblue", ls=":", lw=1.5, alpha=0.85)
        ax.text(ref_or, -0.7, ref_label, color="steelblue",
                ha="center", va="top", fontsize=8, style="italic")

    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Odds Ratio (95% CI)", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8); ax.set_xlim(left=0)
    ax.invert_yaxis(); plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {path.name}")


def forest_comparison(
    left_models:  list[tuple[str, str, pd.DataFrame]],
    right_models: list[tuple[str, str, pd.DataFrame]],
    path: Path,
    left_title:  str = "HR AC1 — OR (95% CI)",
    right_title: str = "HR Euler χ(0) — OR (95% CI)",
) -> None:
    def _rows(models):
        out = []
        for lbl, pred, df in models:
            row = df[df["Variable"] == pred] if not df.empty else pd.DataFrame()
            if not row.empty:
                out.append((lbl, pred, row.iloc[0]))
        return out

    lr = _rows(left_models); rr = _rows(right_models)
    n  = max(len(lr), len(rr))
    if n == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 0.6 * n + 2.2), sharey=True)
    fig.suptitle("eICU External Validation: AC1 vs Euler χ(0)", fontsize=11)

    for ax, rows, xtitle in [(axes[0], lr, left_title), (axes[1], rr, right_title)]:
        labels_drawn = []
        all_hi = [r["CI_hi"] for _, _, r in rows]
        x_max  = max(all_hi + [2.5]) if all_hi else 2.5
        for i, (label, pred, row) in enumerate(rows):
            o, lo, hi, p = row["OR"], row["CI_lo"], row["CI_hi"], row["P"]
            c = "#d62728" if p < 0.05 else "#888888"
            ax.plot([lo, hi], [i, i], color=c, lw=1.8, zorder=2)
            ax.plot(o, i, "o", color=c, ms=7, zorder=3)
            ax.text(x_max * 1.03, i, f"{fmt_or(o, lo, hi)}  p={fmt_p(p)}",
                    va="center", ha="left", fontsize=8, color=c)
            labels_drawn.append(label)
        ax.axvline(1.0, color="grey", ls="--", lw=1.0)
        ax.set_yticks(range(len(labels_drawn)))
        ax.set_yticklabels(labels_drawn, fontsize=9)
        ax.set_xlabel("Odds Ratio", fontsize=9)
        ax.set_title(xtitle, fontsize=10)
        ax.set_xlim(left=0); ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {path.name}")


def timeseries_plot(windows: pd.DataFrame, cohort: pd.DataFrame, path: Path) -> None:
    """shock/control 的 AC1 和 Euler 均值 ±95%CI 时间序列图。"""
    wins = windows.merge(
        cohort[["patientunitstayid", "T0_min", "group"]].drop_duplicates(),
        on=["patientunitstayid", "T0_min"], how="left",
    )
    bins = np.arange(-48, 1, 3)
    wins["h_bin"] = pd.cut(wins["hours_before_T0"], bins=bins,
                           labels=(bins[:-1] + 1.5).round(1))

    metrics = [
        ("ac1_hr",   "low_conf_hr",    "HR AC1",        "AC1"),
        ("euler_hr", "low_conf_euler", "HR Euler χ(0)", "Euler"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    fig.suptitle("eICU: EWS time series (shock vs control)", fontsize=11)

    for ax, (metric, conf_col, ylabel, short) in zip(axes, metrics):
        for grp, color in COLORS.items():
            sub = wins[(wins["group"] == grp) & ~wins[conf_col]]
            agg = (sub.groupby("h_bin", observed=True)[metric]
                   .agg(mean="mean", sem=lambda x: x.sem()))
            x = agg.index.astype(float)
            ax.plot(x, agg["mean"], color=color, label=grp, lw=1.8)
            ax.fill_between(x,
                            agg["mean"] - 1.96 * agg["sem"],
                            agg["mean"] + 1.96 * agg["sem"],
                            alpha=0.15, color=color)
        ax.axvline(-12, color="grey", ls=":", lw=1.0, label="Late window start")
        ax.set_xlabel("Hours before T0", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {path.name}")


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载数据 ───────────────────────────────────────────────────────────
    print("── 加载数据 ─────────────────────────────────────────────────────────")
    cohort  = pd.read_parquet(COHORT_PATH)
    stats   = pd.read_parquet(STATS_PATH)
    windows = pd.read_parquet(WINDOWS_PATH)
    diag    = pd.read_parquet(DIAG_PATH)
    vitals  = pd.read_parquet(VITALS_PATH)

    # 排除清洗不合格患者
    passed  = diag[~diag["excluded"]][["patientunitstayid", "T0_min"]]
    cohort  = cohort.merge(passed, on=["patientunitstayid", "T0_min"])
    stats   = stats.merge(passed, on=["patientunitstayid", "T0_min"])

    # ── 2. 组装建模数据 ───────────────────────────────────────────────────────
    print("── 组装建模数据 ─────────────────────────────────────────────────────")

    # 晚窗原始 HR/MAP 均值
    late_raw = (
        vitals[
            (vitals["hours_before_T0"] >= LATE_LO) &
            (vitals["hours_before_T0"] < LATE_HI)
        ]
        .groupby(["patientunitstayid", "T0_min"], as_index=False)
        .agg(late_hr_mean_raw=("hr_raw", "mean"), late_map_mean_raw=("map_raw", "mean"))
    )

    # 早窗原始 HR/MAP 均值（供 S3 与主仓库等价）
    early_raw = (
        vitals[
            (vitals["hours_before_T0"] >= EARLY_LO) &
            (vitals["hours_before_T0"] < EARLY_HI)
        ]
        .groupby(["patientunitstayid", "T0_min"], as_index=False)
        .agg(early_hr_mean_raw=("hr_raw", "mean"), early_map_mean_raw=("map_raw", "mean"))
    )

    # 调试指标（总变差、中位数阈值 Euler）
    debug_path = OUTPUT_DIR / "euler_debug_extra.parquet"
    if debug_path.exists():
        debug_extra = pd.read_parquet(debug_path).drop(columns=["T0_h","group","matched_pair_id"],
                                                        errors="ignore")
    else:
        debug_extra = pd.DataFrame()

    # SampEn columns may be absent if 03_ews_eicu.py was run before the update
    sampen_cols = [c for c in ["sampen_rel_hr_late_mean", "sampen_rel_hr_early_mean"]
                   if c in stats.columns]

    mdf = (
        cohort
        .merge(stats[["patientunitstayid", "T0_min",
                       "ac1_hr_late_mean", "ac1_hr_early_mean",
                       "euler_hr_late_mean", "euler_hr_early_mean",
                       "n_extrema_hr_late_mean"] + sampen_cols],
               on=["patientunitstayid", "T0_min"], how="inner")
        .merge(late_raw,  on=["patientunitstayid", "T0_min"], how="left")
        .merge(early_raw, on=["patientunitstayid", "T0_min"], how="left")
        .merge(diag[["patientunitstayid", "T0_min", "dominant_map_source"]],
               on=["patientunitstayid", "T0_min"], how="left")
    )
    if not debug_extra.empty:
        mdf = mdf.merge(debug_extra[["patientunitstayid","T0_min",
                                      "total_var_late","euler_median_late"]],
                        on=["patientunitstayid","T0_min"], how="left")
    mdf["binary_group"] = (mdf["group"] == "shock").astype(int)

    # 监测方式：ABP vs NBP（在 pairs 内有变异，可以保留）
    mdf["mon_abp"] = (mdf["dominant_map_source"] == "abp").astype(float)
    # 填充协变量 NaN
    for col in BASE_COVARS:
        if col in mdf.columns:
            mdf[col] = mdf[col].fillna(0).astype(float)

    print(f"  总行数: {len(mdf):,}  shock: {mdf['binary_group'].sum():,}  "
          f"control: {(~mdf['binary_group'].astype(bool)).sum():,}")
    print(f"  ac1_hr_late_mean 缺失: {mdf['ac1_hr_late_mean'].isna().sum()}")
    print(f"  euler_hr_late_mean 缺失: {mdf['euler_hr_late_mean'].isna().sum()}")
    if "sampen_rel_hr_late_mean" in mdf.columns:
        print(f"  sampen_rel_hr_late_mean 缺失: {mdf['sampen_rel_hr_late_mean'].isna().sum()}")

    # ── 3. AC1 模型 ───────────────────────────────────────────────────────────
    print("\n── AC1 条件 logistic 模型 ───────────────────────────────────────────")

    ac1_m1 = fit_clogit(mdf, "ac1_hr_late_mean",  label="M1 主模型")
    ac1_s1 = fit_clogit(mdf, "ac1_hr_late_mean",
                        extra_covars=["late_hr_mean_raw"], label="S1 +晚窗HR均值")
    ac1_s2 = fit_clogit(mdf, "ac1_hr_late_mean",
                        extra_covars=["late_hr_mean_raw", "late_map_mean_raw"],
                        label="S2 +晚窗HR+MAP均值")

    mdf_early       = mdf[mdf["ac1_hr_early_mean"].notna()].copy()
    mdf_early_euler = mdf[mdf["euler_hr_early_mean"].notna()].copy()
    ac1_s3 = fit_clogit(mdf_early, "ac1_hr_early_mean",
                        extra_covars=["early_hr_mean_raw", "early_map_mean_raw"],
                        label="S3 早窗+均值调整")

    for nm, pred, df in [("M1", "ac1_hr_late_mean",  ac1_m1),
                         ("S1", "ac1_hr_late_mean",  ac1_s1),
                         ("S2", "ac1_hr_late_mean",  ac1_s2),
                         ("S3", "ac1_hr_early_mean", ac1_s3)]:
        if not df.empty:
            r = df[df["Variable"] == pred].iloc[0]
            print(f"  {nm}: OR={r['OR']:.2f} ({r['CI_lo']:.2f}–{r['CI_hi']:.2f})"
                  f"  p={fmt_p(r['P'])}  N={r['N_obs']:,}  strata={r['N_strata']:,}")

    # ── 4. Euler 模型 ─────────────────────────────────────────────────────────
    print("\n── Euler χ(0) 条件 logistic 模型 ───────────────────────────────────")

    eul_m1 = fit_clogit(mdf, "euler_hr_late_mean",  label="Euler M1")
    eul_s1 = fit_clogit(mdf, "euler_hr_late_mean",
                        extra_covars=["late_hr_mean_raw"], label="Euler S1 +HR均值")
    eul_s2 = fit_clogit(mdf, "euler_hr_late_mean",
                        extra_covars=["late_hr_mean_raw", "late_map_mean_raw"],
                        label="Euler S2 +HR+MAP均值")
    eul_s3 = fit_clogit(mdf_early_euler, "euler_hr_early_mean",
                        extra_covars=["early_hr_mean_raw", "early_map_mean_raw"],
                        label="Euler S3 早窗+均值调整")

    for nm, pred, df in [("M1", "euler_hr_late_mean",  eul_m1),
                         ("S1", "euler_hr_late_mean",  eul_s1),
                         ("S2", "euler_hr_late_mean",  eul_s2),
                         ("S3", "euler_hr_early_mean", eul_s3)]:
        if not df.empty:
            r = df[df["Variable"] == pred].iloc[0]
            print(f"  {nm}: OR={r['OR']:.2f} ({r['CI_lo']:.2f}–{r['CI_hi']:.2f})"
                  f"  p={fmt_p(r['P'])}  N={r['N_obs']:,}  strata={r['N_strata']:,}")

    # ── 4b. n_extrema + T0 分层敏感性分析 ────────────────────────────────
    print("\n── n_extrema & T0 分层敏感性分析 ───────────────────────────────────")
    print("   MIMIC-IV: AC1 OR=1.60(p=0.005)  n_extrema OR=0.88(p<0.001)  Euler OR=0.75(p<0.001)")

    T0_THRESHOLDS = [("全样本", None), ("T0≥24h", 24), ("T0≥48h", 48), ("T0≥72h", 72)]
    STRAT_METRICS = [
        ("ac1_hr_late_mean",           "AC1",              []),
        ("n_extrema_hr_late_mean",     "n_extrema",        []),
        ("euler_hr_late_mean",         "Euler",            []),
    ] + (
        [("sampen_rel_hr_late_mean",   "SampEn(r=0.5)",    [])]
        if "sampen_rel_hr_late_mean" in mdf.columns else []
    )
    # 参考值来源：euler_ews/scripts/11_euler_logistic.py（M1 主模型，单次去趋势）
    # AC1：ac1/scripts/04_logistic_regression.py Table 2 主模型（1.60，非 1.72）
    # SampEn：13_sampen_comparison.py M1（p=0.178, not significant）
    MIMIC_REF = {"AC1": 1.60, "n_extrema": 0.88, "Euler": 0.75, "SampEn(r=0.5)": 0.86}

    # t0_results[(metric, strat_label)] → fit_clogit DataFrame（可能为空）
    t0_results: dict[tuple, pd.DataFrame] = {}
    t0_rows = []

    for strat_label, th in T0_THRESHOLDS:
        sub = mdf if th is None else mdf[mdf["T0_h"] >= th].copy()
        n_pairs = sub["matched_pair_id"].nunique()
        print(f"\n  [{strat_label}]  患者={len(sub):,}  匹配对≤{n_pairs}")
        for metric, mname, extra in STRAT_METRICS:
            r_df = fit_clogit(sub, metric, extra_covars=extra,
                              label=f"{mname} {strat_label}")
            t0_results[(metric, strat_label)] = r_df
            if not r_df.empty:
                r = r_df[r_df["Variable"] == metric].iloc[0]
                print(f"    {mname:10s}: OR={r['OR']:.2f} ({r['CI_lo']:.2f}–{r['CI_hi']:.2f})"
                      f"  p={fmt_p(r['P'])}  N={r['N_obs']:,}  strata={r['N_strata']:,}")
                t0_rows.append({
                    "T0_strat": strat_label, "metric": mname,
                    "N_eligible_pairs": n_pairs,       # 分层前匹配对数（上限）
                    "N_informative_strata": int(r["N_strata"]),  # 实际进模型 strata 数
                    "N_obs": int(r["N_obs"]),
                    "OR": r["OR"], "CI_lo": r["CI_lo"], "CI_hi": r["CI_hi"], "P": r["P"],
                    "OR_fmt": fmt_or(r["OR"], r["CI_lo"], r["CI_hi"]),
                    "P_fmt": fmt_p(r["P"]),
                    "MIMIC_ref_OR": MIMIC_REF.get(mname, np.nan),
                })
            else:
                t0_rows.append({
                    "T0_strat": strat_label, "metric": mname,
                    "N_eligible_pairs": n_pairs,
                    "N_informative_strata": np.nan,
                    "N_obs": np.nan,
                    "OR": np.nan, "CI_lo": np.nan, "CI_hi": np.nan, "P": np.nan,
                    "OR_fmt": "—", "P_fmt": "—",
                    "MIMIC_ref_OR": MIMIC_REF.get(mname, np.nan),
                })

    t0_tbl = pd.DataFrame(t0_rows)

    # 直接从 t0_results 取 DataFrame 供 models_list 使用
    nex_m1     = t0_results.get(("n_extrema_hr_late_mean", "全样本"),   pd.DataFrame())
    nex_m1_48h = t0_results.get(("n_extrema_hr_late_mean", "T0≥48h"),  pd.DataFrame())
    nex_m1_72h = t0_results.get(("n_extrema_hr_late_mean", "T0≥72h"),  pd.DataFrame())
    eul_m1_48h = t0_results.get(("euler_hr_late_mean",      "T0≥48h"), pd.DataFrame())
    eul_m1_72h = t0_results.get(("euler_hr_late_mean",      "T0≥72h"), pd.DataFrame())
    nex_s1     = fit_clogit(mdf, "n_extrema_hr_late_mean",
                             extra_covars=["late_hr_mean_raw"], label="nExtrema S1 +HR均值")

    # ── 4c. SampEn T0 分层（若列存在）────────────────────────────────────────
    if "sampen_rel_hr_late_mean" in mdf.columns:
        print("\n── SampEn(r=0.5) T0 分层敏感性分析 ─────────────────────────────────")
        print("   MIMIC-IV 参考: SampEn M1 OR=0.86(p=0.178)  M1 不显著")
        for strat_label, th in T0_THRESHOLDS:
            sub = mdf if th is None else mdf[mdf["T0_h"] >= th].copy()
            r_df = fit_clogit(sub, "sampen_rel_hr_late_mean", label=f"SampEn {strat_label}")
            if not r_df.empty:
                r = r_df[r_df["Variable"] == "sampen_rel_hr_late_mean"].iloc[0]
                print(f"  [{strat_label:8s}] OR={r['OR']:.2f} ({r['CI_lo']:.2f}–{r['CI_hi']:.2f})"
                      f"  p={fmt_p(r['P'])}  N={r['N_obs']:,}  strata={r['N_strata']:,}")
                t0_rows.append({
                    "T0_strat": strat_label, "metric": "SampEn(r=0.5)",
                    "N_eligible_pairs": sub["matched_pair_id"].nunique(),
                    "N_informative_strata": int(r["N_strata"]),
                    "N_obs": int(r["N_obs"]),
                    "OR": r["OR"], "CI_lo": r["CI_lo"], "CI_hi": r["CI_hi"], "P": r["P"],
                    "OR_fmt": fmt_or(r["OR"], r["CI_lo"], r["CI_hi"]),
                    "P_fmt": fmt_p(r["P"]),
                    "MIMIC_ref_OR": 0.86,   # SampEn MIMIC M1 (p=0.178, ns)
                })

    # ── 5. 汇总表 ─────────────────────────────────────────────────────────────
    print("\n── 汇总表 ───────────────────────────────────────────────────────────")

    def make_summary(models_list):
        rows = []
        for label, pred, df in models_list:
            if df.empty:
                rows.append({"模型": label, "主预测变量": pred,
                             "N obs": 0, "N strata": 0,
                             "OR": np.nan, "CI_lo": np.nan, "CI_hi": np.nan,
                             "P": np.nan, "OR_fmt": "—", "P_fmt": "—"})
                continue
            r = df[df["Variable"] == pred].iloc[0]
            rows.append({"模型": label, "主预测变量": pred,
                         "N obs": r["N_obs"], "N strata": r["N_strata"],
                         "OR": r["OR"], "CI_lo": r["CI_lo"], "CI_hi": r["CI_hi"],
                         "P": r["P"],
                         "OR_fmt": fmt_or(r["OR"], r["CI_lo"], r["CI_hi"]),
                         "P_fmt": fmt_p(r["P"])})
        return pd.DataFrame(rows)

    ac1_models_list = [
        ("M1 主模型",        "ac1_hr_late_mean",  ac1_m1),
        ("S1 +晚窗HR均值",   "ac1_hr_late_mean",  ac1_s1),
        ("S2 +晚窗HR+MAP",   "ac1_hr_late_mean",  ac1_s2),
        ("S3 早窗+均值调整",  "ac1_hr_early_mean", ac1_s3),
    ]
    eul_models_list = [
        ("Euler M1",         "euler_hr_late_mean",  eul_m1),
        ("Euler S1 +HR均值", "euler_hr_late_mean",  eul_s1),
        ("Euler S2 +HR+MAP", "euler_hr_late_mean",  eul_s2),
        ("Euler S3 早窗+均值调整", "euler_hr_early_mean", eul_s3),
    ]
    nex_models_list = [
        ("nExtrema M1 全样本",  "n_extrema_hr_late_mean", nex_m1),
        ("nExtrema S1 +HR均值", "n_extrema_hr_late_mean", nex_s1),
        ("nExtrema M1 T0≥48h",  "n_extrema_hr_late_mean", nex_m1_48h),
        ("nExtrema M1 T0≥72h",  "n_extrema_hr_late_mean", nex_m1_72h),
    ]

    ac1_tbl = make_summary(ac1_models_list)
    eul_tbl = make_summary(eul_models_list)
    nex_tbl = make_summary(nex_models_list)
    ac1_tbl.to_csv(OUTPUT_DIR / "tbl_validation_ac1.csv",       index=False, encoding="utf-8-sig")
    eul_tbl.to_csv(OUTPUT_DIR / "tbl_validation_euler.csv",     index=False, encoding="utf-8-sig")
    nex_tbl.to_csv(OUTPUT_DIR / "tbl_validation_nExtrema.csv",  index=False, encoding="utf-8-sig")
    t0_tbl.to_csv(OUTPUT_DIR / "tbl_t0_sensitivity.csv",        index=False, encoding="utf-8-sig")

    print("\nAC1 汇总:")
    print(ac1_tbl[["模型", "N obs", "OR_fmt", "P_fmt"]].to_string(index=False))
    print("\nEuler 汇总:")
    print(eul_tbl[["模型", "N obs", "OR_fmt", "P_fmt"]].to_string(index=False))
    print("\nn_extrema + T0 分层汇总:")
    print(nex_tbl[["模型", "N obs", "OR_fmt", "P_fmt"]].to_string(index=False))
    print("\nT0 分层敏感性（完整矩阵）:")
    print(t0_tbl[["T0_strat","metric","N_eligible_pairs","N_informative_strata",
                  "OR_fmt","P_fmt","MIMIC_ref_OR"]].to_string(index=False))

    # ── 6. 读取 MIMIC-IV 参考 OR ─────────────────────────────────────────────
    mimic_ref_path = NAAS_ROOT / "ac1" / "output" / "table2_multivariable.csv"
    # 先用硬编码参考值兜底，避免读到陈旧文件里的旧数值（如 1.72）
    mimic_ref_or: float | None = MIMIC_REF["AC1"]
    if mimic_ref_path.exists():
        try:
            mimic_ref = pd.read_csv(mimic_ref_path)
            ac1_rows  = mimic_ref[mimic_ref.apply(
                lambda r: "ac1" in str(r.get("variable","")).lower() or
                          "ac1" in str(r.get("Variable","")).lower(), axis=1)]
            if not ac1_rows.empty:
                for col in ["OR", "or", "Or"]:
                    if col in ac1_rows.columns:
                        mimic_ref_or = float(ac1_rows.iloc[0][col])
                        break
        except Exception:
            pass

    # ── 7. 图表 ───────────────────────────────────────────────────────────────
    print("\n── 出图 ─────────────────────────────────────────────────────────────")

    timeseries_plot(windows, cohort, OUTPUT_DIR / "fig_timeseries_eicu.png")

    forest_plot(
        ac1_models_list,
        OUTPUT_DIR / "fig_forest_ac1_eicu.png",
        title="eICU External Validation: HR AC1 — Conditional Logistic OR",
        ref_or=mimic_ref_or,
        ref_label=f"MIMIC-IV OR={mimic_ref_or:.2f}" if mimic_ref_or else "",
    )
    forest_plot(
        eul_models_list,
        OUTPUT_DIR / "fig_forest_euler_eicu.png",
        title="eICU External Validation: HR Euler χ(0) — Conditional Logistic OR",
    )
    forest_comparison(
        ac1_models_list, eul_models_list,
        OUTPUT_DIR / "fig_forest_comparison_eicu.png",
    )
    forest_plot(
        nex_models_list,
        OUTPUT_DIR / "fig_forest_nExtrema_eicu.png",
        title="eICU: HR n_extrema — T0-stratified sensitivity (MIMIC-IV ref OR=0.88)",
        ref_or=0.88,
        ref_label="MIMIC-IV OR=0.88",
    )

    # T0 分层三联 forest plot
    _t0_labels = [lbl for lbl, _ in T0_THRESHOLDS]
    _metrics_plot = [
        ("ac1_hr_late_mean",       "HR AC1",        MIMIC_REF["AC1"],       "#1f77b4"),
        ("n_extrema_hr_late_mean", "HR n_extrema",  MIMIC_REF["n_extrema"], "#2ca02c"),
        ("euler_hr_late_mean",     "HR Euler χ(0)", MIMIC_REF["Euler"],     "#d62728"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("eICU: T0-stratified sensitivity analysis — longer pre-event window → stronger signal",
                 fontsize=10)
    for ax, (metric, mname, ref_or, color) in zip(axes, _metrics_plot):
        ors, los, his, ps = [], [], [], []
        for strat_label, _ in T0_THRESHOLDS:
            r_df = t0_results.get((metric, strat_label), pd.DataFrame())
            if not r_df.empty:
                r = r_df[r_df["Variable"] == metric].iloc[0]
                ors.append(r["OR"]); los.append(r["CI_lo"])
                his.append(r["CI_hi"]); ps.append(r["P"])
            else:
                ors.append(np.nan); los.append(np.nan)
                his.append(np.nan); ps.append(1.0)

        x_max = max([h for h in his if not np.isnan(h)] + [2.5])
        for i, (o, lo, hi, p) in enumerate(zip(ors, los, his, ps)):
            if np.isnan(o):
                continue
            c = color if p < 0.05 else "#aaaaaa"
            ms = 9 if p < 0.05 else 6
            ax.plot([lo, hi], [i, i], color=c, lw=2, zorder=2)
            ax.plot(o, i, "o", color=c, ms=ms, zorder=3)
            pstr = "<.001" if p < 0.001 else f"{p:.3f}"
            sig  = " *" if p < 0.05 else ""
            ax.text(x_max * 1.06, i,
                    f"{o:.2f} ({lo:.2f}–{hi:.2f})\np={pstr}{sig}",
                    va="center", ha="left", fontsize=7.5, color=c)
        ax.axvline(1.0, color="grey", ls="--", lw=1.0)
        ax.axvline(ref_or, color="steelblue", ls=":", lw=2, alpha=0.7)
        ax.text(ref_or, -0.65, f"MIMIC\n{ref_or:.2f}", color="steelblue",
                ha="center", va="top", fontsize=7.5, style="italic")
        ax.set_yticks(range(len(_t0_labels)))
        ax.set_yticklabels(_t0_labels, fontsize=9)
        ax.set_xlabel("Odds Ratio (95% CI)", fontsize=9)
        ax.set_title(mname, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_t0strat_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_t0strat_sensitivity.png")

    print(f"\n[完成] 结果输出至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
