"""
探索性分析（不计入投稿稿件）— 方案 C 深化

HR χ(0) / n_extrema 的 Conditional Logistic Regression + 敏感性分析

主模型 (M1)
    shock ~ euler_hr_late + vent + sedation + betablocker + icu_type + monitoring
    strata = matched_pair_id

敏感性分析
    S1  : + late_hr_mean_raw                       ← 复现 AC1"消失"的关键调整
    S2  : + late_hr_mean_raw + late_map_mean_raw
    S3  : 早窗 euler_hr_early_mean 作为主预测变量
    S4  : n_extrema_hr_late_mean 替代主预测变量
    S5  : 无镇静亚组（sedation_before_window=0）
    S6  : 并排对比主文 AC1 结果（OR forest plot）

输出：output/explore/
"""

import warnings
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "DejaVu Sans"],
    "axes.unicode_minus": False,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from statsmodels.discrete.conditional_models import ConditionalLogit

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# euler_ews/scripts/ → euler_ews/ → NaaS/
PROJECT_ROOT = Path(__file__).parent.parent          # euler_ews/
NAAS_ROOT    = PROJECT_ROOT.parent                   # NaaS/  (共享数据在此)

COHORT_PATH  = NAAS_ROOT / "data" / "cohort.parquet"
VITALS_PATH  = NAAS_ROOT / "data" / "vitals_cleaned.parquet"
DIAG_PATH    = NAAS_ROOT / "data" / "cleaning_diagnostics.parquet"
DB_PATH      = NAAS_ROOT / "mimiciv" / "mimiciv.db"
OUTPUT_DIR   = PROJECT_ROOT / "output"

WINDOW_SIZE = 12
MIN_ACTUAL  = 6
EARLY_LO, EARLY_HI = -48, -24
LATE_LO,  LATE_HI  = -12,   0

SEDATIVE_ITEMIDS    = (221385, 221623, 221668, 222168, 225150, 229420)
BETABLOCKER_ITEMIDS = (221429, 225153, 225974)


# ── 欧拉特征计算（与 09 一致）────────────────────────────────────────────────

def euler_at_zero(vals: np.ndarray, is_interp: np.ndarray) -> float:
    actual = vals[~is_interp & ~np.isnan(vals)]
    if len(actual) < MIN_ACTUAL:
        return np.nan
    below = actual <= 0
    if not below.any():
        return 0.0
    n_comp = int(below[0]) + int((np.diff(below.astype(np.int8)) == 1).sum())
    return float(n_comp)


def n_extrema(vals: np.ndarray, is_interp: np.ndarray) -> float:
    actual = vals[~is_interp & ~np.isnan(vals)]
    if len(actual) < MIN_ACTUAL:
        return np.nan
    n_min = len(argrelextrema(actual, np.less,    order=1)[0])
    n_max = len(argrelextrema(actual, np.greater, order=1)[0])
    return float(n_min + n_max)


def window_features(ts: pd.DataFrame, T0: pd.Timestamp) -> pd.DataFrame:
    ts = ts.sort_values("charttime").reset_index(drop=True)
    if len(ts) < WINDOW_SIZE:
        return pd.DataFrame()
    hr_vals   = ts["hr_residual"].to_numpy(float)
    hr_interp = ts["hr_is_interpolated"].to_numpy(bool)
    rows = []
    for i in range(len(ts) - WINDOW_SIZE + 1):
        sl = slice(i, i + WINDOW_SIZE)
        hv, hi  = hr_vals[sl], hr_interp[sl]
        n_act   = int((~hi & ~np.isnan(hv)).sum())
        low     = n_act < MIN_ACTUAL
        center  = ts["charttime"].iloc[i + WINDOW_SIZE // 2]
        h_before = (center - T0).total_seconds() / 3600
        rows.append({
            "hours_before_T0": round(h_before, 1),
            "euler_hr":        euler_at_zero(hv, hi),
            "n_extrema_hr":    n_extrema(hv, hi),
            "low_conf":        low,
        })
    return pd.DataFrame(rows)


def build_euler_features(vitals: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """计算每个 (stay_id, T0) 的早/晚窗 euler_hr 和 n_extrema_hr 均值。"""
    print("── 计算欧拉特征窗口 ─────────────────────────────────────────────")
    pairs   = vitals.groupby(["stay_id", "T0"], sort=False)
    n_total = pairs.ngroups
    rows = []
    for i, ((sid, t0), grp) in enumerate(pairs, 1):
        wins = window_features(grp, pd.Timestamp(t0))
        if wins.empty:
            continue
        def wmean(lo, hi):
            m = (wins["hours_before_T0"] >= lo) & (wins["hours_before_T0"] < hi) & (~wins["low_conf"])
            return wins.loc[m, "euler_hr"].dropna().mean(), wins.loc[m, "n_extrema_hr"].dropna().mean()
        ee, en  = wmean(EARLY_LO, EARLY_HI)
        le, ln  = wmean(LATE_LO,  LATE_HI)
        rows.append({
            "stay_id":              sid,
            "T0":                   pd.Timestamp(t0),
            "euler_hr_early_mean":  ee,
            "euler_hr_late_mean":   le,
            "n_extrema_hr_early_mean": en,
            "n_extrema_hr_late_mean":  ln,
        })
        if i % 500 == 0:
            print(f"  {i:,}/{n_total:,}...", flush=True)
    return pd.DataFrame(rows)


# ── 临床协变量（DuckDB）─────────────────────────────────────────────────────

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
        # ICU 类型
        con.register("_ids", df[["stay_id", "subject_id"]].drop_duplicates())
        demog = con.execute("""
            SELECT ie.stay_id, ie.first_careunit
            FROM _ids i
            INNER JOIN mimiciv_icu.icustays ie ON ie.stay_id = i.stay_id
        """).df()

        # 有创通气 at −12h
        con.register("_st0", df[["stay_id", "T0"]].drop_duplicates())
        vent = con.execute("""
            SELECT DISTINCT s.stay_id, s.T0, 1 AS vent_before_window
            FROM mimiciv_derived.ventilation v
            INNER JOIN _st0 s ON v.stay_id = s.stay_id
            WHERE v.ventilation_status = 'InvasiveVent'
              AND v.starttime <= s.T0 - INTERVAL '12 hours'
              AND (v.endtime IS NULL OR v.endtime >= s.T0 - INTERVAL '12 hours')
        """).df()

        # 镇静药 prior 12h
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

        # β 受体阻滞剂 prior 12h
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
    mdf["icu_type"]   = mdf["first_careunit"].apply(simplify_careunit)
    mdf["icu_SICU"]     = (mdf["icu_type"] == "SICU").astype(float)
    mdf["icu_CCU_CSRU"] = (mdf["icu_type"] == "CCU_CSRU").astype(float)
    mdf["icu_Neuro"]    = (mdf["icu_type"] == "Neuro_NICU").astype(float)
    mdf["icu_Other"]    = (mdf["icu_type"] == "Other").astype(float)
    return mdf


# ── 条件 Logistic 拟合 ────────────────────────────────────────────────────────

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
    extra = extra_covars or []
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
            "Variable":  var,
            "OR":        np.exp(result.params[i]),
            "CI_lo":     np.exp(cis[i, 0]),
            "CI_hi":     np.exp(cis[i, 1]),
            "P":         float(result.pvalues[i]),
            "N_obs":     len(df2),
            "N_strata":  len(valid),
            "model":     label,
        })
    return pd.DataFrame(rows)


# ── 格式化工具 ────────────────────────────────────────────────────────────────

def fmt_or(or_: float, lo: float, hi: float) -> str:
    return f"{or_:.2f} ({lo:.2f}–{hi:.2f})"


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def print_result(df: pd.DataFrame, predictor: str, title: str) -> None:
    row = df[df["Variable"] == predictor].iloc[0]
    print(f"  {title}: OR={row['OR']:.2f} ({row['CI_lo']:.2f}–{row['CI_hi']:.2f}), "
          f"p={fmt_p(row['P'])}, N={row['N_obs']:,}, strata={row['N_strata']:,}")


# ── Forest plot ───────────────────────────────────────────────────────────────

def forest_plot(
    models: list[tuple[str, str, pd.DataFrame]],  # (label, predictor, df)
    path: Path,
    title: str,
    ref_or: float | None = None,
    ref_label: str = "",
) -> None:
    n = len(models)
    fig, ax = plt.subplots(figsize=(9, 0.65 * n + 1.8))

    ys, ors, los, his, labels, pvals = [], [], [], [], [], []
    for label, pred, df in models:
        row = df[df["Variable"] == pred]
        if row.empty:
            continue
        row = row.iloc[0]
        ys.append(len(ys))
        ors.append(row["OR"])
        los.append(row["CI_lo"])
        his.append(row["CI_hi"])
        labels.append(label)
        pvals.append(row["P"])

    x_hi   = max(his + [2.0])
    x_text = x_hi * 1.05

    for y, o, lo, hi, p in zip(ys, ors, los, his, pvals):
        color = "#c0392b" if p < 0.05 else "#7f8c8d"
        ax.plot([lo, hi], [y, y], color=color, lw=2.0, zorder=2,
                solid_capstyle="round")
        ax.plot(o, y, "o", color=color, ms=8, zorder=3,
                markeredgewidth=0.5, markeredgecolor="white")
        ax.text(x_text, y,
                f"OR {fmt_or(o, lo, hi)}\np={fmt_p(p)}",
                va="center", ha="left", fontsize=8.5, color=color, linespacing=1.4)

    ax.axvline(1.0, color="#95a5a6", ls="--", lw=1.0, zorder=1)
    if ref_or is not None:
        ax.axvline(ref_or, color="#2980b9", ls=":", lw=1.5, alpha=0.85, zorder=1)
        ax.text(ref_or, -0.65, ref_label, color="#2980b9",
                ha="center", va="top", fontsize=8, style="italic")

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Odds Ratio (95% CI)", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8, fontweight="bold")
    ax.set_xlim(left=0, right=x_text + 0.5)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def forest_comparison(
    euler_models: list[tuple[str, str, pd.DataFrame]],
    ac1_models:   list[tuple[str, str, pd.DataFrame]],
    path: Path,
) -> None:
    """左右两列分别展示 euler_hr 和 AC1 同名模型的 OR。

    标签与绘图点来源相同（以 pred 变量实际存在于 df 为准），
    避免预先按固定变量名过滤标签导致标签数与点数不一致。
    """
    def _rows(models):
        """提取可成功绘制的行，返回 list[(label, pred, Series)]。"""
        out = []
        for lbl, pred, df in models:
            row = df[df["Variable"] == pred]
            if not row.empty:
                out.append((lbl, pred, row.iloc[0]))
        return out

    euler_rows = _rows(euler_models)
    ac1_rows   = _rows(ac1_models)
    n = max(len(euler_rows), len(ac1_rows))

    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * n + 2.0), sharey=True)
    fig.suptitle("Euler χ(0) vs HR AC1: OR comparison across models", fontsize=11)

    # Global x range across both panels for visual consistency
    all_his_global = [r["CI_hi"] for rows in [euler_rows, ac1_rows] for _, _, r in rows]
    x_hi_global = max(all_his_global + [2.5]) if all_his_global else 2.5
    x_text = x_hi_global * 1.04

    for ax, rows, x_label in [
        (axes[0], euler_rows, r"HR Euler $\chi(0)$ — OR (95% CI)"),
        (axes[1], ac1_rows,   "HR AC1 — OR (95% CI)"),
    ]:
        labels_drawn = []

        for i, (label, pred, row) in enumerate(rows):
            o, lo, hi, p = row["OR"], row["CI_lo"], row["CI_hi"], row["P"]
            color = "#c0392b" if p < 0.05 else "#7f8c8d"
            ax.plot([lo, hi], [i, i], color=color, lw=2.0, zorder=2,
                    solid_capstyle="round")
            ax.plot(o, i, "o", color=color, ms=8, zorder=3,
                    markeredgewidth=0.5, markeredgecolor="white")
            ax.text(x_text, i,
                    f"{fmt_or(o, lo, hi)}  p={fmt_p(p)}",
                    va="center", ha="left", fontsize=8.5, color=color)
            labels_drawn.append(label)

        ax.axvline(1.0, color="#95a5a6", ls="--", lw=1.0)
        ax.set_yticks(range(len(labels_drawn)))
        ax.set_yticklabels(labels_drawn, fontsize=9.5)
        ax.set_xlabel("Odds Ratio", fontsize=9.5)
        ax.set_title(x_label, fontsize=10.5, fontweight="bold", pad=6)
        ax.set_xlim(left=0, right=x_text + 0.55)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 数据加载 ───────────────────────────────────────────────────────────
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

    # ── 2. 欧拉特征 ──────────────────────────────────────────────────────────
    euler_feat = build_euler_features(vitals, cohort)

    # ── 3. 晚窗 / 早窗原始 HR/MAP 均值（敏感性用）──────────────────────────
    late_raw = (
        vitals.loc[
            (vitals["charttime"] > vitals["T0"] - pd.Timedelta(hours=12)) &
            (vitals["charttime"] <= vitals["T0"])
        ]
        .groupby(["stay_id", "T0"], as_index=False)
        .agg(late_hr_mean_raw=("hr_raw", "mean"), late_map_mean_raw=("map_raw", "mean"))
    )
    # 早窗原始均值：与 07_multivariable_model.py 的规格对齐（用于早窗模型调整）
    early_raw = (
        vitals.loc[
            (vitals["charttime"] > vitals["T0"] - pd.Timedelta(hours=48)) &
            (vitals["charttime"] <= vitals["T0"] - pd.Timedelta(hours=24))
        ]
        .groupby(["stay_id", "T0"], as_index=False)
        .agg(early_hr_mean_raw=("hr_raw", "mean"), early_map_mean_raw=("map_raw", "mean"))
    )

    # ── 4. 组装建模数据 ──────────────────────────────────────────────────────
    print("\n── 组装建模数据（DuckDB 协变量）───────────────────────────────")
    mdf = (
        cohort[["stay_id", "T0", "subject_id", "group"]]
        .merge(euler_feat, on=["stay_id", "T0"], how="inner")
        .merge(diag[["stay_id", "T0", "dominant_source"]], on=["stay_id", "T0"], how="left")
        .merge(late_raw,  on=["stay_id", "T0"], how="left")
        .merge(early_raw, on=["stay_id", "T0"], how="left")
    )
    mdf["binary_group"] = (mdf["group"] == "shock").astype(int)
    mdf["monitoring"]   = mdf["dominant_source"].str.lower().fillna("nbp")
    mdf["mon_abp"]      = (mdf["monitoring"] == "abp").astype(float)
    mdf["mon_mixed"]    = (mdf["monitoring"] == "mixed").astype(float)

    mdf = pull_covariates(mdf)
    print(f"  建模样本: {len(mdf):,}  shock={mdf['binary_group'].sum():,}  "
          f"control={(~mdf['binary_group'].astype(bool)).sum():,}")
    print(f"  euler_hr_late_mean 缺失: {mdf['euler_hr_late_mean'].isna().sum()}")
    print(f"  n_extrema_hr_late_mean 缺失: {mdf['n_extrema_hr_late_mean'].isna().sum()}")

    # ── 5. 主模型 + 敏感性 ────────────────────────────────────────────────────
    print("\n── 拟合条件 Logistic 模型 ──────────────────────────────────────")

    m1 = fit_clogit(mdf, "euler_hr_late_mean",
                    label="M1 Base model")
    print_result(m1, "euler_hr_late_mean", "M1 euler_hr_late (base)")

    s1 = fit_clogit(mdf, "euler_hr_late_mean",
                    extra_covars=["late_hr_mean_raw"],
                    label="S1 +late HR mean")
    print_result(s1, "euler_hr_late_mean", "S1 +late_hr_mean_raw")

    s2 = fit_clogit(mdf, "euler_hr_late_mean",
                    extra_covars=["late_hr_mean_raw", "late_map_mean_raw"],
                    label="S2 +late HR+MAP means")
    print_result(s2, "euler_hr_late_mean", "S2 +late_hr+map_mean_raw")

    # S3: early-window euler_hr, adjusted for early-window raw HR/MAP means
    s3 = fit_clogit(mdf, "euler_hr_early_mean",
                    extra_covars=["early_hr_mean_raw", "early_map_mean_raw"],
                    label="S3 Early window")
    print_result(s3, "euler_hr_early_mean", "S3 early euler_hr")

    s4 = fit_clogit(mdf, "n_extrema_hr_late_mean",
                    label="S4 N_extrema")
    print_result(s4, "n_extrema_hr_late_mean", "S4 n_extrema late")

    s4b = fit_clogit(mdf, "n_extrema_hr_late_mean",
                     extra_covars=["late_hr_mean_raw"],
                     label="S4b N_extrema +HR mean")
    print_result(s4b, "n_extrema_hr_late_mean", "S4b n_extrema +late_hr")

    mdf_no_sed = mdf[mdf["sedation_before_window"] == 0].copy()
    s5 = fit_clogit(mdf_no_sed, "euler_hr_late_mean",
                    include_sedation=False,
                    label="S5 No-sedation subgroup")
    print_result(s5, "euler_hr_late_mean", "S5 no-sedation subgroup")

    # ── 6. 汇总结果表 ─────────────────────────────────────────────────────────
    all_results = [m1, s1, s2, s3, s4, s4b, s5]
    summary_rows = []
    for df in all_results:
        label = df["model"].iloc[0]
        pred  = df.iloc[0]["Variable"]
        row   = df.iloc[0]
        summary_rows.append({
            "Model": label,
            "Predictor": pred,
            "N obs": row["N_obs"],
            "N strata": row["N_strata"],
            "OR": row["OR"],
            "CI_lo": row["CI_lo"],
            "CI_hi": row["CI_hi"],
            "P": row["P"],
            "OR_fmt": fmt_or(row["OR"], row["CI_lo"], row["CI_hi"]),
            "P_fmt": fmt_p(row["P"]),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "tbl_euler_logistic.csv", index=False, encoding="utf-8-sig")
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(summary_df[["Model", "Predictor", "N obs", "OR_fmt", "P_fmt"]].to_string(index=False))

    # ── 7. AC1 对照模型（复现，用于并排比较）────────────────────────────────
    print("\n── 加载 AC1 特征，拟合对照模型 ─────────────────────────────────")
    ews_wins = pd.read_parquet(NAAS_ROOT / "data" / "ews_windows.parquet")
    ews_wins["T0"] = pd.to_datetime(ews_wins["T0"])

    def ac1_mean(lo, hi):
        return (
            ews_wins[
                (ews_wins["hours_before_T0"] >= lo) &
                (ews_wins["hours_before_T0"] < hi) &
                (~ews_wins["low_conf_hr"])
            ]
            .groupby(["stay_id", "T0"])["ac1_hr"].mean()
            .reset_index()
            .rename(columns={"ac1_hr": f"ac1_hr_{'early' if lo < -24 else 'late'}_mean"})
        )

    ac1_late  = ac1_mean(LATE_LO,  LATE_HI)
    ac1_early = ac1_mean(EARLY_LO, EARLY_HI)
    mdf_ac1 = mdf.merge(ac1_late,  on=["stay_id", "T0"], how="left")
    mdf_ac1 = mdf_ac1.merge(ac1_early, on=["stay_id", "T0"], how="left")

    ac1_m1  = fit_clogit(mdf_ac1, "ac1_hr_late_mean",  label="AC1 M1 Base model")
    ac1_s1  = fit_clogit(mdf_ac1, "ac1_hr_late_mean",
                         extra_covars=["late_hr_mean_raw"], label="AC1 S1 +late HR mean")
    ac1_s2  = fit_clogit(mdf_ac1, "ac1_hr_late_mean",
                         extra_covars=["late_hr_mean_raw", "late_map_mean_raw"],
                         label="AC1 S2 +late HR+MAP means")
    ac1_s3  = fit_clogit(mdf_ac1, "ac1_hr_early_mean",
                         extra_covars=["early_hr_mean_raw", "early_map_mean_raw"],
                         label="AC1 S3 Early window")
    mdf_no_sed_ac1 = mdf_ac1[mdf_ac1["sedation_before_window"] == 0].copy()
    ac1_s5  = fit_clogit(mdf_no_sed_ac1, "ac1_hr_late_mean",
                         include_sedation=False, label="AC1 S5 No-sedation")

    print_result(ac1_m1, "ac1_hr_late_mean",  "AC1 M1 base")
    print_result(ac1_s1, "ac1_hr_late_mean",  "AC1 S1 +late_hr")
    print_result(ac1_s2, "ac1_hr_late_mean",  "AC1 S2 +late_hr+map")
    print_result(ac1_s3, "ac1_hr_early_mean", "AC1 S3 early")
    print_result(ac1_s5, "ac1_hr_late_mean",  "AC1 S5 no-sed")

    # ── 8. 图表 ───────────────────────────────────────────────────────────────
    print("\n── 出图 ─────────────────────────────────────────────────────────")

    # Forest plot: euler_hr 所有模型
    euler_forest_data = [
        ("M1 Base model",           "euler_hr_late_mean",     m1),
        ("S1 +late HR mean",        "euler_hr_late_mean",     s1),
        ("S2 +late HR+MAP means",   "euler_hr_late_mean",     s2),
        ("S3 Early window",         "euler_hr_early_mean",    s3),
        ("S4 N_extrema",            "n_extrema_hr_late_mean", s4),
        ("S4b N_extrema +HR mean",  "n_extrema_hr_late_mean", s4b),
        ("S5 No-sedation subgroup", "euler_hr_late_mean",     s5),
    ]
    ac1_m1_or   = float(ac1_m1.loc[ac1_m1["Variable"] == "ac1_hr_late_mean", "OR"].iloc[0])
    ac1_m1_or_fmt = f"{ac1_m1_or:.2f}"
    forest_plot(
        euler_forest_data,
        OUTPUT_DIR / "fig_euler_forest.png",
        title=r"HR Euler $\chi(0)$: Conditional Logistic OR (95% CI)",
        ref_or=ac1_m1_or, ref_label=f"AC1 M1 OR={ac1_m1_or_fmt}",
    )

    # 并排比较 forest plot
    euler_compare = [
        ("M1 Base model",           "euler_hr_late_mean",   m1),
        ("S1 +late HR mean",        "euler_hr_late_mean",   s1),
        ("S2 +late HR+MAP means",   "euler_hr_late_mean",   s2),
        ("S3 Early window",         "euler_hr_early_mean",  s3),
        ("S5 No-sedation subgroup", "euler_hr_late_mean",   s5),
    ]
    ac1_compare = [
        ("M1 Base model",           "ac1_hr_late_mean",   ac1_m1),
        ("S1 +late HR mean",        "ac1_hr_late_mean",   ac1_s1),
        ("S2 +late HR+MAP means",   "ac1_hr_late_mean",   ac1_s2),
        ("S3 Early window",         "ac1_hr_early_mean",  ac1_s3),
        ("S5 No-sedation subgroup", "ac1_hr_late_mean",   ac1_s5),
    ]
    forest_comparison(euler_compare, ac1_compare, OUTPUT_DIR / "fig_euler_vs_ac1_forest.png")

    # ── 9. 完整系数表（主模型）────────────────────────────────────────────────
    coef_table = m1.copy()
    coef_table["OR_fmt"] = coef_table.apply(
        lambda r: fmt_or(r["OR"], r["CI_lo"], r["CI_hi"]), axis=1)
    coef_table["P_fmt"] = coef_table["P"].map(fmt_p)
    coef_table[["Variable", "OR_fmt", "P_fmt", "N_obs", "N_strata"]].to_csv(
        OUTPUT_DIR / "tbl_euler_logistic_full.csv", index=False, encoding="utf-8-sig")

    print("\n[完成] 输出目录:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
