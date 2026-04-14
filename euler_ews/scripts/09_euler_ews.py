"""
探索性分析（不计入投稿稿件）— 方案 C

一维时间序列的拓扑指标：
  euler_map / euler_hr
      MAP/HR 残差序列在窗口内"低于 0 的连通分量数"
      = sublevel-set filtration 的 Euler characteristic χ(0)
      生理直觉：残差围绕 0 振荡，低于 0 的段落个数反映信号的振荡复杂度。
      χ(0)=1 → 整段都在 0 以下（单调下压）；
      χ(0)=0 → 整段都在 0 以上（单调上扬）；
      χ(0)↑  → 频繁穿越 0 轴（高度振荡）。

  n_extrema_map / n_extrema_hr
      局部极值点数量（局部极小 + 局部极大），衡量振荡次数。

  total_var_map / total_var_hr
      总变差 Σ|Δresidual|，量化序列的累积波动幅度。

分析框架与 04_ews_analysis.py 一致：
  早窗 [−48, −24 h)  晚窗 [−12, 0 h)
  GEE（Gaussian, Exchangeable, stay_id 聚类）组间比较
  LMM（time * group）趋势检验
  全局 Holm 校正

输出：output/explore/
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "DejaVu Sans"],
    "axes.unicode_minus": False,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import argrelextrema
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# euler_ews/scripts/ → euler_ews/ → NaaS/
PROJECT_ROOT = Path(__file__).parent.parent          # euler_ews/
NAAS_ROOT    = PROJECT_ROOT.parent                   # NaaS/  (共享数据在此)

COHORT_PATH = NAAS_ROOT / "data" / "cohort.parquet"
VITALS_PATH = NAAS_ROOT / "data" / "vitals_cleaned.parquet"
DIAG_PATH   = NAAS_ROOT / "data" / "cleaning_diagnostics.parquet"
OUTPUT_DIR  = PROJECT_ROOT / "output"

WINDOW_SIZE = 12
MIN_ACTUAL = 6          # 窗口内最少实测点数（方案 C 指标比 AC1 容忍度略低）
ALPHA = 0.05

EARLY_LO, EARLY_HI = -48, -24
LATE_LO,  LATE_HI  = -12,   0

COLORS = {"shock": "#d62728", "control": "#1f77b4"}
METRICS = [
    ("euler_map",     r"MAP Euler $\chi(0)$",   "low_conf_map"),
    ("euler_hr",      r"HR Euler $\chi(0)$",    "low_conf_hr"),
    ("n_extrema_map", r"MAP $N_\mathrm{ext}$",  "low_conf_map"),
    ("n_extrema_hr",  r"HR $N_\mathrm{ext}$",   "low_conf_hr"),
    ("total_var_map", "MAP total variation",     "low_conf_map"),
    ("total_var_hr",  "HR total variation",      "low_conf_hr"),
]

sns.set_theme(style="whitegrid", font_scale=1.05)


# ── 拓扑指标计算 ─────────────────────────────────────────────────────────────

def euler_at_zero(vals: np.ndarray, is_interp: np.ndarray) -> float:
    """
    χ(0) = #{连通分量 of {t : residual(t) ≤ 0}}

    对仅有实测（非插值）点计算。
    """
    actual = vals[~is_interp & ~np.isnan(vals)]
    if len(actual) < MIN_ACTUAL:
        return np.nan
    below = actual <= 0
    if not below.any():
        return 0.0
    transitions = np.diff(below.astype(np.int8))
    n_components = int(below[0]) + int((transitions == 1).sum())
    return float(n_components)


def n_extrema(vals: np.ndarray, is_interp: np.ndarray) -> float:
    """局部极值点个数（局部极小 + 局部极大），order=1。"""
    actual = vals[~is_interp & ~np.isnan(vals)]
    if len(actual) < MIN_ACTUAL:
        return np.nan
    n_min = len(argrelextrema(actual, np.less,    order=1)[0])
    n_max = len(argrelextrema(actual, np.greater, order=1)[0])
    return float(n_min + n_max)


def total_variation(vals: np.ndarray, is_interp: np.ndarray) -> float:
    """
    均步长归一化总变差：Σ|Δresidual| / (n_actual - 1)

    除以实测点步数，避免将"观测更密"与"波动更大"混淆。
    使不同有效点数的窗口具有可比的量纲。
    """
    actual = vals[~is_interp & ~np.isnan(vals)]
    if len(actual) < 2:
        return np.nan
    return float(np.sum(np.abs(np.diff(actual))) / (len(actual) - 1))


# ── 滑动窗口计算 ──────────────────────────────────────────────────────────────

def compute_windows(ts: pd.DataFrame, T0: pd.Timestamp) -> pd.DataFrame:
    ts = ts.sort_values("charttime").reset_index(drop=True)
    if len(ts) < WINDOW_SIZE:
        return pd.DataFrame()

    map_vals  = ts["map_residual"].to_numpy(float)
    hr_vals   = ts["hr_residual"].to_numpy(float)
    map_interp = ts["map_is_interpolated"].to_numpy(bool)
    hr_interp  = ts["hr_is_interpolated"].to_numpy(bool)

    rows = []
    for i in range(len(ts) - WINDOW_SIZE + 1):
        sl_map = slice(i, i + WINDOW_SIZE)
        mv, mi = map_vals[sl_map], map_interp[sl_map]
        hv, hi = hr_vals[sl_map], hr_interp[sl_map]

        n_act_map = int((~mi & ~np.isnan(mv)).sum())
        n_act_hr  = int((~hi & ~np.isnan(hv)).sum())
        low_map = n_act_map < MIN_ACTUAL
        low_hr  = n_act_hr  < MIN_ACTUAL

        center_t  = ts["charttime"].iloc[i + WINDOW_SIZE // 2]
        h_before  = (center_t - T0).total_seconds() / 3600

        rows.append({
            "hours_before_T0": round(h_before, 1),
            "euler_map":       euler_at_zero(mv, mi),
            "euler_hr":        euler_at_zero(hv, hi),
            "n_extrema_map":   n_extrema(mv, mi),
            "n_extrema_hr":    n_extrema(hv, hi),
            "total_var_map":   total_variation(mv, mi),
            "total_var_hr":    total_variation(hv, hi),
            "n_actual_map":    n_act_map,
            "n_actual_hr":     n_act_hr,
            "low_conf_map":    low_map,
            "low_conf_hr":     low_hr,
            "low_confidence":  low_map or low_hr,
        })
    return pd.DataFrame(rows)


# ── 患者级早/晚窗摘要 ─────────────────────────────────────────────────────────

def window_mean(wins: pd.DataFrame, metric: str, lo: int, hi: int, conf_col: str) -> float:
    mask = (
        (wins["hours_before_T0"] >= lo)
        & (wins["hours_before_T0"] < hi)
        & (~wins[conf_col])
    )
    vals = wins.loc[mask, metric].dropna()
    return float(vals.mean()) if len(vals) else np.nan


def summarize_patient(wins: pd.DataFrame) -> dict:
    d = {}
    for metric, _, conf_col in METRICS:
        early = window_mean(wins, metric, EARLY_LO, EARLY_HI, conf_col)
        late  = window_mean(wins, metric, LATE_LO,  LATE_HI,  conf_col)
        d[f"early_{metric}"] = early
        d[f"late_{metric}"]  = late
        d[f"delta_{metric}"] = (late - early) if pd.notna(early) and pd.notna(late) else np.nan
    return d


# ── 统计检验 ──────────────────────────────────────────────────────────────────

def format_p(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def gee_pvalue(data: pd.DataFrame, outcome: str) -> tuple[float, float]:
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


def fit_lmm(windows_df: pd.DataFrame, metric: str, conf_col: str) -> dict:
    tmp = windows_df.loc[
        ~windows_df[conf_col],
        ["stay_id", "T0", "group", "hours_before_T0", metric],
    ].dropna().copy()
    if len(tmp) < 50:
        return {"metric": metric, "beta": np.nan, "p": np.nan, "re": "insufficient"}

    tmp["binary_group"]   = (tmp["group"] == "shock").astype(int)
    tmp["time_centered"]  = tmp["hours_before_T0"] + 24.0
    tmp["pair_key"] = (
        tmp["stay_id"].astype(str)
        + "__"
        + pd.to_datetime(tmp["T0"]).dt.strftime("%Y%m%d%H%M%S")
    )
    for re_formula, re_label in [("~time_centered", "intercept+slope"), ("1", "intercept_only")]:
        try:
            res = MixedLM.from_formula(
                f"{metric} ~ time_centered * binary_group",
                groups="pair_key",
                re_formula=re_formula,
                data=tmp,
            ).fit(reml=False, method="lbfgs", maxiter=200, disp=False)
            beta = float(res.params.get("time_centered:binary_group", np.nan))
            p    = float(res.pvalues.get("time_centered:binary_group", np.nan))
            return {"metric": metric, "beta": beta, "p": p, "re": re_label}
        except Exception:
            continue
    return {"metric": metric, "beta": np.nan, "p": np.nan, "re": "failed"}


def build_comparison_table(stats_df: pd.DataFrame) -> pd.DataFrame:
    stats_df = stats_df.copy()
    stats_df["binary_group"] = (stats_df["group"] == "shock").astype(int)
    rows = []
    shock   = stats_df[stats_df["group"] == "shock"]
    control = stats_df[stats_df["group"] == "control"]

    for metric, label, _ in METRICS:
        ec = f"early_{metric}"
        lc = f"late_{metric}"
        dc = f"delta_{metric}"
        _, p_e = gee_pvalue(stats_df, ec)
        _, p_l = gee_pvalue(stats_df, lc)
        _, p_d = gee_pvalue(stats_df, dc)
        rows.append({
            "Metric": label,
            "N_shock":   int((stats_df["group"] == "shock").sum()),
            "N_control": int((stats_df["group"] == "control").sum()),
            "Early_shock":   shock[ec].dropna().mean(),
            "Early_control": control[ec].dropna().mean(),
            "Early_p_raw":   p_e,
            "Late_shock":    shock[lc].dropna().mean(),
            "Late_control":  control[lc].dropna().mean(),
            "Late_p_raw":    p_l,
            "Delta_shock":   shock[dc].dropna().mean(),
            "Delta_control": control[dc].dropna().mean(),
            "Delta_p_raw":   p_d,
        })

    tbl = pd.DataFrame(rows)
    # Holm 校正
    raw_cols = ["Early_p_raw", "Late_p_raw", "Delta_p_raw"]
    all_p = tbl[raw_cols].values.ravel()
    corr = np.full(len(all_p), np.nan)
    valid = ~np.isnan(all_p)
    if valid.any():
        corr[valid] = multipletests(all_p[valid], method="holm")[1]
    corr = corr.reshape(tbl.shape[0], len(raw_cols))
    for j, col in enumerate(raw_cols):
        holm_col = col.replace("_raw", "_holm")
        tbl[holm_col] = corr[:, j]
        tbl[f"{holm_col}_fmt"] = tbl[holm_col].map(format_p)
        tbl[f"{col}_fmt"] = tbl[col].map(format_p)
    return tbl


# ── 图表 ──────────────────────────────────────────────────────────────────────

YLABEL_MAP = {
    "euler_map":     r"MAP Euler $\chi(0)$",
    "euler_hr":      r"HR Euler $\chi(0)$",
    "n_extrema_map": r"MAP $N_\mathrm{ext}$",
    "n_extrema_hr":  r"HR $N_\mathrm{ext}$",
    "total_var_map": "MAP total variation",
    "total_var_hr":  "HR total variation",
}

GROUP_DISPLAY = {"shock": "Shock", "control": "Control"}


def fig_timeseries(windows_df: pd.DataFrame, path: Path) -> None:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    sns.set_theme(style="white", font_scale=1.0)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    fig.suptitle(r"HR & MAP topological indicators — mean $\pm$ 95% CI",
                 fontsize=13, fontweight="bold", y=1.01)

    plot_pairs = [
        ("euler_map",     "low_conf_map"),
        ("euler_hr",      "low_conf_hr"),
        ("n_extrema_map", "low_conf_map"),
        ("n_extrema_hr",  "low_conf_hr"),
        ("total_var_map", "low_conf_map"),
        ("total_var_hr",  "low_conf_hr"),
    ]

    for ax, (metric, conf_col) in zip(axes.ravel(), plot_pairs):
        ylabel = YLABEL_MAP[metric]
        w = windows_df[~windows_df[conf_col]]
        agg = (
            w.groupby(["hours_before_T0", "group"])[metric]
            .agg(mean="mean", sem=lambda x: x.sem())
            .reset_index()
        )
        for grp in ("shock", "control"):
            sub = agg[agg["group"] == grp].sort_values("hours_before_T0")
            if sub.empty:
                continue
            x  = sub["hours_before_T0"].to_numpy()
            y  = sub["mean"].to_numpy()
            ci = 1.96 * sub["sem"].to_numpy()

            # Smooth the mean and CI bounds with LOWESS to remove hourly jitter
            frac = 0.30
            sm_mean = lowess(y,        x, frac=frac, return_sorted=True)
            sm_lo   = lowess(y - ci,   x, frac=frac, return_sorted=True)
            sm_hi   = lowess(y + ci,   x, frac=frac, return_sorted=True)

            ax.fill_between(sm_lo[:, 0], sm_lo[:, 1], sm_hi[:, 1],
                            color=COLORS[grp], alpha=0.18, linewidth=0)
            ax.plot(sm_mean[:, 0], sm_mean[:, 1],
                    color=COLORS[grp], lw=2.2, label=GROUP_DISPLAY[grp])

        # Reference lines
        ax.axvline(0,        color="#7f8c8d", ls="--", lw=1.1, zorder=1)
        ax.axvline(EARLY_HI, color="#27ae60", ls=":",  lw=1.4, alpha=0.9, zorder=1)
        ax.axvline(LATE_LO,  color="#e67e22", ls=":",  lw=1.4, alpha=0.9, zorder=1)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(right=0)
        ax.spines[["top", "right"]].set_visible(False)

    # Legend only in top-right panel
    axes[0, 1].legend(fontsize=9, frameon=False)
    # x-axis label only on bottom row
    for ax in axes[2]:
        ax.set_xlabel("Hours before $T_0$", fontsize=10)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def fig_delta_boxplot(stats_df: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="white", font_scale=1.0)

    BOX_PALETTE = {"control": "#2980b9", "shock": "#c0392b"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(r"Early-to-late $\Delta$ by group",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, (metric, label, _) in zip(axes.ravel(), METRICS):
        col = f"delta_{metric}"
        rows = [
            {"group": g, "delta": v}
            for g, sub in stats_df.groupby("group")
            for v in sub[col].dropna()
        ]
        df = pd.DataFrame(rows)
        if df.empty:
            ax.set_visible(False)
            continue
        sns.boxplot(
            data=df, x="group", y="delta", hue="group",
            palette=BOX_PALETTE, order=["control", "shock"],
            width=0.48, fliersize=1.5, linewidth=0.9,
            flierprops={"alpha": 0.35},
            legend=False, ax=ax,
        )
        ax.axhline(0, color="#7f8c8d", ls="--", lw=1.0)
        ax.set_title(label, fontsize=10, pad=5)
        ax.set_xlabel("")
        ax.set_ylabel(r"Late $-$ Early", fontsize=9)
        ax.set_xticklabels(["Control", "Shock"], fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort = (
        pd.read_parquet(COHORT_PATH)
        .drop_duplicates(["stay_id", "T0"])[["stay_id", "T0", "group"]]
    )
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
    print(f"分析 (stay_id, T0) 对数: {vitals.groupby(['stay_id', 'T0']).ngroups:,}")

    # ── 滑动窗口 ──────────────────────────────────────────────────────────────
    print("\n── 计算滑动窗口 ─────────────────────────────────────────────────")
    all_windows = []
    pairs = vitals.groupby(["stay_id", "T0"], sort=False)
    n_total = pairs.ngroups
    for i, ((sid, t0), grp) in enumerate(pairs, 1):
        wins = compute_windows(grp, pd.Timestamp(t0))
        if not wins.empty:
            wins.insert(0, "stay_id", sid)
            wins.insert(1, "T0", pd.Timestamp(t0))
            all_windows.append(wins)
        if i % 500 == 0:
            print(f"  {i:,}/{n_total:,}...", flush=True)

    windows_df = pd.concat(all_windows, ignore_index=True)
    windows_df = windows_df.merge(cohort, on=["stay_id", "T0"])
    windows_df = windows_df.merge(
        diag[["stay_id", "T0", "dominant_source"]], on=["stay_id", "T0"], how="left"
    )
    print(f"  窗口总数: {len(windows_df):,}")

    # ── 患者级汇总 ────────────────────────────────────────────────────────────
    print("\n── 汇总 early / late / delta ───────────────────────────────────")
    patient_rows = []
    for (sid, t0), wins in windows_df.groupby(["stay_id", "T0"]):
        r = summarize_patient(wins)
        r["stay_id"] = sid
        r["T0"]      = t0
        patient_rows.append(r)

    stats_df = (
        pd.DataFrame(patient_rows)
        .merge(cohort, on=["stay_id", "T0"])
        .merge(diag[["stay_id", "T0", "dominant_source"]], on=["stay_id", "T0"], how="left")
    )

    # ── GEE 组间比较 + Holm ───────────────────────────────────────────────────
    print("\n── GEE 组间比较（Holm 校正）────────────────────────────────────")
    tbl = build_comparison_table(stats_df)
    tbl_path = OUTPUT_DIR / "tbl_euler_comparison.csv"
    tbl.to_csv(tbl_path, index=False, encoding="utf-8-sig")
    print(tbl[["Metric", "Early_shock", "Early_control", "Early_p_holm_fmt",
               "Late_shock", "Late_control", "Late_p_holm_fmt",
               "Delta_shock", "Delta_control", "Delta_p_holm_fmt"]].to_string(index=False))

    # ── LMM 趋势 ─────────────────────────────────────────────────────────────
    print("\n── LMM(time × group) ────────────────────────────────────────────")
    lmm_rows = [fit_lmm(windows_df, m, c) for m, _, c in METRICS]
    lmm_df   = pd.DataFrame(lmm_rows)
    lmm_df["p_fmt"] = lmm_df["p"].map(format_p)
    lmm_df.to_csv(OUTPUT_DIR / "tbl_euler_lmm.csv", index=False, encoding="utf-8-sig")
    print(lmm_df[["metric", "beta", "p_fmt", "re"]].to_string(index=False))

    # ── 图表 ─────────────────────────────────────────────────────────────────
    print("\n── 出图 ─────────────────────────────────────────────────────────")
    fig_timeseries(windows_df, OUTPUT_DIR / "fig_euler_timeseries.png")
    fig_delta_boxplot(stats_df, OUTPUT_DIR / "fig_euler_delta.png")

    print("\n[完成] 输出目录:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
