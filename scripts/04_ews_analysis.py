"""
Step 4: EWS 计算与统计检验

主指标:
  Variance(MAP residual) -- 12h 滑动窗口方差
  AC1(HR residual)       -- 12h 滑动窗口 lag-1 自相关

本版修复:
  - AC1 仅使用相邻且双端都为真实观测的点对
  - 放弃 Hamed-Rao MK，改为 early/late 直接比较 + LMM(time * group)
  - 增加 early / late 窗口中心（window-center）分析
  - 亚组分析与全队列一起做 Holm 校正

输出:
  data/ews_windows.parquet
  data/ews_patient_stats.parquet
  output/fig1_timeseries.png
  output/fig2_delta_boxplot.png
  output/figS1_subgroup.png
  output/table3_ews_comparison.csv
  output/table3_lmm_summary.csv
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent
COHORT_PATH = PROJECT_ROOT / "data" / "cohort.parquet"
VITALS_PATH = PROJECT_ROOT / "data" / "vitals_cleaned.parquet"
DIAG_PATH = PROJECT_ROOT / "data" / "cleaning_diagnostics.parquet"
WINDOWS_PATH = PROJECT_ROOT / "data" / "ews_windows.parquet"
STATS_PATH = PROJECT_ROOT / "data" / "ews_patient_stats.parquet"
OUTPUT_DIR = PROJECT_ROOT / "output"

WINDOW_SIZE = 12
MIN_ACTUAL_WIN = 8
MIN_AC1_PAIRS = 8
ALPHA = 0.05

EARLY_WINDOW_LO = -48
EARLY_WINDOW_HI = -24
LATE_WINDOW_LO = -12
LATE_WINDOW_HI = 0

COLORS = {"shock": "#d62728", "control": "#1f77b4"}
sns.set_theme(style="whitegrid", font_scale=1.05)


def tagged_path(path: Path, tag: str) -> Path:
    if not tag:
        return path
    return path.with_name(f"{path.stem}{tag}{path.suffix}")


def format_p(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def _ac1(vals: np.ndarray, is_interp: np.ndarray) -> float:
    """Lag-1 AC1，仅保留双端均非插值的相邻实测点对。"""
    x, y = vals[:-1], vals[1:]
    ix, iy = is_interp[:-1], is_interp[1:]
    m = ~(np.isnan(x) | np.isnan(y) | ix | iy)
    if m.sum() < MIN_AC1_PAIRS:
        return np.nan
    xm, ym = x[m], y[m]
    if xm.std() == 0 or ym.std() == 0:
        return np.nan
    return float(np.corrcoef(xm, ym)[0, 1])


def compute_windows(ts: pd.DataFrame, T0: pd.Timestamp) -> pd.DataFrame:
    """对单个 (stay_id, T0) 计算 12h 滑动窗口统计。"""
    ts = ts.sort_values("charttime").reset_index(drop=True)
    if len(ts) < WINDOW_SIZE:
        return pd.DataFrame()

    rows = []
    for i in range(len(ts) - WINDOW_SIZE + 1):
        win = ts.iloc[i : i + WINDOW_SIZE]
        n_act_map = int((~win["map_is_interpolated"] & win["map_residual"].notna()).sum())
        n_act_hr = int((~win["hr_is_interpolated"] & win["hr_residual"].notna()).sum())
        low_conf_map = n_act_map < MIN_ACTUAL_WIN
        low_conf_hr = n_act_hr < MIN_ACTUAL_WIN
        # 以 12h rolling window 的中心点记录时间，而不是窗口终点。
        center_t = win["charttime"].iloc[WINDOW_SIZE // 2]
        h_before = (center_t - T0).total_seconds() / 3600

        rows.append(
            {
                "hours_before_T0": round(h_before, 1),
                "var_map": float(np.nanvar(win["map_residual"].to_numpy(float))),
                "ac1_hr": _ac1(
                    win["hr_residual"].to_numpy(float),
                    win["hr_is_interpolated"].to_numpy(bool),
                ),
                "n_actual_map": n_act_map,
                "n_actual_hr": n_act_hr,
                "low_conf_map": low_conf_map,
                "low_conf_hr": low_conf_hr,
                "low_confidence": low_conf_map or low_conf_hr,
            }
        )
    return pd.DataFrame(rows)


def window_mean(
    windows: pd.DataFrame,
    metric: str,
    lo: int,
    hi: int,
    conf_col: str,
) -> float:
    mask = (
        (windows["hours_before_T0"] >= lo)
        & (windows["hours_before_T0"] < hi)
        & (~windows[conf_col])
    )
    vals = windows.loc[mask, metric].dropna()
    return float(vals.mean()) if len(vals) else np.nan


def summarize_patient_windows(windows: pd.DataFrame) -> dict:
    """为每个 (stay_id, T0) 生成按窗口中心分层的 early/late/delta 摘要。"""
    early_map = window_mean(windows, "var_map", EARLY_WINDOW_LO, EARLY_WINDOW_HI, "low_conf_map")
    late_map = window_mean(windows, "var_map", LATE_WINDOW_LO, LATE_WINDOW_HI, "low_conf_map")
    early_hr = window_mean(windows, "ac1_hr", EARLY_WINDOW_LO, EARLY_WINDOW_HI, "low_conf_hr")
    late_hr = window_mean(windows, "ac1_hr", LATE_WINDOW_LO, LATE_WINDOW_HI, "low_conf_hr")

    return {
        "early_map_mean": early_map,
        "late_map_mean": late_map,
        "delta_map": late_map - early_map if pd.notna(early_map) and pd.notna(late_map) else np.nan,
        "early_hr_mean": early_hr,
        "late_hr_mean": late_hr,
        "delta_hr": late_hr - early_hr if pd.notna(early_hr) and pd.notna(late_hr) else np.nan,
        "n_windows_map": int((~windows["low_conf_map"]).sum()),
        "n_windows_hr": int((~windows["low_conf_hr"]).sum()),
    }


def gee_group_pvalue(data: pd.DataFrame, outcome: str) -> tuple[float, float]:
    """以 stay_id 为聚类单元比较 shock vs control。"""
    tmp = data[["stay_id", "binary_group", outcome]].dropna().copy()
    if tmp.empty or tmp["binary_group"].nunique() < 2 or tmp["stay_id"].nunique() < 2:
        return np.nan, np.nan
    try:
        model = GEE.from_formula(
            f"{outcome} ~ binary_group",
            groups="stay_id",
            data=tmp,
            family=Gaussian(),
            cov_struct=Exchangeable(),
        )
        res = model.fit(maxiter=100)
        return float(res.params["binary_group"]), float(res.pvalues["binary_group"])
    except Exception:
        return np.nan, np.nan


def build_group_comparison_table(stats_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """构建 early/late/delta 的组间比较表。"""
    stats_df = stats_df.copy()
    stats_df["binary_group"] = (stats_df["group"] == "shock").astype(int)
    rows = []

    for indicator, early_col, late_col, delta_col in [
        ("Variance-MAP", "early_map_mean", "late_map_mean", "delta_map"),
        ("AC1-HR", "early_hr_mean", "late_hr_mean", "delta_hr"),
    ]:
        shock = stats_df[stats_df["group"] == "shock"]
        control = stats_df[stats_df["group"] == "control"]
        _, p_early = gee_group_pvalue(stats_df, early_col)
        _, p_late = gee_group_pvalue(stats_df, late_col)
        _, p_delta = gee_group_pvalue(stats_df, delta_col)

        rows.append(
            {
                "Subgroup": label,
                "Indicator": indicator,
                "N_shock": int((stats_df["group"] == "shock").sum()),
                "N_control": int((stats_df["group"] == "control").sum()),
                "Early_shock_mean": float(shock[early_col].dropna().mean()) if shock[early_col].notna().any() else np.nan,
                "Early_control_mean": float(control[early_col].dropna().mean()) if control[early_col].notna().any() else np.nan,
                "Early_p_cluster": p_early,
                "Late_shock_mean": float(shock[late_col].dropna().mean()) if shock[late_col].notna().any() else np.nan,
                "Late_control_mean": float(control[late_col].dropna().mean()) if control[late_col].notna().any() else np.nan,
                "Late_p_cluster": p_late,
                "Delta_shock_mean": float(shock[delta_col].dropna().mean()) if shock[delta_col].notna().any() else np.nan,
                "Delta_control_mean": float(control[delta_col].dropna().mean()) if control[delta_col].notna().any() else np.nan,
                "Delta_p_cluster": p_delta,
            }
        )

    return pd.DataFrame(rows)


def apply_global_holm(tbl: pd.DataFrame) -> pd.DataFrame:
    """对全队列 + 亚组全部 early/late/delta 检验统一做 Holm 校正。"""
    tbl = tbl.copy()
    raw_cols = ["Early_p_cluster", "Late_p_cluster", "Delta_p_cluster"]
    values = []
    slots = []
    for idx, row in tbl.iterrows():
        for col in raw_cols:
            values.append(row[col])
            slots.append((idx, f"{col}_holm"))

    corrected = np.full(len(values), np.nan)
    arr = np.asarray(values, dtype=float)
    valid = ~np.isnan(arr)
    if np.any(valid):
        corrected[valid] = multipletests(arr[valid], method="holm")[1]

    for corr, (idx, col) in zip(corrected, slots):
        tbl.loc[idx, col] = corr

    for col in raw_cols + [f"{c}_holm" for c in raw_cols]:
        tbl[f"{col}_fmt"] = tbl[col].map(format_p)
    return tbl


def fit_lmm_trend(windows_df: pd.DataFrame, metric: str, conf_col: str, label: str) -> dict:
    """LMM: metric ~ time * group + (1 + time | pair_key)."""
    tmp = windows_df.loc[
        ~windows_df[conf_col],
        ["stay_id", "T0", "group", "hours_before_T0", metric],
    ].dropna().copy()
    if len(tmp) < 50:
        return {
            "Subgroup": label,
            "Metric": metric,
            "Interaction_beta": np.nan,
            "Interaction_p": np.nan,
            "Random_effects": "insufficient",
        }

    tmp["binary_group"] = (tmp["group"] == "shock").astype(int)
    tmp["time_centered"] = tmp["hours_before_T0"] + 24.0
    tmp["pair_key"] = (
        tmp["stay_id"].astype(str)
        + "__"
        + pd.to_datetime(tmp["T0"]).dt.strftime("%Y%m%d%H%M%S")
    )

    try:
        model = MixedLM.from_formula(
            f"{metric} ~ time_centered * binary_group",
            groups="pair_key",
            re_formula="~time_centered",
            data=tmp,
        )
        res = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        return {
            "Subgroup": label,
            "Metric": metric,
            "Interaction_beta": float(res.params.get("time_centered:binary_group", np.nan)),
            "Interaction_p": float(res.pvalues.get("time_centered:binary_group", np.nan)),
            "Random_effects": "intercept+slope",
        }
    except Exception:
        try:
            model = MixedLM.from_formula(
                f"{metric} ~ time_centered * binary_group",
                groups="pair_key",
                re_formula="1",
                data=tmp,
            )
            res = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
            return {
                "Subgroup": label,
                "Metric": metric,
                "Interaction_beta": float(res.params.get("time_centered:binary_group", np.nan)),
                "Interaction_p": float(res.pvalues.get("time_centered:binary_group", np.nan)),
                "Random_effects": "intercept_only",
            }
        except Exception:
            return {
                "Subgroup": label,
                "Metric": metric,
                "Interaction_beta": np.nan,
                "Interaction_p": np.nan,
                "Random_effects": "failed",
            }


def export_table3(stats_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """导出 full + subgroup 的 early/late/delta 比较表。"""
    tables = [build_group_comparison_table(stats_df, "Full cohort")]
    for src in ["abp", "nbp"]:
        sub = stats_df[stats_df["dominant_source"] == src].copy()
        if len(sub) >= 10:
            tables.append(build_group_comparison_table(sub, f"{src.upper()}-dominant"))
    tbl = apply_global_holm(pd.concat(tables, ignore_index=True))
    tbl.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[Table 3 CSV → {output_path}]")
    print(tbl.to_string(index=False))
    return tbl


def fig1_timeseries(windows_df: pd.DataFrame, path: Path) -> None:
    """MAP variance + HR AC1 时序均值 ±95% CI，叠加 LOESS 平滑趋势线。"""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    def agg(metric: str, conf_col: str) -> pd.DataFrame:
        w = windows_df[~windows_df[conf_col]]
        return (
            w.groupby(["hours_before_T0", "group"])[metric]
            .agg(mean="mean", sem=lambda x: x.sem())
            .reset_index()
        )

    # Darken a hex colour for the LOESS overlay line
    def _darken(hex_color: str, factor: float = 0.65) -> str:
        import colorsys
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        r2, g2, b2 = colorsys.hls_to_rgb(h, max(0, l * factor), s)
        return "#{:02x}{:02x}{:02x}".format(int(r2 * 255), int(g2 * 255), int(b2 * 255))

    agg_map = agg("var_map", "low_conf_map")
    agg_hr = agg("ac1_hr", "low_conf_hr")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for ax, agg_df, ylabel in [
        (axes[0], agg_map, "MAP variance (residual)"),
        (axes[1], agg_hr, "HR AC1 (residual)"),
    ]:
        for grp in ("shock", "control"):
            sub = agg_df[agg_df["group"] == grp].sort_values("hours_before_T0")
            if sub.empty:
                continue
            x = sub["hours_before_T0"].to_numpy()
            y = sub["mean"].to_numpy()
            ci = 1.96 * sub["sem"].to_numpy()
            # Raw mean ± CI band (thin, translucent)
            ax.plot(x, y, color=COLORS[grp], linewidth=1.2, alpha=0.45, label=f"{grp} (mean)")
            ax.fill_between(x, y - ci, y + ci, color=COLORS[grp], alpha=0.15)
            # LOESS smoothed trend line (thicker, darker)
            smoothed = lowess(y, x, frac=0.3, return_sorted=True)
            ax.plot(
                smoothed[:, 0],
                smoothed[:, 1],
                color=_darken(COLORS[grp]),
                linewidth=2.5,
                label=f"{grp} (LOESS)",
            )
        ax.axvline(0, color="grey", linestyle="--", linewidth=1.2)
        ax.axvline(EARLY_WINDOW_HI, color="green", linestyle=":", linewidth=1.2, alpha=0.9)
        ax.axvline(LATE_WINDOW_LO, color="orange", linestyle=":", linewidth=1.2, alpha=0.9)
        ax.set_ylabel(ylabel, fontsize=10)

    axes[1].set_xlabel("Hours before T0", fontsize=10)
    axes[0].legend(fontsize=9)
    # ensure T0 (x=0) and the reference lines at -24 h / -12 h are fully visible
    for ax in axes:
        ax.set_xlim(right=0)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


def fig2_delta_boxplot(stats_df: pd.DataFrame, path: Path) -> None:
    """Early-to-late change 的箱线图：两个指标使用独立子图和 y 轴。"""
    metrics = [
        ("delta_map", "Variance-MAP", "Late − Early\nMAP variance"),
        ("delta_hr",  "AC1-HR",       "Late − Early\nHR AC1"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    fig.suptitle("Figure 2: Early-to-late indicator change", fontsize=12)

    for ax, (col, title, ylabel) in zip(axes, metrics):
        plot_rows = []
        for grp, sub in stats_df.groupby("group"):
            for v in sub[col].dropna():
                plot_rows.append({"group": grp, "delta": v})
        df = pd.DataFrame(plot_rows)
        if df.empty:
            ax.set_visible(False)
            continue
        sns.boxplot(
            data=df,
            x="group",
            y="delta",
            palette=COLORS,
            order=["control", "shock"],
            width=0.5,
            fliersize=2,
            ax=ax,
        )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Group", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


def figS1_subgroup(windows_df: pd.DataFrame, stats_df: pd.DataFrame, path: Path) -> None:
    """ABP-dominant vs NBP-dominant 亚组时序图。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    fig.suptitle("Figure S1: Sensitivity — ABP-dominant vs NBP-dominant", fontsize=13)

    for col_i, src in enumerate(["abp", "nbp"]):
        sids = stats_df[stats_df["dominant_source"] == src][["stay_id", "T0"]]
        w_src = windows_df.merge(sids, on=["stay_id", "T0"])
        for row_i, (metric, ylabel, conf_col) in enumerate(
            [("var_map", "MAP Variance", "low_conf_map"), ("ac1_hr", "HR AC1", "low_conf_hr")]
        ):
            ax = axes[row_i][col_i]
            w = w_src[~w_src[conf_col]]
            agg = (
                w.groupby(["hours_before_T0", "group"])[metric]
                .agg(mean="mean", sem=lambda x: x.sem())
                .reset_index()
            )
            for grp in ("shock", "control"):
                sub = agg[agg["group"] == grp].sort_values("hours_before_T0")
                if sub.empty:
                    continue
                x = sub["hours_before_T0"]
                y = sub["mean"]
                ci = 1.96 * sub["sem"]
                ax.plot(x, y, color=COLORS[grp], label=grp, linewidth=1.5)
                ax.fill_between(x, y - ci, y + ci, color=COLORS[grp], alpha=0.2)
            ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
            ax.set_title(f"{src.upper()}-dominant", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=9)

    axes[0][0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Hours before T0", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vitals_path = tagged_path(VITALS_PATH, args.suffix)
    diag_path = tagged_path(DIAG_PATH, args.suffix)
    windows_path = tagged_path(WINDOWS_PATH, args.suffix)
    stats_path = tagged_path(STATS_PATH, args.suffix)
    table3_path = tagged_path(OUTPUT_DIR / "table3_ews_comparison.csv", args.suffix)
    lmm_path = tagged_path(OUTPUT_DIR / "table3_lmm_summary.csv", args.suffix)
    fig1_path = tagged_path(OUTPUT_DIR / "fig1_timeseries.png", args.suffix)
    fig2_path = tagged_path(OUTPUT_DIR / "fig2_delta_boxplot.png", args.suffix)
    figS1_path = tagged_path(OUTPUT_DIR / "figS1_subgroup.png", args.suffix)

    cohort = (
        pd.read_parquet(COHORT_PATH)
        .drop_duplicates(["stay_id", "T0"])[["stay_id", "T0", "group"]]
    )
    cohort["T0"] = pd.to_datetime(cohort["T0"])
    diag = (
        pd.read_parquet(diag_path)
        .drop_duplicates(["stay_id", "T0"])[["stay_id", "T0", "dominant_source", "excluded"]]
    )
    diag["T0"] = pd.to_datetime(diag["T0"])
    vitals = pd.read_parquet(vitals_path)
    vitals["charttime"] = pd.to_datetime(vitals["charttime"])
    vitals["T0"] = pd.to_datetime(vitals["T0"])

    passed = diag[~diag["excluded"]][["stay_id", "T0"]]
    vitals = vitals.merge(passed, on=["stay_id", "T0"])
    print(f"分析 (stay_id, T0) 对数: {vitals.groupby(['stay_id', 'T0']).ngroups:,}")

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
        diag[["stay_id", "T0", "dominant_source"]],
        on=["stay_id", "T0"],
        how="left",
    )
    windows_df.to_parquet(windows_path, index=False)
    print(f"  窗口总数: {len(windows_df):,}")
    print(f"  low_conf_map 比例: {windows_df['low_conf_map'].mean():.1%}")
    print(f"  low_conf_hr  比例: {windows_df['low_conf_hr'].mean():.1%}")

    print("\n── 汇总 early / late / delta ───────────────────────────────────")
    patient_stats = []
    for (sid, t0), wins in windows_df.groupby(["stay_id", "T0"]):
        res = summarize_patient_windows(wins)
        res["stay_id"] = sid
        res["T0"] = t0
        patient_stats.append(res)

    stats_df = (
        pd.DataFrame(patient_stats)
        .merge(cohort, on=["stay_id", "T0"])
        .merge(
            diag[["stay_id", "T0", "dominant_source"]],
            on=["stay_id", "T0"],
            how="left",
        )
    )
    stats_df.to_parquet(stats_path, index=False)

    print("\n── LMM(time * group) ────────────────────────────────────────────")
    lmm_rows = [
        fit_lmm_trend(windows_df, "var_map", "low_conf_map", "Full cohort"),
        fit_lmm_trend(windows_df, "ac1_hr", "low_conf_hr", "Full cohort"),
    ]
    for src in ["abp", "nbp"]:
        sub_w = windows_df[windows_df["dominant_source"] == src]
        lmm_rows.append(fit_lmm_trend(sub_w, "var_map", "low_conf_map", f"{src.upper()}-dominant"))
        lmm_rows.append(fit_lmm_trend(sub_w, "ac1_hr", "low_conf_hr", f"{src.upper()}-dominant"))
    lmm_tbl = pd.DataFrame(lmm_rows)
    lmm_tbl["Interaction_p_fmt"] = lmm_tbl["Interaction_p"].map(format_p)
    lmm_tbl.to_csv(lmm_path, index=False, encoding="utf-8-sig")
    print(lmm_tbl.to_string(index=False))

    print("\n── 导出 Table 3 (early / late / delta) ─────────────────────────")
    table3 = export_table3(stats_df, table3_path)

    print("\n── 出图 ─────────────────────────────────────────────────────────")
    fig1_timeseries(windows_df, fig1_path)
    fig2_delta_boxplot(stats_df, fig2_path)
    figS1_subgroup(windows_df, stats_df, figS1_path)

    print("\n── 结果摘要 ─────────────────────────────────────────────────────")
    for _, row in table3.iterrows():
        print(
            f"[{row['Subgroup']} | {row['Indicator']}] "
            f"early Holm={row['Early_p_cluster_holm_fmt']}, "
            f"late Holm={row['Late_p_cluster_holm_fmt']}, "
            f"delta Holm={row['Delta_p_cluster_holm_fmt']}"
        )

    print("\n[完成]")


if __name__ == "__main__":
    main()
