"""
探索性分析（不计入投稿稿件）— 方案 A

MAP-HR 联合相图的持续同调（Persistent Homology）分析

原理：
  将每个 12h 窗口内的 (MAP_residual_t, HR_residual_t) 视为二维点云，
  计算 Vietoris-Rips 持续同调，提取 H₁（一维环）的拓扑特征：

  h1_max_pers:   H₁ 最大持续度（最显著的"环"的生命期 = death - birth）
                 → 高值 = MAP-HR 相图存在稳定的圆形/椭圆轨迹（协调振荡）
                 → 低值 = 轨迹退化为点团或直线（动态僵化）
  h1_total_pers: 所有 H₁ 生成子持续度之和（总"圆形性"）
  h1_count:      H₁ 持续度 > 噪声阈值（NOISE_FRAC × H₀_max_persistence）的环数量
  h0_max_pers:   H₀ 最大持续度（最后一对组分合并前的距离 = 点云直径代理）
                 → 高值 = 点云散而碎（相图碎片化）

预处理：
  每窗口对 MAP / HR 残差各自 z-score（窗口内标准化），
  使两轴尺度可比后计算欧氏距离矩阵。
  窗口内需 ≥ MIN_PH_PTS 个双端实测（非插值）时间步。

分析框架与 04 / 09 脚本一致：
  早窗 [−48, −24 h)  晚窗 [−12, 0 h)
  GEE（Gaussian, Exchangeable, stay_id 聚类）
  LMM（time × group）趋势检验
  全局 Holm 校正

输出：output/explore/
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ripser import ripser as compute_ph
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

COHORT_PATH  = NAAS_ROOT / "data" / "cohort.parquet"
VITALS_PATH  = NAAS_ROOT / "data" / "vitals_cleaned.parquet"
DIAG_PATH    = NAAS_ROOT / "data" / "cleaning_diagnostics.parquet"
OUTPUT_DIR   = PROJECT_ROOT / "output"

WINDOW_SIZE = 12
MIN_PH_PTS  = 6     # 每窗口最少双端实测点数（需 ≥ 3 才能形成 H₁ loop）
NOISE_FRAC  = 0.10  # H₁ 过滤噪声：持续度 < noise_frac × H₀_max 视为噪声
ALPHA = 0.05

EARLY_LO, EARLY_HI = -48, -24
LATE_LO,  LATE_HI  = -12,   0

COLORS = {"shock": "#d62728", "control": "#1f77b4"}
METRICS = [
    ("h1_max_pers",   "H₁ max persistence",   "low_conf_ph"),
    ("h1_total_pers", "H₁ total persistence",  "low_conf_ph"),
    ("h1_count",      "H₁ count (> noise)",    "low_conf_ph"),
    ("h0_max_pers",   "H₀ max persistence",    "low_conf_ph"),
]

sns.set_theme(style="whitegrid", font_scale=1.05)


# ── 持续同调特征提取 ──────────────────────────────────────────────────────────

def ph_features(
    map_vals: np.ndarray,
    hr_vals:  np.ndarray,
    map_interp: np.ndarray,
    hr_interp:  np.ndarray,
) -> dict:
    """
    从窗口内的双端实测点对提取 Vietoris-Rips H₀/H₁ 特征。

    返回字典含：h1_max_pers, h1_total_pers, h1_count, h0_max_pers, low_conf_ph。
    """
    # 仅保留双端均非插值、均非 NaN 的时间步
    valid = (
        ~map_interp & ~hr_interp
        & ~np.isnan(map_vals) & ~np.isnan(hr_vals)
    )
    mv = map_vals[valid]
    hv = hr_vals[valid]

    n = len(mv)
    if n < MIN_PH_PTS:
        return {
            "h1_max_pers":   np.nan,
            "h1_total_pers": np.nan,
            "h1_count":      np.nan,
            "h0_max_pers":   np.nan,
            "low_conf_ph":   True,
        }

    # 窗口内 z-score（使 MAP/HR 尺度可比）
    for arr in [mv, hv]:
        std = arr.std()
        if std < 1e-9:
            # 某轴近乎常数 → 点云退化为 1D，H₁ 不可计算，h0_max_pers 也失去意义。
            # 标记 low_conf_ph=True 使该窗口被所有后续分析均匀排除，
            # 避免 NaN 被静默丢弃而系统性偏高 h0_max_pers 汇总值。
            return {
                "h1_max_pers":   np.nan,
                "h1_total_pers": np.nan,
                "h1_count":      np.nan,
                "h0_max_pers":   np.nan,
                "low_conf_ph":   True,
            }

    mv_z = (mv - mv.mean()) / mv.std()
    hv_z = (hv - hv.mean()) / hv.std()
    pts   = np.column_stack([mv_z, hv_z])   # (n, 2)

    result = compute_ph(pts, maxdim=1)
    dgm0   = result["dgms"][0]  # H₀: (n_components, 2)
    dgm1   = result["dgms"][1]  # H₁: (n_loops, 2)

    # H₀ 最大持续度（排除 death=∞ 的那个全局分量）
    finite0 = dgm0[np.isfinite(dgm0[:, 1])]
    h0_max  = float(np.max(finite0[:, 1] - finite0[:, 0])) if len(finite0) else np.nan

    # 噪声阈值：H₀ 最大持续度 × NOISE_FRAC
    noise_thresh = h0_max * NOISE_FRAC if not np.isnan(h0_max) else 0.0

    if len(dgm1) == 0:
        return {
            "h1_max_pers":   0.0,
            "h1_total_pers": 0.0,
            "h1_count":      0.0,
            "h0_max_pers":   h0_max,
            "low_conf_ph":   False,
        }

    # 过滤 H₁ 中 death=∞ 的条（Vietoris-Rips 在 R² 中 H₁ 无无穷条，但保险起见）
    finite1  = dgm1[np.isfinite(dgm1[:, 1])]
    if len(finite1) == 0:
        return {
            "h1_max_pers":   0.0,
            "h1_total_pers": 0.0,
            "h1_count":      0.0,
            "h0_max_pers":   h0_max,
            "low_conf_ph":   False,
        }

    pers1 = finite1[:, 1] - finite1[:, 0]   # 所有 H₁ 持续度
    sig1  = pers1[pers1 > noise_thresh]      # 过滤噪声后的显著环

    return {
        "h1_max_pers":   float(pers1.max()),
        "h1_total_pers": float(pers1.sum()),
        "h1_count":      float(len(sig1)),
        "h0_max_pers":   h0_max,
        "low_conf_ph":   False,
    }


# ── 滑动窗口 ──────────────────────────────────────────────────────────────────

def compute_windows(ts: pd.DataFrame, T0: pd.Timestamp) -> pd.DataFrame:
    ts = ts.sort_values("charttime").reset_index(drop=True)
    if len(ts) < WINDOW_SIZE:
        return pd.DataFrame()

    map_vals   = ts["map_residual"].to_numpy(float)
    hr_vals    = ts["hr_residual"].to_numpy(float)
    map_interp = ts["map_is_interpolated"].to_numpy(bool)
    hr_interp  = ts["hr_is_interpolated"].to_numpy(bool)

    rows = []
    for i in range(len(ts) - WINDOW_SIZE + 1):
        sl = slice(i, i + WINDOW_SIZE)
        feats = ph_features(
            map_vals[sl], hr_vals[sl],
            map_interp[sl], hr_interp[sl],
        )
        center_t = ts["charttime"].iloc[i + WINDOW_SIZE // 2]
        h_before = (center_t - T0).total_seconds() / 3600
        feats["hours_before_T0"] = round(h_before, 1)
        rows.append(feats)

    return pd.DataFrame(rows)


# ── 患者级早/晚窗摘要 ─────────────────────────────────────────────────────────

def window_mean(wins: pd.DataFrame, metric: str, lo: int, hi: int) -> float:
    mask = (
        (wins["hours_before_T0"] >= lo)
        & (wins["hours_before_T0"] < hi)
        & (~wins["low_conf_ph"])
    )
    vals = wins.loc[mask, metric].dropna()
    return float(vals.mean()) if len(vals) else np.nan


def summarize_patient(wins: pd.DataFrame) -> dict:
    d = {}
    for metric, _, _ in METRICS:
        early = window_mean(wins, metric, EARLY_LO, EARLY_HI)
        late  = window_mean(wins, metric, LATE_LO,  LATE_HI)
        d[f"early_{metric}"] = early
        d[f"late_{metric}"]  = late
        d[f"delta_{metric}"] = (late - early) if pd.notna(early) and pd.notna(late) else np.nan
    return d


# ── 统计 ──────────────────────────────────────────────────────────────────────

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


def fit_lmm(windows_df: pd.DataFrame, metric: str) -> dict:
    tmp = windows_df.loc[
        ~windows_df["low_conf_ph"],
        ["stay_id", "T0", "group", "hours_before_T0", metric],
    ].dropna().copy()
    if len(tmp) < 50:
        return {"metric": metric, "beta": np.nan, "p": np.nan, "re": "insufficient"}

    tmp["binary_group"]  = (tmp["group"] == "shock").astype(int)
    tmp["time_centered"] = tmp["hours_before_T0"] + 24.0
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
    shock   = stats_df[stats_df["group"] == "shock"]
    control = stats_df[stats_df["group"] == "control"]
    rows = []
    for metric, label, _ in METRICS:
        ec, lc, dc = f"early_{metric}", f"late_{metric}", f"delta_{metric}"
        _, p_e = gee_pvalue(stats_df, ec)
        _, p_l = gee_pvalue(stats_df, lc)
        _, p_d = gee_pvalue(stats_df, dc)
        rows.append({
            "Metric": label,
            "N_shock":        int((stats_df["group"] == "shock").sum()),
            "N_control":      int((stats_df["group"] == "control").sum()),
            "Early_shock":    shock[ec].dropna().mean(),
            "Early_control":  control[ec].dropna().mean(),
            "Early_p_raw":    p_e,
            "Late_shock":     shock[lc].dropna().mean(),
            "Late_control":   control[lc].dropna().mean(),
            "Late_p_raw":     p_l,
            "Delta_shock":    shock[dc].dropna().mean(),
            "Delta_control":  control[dc].dropna().mean(),
            "Delta_p_raw":    p_d,
        })
    tbl = pd.DataFrame(rows)
    raw_cols = ["Early_p_raw", "Late_p_raw", "Delta_p_raw"]
    all_p  = tbl[raw_cols].values.ravel()
    corr   = np.full(len(all_p), np.nan)
    valid  = ~np.isnan(all_p)
    if valid.any():
        corr[valid] = multipletests(all_p[valid], method="holm")[1]
    corr = corr.reshape(tbl.shape[0], len(raw_cols))
    for j, col in enumerate(raw_cols):
        hc = col.replace("_raw", "_holm")
        tbl[hc] = corr[:, j]
        tbl[f"{hc}_fmt"]   = tbl[hc].map(format_p)
        tbl[f"{col}_fmt"]  = tbl[col].map(format_p)
    return tbl


# ── 图表 ──────────────────────────────────────────────────────────────────────

def fig_timeseries(windows_df: pd.DataFrame, path: Path) -> None:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Topological EWS (Plan A): MAP-HR phase portrait H₁/H₀ — mean ± 95% CI",
                 fontsize=11)

    for ax, (metric, label, _) in zip(axes.ravel(), METRICS):
        w = windows_df[~windows_df["low_conf_ph"]]
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
            ax.plot(x, y, color=COLORS[grp], lw=1.2, alpha=0.45, label=grp)
            ax.fill_between(x, y - ci, y + ci, color=COLORS[grp], alpha=0.15)
            sm = lowess(y, x, frac=0.3, return_sorted=True)
            ax.plot(sm[:, 0], sm[:, 1], color=COLORS[grp], lw=2.2)
        ax.axvline(0,        color="grey",   ls="--", lw=1.0)
        ax.axvline(EARLY_HI, color="green",  ls=":",  lw=1.0, alpha=0.85)
        ax.axvline(LATE_LO,  color="orange", ls=":",  lw=1.0, alpha=0.85)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xlim(right=0)

    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Hours before T0", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def fig_delta_boxplot(stats_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    fig.suptitle("Topological EWS (Plan A): Early-to-late Δ by group", fontsize=11)

    for ax, (metric, label, _) in zip(axes, METRICS):
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
            palette=COLORS, order=["control", "shock"],
            width=0.5, fliersize=2, legend=False, ax=ax,
        )
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("Late − Early", fontsize=9)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def fig_persistence_scatter(stats_df: pd.DataFrame, path: Path) -> None:
    """晚窗 H₁ max vs H₀ max 散点图，按组着色。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for grp in ("control", "shock"):
        sub = stats_df[stats_df["group"] == grp]
        ax.scatter(
            sub["late_h0_max_pers"], sub["late_h1_max_pers"],
            c=COLORS[grp], alpha=0.35, s=14, label=grp, edgecolors="none",
        )
    ax.set_xlabel("Late-window H₀ max persistence (point cloud diameter proxy)", fontsize=9)
    ax.set_ylabel("Late-window H₁ max persistence (loop prominence)", fontsize=9)
    ax.set_title("Phase portrait topology — late window", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
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
    print("\n── 计算持续同调窗口 ────────────────────────────────────────────")
    all_windows = []
    pairs   = vitals.groupby(["stay_id", "T0"], sort=False)
    n_total = pairs.ngroups
    for i, ((sid, t0), grp) in enumerate(pairs, 1):
        wins = compute_windows(grp, pd.Timestamp(t0))
        if not wins.empty:
            wins.insert(0, "stay_id", sid)
            wins.insert(1, "T0", pd.Timestamp(t0))
            all_windows.append(wins)
        if i % 200 == 0:
            print(f"  {i:,}/{n_total:,}...", flush=True)

    windows_df = pd.concat(all_windows, ignore_index=True)
    windows_df = windows_df.merge(cohort, on=["stay_id", "T0"])
    windows_df = windows_df.merge(
        diag[["stay_id", "T0", "dominant_source"]], on=["stay_id", "T0"], how="left"
    )
    print(f"  窗口总数: {len(windows_df):,}")
    print(f"  low_conf_ph 比例: {windows_df['low_conf_ph'].mean():.1%}")
    h1_nonzero = (windows_df.loc[~windows_df["low_conf_ph"], "h1_max_pers"] > 0).mean()
    print(f"  H₁ 环存在比例（非噪声窗口中）: {h1_nonzero:.1%}")

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

    # ── GEE + Holm ────────────────────────────────────────────────────────────
    print("\n── GEE 组间比较（Holm 校正）────────────────────────────────────")
    tbl = build_comparison_table(stats_df)
    tbl.to_csv(OUTPUT_DIR / "tbl_ph_comparison.csv", index=False, encoding="utf-8-sig")
    print(tbl[[
        "Metric",
        "Early_shock", "Early_control", "Early_p_holm_fmt",
        "Late_shock",  "Late_control",  "Late_p_holm_fmt",
        "Delta_shock", "Delta_control", "Delta_p_holm_fmt",
    ]].to_string(index=False))

    # ── LMM ──────────────────────────────────────────────────────────────────
    print("\n── LMM(time × group) ────────────────────────────────────────────")
    lmm_rows = [fit_lmm(windows_df, m) for m, _, _ in METRICS]
    lmm_df   = pd.DataFrame(lmm_rows)
    lmm_df["p_fmt"] = lmm_df["p"].map(format_p)
    lmm_df.to_csv(OUTPUT_DIR / "tbl_ph_lmm.csv", index=False, encoding="utf-8-sig")
    print(lmm_df[["metric", "beta", "p_fmt", "re"]].to_string(index=False))

    # ── 图表 ─────────────────────────────────────────────────────────────────
    print("\n── 出图 ─────────────────────────────────────────────────────────")
    fig_timeseries(windows_df, OUTPUT_DIR / "fig_ph_timeseries.png")
    fig_delta_boxplot(stats_df, OUTPUT_DIR / "fig_ph_delta.png")
    fig_persistence_scatter(stats_df, OUTPUT_DIR / "fig_ph_scatter.png")

    print("\n[完成] 输出目录:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
