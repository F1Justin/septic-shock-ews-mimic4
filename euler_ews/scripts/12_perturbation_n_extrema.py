"""
探索性分析 — 护理翻身后 HR 复杂性响应（n_extrema）

背景：
  05_perturbation_recovery.py 用翻身后 30min MAP 的偏差积分（AUC_recovery）
  检验心血管弹性；此脚本改用 HR n_extrema 作为互补的"振荡复杂性"指标，
  测量翻身后 HR 的局部方向反转次数，反映心率调节的灵活程度。

假设：
  接近脓毒性休克的患者（晚窗 vs 早窗）心率调节能力下降，
  翻身后 4h 内 HR n_extrema 呈现与主分析一致的降低趋势（OR < 1）。

方法：
  数据源：
    - data/perturbation_events.parquet  （已有翻身事件，含 turn_time、T0、group）
    - data/vitals_cleaned.parquet       （小时级 HR，含 hr_residual、hr_is_interpolated）

  每个翻身事件：
    - 取后续 POST_WIN_H 小时（默认 4h）内的非插值 hr_residual
    - 需 >= MIN_HR_PTS 个有效点，否则排除
    - 计算 n_extrema = 局部极小 + 局部极大（order=1）
    - 同时记录 pre_hr_mean（翻身前 2h 均值）和 post_hr_mean（翻身后 2h 均值）
      以计算 HR 变化量 hr_delta = post - pre

  患者对（stay_id, T0）级别聚合：
    - 取各期内该患者所有合格事件的 n_extrema 中位数（Fix 4 策略）
    - 再对 patient-level 中位数做组间比较

  统计检验（Wilcoxon rank-sum，非配对）：
    1. Shock:   early 子集 vs late 子集  （两个独立的患者-期次边际样本，非配对；
                                          不能解读为"同一患者从早到晚的变化"）
    2. Control: early 子集 vs late 子集  （阴性对照，同样为非配对边际比较）
    3. Late: shock vs control            （最直接的组间差异）

  注意：因为患者可能仅在 early 或 late 其中一个期次有合格事件，
  early/late 两个子集的患者组成不完全重叠，故无法做配对检验。
  如需真正的 within-patient 配对分析，需在同一患者同时有
  early 和 late 合格事件后才能对齐，样本量将大幅缩小。

  时段定义（与 05 脚本一致）：
    early: (-24, -12) h before T0
    late:  (-6,   0)  h before T0

限制：
  vitals_cleaned 为小时级，POST_WIN_H=4h 时典型约 4-5 个点，
  n_extrema 取值范围通常 0-3，不如分钟级数据精细。
  本分析标记为 exploratory，不作为主结论支柱。

输出：
  euler_ews/output/fig_perturb_n_extrema.png
  euler_ews/output/tbl_perturb_n_extrema_summary.csv
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.sans-serif": ["Arial Unicode MS", "DejaVu Sans"],
                             "axes.unicode_minus": False})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.signal import argrelextrema

warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent          # euler_ews/
NAAS_ROOT     = PROJECT_ROOT.parent                   # NaaS/
EVENTS_PATH   = NAAS_ROOT / "data" / "perturbation_events.parquet"
VITALS_PATH   = NAAS_ROOT / "data" / "vitals_cleaned.parquet"
OUTPUT_DIR    = PROJECT_ROOT / "output"

# ── 参数 ──────────────────────────────────────────────────────────────────────
POST_WIN_H  = 4     # 翻身后 HR 窗口（小时）
PRE_WIN_H   = 2     # 翻身前基线窗口（小时）
MIN_HR_PTS  = 3     # 最少非插值 HR 点数
EXTREMA_ORD = 1     # argrelextrema order 参数

PERIODS = {
    "early": (-24, -12),
    "late":  (-6,   0),
}
COLORS_PERIOD = {"early": "#2196F3", "late": "#F44336"}
COLORS_GROUP  = {"shock": "#d62728", "control": "#1f77b4"}

sns.set_theme(style="whitegrid", font_scale=1.05)


# ── 核心计算 ──────────────────────────────────────────────────────────────────

def n_extrema_1d(vals: np.ndarray) -> int:
    """计算序列的局部极值（极大 + 极小）个数，order=EXTREMA_ORD。"""
    if len(vals) < 3:
        return 0
    n_min = len(argrelextrema(vals, np.less,    order=EXTREMA_ORD)[0])
    n_max = len(argrelextrema(vals, np.greater, order=EXTREMA_ORD)[0])
    return n_min + n_max


def compute_event_features(
    turn_time: pd.Timestamp,
    stay_vitals: pd.DataFrame,
) -> dict | None:
    """
    对单次翻身事件计算 HR 特征。
    返回 {n_extrema, pre_hr_mean, post_hr_mean, hr_delta, n_pts} 或 None。
    """
    T = pd.Timestamp(turn_time)
    cv = stay_vitals

    # 翻身后窗口（非插值 HR 残差）
    post_win = cv[
        (cv["charttime"] >= T) &
        (cv["charttime"] <= T + pd.Timedelta(hours=POST_WIN_H))
    ]
    actual_post = post_win[~post_win["hr_is_interpolated"] & post_win["hr_residual"].notna()]
    if len(actual_post) < MIN_HR_PTS:
        return None

    vals = actual_post["hr_residual"].values
    nex  = n_extrema_1d(vals)

    # 翻身前均值（raw HR，供 hr_delta 使用）
    pre_win = cv[
        (cv["charttime"] >= T - pd.Timedelta(hours=PRE_WIN_H)) &
        (cv["charttime"] <  T)
    ]
    pre_mean = pre_win["hr_raw"].mean() if len(pre_win) >= 1 else np.nan
    post_mean_2h = cv[
        (cv["charttime"] >= T) &
        (cv["charttime"] <= T + pd.Timedelta(hours=2))
    ]["hr_raw"].mean()

    return {
        "n_extrema":   nex,
        "pre_hr_mean": float(pre_mean),
        "post_hr_mean":float(post_mean_2h),
        "hr_delta":    float(post_mean_2h - pre_mean) if not (np.isnan(pre_mean) or np.isnan(post_mean_2h)) else np.nan,
        "n_pts":       len(actual_post),
    }


# ── 期标签 ───────────────────────────────────────────────────────────────────

def get_period(h: float) -> str | None:
    for name, (lo, hi) in PERIODS.items():
        if lo <= h < hi:
            return name
    return None


# ── 患者对聚合 ────────────────────────────────────────────────────────────────

def patient_median(events: pd.DataFrame, period: str) -> pd.DataFrame:
    """Fix 4: 每个 (stay_id, T0) 对取中位数，消除多事件伪重复。"""
    win = events[events["period"] == period].copy()
    return (
        win.groupby(["stay_id", "T0", "group"])["n_extrema"]
        .median()
        .reset_index()
    )


# ── 统计分析 ──────────────────────────────────────────────────────────────────

def run_statistics(events: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for grp in ["shock", "control"]:
        e = patient_median(events, "early")
        l = patient_median(events, "late")
        eg = e[e["group"] == grp]["n_extrema"]
        lg = l[l["group"] == grp]["n_extrema"]

        p = stats.ranksums(eg, lg).pvalue if len(eg) >= 5 and len(lg) >= 5 else np.nan
        rows.append({
            "Comparison":     f"{grp}: early-cohort vs late-cohort (non-paired marginal)",
            "n_early_pairs":  len(eg),
            "n_late_pairs":   len(lg),
            "early_nex_med":  float(eg.median()) if len(eg) else np.nan,
            "early_nex_IQR":  f"{eg.quantile(.25):.1f}–{eg.quantile(.75):.1f}" if len(eg) else "—",
            "late_nex_med":   float(lg.median()) if len(lg) else np.nan,
            "late_nex_IQR":   f"{lg.quantile(.25):.1f}–{lg.quantile(.75):.1f}" if len(lg) else "—",
            "p_value":        float(p) if not np.isnan(p) else np.nan,
        })

    # late shock vs control
    l   = patient_median(events, "late")
    sl  = l[l["group"] == "shock"]["n_extrema"]
    cl  = l[l["group"] == "control"]["n_extrema"]
    p_sc = stats.ranksums(sl, cl).pvalue if len(sl) >= 5 and len(cl) >= 5 else np.nan
    rows.append({
        "Comparison":     "late: shock vs control",
        "n_early_pairs":  np.nan,
        "n_late_pairs":   len(sl) + len(cl),
        "early_nex_med":  np.nan,
        "early_nex_IQR":  "—",
        "late_nex_med":   np.nan,
        "late_nex_IQR":   "—",
        "p_value":        float(p_sc) if not np.isnan(p_sc) else np.nan,
        "shock_late_med":  float(sl.median()) if len(sl) else np.nan,
        "control_late_med":float(cl.median()) if len(cl) else np.nan,
    })

    return pd.DataFrame(rows)


# ── 可视化 ────────────────────────────────────────────────────────────────────

def plot_results(events: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Exploratory: HR n_extrema after Nursing Turn\n"
        f"(post-turn window = {POST_WIN_H}h, hourly HR, MIMIC-IV)",
        fontsize=11,
    )

    # ── Panel A: n_extrema 分布（boxplot, shock vs control × period）──────────
    ax = axes[0]
    plot_df = []
    for period in ["early", "late"]:
        pm = patient_median(events, period)
        for grp in ["shock", "control"]:
            for v in pm[pm["group"] == grp]["n_extrema"]:
                plot_df.append({"Period": period, "Group": grp, "n_extrema": v})
    pdf = pd.DataFrame(plot_df)
    if not pdf.empty:
        palette = {"shock": COLORS_GROUP["shock"], "control": COLORS_GROUP["control"]}
        sns.boxplot(data=pdf, x="Period", y="n_extrema", hue="Group",
                    palette=palette, width=0.55, fliersize=3, ax=ax)
        ax.set_title("HR n_extrema (patient-level median)", fontsize=10)
        ax.set_ylabel("n_extrema post-turn")
        ax.set_xlabel("Period")
        # p 值注释
        for i, row in summary.iterrows():
            if "early vs late" in row["Comparison"]:
                grp_label = row["Comparison"].split(":")[0]
                p = row["p_value"]
                if not np.isnan(p):
                    x_pos = 0.5 if grp_label == "shock" else 0.5
                    offset = -0.15 if grp_label == "shock" else 0.15
                    ax.text(1.0 + offset, ax.get_ylim()[1] * 0.95,
                            f"p={p:.3f}", ha="center", fontsize=7.5,
                            color=COLORS_GROUP[grp_label])
        row_sc = summary[summary["Comparison"] == "late: shock vs control"].iloc[0]
        p_sc = row_sc["p_value"]
        if not np.isnan(p_sc):
            ax.text(0.98, 0.98, f"late S vs C: p={p_sc:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="grey",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # ── Panel B: aligned HR residual trajectory（以 turn 为零点）─────────────
    ax = axes[1]
    ax.set_title(f"HR residual trajectory (±{POST_WIN_H}h, mean±SE)", fontsize=10)
    ax.axvline(0, color="grey", ls=":", lw=0.8)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Hours after turn")
    ax.set_ylabel("HR residual (bpm)")

    aligned_rows = []
    for row in events.itertuples(index=False):
        if row.period is None:
            continue
        sv = events._vitals_cache.get(row.stay_id, pd.DataFrame()) if hasattr(events, "_vitals_cache") else pd.DataFrame()
        # aligned 数据已在 events 里预存
        aligned_rows.append({"group": row.group, "period": row.period,
                              "t_h": 0, "hr_res": row.n_extrema})   # placeholder

    # 实际用 events 里的 aligned 列（如果 prepare_aligned 有存）
    if "aligned_t_h" in events.columns:
        adf = events.explode(["aligned_t_h", "aligned_hr_res"])
        adf = adf[adf["period"].notna()].dropna(subset=["aligned_t_h","aligned_hr_res"])
        for grp in ["shock", "control"]:
            for period in ["early", "late"]:
                sub = adf[(adf["group"]==grp) & (adf["period"]==period)]
                if sub.empty: continue
                agg = sub.groupby("aligned_t_h")["aligned_hr_res"].agg(
                    mean="mean", sem=lambda x: x.sem()).reset_index()
                x, y, se = agg["aligned_t_h"], agg["mean"], 1.96*agg["sem"]
                color = COLORS_GROUP[grp]
                ls    = "--" if period == "early" else "-"
                ax.plot(x, y, color=color, ls=ls, lw=1.8,
                        label=f"{grp} {period}")
                ax.fill_between(x, y-se, y+se, color=color, alpha=0.15)
        ax.legend(fontsize=7.5, loc="upper right")
    else:
        ax.text(0.5, 0.5, "Trajectory data\nnot available",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)

    # ── Panel C: hr_delta 分布（翻身后 2h HR 均值变化）────────────────────────
    ax = axes[2]
    ax.set_title("HR delta after turn\n(post2h_mean − pre2h_mean)", fontsize=10)
    if "hr_delta" in events.columns:
        plot_d = []
        for period in ["early", "late"]:
            sub = events[events["period"] == period].dropna(subset=["hr_delta"])
            pat = sub.groupby(["stay_id","T0","group"])["hr_delta"].median().reset_index()
            for grp in ["shock","control"]:
                for v in pat[pat["group"]==grp]["hr_delta"]:
                    plot_d.append({"Period":period,"Group":grp,"hr_delta":v})
        ddf = pd.DataFrame(plot_d)
        if not ddf.empty:
            sns.boxplot(data=ddf, x="Period", y="hr_delta", hue="Group",
                        palette=palette, width=0.55, fliersize=3, ax=ax)
            ax.axhline(0, color="grey", ls="--", lw=0.8)
            ax.set_ylabel("HR delta (bpm)")
            ax.set_xlabel("Period")

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("加载数据 …")
    pe  = pd.read_parquet(EVENTS_PATH)
    vit = pd.read_parquet(VITALS_PATH)

    pe["turn_time"]  = pd.to_datetime(pe["turn_time"])
    pe["T0"]         = pd.to_datetime(pe["T0"])
    vit["charttime"] = pd.to_datetime(vit["charttime"])

    print(f"  翻身事件: {len(pe):,}  (shock={( pe.group=='shock').sum()}, "
          f"control={(pe.group=='control').sum()})")
    print(f"  有效 stay_id 交集: {pe['stay_id'].nunique()}")

    # 预构建 vitals 索引（按 stay_id）
    vit_by_stay = {sid: grp.sort_values("charttime").reset_index(drop=True)
                   for sid, grp in vit.groupby("stay_id")}

    # ── 计算每个翻身事件的特征 ────────────────────────────────────────────────
    print(f"\n计算翻身后 {POST_WIN_H}h HR n_extrema …")
    records   = []
    n_excl    = 0

    # 同时收集 aligned HR 轨迹用于 Panel B
    aligned_records = []

    for row in pe.itertuples(index=False):
        T   = row.turn_time
        sid = row.stay_id
        sv  = vit_by_stay.get(sid, pd.DataFrame())
        if sv.empty:
            n_excl += 1
            continue

        feat = compute_event_features(T, sv)
        if feat is None:
            n_excl += 1
            continue

        period = get_period(float(row.hours_before_T0))

        records.append({
            "stay_id":          sid,
            "T0":               row.T0,
            "group":            row.group,
            "turn_time":        T,
            "hours_before_T0":  row.hours_before_T0,
            "period":           period,
            "n_extrema":        feat["n_extrema"],
            "hr_delta":         feat["hr_delta"],
            "pre_hr_mean":      feat["pre_hr_mean"],
            "post_hr_mean":     feat["post_hr_mean"],
            "n_pts":            feat["n_pts"],
            "has_vasopressor":  row.has_vasopressor,
        })

        # 收集 aligned HR 轨迹
        if period is not None:
            post_win = sv[
                (sv["charttime"] >= T) &
                (sv["charttime"] <= T + pd.Timedelta(hours=POST_WIN_H))
            ]
            actual_post = post_win[~post_win["hr_is_interpolated"] & post_win["hr_residual"].notna()]
            for _, vrow in actual_post.iterrows():
                t_h = (vrow["charttime"] - T).total_seconds() / 3600
                aligned_records.append({
                    "stay_id": sid, "T0": row.T0,
                    "group": row.group, "period": period,
                    "t_h": t_h, "hr_res": vrow["hr_residual"],
                })

    events = pd.DataFrame(records)
    aligned = pd.DataFrame(aligned_records)

    print(f"  有效事件: {len(events):,}  排除: {n_excl:,}")
    print(f"  period 分布:")
    print(f"    {events.groupby(['group','period']).size().unstack(fill_value=0).to_string()}")

    # 期内统计
    for period in ["early","late"]:
        pm = patient_median(events[events["period"].notna()], period)
        print(f"\n  [{period}] 患者对数: shock={len(pm[pm.group=='shock'])}, "
              f"control={len(pm[pm.group=='control'])}")
        for grp in ["shock","control"]:
            g = pm[pm.group==grp]["n_extrema"]
            print(f"    {grp}: med={g.median():.2f} IQR=[{g.quantile(.25):.2f}–{g.quantile(.75):.2f}] n={len(g)}")

    # ── 统计检验 ──────────────────────────────────────────────────────────────
    # 注：early/late 两个子集的患者组成不完全重叠（有患者只在其中一个期次有合格事件），
    #     故使用非配对 Wilcoxon rank-sum。"early vs late"比较的是两个边际样本，
    #     不能解释为"同一患者从早期到晚期的变化趋势"。
    print("\n统计检验（Wilcoxon rank-sum，非配对边际比较 — 非 within-patient 配对）:")
    events_with_period = events[events["period"].notna()].copy()
    summary = run_statistics(events_with_period)

    for _, row in summary.iterrows():
        p = row["p_value"]
        p_str = f"p={p:.4f}" if not np.isnan(p) else "p=N/A"
        print(f"  {row['Comparison']:40s}: {p_str}")
        if "early vs late" in row["Comparison"]:
            print(f"    early: {row['early_nex_med']:.2f} [{row['early_nex_IQR']}] n={int(row['n_early_pairs'])}")
            print(f"    late:  {row['late_nex_med']:.2f} [{row['late_nex_IQR']}] n={int(row['n_late_pairs'])}")
        else:
            if "shock_late_med" in row and not np.isnan(row.get("shock_late_med", np.nan)):
                print(f"    shock={row['shock_late_med']:.2f}  control={row['control_late_med']:.2f}")

    # 无升压药亚组
    print("\n  无升压药亚组:")
    no_vaso = events_with_period[~events_with_period["has_vasopressor"]].copy()
    if len(no_vaso) >= 30:
        sum_nv = run_statistics(no_vaso)
        for _, row in sum_nv.iterrows():
            p = row["p_value"]
            p_str = f"p={p:.4f}" if not np.isnan(p) else "p=N/A"
            print(f"    {row['Comparison']:40s}: {p_str}")
    else:
        print(f"    样本不足 (n={len(no_vaso)})")

    # ── 输出 ──────────────────────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "tbl_perturb_n_extrema_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n  → {summary_path.name}")

    events_path = OUTPUT_DIR / "perturb_n_extrema_events.parquet"
    events.to_parquet(events_path, index=False)
    print(f"  → {events_path.name}")

    # ── 图 ─────────────────────────────────────────────────────────────────────
    print("\n绘图 …")
    _make_figure(events_with_period, aligned, summary,
                 OUTPUT_DIR / "fig_perturb_n_extrema.png")

    print("\n[完成]")


def _make_figure(events: pd.DataFrame, aligned: pd.DataFrame,
                 summary: pd.DataFrame, path: Path) -> None:
    palette = {"shock": COLORS_GROUP["shock"], "control": COLORS_GROUP["control"]}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Exploratory: HR Complexity (n_extrema) after Nursing Turn — MIMIC-IV\n"
        f"Post-turn window = {POST_WIN_H}h  |  Hourly HR residual  |  "
        f"Patient-level median per period",
        fontsize=10,
    )

    # ── A: n_extrema boxplot ─────────────────────────────────────────────────
    ax = axes[0]
    plot_rows = []
    for period in ["early", "late"]:
        pm = patient_median(events, period)
        for grp in ["shock", "control"]:
            for v in pm[pm.group == grp]["n_extrema"]:
                plot_rows.append({"Period": period.capitalize(), "Group": grp, "n_extrema": float(v)})
    pdf = pd.DataFrame(plot_rows)
    if not pdf.empty:
        sns.boxplot(data=pdf, x="Period", y="n_extrema", hue="Group",
                    palette=palette, width=0.55, fliersize=3, linewidth=1.2, ax=ax)
        # 统计注释
        p_map = {}
        for _, row in summary.iterrows():
            p_map[row["Comparison"]] = row["p_value"]

        y_top = pdf["n_extrema"].max() + 0.3
        for i_period, period_cap in enumerate(["Early", "Late"]):
            period = period_cap.lower()
            shock_key   = f"shock: early vs late"
            ctrl_key    = f"control: early vs late"
            if i_period == 0:  # early
                pass  # 早窗是参照，不单独标注
            else:  # late
                p_sh = p_map.get(shock_key, np.nan)
                p_ct = p_map.get(ctrl_key, np.nan)
                sc_key = "late: shock vs control"
                p_sc = p_map.get(sc_key, np.nan)
                for p_v, y_off, color in [(p_sh, y_top, COLORS_GROUP["shock"]),
                                           (p_ct, y_top+0.3, COLORS_GROUP["control"])]:
                    if not np.isnan(p_v):
                        ax.text(1.5, y_off, f"early→late p={p_v:.3f}",
                                ha="center", fontsize=7, color=color)
                if not np.isnan(p_sc):
                    ax.text(1.5, y_top + 0.6, f"S vs C p={p_sc:.3f}",
                            ha="center", fontsize=7.5, color="black",
                            fontweight="bold")

        ax.set_title("n_extrema (patient median)", fontsize=10)
        ax.set_xlabel("Period"); ax.set_ylabel("n_extrema (post-turn HR)")
        ax.legend(title="", fontsize=8)

    # ── B: aligned HR residual 轨迹 ──────────────────────────────────────────
    ax = axes[1]
    ax.set_title(f"Aligned HR residual (mean ± SE)", fontsize=10)
    ax.axvline(0, color="grey", ls=":", lw=0.8, label="turn")
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    if not aligned.empty:
        for grp in ["shock", "control"]:
            for period in ["early", "late"]:
                sub = aligned[(aligned.group == grp) & (aligned.period == period)]
                if sub.empty: continue
                sub = sub.copy()
                sub["t_bin"] = (sub["t_h"] * 2).round() / 2   # 30-min bins
                agg = sub.groupby("t_bin")["hr_res"].agg(
                    mean="mean", sem=lambda x: x.sem() if len(x) > 1 else 0.0
                ).reset_index()
                x, y, se = agg["t_bin"], agg["mean"], 1.96 * agg["sem"]
                ls    = "--" if period == "early" else "-"
                color = COLORS_GROUP[grp]
                ax.plot(x, y, color=color, ls=ls, lw=1.8,
                        label=f"{grp} {period}")
                ax.fill_between(x, y - se, y + se, color=color, alpha=0.15)
        ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax.set_xlabel("Hours after turn"); ax.set_ylabel("HR residual (bpm)")

    # ── C: hr_delta 箱线图 ───────────────────────────────────────────────────
    ax = axes[2]
    ax.set_title("HR delta (post2h − pre2h, bpm)", fontsize=10)
    if "hr_delta" in events.columns:
        delta_rows = []
        for period in ["early", "late"]:
            sub = events[events.period == period].dropna(subset=["hr_delta"])
            pat = sub.groupby(["stay_id","T0","group"])["hr_delta"].median().reset_index()
            for grp in ["shock","control"]:
                for v in pat[pat.group==grp]["hr_delta"]:
                    delta_rows.append({"Period": period.capitalize(), "Group": grp, "hr_delta": float(v)})
        ddf = pd.DataFrame(delta_rows)
        if not ddf.empty:
            sns.boxplot(data=ddf, x="Period", y="hr_delta", hue="Group",
                        palette=palette, width=0.55, fliersize=3, linewidth=1.2, ax=ax)
            ax.axhline(0, color="grey", ls="--", lw=0.8)
            ax.set_xlabel("Period"); ax.set_ylabel("HR change (bpm)")
            ax.legend(title="", fontsize=8)

            # Wilcoxon 检验 hr_delta
            for period, x_pos in [("early", 0), ("late", 1)]:
                pp = ddf[ddf.Period==period.capitalize()]
                s = pp[pp.Group=="shock"]["hr_delta"]
                c = pp[pp.Group=="control"]["hr_delta"]
                if len(s) >= 5 and len(c) >= 5:
                    _, pv = stats.ranksums(s, c)
                    ax.text(x_pos, ax.get_ylim()[1]*0.95,
                            f"p={pv:.3f}", ha="center", fontsize=7.5)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


if __name__ == "__main__":
    main()
