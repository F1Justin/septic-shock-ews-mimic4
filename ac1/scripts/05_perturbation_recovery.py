"""
Step 5: 护理扰动恢复分析 (探索性)

明确标注为 exploratory analysis，不进入主结论。

方法:
  事件: Turn (chartevents itemid 224082)
  指标: AUC_recovery = integral of |MAP(t) - baseline| for t in [Turn, Turn+30min]
  基线: Turn 前 15min 内 MAP 均值
  排除: 事件前后 30min 内升压药剂量变化 OR 总数据点 < 3 OR 恢复点 < 2

升压药混杂控制:
  策略A: norepinephrine_equivalent_dose 作为 LMM 协变量 (未实现，需 R/statsmodels)
  策略B: 无升压药亚组单独报告

设计变更 (Fix 1, Fix 4):
  - 保留全部 (stay_id, T0) 配对 (Fix 1): 不对 stay_id 单独去重
  - 统计检验在患者层面进行 (Fix 4): 先对每个 (stay_id, T0) 对聚合中位数
    AUC，再执行 Wilcoxon 检验，避免同一患者多事件造成伪重复

输出:
  data/perturbation_events.parquet   -- 每个有效 Turn 事件的 AUC_recovery
  output/tableS7_perturbation_summary.csv -- 患者对级别统计汇总表
  output/fig3_recovery.png           -- 扰动恢复叠加图

用法: python scripts/05_perturbation_recovery.py
预计运行时间: 15-30 分钟 (chartevents 多次扫描)
"""

import warnings
import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
NAAS_ROOT    = PROJECT_ROOT.parent               # NaaS/  (共享数据在此)
DB_PATH      = NAAS_ROOT / "mimiciv" / "mimiciv.db"
COHORT_PATH  = NAAS_ROOT / "data" / "cohort.parquet"
EVENTS_PATH  = NAAS_ROOT / "data" / "perturbation_events.parquet"
OUTPUT_DIR   = PROJECT_ROOT / "output"
SUMMARY_PATH = OUTPUT_DIR / "tableS7_perturbation_summary.csv"

TURN_ITEMID      = 224082
MAP_ITEMIDS      = (220052, 220181)     # ABPm, NBPm
VASO_ITEMIDS     = (221906, 221289, 221662, 221749, 222315)
BASELINE_MIN     = 15    # Turn 前基线窗口 (分钟)
RECOVERY_MIN     = 30    # Turn 后恢复观察窗口 (分钟)
EXCL_WINDOW_MIN  = 30    # 升压药变化排除窗口 (分钟)
MIN_PTS          = 3     # 事件前后最少数据点数

# 分析时段定义 (相对于 T0 的小时数)
PERIODS = {
    "early": (-24, -12),
    "late":  (-6,   0),
}

COLORS_PERIOD = {"early": "#2196F3", "late": "#F44336"}
COLORS_GROUP  = {"shock": "#d62728", "control": "#1f77b4"}
matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          10,
    "axes.labelsize":     10,
    "axes.titlesize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.7,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "grid.color":         "#D8D8D8",
    "grid.linewidth":     0.8,
    "grid.alpha":         1.0,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "savefig.facecolor":  "white",
})
sns.set_theme(style="whitegrid", font_scale=1.0)


# ── 数据提取 ───────────────────────────────────────────────────────────────────

def extract_turn_events(con: duckdb.DuckDBPyConnection,
                        cohort: pd.DataFrame) -> pd.DataFrame:
    """
    提取队列患者 T0 前 48h 内所有 Turn 事件。
    cohort 含所有 (stay_id, T0) 对，同一 stay_id 的多 T0 行均会参与 JOIN，
    产生的 turn 事件行已带有对应 T0 (Fix 1)。
    """
    cohort_t0 = cohort[["stay_id", "T0", "group"]].copy()
    cohort_t0["T0"] = pd.to_datetime(cohort_t0["T0"])
    con.register("cohort_t0", cohort_t0)

    return con.execute(f"""
        SELECT
            ce.stay_id,
            ce.charttime AS turn_time,
            c.group,
            c.T0,
            DATEDIFF('minute', ce.charttime, c.T0) AS min_before_T0
        FROM mimiciv_icu.chartevents ce
        INNER JOIN cohort_t0 c ON ce.stay_id = c.stay_id
        WHERE ce.itemid = {TURN_ITEMID}
          AND ce.charttime BETWEEN c.T0 - INTERVAL '48' HOUR AND c.T0
    """).df()


def extract_map_around_turns(con: duckdb.DuckDBPyConnection,
                              turns: pd.DataFrame) -> pd.DataFrame:
    """
    提取每个 Turn 前后的 MAP 值。
    一次性拉取所有相关 stay 在最宽时间范围内的 MAP 记录。
    """
    stay_windows = turns.groupby("stay_id").agg(
        t_min=("turn_time", lambda x: x.min() - pd.Timedelta(minutes=BASELINE_MIN)),
        t_max=("turn_time", lambda x: x.max() + pd.Timedelta(minutes=RECOVERY_MIN)),
    ).reset_index()
    con.register("stay_windows", stay_windows)

    itemids = ",".join(map(str, MAP_ITEMIDS))
    return con.execute(f"""
        SELECT
            ce.stay_id,
            ce.charttime,
            median(ce.valuenum) AS map_value
        FROM mimiciv_icu.chartevents ce
        INNER JOIN stay_windows sw ON ce.stay_id = sw.stay_id
        WHERE ce.itemid IN ({itemids})
          AND ce.valuenum IS NOT NULL
          AND ce.charttime BETWEEN sw.t_min AND sw.t_max
        GROUP BY ce.stay_id, ce.charttime
        ORDER BY ce.stay_id, ce.charttime
    """).df()


def extract_vaso_events(con: duckdb.DuckDBPyConnection,
                        turns: pd.DataFrame) -> pd.DataFrame:
    """提取升压药 inputevents，用于判断 Turn 窗口内是否有剂量变化。"""
    stay_windows = turns.groupby("stay_id").agg(
        t_min=("turn_time", lambda x: x.min() - pd.Timedelta(minutes=EXCL_WINDOW_MIN)),
        t_max=("turn_time", lambda x: x.max() + pd.Timedelta(minutes=EXCL_WINDOW_MIN)),
    ).reset_index()
    con.register("vaso_windows", stay_windows)

    itemids = ",".join(map(str, VASO_ITEMIDS))
    return con.execute(f"""
        SELECT ie.stay_id, ie.starttime AS vaso_change_time
        FROM mimiciv_icu.inputevents ie
        INNER JOIN vaso_windows vw ON ie.stay_id = vw.stay_id
        WHERE ie.itemid IN ({itemids})
          AND ie.starttime BETWEEN vw.t_min AND vw.t_max
    """).df()


def extract_ned(con: duckdb.DuckDBPyConnection,
                turns: pd.DataFrame) -> pd.DataFrame:
    """获取每个 Turn 时刻的 norepinephrine equivalent dose (策略 A 协变量)。"""
    con.register("turn_times",
                 turns[["stay_id", "turn_time"]].drop_duplicates())

    return con.execute("""
        SELECT
            t.stay_id,
            t.turn_time,
            COALESCE(ned.norepinephrine_equivalent_dose, 0) AS ned_mcg_kg_min
        FROM turn_times t
        LEFT JOIN mimiciv_derived.norepinephrine_equivalent_dose ned
            ON t.stay_id = ned.stay_id
            AND t.turn_time BETWEEN ned.starttime AND ned.endtime
    """).df()


# ── 单事件计算 ─────────────────────────────────────────────────────────────────

def compute_auc_recovery(turn_time: pd.Timestamp,
                         map_df: pd.DataFrame) -> dict | None:
    """对单个 Turn 事件计算 AUC_recovery。返回 dict 或 None (数据不足时)。"""
    if map_df.empty or "charttime" not in map_df.columns:
        return None

    map_df = map_df.sort_values("charttime", kind="mergesort").reset_index(drop=True)

    T = pd.Timestamp(turn_time)
    ct = map_df["charttime"]
    baseline_win = map_df[(ct >= T - pd.Timedelta(minutes=BASELINE_MIN)) & (ct < T)]
    recovery_win = map_df[(ct >= T) & (ct <= T + pd.Timedelta(minutes=RECOVERY_MIN))]
    recovery_win = recovery_win.sort_values("charttime", kind="mergesort").reset_index(drop=True)

    n_total = len(baseline_win) + len(recovery_win)
    if n_total < MIN_PTS or len(recovery_win) < 2:
        return None

    baseline = baseline_win["map_value"].mean()
    if pd.isna(baseline):
        baseline = recovery_win.iloc[0]["map_value"]

    t_pts = recovery_win["charttime"].values
    v_pts = recovery_win["map_value"].values
    dev   = np.abs(v_pts.astype(float) - float(baseline))
    t_min = (t_pts.astype("datetime64[s]") -
             np.datetime64(T.to_pydatetime(), "s")).astype(float) / 60.0

    auc = float(np.trapezoid(dev, t_min))

    return {
        "baseline_map":   float(baseline),
        "auc_recovery":   auc,
        "n_pts_total":    n_total,
        "n_pts_recovery": len(recovery_win),
    }


# ── 主处理 ─────────────────────────────────────────────────────────────────────

def process_events(turns: pd.DataFrame,
                   map_all: pd.DataFrame,
                   vaso_changes: pd.DataFrame,
                   ned_df: pd.DataFrame) -> pd.DataFrame:
    """对每个 Turn 事件：排除 → 计算 AUC → 附加 NED。"""
    turns["turn_time"] = pd.to_datetime(turns["turn_time"])
    turns["T0"]        = pd.to_datetime(turns["T0"])
    map_all["charttime"] = pd.to_datetime(map_all["charttime"])
    vaso_changes["vaso_change_time"] = pd.to_datetime(vaso_changes["vaso_change_time"])
    ned_df["turn_time"] = pd.to_datetime(ned_df["turn_time"])
    turns = turns.sort_values(["stay_id", "T0", "turn_time"], kind="mergesort").reset_index(drop=True)
    map_all = map_all.sort_values(["stay_id", "charttime"], kind="mergesort").reset_index(drop=True)
    vaso_changes = vaso_changes.sort_values(["stay_id", "vaso_change_time"], kind="mergesort").reset_index(drop=True)
    ned_df = ned_df.sort_values(["stay_id", "turn_time"], kind="mergesort").reset_index(drop=True)

    import bisect
    vaso_by_stay: dict[int, list] = {}
    for _, vc in vaso_changes.iterrows():
        vaso_by_stay.setdefault(int(vc["stay_id"]), []).append(vc["vaso_change_time"])
    vaso_by_stay = {sid: sorted(ts) for sid, ts in vaso_by_stay.items()}

    map_by_stay = {sid: grp.reset_index(drop=True)
                   for sid, grp in map_all.groupby("stay_id")}

    ned_lookup: dict[tuple, float] = {}
    for _, nr in ned_df.iterrows():
        key = (int(nr["stay_id"]), nr["turn_time"])
        ned_lookup[key] = max(ned_lookup.get(key, 0.0),
                              float(nr["ned_mcg_kg_min"]))

    records = []
    n_excl_vaso = n_excl_data = 0

    for row in turns.itertuples(index=False):
        T   = row.turn_time
        sid = int(row.stay_id)

        T_excl_start = T - pd.Timedelta(minutes=EXCL_WINDOW_MIN)
        T_excl_end   = T + pd.Timedelta(minutes=EXCL_WINDOW_MIN)
        vaso_times = vaso_by_stay.get(sid, [])
        lo = bisect.bisect_left(vaso_times, T_excl_start)
        vaso_in_win = lo < len(vaso_times) and vaso_times[lo] <= T_excl_end
        if vaso_in_win:
            n_excl_vaso += 1
            continue

        stay_map = map_by_stay.get(sid, pd.DataFrame())
        res = compute_auc_recovery(T, stay_map)
        if res is None:
            n_excl_data += 1
            continue

        hours_before_T0 = (T - row.T0).total_seconds() / 3600   # 负值

        ned = ned_lookup.get((sid, T), 0.0)
        has_vaso = ned > 0

        records.append({
            "stay_id":         sid,
            "T0":              row.T0,    # Fix 1: 保留 T0 以标识所属配对
            "group":           row.group,
            "turn_time":       T,
            "hours_before_T0": hours_before_T0,
            "auc_recovery":    res["auc_recovery"],
            "baseline_map":    res["baseline_map"],
            "n_pts_recovery":  res["n_pts_recovery"],
            "ned_mcg_kg_min":  ned,
            "has_vasopressor": has_vaso,
        })

    print(f"  排除 (升压药变化): {n_excl_vaso:,}")
    print(f"  排除 (数据不足)  : {n_excl_data:,}")
    return pd.DataFrame(records)


# ── 统计分析 ───────────────────────────────────────────────────────────────────

def patient_auc(events: pd.DataFrame,
                period_lo: float, period_hi: float) -> pd.DataFrame:
    """
    Fix 4: 聚合至患者 (stay_id, T0) 对的中位数 AUC_recovery。
    避免同一患者多个 Turn 事件造成伪重复。
    """
    window = events[(events["hours_before_T0"] >= period_lo) &
                    (events["hours_before_T0"] <  period_hi)].copy()
    return (window.groupby(["stay_id", "T0", "group"])["auc_recovery"]
                  .median()
                  .reset_index())


def summarize_analysis(events: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict]:
    """汇总患者对级别 Step 5 统计，返回 (summary_table, p_values)."""
    pt_early = patient_auc(events, *PERIODS["early"])
    pt_late  = patient_auc(events, *PERIODS["late"])

    rows = []
    pvals = {
        "label": label,
        "within_shock_p": np.nan,
        "within_control_p": np.nan,
        "late_shock_vs_control_p": np.nan,
    }

    for grp in ["shock", "control"]:
        e = pt_early[pt_early["group"] == grp]["auc_recovery"]
        l = pt_late[pt_late["group"]  == grp]["auc_recovery"]
        p = stats.ranksums(e, l).pvalue if len(e) >= 5 and len(l) >= 5 else np.nan
        if grp == "shock":
            pvals["within_shock_p"] = p
        else:
            pvals["within_control_p"] = p

        rows.append({
            "Analysis": label,
            "Group": grp,
            "Comparison": "early vs late",
            "n_early_pairs": int(len(e)),
            "n_late_pairs": int(len(l)),
            "early_auc_median": float(e.median()) if len(e) else np.nan,
            "early_auc_q1": float(e.quantile(.25)) if len(e) else np.nan,
            "early_auc_q3": float(e.quantile(.75)) if len(e) else np.nan,
            "late_auc_median": float(l.median()) if len(l) else np.nan,
            "late_auc_q1": float(l.quantile(.25)) if len(l) else np.nan,
            "late_auc_q3": float(l.quantile(.75)) if len(l) else np.nan,
            "p_value": float(p) if not np.isnan(p) else np.nan,
        })

    s_late = pt_late[pt_late["group"] == "shock"]["auc_recovery"]
    c_late = pt_late[pt_late["group"] == "control"]["auc_recovery"]
    p_sc = stats.ranksums(s_late, c_late).pvalue if len(s_late) >= 5 and len(c_late) >= 5 else np.nan
    pvals["late_shock_vs_control_p"] = p_sc
    rows.append({
        "Analysis": label,
        "Group": "shock_vs_control",
        "Comparison": "late shock vs control",
        "n_early_pairs": np.nan,
        "n_late_pairs": int(len(s_late) + len(c_late)),
        "early_auc_median": np.nan,
        "early_auc_q1": np.nan,
        "early_auc_q3": np.nan,
        "late_auc_median": np.nan,
        "late_auc_q1": np.nan,
        "late_auc_q3": np.nan,
        "shock_late_auc_median": float(s_late.median()) if len(s_late) else np.nan,
        "control_late_auc_median": float(c_late.median()) if len(c_late) else np.nan,
        "p_value": float(p_sc) if not np.isnan(p_sc) else np.nan,
        "n_shock_late_pairs": int(len(s_late)),
        "n_control_late_pairs": int(len(c_late)),
    })

    raw_e = events[(events["hours_before_T0"] >= PERIODS["early"][0]) &
                   (events["hours_before_T0"] <  PERIODS["early"][1])]
    raw_l = events[(events["hours_before_T0"] >= PERIODS["late"][0]) &
                   (events["hours_before_T0"] <  PERIODS["late"][1])]
    for grp in ["shock", "control"]:
        rows.append({
            "Analysis": label,
            "Group": grp,
            "Comparison": "raw_event_counts",
            "n_early_pairs": int(len(raw_e[raw_e["group"] == grp])),
            "n_late_pairs": int(len(raw_l[raw_l["group"] == grp])),
            "early_auc_median": np.nan,
            "early_auc_q1": np.nan,
            "early_auc_q3": np.nan,
            "late_auc_median": np.nan,
            "late_auc_q1": np.nan,
            "late_auc_q3": np.nan,
            "p_value": np.nan,
        })

    return pd.DataFrame(rows), pvals


def analyze(events: pd.DataFrame, label: str = "全队列") -> tuple[pd.DataFrame, dict]:
    """
    比较 early vs late 时段的 AUC_recovery。
    Fix 4: 先聚合至 (stay_id, T0) 患者对中位数，再做两样本非配对
    Wilcoxon rank-sum 检验（stats.ranksums）。
    """
    print(f"\n── {label} ──────────────────────────────────────────────────────")
    summary, pvals = summarize_analysis(events, label)

    for grp in ["shock", "control"]:
        row = summary[(summary["Group"] == grp) & (summary["Comparison"] == "early vs late")].iloc[0]
        e_n = int(row["n_early_pairs"])
        l_n = int(row["n_late_pairs"])
        p = row["p_value"]

        print(f"  {grp} (患者对级别 n_early={e_n}, n_late={l_n}):")
        if e_n >= 1:
            print(f"    early  AUC={row['early_auc_median']:.2f} "
                  f"[IQR {row['early_auc_q1']:.2f}–{row['early_auc_q3']:.2f}]")
        if l_n >= 1:
            print(f"    late   AUC={row['late_auc_median']:.2f} "
                  f"[IQR {row['late_auc_q1']:.2f}–{row['late_auc_q3']:.2f}]")
        if not np.isnan(p):
            print(f"    early vs late: Wilcoxon rank-sum (unpaired) p={p:.4f}")

    p_sc = pvals["late_shock_vs_control_p"]
    late_row = summary[(summary["Group"] == "shock_vs_control") &
                       (summary["Comparison"] == "late shock vs control")].iloc[0]
    if not np.isnan(p_sc):
        print(f"\n  Late 时段 shock vs control (患者对中位数): "
              f"Wilcoxon rank-sum (unpaired) p={p_sc:.4f}  "
              f"n=({int(late_row['n_shock_late_pairs'])},{int(late_row['n_control_late_pairs'])})")

    # 补充: 原始事件级别统计 (仅供参考，不作为主要结果)
    print(f"\n  [参考] 原始事件级别 (n_events):")
    for grp in ["shock", "control"]:
        raw_row = summary[(summary["Group"] == grp) &
                          (summary["Comparison"] == "raw_event_counts")].iloc[0]
        print(f"    {grp}: early n={int(raw_row['n_early_pairs'])}, "
              f"late n={int(raw_row['n_late_pairs'])}")

    return summary, pvals


# ── 图表 ───────────────────────────────────────────────────────────────────────

def fig3_recovery(events: pd.DataFrame,
                  map_all: pd.DataFrame,
                  path: Path,
                  pvals: dict | None = None) -> None:
    """Figure 3: 以 Turn 为零点对齐的平均 MAP 恢复曲线。"""
    map_all["charttime"] = pd.to_datetime(map_all["charttime"])
    events["turn_time"]  = pd.to_datetime(events["turn_time"])
    map_by_stay = {sid: grp for sid, grp in map_all.groupby("stay_id")}

    aligned = []
    for row in events.itertuples(index=False):
        h = row.hours_before_T0
        period = None
        for pname, (plo, phi) in PERIODS.items():
            if plo <= h < phi:
                period = pname
                break
        if period is None:
            continue

        stay_map = map_by_stay.get(row.stay_id, pd.DataFrame())
        if stay_map.empty:
            continue

        T = row.turn_time
        win = stay_map[
            (stay_map["charttime"] >= T) &
            (stay_map["charttime"] <= T + pd.Timedelta(minutes=RECOVERY_MIN))
        ]
        if win.empty:
            continue

        baseline = row.baseline_map
        for _, mrow in win.iterrows():
            t_min = (mrow["charttime"] - T).total_seconds() / 60
            aligned.append({
                "group": row.group,
                "period": period,
                "t_min": t_min,
                "deviation": mrow["map_value"] - baseline,
            })

    if not aligned:
        print("  [WARN] fig3: no aligned data points")
        return

    df = pd.DataFrame(aligned)
    df["t_bin"] = (df["t_min"] // 5 * 5).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), sharey=True)
    fig.suptitle(
        "MAP recovery after Turn events (exploratory)",
        fontsize=11,
        fontweight="bold",
        color="#2C2C2C",
        y=0.98,
    )

    for ax_idx, (ax, grp) in enumerate(zip(axes, ["shock", "control"])):
        sub = df[df["group"] == grp]
        for period, color in COLORS_PERIOD.items():
            p_sub = sub[sub["period"] == period]
            if p_sub.empty:
                continue
            agg = (p_sub.groupby("t_bin")["deviation"]
                        .agg(mean="mean", sem=lambda x: x.sem())
                        .reset_index())
            if agg.empty:
                continue
            x, y, ci = agg["t_bin"], agg["mean"], 1.96 * agg["sem"]
            label = "Early (-24 h to -12 h)" if period == "early" else "Late (-6 h to 0 h)"
            ax.plot(x, y, color=color, label=label, linewidth=2.2)
            ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.16)
        ax.axhline(0, color="#7F8C8D", linestyle="--", linewidth=1.0)
        ax.axvline(0, color="#BDBDBD", linestyle=":", linewidth=1.0)
        ax.set_title(grp.capitalize(), fontsize=11)
        ax.set_xlabel("Minutes after Turn", fontsize=10)
        ax.spines["left"].set_color("#BBBBBB")
        ax.spines["bottom"].set_color("#BBBBBB")
        ax.text(
            0.0, 1.10, chr(ord("A") + ax_idx),
            transform=ax.transAxes,
            fontsize=14, fontweight="bold",
            va="top", ha="right", color="#333333",
        )

    axes[0].set_ylabel("MAP deviation from baseline (mmHg)", fontsize=10)

    if pvals is None:
        pvals = {}
    within_p = pvals.get("within_shock_p", np.nan)
    between_p = pvals.get("late_shock_vs_control_p", np.nan)
    p_annotations = {
        "shock":   f"Within-group (late vs early): p = {within_p:.3f}\n"
                   f"Between-group (late window): p = {between_p:.3f}",
        "control": f"Between-group (late window): p = {between_p:.3f}",
    }
    for ax, grp in zip(axes, ["shock", "control"]):
        ax.text(
            0.03, 0.97, p_annotations[grp],
            transform=ax.transAxes,
            fontsize=7.4,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="#DDDDDD", linewidth=0.6, alpha=0.88),
        )

    axes[0].legend(frameon=False, loc="upper right", fontsize=8.3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


def export_summary_table(tbl: pd.DataFrame, path: Path) -> None:
    """导出 Step 5 统计汇总表，便于审计。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sort_cols = [c for c in ["Analysis", "Group", "Comparison"] if c in tbl.columns]
    if sort_cols:
        tbl = tbl.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    tbl.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  saved → {path.name}")


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fix 1: 保留全部 (stay_id, T0) 对，不对 stay_id 单独去重
    cohort = pd.read_parquet(COHORT_PATH).drop_duplicates(["stay_id", "T0"])
    cohort["T0"] = pd.to_datetime(cohort["T0"])
    print(f"队列 (stay_id, T0) 对数: {len(cohort):,}  "
          f"(唯一 stay_id: {cohort['stay_id'].nunique():,})\n")

    con = duckdb.connect(str(DB_PATH))
    con.execute("PRAGMA threads=8")

    # ── 1. 提取 Turn 事件 ──────────────────────────────────────────────────
    print("── 提取 Turn 事件 ──────────────────────────────────────────────")
    print("  扫描 chartevents...", flush=True)
    turns = extract_turn_events(con, cohort)
    turns["turn_time"] = pd.to_datetime(turns["turn_time"])
    turns["T0"]        = pd.to_datetime(turns["T0"])
    print(f"  T0 前 48h 内 Turn 事件: {len(turns):,}  "
          f"(涉及 {turns['stay_id'].nunique():,} 个 stay)")

    if turns.empty:
        print("[WARN] 无 Turn 事件，退出")
        con.close()
        return

    # ── 2. 提取 MAP 和升压药 ───────────────────────────────────────────────
    print("\n── 提取 MAP + 升压药变化 ────────────────────────────────────────")
    print("  扫描 MAP...", flush=True)
    map_all = extract_map_around_turns(con, turns)
    map_all["charttime"] = pd.to_datetime(map_all["charttime"])
    print(f"  MAP 记录: {len(map_all):,}")

    print("  扫描 inputevents...", flush=True)
    vaso_changes = extract_vaso_events(con, turns)
    print(f"  升压药变化记录: {len(vaso_changes):,}")

    print("  获取 NED...", flush=True)
    ned_df = extract_ned(con, turns)
    con.close()

    # ── 3. 计算 AUC_recovery ───────────────────────────────────────────────
    print("\n── 计算 AUC_recovery ────────────────────────────────────────────")
    events = process_events(turns, map_all, vaso_changes, ned_df)
    print(f"  有效事件: {len(events):,}  "
          f"(涉及 {events['stay_id'].nunique():,} 个患者, "
          f"{events.groupby(['stay_id','T0']).ngroups:,} 个配对对)")

    if events.empty:
        print("[WARN] 无有效事件，退出")
        return

    print(f"  shock 事件: {(events.group=='shock').sum():,}")
    print(f"  control 事件: {(events.group=='control').sum():,}")
    print(f"  有升压药事件: {events.has_vasopressor.sum():,}  "
          f"({events.has_vasopressor.mean():.1%})")

    # 多事件患者分布
    multi = (events.groupby(["stay_id", "T0"]).size()
                   .rename("n_events"))
    print(f"  每个 (stay_id,T0) 对的事件数: "
          f"median={multi.median():.0f}, max={multi.max()}")
    print(f"  (Fix 4: 统计检验将先聚合至患者对中位数)")

    events.to_parquet(EVENTS_PATH, index=False)

    # ── 4. 统计分析 ────────────────────────────────────────────────────────
    print("\n── 统计分析 ─────────────────────────────────────────────────────")
    tbl_all, pvals_all = analyze(events, "全队列")

    no_vaso = events[~events["has_vasopressor"]]
    summary_tables = [tbl_all]
    if len(no_vaso) >= 20:
        tbl_no_vaso, _ = analyze(no_vaso, "策略B: 无升压药亚组")
        summary_tables.append(tbl_no_vaso)
    else:
        print(f"\n  [WARN] 无升压药亚组事件太少 (n={len(no_vaso)})，跳过")

    export_summary_table(pd.concat(summary_tables, ignore_index=True), SUMMARY_PATH)

    # ── 5. 出图 ────────────────────────────────────────────────────────────
    print("\n── 出图 ─────────────────────────────────────────────────────────")
    fig3_recovery(events, map_all, OUTPUT_DIR / "fig3_recovery.png", pvals=pvals_all)

    print("\n[完成]")
    print(f"  {EVENTS_PATH}")


if __name__ == "__main__":
    main()
