"""
eICU 外部验证 — Step 3: EWS 特征计算

在清洗后的每小时格点数据上计算滑动窗口统计量，与 ac1/scripts/04_ews_analysis.py
的实现保持一致（WINDOW_SIZE=12, MIN_ACTUAL_WIN=8, MIN_AC1_PAIRS=8）。

指标:
  AC1(HR residual)          — lag-1 自相关（仅真实相邻点对）
  Var(MAP residual)         — 窗口方差
  Euler χ(0)(HR residual)   — 零轴以下连通分量数（与 euler_ews 一致）
  n_extrema(HR residual)    — 局部极值数

时间窗口汇聚:
  晚窗 (LATE)  : hours_before_T0 ∈ [-12, 0)   — 所有 T0 >= 12h 的患者
  早窗 (EARLY) : hours_before_T0 ∈ [-24, -12)  — 仅 T0 >= 24h 的患者

输出:
  eicu_validation/output/ews_windows_eicu.parquet  (逐窗口)
  eicu_validation/output/ews_patient_eicu.parquet  (患者级汇总)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).parent.parent
NAAS_ROOT    = PROJECT_ROOT.parent
OUTPUT_DIR   = PROJECT_ROOT / "output"

COHORT_PATH  = OUTPUT_DIR / "cohort_eicu.parquet"
VITALS_PATH  = OUTPUT_DIR / "vitals_eicu.parquet"
DIAG_PATH    = OUTPUT_DIR / "cleaning_diagnostics_eicu.parquet"
WINDOWS_PATH = OUTPUT_DIR / "ews_windows_eicu.parquet"
STATS_PATH   = OUTPUT_DIR / "ews_patient_eicu.parquet"

WINDOW_SIZE    = 12
MIN_ACTUAL_WIN = 8
MIN_AC1_PAIRS  = 8

EARLY_LO, EARLY_HI = -24, -12
LATE_LO,  LATE_HI  = -12,   0

MIN_ACTUAL_EULER  = 6   # Euler 指标的宽松阈值（与 euler_ews 对齐）
MIN_ACTUAL_SAMPEN = 8   # SampEn 需要更多点
SAMPEN_R_STD      = 0.2 # 报告用（NaN 率高）
SAMPEN_R_RELAXED  = 0.5 # 回归用（NaN 率低）


# ── 指标函数 ──────────────────────────────────────────────────────────────────

def _ac1(vals: np.ndarray, is_interp: np.ndarray) -> float:
    """Lag-1 自相关，仅保留双端均为真实点的相邻对。"""
    x, y   = vals[:-1], vals[1:]
    ix, iy = is_interp[:-1], is_interp[1:]
    m = ~(np.isnan(x) | np.isnan(y) | ix | iy)
    if m.sum() < MIN_AC1_PAIRS:
        return np.nan
    xm, ym = x[m], y[m]
    if xm.std() == 0 or ym.std() == 0:
        return np.nan
    return float(np.corrcoef(xm, ym)[0, 1])


def _var_map(vals: np.ndarray) -> float:
    v = vals[~np.isnan(vals)]
    return float(np.var(v)) if len(v) >= 2 else np.nan


def _euler_at_zero(vals: np.ndarray, is_interp: np.ndarray) -> float:
    """零轴以下连通分量数 χ(0)，与 euler_ews/scripts/09 定义一致。"""
    actual = vals[~is_interp & ~np.isnan(vals)]
    if len(actual) < MIN_ACTUAL_EULER:
        return np.nan
    below = actual <= 0
    if not below.any():
        return 0.0
    n_comp = int(below[0]) + int((np.diff(below.astype(np.int8)) == 1).sum())
    return float(n_comp)


def _n_extrema(vals: np.ndarray, is_interp: np.ndarray) -> float:
    actual = vals[~is_interp & ~np.isnan(vals)]
    if len(actual) < MIN_ACTUAL_EULER:
        return np.nan
    n_min = len(argrelextrema(actual, np.less,    order=1)[0])
    n_max = len(argrelextrema(actual, np.greater, order=1)[0])
    return float(n_min + n_max)


def _sampen(
    vals: np.ndarray,
    is_interp: np.ndarray,
    m: int = 1,
    r: float | None = None,
) -> float:
    """
    Sample Entropy（Richman & Moorman 2000）。
    r: 绝对容忍度，由调用方传入（= r_factor × patient_48h_std）。
    A=0 时返回 NaN（而非 +∞），保持与 MIMIC 分析一致。
    """
    actual = vals[~is_interp & ~np.isnan(vals)]
    if len(actual) < MIN_ACTUAL_SAMPEN or r is None or r <= 0:
        return np.nan
    N = len(actual)

    def _count(tl: int) -> int:
        c = 0
        for i in range(N - tl):
            tmpl = actual[i : i + tl]
            for j in range(i + 1, N - tl):
                if np.max(np.abs(actual[j : j + tl] - tmpl)) < r:
                    c += 1
        return c

    B = _count(m)
    if B == 0:
        return np.nan
    A = _count(m + 1)
    if A == 0:
        return np.nan
    return float(-np.log(A / B))


# ── 滑动窗口计算 ───────────────────────────────────────────────────────────────

def compute_windows(
    ts: pd.DataFrame,
    global_r_std: float | None = None,
    global_r_rel: float | None = None,
) -> pd.DataFrame:
    """
    对单个 (patientunitstayid, T0_min) 的时间序列计算 12h 滑动窗口统计量。
    ts 需已按 hours_before_T0 排序，且包含以下列:
      hours_before_T0, hr_residual, map_residual,
      hr_is_interpolated, map_is_interpolated

    global_r_std / global_r_rel: 患者级绝对容忍度
        std = 0.2 × patient_series_std（标准，高 NaN；用于报告）
        rel = 0.5 × patient_series_std（宽松，低 NaN；用于回归）
    """
    ts = ts.sort_values("hours_before_T0").reset_index(drop=True)
    if len(ts) < WINDOW_SIZE:
        return pd.DataFrame()

    rows = []
    for i in range(len(ts) - WINDOW_SIZE + 1):
        win = ts.iloc[i : i + WINDOW_SIZE]

        hr_vals   = win["hr_residual"].to_numpy(float)
        hr_interp = win["hr_is_interpolated"].to_numpy(bool)
        map_vals  = win["map_residual"].to_numpy(float)
        map_interp= win["map_is_interpolated"].to_numpy(bool)

        n_act_hr  = int((~hr_interp  & ~np.isnan(hr_vals)).sum())
        n_act_map = int((~map_interp & ~np.isnan(map_vals)).sum())

        low_conf_hr     = n_act_hr  < MIN_ACTUAL_WIN
        low_conf_map    = n_act_map < MIN_ACTUAL_WIN
        low_conf_euler  = n_act_hr  < MIN_ACTUAL_EULER
        low_conf_sampen = n_act_hr  < MIN_ACTUAL_SAMPEN

        center_h = float(win["hours_before_T0"].iloc[WINDOW_SIZE // 2])

        rows.append({
            "hours_before_T0":   round(center_h, 1),
            "ac1_hr":            _ac1(hr_vals, hr_interp),
            "var_map":           _var_map(map_vals),
            "euler_hr":          _euler_at_zero(hr_vals, hr_interp),
            "n_extrema_hr":      _n_extrema(hr_vals, hr_interp),
            # SampEn: standard r (NaN diagnostic) and relaxed r (regression)
            "sampen_std_hr":     _sampen(hr_vals, hr_interp, m=1, r=global_r_std),
            "sampen_rel_hr":     _sampen(hr_vals, hr_interp, m=1, r=global_r_rel),
            "n_actual_hr":       n_act_hr,
            "n_actual_map":      n_act_map,
            "low_conf_hr":       low_conf_hr,
            "low_conf_map":      low_conf_map,
            "low_conf_euler":    low_conf_euler,
            "low_conf_sampen":   low_conf_sampen,
        })
    return pd.DataFrame(rows)


def window_mean(
    windows:   pd.DataFrame,
    metric:    str,
    lo:        float,
    hi:        float,
    conf_col:  str,
) -> float:
    mask = (
        (windows["hours_before_T0"] >= lo)
        & (windows["hours_before_T0"] < hi)
        & ~windows[conf_col]
    )
    vals = windows.loc[mask, metric].dropna()
    return float(vals.mean()) if len(vals) else np.nan


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载数据 ───────────────────────────────────────────────────────────
    print("── 加载数据 ─────────────────────────────────────────────────────────")
    cohort = pd.read_parquet(COHORT_PATH)[
        ["patientunitstayid", "T0_min", "group", "matched_pair_id"]
    ].drop_duplicates(["patientunitstayid", "T0_min"])

    diag = pd.read_parquet(DIAG_PATH)
    passed = diag[~diag["excluded"]][["patientunitstayid", "T0_min"]]

    vitals = pd.read_parquet(VITALS_PATH)
    vitals = vitals.merge(passed, on=["patientunitstayid", "T0_min"])
    print(f"  通过清洗检查的 (patient, T0) 对: {len(passed):,}")

    # ── 2. 逐患者计算滑动窗口 ─────────────────────────────────────────────────
    print("── 计算滑动窗口 ─────────────────────────────────────────────────────")
    groups   = vitals.groupby(["patientunitstayid", "T0_min"], sort=False)
    n_total  = groups.ngroups
    all_wins = []
    pat_rows = []

    for j, ((sid, t0_m), grp) in enumerate(groups, 1):
        # ── 患者级 global_r（用于 SampEn 容忍度）─────────────────────────────
        actual_hr = grp.loc[
            ~grp["hr_is_interpolated"] & grp["hr_residual"].notna(),
            "hr_residual"
        ].to_numpy(float)
        if len(actual_hr) >= 2:
            sd = actual_hr.std(ddof=1)
            global_r_std = SAMPEN_R_STD     * sd if sd > 0 else None
            global_r_rel = SAMPEN_R_RELAXED * sd if sd > 0 else None
        else:
            global_r_std = global_r_rel = None

        wins = compute_windows(grp, global_r_std=global_r_std, global_r_rel=global_r_rel)
        if wins.empty:
            continue

        wins["patientunitstayid"] = sid
        wins["T0_min"]            = t0_m
        all_wins.append(wins)

        # 患者级汇总（晚窗 + 早窗）
        row_info = cohort[(cohort["patientunitstayid"] == sid) & (cohort["T0_min"] == t0_m)]
        t0_h = float(t0_m) / 60

        pat_rows.append({
            "patientunitstayid":           sid,
            "T0_min":                      t0_m,
            "group":                       row_info["group"].iloc[0] if not row_info.empty else np.nan,
            "matched_pair_id":             row_info["matched_pair_id"].iloc[0] if not row_info.empty else np.nan,
            # 晚窗 (所有患者均有)
            "ac1_hr_late_mean":            window_mean(wins, "ac1_hr",        LATE_LO,  LATE_HI,  "low_conf_hr"),
            "var_map_late_mean":           window_mean(wins, "var_map",       LATE_LO,  LATE_HI,  "low_conf_map"),
            "euler_hr_late_mean":          window_mean(wins, "euler_hr",      LATE_LO,  LATE_HI,  "low_conf_euler"),
            "n_extrema_hr_late_mean":      window_mean(wins, "n_extrema_hr",  LATE_LO,  LATE_HI,  "low_conf_euler"),
            "sampen_rel_hr_late_mean":     window_mean(wins, "sampen_rel_hr", LATE_LO,  LATE_HI,  "low_conf_sampen"),
            "sampen_std_hr_late_mean":     window_mean(wins, "sampen_std_hr", LATE_LO,  LATE_HI,  "low_conf_sampen"),
            # 早窗 (仅 T0 >= 24h 有意义)
            "ac1_hr_early_mean":           window_mean(wins, "ac1_hr",        EARLY_LO, EARLY_HI, "low_conf_hr")      if t0_h >= 24 else np.nan,
            "var_map_early_mean":          window_mean(wins, "var_map",       EARLY_LO, EARLY_HI, "low_conf_map")     if t0_h >= 24 else np.nan,
            "euler_hr_early_mean":         window_mean(wins, "euler_hr",      EARLY_LO, EARLY_HI, "low_conf_euler")   if t0_h >= 24 else np.nan,
            "n_extrema_hr_early_mean":     window_mean(wins, "n_extrema_hr",  EARLY_LO, EARLY_HI, "low_conf_euler")   if t0_h >= 24 else np.nan,
            "sampen_rel_hr_early_mean":    window_mean(wins, "sampen_rel_hr", EARLY_LO, EARLY_HI, "low_conf_sampen")  if t0_h >= 24 else np.nan,
            "sampen_std_hr_early_mean":    window_mean(wins, "sampen_std_hr", EARLY_LO, EARLY_HI, "low_conf_sampen")  if t0_h >= 24 else np.nan,
            "n_windows_hr":                int((~wins["low_conf_hr"]).sum()),
            "n_windows_map":               int((~wins["low_conf_map"]).sum()),
        })

        if j % 200 == 0:
            print(f"  {j:,}/{n_total:,}...", flush=True)

    windows_df = pd.concat(all_wins, ignore_index=True)
    stats_df   = pd.DataFrame(pat_rows)

    # ── 3. 保存 ───────────────────────────────────────────────────────────────
    windows_df.to_parquet(WINDOWS_PATH, index=False)
    stats_df.to_parquet(STATS_PATH, index=False)
    print(f"\n  → {WINDOWS_PATH}  ({len(windows_df):,} 个窗口)")
    print(f"  → {STATS_PATH}  ({len(stats_df):,} 患者)")

    # 简要统计
    merged = stats_df.merge(
        cohort[["patientunitstayid","T0_min","group"]].drop_duplicates(),
        on=["patientunitstayid","T0_min"], how="left", suffixes=("","_c")
    )
    for grp, sub in merged.groupby("group"):
        late_ac1 = sub["ac1_hr_late_mean"].dropna()
        print(f"  {grp}: n={len(sub)}, AC1 late mean={late_ac1.mean():.3f}±{late_ac1.std():.3f}"
              f"  (n_valid={len(late_ac1)})")


if __name__ == "__main__":
    main()
