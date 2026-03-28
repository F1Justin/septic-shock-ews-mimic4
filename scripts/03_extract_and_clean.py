"""
Step 3: 时间序列提取与清洗

对每个队列 (stay_id, T0) 对提取 T0 前 72h 的 MAP/HR，
其中前 24h 仅作为 burn-in，用于稳定 24h trailing mean；
真正分析窗口仍为后 48h。

设计变更 (Fix 1, Fix 2):
  - 以 (stay_id, T0) 为处理单元，而非仅 stay_id。
    对照患者在风险集设计中可有多个不同 T0 (对应不同配对)，每个 (stay_id, T0)
    独立提取窗口并计算特征。
  - is_interpolated 拆分为 map_is_interpolated / hr_is_interpolated。

清洗流程:
  1. 生理范围过滤: MAP [30,200], HR [20,300]
  2. 1h 重采样 (中位数聚合); MAP 优先使用 ABPm，补充 NBPm
  3. 因果去趋势: 仅使用当前及既往 24h 的 trailing mean
  4. mu±3sigma 异常值替换为 mu (仅作用于实测点)
  5. 排除: 后 48h 分析窗口内实测点 < 24 的患者

输出:
  data/vitals_cleaned.parquet      -- 含 T0 列，以 (stay_id, T0) 标识窗口
  data/cleaning_diagnostics.parquet -- 含 T0 列

用法: python scripts/03_extract_and_clean.py
预计运行时间: chartevents 扫描 5-10 分钟 + Python 处理 2-5 分钟
"""

import argparse
import sys
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH      = PROJECT_ROOT / "mimiciv" / "mimiciv.db"
COHORT_PATH  = PROJECT_ROOT / "data" / "cohort.parquet"
VITALS_PATH  = PROJECT_ROOT / "data" / "vitals_cleaned.parquet"
DIAG_PATH    = PROJECT_ROOT / "data" / "cleaning_diagnostics.parquet"

# itemids
MAP_ABP_ID = 220052   # Arterial BP mean (有创)
MAP_NBP_ID = 220181   # Non-invasive BP mean (无创)
HR_ID      = 220045   # Heart rate

MAP_RANGE     = (30.0, 200.0)
HR_RANGE      = (20.0, 300.0)
EXTRACT_HOURS = 72
WINDOW_HOURS  = 48
BURNIN_HOURS  = EXTRACT_HOURS - WINDOW_HOURS
TREND_WINDOW  = 24    # 小时; causal trailing window
MIN_ACTUAL    = 24    # 排除阈值: 48h 内实测点数
LOCAL_LINEAR_WINDOW = 12


def tagged_path(path: Path, tag: str) -> Path:
    """在文件名后添加 tag；空 tag 时返回原路径。"""
    if not tag:
        return path
    return path.with_name(f"{path.stem}{tag}{path.suffix}")


# ── 单患者处理 ─────────────────────────────────────────────────────────────────

def resample_hourly(df_abp: pd.DataFrame,
                    df_nbp: pd.DataFrame,
                    df_hr:  pd.DataFrame,
                    hours_index: pd.DatetimeIndex):
    """
    将原始记录聚合到 hours_index 定义的整点网格。
    MAP: ABPm 优先，缺失时用 NBPm 补充。
    返回 (map_raw, map_src, map_actual, hr_raw, hr_actual)，均以 hours_index 为索引。
    """
    def hourly_median(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        return df.assign(hour=df["charttime"].dt.floor("h")) \
                 .groupby("hour")["valuenum"].median()

    abp_h = hourly_median(df_abp)
    nbp_h = hourly_median(df_nbp)
    hr_h  = hourly_median(df_hr)

    map_raw = abp_h.reindex(hours_index)
    map_src = pd.Series(
        np.where(map_raw.notna(), "abp", None), index=hours_index, dtype=object
    )
    missing = map_raw.isna()
    nbp_fill = nbp_h.reindex(hours_index)
    map_raw[missing] = nbp_fill[missing]
    map_src[missing & map_raw.notna()] = "nbp"

    hr_raw = hr_h.reindex(hours_index)

    return (map_raw,
            map_src,
            map_raw.notna(),
            hr_raw,
            hr_raw.notna())


def causal_detrend(raw: pd.Series, window_hours: int = TREND_WINDOW) -> pd.Series:
    """
    1. 线性插值填充 NaN
    2. 用当前及既往 window_hours 的 trailing mean 估计趋势
    3. 残差 = raw - trend (NaN 原位保留)
    """
    filled = raw.interpolate("linear").ffill().bfill()
    if filled.isna().all():
        return pd.Series(np.nan, index=raw.index)
    trend = filled.rolling(window=window_hours, min_periods=window_hours).mean()
    return raw - trend


def causal_local_linear_fit(series: pd.Series,
                            window_hours: int = LOCAL_LINEAR_WINDOW) -> pd.Series:
    """
    对已填充序列做 trailing local linear fit，返回每个时点的拟合值。
    仅使用当前及既往 window_hours 个点，保持因果性。
    """
    x = np.arange(window_hours, dtype=float)
    x_centered = x - x.mean()
    denom = float(np.square(x_centered).sum())

    def _predict_last(win: np.ndarray) -> float:
        if np.isnan(win).any():
            return np.nan
        y = win.astype(float)
        y_mean = y.mean()
        slope = float(np.dot(y - y_mean, x_centered) / denom)
        intercept = y_mean - slope * x.mean()
        return intercept + slope * x[-1]

    return series.rolling(window=window_hours, min_periods=window_hours).apply(
        _predict_last,
        raw=True,
    )


def double_detrend(raw: pd.Series) -> pd.Series:
    """
    Double detrending:
      1) 24h trailing mean
      2) 在残差上再做 12h trailing local linear detrending
    """
    filled = raw.interpolate("linear").ffill().bfill()
    if filled.isna().all():
        return pd.Series(np.nan, index=raw.index)
    mean_trend = filled.rolling(window=TREND_WINDOW, min_periods=TREND_WINDOW).mean()
    first_pass = filled - mean_trend
    local_trend = causal_local_linear_fit(first_pass, LOCAL_LINEAR_WINDOW)
    return raw - mean_trend - local_trend


def clip_outliers(residual: pd.Series, actual: pd.Series) -> pd.Series:
    """对实测点: |res - mu| > 3σ 则替换为 mu."""
    act_res = residual[actual]
    if len(act_res) < 3:
        return residual
    mu, sigma = act_res.mean(), act_res.std()
    if sigma == 0:
        return residual
    result = residual.copy()
    result[actual & ((residual - mu).abs() > 3 * sigma)] = mu
    return result


def gap_stats(actual: pd.Series) -> tuple[float, int]:
    """返回 (最大连续缺口小时数, >2h 缺口个数)."""
    idx = actual.index[actual]
    if len(idx) < 2:
        return float(WINDOW_HOURS), 0
    gaps = [(idx[i + 1] - idx[i]).total_seconds() / 3600 for i in range(len(idx) - 1)]
    return max(gaps), sum(g > 2 for g in gaps)


def process_stay(stay_id: int,
                 raw: pd.DataFrame,
                 T0: pd.Timestamp,
                 detrend_mode: str = "standard") -> tuple:
    """
    处理单个 (stay_id, T0) 窗口，返回 (ts_df | None, diag_dict)。
    ts_df is None 表示被排除。
    raw 可包含超出 T0-48h 窗口的数据；先按当前 T0 截断，再计算 dominant_source，
    避免同一 stay 的其他 T0 窗口记录污染本窗口的 ABP/NBP 计数。
    """
    hours_index = pd.date_range(end=T0.floor("h"), periods=EXTRACT_HOURS, freq="h")
    raw_window = raw.loc[
        (raw["charttime"] >= hours_index.min()) &
        (raw["charttime"] <= hours_index.max())
    ]

    abp = raw_window.loc[(raw_window["itemid"] == MAP_ABP_ID) &
                         raw_window["valuenum"].between(*MAP_RANGE)]
    nbp = raw_window.loc[(raw_window["itemid"] == MAP_NBP_ID) &
                         raw_window["valuenum"].between(*MAP_RANGE)]
    hr  = raw_window.loc[(raw_window["itemid"] == HR_ID) &
                         raw_window["valuenum"].between(*HR_RANGE)]

    n_raw_map = len(abp) + len(nbp)
    n_raw_hr  = len(hr)
    n_abp, n_nbp = len(abp), len(nbp)

    map_raw, map_src, map_act, hr_raw, hr_act = \
        resample_hourly(abp, nbp, hr, hours_index)

    analysis_mask = pd.Series(False, index=hours_index)
    analysis_mask.iloc[-WINDOW_HOURS:] = True

    map_raw_48 = map_raw[analysis_mask]
    hr_raw_48 = hr_raw[analysis_mask]
    map_src_48 = map_src[analysis_mask]
    map_act_48 = map_act[analysis_mask]
    hr_act_48 = hr_act[analysis_mask]

    n_act_map = int(map_act_48.sum())
    n_act_hr  = int(hr_act_48.sum())

    tot = n_abp + n_nbp
    dominant = ("abp"  if tot > 0 and n_abp / tot > 0.7 else
                "nbp"  if tot > 0 and n_nbp / tot > 0.7 else
                "none" if tot == 0 else "mixed")

    max_gap, n_gaps_gt2 = gap_stats(map_act_48)
    var_before = map_raw_48[map_act_48].var() if n_act_map > 1 else np.nan

    excluded = (n_act_map < MIN_ACTUAL) or (n_act_hr < MIN_ACTUAL)

    diag: dict = {
        "stay_id":           stay_id,
        "T0":                T0,
        "n_raw_map":         n_raw_map,
        "n_raw_hr":          n_raw_hr,
        "n_actual_map":      n_act_map,
        "n_actual_hr":       n_act_hr,
        "interp_ratio_map":  1 - n_act_map / WINDOW_HOURS,
        "interp_ratio_hr":   1 - n_act_hr  / WINDOW_HOURS,
        "dominant_source":   dominant,
        "n_abp":             n_abp,
        "n_nbp":             n_nbp,
        "max_gap_hours_map": max_gap,
        "n_gaps_gt2h_map":   n_gaps_gt2,
        "excluded":          excluded,
        "var_ratio_map":     np.nan,
        "var_ratio_hr":      np.nan,
        "detrend_mode":      detrend_mode,
    }

    if excluded:
        return None, diag

    detrend_fn = double_detrend if detrend_mode == "double" else causal_detrend
    map_res = detrend_fn(map_raw)
    hr_res  = detrend_fn(hr_raw)

    map_res = clip_outliers(map_res, map_act)
    hr_res  = clip_outliers(hr_res,  hr_act)

    map_res_48 = map_res[analysis_mask]
    hr_res_48 = hr_res[analysis_mask]

    var_after = map_res_48[map_act_48].var() if n_act_map > 1 else np.nan
    diag["var_ratio_map"] = var_after / var_before if var_before > 0 else np.nan

    var_before_hr = hr_raw_48[hr_act_48].var() if n_act_hr > 1 else np.nan
    var_after_hr  = hr_res_48[hr_act_48].var() if n_act_hr  > 1 else np.nan
    diag["var_ratio_hr"] = (var_after_hr / var_before_hr
                            if var_before_hr and var_before_hr > 0 else np.nan)

    ts = pd.DataFrame({
        "stay_id":              stay_id,
        "T0":                   T0,           # Fix 1: 标识所属配对窗口
        "charttime":            hours_index[analysis_mask],
        "map_residual":         map_res_48.to_numpy(float),
        "hr_residual":          hr_res_48.to_numpy(float),
        "map_raw":              map_raw_48.to_numpy(float),
        "hr_raw":               hr_raw_48.to_numpy(float),
        "map_source":           map_src_48.to_numpy(object),
        # Fix 2: 拆分为独立的插值标志列
        "map_is_interpolated":  (~map_act_48).to_numpy(bool),
        "hr_is_interpolated":   (~hr_act_48).to_numpy(bool),
    })
    return ts, diag


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detrend-mode",
        choices=["standard", "double"],
        default="standard",
        help="standard=24h trailing mean; double=24h trailing mean + 12h local linear detrending",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="附加到输出文件名的后缀，例如 _double_detrend",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vitals_path = tagged_path(VITALS_PATH, args.suffix)
    diag_path = tagged_path(DIAG_PATH, args.suffix)

    # Fix 1: 保留所有 (stay_id, T0) 对，不对 stay_id 去重
    cohort = pd.read_parquet(COHORT_PATH)
    cohort["T0"] = pd.to_datetime(cohort["T0"])
    cohort = cohort.drop_duplicates(["stay_id", "T0"])  # 仅去除完全重复行
    print(f"队列 (stay_id, T0) 对数: {len(cohort):,}  "
          f"(唯一 stay_id: {cohort['stay_id'].nunique():,})\n")
    print(f"去趋势模式: {args.detrend_mode}")
    print(f"输出后缀  : {args.suffix or '(none)'}\n")

    # ── 1. 从 DuckDB 提取原始 chartevents ──────────────────────────────────
    print("── 提取 chartevents ─────────────────────────────────────────────")
    print("  扫描 chartevents.csv.gz（约 5-10 分钟）...", flush=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute("PRAGMA threads=8")

    # Fix 1: 每个 stay 取所有 T0 的并集窗口 (min(T0)-48h 到 max(T0))，避免
    # 多 T0 JOIN 产生重复行。process_stay 通过 reindex 自然截断至各自窗口。
    stay_ranges = (cohort.groupby("stay_id")["T0"]
                  .agg(t_min=lambda x: x.min() - pd.Timedelta(hours=EXTRACT_HOURS),
                        t_max="max")
                   .reset_index())
    con.register("stay_ranges_tmp", stay_ranges)

    vitals = con.execute(f"""
        SELECT ce.stay_id, ce.charttime, ce.itemid, ce.valuenum
        FROM mimiciv_icu.chartevents AS ce
        INNER JOIN stay_ranges_tmp AS sr ON ce.stay_id = sr.stay_id
        WHERE ce.itemid IN ({MAP_ABP_ID}, {MAP_NBP_ID}, {HR_ID})
          AND ce.valuenum IS NOT NULL
          AND ce.charttime >= sr.t_min
          AND ce.charttime <= sr.t_max
    """).df()
    con.close()

    vitals["charttime"] = pd.to_datetime(vitals["charttime"])
    print(f"  原始记录: {len(vitals):,} 条，涉及 {vitals['stay_id'].nunique():,} 个 stay\n")

    # ── 2. 按 (stay_id, T0) 处理 ───────────────────────────────────────────
    print("── 清洗 ─────────────────────────────────────────────────────────")
    _empty_vitals = pd.DataFrame(columns=["stay_id", "charttime", "itemid", "valuenum"])
    vitals_by_stay = {sid: grp for sid, grp in vitals.groupby("stay_id", sort=False)}

    # 所有 (stay_id, T0) 对
    pairs = cohort[["stay_id", "T0"]].drop_duplicates().values.tolist()
    n_total = len(pairs)

    all_ts   = []
    all_diag = []

    for i, (sid, t0) in enumerate(pairs, 1):
        T0  = pd.Timestamp(t0)
        grp = vitals_by_stay.get(sid, _empty_vitals)
        ts, diag = process_stay(sid, grp, T0, detrend_mode=args.detrend_mode)
        if ts is not None:
            all_ts.append(ts)
        all_diag.append(diag)
        if i % 2000 == 0:
            print(f"  {i:,}/{n_total:,} (stay_id, T0) 对已处理...", flush=True)

    # 队列中没有 chartevents 记录的对 也计入诊断（全排除）
    processed_keys = {(d["stay_id"], d["T0"]) for d in all_diag}
    for _, row in cohort.iterrows():
        key = (row["stay_id"], row["T0"])
        if key not in processed_keys:
            all_diag.append({
                "stay_id": row["stay_id"], "T0": row["T0"], "excluded": True,
                "n_raw_map": 0, "n_raw_hr": 0,
                "n_actual_map": 0, "n_actual_hr": 0,
                "interp_ratio_map": 1.0, "interp_ratio_hr": 1.0,
                "dominant_source": "none", "n_abp": 0, "n_nbp": 0,
                "max_gap_hours_map": WINDOW_HOURS, "n_gaps_gt2h_map": 0,
                "var_ratio_map": np.nan, "var_ratio_hr": np.nan,
                "detrend_mode": args.detrend_mode,
            })

    # ── 3. 保存 ────────────────────────────────────────────────────────────
    diag_df = pd.DataFrame(all_diag)
    n_kept = (~diag_df["excluded"]).sum()
    n_excl = diag_df["excluded"].sum()

    print(f"\n── 结果 ─────────────────────────────────────────────────────────")
    print(f"  队列 (stay_id,T0) 对  : {len(cohort):,}")
    print(f"  有 chartevents        : {len([d for d in all_diag if d.get('n_raw_map', 0) > 0 or d.get('n_raw_hr', 0) > 0]):,}")
    print(f"  通过清洗              : {n_kept:,}")
    print(f"  被排除                : {n_excl:,}  (实测点 < {MIN_ACTUAL} 或无 chartevents)")

    src_counts = diag_df.loc[~diag_df["excluded"], "dominant_source"].value_counts()
    print(f"\n  dominant_source 分布 (通过清洗):")
    for src, cnt in src_counts.items():
        print(f"    {src:6s}: {cnt:,} ({cnt/n_kept*100:.1f}%)")

    vitals_df = pd.concat(all_ts, ignore_index=True)
    vitals_df.to_parquet(vitals_path, index=False)
    diag_df.to_parquet(diag_path, index=False)

    print(f"\n[完成]")
    print(f"  {vitals_path}")
    print(f"  {diag_path}")


if __name__ == "__main__":
    main()
