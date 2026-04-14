"""
eICU 外部验证 — Step 2: 生命体征提取与清洗

输入:
  eicu_validation/output/cohort_eicu.parquet
  eicu-crd/2.0/vitalPeriodic.csv.gz
  eicu-crd/2.0/vitalAperiodic.csv.gz

处理流程:
  1. 仅提取队列患者（patientunitstayid）的 HR / MAP 数据
  2. 时间窗口: T0 前 (EXTRACT_HOURS + BURNIN_HOURS) 小时内
     - BURNIN_HOURS = 24h: 仅用于稳定 24h trailing mean，不参与分析
     - EXTRACT_HOURS = 48h: 实际分析窗口 (= 24h早窗 + 24h晚窗)
  3. 5 分钟数据 → 按整点小时分箱取中位数
  4. MAP 来源: 有创 (systemicmean) 优先，缺失时补充无创 (noninvasivemean)
  5. 生理范围过滤: MAP [30, 200], HR [20, 300]
  6. 24h 因果拖尾均值去趋势 → hr_residual, map_residual
  7. ±3σ 异常点替换为均值（仅实测点）
  8. 线性插值（最多连续 2 格），并标记 is_interpolated

时间轴约定:
  - hours_before_T0 = (observationoffset_min - T0_min) / 60
  - 负值 = T0 之前；0 = T0
  - 分析窗口: LATE [-12, 0), EARLY [-24, -12)  (T0 >= 24h 时启用)

输出:
  eicu_validation/output/vitals_eicu.parquet
  eicu_validation/output/cleaning_diagnostics_eicu.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
NAAS_ROOT    = PROJECT_ROOT.parent
EICU_DIR     = NAAS_ROOT / "eicu-crd" / "2.0"
OUTPUT_DIR   = PROJECT_ROOT / "output"

COHORT_PATH  = OUTPUT_DIR / "cohort_eicu.parquet"
VITALS_PATH  = OUTPUT_DIR / "vitals_eicu.parquet"
DIAG_PATH    = OUTPUT_DIR / "cleaning_diagnostics_eicu.parquet"

MAP_RANGE      = (30.0, 200.0)
HR_RANGE       = (20.0, 300.0)
EXTRACT_HOURS  = 48    # 实际分析用: [-48h, 0h)
BURNIN_HOURS   = 24    # 额外拉取用于去趋势计算
TOTAL_HOURS    = EXTRACT_HOURS + BURNIN_HOURS   # 72h 总窗口
TREND_WINDOW   = 24    # trailing mean 回溯小时数
MAX_INTERP_GAP = 2     # 最多插值连续缺失格数
MIN_ACTUAL_LATE = 6    # 晚窗 [-12h, 0h) 内最少实测点数（否则排除）


# ── 单患者处理 ─────────────────────────────────────────────────────────────────

def resample_hourly_eicu(
    hr_raw:  pd.DataFrame,   # columns: hour_idx, value
    map_abp: pd.DataFrame,   # columns: hour_idx, value (有创)
    map_nbp: pd.DataFrame,   # columns: hour_idx, value (无创)
    hour_index: np.ndarray,  # 整点 hour_idx 数组
) -> pd.DataFrame:
    """
    将原始 5-min 记录聚合到 hour_index 定义的整点网格。
    MAP: ABP 优先；若某小时无 ABP 则用 NBP；记录 map_source ('abp'/'nbp'/NaN)。
    返回 DataFrame，索引为 hour_idx。
    """
    def hourly_median(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        return df.groupby("hour_idx")["value"].median()

    hr_h  = hourly_median(hr_raw)
    abp_h = hourly_median(map_abp)
    nbp_h = hourly_median(map_nbp)

    result = pd.DataFrame(index=hour_index)
    result["hr_raw"] = hr_h.reindex(hour_index)

    map_vals  = abp_h.reindex(hour_index)
    map_src   = pd.Series(["abp" if v else None for v in map_vals.notna()], index=hour_index, dtype=object)
    nbp_fill  = nbp_h.reindex(hour_index)
    mask_fill = map_vals.isna() & nbp_fill.notna()
    map_vals[mask_fill] = nbp_fill[mask_fill]
    map_src[mask_fill]  = "nbp"

    result["map_raw"]    = map_vals
    result["map_source"] = map_src
    return result


def causal_trailing_mean(series: pd.Series, window: int) -> pd.Series:
    """
    因果拖尾均值: 每点仅用当前及之前 window 个实测值。
    对 NaN 格做前向传播（不改变 series 本身）。
    """
    filled = series.ffill()
    return filled.rolling(window=window, min_periods=1).mean()


def detrend_series(raw: pd.Series, trend_window: int = TREND_WINDOW) -> tuple[pd.Series, pd.Series]:
    """去趋势 + ±3σ 异常点替换，返回 (residual, is_interpolated)。"""
    # trailing mean
    trend = causal_trailing_mean(raw, trend_window)
    resid = raw - trend

    # ±3σ: 仅对实测点（raw 非 NaN）操作
    actual_mask = raw.notna()
    if actual_mask.sum() >= 3:
        mu    = resid[actual_mask].mean()
        sigma = resid[actual_mask].std()
        if sigma > 0:
            outlier = actual_mask & ((resid - mu).abs() > 3 * sigma)
            resid[outlier] = mu

    # 线性插值（最多 MAX_INTERP_GAP 连续 NaN）
    interp = resid.interpolate(method="linear", limit=MAX_INTERP_GAP)
    is_interp = (resid.isna() & interp.notna())

    return interp, is_interp


def process_patient(
    sid:    int,
    t0_min: float,
    hr_df:  pd.DataFrame,
    abp_df: pd.DataFrame,
    nbp_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    处理单个患者，返回 (hourly_df, diag_dict)。
    hour_index 覆盖 [-TOTAL_HOURS, 0] (相对 T0)。
    """
    # 时间范围（分钟）
    lo_min = t0_min - TOTAL_HOURS * 60
    hi_min = t0_min

    # 转换为 hour_idx (相对 T0 的整点)
    def to_hour_idx(df: pd.DataFrame, offset_col: str, value_col: str) -> pd.DataFrame:
        d = df[(df[offset_col] >= lo_min) & (df[offset_col] < hi_min)].copy()
        d["hour_idx"] = np.floor((d[offset_col] - t0_min) / 60).astype(int)
        return d.rename(columns={value_col: "value"})[["hour_idx", "value"]]

    hr_raw_p  = to_hour_idx(hr_df,  "observationoffset", "heartrate")
    abp_raw_p = to_hour_idx(abp_df, "observationoffset", "systemicmean")
    nbp_raw_p = to_hour_idx(nbp_df, "observationoffset", "noninvasivemean")

    hour_index = np.arange(-TOTAL_HOURS, 0, dtype=int)  # [-72, -71, ..., -1]
    grid = resample_hourly_eicu(hr_raw_p, abp_raw_p, nbp_raw_p, hour_index)

    # 生理范围过滤
    grid.loc[(grid["hr_raw"]  < HR_RANGE[0]) | (grid["hr_raw"]  > HR_RANGE[1]), "hr_raw"]  = np.nan
    grid.loc[(grid["map_raw"] < MAP_RANGE[0])| (grid["map_raw"] > MAP_RANGE[1]),"map_raw"] = np.nan

    # 去趋势
    hr_resid,  hr_interp  = detrend_series(grid["hr_raw"])
    map_resid, map_interp = detrend_series(grid["map_raw"])

    grid["hr_residual"]        = hr_resid
    grid["map_residual"]       = map_resid
    grid["hr_is_interpolated"] = hr_interp
    grid["map_is_interpolated"]= map_interp

    grid["patientunitstayid"]  = sid
    grid["T0_min"]             = t0_min
    grid["hours_before_T0"]    = grid.index.astype(float)  # hour_idx already relative

    # 诊断信息
    # 晚窗 [-12h, 0h) 实测点数（用于排除判断）
    late_mask    = (grid.index >= -12) & (grid.index < 0)
    n_actual_hr  = int((grid.loc[late_mask, "hr_raw"].notna() &
                        ~grid.loc[late_mask, "hr_is_interpolated"]).sum())
    # 48h 全窗口 MAP 统计（用于报告）
    analysis_mask = grid.index >= -EXTRACT_HOURS
    n_actual_map = int((grid.loc[analysis_mask, "map_raw"].notna() &
                        ~grid.loc[analysis_mask, "map_is_interpolated"]).sum())
    map_srcs = grid.loc[analysis_mask & grid["map_raw"].notna(), "map_source"].value_counts().to_dict()
    dominant_src = max(map_srcs, key=map_srcs.get) if map_srcs else "none"

    excluded = n_actual_hr < MIN_ACTUAL_LATE

    diag = {
        "patientunitstayid": sid,
        "T0_min":            t0_min,
        "n_actual_hr_late":  n_actual_hr,
        "n_actual_map_48h":  n_actual_map,
        "dominant_map_source": dominant_src,
        "excluded":          excluded,
    }

    return grid.reset_index().rename(columns={"index": "hour_idx"}), diag


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载队列 ───────────────────────────────────────────────────────────
    print("── 加载队列 ─────────────────────────────────────────────────────────")
    cohort = pd.read_parquet(COHORT_PATH)[["patientunitstayid", "T0_min"]].drop_duplicates()
    sids   = set(cohort["patientunitstayid"])
    print(f"  患者数: {len(sids):,}")

    # ── 2. 加载生命体征原始数据（仅读取队列患者）───────────────────────────
    print("── 加载 vitalPeriodic（HR + ABP）────────────────────────────────────")
    vp = pd.read_csv(
        EICU_DIR / "vitalPeriodic.csv.gz",
        usecols=["patientunitstayid", "observationoffset", "heartrate", "systemicmean"],
    )
    vp = vp[vp["patientunitstayid"].isin(sids)].copy()
    print(f"  记录数: {len(vp):,}")

    print("── 加载 vitalAperiodic（NBP）────────────────────────────────────────")
    va = pd.read_csv(
        EICU_DIR / "vitalAperiodic.csv.gz",
        usecols=["patientunitstayid", "observationoffset", "noninvasivemean"],
    )
    va = va[va["patientunitstayid"].isin(sids)].copy()
    print(f"  记录数: {len(va):,}")

    # ── 3. 逐患者处理 ─────────────────────────────────────────────────────────
    print("── 逐患者提取与清洗 ─────────────────────────────────────────────────")
    all_vitals = []
    all_diags  = []
    n_total    = len(cohort)

    hr_by_id  = {sid: grp for sid, grp in vp.groupby("patientunitstayid")}
    abp_by_id = {sid: grp for sid, grp in vp.groupby("patientunitstayid")}
    nbp_by_id = {sid: grp for sid, grp in va.groupby("patientunitstayid")}

    for i, (_, crow) in enumerate(cohort.iterrows(), 1):
        sid   = int(crow["patientunitstayid"])
        t0_m  = float(crow["T0_min"])

        hr_df  = hr_by_id.get(sid, pd.DataFrame(columns=["observationoffset","heartrate"]))
        abp_df = abp_by_id.get(sid, pd.DataFrame(columns=["observationoffset","systemicmean"]))
        nbp_df = nbp_by_id.get(sid, pd.DataFrame(columns=["observationoffset","noninvasivemean"]))

        grid, diag = process_patient(sid, t0_m, hr_df, abp_df, nbp_df)
        all_vitals.append(grid)
        all_diags.append(diag)

        if i % 200 == 0:
            print(f"  {i:,}/{n_total:,}...", flush=True)

    vitals_df = pd.concat(all_vitals, ignore_index=True)
    diag_df   = pd.DataFrame(all_diags)

    # ── 4. 统计与保存 ─────────────────────────────────────────────────────────
    n_excl = diag_df["excluded"].sum()
    print(f"\n  总处理: {n_total:,}  排除（HR实测点不足）: {n_excl:,}")
    print(f"  HR 晚窗中位实测点: {diag_df['n_actual_hr_late'].median():.0f}")
    print(f"  MAP dominant source: {diag_df['dominant_map_source'].value_counts().to_dict()}")

    vitals_df.to_parquet(VITALS_PATH, index=False)
    diag_df.to_parquet(DIAG_PATH, index=False)
    print(f"\n  → {VITALS_PATH}")
    print(f"  → {DIAG_PATH}")
    print(f"  vitals shape: {vitals_df.shape}")


if __name__ == "__main__":
    main()
