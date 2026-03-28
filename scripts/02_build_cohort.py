"""
Step 2: 队列构建 + 风险集匹配

本版修复:
  1. 休克定义增加液体复苏与升压药持续性约束，降低宽泛表型噪声
  2. 先按未来分析窗口的数据可分析性预筛，再进入 risk-set matching
  3. 为 72h 提取 + 24h burn-in 预留足够 ICU 观察长度

输出:
  data/cohort.parquet
  output/t0_window_sensitivity.csv
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "mimiciv" / "mimiciv.db"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cohort.parquet"
SENS_PATH = PROJECT_ROOT / "output" / "t0_window_sensitivity.csv"
FLUID_SENS_PATH = PROJECT_ROOT / "output" / "fluid_threshold_sensitivity.csv"

VASOPRESSOR_ITEMIDS = (221906, 221289, 221662, 221749, 222315)
LACTATE_ITEMIDS = (50813, 52442, 53154)
ISOTONIC_CRYSTALLOID_ITEMIDS = (225158, 225828, 225827)
MAP_ITEMIDS = (220052, 220181)
HR_ITEMID = 220045

LACTATE_THRESHOLD = 2.0
VASO_LACTATE_COEXIST_H = 24
FLUID_LOOKBACK_H = 24
MIN_CRYSTALLOID_ML = 1000.0
MIN_VASO_DURATION_H = 1

MATCH_RATIO = 2
SOFA_WINDOW = 2
RANDOM_SEED = 42

ANALYSIS_HOURS = 48
BURNIN_HOURS = 24
PREMATCH_OBS_HOURS = ANALYSIS_HOURS + BURNIN_HOURS
MIN_ACTUAL = 24

MAP_RANGE = (30.0, 200.0)
HR_RANGE = (20.0, 300.0)


def get_sepsis_stays(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Sepsis-3 ICU stays，附带 ICU 入出时间。"""
    return con.execute("""
        SELECT
            s.subject_id,
            s.stay_id,
            ie.hadm_id,
            ie.intime AS icu_intime,
            ie.outtime AS icu_outtime,
            DATEDIFF('hour', ie.intime, ie.outtime) AS icu_los_hours
        FROM mimiciv_derived.sepsis3 AS s
        INNER JOIN mimiciv_icu.icustays AS ie
            ON s.stay_id = ie.stay_id
        WHERE s.sepsis3 = TRUE
    """).df()


def compute_shock_T0(
    con: duckdb.DuckDBPyConnection,
    sepsis_df: pd.DataFrame,
    coexist_h: int = VASO_LACTATE_COEXIST_H,
    fluid_min_ml: float = MIN_CRYSTALLOID_ML,
) -> pd.DataFrame:
    """
    更严格的操作性休克定义:
      - ICU 内首次持续 >= 1h 的升压药开始
      - 该时间点前后 coexist_h 小时内有 lactate > 2
      - 升压药前 24h 内累计等渗晶体液 >= fluid_min_ml
    """
    vaso_ids = ",".join(map(str, VASOPRESSOR_ITEMIDS))
    lac_ids = ",".join(map(str, LACTATE_ITEMIDS))
    fluid_ids = ",".join(map(str, ISOTONIC_CRYSTALLOID_ITEMIDS))

    con.register(
        "sepsis_tmp",
        sepsis_df[["stay_id", "hadm_id", "icu_intime", "icu_outtime"]].copy(),
    )

    return con.execute(f"""
        WITH vaso_starts AS (
            SELECT
                ie.stay_id,
                ie.starttime AS vaso_time,
                ie.endtime AS vaso_endtime
            FROM mimiciv_icu.inputevents AS ie
            INNER JOIN sepsis_tmp AS s
                ON ie.stay_id = s.stay_id
            WHERE ie.itemid IN ({vaso_ids})
              AND COALESCE(ie.amount, 0) > 0
              AND ie.starttime BETWEEN s.icu_intime AND s.icu_outtime
              AND ie.endtime >= ie.starttime + INTERVAL '{MIN_VASO_DURATION_H}' HOUR
        ),
        qualifying AS (
            SELECT
                vs.stay_id,
                vs.vaso_time
            FROM vaso_starts AS vs
            INNER JOIN sepsis_tmp AS s
                ON vs.stay_id = s.stay_id
            WHERE EXISTS (
                SELECT 1
                FROM mimiciv_hosp.labevents AS le
                WHERE le.hadm_id = s.hadm_id
                  AND le.itemid IN ({lac_ids})
                  AND le.valuenum > {LACTATE_THRESHOLD}
                  AND le.charttime BETWEEN
                        vs.vaso_time - INTERVAL '{coexist_h}' HOUR
                    AND vs.vaso_time + INTERVAL '{coexist_h}' HOUR
            )
              AND (
                SELECT COALESCE(SUM(COALESCE(ie.totalamount, ie.amount)), 0)
                FROM mimiciv_icu.inputevents AS ie
                WHERE ie.stay_id = vs.stay_id
                  AND ie.itemid IN ({fluid_ids})
                  AND COALESCE(ie.amount, 0) > 0
                  AND ie.starttime BETWEEN
                        vs.vaso_time - INTERVAL '{FLUID_LOOKBACK_H}' HOUR
                    AND vs.vaso_time
              ) >= {fluid_min_ml}
        )
        SELECT stay_id, MIN(vaso_time) AS T0
        FROM qualifying
        GROUP BY stay_id
    """).df()


def cohort_sensitivity_by_window(
    con: duckdb.DuckDBPyConnection,
    sepsis_df: pd.DataFrame,
    windows: list[int],
) -> pd.DataFrame:
    rows = []
    for h in windows:
        shock_t0 = compute_shock_T0(con, sepsis_df, coexist_h=h)
        rows.append(
            {
                "coexist_window_hours": h,
                "shock_stays": int(len(shock_t0)),
            }
        )
    return pd.DataFrame(rows)


def cohort_sensitivity_by_fluid(
    con: duckdb.DuckDBPyConnection,
    sepsis_df: pd.DataFrame,
    thresholds_ml: list[float],
) -> pd.DataFrame:
    rows = []
    for thr in thresholds_ml:
        shock_t0 = compute_shock_T0(con, sepsis_df, fluid_min_ml=thr)
        rows.append(
            {
                "fluid_threshold_ml": float(thr),
                "shock_stays": int(len(shock_t0)),
            }
        )
    return pd.DataFrame(rows)


def get_admission_sofa(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """入 ICU 前 24h 内最高 SOFA 作为 admission SOFA。"""
    return con.execute("""
        SELECT
            s.stay_id,
            MAX(s.sofa_24hours) AS sofa_admission
        FROM mimiciv_derived.sofa AS s
        INNER JOIN mimiciv_icu.icustays AS ie
            ON s.stay_id = ie.stay_id
        WHERE s.endtime <= ie.intime + INTERVAL '24' HOUR
        GROUP BY s.stay_id
    """).df()


def build_hourly_vitals_coverage(
    con: duckdb.DuckDBPyConnection,
    sepsis_df: pd.DataFrame,
) -> pd.DataFrame:
    """按 stay_id + 整点小时聚合 MAP/HR 是否有实测值。"""
    map_ids = ",".join(map(str, MAP_ITEMIDS))
    con.register("stay_ids_tmp", sepsis_df[["stay_id"]].drop_duplicates())
    return con.execute(f"""
        SELECT
            ce.stay_id,
            DATE_TRUNC('hour', ce.charttime) AS chart_hour,
            MAX(
                CASE
                    WHEN ce.itemid IN ({map_ids})
                     AND ce.valuenum BETWEEN {MAP_RANGE[0]} AND {MAP_RANGE[1]}
                    THEN 1 ELSE 0
                END
            ) AS has_map,
            MAX(
                CASE
                    WHEN ce.itemid = {HR_ITEMID}
                     AND ce.valuenum BETWEEN {HR_RANGE[0]} AND {HR_RANGE[1]}
                    THEN 1 ELSE 0
                END
            ) AS has_hr
        FROM mimiciv_icu.chartevents AS ce
        INNER JOIN stay_ids_tmp AS s
            ON ce.stay_id = s.stay_id
        WHERE ce.itemid IN ({map_ids}, {HR_ITEMID})
          AND ce.valuenum IS NOT NULL
        GROUP BY ce.stay_id, DATE_TRUNC('hour', ce.charttime)
    """).df()


def prescreen_data_quality(
    stay_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    window_hours: int = ANALYSIS_HOURS,
    min_actual: int = MIN_ACTUAL,
) -> dict[int, set[pd.Timestamp]]:
    """
    为每个 stay 预先计算“哪些整点 end_hour 的前 48h 同时满足
    MAP/HR 各 >= min_actual 实测小时”。
    """
    stay_meta = (
        stay_df[["stay_id", "icu_intime", "icu_outtime"]]
        .drop_duplicates("stay_id")
        .copy()
    )
    stay_meta["icu_intime"] = pd.to_datetime(stay_meta["icu_intime"])
    stay_meta["icu_outtime"] = pd.to_datetime(stay_meta["icu_outtime"])

    hourly_df = hourly_df.copy()
    hourly_df["chart_hour"] = pd.to_datetime(hourly_df["chart_hour"])
    hourly_df["has_map"] = hourly_df["has_map"].astype(int)
    hourly_df["has_hr"] = hourly_df["has_hr"].astype(int)
    grouped = {sid: grp.set_index("chart_hour") for sid, grp in hourly_df.groupby("stay_id")}

    eligible_end_hours: dict[int, set[pd.Timestamp]] = {}
    for row in stay_meta.itertuples(index=False):
        dense_hours = pd.date_range(
            start=pd.Timestamp(row.icu_intime).floor("h"),
            end=pd.Timestamp(row.icu_outtime).floor("h"),
            freq="h",
        )
        if len(dense_hours) < window_hours:
            eligible_end_hours[row.stay_id] = set()
            continue

        raw = grouped.get(row.stay_id)
        if raw is None:
            eligible_end_hours[row.stay_id] = set()
            continue

        has_map = raw["has_map"].reindex(dense_hours, fill_value=0)
        has_hr = raw["has_hr"].reindex(dense_hours, fill_value=0)
        map_roll = has_map.rolling(window_hours, min_periods=window_hours).sum()
        hr_roll = has_hr.rolling(window_hours, min_periods=window_hours).sum()
        mask = (map_roll >= min_actual) & (hr_roll >= min_actual)
        eligible_end_hours[row.stay_id] = set(dense_hours[mask.to_numpy()])

    return eligible_end_hours


def window_is_eligible(
    stay_id: int,
    t0: pd.Timestamp,
    eligible_end_hours: dict[int, set[pd.Timestamp]],
) -> bool:
    """给定 T0，检查其 floor 到整点后是否通过 48h 数据质量预筛。"""
    return pd.Timestamp(t0).floor("h") in eligible_end_hours.get(stay_id, set())


def risk_set_match(
    shock_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    eligible_end_hours: dict[int, set[pd.Timestamp]],
) -> pd.DataFrame:
    """
    先以数据可分析性筛过候选控制，再执行 risk-set matching。
    """
    rng = np.random.default_rng(RANDOM_SEED)
    records = []
    unmatched = 0
    no_data_match = 0

    pool_df = pool_df.copy()
    pool_df["icu_intime"] = pd.to_datetime(pool_df["icu_intime"])
    pool_df["T0"] = pd.to_datetime(pool_df["T0"])
    pool_df["shock_hour"] = (
        (pool_df["T0"] - pool_df["icu_intime"]).dt.total_seconds() / 3600
    )

    for pair_id, row in enumerate(shock_df.itertuples(index=False), start=1):
        H = (pd.Timestamp(row.T0) - pd.Timestamp(row.icu_intime)).total_seconds() / 3600
        sofa = row.sofa_admission
        case_end_hour = pd.Timestamp(row.T0).floor("h")

        if H < PREMATCH_OBS_HOURS or case_end_hour not in eligible_end_hours.get(row.stay_id, set()):
            no_data_match += 1
            continue

        eligible = pool_df[
            (pool_df["stay_id"] != row.stay_id)
            & (pool_df["icu_los_hours"] >= H)
            & (
                pool_df["shock_hour"].isna()
                | (pool_df["shock_hour"] > H)
            )
        ].copy()

        if pd.notna(sofa) and len(eligible) > 0:
            sofa_matched = eligible[
                eligible["sofa_admission"].between(sofa - SOFA_WINDOW, sofa + SOFA_WINDOW)
            ]
            if len(sofa_matched) > 0:
                eligible = sofa_matched

        if len(eligible) == 0:
            unmatched += 1
            continue

        eligible["pseudo_T0"] = eligible["icu_intime"] + pd.to_timedelta(H, unit="h")
        eligible = eligible[
            [
                window_is_eligible(sid, t0, eligible_end_hours)
                for sid, t0 in zip(eligible["stay_id"], eligible["pseudo_T0"])
            ]
        ]

        if len(eligible) == 0:
            unmatched += 1
            continue

        n_sample = min(MATCH_RATIO, len(eligible))
        sampled = eligible.sample(
            n=n_sample,
            random_state=int(rng.integers(0, 2**31)),
        )

        records.append(
            {
                "subject_id": row.subject_id,
                "stay_id": row.stay_id,
                "group": "shock",
                "T0": row.T0,
                "icu_intime": row.icu_intime,
                "sofa_admission": sofa,
                "matched_pair_id": pair_id,
            }
        )

        for ctrl in sampled.itertuples(index=False):
            records.append(
                {
                    "subject_id": ctrl.subject_id,
                    "stay_id": ctrl.stay_id,
                    "group": "control",
                    "T0": ctrl.pseudo_T0,
                    "icu_intime": ctrl.icu_intime,
                    "sofa_admission": ctrl.sofa_admission,
                    "matched_pair_id": pair_id,
                }
            )

    print(f"  shock 因自身窗口不满足预筛而排除: {no_data_match:,}")
    print(f"  shock 未匹配到对照: {unmatched:,}")
    return pd.DataFrame(records)


def main() -> None:
    print(f"DB: {DB_PATH}\n")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("── 提取基础队列 ───────────────────────────────────────────────────")
    print("  sepsis stays ...", end="", flush=True)
    sepsis = get_sepsis_stays(con)
    print(f" {len(sepsis):,} stays")

    print("  stricter shock T0 ...", end="", flush=True)
    shock_t0 = compute_shock_T0(con, sepsis)
    print(f" {len(shock_t0):,} shock stays with valid T0")

    sens = cohort_sensitivity_by_window(con, sepsis, windows=[6, 12, 24])
    fluid_sens = cohort_sensitivity_by_fluid(con, sepsis, thresholds_ml=[1000, 1500, 2000, 2400, 3000])
    SENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    sens.to_csv(SENS_PATH, index=False, encoding="utf-8-sig")
    fluid_sens.to_csv(FLUID_SENS_PATH, index=False, encoding="utf-8-sig")
    print(
        "  T0 window sensitivity ... "
        + ", ".join(
            f"±{int(r.coexist_window_hours)}h={int(r.shock_stays):,}"
            for r in sens.itertuples(index=False)
        )
    )
    print(
        "  fluid threshold sensitivity ... "
        + ", ".join(
            f"{int(r.fluid_threshold_ml)}mL={int(r.shock_stays):,}"
            for r in fluid_sens.itertuples(index=False)
        )
    )

    print("  admission SOFA ...", end="", flush=True)
    sofa_adm = get_admission_sofa(con)
    print(f" {len(sofa_adm):,} stays")

    print("  hourly coverage prescreen ...", end="", flush=True)
    hourly_coverage = build_hourly_vitals_coverage(con, sepsis)
    print(f" {len(hourly_coverage):,} stay-hour rows")
    con.close()

    print("\n── 合并 & 标记休克 ───────────────────────────────────────────────")
    df = sepsis.merge(sofa_adm, on="stay_id", how="left")
    df = df.merge(shock_t0, on="stay_id", how="left")
    df["icu_intime"] = pd.to_datetime(df["icu_intime"])
    df["icu_outtime"] = pd.to_datetime(df["icu_outtime"])
    df["T0"] = pd.to_datetime(df["T0"])

    t0_in_icu = df["T0"].between(df["icu_intime"], df["icu_outtime"])
    df["is_shock"] = df["T0"].notna() & t0_in_icu
    df["hours_to_t0"] = (
        (df["T0"] - df["icu_intime"]).dt.total_seconds() / 3600
    )

    print(f"  脓毒症总 stays        : {len(df):,}")
    print(f"  休克 stays            : {int(df['is_shock'].sum()):,}")
    print(f"  非休克 stays          : {int((~df['is_shock']).sum()):,}")

    print("\n── 预筛数据可分析性（先筛后匹配）──────────────────────────────────")
    eligible_end_hours = prescreen_data_quality(df, hourly_coverage)
    df["has_any_eligible_window"] = df["stay_id"].map(
        lambda sid: bool(eligible_end_hours.get(sid))
    )

    shock_df = df[df["is_shock"]].copy()
    shock_df = shock_df[shock_df["hours_to_t0"] >= PREMATCH_OBS_HOURS].copy()
    shock_df = shock_df[
        [
            window_is_eligible(sid, t0, eligible_end_hours)
            for sid, t0 in zip(shock_df["stay_id"], shock_df["T0"])
        ]
    ].copy()
    pool_df = df[(df["icu_los_hours"] >= PREMATCH_OBS_HOURS) & df["has_any_eligible_window"]].copy()

    print(f"  满足 72h 观察长度的 shock stays : {len(shock_df):,}")
    print(f"  可进入风险集的 pool stays      : {len(pool_df):,}")

    print("\n── 风险集匹配 (1:2, SOFA ±2) ────────────────────────────────────")
    cohort = risk_set_match(shock_df, pool_df, eligible_end_hours)

    n_shock = int((cohort["group"] == "shock").sum())
    n_control = int((cohort["group"] == "control").sum())
    n_pairs = int(cohort["matched_pair_id"].nunique())
    ctrl_counts = cohort[cohort["group"] == "control"].groupby("stay_id").size()

    print(f"  匹配后 shock          : {n_shock:,}")
    print(
        f"  匹配后 control        : {n_control:,}  "
        f"(唯一 stay_id: {ctrl_counts.index.nunique():,}，最多重复 {int(ctrl_counts.max()) if len(ctrl_counts) else 0} 次)"
    )
    print(f"  配对数                : {n_pairs:,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cohort[
        [
            "subject_id",
            "stay_id",
            "group",
            "T0",
            "icu_intime",
            "sofa_admission",
            "matched_pair_id",
        ]
    ].to_parquet(OUTPUT_PATH, index=False)
    print(f"\n[完成] 已保存 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
