"""
一次性提取：翻身事件相关患者的分钟级 HR 数据

从 mimiciv.db chartevents 提取队列患者（来自 perturbation_events.parquet）
每个 stay 的查询窗口为 [最早翻身时间 - 1h, T0 + POST_HOURS]，
涵盖翻身前基线及翻身后恢复期所需的 HR 记录（itemid=220045），
保存为 data/hr_minutelevel_perturb.parquet，供后续扰动分析复用。

注：数据提取结果（中位间隔 60min）表明 MIMIC-IV chartevents HR 为
护士手动记录（小时级），并非分钟级连续监测数据。

运行一次即可，之后分析脚本直接读 parquet。

预计运行时间: 60-120 秒（DuckDB 按 stay_id IN 过滤）
"""

import warnings
from pathlib import Path

import duckdb
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
NAAS_ROOT    = PROJECT_ROOT.parent
DB_PATH      = NAAS_ROOT / "mimiciv" / "mimiciv.db"
EVENTS_PATH  = NAAS_ROOT / "data" / "perturbation_events.parquet"
OUT_PATH     = NAAS_ROOT / "data" / "hr_minutelevel_perturb.parquet"

HR_ITEMID     = 220045
POST_HOURS    = 1     # T0 后缓冲小时数


def main() -> None:
    print("加载翻身事件 …")
    pe = pd.read_parquet(EVENTS_PATH)
    pe["turn_time"] = pd.to_datetime(pe["turn_time"])
    pe["T0"]        = pd.to_datetime(pe["T0"])

    # 每个 stay_id 的查询时间窗：[最早翻身时间 - 1h, T0 + POST_HOURS]
    # 1h 前置窗口足以覆盖翻身前基线（PRE_WIN_H=2h 时仍需向前看，
    # 但 perturbation_events 中 turn_time 本身已在 T0 前 48h 内，
    # 故 1h 余量是对最早翻身的保守缓冲，不是 T0-56h 的全量窗口）
    stay_windows = (
        pe.groupby("stay_id")
        .agg(
            t_min=("turn_time", lambda x: x.min() - pd.Timedelta(hours=1)),
            t_max=("T0",        lambda x: x.max() + pd.Timedelta(hours=POST_HOURS)),
        )
        .reset_index()
    )
    stay_ids = stay_windows["stay_id"].tolist()
    print(f"  涉及 stay_id: {len(stay_ids)}")

    # 构造 IN 列表字符串
    ids_str = ", ".join(str(s) for s in stay_ids)

    print(f"\n连接 DuckDB ({DB_PATH.name}) …")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("PRAGMA threads=4")

    # 注册时间窗 DataFrame
    con.register("stay_windows", stay_windows)

    print("提取分钟级 HR（itemid=220045）…")
    query = f"""
        SELECT
            ce.stay_id,
            ce.charttime,
            ce.valuenum AS hr
        FROM mimiciv_icu.chartevents ce
        INNER JOIN stay_windows sw ON ce.stay_id = sw.stay_id
        WHERE ce.itemid = {HR_ITEMID}
          AND ce.valuenum IS NOT NULL
          AND ce.valuenum BETWEEN 20 AND 300
          AND ce.charttime BETWEEN sw.t_min AND sw.t_max
        ORDER BY ce.stay_id, ce.charttime
    """
    df = con.execute(query).df()
    con.close()

    df["charttime"] = pd.to_datetime(df["charttime"])
    print(f"  提取行数: {len(df):,}")
    print(f"  stay_id 数: {df['stay_id'].nunique()}")

    # 抽样检查时间间隔
    sample_sid = df["stay_id"].iloc[0]
    sample = df[df["stay_id"] == sample_sid].sort_values("charttime")
    diffs = sample["charttime"].diff().dt.total_seconds().div(60).dropna()
    print(f"\n示例 stay {sample_sid} 时间间隔（分钟）：")
    print(f"  median={diffs.median():.1f}  mean={diffs.mean():.1f}  "
          f"min={diffs.min():.1f}  max={diffs.max():.1f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\n已保存 → {OUT_PATH}")
    print(f"文件大小: {OUT_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
