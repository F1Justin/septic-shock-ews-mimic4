"""
Step 1: 数据基础设施
- 创建 DuckDB schema 别名 (mimiciv_icu / mimiciv_hosp / mimiciv_derived)
- 将原始 CSV.gz 文件映射为 schema 内的 VIEW
- 按依赖顺序执行 sepsis3 最小 SQL 链 (~25 个 SQL)

用法: python scripts/01_setup_db.py
预计运行时间: 20-40 分钟 (受 chartevents.csv.gz ~30GB 制约)
"""

import sys
import time
import duckdb
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH      = PROJECT_ROOT / "mimiciv" / "mimiciv.db"
DATA_DIR     = PROJECT_ROOT / "mimiciv" / "3.1"
SQL_DIR      = PROJECT_ROOT / "mimic-code-ref" / "mimic-iv" / "concepts_duckdb"

# ── sepsis3 最小依赖链 (保持 duckdb.sql 中的原始顺序) ─────────────────────────
SEPSIS3_CHAIN = [
    # 依赖基础
    "demographics/icustay_times.sql",
    "demographics/icustay_hourly.sql",
    "demographics/weight_durations.sql",
    "measurement/urine_output.sql",
    "organfailure/kdigo_uo.sql",
    # 测量指标 (sofa 需要)
    "measurement/bg.sql",
    "measurement/chemistry.sql",
    "measurement/complete_blood_count.sql",
    "measurement/enzyme.sql",
    "measurement/gcs.sql",
    "measurement/urine_output_rate.sql",
    "measurement/ventilator_setting.sql",
    "measurement/oxygen_delivery.sql",
    "measurement/vitalsign.sql",
    # 药物 (sofa 心血管评分 + suspicion_of_infection)
    "medication/antibiotic.sql",
    "medication/dobutamine.sql",
    "medication/dopamine.sql",
    "medication/epinephrine.sql",
    "medication/norepinephrine.sql",
    "medication/phenylephrine.sql",
    "medication/vasopressin.sql",
    "medication/milrinone.sql",
    # 治疗 (sofa 呼吸评分需要 ventilation)
    "treatment/ventilation.sql",
    # 评分
    "score/sofa.sql",
    # 脓毒症定义
    "sepsis/suspicion_of_infection.sql",
    "sepsis/sepsis3.sql",
    # 副线: 升压药协变量
    "medication/vasoactive_agent.sql",
    "medication/norepinephrine_equivalent_dose.sql",
]

# ── 原始表映射 ─────────────────────────────────────────────────────────────────
# schema -> {table_name: relative_path_in_DATA_DIR}
RAW_TABLES = {
    "mimiciv_icu": {
        "caregiver":         "icu/caregiver.csv.gz",
        "chartevents":       "icu/chartevents.csv.gz",
        "d_items":           "icu/d_items.csv.gz",
        "datetimeevents":    "icu/datetimeevents.csv.gz",
        "icustays":          "icu/icustays.csv.gz",
        "ingredientevents":  "icu/ingredientevents.csv.gz",
        "inputevents":       "icu/inputevents.csv.gz",
        "outputevents":      "icu/outputevents.csv.gz",
        "procedureevents":   "icu/procedureevents.csv.gz",
    },
    "mimiciv_hosp": {
        "admissions":        "hosp/admissions.csv.gz",
        "d_hcpcs":           "hosp/d_hcpcs.csv.gz",
        "d_icd_diagnoses":   "hosp/d_icd_diagnoses.csv.gz",
        "d_icd_procedures":  "hosp/d_icd_procedures.csv.gz",
        "d_labitems":        "hosp/d_labitems.csv.gz",
        "diagnoses_icd":     "hosp/diagnoses_icd.csv.gz",
        "drgcodes":          "hosp/drgcodes.csv.gz",
        "emar":              "hosp/emar.csv.gz",
        "emar_detail":       "hosp/emar_detail.csv.gz",
        "hcpcsevents":       "hosp/hcpcsevents.csv.gz",
        "labevents":         "hosp/labevents.csv.gz",
        "microbiologyevents":"hosp/microbiologyevents.csv.gz",
        "omr":               "hosp/omr.csv.gz",
        "patients":          "hosp/patients.csv.gz",
        "pharmacy":          "hosp/pharmacy.csv.gz",
        "poe":               "hosp/poe.csv.gz",
        "poe_detail":        "hosp/poe_detail.csv.gz",
        "prescriptions":     "hosp/prescriptions.csv.gz",
        "procedures_icd":    "hosp/procedures_icd.csv.gz",
        "provider":          "hosp/provider.csv.gz",
        "services":          "hosp/services.csv.gz",
        "transfers":         "hosp/transfers.csv.gz",
    },
}


def setup_schemas(con: duckdb.DuckDBPyConnection) -> None:
    """创建三个 schema."""
    for schema in ("mimiciv_icu", "mimiciv_hosp", "mimiciv_derived"):
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    print("  [OK] schemas created")


def setup_raw_views(con: duckdb.DuckDBPyConnection) -> None:
    """为每个原始 CSV.gz 文件创建 schema-qualified VIEW."""
    for schema, tables in RAW_TABLES.items():
        for table, rel_path in tables.items():
            csv_path = DATA_DIR / rel_path
            if not csv_path.exists():
                print(f"  [WARN] missing: {csv_path} — skipping {schema}.{table}")
                continue
            # 使用绝对路径; timestampformat 让 DuckDB 自动检测时间戳
            sql = (
                f"CREATE OR REPLACE VIEW {schema}.{table} AS "
                f"SELECT * FROM read_csv_auto('{csv_path}', "
                f"header=true, compression='gzip');"
            )
            con.execute(sql)
    print(f"  [OK] raw views created")


CHECKPOINT_FILE = PROJECT_ROOT / "data" / ".setup_checkpoint"


def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        return set(CHECKPOINT_FILE.read_text().splitlines())
    return set()


def save_checkpoint(done: set) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text("\n".join(sorted(done)))


def run_sql_chain(con: duckdb.DuckDBPyConnection) -> None:
    """按依赖顺序执行 sepsis3 最小 SQL 链，支持断点续跑."""
    done = load_checkpoint()
    total = len(SEPSIS3_CHAIN)

    for i, rel_sql in enumerate(SEPSIS3_CHAIN, 1):
        if rel_sql in done:
            print(f"  [{i:02d}/{total}] {rel_sql} ... skipped (checkpoint)")
            continue

        sql_path = SQL_DIR / rel_sql
        if not sql_path.exists():
            print(f"  [ERROR] SQL file not found: {sql_path}")
            sys.exit(1)

        sql_text = sql_path.read_text()
        t0 = time.time()
        print(f"  [{i:02d}/{total}] {rel_sql} ...", end="", flush=True)
        try:
            con.execute(sql_text)
            elapsed = time.time() - t0
            print(f" done ({elapsed:.1f}s)")
            done.add(rel_sql)
            save_checkpoint(done)
        except Exception as e:
            print(f"\n  [ERROR] failed on {rel_sql}:\n  {e}")
            sys.exit(1)


def verify(con: duckdb.DuckDBPyConnection) -> None:
    """简单验证: sepsis3 表是否存在且有数据."""
    tables_to_check = [
        ("mimiciv_derived", "sepsis3"),
        ("mimiciv_derived", "sofa"),
        ("mimiciv_derived", "vasoactive_agent"),
        ("mimiciv_derived", "norepinephrine_equivalent_dose"),
    ]
    print("\n── 验证 ──────────────────────────────────────────────────────────")
    for schema, table in tables_to_check:
        try:
            n = con.execute(
                f"SELECT COUNT(*) FROM {schema}.{table}"
            ).fetchone()[0]
            print(f"  {schema}.{table}: {n:,} 行")
        except Exception as e:
            print(f"  [ERROR] {schema}.{table}: {e}")


def main() -> None:
    print(f"DuckDB: {DB_PATH}")
    print(f"Data:   {DATA_DIR}")
    print(f"SQL:    {SQL_DIR}\n")

    if not DATA_DIR.exists():
        print(f"[ERROR] data directory not found: {DATA_DIR}")
        sys.exit(1)
    if not SQL_DIR.exists():
        print(f"[ERROR] SQL directory not found: {SQL_DIR}")
        sys.exit(1)

    con = duckdb.connect(str(DB_PATH))
    # 允许 DuckDB 使用多线程加速
    con.execute("PRAGMA threads=8")

    print("── Step 1: 创建 schema ───────────────────────────────────────────")
    setup_schemas(con)

    print("\n── Step 2: 创建原始表 VIEW ──────────────────────────────────────")
    setup_raw_views(con)

    print("\n── Step 3: 执行 sepsis3 最小 SQL 链 ─────────────────────────────")
    t_start = time.time()
    run_sql_chain(con)
    elapsed_total = time.time() - t_start
    print(f"\n  总计: {elapsed_total/60:.1f} 分钟")

    verify(con)
    con.close()
    print("\n[完成] mimiciv.db 已就绪，可运行 02_build_cohort.py")


if __name__ == "__main__":
    main()
