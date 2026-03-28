"""
Step 6 (T1.1): 基线对比表 — analyzable matched cohort 内 Shock vs Control

数据来源
--------
  cleaning_diagnostics.parquet  → excluded 标记, dominant_source
  cohort.parquet                → subject_id, sofa_admission, group
  mimiciv_hosp.patients         → anchor_age, gender
  mimiciv_hosp.admissions       → race, hospital_expire_flag, insurance
  mimiciv_icu.icustays          → first_careunit, los
  mimiciv_derived.ventilation   → T0 前机械通气
  mimiciv_derived.norepinephrine_equivalent_dose → T0 前升压药

统计方法
--------
  连续变量 : Shapiro-Wilk 判断正态性 (N≤5000 子样本) →
            正态→ Welch t, 否则 → Wilcoxon 秩和检验
            汇报 median [IQR]
  分类变量 : Chi-square (期望频数 < 5 时用 Fisher exact)
            汇报 N (%)
  SMD      : 连续变量 Cohen's d (pooled SD),
            二分变量 Austin-Stuart formula,
            多分类变量: 按频率最高的虚拟编码 pooled SMD

输出
----
  output/table1_baseline.csv  — 机器可读
  output/table1_baseline.html — 人类可读
  logs/06_baseline_table.log  — 运行日志

用法: python scripts/06_baseline_table.py
"""

import sys
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH      = PROJECT_ROOT / "mimiciv" / "mimiciv.db"
LOG_DIR      = PROJECT_ROOT / "logs"
OUTPUT_DIR   = PROJECT_ROOT / "output"

LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "06_baseline_table.log"

# ── 日志 ──────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    log = logging.getLogger("baseline")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(logging.Formatter(fmt))
        log.addHandler(h)
    return log


log = setup_logging()

# ── Race 合并映射 ──────────────────────────────────────────────────────────────
RACE_MAP: dict[str, str] = {
    "WHITE": "White",
    "WHITE - OTHER EUROPEAN": "White",
    "WHITE - RUSSIAN": "White",
    "WHITE - BRAZILIAN": "White",
    "WHITE - EASTERN EUROPEAN": "White",
    "BLACK/AFRICAN AMERICAN": "Black",
    "BLACK/AFRICAN": "Black",
    "BLACK/CAPE VERDEAN": "Black",
    "BLACK/CARIBBEAN ISLAND": "Black",
    "HISPANIC OR LATINO": "Hispanic",
    "HISPANIC/LATINO - PUERTO RICAN": "Hispanic",
    "HISPANIC/LATINO - DOMINICAN": "Hispanic",
    "HISPANIC/LATINO - GUATEMALAN": "Hispanic",
    "HISPANIC/LATINO - CUBAN": "Hispanic",
    "HISPANIC/LATINO - SALVADORAN": "Hispanic",
    "HISPANIC/LATINO - CENTRAL AMERICAN": "Hispanic",
    "HISPANIC/LATINO - MEXICAN": "Hispanic",
    "HISPANIC/LATINO - COLOMBIAN": "Hispanic",
    "HISPANIC/LATINO - HONDURAN": "Hispanic",
    "ASIAN": "Asian",
    "ASIAN - CHINESE": "Asian",
    "ASIAN - SOUTH EAST ASIAN": "Asian",
    "ASIAN - KOREAN": "Asian",
    "ASIAN - ASIAN INDIAN": "Asian",
    "UNKNOWN": "Unknown/Other",
    "UNABLE TO OBTAIN": "Unknown/Other",
    "PATIENT DECLINED TO ANSWER": "Unknown/Other",
}


def map_race(r: str) -> str:
    if pd.isna(r):
        return "Unknown/Other"
    r_upper = str(r).upper().strip()
    for key, val in RACE_MAP.items():
        if r_upper.startswith(key):
            return val
    return "Other"


# ── ICU type 简化 ──────────────────────────────────────────────────────────────
def simplify_careunit(cu: str) -> str:
    if pd.isna(cu):
        return "Other"
    cu = str(cu).upper()
    if "MICU" in cu or "MEDICAL" in cu:
        return "MICU"
    if "SICU" in cu or "SURGICAL" in cu or "TSICU" in cu:
        return "SICU"
    if "CCU" in cu or "CARDIAC" in cu or "CSRU" in cu or "CVICU" in cu:
        return "CCU/CSRU"
    if "NICU" in cu or "NEURO" in cu:
        return "Neuro/NICU"
    return "Other"


# ── 统计工具函数 ───────────────────────────────────────────────────────────────

def smd_continuous(x1: np.ndarray, x2: np.ndarray) -> float:
    """Cohen's d with pooled SD (ignores NaN)."""
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]
    if len(x1) < 2 or len(x2) < 2:
        return np.nan
    pooled_sd = np.sqrt((np.var(x1, ddof=1) + np.var(x2, ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0
    return (np.mean(x1) - np.mean(x2)) / pooled_sd


def smd_binary(p1: float, p2: float) -> float:
    """Austin-Stuart (2015) SMD for binary proportions."""
    denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    if denom == 0:
        return 0.0
    return (p1 - p2) / denom


def test_normality(x: np.ndarray, max_n: int = 5000) -> bool:
    """
    True if Shapiro-Wilk cannot reject normality (p >= 0.05) on a random sample.
    For very large N, always returns False (use non-parametric).
    """
    x = x[~np.isnan(x)]
    if len(x) < 8:
        return False
    if len(x) > 100_000:
        return False
    sample = x if len(x) <= max_n else np.random.default_rng(42).choice(
        x, size=max_n, replace=False
    )
    _, p = stats.shapiro(sample)
    return p >= 0.05


def test_continuous(x1: np.ndarray, x2: np.ndarray) -> tuple[float, str]:
    """
    Returns (p_value, test_name).
    Chooses Welch t if both groups appear normal, else Wilcoxon (Mann-Whitney U).
    """
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]
    if len(x1) < 3 or len(x2) < 3:
        return np.nan, "n/a"
    normal = test_normality(x1) and test_normality(x2)
    if normal:
        _, p = stats.ttest_ind(x1, x2, equal_var=False)
        return p, "Welch t"
    else:
        _, p = stats.mannwhitneyu(x1, x2, alternative="two-sided")
        return p, "Wilcoxon"


def test_categorical(counts1: np.ndarray, counts2: np.ndarray) -> tuple[float, str]:
    """
    Chi-square test on contingency table; falls back to Fisher exact for 2×2 with
    expected cell < 5.
    """
    table = np.array([counts1, counts2])
    # check expected frequencies
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    if dof == 1 and (expected < 5).any():
        _, p = stats.fisher_exact(table)
        return p, "Fisher"
    return p, "Chi-sq"


def format_p(p: float) -> str:
    if np.isnan(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def format_smd(s: float) -> str:
    if np.isnan(s):
        return "—"
    return f"{s:.3f}"


# ── 数据加载与分组 ────────────────────────────────────────────────────────────

def load_and_group() -> pd.DataFrame:
    """加载 analyzable matched cohort，构建每个 (stay_id, T0) 一行的主表。"""
    diag   = pd.read_parquet(PROJECT_ROOT / "data" / "cleaning_diagnostics.parquet")
    cohort = pd.read_parquet(PROJECT_ROOT / "data" / "cohort.parquet")

    # 去除 cohort 中的重复 (stay_id, T0) (16 行, 不影响主分析)
    cohort = cohort.drop_duplicates(["stay_id", "T0"])

    log.info(f"cleaning_diagnostics: {len(diag):,} rows; "
             f"analysable={int((~diag['excluded']).sum()):,}, "
             f"excluded={int(diag['excluded'].sum()):,}")
    log.info(f"cohort (deduped): {len(cohort):,} rows")

    diag = diag.drop_duplicates(["stay_id", "T0"])
    analyzable = (
        cohort
        .merge(
            diag[["stay_id", "T0", "excluded", "dominant_source"]],
            on=["stay_id", "T0"],
            how="inner",
        )
        .loc[lambda d: ~d["excluded"]]
        .copy()
    )
    shock_n = int((analyzable["group"] == "shock").sum())
    control_n = int((analyzable["group"] == "control").sum())
    log.info(
        f"Analyzable matched cohort → shock={shock_n:,}, "
        f"control={control_n:,}, total={len(analyzable):,}"
    )
    return analyzable


# ── DuckDB 查询 ───────────────────────────────────────────────────────────────

def pull_demographics(con: duckdb.DuckDBPyConnection,
                      base: pd.DataFrame) -> pd.DataFrame:
    """年龄, 性别, 种族, 住院死亡率, 保险, ICU 类型, ICU LOS."""
    con.register("_base", base[["stay_id", "subject_id"]].drop_duplicates())
    df = con.execute("""
        SELECT
            ie.stay_id,
            p.anchor_age            AS age,
            p.gender,
            a.race,
            a.hospital_expire_flag  AS hospital_mortality,
            a.insurance,
            ie.first_careunit,
            ie.los                  AS icu_los_days
        FROM _base b
        INNER JOIN mimiciv_icu.icustays        ie ON ie.stay_id    = b.stay_id
        INNER JOIN mimiciv_hosp.patients       p  ON p.subject_id  = b.subject_id
        INNER JOIN mimiciv_hosp.admissions     a  ON a.hadm_id     = ie.hadm_id
    """).df()
    log.info(f"Demographics: {len(df):,} rows returned")
    return df


def pull_ventilation(con: duckdb.DuckDBPyConnection,
                     stay_t0: pd.DataFrame) -> pd.DataFrame:
    """T0 前是否接受呼吸支持 (InvasiveVent / NonInvasiveVent / HFNC)."""
    con.register("_st0", stay_t0[["stay_id", "T0"]])
    df = con.execute("""
        SELECT DISTINCT s.stay_id, s.T0, 1 AS vent_before_t0
        FROM mimiciv_derived.ventilation v
        INNER JOIN _st0 s ON v.stay_id = s.stay_id
        WHERE v.starttime < s.T0
          AND v.ventilation_status IN ('InvasiveVent', 'NonInvasiveVent', 'HFNC')
    """).df()
    log.info(f"Respiratory support: {len(df):,} stay-T0 pairs before T0")
    return df


def pull_vasopressor(con: duckdb.DuckDBPyConnection,
                     stay_t0: pd.DataFrame) -> pd.DataFrame:
    """T0 前是否有升压药 (norepinephrine_equivalent_dose > 0)."""
    con.register("_st0v", stay_t0[["stay_id", "T0"]])
    df = con.execute("""
        SELECT DISTINCT s.stay_id, s.T0, 1 AS vaso_before_t0
        FROM mimiciv_derived.norepinephrine_equivalent_dose n
        INNER JOIN _st0v s ON n.stay_id = s.stay_id
        WHERE n.starttime < s.T0
          AND n.norepinephrine_equivalent_dose > 0
    """).df()
    log.info(f"Vasopressor: {len(df):,} stay-T0 pairs with vasopressor before T0")
    return df


# ── 主表构建 ──────────────────────────────────────────────────────────────────

def build_master_table(base: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        demog = pull_demographics(con, base)
        vent  = pull_ventilation(con,  base[["stay_id", "T0"]])
        vaso  = pull_vasopressor(con,  base[["stay_id", "T0"]])
    finally:
        con.close()

    master = (
        base
        .merge(demog, on="stay_id", how="left")
        .merge(vent,  on=["stay_id", "T0"], how="left")
        .merge(vaso,  on=["stay_id", "T0"], how="left")
    )
    master["vent_before_t0"] = master["vent_before_t0"].fillna(0).astype(int)
    master["vaso_before_t0"] = master["vaso_before_t0"].fillna(0).astype(int)

    # 派生变量
    master["race_group"]       = master["race"].apply(map_race)
    master["icu_type"]         = master["first_careunit"].apply(simplify_careunit)
    master["hospital_mortality"] = master["hospital_mortality"].fillna(0).astype(int)
    master["male"]             = (master["gender"] == "M").astype(int)
    master["monitoring_mode"]  = master["dominant_source"].str.upper().fillna("UNKNOWN")

    log.info(f"Master table: {len(master):,} rows, {master.columns.tolist()}")
    return master


# ── Table 1 统计生成 ───────────────────────────────────────────────────────────

def build_table1(master: pd.DataFrame) -> pd.DataFrame:
    """
    构建 Table 1: 每个变量一行, 列: Variable, Shock, Control, P-value, SMD, Test.
    """
    g_s = master[master["group"] == "shock"]
    g_c = master[master["group"] == "control"]
    N_s = len(g_s)
    N_c = len(g_c)

    log.info(f"Table 1 groups — Shock N={N_s:,}, Control N={N_c:,}")

    rows: list[dict] = []

    def add_row(var: str, val_s: str, val_c: str,
                p: float, smd: float, test: str) -> None:
        rows.append({
            "Variable":   var,
            f"Shock (N={N_s:,})": val_s,
            f"Control (N={N_c:,})":   val_c,
            "P-value":    format_p(p),
            "SMD":        format_smd(smd),
            "Test":       test,
        })

    # ── 连续变量 ─────────────────────────────────────────────
    def add_continuous(col: str, label: str, decimals: int = 1) -> None:
        x1 = g_s[col].dropna().values.astype(float)
        x2 = g_c[col].dropna().values.astype(float)
        p, test = test_continuous(x1, x2)
        s = smd_continuous(x1, x2)

        def fmt(x: np.ndarray) -> str:
            med = np.median(x)
            q1  = np.percentile(x, 25)
            q3  = np.percentile(x, 75)
            fmt_str = f"{{:.{decimals}f}}"
            return (f"{fmt_str.format(med)} "
                    f"[{fmt_str.format(q1)}–{fmt_str.format(q3)}]")

        log.info(f"  {label}: shock median={np.median(x1):.2f}, "
                 f"control median={np.median(x2):.2f}, p={format_p(p)}, SMD={format_smd(s)}")
        add_row(label, fmt(x1), fmt(x2), p, s, test)

    add_continuous("age",           "Age, years")
    add_continuous("sofa_admission","Admission SOFA score", decimals=0)
    add_continuous("icu_los_days",  "ICU LOS, days", decimals=1)

    # ── 二分变量 ─────────────────────────────────────────────
    def add_binary(col: str, label: str, positive_val=1) -> None:
        x1 = (g_s[col] == positive_val).astype(int)
        x2 = (g_c[col] == positive_val).astype(int)
        n1, n2 = x1.sum(), x2.sum()
        p1, p2 = x1.mean(), x2.mean()
        counts1 = np.array([n1, N_s - n1])
        counts2 = np.array([n2, N_c - n2])
        p, test = test_categorical(counts1, counts2)
        s = smd_binary(p1, p2)
        log.info(f"  {label}: shock={p1*100:.1f}%, control={p2*100:.1f}%, "
                 f"p={format_p(p)}, SMD={format_smd(s)}")
        add_row(
            label,
            f"{n1:,} ({p1*100:.1f}%)",
            f"{n2:,} ({p2*100:.1f}%)",
            p, s, test,
        )

    add_binary("male",             "Male sex")
    add_binary("hospital_mortality", "Hospital mortality")
    add_binary("vent_before_t0",   "Respiratory support before T0")
    add_binary("vaso_before_t0",   "Vasopressor use before T0")

    # ── 多分类变量辅助 ────────────────────────────────────────
    def add_categorical_header(label: str) -> None:
        rows.append({
            "Variable": label,
            f"Shock (N={N_s:,})": "",
            f"Control (N={N_c:,})":   "",
            "P-value": "",
            "SMD":     "",
            "Test":    "",
        })

    def add_categorical(col: str, label: str, order: list[str] | None = None) -> None:
        cats_a = g_s[col].value_counts()
        cats_e = g_c[col].value_counts()
        all_cats = sorted(set(cats_a.index) | set(cats_e.index))
        if order:
            all_cats = [c for c in order if c in all_cats] + \
                       [c for c in all_cats if c not in order]

        # Chi-square on full contingency table
        counts_a = np.array([cats_a.get(c, 0) for c in all_cats])
        counts_e = np.array([cats_e.get(c, 0) for c in all_cats])
        try:
            p, test = test_categorical(counts_a, counts_e)
        except Exception:
            p, test = np.nan, "n/a"

        # pooled SMD (multivariate, reference = first category)
        if len(all_cats) > 1:
            smd_vals = []
            for c in all_cats[1:]:
                p1 = cats_a.get(c, 0) / N_s
                p2 = cats_e.get(c, 0) / N_c
                smd_vals.append(smd_binary(p1, p2) ** 2)
            pooled_smd = np.sqrt(np.mean(smd_vals)) if smd_vals else np.nan
        else:
            pooled_smd = np.nan

        add_categorical_header(f"{label} [Chi-sq p={format_p(p)}, SMD={format_smd(pooled_smd)}]")
        for i, c in enumerate(all_cats):
            n1 = cats_a.get(c, 0)
            n2 = cats_e.get(c, 0)
            p1 = n1 / N_s
            p2 = n2 / N_c
            s  = smd_binary(p1, p2)
            log.info(f"    {label}/{c}: shock={p1*100:.1f}%, control={p2*100:.1f}%, "
                     f"per-cat SMD={format_smd(s)}")
            add_row(
                f"  {c}",
                f"{n1:,} ({p1*100:.1f}%)",
                f"{n2:,} ({p2*100:.1f}%)",
                p if i == 0 else np.nan,   # only show p on first sub-row
                s,
                test if i == 0 else "",
            )

    add_categorical(
        "race_group", "Race/ethnicity",
        order=["White", "Black", "Hispanic", "Asian", "Other", "Unknown/Other"],
    )

    add_categorical(
        "icu_type", "ICU type (first care unit)",
        order=["MICU", "SICU", "CCU/CSRU", "Neuro/NICU", "Other"],
    )

    add_categorical(
        "monitoring_mode", "Monitoring modality",
        order=["ABP", "NBP", "MIXED", "UNKNOWN"],
    )

    return pd.DataFrame(rows)


# ── HTML 样式输出 ─────────────────────────────────────────────────────────────

HTML_CSS = """
<style>
body { font-family: Arial, sans-serif; font-size: 13px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
th { background: #2c3e50; color: white; }
tr:nth-child(even) { background: #f8f8f8; }
tr.header-row td { background: #dce8f5; font-weight: bold; }
td.smd-high { color: #c0392b; font-weight: bold; }
caption { font-size: 15px; font-weight: bold; margin-bottom: 8px; }
</style>
"""


def df_to_html(tbl: pd.DataFrame, n_s: int, n_c: int) -> str:
    lines = [HTML_CSS,
             "<table>",
             f"<caption>Table 1. Baseline Characteristics: "
             f"Analyzable matched cohort, Shock (N={n_s:,}) vs Control (N={n_c:,})</caption>",
             "<thead><tr>"]
    for col in tbl.columns:
        lines.append(f"<th>{col}</th>")
    lines.append("</tr></thead><tbody>")

    for _, row in tbl.iterrows():
        var = row["Variable"]
        is_header = var.startswith(("Age", "Admission", "ICU LOS",
                                    "Male", "Hospital", "Mechanical",
                                    "Vasopressor", "Race", "ICU type",
                                    "Monitoring", "Septic"))
        row_cls = ' class="header-row"' if not var.startswith("  ") else ""
        lines.append(f"<tr{row_cls}>")
        for col in tbl.columns:
            val = row[col]
            cell_cls = ""
            if col == "SMD":
                try:
                    if abs(float(val)) >= 0.1:
                        cell_cls = ' class="smd-high"'
                except (ValueError, TypeError):
                    pass
            lines.append(f"<td{cell_cls}>{val}</td>")
        lines.append("</tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 70)
    log.info("Step 6 (T1.1): Baseline Table — Analyzable matched cohort (Shock vs Control)")
    log.info("=" * 70)

    # 1. 构建分组基础表
    base = load_and_group()

    # 2. 从 DuckDB 拉取协变量并合并
    master = build_master_table(base)

    # 3. 统计分析
    log.info("Building Table 1 statistics...")
    tbl = build_table1(master)

    # 4. 输出
    csv_path  = OUTPUT_DIR / "table1_baseline.csv"
    html_path = OUTPUT_DIR / "table1_baseline.html"

    tbl.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info(f"CSV  saved → {csv_path}")

    n_s = int((master["group"] == "shock").sum())
    n_c = int((master["group"] == "control").sum())
    html = df_to_html(tbl, n_s, n_c)
    html_path.write_text(html, encoding="utf-8")
    log.info(f"HTML saved → {html_path}")

    # 5. 终端摘要
    log.info("")
    log.info("━" * 70)
    log.info("TABLE 1 PREVIEW")
    log.info("━" * 70)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 160)
    log.info("\n" + tbl.to_string(index=False))
    log.info("━" * 70)
    log.info("Step 6 complete.")


if __name__ == "__main__":
    main()
