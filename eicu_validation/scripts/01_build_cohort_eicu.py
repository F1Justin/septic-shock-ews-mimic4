"""
eICU 外部验证 — Step 1: 队列构建与1:1匹配

病例定义（septic shock）:
  - diagnosis.diagnosisstring 含 "septic shock"
  - infusionDrug 中有升压药记录（NE/VP/EP/PE/DA）
  - T0 = 首次升压药 infusionoffset（分钟）
  - 要求 T0 >= 12h（晚窗 [-12h, 0h) 需要足够前驱数据）
  - 要求 ICU 总时长 >= T0 + 2h

对照定义:
  - 无任何 sepsis 相关诊断（diagnosisstring 不含 "sepsis"/"septic"）
  - pseudo-T0 = 与配对病例的 T0 时长相近（random 分配，确保对照有相同的 ICU 暴露时长）

匹配条件（1:1，无放回）:
  - age_group (5岁分组，> 89 处理为 90)
  - gender
  - unittype_simplified (Med-Surg/MICU/CCU/SICU/Neuro/Other)
  - hospitaldischargeyear

输出:
  eicu_validation/output/cohort_eicu.parquet
  eicu_validation/output/t0_sensitivity_eicu.csv
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent   # eicu_validation/
NAAS_ROOT    = PROJECT_ROOT.parent            # NaaS/
EICU_DIR     = NAAS_ROOT / "eicu-crd" / "2.0"
OUTPUT_DIR   = PROJECT_ROOT / "output"

VASOPRESSOR_NAMES = (
    "norepinephrine", "epinephrine", "vasopressin", "phenylephrine", "dopamine"
)
SEDATION_NAMES = (
    "propofol", "midazolam", "fentanyl", "lorazepam",
    "dexmedetomidine", "ketamine",
)
BETA_BLOCKER_NAMES = (
    "metoprolol", "atenolol", "carvedilol", "bisoprolol",
    "labetalol", "propranolol",
)

MIN_T0_H         = 12     # 晚窗至少需要 T0 >= 12h
MIN_ICU_MARGIN_H = 2      # T0 后至少还要有 2h 的 ICU 数据
RANDOM_SEED      = 114514
MATCH_RATIO      = 1


def simplify_unittype(ut: str) -> str:
    if pd.isna(ut):
        return "Other"
    ut = str(ut).upper()
    if "MED-SURG" in ut or "MEDICAL-SURGICAL" in ut:
        return "Med-Surg"
    if "MICU" in ut or ("MEDICAL" in ut and "SURG" not in ut):
        return "MICU"
    if "SICU" in ut or "SURGICAL" in ut or "TRAUMA" in ut:
        return "SICU"
    if "CCU" in ut or "CARDIAC" in ut or "CSICU" in ut or "CTICU" in ut:
        return "CCU"
    if "NEURO" in ut:
        return "Neuro"
    return "Other"


def age_to_numeric(age: str) -> float:
    """eICU age 是字符串，'> 89' → 90，其余转 float。"""
    if pd.isna(age):
        return np.nan
    age = str(age).strip()
    if age.startswith(">"):
        return 90.0
    try:
        return float(age)
    except ValueError:
        return np.nan


def age_group(age: float, width: int = 5) -> int:
    """将年龄归入 5 岁组，NaN → -1。"""
    if pd.isna(age):
        return -1
    return int(age // width) * width


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    # ── 1. 加载基础表 ──────────────────────────────────────────────────────────
    print("── 加载 patient.csv.gz ───────────────────────────────────────────────")
    pt = pd.read_csv(
        EICU_DIR / "patient.csv.gz",
        usecols=[
            "patientunitstayid", "uniquepid", "gender", "age",
            "unittype", "hospitalid", "hospitaldischargeyear",
            "unitdischargeoffset", "hospitaladmitoffset",
        ],
    )
    print(f"  总 ICU stays: {len(pt):,}")

    print("── 加载 diagnosis.csv.gz ─────────────────────────────────────────────")
    dx = pd.read_csv(
        EICU_DIR / "diagnosis.csv.gz",
        usecols=["patientunitstayid", "diagnosisoffset", "diagnosisstring"],
    )

    print("── 加载 infusionDrug.csv.gz ──────────────────────────────────────────")
    inf = pd.read_csv(
        EICU_DIR / "infusionDrug.csv.gz",
        usecols=["patientunitstayid", "infusionoffset", "drugname"],
    )

    # ── 2. 识别病例 ────────────────────────────────────────────────────────────
    print("── 识别 septic shock 病例 ────────────────────────────────────────────")

    # 有 septic shock 诊断
    shock_dx = dx[dx["diagnosisstring"].str.contains("septic shock", case=False, na=False)]
    shock_ids_dx = set(shock_dx["patientunitstayid"])
    print(f"  含 septic shock 诊断: {len(shock_ids_dx):,} stays")

    # 有升压药记录
    vaso_pat = "|".join(VASOPRESSOR_NAMES)
    vaso = inf[inf["drugname"].str.contains(vaso_pat, case=False, na=False)]
    t0_vaso = (
        vaso[vaso["patientunitstayid"].isin(shock_ids_dx)]
        .groupby("patientunitstayid")["infusionoffset"]
        .min()
        .reset_index()
        .rename(columns={"infusionoffset": "T0_min"})
    )
    t0_vaso["T0_h"] = t0_vaso["T0_min"] / 60
    print(f"  有升压药 + septic shock: {len(t0_vaso):,} stays")

    # 合并 patient 信息
    cases = pt.merge(t0_vaso, on="patientunitstayid", how="inner")
    cases["icu_duration_h"] = cases["unitdischargeoffset"] / 60

    # 应用纳入标准
    cases = cases[cases["T0_h"] >= MIN_T0_H].copy()
    cases = cases[cases["icu_duration_h"] >= cases["T0_h"] + MIN_ICU_MARGIN_H].copy()
    print(f"  T0 >= {MIN_T0_H}h + ICU margin: {len(cases):,} 有效病例")

    # T0 分布敏感性
    sens_rows = []
    for th in [6, 8, 10, 12, 18, 24, 36, 48]:
        n = (t0_vaso["T0_h"] >= th).sum()
        sens_rows.append({"min_T0_h": th, "n_cases": n})
    pd.DataFrame(sens_rows).to_csv(OUTPUT_DIR / "t0_sensitivity_eicu.csv", index=False)

    # ── 3. 识别对照 ────────────────────────────────────────────────────────────
    print("── 识别对照（无 sepsis 诊断）────────────────────────────────────────")
    sepsis_ids = set(dx[dx["diagnosisstring"].str.contains(
        "sepsis|septic", case=False, na=False)]["patientunitstayid"])
    controls_all = pt[~pt["patientunitstayid"].isin(sepsis_ids)].copy()
    controls_all["icu_duration_h"] = controls_all["unitdischargeoffset"] / 60
    print(f"  无 sepsis 诊断的 stays: {len(controls_all):,}")

    # ── 4. 构建匹配键 ──────────────────────────────────────────────────────────
    print("── 构建匹配键 ────────────────────────────────────────────────────────")
    for df in [cases, controls_all]:
        df["age_num"]   = df["age"].apply(age_to_numeric)
        df["age_grp"]   = df["age_num"].apply(age_group)
        df["unit_type"] = df["unittype"].apply(simplify_unittype)
        df["gender_std"]= df["gender"].str.strip().str.title().fillna("Unknown")
        df["match_key"] = (
            df["age_grp"].astype(str) + "|"
            + df["gender_std"] + "|"
            + df["unit_type"] + "|"
            + df["hospitaldischargeyear"].astype(str)
        )

    # ── 5. 1:1 无放回匹配 ──────────────────────────────────────────────────────
    print("── 1:1 匹配 ─────────────────────────────────────────────────────────")

    # 对照按 match_key 建索引（转为 dict 以便 remove 操作）
    ctrl_pool: dict[str, list[dict]] = {}
    for _, row in controls_all.iterrows():
        k = row["match_key"]
        ctrl_pool.setdefault(k, []).append(row.to_dict())
    for v in ctrl_pool.values():
        rng.shuffle(v)

    # 病例顺序随机化——消除 iterrows 遍历顺序对匹配结果的隐含依赖
    cases = cases.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    matched_pairs = []
    pair_id = 0
    unmatched = 0

    for _, case_row in cases.iterrows():
        k = case_row["match_key"]
        pool = ctrl_pool.get(k, [])
        # 对照也需要有足够 ICU 时长覆盖同等 pseudo-T0
        eligible = [c for c in pool if c["icu_duration_h"] >= case_row["T0_h"] + MIN_ICU_MARGIN_H]
        if not eligible:
            unmatched += 1
            continue
        ctrl_row_dict = eligible[0]
        pool.remove(ctrl_row_dict)
        ctrl_row = ctrl_row_dict

        for row, grp in [(case_row, "shock"), (ctrl_row, "control")]:
            matched_pairs.append({
                "patientunitstayid": int(row["patientunitstayid"]),
                "uniquepid":         row["uniquepid"],
                "hospitalid":        int(row["hospitalid"]),
                "group":             grp,
                "T0_min":            float(case_row["T0_min"]) if grp == "shock" else float(case_row["T0_min"]),
                "T0_h":              float(case_row["T0_h"]),
                "icu_duration_h":    float(row["icu_duration_h"]),
                "age_num":           float(row["age_num"]) if not pd.isna(row["age_num"]) else np.nan,
                "gender":            row["gender_std"],
                "unit_type":         row["unit_type"],
                "year":              int(row["hospitaldischargeyear"]),
                "matched_pair_id":   pair_id,
            })
        pair_id += 1

    cohort = pd.DataFrame(matched_pairs)
    print(f"  匹配对数: {pair_id:,}  未匹配病例: {unmatched:,}")
    print(f"  cohort 行数: {len(cohort):,}")

    # ── 6. 协变量补充（镇静药、β阻滞剂、机械通气）─────────────────────────
    print("── 补充协变量 ───────────────────────────────────────────────────────")

    # 镇静药：infusionDrug 在 T0 前 12h 内
    sed_pat = "|".join(SEDATION_NAMES)
    sed = inf[inf["drugname"].str.contains(sed_pat, case=False, na=False)][
        ["patientunitstayid", "infusionoffset"]
    ].copy()

    # β 阻滞剂：medication
    med = pd.read_csv(
        EICU_DIR / "medication.csv.gz",
        usecols=["patientunitstayid", "drugstartoffset", "drugname"],
    )
    bb_pat = "|".join(BETA_BLOCKER_NAMES)
    bb = med[med["drugname"].str.contains(bb_pat, case=False, na=False)][
        ["patientunitstayid", "drugstartoffset"]
    ].copy()

    # 机械通气：respiratoryCare.ventstartoffset / ventendoffset
    rc = pd.read_csv(
        EICU_DIR / "respiratoryCare.csv.gz",
        usecols=["patientunitstayid", "ventstartoffset", "ventendoffset"],
    )
    rc = rc[rc["ventstartoffset"].notna()].copy()

    # 对每个 (patientunitstayid, T0_min) 判断是否有 prior 12h 内的暴露
    cov_rows = []
    for _, row in cohort.drop_duplicates("patientunitstayid").iterrows():
        sid  = row["patientunitstayid"]
        t0   = row["T0_min"]
        win_start = t0 - 12 * 60   # T0-12h (分钟)

        # 镇静药
        s_sub = sed[sed["patientunitstayid"] == sid]
        sed_flag = int(((s_sub["infusionoffset"] < t0) & (s_sub["infusionoffset"] >= win_start)).any())

        # β 阻滞剂
        b_sub = bb[bb["patientunitstayid"] == sid]
        bb_flag = int(((b_sub["drugstartoffset"] < t0) & (b_sub["drugstartoffset"] >= win_start)).any())

        # 机械通气 (active at win_start)
        v_sub = rc[rc["patientunitstayid"] == sid]
        vent_flag = 0
        if not v_sub.empty:
            vent_flag = int((
                (v_sub["ventstartoffset"] <= win_start) &
                ((v_sub["ventendoffset"] >= win_start) | (v_sub["ventendoffset"] == 0))
            ).any())

        cov_rows.append({
            "patientunitstayid":     sid,
            "sedation_before_window": sed_flag,
            "betablocker_before_window": bb_flag,
            "vent_before_window":    vent_flag,
        })

    cov_df = pd.DataFrame(cov_rows)
    cohort = cohort.merge(cov_df, on="patientunitstayid", how="left")
    for col in ["sedation_before_window", "betablocker_before_window", "vent_before_window"]:
        cohort[col] = cohort[col].fillna(0).astype(int)

    # ── 7. 保存 ───────────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "cohort_eicu.parquet"
    cohort.to_parquet(out_path, index=False)
    print(f"\n  → {out_path}")
    print(f"  shock: {(cohort['group']=='shock').sum()}, control: {(cohort['group']=='control').sum()}")
    print(f"  matched pairs: {cohort['matched_pair_id'].nunique()}")
    print(f"  hospitals: {cohort['hospitalid'].nunique()}")

    # 简单汇总
    print("\n── 队列特征简报 ─────────────────────────────────────────────────────")
    for grp, sub in cohort.groupby("group"):
        print(f"  {grp}: n={len(sub)}, age={sub['age_num'].mean():.1f}±{sub['age_num'].std():.1f}, "
              f"T0={sub['T0_h'].mean():.1f}h (median {sub['T0_h'].median():.1f}h)")


if __name__ == "__main__":
    main()
