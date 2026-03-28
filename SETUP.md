# EWS 项目当前正式分析链

本文件只描述当前与 `report.tex`、`output/*.csv`、`scripts/0*.py` 一致的正式分析流程。
历史试验性结果、已废弃的主链、旧版大样本中间结果均不再作为当前稿件依据。

## 环境依赖

```bash
cd /Users/justin/0Projects/NaaS
pip install -r requirements.txt
```

当前正式流程主要依赖：
- `duckdb`
- `pandas`
- `numpy`
- `scipy`
- `statsmodels`
- `seaborn`
- `pyarrow`

## Step 1: 数据库初始化

运行：

```bash
python scripts/01_setup_db.py
```

用途：
- 构建 `mimiciv_icu` / `mimiciv_hosp` 懒加载视图
- 生成 `mimiciv_derived.sepsis3`、`mimiciv_derived.sofa`、`mimiciv_derived.norepinephrine_equivalent_dose` 等分析所需派生表

当前基数：
- `Sepsis-3 ICU stays`: `41,296`

## Step 2: 队列构建与风险集匹配

运行：

```bash
python scripts/02_build_cohort.py
```

方法：
- 基础人群：`mimiciv_derived.sepsis3`
- septic shock 操作性定义：
  - ICU 内首次持续 `>=1h` 的升压药开始
  - 该时间点前后 `±24h` 内存在 lactate `>2 mmol/L`
  - 该时间点前 `24h` 内累计等渗晶体液 `>=1000 mL`
- `T0` 为满足以上条件的最早升压药开始时间
- 先做数据可分析性预筛，再做 `1:2` risk-set matching
- 匹配变量：入 ICU 后前 `24h` 的 admission SOFA，卡钳 `±2`
- 对照 `T0 = icu_intime + H`，其中 `H = case T0 - case icu_intime`

当前正式结果：
- `Sepsis-3 stays`: `41,296`
- `Shock stays` under main operational definition: `10,310`
- `T0 window sensitivity`: `±6h=9,239`, `±12h=9,909`, `±24h=10,310`
- `Fluid threshold sensitivity`:
  - `1000 mL = 10,310`
  - `1500 mL = 9,839`
  - `2000 mL = 9,369`
  - `2400 mL = 8,829`
  - `3000 mL = 8,124`
- 通过 72h 观察长度与数据质量预筛的 shock stays: `725`
- 匹配后：
  - `shock = 725`
  - `control = 1,450`
  - `matched pairs = 725`

输出：
- `data/cohort.parquet`
- `output/t0_window_sensitivity.csv`
- `output/fluid_threshold_sensitivity.csv`

说明：
- `data/cohort.parquet` 是匹配完成后的原始配对集；后续清洗后分析集为 `2,172` 行，因为有 `3` 个 matched windows 在最终清洗时被排除。

## Step 3: 时间序列提取与清洗

运行：

```bash
python scripts/03_extract_and_clean.py
```

可选双重去趋势敏感性：

```bash
python scripts/03_extract_and_clean.py --detrend-mode double --suffix _double_detrend
```

当前主流程方法：
- 每个分析单元为 `(stay_id, T0)`，不是仅按 `stay_id`
- 对每个单元提取 `T0` 前 `72h` 的 MAP/HR
- 前 `24h` 仅作为 burn-in，用于稳定 trailing mean
- 真正分析窗口为后 `48h`
- MAP：`ABPm` 优先，`NBPm` 补充
- 清洗顺序：
  - 生理范围过滤
  - 1h 重采样（中位数）
  - 24h causal trailing mean 去趋势
  - 实测点 `mu ± 3 sigma` 异常值回缩
- 排除标准：后 `48h` 内 MAP 或 HR 实测点 `<24`

当前正式结果：
- `cohort (stay_id, T0) pairs`: `2,175`
- `passed final cleaning`: `2,172`
- `excluded`: `3`
- `dominant_source` among analysable pairs:
  - `NBP = 1,097 (50.5%)`
  - `ABP = 823 (37.9%)`
  - `Mixed = 252 (11.6%)`

输出：
- `data/vitals_cleaned.parquet`
- `data/cleaning_diagnostics.parquet`

双重去趋势敏感性输出：
- `data/vitals_cleaned_double_detrend.parquet`
- `data/cleaning_diagnostics_double_detrend.parquet`

## Step 4: EWS 计算与组间比较

运行：

```bash
python scripts/04_ews_analysis.py
```

双重去趋势敏感性：

```bash
python scripts/04_ews_analysis.py --suffix _double_detrend
```

当前正式方法：
- 指标：
  - `Variance-MAP`
  - `AC1-HR`
- 使用 `12h` rolling window，步长 `1h`
- 每个 rolling window 用窗口中心时间 `hours_before_T0` 标记，而不是窗口终点
- 在当前 48h analysable trajectory 上，实际 window centers 大致从 `-42h` 到 `-5h`
- 早/晚窗口定义基于窗口中心：
  - `early`: centers in `[-48h, -24h)`
  - `late`: centers in `[-12h, 0h)`，实际可实现到 `-5h`
- 主比较为：
  - early mean
  - late mean
  - late minus early
- 组间比较：stay-level clustered GEE
- 轨迹差异：LMM `indicator ~ time * group`
- 多重比较：Holm across full cohort + modality subgroups

当前正式结果（full cohort）：
- `AC1-HR`
  - early mean: `0.345 vs 0.323`, raw `p=0.0395`, Holm `p=0.3157`
  - late mean: `0.382 vs 0.323`, raw `p<0.001`, Holm `p<0.001`
  - delta: `0.036 vs -0.002`, raw `p=0.0208`, Holm `p=0.1872`
- `Variance-MAP`
  - early mean: `64.20 vs 72.30`, raw `p=0.0062`, Holm `p=0.0685`
  - late mean: `77.00 vs 74.99`, raw `p=0.5129`, Holm `p=1.0000`
  - delta: `12.78 vs 2.43`, raw `p=0.0016`, Holm `p=0.0249`
- LMM:
  - `AC1-HR time-by-group beta = 0.00143`, `p=0.0241`

输出：
- `data/ews_windows.parquet`
- `data/ews_patient_stats.parquet`
- `output/table3_ews_comparison.csv`
- `output/table3_lmm_summary.csv`
- `output/fig1_timeseries.png`
- `output/fig2_delta_boxplot.png`
- `output/figS1_subgroup.png`

双重去趋势敏感性输出：
- `data/ews_windows_double_detrend.parquet`
- `data/ews_patient_stats_double_detrend.parquet`
- `output/table3_ews_comparison_double_detrend.csv`
- `output/table3_lmm_summary_double_detrend.csv`

## Step 5: 护理扰动恢复分析（探索性）

运行：

```bash
python scripts/05_perturbation_recovery.py
```

当前定位：
- 探索性、阴性为主
- 当前稿件不将其作为主结果支柱

当前正式结果：
- 有效 Turn events: `703`
- within-shock early vs late: `p=0.303`
- late shock vs control: `p=0.129`
- no-vasopressor subgroup: `p=0.067`

输出：
- `output/fig3_recovery.png`
- 对应表格与中间数据见 `output/` / `data/`

## Step 6: 基线表（Table 1）

运行：

```bash
python scripts/06_baseline_table.py
```

当前正式定义：
- 不再做 `analysable vs excluded` 的旧版审计表
- 当前 `Table 1` 为 analyzable matched cohort 内的 `Shock vs Control`

当前正式结果：
- `Shock = 725`
- `Control = 1,447`

输出：
- `output/table1_baseline.csv`
- `output/table1_baseline.html`

## Step 7: 匹配条件 Logistic 主分析

运行：

```bash
python scripts/07_multivariable_model.py
```

双重去趋势敏感性：

```bash
python scripts/07_multivariable_model.py --suffix _double_detrend
```

当前正式主分析：
- 模型：conditional logistic regression
- strata: `matched_pair_id`
- 主指标：`HR AC1, late-centred window mean`
- 协变量：
  - `vent_before_window`
  - `sedation_before_window`
  - `betablocker_before_window`
  - `icu_type`
  - `monitoring`

当前正式结果：
- Primary:
  - `HR AC1 late-centred window mean OR = 1.60 (1.15–2.22), p=0.005`
- S4:
  - 加入 raw final-12h MAP/HR 后：
  - `OR = 1.23 (0.85–1.78), p=0.271`
- S5A:
  - early-centred AC1 + early raw MAP/HR：
  - `OR = 1.12 (0.73–1.72), p=0.612`
- S5B:
  - late-centred AC1 + early raw MAP/HR：
  - `OR = 1.48 (1.05–2.08), p=0.025`
- S6:
  - no-sedation subgroup：
  - `OR = 1.07 (0.59–1.95), p=0.815`

双重去趋势敏感性：
- primary:
  - `OR = 1.29 (0.91–1.83), p=0.159`
  - 加入 raw final-12h MAP/HR 后：
  - `OR = 0.92 (0.62–1.38), p=0.701`

输出：
- `output/table2_multivariable.csv`
- `output/table2_diagnostics.csv`
- `output/tableS4_late_vitals_sensitivity.csv`
- `output/tableS5_early_window_independence.csv`
- `output/tableS6_no_sedation_subgroup.csv`

双重去趋势敏感性输出：
- `output/table2_multivariable_double_detrend.csv`
- `output/table2_diagnostics_double_detrend.csv`
- `output/tableS4_late_vitals_sensitivity_double_detrend.csv`
- `output/tableS5_early_window_independence_double_detrend.csv`
- `output/tableS6_no_sedation_subgroup_double_detrend.csv`

## Step 8: 非独立观测稳健性分析

运行：

```bash
python scripts/08_cluster_sensitivity.py
```

当前正式结果：
- GEE adjusted:
  - `OR = 1.61 (1.17–2.22), p=0.003`
- 去重复子集：
  - `OR = 1.65 (1.18–2.31), p=0.003`
- bootstrap 95% CI:
  - `Δ late-centred AC1-HR mean = [0.0331, 0.0862]`
  - `Δ early-window AC1-HR mean = [0.0033, 0.0465]`
  - `Δ early-to-late AC1-HR change = [0.0041, 0.0647]`

输出：
- `output/tableS1_gee.csv`
- `output/tableS2_dedup.csv`
- `output/tableS3_bootstrap.csv`

## 当前稿件应采用的解释边界

当前正式链支持的结论是：
- `HR AC1` 的信号主要集中在窗口中心约位于 `-12h` 到 `-5h` 的 late-centred rolling-window interval
- 它不是一个已被证明独立于同时期原始 `MAP/HR` 的 occult early-warning signal
- no-sedation subgroup 与 double detrending sensitivity 都削弱了更强的机制性主张
- 因此最稳妥的表述是：
  - `HR AC1` 是 late physiological rigidity 的量化描述符
  - 它与 overt haemodynamic deterioration 共演化
  - 目前不应被表述为静默期、独立、可直接部署的预警器
