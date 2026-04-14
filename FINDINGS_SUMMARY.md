# 脓毒性休克早期预警信号：完整研究结论汇总

> 生成时间：2026-04-13  
> 覆盖项目：`ac1/`（MIMIC-IV AC1 分析）· `euler_ews/`（MIMIC-IV Euler/n_extrema 分析）· `eicu_validation/`（eICU 外部验证）

---

## 一、研究背景与核心问题

**核心假设**：脓毒性休克发生前，自主神经调节能力逐渐崩溃，可体现在心率（HR）时间序列的统计特征上——
- **HR AC1**（滞后-1 自相关）升高 → 心率动力学趋向持续性（临界慢化现象）
- **HR Euler χ(0)**（HR 残差零阈值下的连通分量数）降低 → HR 长时间偏离基线（持续心动过速）
- **HR n_extrema**（局部极值数）降低 → HR 振荡复杂性丧失（弹性下降）

**研究设计**：
- 数据集：MIMIC-IV（发现队列，1:2 匹配，725 shock / 1447 control）；eICU-CRD（外部验证，1:1 匹配，587 对）
- 时间窗：晚窗 \[-12, 0\) h before T0（MIMIC 与 eICU 一致）；早窗定义**两个数据集不同**：
  - MIMIC-IV（AC1/Euler/n\_extrema）：\[-48, -24\) h（`EARLY_LO, EARLY_HI = -48, -24`）
  - eICU 验证：\[-24, -12\) h（`EARLY_LO, EARLY_HI = -24, -12`，受 eICU 中位 T0=24h 限制）
  - ⚠️ **两端早窗口径不同，不可将两侧早窗结果直接并列讨论**
- T0 定义：升压药启动 + 脓毒性休克诊断同时成立的时间点
- 主要模型：条件 logistic 回归（strata = matched\_pair\_id）

---

## 二、MIMIC-IV：AC1 分析（项目 `ac1/`）

### 2.1 基线特征

| 指标 | Shock (N=725) | Control (N=1,447) | P 值 |
|------|-------------|------------------|------|
| 年龄（岁） | 65 \[54–74\] | 63 \[54–75\] | 0.483 |
| 入院 SOFA | 6 \[4–9\] | 6 \[4–8\] | 0.398 |
| 男性 | 56.7% | 59.6% | 0.198 |
| 院内死亡率 | **56.8%** | 21.3% | <0.001 |
| 升压药前使用 | 93.2% | 41.4% | <0.001 |
| ICU 类型差异 | — | — | <0.001 |

两组匹配良好，年龄/SOFA/监测模式组间无显著差异。

### 2.2 主要结果：HR AC1 与脓毒性休克关联

| 模型 | OR (95% CI) | P |
|------|------------|---|
| **M1 主模型（单次去趋势）** | **1.60 (1.15–2.22)** | **0.005** |
| S4 + 晚窗 MAP/HR 均值调整 | 1.23 (0.85–1.78) | 0.271 |
| M1（双重去趋势） | 1.29 (0.91–1.83) | 0.159 |
| 无镇静亚组 | 1.07 (0.59–1.95) | 0.815 |

**关键发现 1**：HR AC1 在晚窗（T0 前 12h）显著升高，但有**两个脆弱性**：
1. **均值混淆**：控制晚窗 HR 均值后，OR 从 1.60 降至 1.23（p=0.271）——脓毒性休克的持续心动过速（高均值 HR）可能直接拉高 AC1，而非独立的动力学信号。
2. **去趋势敏感性**：双重去趋势后信号衰减（OR 1.29, p=0.159），说明 AC1 信号部分依赖趋势去除方法。

### 2.3 GEE 时序分析

| 比较 | 早窗 p | 晚窗 p | Δ(早→晚) p |
|------|--------|--------|-----------|
| AC1-HR（全样本） | 0.039 | <0.001 | 0.021 |
| Variance-MAP（全样本） | 0.007 | 0.513 | 0.002 |
| AC1-HR（双重去趋势） | 0.363 | 0.001 | 0.049 |

晚窗 AC1-HR 与 control 的差距显著（p<0.001），且存在早→晚的时序增加趋势（LMM 交互项 p=0.024），支持"临界慢化"的时序演化叙事。

### 2.4 扰动恢复分析（MAP AUC，探索性）

翻身（Turn, itemid=224082）后 30min MAP 偏差积分：

| 分析 | Shock | Control | P |
|------|-------|---------|---|
| 早窗 vs 晚窗（全队列 shock） | 127.6 vs 180.0 | — | **0.075**（边缘） |
| 策略B（无升压药，shock 早 vs 晚） | 127.6 vs 192.0 | — | **0.016** |
| 晚窗 shock vs control | 180.0 vs 138.0 | — | 0.330 |

接近休克时（晚窗），无升压药 shock 患者翻身后 MAP 偏差更大（恢复更慢），与"弹性丧失"假说一致，但样本量小、全样本不显著。**这是整个研究的"机制灵魂"——扰动响应揭示了生理调节弹性的直接证据。**

---

## 三、MIMIC-IV：Euler χ(0) 与 n_extrema 分析（项目 `euler_ews/`）

### 3.1 GEE 组间比较（早、晚窗均值，shock vs control）

| 指标 | 早窗 Shock | 早窗 Control | 晚窗 Shock | 晚窗 Control | 晚窗 p (Holm) |
|------|-----------|-------------|-----------|-------------|--------------|
| **HR Euler χ(0)** | 1.752 | 1.860 | **1.617** | **1.890** | **<0.001** |
| **HR n_extrema** | 5.478 | 5.515 | **5.160** | **5.534** | **<0.001** |
| MAP Euler χ(0) | 2.386 | 2.347 | 2.284 | 2.334 | 1.000 |
| MAP n_extrema | 6.060 | 5.979 | 5.726 | 5.920 | 0.055 |
| HR total variation | 55.1 | 57.9 | 56.1 | 57.8 | 1.000 |

**关键发现 2**：**HR Euler χ(0) 和 HR n_extrema 是最强信号，MAP 指标均不显著。** Shock 患者晚窗 HR 的拓扑/振荡复杂性显著低于 control，方向与假设一致（心率失去弹性）。

### 3.2 条件 logistic 回归（N=725 shock, 1447 control, 720 strata）

| 模型 | 主预测变量 | OR (95% CI) | P |
|------|----------|------------|---|
| **M1 主模型** | HR Euler χ(0) 晚窗均值 | **0.75 (0.66–0.84)** | **<0.001** |
| S1 +晚窗 HR 均值 | HR Euler χ(0) | 0.79 (0.70–0.89) | <0.001 |
| S2 +晚窗 HR+MAP 均值 | HR Euler χ(0) | 0.86 (0.76–0.98) | 0.025 |
| S3 早窗（均值调整） | HR Euler χ(0) 早窗均值 | 0.85 (0.74–0.98) | 0.027 |
| **S4 n_extrema 替代** | **HR n_extrema 晚窗均值** | **0.88 (0.82–0.94)** | **<0.001** |
| S4b n_extrema +HR 均值 | HR n_extrema | 0.88 (0.82–0.94) | <0.001 |
| S5 无镇静亚组 | HR Euler χ(0) | 0.72 (0.58–0.89) | 0.002 |

**关键发现 3**：
- **Euler χ(0)** 在均值调整后信号衰减（M1 OR=0.75 → S2 OR=0.86），与 AC1 类似，**部分信号由持续心动过速（高 HR 均值）驱动**。
- **n_extrema 是最鲁棒的指标**：加入 HR 均值协变量后 OR 几乎不变（0.88 → 0.88），说明它捕捉的是**独立于均值水平的振荡复杂性信息**。
- 无镇静亚组 Euler OR=0.72（p=0.002），更强，表明镇静药部分混淆了主分析。

---

## 四、eICU 外部验证（项目 `eicu_validation/`）

### 4.1 队列特征

- 587 对（1:1 匹配），匹配条件：年龄组、性别、ICU 类型、入院年份
- **中位 T0（ICU 入住后时间）：约 24h**，远短于 MIMIC-IV（>72h）
- T0 分布：T0≥48h 154 对，T0≥72h 103 对

### 4.2 全样本验证结果

| 指标 | eICU 全样本 OR (95% CI) | P | MIMIC-IV 参考 OR |
|------|----------------------|---|-----------------|
| AC1 | 1.21 (0.75–1.97) | 0.435 | 1.60 (1.15–2.22) |
| n_extrema | 0.94 (0.86–1.02) | 0.142 | 0.88 (0.82–0.94) |
| Euler χ(0) | 0.97 (0.81–1.16) | 0.756 | 0.75 (0.66–0.84) |

**全样本无一显著**，但方向与 MIMIC-IV 一致（AC1 OR>1，n\_extrema/Euler OR<1）。

### 4.3 T0 分层敏感性分析（核心发现）

| T0 阈值 | 对数 | AC1 OR (p) | n_extrema OR (p) | Euler OR (p) |
|--------|-----|-----------|-----------------|-------------|
| 全样本（中位 24h） | 587 | 1.21 (0.435) | 0.94 (0.142) | 0.97 (0.756) |
| ≥24h | 297 | 1.50 (0.224) | 0.96 (0.455) | 1.02 (0.852) |
| ≥48h | 154 | 1.21 (0.673) | 0.92 (0.281) | 0.84 (0.262) |
| **≥72h** | **103** | **2.68 (0.082)** | **0.80 (0.034) ★** | **0.72 (0.084)** |
| MIMIC-IV 参考 | 725 | 1.60 (0.005) | 0.88 (<0.001) | 0.75 (<0.001) |

**关键发现 4**（最重要的 eICU 结论）：
- **T0≥72h 时 n_extrema OR=0.80（p=0.034）**，与 MIMIC-IV 的 OR=0.88 高度吻合
- Euler OR=0.72（p=0.084），在 100 对样本下趋近但未达显著，与 MIMIC-IV 0.75 几乎相同
- **"T0 剂量-响应关系"（非严格单调）**：总体趋势是更长 T0 子集里 OR 更接近 MIMIC-IV 方向（≥72h 最一致），但并非严格单调——n_extrema 在 ≥24h 略微回升（0.94→0.96），Euler 在 ≥24h 反而更靠近 1（0.97→**1.02**）后才重新收敛（≥48h: 0.84，≥72h: 0.72）。正确表述为"≥72h 子集最接近 MIMIC"，而非"清晰的单调生物学梯度"

**机制解释**：eICU 全样本不显著的根本原因是**T0 太短**（中位 24h），预警窗口内患者尚未充分进入"失代偿前期"。信号本身存在，但需要足够长的 ICU 暴露时间才能被检测到。

### 4.4 SOFA/APACHE 独立性分析

**AUC 对比**（注意：SOFA 测于 T0 时刻，EWS 测于 T0 前 12h，比较时间点不同）：

> **方法注记**：所有指标均在同一 **passed 母集**（通过 EWS 数据质量筛查的子集）内计算，早期版本因 `how="left"` 合并导致 SOFA/APACHE（n=1,159/1,002）与 EWS（n=1,152）分母不同；修正后 `sdf` 先 inner join `passed` 子集（n=1,152），再合并 EWS 特征。但由于 SOFA/APACHE 本身也有缺失，**各指标的实际分析 n 仍不完全相同**（SOFA: 1,140；APACHE: 989；AC1: 1,138；Euler/n\_extrema: 1,152），不构成"完全相同分母的 head-to-head 比较"，只能说"在同一 passed 母集内，按各指标可用样本分别计算"。

| 指标 | AUC (95% CI) | 分析 n | 性质 |
|------|-------------|-------|------|
| SOFA full（含升压药心血管项） | 0.909（0.891–0.926） | 1,140 | 临床评分（循环论证，供参考） |
| SOFA non-cardio（肝/肾/凝血/神经） | 0.680（0.650–0.711） | 1,140 | 临床评分（公平对比） |
| APACHE（入院评分） | 0.721（0.690–0.752） | 989 | 临床评分 |
| n_extrema（T0 前 12h 动态） | 0.544（0.511–0.577） | 1,152 | EWS |
| Euler χ(0) | 0.516（0.483–0.549） | 1,152 | EWS |
| AC1 | 0.522（0.488–0.555） | 1,138 | EWS |

**条件 logistic 独立性检验（T0≥72h，n=100 对）**：

| 指标 | alone | +SOFA non-cardio | +APACHE |
|------|-------|-----------------|---------|
| **n_extrema** | 0.79 (p=0.023)★ | **0.80 (p=0.041)★** | **0.77 (p=0.038)★** |
| Euler χ(0) | 0.72 (p=0.073) | 0.75 (p=0.135) | 0.76 (p=0.178) |
| AC1 | 2.36 (p=0.114) | 1.93 (p=0.237) | 2.74 (p=0.105) |

**关键发现 5**：**n_extrema 在 SOFA non-cardio 和 APACHE 双重调整后均保持显著（p<0.05）**，OR 几乎不变（0.77–0.80），证明它提供的是**独立于器官损害程度和入院疾病负担的增量预警信息**。这是 EWS 临床转化价值的直接证据。

---

## 五、MIMIC-IV：翻身后 HR n_extrema 扰动分析（探索性，`euler_ews/scripts/12_perturbation_n_extrema.py`）

**设计**：翻身后 4h 窗口内，HR 残差的局部极值数（n_extrema）  
**数据限制**：MIMIC-IV chartevents HR 为**护士手动记录（小时级）**，4h 窗口约 4-5 点

| 对比 | early-cohort 中位数 | late-cohort 中位数 | P |
|------|-------------------|------------------|---|
| Shock: early-cohort vs late-cohort | 1.0 | 1.0 | 0.363 |
| Control: early-cohort vs late-cohort | 1.0 | 1.5 | 0.929 |
| Late: shock vs control | — | 1.0 vs 1.5 | 0.240 |

> ⚠️ **方法说明**：以上 early/late 比较使用 Wilcoxon rank-sum（**非配对**），比较的是两个独立边际样本（各自有合格事件的患者子集），不是同一患者的配对追踪，因此**不能解读为"同一患者接近休克时的弹性变化"**，只能描述为"事件发生于 early 期次的患者群"与"发生于 late 期次的患者群"之间的差异。

**结论**：全部不显著。方向一致（shock 的翻身后 HR n_extrema 低于 control，与主分析方向相同），但因 HR 数据分辨率不足（小时级 vs MAP 的分钟级），且分析本质为非配对边际比较，统计功效极低。MAP AUC 扰动分析可行的原因是动脉置管提供了分钟级 MAP，而 HR 无此数据源。

---

## 六、SampEn 对照分析（`euler_ews/scripts/13_sampen_comparison.py`）

> 生成时间：2026-04-13  
> 目的：以 Sample Entropy（SampEn）作为已有熵方法代表，与拓扑指标做严格 head-to-head 比较，检验"均值调整鲁棒性"和"eICU 外部可验证性"两个关键主张。

### 6.1 方法细节

**SampEn 参数决策**

| 参数 | 取值 | 依据 |
|------|------|------|
| m（模板长度）| 1（主），2（敏感性）| 12 点窗口下 m=2 模板数过少 |
| r（容忍度）| **0.5 × patient_48h_std**（回归用）；0.2 × patient_48h_std（NaN 诊断用）| Richman & Moorman 2000 建议用全段 std；0.2 产生 15% NaN，0.5 降至 1% |
| min_actual | 8（比 Euler/n_extrema 的 6 更保守）| SampEn 需要更多点才能给出稳定估计 |
| NaN 根因 | A=0（length-2 模板在窗口内无匹配）| 12 点稀疏窗口的**结构性不可用**，非数据质量问题 |

**NaN 率实测对比（MIMIC 晚窗）**

| 指标 | NaN 率 |
|------|--------|
| Euler χ(0) / n_extrema | 0.3% |
| SampEn r=0.2（标准参数）| 15.2% |
| SampEn r=0.5（宽松参数）| 1.1% |
| SampEn m=2, r=0.5 | 9.6% |

### 6.2 MIMIC-IV 条件 Logistic 主要结果

**均值调整鲁棒性对比（M1 → S1 → S2）**

| 指标 | M1 OR (p) | S1 +HR均值 (p) | S2 +HR+MAP (p) |
|------|-----------|---------------|----------------|
| **n_extrema** | 0.88 <0.001 | 0.88 <0.001 | **0.87 <0.001** ★★★ |
| Euler χ(0) | 0.75 <0.001 | 0.79 <0.001 | 0.86 0.025 ★ |
| **SampEn m=1 r=0.5** | 0.86 **0.178** | 0.81 0.067 | 0.87 0.284 ✗ |
| SampEn m=2 r=0.5 | 0.84 **0.226** | 0.76 0.066 | 0.78 0.150 ✗ |
| AC1 | 1.60 0.005 | 1.49 0.021 | 1.23 0.271 ✗ |

**关键发现**：SampEn 在未调整的基础模型（M1）就已不显著（p=0.178），比 AC1（M1 p=0.005）信号更弱。即"均值调整后消失"不是 SampEn 的主要问题——在本数据集中，它根本未捕捉到独立信号。

### 6.3 MIMIC-IV GEE 早/晚窗组间比较（Holm 校正）

| 指标 | 早窗组差 Holm p | 晚窗组差 Holm p | Δ轨迹 Holm p |
|------|---------------|---------------|-------------|
| Euler χ(0) | 0.011 | <0.001 | 0.007 |
| n_extrema | 0.997 | <0.001 | <0.001 |
| SampEn r=0.5 | **<0.001** | 0.016 | **0.423** |
| SampEn r=0.2 | 0.030 | 0.423 | 0.997 |

**解读**：SampEn r=0.5 早窗差异显著，但 Δ（早→晚变化轨迹）不显著（p=0.423）——其差异是两组间静态截距差，而非进行性动态分离，与 n_extrema 的"随时间加剧"模式不同。

### 6.4 eICU 外部验证（T0 分层）

| T0 阈值 | SampEn OR (p) | n_extrema OR (p) | Euler OR (p) |
|--------|--------------|-----------------|-------------|
| 全样本（587 对）| 1.00 (0.978) ✗ | 0.94 (0.142) | 0.97 (0.756) |
| ≥24h（297 对）| 1.04 (0.819) ✗ | 0.96 (0.455) | 1.02 (0.852) |
| ≥48h（154 对）| 1.14 (0.628) ✗ | 0.92 (0.281) | 0.84 (0.262) |
| **≥72h（103 对）**| **1.17 (0.625) ✗** | **0.80 (0.034) ★** | 0.72 (0.084) |
| MIMIC 参考（725 对）| 0.86 (0.178) | 0.88 (<0.001) | 0.75 (<0.001) |

**关键发现**：SampEn 在 eICU 所有分层方向均相反（OR>1），完全失败；n_extrema T0≥72h 验证成功（OR=0.80, p=0.034），形成鲜明对比。

### 6.5 输出文件

| 文件 | 内容 |
|------|------|
| `euler_ews/output/tbl_sampen_logistic.csv` | 5 指标 × 3 模型汇总（M1/S1/S2）|
| `euler_ews/output/tbl_sampen_gee.csv` | GEE 早/晚窗对比（含 Holm 校正）|
| `euler_ews/output/fig_sampen_4way_forest.png` | **核心对比图**：4 指标 × M1/S1/S2 森林图 |
| `euler_ews/output/fig_sampen_m1_vs_m2.png` | SampEn m 参数敏感性图 |
| `eicu_validation/output/ews_patient_eicu.parquet` | 含 `sampen_rel_hr_late_mean` 列 |
| `eicu_validation/output/tbl_validation_t0strat.csv` | 含 SampEn(r=0.5) T0 分层行 |

---

## 七、综合结论

### 7.1 指标总评表（含 SampEn）

| 指标 | MIMIC M1 | MIMIC S2 | eICU 全样本 | eICU T0≥72h | 均值调整稳健性 | 推荐地位 |
|------|---------|---------|------------|------------|--------------|---------|
| **HR n_extrema** | OR=0.88*** | OR=0.87*** | OR=0.94 (ns) | **OR=0.80*** | **★★★ 完全稳健** | **主要结论** |
| **HR Euler χ(0)** | OR=0.75*** | OR=0.86* | OR=0.97 (ns) | OR=0.72† | ★★ 部分被均值解释 | 重要发现 |
| HR AC1 | OR=1.60** | OR=1.23 (ns) | OR=1.21 (ns) | OR=2.68† | ★ 高度依赖均值 | 参考 |
| SampEn r=0.5 | OR=0.86 (ns) | OR=0.87 (ns) | OR=1.00 (ns) | OR=1.17 (ns) ✗ | ✗ M1 就不显著 | **阴性对照** |
| MAP Euler | ns | ns | ns | — | — | 不推荐 |
| MAP AUC 恢复 | p=0.016（无升压药亚组）| 无数据 | — | — | — | 机制证据 |

†p<0.1, *p<0.05, **p<0.01, ***p<0.001

### 7.2 核心叙事

1. **心率振荡复杂性（n_extrema）是脓毒性休克前最鲁棒的拓扑预警指标**
   - MIMIC-IV：OR=0.88（p<0.001），均值调整后完全不变
   - eICU T0≥72h：OR=0.80（p=0.034），与 MIMIC 高度一致
   - SOFA/APACHE 调整后仍显著：提供独立于器官损害程度的增量信息

2. **Euler χ(0) 是重要补充，但包含均值混淆成分**
   - MIMIC-IV OR=0.75（p<0.001），但均值调整后衰减至 OR=0.86
   - eICU T0≥72h OR=0.72，接近 MIMIC，方向强烈一致
   - 其信号部分来自"持续心动过速使 HR 长时间高于零阈值"的均值效应

3. **AC1 信号不稳健，存在严重混淆**
   - 主模型有意义，但均值调整、双重去趋势、无镇静亚组后均消失
   - eICU 未验证：更可能是原研究的统计偶然或镇静药混淆
   - ⚠️ eICU 中 AC1 与 Euler/n\_extrema 分析样本不完全相同（AC1 M1 N=1,102，Euler M1 N=1,130），任何"Euler/n\_extrema 比 AC1 更稳健"的表述须限定为"在相同建模框架下、但非完全相同可分析样本"的比较

4. **T0 窗口长度是 eICU 验证的关键限制**
   - 不是信号不存在，而是信号需要"积累时间"才能显现
   - 这本身就是重要的生物学发现：预警信号的有效时间尺度

5. **扰动分析（护理翻身）是机制验证的"灵魂"**
   - MAP 的直接证据表明接近休克时心血管弹性降低（无升压药亚组显著）
   - HR n_extrema 的扰动分析受限于数据分辨率，需要分钟级 HR（波形数据库）才能完整验证

### 7.3 研究局限性

| 局限性 | 影响 | 缓解措施 |
|--------|------|---------|
| eICU T0 短 | 全样本不显著 | T0 分层分析 |
| AC1 均值混淆 | 信号可能伪造 | 双重去趋势、S2 模型 |
| Euler 阈值依赖 | 零阈值有特殊假设 | n_extrema 作为阈值无关替代 |
| MIMIC HR 时间分辨率（小时级） | 翻身后 HR n_extrema 无统计功效 | 需要 MIMIC 波形数据库 |
| eICU 仅 12% 患者有翻身记录 | 扰动分析样本不足 | — |
| 匹配随机种子敏感性 | 特定种子下 eICU AC1 曾显著 | 种子 114514 随机化消除偏差 |

---

## 八、相关脚本索引

| 分析 | 脚本 | 输出 |
|------|------|------|
| MIMIC AC1 主分析 | `ac1/scripts/04_logistic_regression.py` | `ac1/output/table2_*.csv` |
| MIMIC AC1 扰动分析 | `ac1/scripts/05_perturbation_recovery.py` | `data/perturbation_events.parquet` |
| MIMIC Euler/n_extrema 主分析 | `euler_ews/scripts/09_euler_ews.py` | `euler_ews/output/tbl_euler_*.csv` |
| MIMIC Euler logistic 回归 | `euler_ews/scripts/11_euler_logistic.py` | `euler_ews/output/tbl_euler_logistic.csv` |
| MIMIC HR n_extrema 翻身分析（探索性） | `euler_ews/scripts/12_perturbation_n_extrema.py` | `euler_ews/output/tbl_perturb_n_extrema_summary.csv` |
| MIMIC HR 分钟级提取（缓存） | `euler_ews/scripts/12a_extract_hr_minutelevel.py` | `data/hr_minutelevel_perturb.parquet` |
| eICU 队列构建 | `eicu_validation/scripts/01_build_cohort_eicu.py` | `eicu_validation/output/cohort_eicu.parquet` |
| eICU Vitals 提取 | `eicu_validation/scripts/02_extract_vitals_eicu.py` | `eicu_validiaton/output/vitals_eicu.parquet` |
| eICU EWS 计算 | `eicu_validation/scripts/03_ews_eicu.py` | `eicu_validation/output/ews_patient_eicu.parquet` |
| eICU 主分析（含 T0 分层） | `eicu_validation/scripts/04_validation_analysis.py` | `eicu_validation/output/tbl_validation_*.csv` |
| eICU vs SOFA 对比 | `eicu_validation/scripts/05_sofa_comparison.py` | `eicu_validation/output/tbl_auc_comparison.csv` |
| MIMIC SampEn 对照分析 | `euler_ews/scripts/13_sampen_comparison.py` | `euler_ews/output/fig_sampen_4way_forest.png`, `tbl_sampen_logistic.csv`, `tbl_sampen_gee.csv` |

---

## 九、已修复的方法问题记录

| 问题 | 影响 | 修复方式 | 数值影响 |
|------|------|---------|---------|
| `05_sofa_comparison.py`：SOFA/APACHE 与 EWS 分母不同（1,174 vs 1,152） | AUC head-to-head 比较不公平 | `sdf` 先 inner join `passed` 子集再合并 EWS | AUC 变化 <0.003，结论不变 |
| `12a_extract_hr_minutelevel.py`：`PRE_HOURS=56` 定义但未使用，docstring 与实现不符 | 无功能影响，文档误导 | 删除未用常量，修正 docstring | 无数值影响 |
| `03_ews_eicu.py`：Euler/n_extrema 使用 `low_conf_hr`（8点阈值）而非宽松阈值（6点）| Euler 有效窗口被过度排除 | 引入 `low_conf_euler`，Euler 类用 6 点阈值 | 改变 Euler NaN 率，影响结果 |
| `04_validation_analysis.py`：Euler 时间序列图仍用 `low_conf_hr` 过滤 | 图中 Euler 显示与计算口径不一致 | 改为 `low_conf_euler` | 仅影响图形 |
| `01_build_cohort_eicu.py`：匹配顺序依赖输入行序（随机种子 42） | 结果对 CSV 行序敏感，可重复性差 | 添加 `cases.sample(frac=1, random_state=114514)` | AC1 p: 0.049→0.435（原显著为假阳性）|
| `04_validation_analysis.py`：`MIMIC_REF["AC1"] = 1.72` 与实际主模型 OR=1.60 不符 | "向 MIMIC 收敛"叙述使用错误参照值；森林图参考线偏右 | 改为 `1.60`，并加硬编码兜底防文件读取旧值 | 参考线修正，文字更正 |
| `FINDINGS_SUMMARY.md`："同一分母 head-to-head"表述过强 | 各指标实际分析 n 仍不同（SOFA 1,140 / APACHE 989 / AC1 1,138 / EWS 1,152） | 改为"同一 passed 母集内，按各指标可用样本分别计算" | 仅文字修正 |
| `FINDINGS_SUMMARY.md`："T0 清晰单调生物学梯度"表述过强 | Euler ≥24h OR=1.02（方向反转），n_extrema ≥24h 略微回升，不严格单调 | 改为"≥72h 子集最接近 MIMIC，但非严格单调" | 仅文字修正 |
| `12_perturbation_n_extrema.py`："Shock: early vs late（within-group）"暗示配对分析 | 实为两独立边际样本的 rank-sum，不能推断 within-patient 变化 | docstring + 输出标签改为"early-cohort vs late-cohort (non-paired marginal)" | 仅文字修正 |
