#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/logs"

RUN_TS="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$ROOT_DIR/logs/rerun_all_${RUN_TS}.log"

# 全程实时显示，同时完整写入总日志
exec > >(tee -a "$LOG_FILE") 2>&1

run_step() {
  local label="$1"
  shift
  echo
  echo "============================================================"
  echo "[$(date '+%F %T')] $label"
  echo "============================================================"
  "$@"
}

echo "Master log : $LOG_FILE"
echo "Project root: $ROOT_DIR"
echo "Start time : $(date '+%F %T')"

# 说明:
# - 默认不重跑 Step 1（建库），避免重复耗时
# - 若中途任一步失败，脚本会立即退出
# - 所有 stdout/stderr 会完整保存到 logs/rerun_all_*.log
# - 建议在项目根目录执行: bash rerun_all.sh

run_step "Step 2: rebuild cohort" \
  python3 scripts/02_build_cohort.py

run_step "Step 3: extract and clean vitals" \
  python3 scripts/03_extract_and_clean.py

run_step "Step 4: EWS analysis" \
  python3 scripts/04_ews_analysis.py

run_step "Step 5: perturbation recovery" \
  python3 scripts/05_perturbation_recovery.py

run_step "Step 6: baseline table" \
  python3 scripts/06_baseline_table.py

run_step "Step 7: multivariable model" \
  python3 scripts/07_multivariable_model.py

run_step "Step 8: cluster sensitivity" \
  python3 scripts/08_cluster_sensitivity.py

run_step "Compile report (pass 1)" \
  pdflatex -interaction=nonstopmode report.tex

run_step "Compile report (pass 2)" \
  pdflatex -interaction=nonstopmode report.tex

echo
echo "Done."
echo "End time   : $(date '+%F %T')"
echo "Master log : $LOG_FILE"
echo "Key outputs:"
echo "  - data/cohort.parquet"
echo "  - data/vitals_cleaned.parquet"
echo "  - data/ews_windows.parquet"
echo "  - output/table1_baseline.csv"
echo "  - output/table2_multivariable.csv"
echo "  - output/table3_ews_comparison.csv"
echo "  - output/tableS1_gee.csv"
echo "  - output/tableS2_dedup.csv"
echo "  - output/tableS3_bootstrap.csv"
echo "  - output/tableS4_conditional_logistic.csv"
echo "  - report.pdf"
