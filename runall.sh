#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/logs"

RUN_TS="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$ROOT_DIR/logs/rerun_all_${RUN_TS}.log"
RUN_SETUP="auto"

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

usage() {
  cat <<'EOF'
Usage: bash runall.sh [--with-setup] [--force-setup] [--skip-setup]

  --with-setup   Always run scripts/01_setup_db.py before the analysis pipeline
  --force-setup  Alias of --with-setup
  --skip-setup   Never run scripts/01_setup_db.py automatically
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-setup|--force-setup)
      RUN_SETUP="always"
      shift
      ;;
    --skip-setup)
      RUN_SETUP="never"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

should_run_setup() {
  if [[ "$RUN_SETUP" == "always" ]]; then
    return 0
  fi
  if [[ "$RUN_SETUP" == "never" ]]; then
    return 1
  fi

  python3 - <<'PY'
from pathlib import Path
import duckdb
import sys

root = Path.cwd()
db_path = root / "mimiciv" / "mimiciv.db"
if not db_path.exists():
    sys.exit(1)

try:
    con = duckdb.connect(str(db_path), read_only=True)
    con.execute("SELECT COUNT(*) FROM mimiciv_icu.icustays").fetchone()
    con.execute("SELECT COUNT(*) FROM mimiciv_hosp.patients").fetchone()
    con.execute("SELECT COUNT(*) FROM mimiciv_derived.sepsis3").fetchone()
    con.close()
except Exception:
    sys.exit(1)

sys.exit(0)
PY
}

echo "Master log : $LOG_FILE"
echo "Project root: $ROOT_DIR"
echo "Start time : $(date '+%F %T')"

# 说明:
# - 默认先做数据库预检; 缺库、schema 缺失、或旧机器绝对路径失效时会自动重跑 Step 1
# - 若中途任一步失败，脚本会立即退出
# - 所有 stdout/stderr 会完整保存到 logs/rerun_all_*.log
# - 建议在项目根目录执行: bash runall.sh

if should_run_setup; then
  run_step "Step 1: setup DuckDB schemas and derived concepts" \
    python3 scripts/01_setup_db.py
else
  echo
  echo "============================================================"
  echo "[$(date '+%F %T')] Step 1: setup DuckDB schemas and derived concepts"
  echo "============================================================"
  echo "Skipping Step 1: existing DuckDB database passed preflight checks"
fi

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
echo "  - output/tableS4_late_vitals_sensitivity.csv"
echo "  - output/tableS5_early_window_independence.csv"
echo "  - output/tableS6_no_sedation_subgroup.csv"
echo "  - output/tableS7_perturbation_summary.csv"
echo "  - output/fig1_timeseries.png"
echo "  - output/fig2_delta_boxplot.png"
echo "  - output/figS1_subgroup.png"
echo "  - output/fig3_recovery.png"
echo "  - report.pdf"
