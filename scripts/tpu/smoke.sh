#!/usr/bin/env bash
# Source runtime.sh first; TASK and SMOKE_PREFIX are set by the TPUQ script.
set -euo pipefail
export WANDB_MODE=disabled
case "$TASK" in
    square) start=1201; end=1204 ;;
    toolhang) start=3001; end=3004 ;;
    transport) start=2401; end=2404 ;;
esac
args=(--task "$TASK" --seed 1 --run-id "ogpo-smoke-$TASK" --group ogpo-smoke
      --prefix "$SMOKE_PREFIX" --data-prefix "$DATA_PREFIX"
      --bc_pi_steps=1 --start_training="$start" --log_interval=1
      --eval_interval_bc=10000 --preemption_backup_interval=4)
python scripts/tpu/run.py "${args[@]}" --online_steps="$end" --eval_interval="$end"
python scripts/tpu/check_smoke.py "$end"
# Force the second process to recover from GCS, preserving the BC artifact.
python - <<'PY'
import os, pathlib, shutil
shutil.rmtree(pathlib.Path(os.environ['TPUQ_TASK_ID']) / 'checkpoints')
PY
python scripts/tpu/run.py "${args[@]}" --online_steps="$((end + 4))" --eval_interval=4
python scripts/tpu/check_smoke.py "$((end + 4))"
echo SMOKE_COMPLETE "$TASK"
