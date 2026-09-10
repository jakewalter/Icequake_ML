#!/bin/bash
# A/B/C SNR-threshold x min-stations-per-event ablation, run unattended.
# A (baseline) is already trained; this script handles B, C, then
# re-associates all three under the same relaxed pyocto rule and builds
# comparison plots. Does NOT launch the full-archive run.
set -uo pipefail
cd /home/jwalter/Icequake_ML

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

CKPT_A=full_catalog_pipeline/checkpoints/best_model_state_only.pth
STATIONS="BAUM DRSC EPJZ FRST JULA OKGS WICH"
QM_T2=/scratch2/qm/t2/quakeml/output/working_files/events.json

run_or_die() {
  "$@"
  status=$?
  if [ "$status" -ne 0 ]; then
    log "COMMAND FAILED (exit $status): $*"
    exit 1
  fi
}

# ---------- Stage 1: curate dataset C ----------
log "=== STAGE 1: curating dataset C (SNR>=3dB, min_stations>=3) ==="
run_or_die env FCP_RUN_SUFFIX=_C python3 full_catalog_pipeline/04a_two_stage_snr_filter.py \
  --window-threshold-db 3.0 --min-stations 3 --force
run_or_die env FCP_RUN_SUFFIX=_C python3 full_catalog_pipeline/05_pack_seisbench.py --force

# ---------- Stage 2: train B and C ----------
log "=== STAGE 2: training model B ==="
run_or_die python3 full_catalog_pipeline/train.py \
  --config full_catalog_pipeline/train_config_B.json \
  --checkpoint-dir full_catalog_pipeline/checkpoints_B/
log "model B training complete"

log "=== STAGE 2b: training model C ==="
run_or_die python3 full_catalog_pipeline/train.py \
  --config full_catalog_pipeline/train_config_C.json \
  --checkpoint-dir full_catalog_pipeline/checkpoints_C/
log "model C training complete"

# ---------- Stage 3: pilot continuous detection for B and C ----------
for VARIANT in B C; do
  log "=== STAGE 3 ($VARIANT): pilot continuous detection, T2 array, July 2020 ==="
  mkdir -p full_catalog_pipeline/artifacts/pilot_$VARIANT
  CKPT=full_catalog_pipeline/checkpoints_${VARIANT}/best_model_state_only.pth
  PIDS=""
  for STA in $STATIONS; do
    nohup python3 full_catalog_pipeline/batch_classify.py \
      --stations $STA \
      --start-date 2020-07-01 --end-date 2020-07-31 \
      --checkpoint $CKPT \
      --out-csv full_catalog_pipeline/artifacts/pilot_${VARIANT}/picks_${STA}_202007.csv \
      > full_catalog_pipeline/artifacts/pilot_${VARIANT}/log_${STA}.log 2>&1 &
    PIDS="$PIDS $!"
  done
  log "  launched PIDs:$PIDS"
  for p in $PIDS; do wait $p; done
  log "  ($VARIANT) pilot detection finished"
done

# ---------- Stage 4: associate A/B/C with pyocto (n_p_and_s_picks=3) ----------
log "=== STAGE 4: pyocto association (n_p_and_s_picks=3) for A, B, C ==="
run_or_die python3 full_catalog_pipeline/associate_pyocto.py \
  --picks-glob "full_catalog_pipeline/artifacts/pilot/picks_*_202007.csv" \
  --out-events full_catalog_pipeline/artifacts/pilot/pyocto_events_naps3.csv \
  --out-assignments full_catalog_pipeline/artifacts/pilot/pyocto_assignments_naps3.csv \
  --n-p-and-s-picks 3 2>&1 | grep -v Warning

for VARIANT in B C; do
  run_or_die python3 full_catalog_pipeline/associate_pyocto.py \
    --picks-glob "full_catalog_pipeline/artifacts/pilot_${VARIANT}/picks_*_202007.csv" \
    --out-events full_catalog_pipeline/artifacts/pilot_${VARIANT}/pyocto_events.csv \
    --out-assignments full_catalog_pipeline/artifacts/pilot_${VARIANT}/pyocto_assignments.csv \
    --n-p-and-s-picks 3 2>&1 | grep -v Warning
done

# ---------- Stage 5: plots per variant ----------
log "=== STAGE 5: building comparison plots for A, B, C ==="
run_or_die python3 full_catalog_pipeline/plot_pilot_comparison.py \
  --pyocto-events full_catalog_pipeline/artifacts/pilot/pyocto_events_naps3.csv \
  --out-dir full_catalog_pipeline/artifacts/pilot \
  --label "Model A (SNR>=5dB, >=4 stations)" 2>&1 | grep -v Warning

run_or_die python3 full_catalog_pipeline/plot_pilot_comparison.py \
  --pyocto-events full_catalog_pipeline/artifacts/pilot_B/pyocto_events.csv \
  --out-dir full_catalog_pipeline/artifacts/pilot_B \
  --label "Model B (SNR>=5dB, >=3 stations)" 2>&1 | grep -v Warning

run_or_die python3 full_catalog_pipeline/plot_pilot_comparison.py \
  --pyocto-events full_catalog_pipeline/artifacts/pilot_C/pyocto_events.csv \
  --out-dir full_catalog_pipeline/artifacts/pilot_C \
  --label "Model C (SNR>=3dB, >=3 stations)" 2>&1 | grep -v Warning

log "=== STAGE 6: building 3-way summary figure ==="
run_or_die python3 full_catalog_pipeline/plot_abc_summary.py

log "ABLATION_COMPLETE"
