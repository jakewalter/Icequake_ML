#!/bin/bash
# Follow-on to the A/B/C ablation: how far can loosening go? D pushes the
# SNR floor down to 1 dB (near the noise floor) while holding the
# min-stations floor at 3 (same as C), isolating further SNR-axis loosening.
# At this threshold 04c_two_stage_filtering_analysis.py shows 24,561/25,175
# events (97.6%) qualify, so D is close to "no SNR filter at all" and is a
# good test of whether the recall/precision curve is still climbing or has
# started to collapse.
set -uo pipefail
cd /home/jwalter/Icequake_ML

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

STATIONS="BAUM DRSC EPJZ FRST JULA OKGS WICH"

run_or_die() {
  "$@"
  status=$?
  if [ "$status" -ne 0 ]; then
    log "COMMAND FAILED (exit $status): $*"
    exit 1
  fi
}

# ---------- Stage 1: curate dataset D ----------
log "=== STAGE 1: curating dataset D (SNR>=1dB, min_stations>=3) ==="
run_or_die env FCP_RUN_SUFFIX=_D python3 full_catalog_pipeline/04a_two_stage_snr_filter.py \
  --window-threshold-db 1.0 --min-stations 3 --force
run_or_die env FCP_RUN_SUFFIX=_D python3 full_catalog_pipeline/05_pack_seisbench.py --force

# ---------- Stage 2: train D ----------
log "=== STAGE 2: training model D ==="
run_or_die python3 full_catalog_pipeline/train.py \
  --config full_catalog_pipeline/train_config_D.json \
  --checkpoint-dir full_catalog_pipeline/checkpoints_D/
log "model D training complete"

# ---------- Stage 3: pilot continuous detection for D ----------
log "=== STAGE 3: pilot continuous detection, T2 array, July 2020 ==="
mkdir -p full_catalog_pipeline/artifacts/pilot_D
CKPT=full_catalog_pipeline/checkpoints_D/best_model_state_only.pth
PIDS=""
for STA in $STATIONS; do
  nohup python3 full_catalog_pipeline/batch_classify.py \
    --stations $STA \
    --start-date 2020-07-01 --end-date 2020-07-31 \
    --checkpoint $CKPT \
    --out-csv full_catalog_pipeline/artifacts/pilot_D/picks_${STA}_202007.csv \
    > full_catalog_pipeline/artifacts/pilot_D/log_${STA}.log 2>&1 &
  PIDS="$PIDS $!"
done
log "  launched PIDs:$PIDS"
for p in $PIDS; do wait $p; done
log "  (D) pilot detection finished"

# ---------- Stage 4: associate D with pyocto (n_p_and_s_picks=3) ----------
log "=== STAGE 4: pyocto association (n_p_and_s_picks=3) for D ==="
run_or_die python3 full_catalog_pipeline/associate_pyocto.py \
  --picks-glob "full_catalog_pipeline/artifacts/pilot_D/picks_*_202007.csv" \
  --out-events full_catalog_pipeline/artifacts/pilot_D/pyocto_events.csv \
  --out-assignments full_catalog_pipeline/artifacts/pilot_D/pyocto_assignments.csv \
  --n-p-and-s-picks 3 2>&1 | grep -v Warning

# ---------- Stage 5: comparison plot for D ----------
log "=== STAGE 5: building comparison plots for D ==="
run_or_die python3 full_catalog_pipeline/plot_pilot_comparison.py \
  --pyocto-events full_catalog_pipeline/artifacts/pilot_D/pyocto_events.csv \
  --out-dir full_catalog_pipeline/artifacts/pilot_D \
  --label "Model D (SNR>=1dB, >=3 stations)" 2>&1 | grep -v Warning

# ---------- Stage 6: 4-way summary figure (A/B/C/D) ----------
log "=== STAGE 6: building 4-way summary figure ==="
run_or_die python3 full_catalog_pipeline/plot_abc_summary.py

log "ABLATION_D_COMPLETE"
