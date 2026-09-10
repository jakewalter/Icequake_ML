#!/bin/bash
# Follow-on to A/B/C/D: D (SNR>=1dB) reversed and came in worse than C
# (SNR>=3dB) on both precision and recall, showing the curve is not
# monotonic in the SNR floor. E fills in the gap at 2 dB, holding the
# min-stations floor at 3 (same as C/D), to bracket where the true peak
# sits between C and D.
# 04c_two_stage_filtering_analysis.py --window-threshold 2.0 shows 91,855
# windows / 20,830 events qualify at this threshold (vs C: 58,977/14,881,
# D: 133,266/24,561) -- squarely between C and D as intended.
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

# ---------- Stage 1: curate dataset E ----------
log "=== STAGE 1: curating dataset E (SNR>=2dB, min_stations>=3) ==="
run_or_die env FCP_RUN_SUFFIX=_E python3 full_catalog_pipeline/04a_two_stage_snr_filter.py \
  --window-threshold-db 2.0 --min-stations 3 --force
run_or_die env FCP_RUN_SUFFIX=_E python3 full_catalog_pipeline/05_pack_seisbench.py --force

# ---------- Stage 2: train E ----------
log "=== STAGE 2: training model E ==="
run_or_die python3 full_catalog_pipeline/train.py \
  --config full_catalog_pipeline/train_config_E.json \
  --checkpoint-dir full_catalog_pipeline/checkpoints_E/
log "model E training complete"

# ---------- Stage 3: pilot continuous detection for E ----------
log "=== STAGE 3: pilot continuous detection, T2 array, July 2020 ==="
mkdir -p full_catalog_pipeline/artifacts/pilot_E
CKPT=full_catalog_pipeline/checkpoints_E/best_model_state_only.pth
PIDS=""
for STA in $STATIONS; do
  nohup python3 full_catalog_pipeline/batch_classify.py \
    --stations $STA \
    --start-date 2020-07-01 --end-date 2020-07-31 \
    --checkpoint $CKPT \
    --out-csv full_catalog_pipeline/artifacts/pilot_E/picks_${STA}_202007.csv \
    > full_catalog_pipeline/artifacts/pilot_E/log_${STA}.log 2>&1 &
  PIDS="$PIDS $!"
done
log "  launched PIDs:$PIDS"
for p in $PIDS; do wait $p; done
log "  (E) pilot detection finished"

# ---------- Stage 4: associate E with pyocto (n_p_and_s_picks=3) ----------
log "=== STAGE 4: pyocto association (n_p_and_s_picks=3) for E ==="
run_or_die python3 full_catalog_pipeline/associate_pyocto.py \
  --picks-glob "full_catalog_pipeline/artifacts/pilot_E/picks_*_202007.csv" \
  --out-events full_catalog_pipeline/artifacts/pilot_E/pyocto_events.csv \
  --out-assignments full_catalog_pipeline/artifacts/pilot_E/pyocto_assignments.csv \
  --n-p-and-s-picks 3 2>&1 | grep -v Warning

# ---------- Stage 5: comparison plot for E ----------
log "=== STAGE 5: building comparison plots for E ==="
run_or_die python3 full_catalog_pipeline/plot_pilot_comparison.py \
  --pyocto-events full_catalog_pipeline/artifacts/pilot_E/pyocto_events.csv \
  --out-dir full_catalog_pipeline/artifacts/pilot_E \
  --label "Model E (SNR>=2dB, >=3 stations)" 2>&1 | grep -v Warning

# ---------- Stage 6: 5-way summary figure (A/B/C/D/E) ----------
log "=== STAGE 6: building 5-way summary figure ==="
run_or_die python3 full_catalog_pipeline/plot_abc_summary.py

log "ABLATION_E_COMPLETE"
