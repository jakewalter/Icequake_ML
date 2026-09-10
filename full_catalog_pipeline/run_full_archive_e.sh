#!/bin/bash
# Full 14-station/2-year archive run using model E (SNR>=2dB, >=3 stations),
# the winner of the A/B/C/D/E ablation (see snr_station_ablation memory).
# Detection is per-station and array-agnostic, but ASSOCIATION is done
# SEPARATELY for T1 and T2 -- they are distinct physical arrays on the
# glacier with their own QuakeMigrate catalogs, so pooling all 14 stations
# into one pyocto associator would build one oversized bounding box and risk
# spurious cross-array associations. Each array is associated and compared
# against its own QuakeMigrate catalog independently.
set -uo pipefail
cd /home/jwalter/Icequake_ML

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

run_or_die() {
  "$@"
  status=$?
  if [ "$status" -ne 0 ]; then
    log "COMMAND FAILED (exit $status): $*"
    exit 1
  fi
}

CKPT=full_catalog_pipeline/checkpoints_E/best_model_state_only.pth
START_DATE=2019-12-29
END_DATE=2021-12-26
T1_STATIONS="DEEJ ELZA LILA LOUS OTIS SQIG TJTJ"
T2_STATIONS="BAUM DRSC EPJZ FRST JULA OKGS WICH"

mkdir -p full_catalog_pipeline/artifacts/full_run/T1 full_catalog_pipeline/artifacts/full_run/T2

# ---------- Stage 1: launch continuous detection, all 14 stations in parallel ----------
log "=== STAGE 1: launching full-archive detection (14 stations, $START_DATE..$END_DATE, model E) ==="
PIDS=""
for STA in $T1_STATIONS; do
  nohup python3 full_catalog_pipeline/batch_classify.py \
    --stations $STA \
    --start-date $START_DATE --end-date $END_DATE \
    --checkpoint $CKPT \
    --out-csv full_catalog_pipeline/artifacts/full_run/T1/picks_${STA}_full.csv \
    > full_catalog_pipeline/artifacts/full_run/T1/log_${STA}.log 2>&1 &
  PIDS="$PIDS $!"
done
for STA in $T2_STATIONS; do
  nohup python3 full_catalog_pipeline/batch_classify.py \
    --stations $STA \
    --start-date $START_DATE --end-date $END_DATE \
    --checkpoint $CKPT \
    --out-csv full_catalog_pipeline/artifacts/full_run/T2/picks_${STA}_full.csv \
    > full_catalog_pipeline/artifacts/full_run/T2/log_${STA}.log 2>&1 &
  PIDS="$PIDS $!"
done
log "  launched PIDs:$PIDS"

# ---------- Stage 2: wait for all 14 to finish (this is the multi-hour part) ----------
log "=== STAGE 2: waiting for all 14 station-detection processes to finish ==="
for p in $PIDS; do wait $p; done
log "  full-archive detection finished"

# ---------- Stage 3: associate T1 picks (pyocto, T1 array only) ----------
log "=== STAGE 3: pyocto association, T1 array only (n_p_and_s_picks=3) ==="
run_or_die python3 full_catalog_pipeline/associate_pyocto.py \
  --picks-glob "full_catalog_pipeline/artifacts/full_run/T1/picks_*_full.csv" \
  --out-events full_catalog_pipeline/artifacts/full_run/T1/pyocto_events.csv \
  --out-assignments full_catalog_pipeline/artifacts/full_run/T1/pyocto_assignments.csv \
  --velocity-model-path full_catalog_pipeline/artifacts/full_run/T1/velocity_model \
  --n-p-and-s-picks 3 2>&1 | grep -v Warning

# ---------- Stage 4: associate T2 picks (pyocto, T2 array only) ----------
log "=== STAGE 4: pyocto association, T2 array only (n_p_and_s_picks=3) ==="
run_or_die python3 full_catalog_pipeline/associate_pyocto.py \
  --picks-glob "full_catalog_pipeline/artifacts/full_run/T2/picks_*_full.csv" \
  --out-events full_catalog_pipeline/artifacts/full_run/T2/pyocto_events.csv \
  --out-assignments full_catalog_pipeline/artifacts/full_run/T2/pyocto_assignments.csv \
  --velocity-model-path full_catalog_pipeline/artifacts/full_run/T2/velocity_model \
  --n-p-and-s-picks 3 2>&1 | grep -v Warning

# ---------- Stage 5: compare T1 catalog vs T1 QuakeMigrate ----------
log "=== STAGE 5: comparing T1 catalog to QuakeMigrate T1 ==="
run_or_die python3 full_catalog_pipeline/compare_catalogs.py \
  --pyocto-events full_catalog_pipeline/artifacts/full_run/T1/pyocto_events.csv \
  --qm-events-json /scratch2/qm/t1/output/working_files/events.json \
  --start-date $START_DATE --end-date $END_DATE --tol-sec 5.0 \
  2>&1 | grep -v Warning

# ---------- Stage 6: compare T2 catalog vs T2 QuakeMigrate ----------
log "=== STAGE 6: comparing T2 catalog to QuakeMigrate T2 ==="
run_or_die python3 full_catalog_pipeline/compare_catalogs.py \
  --pyocto-events full_catalog_pipeline/artifacts/full_run/T2/pyocto_events.csv \
  --qm-events-json /scratch2/qm/t2/quakeml/output/working_files/events.json \
  --start-date $START_DATE --end-date $END_DATE --tol-sec 5.0 \
  2>&1 | grep -v Warning

log "FULL_ARCHIVE_COMPLETE"
