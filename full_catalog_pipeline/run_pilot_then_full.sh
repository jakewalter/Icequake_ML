#!/bin/bash
# Chained driver: wait for the pilot detection run to finish, associate +
# compare it against QuakeMigrate, then automatically launch and process the
# full-archive run (all 14 stations, full 2019-12-29..2021-12-26 continuous
# archive) the same way. Designed to run unattended via nohup.
set -uo pipefail
cd /home/jwalter/Icequake_ML

CKPT=full_catalog_pipeline/checkpoints/best_model_state_only.pth
PILOT_PIDS="3085716 3085717 3085718 3085719 3085720 3085721 3085722"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "=== STAGE 1: waiting for pilot detection PIDs to exit ==="
while true; do
  alive=0
  for p in $PILOT_PIDS; do
    [ -d "/proc/$p" ] && alive=$((alive+1))
  done
  [ "$alive" -eq 0 ] && break
  sleep 20
done
log "pilot detection finished"

log "=== STAGE 2: pyocto association on pilot picks (T2, July 2020) ==="
python3 full_catalog_pipeline/associate_pyocto.py \
  --picks-glob "full_catalog_pipeline/artifacts/pilot/picks_*_202007.csv" \
  --out-events full_catalog_pipeline/artifacts/pilot/pyocto_events.csv \
  --out-assignments full_catalog_pipeline/artifacts/pilot/pyocto_assignments.csv \
  2>&1 | grep -v Warning
assoc_status=${PIPESTATUS[0]}
if [ "$assoc_status" -ne 0 ]; then
  log "PILOT_ASSOCIATION_FAILED (exit $assoc_status) -- aborting chain, NOT launching full archive run"
  exit 1
fi

log "=== STAGE 3: comparing pilot catalog to QuakeMigrate T2 (July 2020) ==="
python3 full_catalog_pipeline/compare_catalogs.py \
  --pyocto-events full_catalog_pipeline/artifacts/pilot/pyocto_events.csv \
  --qm-events-json /scratch2/qm/t2/quakeml/output/working_files/events.json \
  --start-date 2020-07-01 --end-date 2020-07-31 --tol-sec 5.0 \
  2>&1 | grep -v Warning
log "PILOT_COMPARISON_DONE"

log "=== STAGE 4: launching FULL-ARCHIVE detection (14 stations, 2019-12-29..2021-12-26) ==="
mkdir -p full_catalog_pipeline/artifacts/full_run
STATIONS="BAUM DRSC EPJZ FRST JULA OKGS WICH DEEJ ELZA LILA LOUS OTIS SQIG TJTJ"
FULL_PIDS=""
for STA in $STATIONS; do
  nohup python3 full_catalog_pipeline/batch_classify.py \
    --stations $STA \
    --start-date 2019-12-29 --end-date 2021-12-26 \
    --checkpoint $CKPT \
    --out-csv full_catalog_pipeline/artifacts/full_run/picks_${STA}_full.csv \
    > full_catalog_pipeline/artifacts/full_run/log_${STA}.log 2>&1 &
  FULL_PIDS="$FULL_PIDS $!"
done
log "launched full-archive PIDs: $FULL_PIDS"

log "=== STAGE 5: waiting for full-archive detection to finish (this will take hours) ==="
while true; do
  alive=0
  for p in $FULL_PIDS; do
    [ -d "/proc/$p" ] && alive=$((alive+1))
  done
  [ "$alive" -eq 0 ] && break
  sleep 120
done
log "full-archive detection finished"

log "=== STAGE 6: pyocto association on full-archive picks (all 14 stations) ==="
python3 full_catalog_pipeline/associate_pyocto.py \
  --picks-glob "full_catalog_pipeline/artifacts/full_run/picks_*_full.csv" \
  --out-events full_catalog_pipeline/artifacts/full_run/pyocto_events.csv \
  --out-assignments full_catalog_pipeline/artifacts/full_run/pyocto_assignments.csv \
  2>&1 | grep -v Warning
assoc_status=${PIPESTATUS[0]}
if [ "$assoc_status" -ne 0 ]; then
  log "FULL_ASSOCIATION_FAILED (exit $assoc_status)"
  exit 1
fi

log "=== STAGE 7: comparing full-archive catalog to combined QuakeMigrate T1+T2 ==="
python3 full_catalog_pipeline/compare_catalogs.py \
  --pyocto-events full_catalog_pipeline/artifacts/full_run/pyocto_events.csv \
  --qm-events-json "/scratch2/qm/t1/output/working_files/events.json,/scratch2/qm/t2/quakeml/output/working_files/events.json" \
  --start-date 2019-12-29 --end-date 2021-12-26 --tol-sec 5.0 \
  2>&1 | grep -v Warning
log "FULL_COMPARISON_DONE"

log "=== ALL STAGES COMPLETE ==="
