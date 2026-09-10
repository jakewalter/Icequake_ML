#!/bin/bash
# Wait for the pilot detection run to finish, then associate + compare it
# against QuakeMigrate. Stops here -- does NOT launch the full-archive run.
set -uo pipefail
cd /home/jwalter/Icequake_ML

PILOT_PIDS="3085716 3085717 3085718 3085719 3085720 3085721 3085722"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "=== STAGE 1: waiting for pilot detection PIDs to exit ==="
while true; do
  alive=0
  for p in $PILOT_PIDS; do
    [ -d "/proc/$p" ] && alive=$((alive+1))
  done
  [ "$alive" -eq 0 ] && break
  sleep 15
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
  log "PILOT_ASSOCIATION_FAILED (exit $assoc_status)"
  exit 1
fi

log "=== STAGE 3: comparing pilot catalog to QuakeMigrate T2 (July 2020) ==="
python3 full_catalog_pipeline/compare_catalogs.py \
  --pyocto-events full_catalog_pipeline/artifacts/pilot/pyocto_events.csv \
  --qm-events-json /scratch2/qm/t2/quakeml/output/working_files/events.json \
  --start-date 2020-07-01 --end-date 2020-07-31 --tol-sec 5.0 \
  2>&1 | grep -v Warning
log "PILOT_COMPARISON_DONE"
log "=== STOPPING HERE (full-archive run NOT launched -- awaiting review) ==="
