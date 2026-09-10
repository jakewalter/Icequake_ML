#!/usr/bin/env python3
"""Convert a pyocto events/assignments CSV pair into a QuakeML catalog, in the
same style as QuakeMigrate's own exports (/scratch2/qm/{t1,t2}/*_all_events.quakeml):
one obspy Event per pyocto event, one preferred Origin (lat/lon/depth/time),
and one Pick per assigned station-phase, with matching Arrival entries under
the origin. This lets existing relocation pipelines (e.g.
/home/jwalter/seis/hypoDDpy, which reads catalogs via obspy.read_events and
pulls event.preferred_origin()/event.picks) consume the pyocto catalog the
same way they already consume the QuakeMigrate one.

Usage:
    python full_catalog_pipeline/pyocto_to_quakeml.py \
        --pyocto-events full_catalog_pipeline/artifacts/full_run/T1_v5/pyocto_events.csv \
        --pyocto-assignments full_catalog_pipeline/artifacts/full_run/T1_v5/pyocto_assignments.csv \
        --out-quakeml full_catalog_pipeline/artifacts/full_run/T1_v5/pyocto_events.quakeml
"""
import argparse

import pandas as pd
from obspy import UTCDateTime
from obspy.core.event import (
    Arrival,
    Catalog,
    CreationInfo,
    Event,
    Origin,
    Pick,
    WaveformStreamID,
)


def station_code(station_id):
    # pyocto station ids are like "7U.DEEJ." -- strip network code + trailing dot.
    return station_id.split(".")[1]


def build_catalog(events_df, assignments_df):
    catalog = Catalog()
    creation_info = CreationInfo(author="pyocto", version="0.1.9")
    picks_by_event = assignments_df.groupby("event_idx")

    for _, ev_row in events_df.iterrows():
        origin = Origin(
            time=UTCDateTime(ev_row["time"]),
            latitude=ev_row["latitude"],
            longitude=ev_row["longitude"],
            depth=ev_row["depth"] * 1000.0,  # km -> m
            method_id="smi:local/pyocto",
            evaluation_mode="automatic",
        )

        picks, arrivals = [], []
        if ev_row["idx"] in picks_by_event.groups:
            for _, pk_row in picks_by_event.get_group(ev_row["idx"]).iterrows():
                pick = Pick(
                    time=UTCDateTime(pk_row["time"]),
                    waveform_id=WaveformStreamID(network_code="", station_code=station_code(pk_row["station"])),
                    method_id="smi:local/pyocto_pick",
                    phase_hint=pk_row["phase"],
                )
                picks.append(pick)
                arrivals.append(Arrival(
                    pick_id=pick.resource_id,
                    phase=pk_row["phase"],
                    time_residual=pk_row["residual"],
                ))

        origin.arrivals = arrivals
        event = Event(
            origins=[origin],
            picks=picks,
            preferred_origin_id=origin.resource_id,
            creation_info=creation_info,
        )
        catalog.append(event)

    return catalog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyocto-events", required=True)
    ap.add_argument("--pyocto-assignments", required=True)
    ap.add_argument("--out-quakeml", required=True)
    args = ap.parse_args()

    events_df = pd.read_csv(args.pyocto_events)
    assignments_df = pd.read_csv(args.pyocto_assignments)

    catalog = build_catalog(events_df, assignments_df)
    catalog.write(args.out_quakeml, format="QUAKEML")
    print(f"Wrote {len(catalog)} events to {args.out_quakeml}")


if __name__ == "__main__":
    main()
