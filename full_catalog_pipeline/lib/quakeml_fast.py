"""Fast ElementTree-based QuakeML pick extraction.

Adapted from quakeml_to_seisbench.py, but emits one row per raw pick (not
pre-aggregated per station/phase) so dedup can see all candidates, and is
generalized to tag rows with an array label (T1/T2). obspy.read_events() is
too slow for full-scale catalogs (~3 min for T1 alone) so this bypasses obspy
entirely for parsing.
"""

import xml.etree.ElementTree as ET
from collections import defaultdict

NAMESPACES = {
    "q": "http://quakeml.org/xmlns/quakeml/1.2",
    "bed": "http://quakeml.org/xmlns/bed/1.2",
}

METHOD_PREFERENCE = {"modelled": 0, "autopick": 1}


def _strip_smi(value):
    if value and "smi:local/" in value:
        return value.split("smi:local/")[-1]
    return value


def parse_quakeml_picks(xml_path, array_label, network):
    """Parse a QuakeML file into a flat list of raw pick rows.

    Returns one dict per <pick> element:
    event_id, array, origin_time, latitude, longitude, depth_m,
    station, network, phase, pick_time, method_id
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []
    for event in root.findall(".//bed:event", NAMESPACES):
        event_id = _strip_smi(event.get("publicID"))

        origin = event.find("bed:origin", NAMESPACES)
        origin_time = latitude = longitude = depth_m = None
        if origin is not None:
            time_elem = origin.find("bed:time/bed:value", NAMESPACES)
            if time_elem is not None:
                origin_time = time_elem.text
            lat_elem = origin.find("bed:latitude/bed:value", NAMESPACES)
            if lat_elem is not None:
                latitude = float(lat_elem.text)
            lon_elem = origin.find("bed:longitude/bed:value", NAMESPACES)
            if lon_elem is not None:
                longitude = float(lon_elem.text)
            depth_elem = origin.find("bed:depth/bed:value", NAMESPACES)
            if depth_elem is not None:
                depth_m = float(depth_elem.text)

        for pick in event.findall("bed:pick", NAMESPACES):
            phase_elem = pick.find("bed:phaseHint", NAMESPACES)
            time_elem = pick.find("bed:time/bed:value", NAMESPACES)
            waveform_elem = pick.find("bed:waveformID", NAMESPACES)
            method_elem = pick.find("bed:methodID", NAMESPACES)

            if phase_elem is None or time_elem is None or waveform_elem is None:
                continue

            station = waveform_elem.get("stationCode")
            if not station:
                continue

            rows.append({
                "event_id": event_id,
                "array": array_label,
                "origin_time": origin_time,
                "latitude": latitude,
                "longitude": longitude,
                "depth_m": depth_m,
                "station": station,
                "network": network,
                "phase": phase_elem.text,
                "pick_time": time_elem.text,
                "method_id": _strip_smi(method_elem.text if method_elem is not None else ""),
            })

    return rows


def dedupe_picks(rows):
    """Port of 02_organize_quakexml_file.py's selection logic on plain dicts.

    Groups by (event_id, station, phase); prefers method_id == "modelled",
    then "autopick", then the first-encountered row.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[(row["event_id"], row["station"], row["phase"])].append(row)

    deduped = []
    for candidates in groups.values():
        if len(candidates) == 1:
            deduped.append(candidates[0])
            continue
        best = min(
            candidates,
            key=lambda r: METHOD_PREFERENCE.get(r["method_id"], len(METHOD_PREFERENCE)),
        )
        deduped.append(best)

    return deduped
