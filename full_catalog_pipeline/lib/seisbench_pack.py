"""SeisBench packing helpers: event-stratified train/dev/test split (fixes
08_pack_mseed_to_seisbench.py's row-index-based split, which lets the same
event leak across splits via its different stations) and pick-method lookup
for provenance-correct trace_p_status/trace_s_status.
"""

import csv

import numpy as np

import config


def assign_splits_by_event(event_ids, split_ratios=None, random_seed=None):
    """Same floor/remainder arithmetic as 08's assign_dataset_splits, but
    operating on the list of UNIQUE event_ids so every station-event row for
    a given event resolves to the same split."""
    if split_ratios is None:
        split_ratios = config.SPLIT_RATIOS
    if random_seed is None:
        random_seed = config.RANDOM_SEED

    unique_events = sorted(set(event_ids))
    n_events = len(unique_events)

    np.random.seed(random_seed)
    shuffled_events = list(unique_events)
    np.random.shuffle(shuffled_events)

    n_train = int(np.floor(n_events * split_ratios["train"]))
    n_dev = int(np.floor(n_events * split_ratios["dev"]))

    splits = {}
    for i, event_id in enumerate(shuffled_events):
        if i < n_train:
            splits[event_id] = "train"
        elif i < n_train + n_dev:
            splits[event_id] = "dev"
        else:
            splits[event_id] = "test"
    return splits


def load_pick_methods():
    """Returns {(event_id, station, phase): method_id} from picks_deduped.csv."""
    methods = {}
    with open(config.PICKS_DEDUPED_CSV) as f:
        for row in csv.DictReader(f):
            methods[(row["event_id"], row["station"], row["phase"])] = row["method_id"]
    return methods
