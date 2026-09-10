#!/usr/bin/env python3
"""Evaluate a PhaseNet checkpoint on a SeisBench dataset's test split.

Computes, for P and S separately:
  - residual histogram (predicted peak time - ground-truth peak time)
  - recall: fraction of ground-truth peaks matched by a predicted peak within tol_sec
  - precision: fraction of predicted peaks matched by a ground-truth peak within tol_sec
  - F1

Works on any dataset (old small curated set or new full-catalog set) and any
checkpoint, so the same script does both "test new model on new data" and
"compare old vs new model on the same held-out set."

Usage:
    python full_catalog_pipeline/evaluate.py \
        --dataset full_catalog_pipeline/output \
        --checkpoint full_catalog_pipeline/checkpoints/best_model_state_only.pth \
        --label new_model_new_data \
        --out-json full_catalog_pipeline/artifacts/eval_new_model_new_data.json
"""
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks
from tqdm import tqdm

import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm


def load_model(checkpoint_path, device):
    model = sbm.PhaseNet(phases="PSN", norm="peak").to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def match_peaks(pred_times, true_times, tol_sec):
    """Greedy nearest-neighbor matching within tol_sec. Returns (n_matched, n_pred, n_true)."""
    if len(true_times) == 0 or len(pred_times) == 0:
        return 0, len(pred_times), len(true_times)
    true_used = np.zeros(len(true_times), dtype=bool)
    n_matched = 0
    for pt in pred_times:
        diffs = np.abs(true_times - pt)
        diffs[true_used] = np.inf
        j = np.argmin(diffs)
        if diffs[j] <= tol_sec:
            true_used[j] = True
            n_matched += 1
    return n_matched, len(pred_times), len(true_times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="SeisBench dataset dir, e.g. full_catalog_pipeline/output")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--sampling-rate", type=float, default=200.0)
    ap.add_argument("--window-len", type=int, default=1001)
    ap.add_argument("--samples-before", type=int, default=1000)
    ap.add_argument("--windowlen-large", type=int, default=2000)
    ap.add_argument("--peak-height", type=float, default=0.5)
    ap.add_argument("--peak-distance", type=int, default=100)
    ap.add_argument("--tol-sec", type=float, default=0.6)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.label}] device={device}")

    data = sbd.WaveformDataset(args.dataset, sampling_rate=args.sampling_rate)
    train_ds, dev_ds, test_ds = data.train_dev_test()
    ds = {"train": train_ds, "dev": dev_ds, "test": test_ds}[args.split]
    print(f"[{args.label}] {args.split} split size: {len(ds)}")

    phase_dict = {
        "trace_p_arrival_sample": "P",
        "trace_P_arrival_sample": "P",
        "trace_s_arrival_sample": "S",
        "trace_S_arrival_sample": "S",
    }
    augmentations = [
        sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=args.samples_before,
                               windowlen=args.windowlen_large, selection="first", strategy="pad"),
        sbg.RandomWindow(windowlen=args.window_len, strategy="pad"),
        sbg.Normalize(demean_axis=-1, detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(sigma=30, dim=0),
    ]
    gen = sbg.GenericGenerator(ds)
    gen.add_augmentations(augmentations)

    model = load_model(args.checkpoint, device)

    residuals = {"P": [], "S": []}
    match_counts = {"P": [0, 0, 0], "S": [0, 0, 0]}  # matched, n_pred, n_true

    with torch.no_grad():
        for i in tqdm(range(len(gen)), desc=f"Evaluating [{args.label}]", unit="sample"):
            sample = gen[i]
            y_p_peaks, _ = find_peaks(sample["y"][0], height=args.peak_height, distance=args.peak_distance)
            y_s_peaks, _ = find_peaks(sample["y"][1], height=args.peak_height, distance=args.peak_distance)

            pred = model(torch.tensor(sample["X"], device=device).unsqueeze(0))
            pred = pred[0].cpu().numpy()
            p_prob, s_prob = pred[0], pred[1]
            p_peaks, _ = find_peaks(p_prob, height=args.peak_height, distance=args.peak_distance)
            s_peaks, _ = find_peaks(s_prob, height=args.peak_height, distance=args.peak_distance)

            for phase, y_peaks, p_peaks_ in (("P", y_p_peaks, p_peaks), ("S", y_s_peaks, s_peaks)):
                y_times = y_peaks / args.sampling_rate
                p_times = p_peaks_ / args.sampling_rate

                for yt in y_times:
                    if len(p_times) > 0:
                        res = p_times - yt
                        min_res = res[np.argmin(np.abs(res))]
                        residuals[phase].append(float(min_res))

                m, n_pred, n_true = match_peaks(p_times, y_times, args.tol_sec)
                match_counts[phase][0] += m
                match_counts[phase][1] += n_pred
                match_counts[phase][2] += n_true

    results = {"label": args.label, "dataset": args.dataset, "checkpoint": args.checkpoint,
               "split": args.split, "n_windows": len(ds), "tol_sec": args.tol_sec}
    for phase in ("P", "S"):
        matched, n_pred, n_true = match_counts[phase]
        precision = matched / n_pred if n_pred > 0 else float("nan")
        recall = matched / n_true if n_true > 0 else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
        res_arr = np.array(residuals[phase])
        results[phase] = {
            "n_ground_truth_peaks": int(n_true),
            "n_predicted_peaks": int(n_pred),
            "n_matched": int(matched),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "residual_mean_sec": float(np.mean(res_arr)) if len(res_arr) else None,
            "residual_std_sec": float(np.std(res_arr)) if len(res_arr) else None,
            "residual_n": int(len(res_arr)),
        }
        print(f"[{args.label}] {phase}: n_true={n_true} n_pred={n_pred} matched={matched} "
              f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
