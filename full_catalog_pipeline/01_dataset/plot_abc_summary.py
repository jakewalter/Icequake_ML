#!/usr/bin/env python3
"""3-way summary comparison across the A/B/C SNR/min-stations ablation,
reading the pilot_metrics.json each plot_pilot_comparison.py run writes.

Usage:
    python full_catalog_pipeline/plot_abc_summary.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="whitegrid", context="talk")

VARIANTS = [
    ("A", "full_catalog_pipeline/artifacts/pilot/pilot_metrics.json", "A: SNR>=5dB\n>=4 stations\n(13,810 win)"),
    ("B", "full_catalog_pipeline/artifacts/pilot_B/pilot_metrics.json", "B: SNR>=5dB\n>=3 stations\n(25,261 win)"),
    ("C", "full_catalog_pipeline/artifacts/pilot_C/pilot_metrics.json", "C: SNR>=3dB\n>=3 stations\n(58,977 win)"),
]
if __import__("os").path.exists("full_catalog_pipeline/artifacts/pilot_D/pilot_metrics.json"):
    VARIANTS.append(
        ("D", "full_catalog_pipeline/artifacts/pilot_D/pilot_metrics.json", "D: SNR>=1dB\n>=3 stations\n(133,266 win)")
    )
if __import__("os").path.exists("full_catalog_pipeline/artifacts/pilot_E/pilot_metrics.json"):
    VARIANTS.append(
        ("E", "full_catalog_pipeline/artifacts/pilot_E/pilot_metrics.json", "E: SNR>=2dB\n>=3 stations\n(91,855 win)")
    )

COLORS = {"A": "#8c96a6", "B": "#ff7d00", "C": "#15616d", "D": "#8338ec", "E": "#d90429"}

OUT_PATH = "full_catalog_pipeline/artifacts/abc_summary.png"


def main():
    metrics = {}
    for key, path, label in VARIANTS:
        with open(path) as f:
            metrics[key] = json.load(f)
        metrics[key]["axis_label"] = label

    fig, axs = plt.subplots(1, 4, figsize=(22, 5.5))
    keys = [k for k, _, _ in VARIANTS]
    labels = [metrics[k]["axis_label"] for k in keys]
    colors = [COLORS[k] for k in keys]

    ax = axs[0]
    ax.bar(labels, [metrics[k]["n_pyocto_events"] for k in keys], color=colors)
    ax.axhline(metrics["A"]["n_qm_events"], color="black", linestyle="--", linewidth=1.5,
               label=f"QuakeMigrate (n={metrics['A']['n_qm_events']})")
    ax.set_title("pyocto event count")
    ax.legend(fontsize=9)
    for i, k in enumerate(keys):
        ax.text(i, metrics[k]["n_pyocto_events"], str(metrics[k]["n_pyocto_events"]), ha="center", va="bottom", fontsize=11)

    ax = axs[1]
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w/2, [metrics[k]["precision"] for k in keys], w, label="precision", color="#15616d")
    ax.bar(x + w/2, [metrics[k]["recall"] for k in keys], w, label="recall", color="#ff7d00")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision / recall vs QuakeMigrate")
    ax.legend(fontsize=10)
    for i, k in enumerate(keys):
        ax.text(i - w/2, metrics[k]["precision"] + 0.02, f"{metrics[k]['precision']:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, metrics[k]["recall"] + 0.02, f"{metrics[k]['recall']:.2f}", ha="center", fontsize=9)

    ax = axs[2]
    ax.bar(labels, [metrics[k]["n_matched"] for k in keys], color=colors)
    ax.set_title("Matched events (of QM total)")
    for i, k in enumerate(keys):
        ax.text(i, metrics[k]["n_matched"], str(metrics[k]["n_matched"]), ha="center", va="bottom", fontsize=11)

    ax = axs[3]
    ax.bar(labels, [metrics[k]["matched_dist_median_km"] for k in keys], color=colors)
    ax.set_title("Median location error\n(matched events, km)")
    for i, k in enumerate(keys):
        v = metrics[k]["matched_dist_median_km"]
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11)

    fig.suptitle("SNR / min-stations ablation vs QuakeMigrate (T2 array, July 2020)", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH}")

    for k in keys:
        m = metrics[k]
        print(f"{k}: events={m['n_pyocto_events']} matched={m['n_matched']} "
              f"precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
