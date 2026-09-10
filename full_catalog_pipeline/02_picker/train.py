#!/usr/bin/env python3
"""Train PhaseNet on the full-catalog SNR-filtered dataset.

Usage:
    python full_catalog_pipeline/train.py [--config full_catalog_pipeline/train_config.json]
                                          [--checkpoint-dir full_catalog_pipeline/checkpoints/]

Writes checkpoints and loss history to --checkpoint-dir.
"""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding

from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def loss_fn(y_pred, y_true, eps=1e-8):
    """Cross-entropy loss matching SeisBench PhaseNet convention."""
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1).mean()
    return -h


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, checkpoint_dir="checkpoints/", verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = self.checkpoint_dir / "best_model.pth"

    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(model, epoch)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(model, epoch)
            self.counter = 0

    def _save(self, model, epoch):
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "best_score": self.best_score}, self.checkpoint_path)
        if self.verbose:
            print(f"  [checkpoint] validation improved → {self.checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full_catalog_pipeline/train_config.json")
    parser.add_argument("--checkpoint-dir", default="full_catalog_pipeline/checkpoints/")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    set_seed(42)
    device = torch.device("cuda" if cfg["device"]["use_cuda"] and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    data_cfg = cfg["data"]
    print(f"\nLoading dataset from {data_cfg['dataset_name']}...")
    data = sbd.WaveformDataset(data_cfg["dataset_name"], sampling_rate=data_cfg["sampling_rate"])
    train_ds, dev_ds, test_ds = data.train_dev_test()
    print(f"Train: {len(train_ds)}  Dev: {len(dev_ds)}  Test: {len(test_ds)}")

    # Phase dict
    phase_dict = {
        "trace_p_arrival_sample": "P",
        "trace_P_arrival_sample": "P",
        "trace_s_arrival_sample": "S",
        "trace_S_arrival_sample": "S",
    }

    wl_large = data_cfg["windowlen_large"]
    wl = data_cfg["window_len"]
    sb = data_cfg["samples_before"]

    augmentations = [
        sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=sb, windowlen=wl_large,
                               selection="first", strategy="pad"),
        sbg.RandomWindow(windowlen=wl, strategy="pad"),
        sbg.Normalize(demean_axis=-1, detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(sigma=30, dim=0),
    ]

    train_gen = sbg.GenericGenerator(train_ds)
    dev_gen = sbg.GenericGenerator(dev_ds)
    train_gen.add_augmentations(augmentations)
    dev_gen.add_augmentations(augmentations)

    t_cfg = cfg["training"]
    opt_cfg = t_cfg.get("optimization", {})

    train_loader = DataLoader(
        train_gen,
        batch_size=t_cfg["batch_size"],
        shuffle=True,
        num_workers=t_cfg["num_workers"],
        worker_init_fn=worker_seeding,
        pin_memory=opt_cfg.get("pin_memory", True),
        prefetch_factor=opt_cfg.get("prefetch_factor", 2),
        persistent_workers=opt_cfg.get("persistent_workers", True),
    )
    dev_loader = DataLoader(
        dev_gen,
        batch_size=t_cfg["batch_size"],
        shuffle=False,
        num_workers=t_cfg["num_workers"],
        worker_init_fn=worker_seeding,
        pin_memory=opt_cfg.get("pin_memory", True),
        prefetch_factor=opt_cfg.get("prefetch_factor", 2),
        persistent_workers=opt_cfg.get("persistent_workers", True),
    )

    # Model
    model = sbm.PhaseNet(phases="PSN", norm="peak").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=opt_cfg.get("mixed_precision", True))
    early_stop = EarlyStopping(patience=t_cfg["patience"], checkpoint_dir=args.checkpoint_dir)

    loss_weights = t_cfg.get("loss_weights", [0.05, 0.475, 0.475])
    loss_weights_t = torch.tensor(loss_weights, dtype=torch.float32).to(device)

    ckpt_dir = Path(args.checkpoint_dir)
    history = {"train": [], "dev": []}

    print(f"\nStarting training for up to {t_cfg['epochs']} epochs...\n")

    for epoch in range(1, t_cfg["epochs"] + 1):
        # -- Training --
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{t_cfg['epochs']} [train]", leave=False)
        for batch in pbar:
            X = batch["X"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=opt_cfg.get("mixed_precision", True)):
                pred = model(X)
                loss = sum(loss_fn(pred[:, i:i+1, :], y[:, i:i+1, :]) * loss_weights_t[i]
                           for i in range(3))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = np.mean(train_losses)
        history["train"].append(train_loss)

        # -- Validation --
        model.eval()
        dev_losses = []
        with torch.no_grad():
            for batch in dev_loader:
                X = batch["X"].to(device)
                y = batch["y"].to(device)
                with torch.amp.autocast("cuda", enabled=opt_cfg.get("mixed_precision", True)):
                    pred = model(X)
                    loss = sum(loss_fn(pred[:, i:i+1, :], y[:, i:i+1, :]) * loss_weights_t[i]
                               for i in range(3))
                dev_losses.append(loss.item())

        dev_loss = np.mean(dev_losses)
        history["dev"].append(dev_loss)

        print(f"Epoch {epoch:3d}  train={train_loss:.4f}  dev={dev_loss:.4f}")

        # Save loss history each epoch
        with open(ckpt_dir / "loss_history.json", "w") as f:
            json.dump(history, f, indent=2)

        early_stop(dev_loss, model, epoch)
        if early_stop.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    # Save final model
    torch.save({"model_state_dict": model.state_dict(), "config": cfg},
               ckpt_dir / "final_model.pth")

    # Save state-dict only version (for easy inference loading)
    torch.save(model.state_dict(), ckpt_dir / "best_model_state_only.pth")

    print(f"\nTraining complete. Checkpoints in {ckpt_dir}/")


if __name__ == "__main__":
    main()
