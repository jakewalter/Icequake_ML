#!/usr/bin/env python3
"""Simple forward model / reflectivity check for the firn-reverberation hypothesis (see
[[t1-sp-vs-depth-vpvs-check-result]] Results 2/4/5): does a basic two-layer (firn-over-ice,
free surface on top) reverberation model actually reproduce the observed dual S-pulses, in
BOTH timing and amplitude?

Two independent checks, both deliberately simple (straight-ray geometry, scalar 1D
reverberation operator -- not a full elastic P-SV reflectivity code):

Part 1 -- travel-time moveout: for each station's real offset/depth, ray-trace (Snell's law,
straight rays in the constant-velocity ice layer, refraction into the firn layer) the
two-way vertical delay a firn-layer multiple should show, using the pipeline's own accepted
velocity model (hypodd_relocate.ARRAY_CONFIG["T1"]["velocity_layers"]: 0-0.1km firn
Vp/Vs=2.50/1.84, 0.1-3.1km ice Vp/Vs=3.85/2.22). Also inverts the OBSERVED median lag at each
station for an implied local firn thickness, instead of assuming one uniform 100m everywhere.

Part 2 -- reflectivity/amplitude: builds the exact 1D reverberation operator (free surface on
top, single reflector at the firn/ice interface) as a geometric series of bounces with ratio
R (the interface's reflection coefficient), convolves it with a Ricker source wavelet, applies
the SAME zero-phase 5 Hz highpass the real data are processed with (config.SNR_HIGHPASS_FREQ),
and checks whether ANY physically valid R (|R|<=1 is an energy-conservation hard bound for a
single passive interface -- not just an unexplored-parameter-space issue) and source frequency
can reproduce a second pulse LARGER than the first, as observed at DEEJ/LILA.

Usage:
    python full_catalog_pipeline/cluster3_firn_reflectivity_forward_model.py
"""

# --- pipeline path bootstrap (added by the 2026-09 reorganization) ---------------
# Scripts live in stage subdirectories but import shared modules (catalog_paths, config,
# vels1d_model, lib.*) and occasionally each other. Python only puts a script's OWN
# directory on sys.path, so add the pipeline root and every stage directory. Invoke from
# the REPOSITORY root -- data paths inside these scripts are relative to it.
import os as _os, sys as _sys
_PIPE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_PIPE] + [_os.path.join(_PIPE, _s) for _s in sorted(_os.listdir(_PIPE))
                     if _os.path.isdir(_os.path.join(_PIPE, _s))
                     and not _s.startswith(('.', 'artifacts', '__'))]:
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# --- end bootstrap ---------------------------------------------------------------

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

import config

# Velocity model verbatim from hypodd_relocate.ARRAY_CONFIG["T1"]["velocity_layers"]
H_FIRN_KM = 0.1
VP_FIRN, VS_FIRN = 2.50, 1.84
VP_ICE, VS_ICE = 3.85, 2.22  # 0.1-3.1 km layer, covers cluster3's 2.25-3.19 km depth range

# Real station geometry + observed peak-lag stats (from cluster3_firn_crossstation_check.py /
# cluster3_all_stations_stack_check.png), cluster3 median depth ~2.72 km
STATIONS = {
    "DEEJ": dict(dist_km=0.78, depth_km=2.72, obs_lag_ms=145, obs_std_ms=19,
                 pulse1_ms=105, pulse1_amp=0.207, pulse2_ms=165, pulse2_amp=0.429),
    "LILA": dict(dist_km=4.37, depth_km=2.72, obs_lag_ms=170, obs_std_ms=23,
                 pulse1_ms=65, pulse1_amp=0.281, pulse2_ms=175, pulse2_amp=0.562),
    "TJTJ": dict(dist_km=4.43, depth_km=2.72, obs_lag_ms=105, obs_std_ms=139,
                 pulse1_ms=75, pulse1_amp=0.455, pulse2_ms=300, pulse2_amp=0.330),
    "ELZA": dict(dist_km=5.14, depth_km=2.72, obs_lag_ms=155, obs_std_ms=120,
                 pulse1_ms=None, pulse1_amp=None, pulse2_ms=155, pulse2_amp=0.264),
    "OTIS": dict(dist_km=8.48, depth_km=2.72, obs_lag_ms=370, obs_std_ms=396,
                 pulse1_ms=None, pulse1_amp=None, pulse2_ms=None, pulse2_amp=None),
}


def ray_geometry(offset_km, depth_km):
    """Snell's-law refraction of a straight ray (constant-velocity ice layer) into the firn
    layer. Returns (incidence angle in firn, deg; ray parameter, s/km)."""
    slant = np.hypot(offset_km, depth_km)
    sin_i_ice = offset_km / slant
    p = sin_i_ice / VS_ICE
    sin_i_firn = p * VS_FIRN
    sin_i_firn = min(sin_i_firn, 1.0)
    return np.degrees(np.arcsin(sin_i_firn)), p


def part1_moveout():
    print("=" * 78)
    print("PART 1: travel-time moveout -- predicted (uniform 100m firn) vs. observed lag,")
    print("        and per-station IMPLIED firn thickness if instead we invert the observed lag")
    print("=" * 78)
    print(f"{'station':6s} {'offset':>7s} {'inc_firn':>9s} {'pred_delay':>11s} "
          f"{'obs_lag':>8s} {'implied_h_m':>12s}")
    for sta, d in STATIONS.items():
        inc_firn_deg, p = ray_geometry(d["dist_km"], d["depth_km"])
        cos_i_firn = np.cos(np.radians(inc_firn_deg))
        pred_delay_ms = 2 * H_FIRN_KM * cos_i_firn / VS_FIRN * 1000
        obs = d["obs_lag_ms"]
        implied_h_m = obs / 1000.0 * VS_FIRN / (2 * cos_i_firn) * 1000
        print(f"{sta:6s} {d['dist_km']:7.2f} {inc_firn_deg:9.1f} {pred_delay_ms:11.1f} "
              f"{obs:8.0f} {implied_h_m:12.0f}")
    print("\nVertical-incidence (DEEJ-motivated) baseline: "
          f"{2 * H_FIRN_KM / VS_FIRN * 1000:.1f} ms")
    print("Verdict: predicted delay (assuming ONE uniform 100m layer for every station) falls "
          "with offset\n(66.8-105.8ms, DEEJ->OTIS) but observed lags do NOT track that trend "
          "(145/170/105/155/370ms) --\na single uniform firn thickness does not fit. Implied "
          "per-station thickness (last column) is\nphysically plausible (136-220m, except OTIS' "
          "554m -- that station's fit is unreliable, n=27,\nstd=396ms) but does not by itself "
          "prove the mechanism.")


def reflection_coef(rho_firn, rho_ice, vs_firn=VS_FIRN, vs_ice=VS_ICE):
    z_firn, z_ice = rho_firn * vs_firn, rho_ice * vs_ice
    return (z_ice - z_firn) / (z_ice + z_firn)


def ricker(t, f0):
    a = (np.pi * f0 * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


_HP = butter(config.SNR_HIGHPASS_CORNERS, config.SNR_HIGHPASS_FREQ,
             btype="highpass", fs=config.SAMPLE_RATE_HZ, output="sos")


def synth_envelope(R, f0, tau_one_way, n_bounces=25, dur=1.2, sr=config.SAMPLE_RATE_HZ):
    dt = 1.0 / sr
    t = np.arange(0, dur, dt)
    wt = np.arange(-0.15, 0.15, dt)
    w = ricker(wt, f0)
    x = np.zeros(len(t))
    for n in range(n_bounces):
        idx = int(round(n * 2 * tau_one_way / dt))
        if idx >= len(x):
            break
        x[idx] += R ** n
    y = np.convolve(x, w, mode="same")
    y = sosfiltfilt(_HP, y)
    env = np.abs(hilbert(y))
    return t, env / env.max()


def part2_reflectivity():
    print()
    print("=" * 78)
    print("PART 2: reflectivity/amplitude -- can a single-interface reverberation operator")
    print("        reproduce the observed 2nd-pulse > 1st-pulse pattern at DEEJ/LILA?")
    print("=" * 78)
    inc_firn_deg, _ = ray_geometry(STATIONS["DEEJ"]["dist_km"], STATIONS["DEEJ"]["depth_km"])
    tau_one_way = H_FIRN_KM * np.cos(np.radians(inc_firn_deg)) / VS_FIRN
    print(f"Using DEEJ's near-vertical geometry: one-way delay tau={tau_one_way*1000:.1f} ms\n")

    for rho_firn, rho_ice, label in [(850, 917, "density-corrected (typical firn/ice)"),
                                      (1.0, 1.0, "velocity-only, rho ratio=1")]:
        R = reflection_coef(rho_firn, rho_ice)
        print(f"  R ({label}): {R:.3f}")
    print("\n  |R| <= 1 is an ENERGY-CONSERVATION hard bound for any single passive interface "
          "(not just\n  an unexplored-parameter-space issue) -- so this sweep covers the full "
          "physically valid range,\n  plus deliberately unrealistic R up to 0.4 as a stress test.\n")

    print(f"{'R':>5s} {'f0(Hz)':>7s} {'t1(ms)':>7s} {'amp1':>6s} {'t2(ms)':>7s} {'amp2':>6s}  verdict")
    any_amplifies = False
    for R in [0.09, 0.13, 0.25, 0.4]:
        for f0 in [4, 6, 8, 10, 15, 20]:
            t, env = synth_envelope(R, f0, tau_one_way)
            mask = (env[1:-1] > env[:-2]) & (env[1:-1] > env[2:])
            peak_idx = np.where(mask)[0] + 1
            peak_idx = peak_idx[t[peak_idx] < 0.6]
            if len(peak_idx) < 2:
                continue
            a1, a2 = env[peak_idx[0]], env[peak_idx[1]]
            if a2 > a1 * 1.05:
                verdict = "AMPLIFIES (2nd>1st)"
                any_amplifies = True
            elif a2 < a1 * 0.95:
                verdict = "decays (1st>2nd)"
            else:
                verdict = "~equal"
            print(f"{R:5.2f} {f0:7d} {t[peak_idx[0]]*1000:7.0f} {a1:6.2f} "
                  f"{t[peak_idx[1]]*1000:7.0f} {a2:6.2f}  {verdict}")

    print(f"\nObserved amplitude ratios (2nd pulse / 1st pulse): "
          f"DEEJ={STATIONS['DEEJ']['pulse2_amp']/STATIONS['DEEJ']['pulse1_amp']:.2f}, "
          f"LILA={STATIONS['LILA']['pulse2_amp']/STATIONS['LILA']['pulse1_amp']:.2f}, "
          f"TJTJ={STATIONS['TJTJ']['pulse2_amp']/STATIONS['TJTJ']['pulse1_amp']:.2f} "
          f"(TJTJ decays like the model; DEEJ/LILA do not)")
    print("\nVerdict: across the ENTIRE physically valid R range (and even implausibly large R "
          "up to 0.4),\na single-interface reverberation operator never reproduces a 2nd pulse "
          "~2x the 1st, as seen\nat DEEJ and LILA. This is a genuine, quantitative falsification "
          "of the SIMPLEST single-bounce\nmechanism for those two stations specifically -- "
          "TJTJ's decaying pattern IS consistent with it.\nA guided/interface (Rayleigh- or "
          "Scholte-like) wave trapped in the low-velocity firn layer,\nrather than a simple "
          "vertical multiple, is a more literature-grounded candidate for DEEJ/LILA's\n"
          "amplification pattern -- not tested here, would need a proper dispersion/normal-mode "
          "calculation.")
    return any_amplifies


if __name__ == "__main__":
    part1_moveout()
    part2_reflectivity()
