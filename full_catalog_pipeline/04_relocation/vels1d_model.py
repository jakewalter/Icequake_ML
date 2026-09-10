#!/usr/bin/env python3
"""Build hypoDD 1D layered velocity models from the reflection-seismology profile in
`full_catalog_pipeline/vels1d/` (unpacked from the user-supplied vels1d.zip: depz.mat,
vp1d.mat, vs1d.mat -- a 5 m-sampled Vp and Vs profile from 0 to 15 km, measured directly
over the T2 array).

This replaces the previous hand-assembled T1/T2 models in hypodd_relocate.py's ARRAY_CONFIG,
which used a generic 2-layer ice column (2.50 / 3.85 km/s) bolted onto a textbook crustal
Vp progression with a single assumed Vp/Vs ratio.

Two things the profile gives us that the old models did not:

  * a real firn gradient (Vp 3.31 -> 3.85 km/s over the top ~300 m, with a genuinely
    non-monotonic Vs that peaks at ~1.91 km/s near 110 m and dips back to ~1.84 km/s by
    220 m), instead of a flat 2.50 km/s slab over the top 100 m; and
  * a real, depth-resolved Vp/Vs, which sits at ~2.01 through the ice column (the physically
    expected value for glacier ice) rather than the 1.73 the old configs intended.

TWO GOTCHAS worth writing down, because both bit the previous configuration:

1. hypoDD's imod=1 ("variable Vp/Vs") block is TOP / VP / RATIO -- the third line is the
   per-layer Vp/Vs RATIO, not Vs. The previous models wrote Vs values there (1.84 2.22 2.95
   ... for T1), so hypoDD read them as ratios and silently ran with Vs = Vp/2.22 = 1.73 km/s
   in the ice and ~1.73 km/s in every bedrock layer too -- a nearly constant-Vs half space,
   not the intended profile. Confirmed in T1_v5/hypodd/hypoDD_log.txt, which echoes
   "MOD_VS 1.359 1.734 1.729 1.731 ...". build_hypodd_model_lines() below emits ratios.

2. hypoDD's MAXLAY is 50 (include/hypoDD.inc), so the 3001-sample profile has to be
   discretized. LAYER_TOPS_M below concentrates layers where the profile actually has
   structure (firn gradient, ice-bed transition) and uses a handful of thick layers through
   the constant-velocity ice column and bedrock half space. Each layer's velocity is the
   HARMONIC mean of the profile over that layer's depth interval, which preserves the
   vertical traveltime through the layer (an arithmetic mean would not).

T1 adjustment
-------------
The profile was shot over T2, whose ice is ~2.02 km thick; T1's is ~3.24 km (both
BedMachine v3 / Bedmap2 at the array centroid -- the same source associate_pyocto.py uses).
To move the profile to T1 we insert DELTA = 3.24 - 2.02 = 1.22 km of extra deep ice at
INSERT_DEPTH_M and shift the ice-bed transition and everything below it down by that amount.
The firn section is left exactly where it is (firn structure is set by accumulation and
compaction at the surface, not by how much ice is underneath it), and the shape of the
ice-bed transition and the bedrock velocities are carried over unchanged (there is no
T1-specific reflection profile).

Note the transition in the profile is a smooth ~600 m ramp (Vp leaves the ice value at
~1.97 km and reaches the bedrock value by ~2.55 km), steepest at ~2.25 km, rather than a
sharp step at 2.02 km -- expected for a smoothing-regularized reflection inversion. Shifting
by the thickness DIFFERENCE rather than by "wherever you declare the interface to be" means
the answer does not depend on picking a single interface depth out of that ramp.

Usage:
    python full_catalog_pipeline/vels1d_model.py          # print both models
"""
import os

import numpy as np
import scipy.io as sio

# vels1d/ lives at the PIPELINE root, not beside this module. The 2026-09 reorganization
# moved this file into 04_relocation/, so go up one level to find the .mat profiles.
VELS1D_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vels1d")

# Ice thickness at each array centroid (km), BedMachine v3 / Bedmap2 -- same numbers
# associate_pyocto.py's ICE_MODEL uses.
ICE_THICKNESS_KM = {"T1": 3.24, "T2": 2.02}
PROFILE_SITE = "T2"  # the array the reflection profile was shot over

# Depth (m) at which the extra deep ice is inserted for T1: below the firn gradient and
# well above the start of the ice-bed transition, i.e. inside the constant-velocity ice.
INSERT_DEPTH_M = 1600.0

# hypoDD layer tops (m). Fine through the firn gradient (0-300 m) and the ice-bed
# transition (1900-2550 m), coarse through the constant ice and the bedrock half space.
LAYER_TOPS_M = [
    0, 20, 40, 60, 80, 100, 150, 200, 250, 300,       # firn gradient
    500, 800, 1200, 1600,                              # ice column
    1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250,    # ice-bed transition
    2300, 2350, 2400, 2450, 2500, 2550,
    2700, 3500, 6000,                                  # bedrock half space
]
MAXLAY = 50  # from HYPODD/include/hypoDD.inc


def load_profile():
    """Return (depth_m, vp_m_s, vs_m_s) as float arrays, 5 m sampling, 0-15000 m."""
    depth = sio.loadmat(os.path.join(VELS1D_DIR, "depz.mat"))["depz"].ravel().astype(float)
    vp = sio.loadmat(os.path.join(VELS1D_DIR, "vp1d.mat"))["vp1d"].ravel().astype(float)
    vs = sio.loadmat(os.path.join(VELS1D_DIR, "vs1d.mat"))["vs1d"].ravel().astype(float)
    return depth, vp, vs


def _harmonic_mean(depth, vel, z0, z1):
    """Traveltime-preserving (harmonic) mean of `vel` over [z0, z1).

    The vertical traveltime through the interval is integral(dz/v), so the velocity that
    reproduces it is (z1-z0) / integral(dz/v) -- the harmonic mean, not the arithmetic one.
    The profile is uniformly sampled, so a plain mean of the slownesses does it.
    """
    mask = (depth >= z0) & (depth < z1)
    if not mask.any():  # interval past the end of the profile: use the deepest sample
        return float(vel[-1])
    return float(1.0 / np.mean(1.0 / vel[mask]))


def shifted_profile(site):
    """The reflection profile mapped onto `site`'s ice thickness, on the same 5 m grid.

    Above INSERT_DEPTH_M nothing moves. From INSERT_DEPTH_M down, `delta` metres of deep ice
    (the profile's velocity at INSERT_DEPTH_M) are spliced in, and the rest of the profile --
    the ice-bed transition and the whole bedrock section -- rides down by `delta`. For the
    profile's own site delta is 0 and this returns the profile unchanged.
    """
    depth, vp, vs = load_profile()
    delta = (ICE_THICKNESS_KM[site] - ICE_THICKNESS_KM[PROFILE_SITE]) * 1000.0
    if delta == 0:
        return depth, vp, vs
    if delta < 0:
        raise NotImplementedError(
            f"{site} is thinner than the profile site {PROFILE_SITE}; splicing ice IN only")

    dz = depth[1] - depth[0]
    new_depth = np.arange(depth[0], depth[-1] + delta + dz, dz)
    # Where in the ORIGINAL profile each new depth samples from.
    src_depth = np.where(
        new_depth < INSERT_DEPTH_M, new_depth,
        np.where(new_depth < INSERT_DEPTH_M + delta, INSERT_DEPTH_M, new_depth - delta),
    )
    idx = np.clip(np.round((src_depth - depth[0]) / dz).astype(int), 0, len(depth) - 1)
    return new_depth, vp[idx], vs[idx]


def site_layer_tops_m(site):
    """LAYER_TOPS_M with everything at/below INSERT_DEPTH_M pushed down by the thickness
    difference, plus a top for the spliced-in deep-ice layer itself."""
    delta = (ICE_THICKNESS_KM[site] - ICE_THICKNESS_KM[PROFILE_SITE]) * 1000.0
    tops = [t if t < INSERT_DEPTH_M else t + delta for t in LAYER_TOPS_M]
    if delta > 0 and INSERT_DEPTH_M not in tops:
        tops.append(INSERT_DEPTH_M)
    return sorted(tops)


def build_layers(site):
    """Return [(top_km, vp_km_s, vs_km_s), ...] for `site`, velocities harmonic-averaged
    over each layer from that site's (possibly thickness-shifted) profile."""
    depth, vp, vs = shifted_profile(site)
    tops = site_layer_tops_m(site)
    layers = []
    for i, top in enumerate(tops):
        bottom = tops[i + 1] if i + 1 < len(tops) else float(depth[-1] + 5.0)
        layers.append((
            top / 1000.0,
            _harmonic_mean(depth, vp, top, bottom) / 1000.0,
            _harmonic_mean(depth, vs, top, bottom) / 1000.0,
        ))
    if len(layers) > MAXLAY:
        raise ValueError(f"{len(layers)} layers exceeds hypoDD's MAXLAY={MAXLAY}")
    return layers


def bed_markers(site):
    """The ice-bed transition AS THE REFLECTION PROFILE MEASURES IT, in km, for `site`.

    Returns dict(ice_base, steepest, bedrock_top). This -- not BedMachine's single number --
    is the bed reference for this project: the profile was shot over the site and resolves the
    transition as a ~510 m ramp rather than an interface. At T2 the ice base (where Vp first
    departs the ice value) is 2.00 km, the steepest gradient is at 2.25 km, and bedrock
    velocity is reached at 2.51 km; BedMachine's 2.02 km coincides with the ice base only.

    Any depth structure should be read against ice_base, and anything between ice_base and
    bedrock_top is inside the transition, where "ice or bed?" is not a sharp question.
    """
    depth, vp, _ = load_profile()
    delta = (ICE_THICKNESS_KM[site] - ICE_THICKNESS_KM[PROFILE_SITE]) * 1000.0
    ice = float(np.median(vp[(depth > 500) & (depth < 1500)]))
    rock = float(np.median(vp[(depth > 3000) & (depth < 5000)]))

    def first(mask):
        idx = np.flatnonzero(mask)
        return float(depth[idx[0]]) if len(idx) else float("nan")

    grad = np.gradient(vp, depth)
    return {
        "ice_base": (first(vp > ice * 1.005) + delta) / 1000.0,
        "steepest": (float(depth[int(np.argmax(grad))]) + delta) / 1000.0,
        "bedrock_top": (first(vp > rock * 0.995) + delta) / 1000.0,
    }


def build_hypodd_model_lines(layers, ndigits=4):
    """Return the three hypoDD imod=1 model lines: TOP, VP, and Vp/Vs RATIO (NOT Vs)."""
    tops = " ".join(f"{round(top, ndigits):g}" for top, _, _ in layers)
    vps = " ".join(f"{round(vp, ndigits):g}" for _, vp, _ in layers)
    ratios = " ".join(f"{round(vp / vs, ndigits):g}" for _, vp, vs in layers)
    return tops, vps, ratios


def main():
    for site in ["T2", "T1"]:
        layers = build_layers(site)
        print(f"\n=== {site} ({len(layers)} layers, ice thickness {ICE_THICKNESS_KM[site]} km) ===")
        print(f"{'top_km':>8} {'vp_km/s':>9} {'vs_km/s':>9} {'vp/vs':>7}")
        for top, vp, vs in layers:
            print(f"{top:8.3f} {vp:9.4f} {vs:9.4f} {vp / vs:7.4f}")
        tops, vps, ratios = build_hypodd_model_lines(layers)
        print("hypoDD.inp model block:")
        print(f"  {tops}\n  {vps}\n  {ratios}")


if __name__ == "__main__":
    main()
