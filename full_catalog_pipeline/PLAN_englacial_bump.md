# Is T2's ~1.35 km englacial bump real?

> **Superseded (2026-08-28).** Its depth conclusions do not stand: see
> `RESULTS_depth_resolvability.md`. T2 depth is not resolved, and the between-cluster
> depth order inverts under a physically shared Vp/Vs.

Status: **planned, not started.** Written 2026-08-26, after the hypoDD tuning sweep
(`hypodd_tune.py`) surfaced the feature. Ordered cheapest/most-decisive first.

## What the feature is

Every one of the 32 T2 hypoDD configs produces a **two-peaked depth distribution**: a basal
peak whose depth is config-dependent (1.84-2.04 km, straddling the measured 2.00 km ice base)
and a shallower peak at **~1.35 km that is identical in every config** -- same depth, same
amplitude, same shape. See `tune/t2_depth_consensus.png`.

## What already argues it is real

1. **Config-invariant.** Identical across all 32 configs, which span damping schedules,
   iteration counts, residual cutoffs, clustering thresholds, two velocity models, and
   dt.cc filters. Nothing in the inversion setup controls it.
2. **Survives data perturbation.** It does not move when 20% of the cross-correlation data is
   withheld, in either split seed, including in the configs whose *basal* peak flips by 2.4 km
   (`tune/t2_tune_stability.png`).
3. **No velocity structure to blame.** The reflection profile is a constant 3.85 km/s from
   ~0.3 km to ~2.0 km. There is no discontinuity, gradient, or layer boundary anywhere near
   1.35 km, so it cannot be hypocentres piling on a model interface -- the classic artifact.

## What has NOT been ruled out

### A. Inherited from the input rather than resolved
If pyocto's initial locations already contain the bump and hypoDD simply fails to move those
events, the feature is an association artifact, not a relocation result.
**Test:** depth histogram of `prod/hypoDD.loc` (initial) against `prod/hypoDD.reloc` (final),
plus per-event displacement for bump members. Cost: minutes, files already on disk.
**Kills the feature if:** the bump is present and equally sharp in the input, and bump events
move less than the rest of the catalog.

### B. A default-depth sink for poorly constrained events
Events whose depth the geometry cannot resolve can settle at a preferred depth and pile up.
**Test:** SVD formal errors (`hypodd_svd_cluster_errors_t2.py`) for bump members vs basal
members -- compare EZ, and the fraction with EZ reported as exactly zero. Also compare the
number of P/S observations per event between the two populations.
**Kills the feature if:** bump events have systematically larger EZ, or fewer observations,
than basal events.

### C. Geometry: is it a surface, a set of clusters, or a diffuse cloud?
"Real" is not one question. A planar feature at constant depth, a set of compact repeating
clusters, and a diffuse haze imply very different physics.
**Test:** DBSCAN + PCA/Woodcock K on the bump population, reusing
`plot_hypodd_t2_basal_clusters.py`'s `cluster_basal`/`woodcock_shape` with the depth window
retargeted to the bump. Report K (cluster-like vs girdle-like), best-fit plane dip, and
lateral extent; map it against the basal population.

### D. Are the sources physically distinct from the basal events?
**Test:** waveform character of bump vs basal events at shared stations -- CC-aligned stacks
(`cluster_full_stack.py`), dominant frequency, S/P amplitude ratio, first-motion consistency
(`cluster_p_stack_polarity.py`). Englacial sources should differ from basal-slip sources.

### E. Independent absolute-depth check
**Test:** `test_sp_absolute_depth.py` restricted to bump members. S-P depends on hypocentral
distance and not on origin time, so it is an absolute constraint the double-difference
inversion never used. Compare observed-minus-predicted S-P for bump events against the
catalog as a whole, and against synthetic bump depths of 1.0/1.35/1.7 km to see whether S-P
prefers the observed depth.

### F. Depth-vs-distance trade-off
A ring of events at fixed *hypocentral* distance from the array centre would masquerade as a
fixed depth.
**Test:** plot bump-member depth against epicentral distance and against azimuth. A real
horizon is flat in depth across the array; a trade-off artifact curves with distance.

## Interpretation to reach for only after A/B are cleared

At ~1.35 km in ~2.0 km of ice, this is well below the firn (which the profile puts in the top
~300 m) and well above the bed. Candidate mechanisms worth weighing then: englacial fracture
at a rheological or fabric boundary, a shear-band or thrust structure within the ice column,
or hydrofracture/water-filled crevasse tips. Nothing here should be asserted before the
artifact tests above are done.

## Dependency

Runs on T2's settled authoritative relocation. Do not start until the config is fixed --
the bump is config-invariant, but cluster membership and formal errors are not.
