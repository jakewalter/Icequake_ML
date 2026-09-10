# Is T2's cluster depth structure real, or an artifact? — test plan

> **Answered — see `RESULTS_depth_resolvability.md` (2026-08-28).** Verdict: depth is not
> resolved at T2. Note the correction to Test 3's ghost geometry recorded there.

Written 2026-08-27, before running anything. Supersedes the depth conclusions in
`PLAN_englacial_bump.md`, which were based on catalog-pick S-P and are now known circular.

## Where we actually are

Cross-correlation refinement (`cc_refine_component.py`, MCCC on raw single components,
independent of hypoDD's dt.cc) gives, for every cluster and component tested:

| cluster · station · comp | catalog-pick r | CC dS-P/dz | model requires | ratio |
|---|---|---|---|---|
| 1 · DRSC · Z | +0.553 | −4 ± 5 ms/km | +131 | −0.03 |
| 1 · DRSC · N | +0.553 | +4 ± 9 | +131 | +0.03 |
| 0 · JULA · Z | +0.100 | +11 ± 3 | +255 | 0.04 |
| 2 · DRSC · Z | +0.615 | +2 ± 2 | +125 | 0.02 |

So the **within-cluster depth spread is not carried by the waveforms**. The strong
catalog-pick correlations are the signature of the problem, not evidence against it: hypoDD's
depths and the catalog S-P are both functions of the same S picks and co-vary regardless of
whether the depths mean anything.

Three separate questions remain, and only the first is answered:

1. **Within-cluster spread** — answered: not supported. Events in a cluster sit at essentially
   one hypocentral distance per station.
2. **Between-cluster depth differences** — untested. Clusters differ in median depth by up to
   1 km; nothing so far bears on whether that is real.
3. **Absolute depth of any cluster** — untested. A bulk offset shared by every event in a
   cluster is invisible to a slope test by construction.

Everything below targets 2 and 3. Ordered by decisiveness per unit of compute.

---

## Test 0 — Does MCCC preserve moveout, or erase it? (PREREQUISITE)

Every conclusion above rests on trusting the CC refinement. The failure mode to exclude is
that MCCC, by aligning every trace to a common stack, destroys real differential moveout and
returns ~0 whatever the truth.

**Method.** Take a real cluster's waveform set. Apply a *known* synthetic depth-dependent
shift to each event's S window — `dt_i = slope_true × (z_i − z̄)` for slope_true = +131 ms/km
and the events' own hypoDD depths — then run the identical MCCC pipeline and regress the
recovered S-P against z. Repeat for slope_true ∈ {0, 65, 131, 262} ms/km.

**Reads.** Recovered slope vs injected slope. Also the recovery as a function of the
`--max-lag` search half-width, since a lag window narrower than the true moveout would clip it.

**Kill criterion.** If recovered ≈ 0 for a large injected slope, MCCC is erasing moveout, the
flat results above are meaningless, and Tests 1–5 must be rebuilt on a method that does not
align to a common reference (e.g. pairwise CC with an explicit moveout term).

**Cost.** ~30 min. Nothing else should be believed until this passes.

---

## Test 1 — Per-cluster, per-station CC stacks

Not a test in itself; the input to Tests 2–4, and worth inspecting on its own.

**Method.** For each cluster × each of the 7 stations: MCCC-align on P, stack the aligned
traces (Z, N, E separately — no envelopes, which discard polarity and smear onsets), and
record the composite P and S arrival times with their alignment scatter.

**Reads.** Stack SNR and coherence per station; the composite S-P per station with a real
uncertainty; whether the S arrival on the stack is a clean onset or a wavetrain.

---

## Test 2 — Depth-pinned composite relocation (the "pin it to the ice base" test)

The quantitative form of pinning a cluster centroid to the ice base and reading the residuals.

**Method.** Treat each cluster as ONE composite event using the Test-1 stacked arrival times
at all stations. Grid over trial depth z ∈ [0.2 … 3.0] km in 25 m steps. At each z, solve
least squares for the remaining free parameters (x, y, t0) against the composite P and S times,
using the reflection-model traveltimes. Record the weighted RMS residual → an RMS(z) curve.

**Reads.**
* A sharp minimum ⇒ depth IS resolved; its location is the data-preferred depth and the
  curvature (Δχ² = 1) gives the uncertainty.
* A flat curve ⇒ depth is not resolved at all; hypoDD's value is arbitrary within the flat
  range, and no depth-based interpretation of these clusters is defensible.
* Specifically report RMS at z = 2.00 km (measured ice base) vs z = the cluster's hypoDD
  median, and whether the difference exceeds the arrival-time precision from Test 1.

**Why this is not circular.** Arrival times come from waveform CC, not the pick table; the
inversion is independent of hypoDD; the traveltimes come from the reflection profile.

---

## Test 3 — Surface-reflected (depth) phase in the stacks — the strongest available test

A source at depth z beneath the free ice surface radiates an upgoing ray that reflects and
returns: a pP-equivalent ghost, delayed after the direct P by ≈ 2z·cos(i)/Vp. **That delay
depends only on depth and velocity — not on origin time, not on epicentral distance, not on
the epicentre.** It is the one observable that measures absolute depth directly.

At Vp = 3.85 km/s and near-vertical takeoff: z = 1.35 km → ~0.70 s; z = 2.00 km (ice base) →
~1.04 s. The equivalent sP and the ice-bed reflection from below give further constraints.

**Method.** Search the Test-1 stacks for coherent arrivals at plausible ghost delays; measure
the delay per station; check it is consistent across stations after the cos(i) correction
(a true depth phase is, a scattered arrival is not); invert the delay for z.

**Context.** This dataset is known to contain such reverberation — the T1 DEEJ "tight 1:2
harmonic bounce pair" and the firn-reverberation work
([[t1-cluster3-wrongpulse-refuted-deej-onset-finding]]). Those were treated as a nuisance;
here they are the signal.

**Reads.** A coherent ghost at a consistent delay pins absolute depth to ~±50 m and settles
question 3 outright. Its absence is uninformative (it may simply be too weak), so this test
can confirm but not refute.

---

## Test 4 — Between-cluster differential depth

Even where absolute depth is weak, the *difference* between two clusters may be resolved, and
that is what "the englacial bump is shallower than the basal population" actually asserts.

**Method.** For each cluster pair, difference their composite S-P at a common station. With
epicentres well determined (the map view is identical across all 32 hypoDD configs — the one
thing that IS robust), the S-P difference maps to a depth difference. Do it at every station
and check consistency; a real depth difference produces a station-dependent pattern set by
each station's own z/r, whereas a common time-shift artifact produces a constant offset.

---

## Test 5 — Station-differential S-P (origin-time-free cross-check)

The vector of S-P across the 7 stations is a fingerprint of (epicentre, depth) in which origin
time cancels exactly. Fit that vector at trial depths and compare with Test 2. Independent
formulation, same question — agreement between 2 and 5 is worth more than either alone.

---

## Decision table

| Test 2 RMS(z) | Test 3 ghost | Conclusion |
|---|---|---|
| sharp minimum | consistent, same z | depth resolved and real — interpret it |
| sharp minimum | absent | depth resolved by traveltimes alone; interpret with the model caveat |
| flat | consistent | absolute depth from the ghost; traveltime geometry cannot see it |
| flat | absent | **depth is unresolved at T2** — report locations as epicentres only, drop all depth-based structure |

---

## Caveats to carry through

* **Velocity model uncertainty propagates into every depth.** The absolute S-P test measured a
  best-fit scale factor of 0.977 on the reflection model, i.e. it over-predicts S-P by ~2.3%.
  Propagate ±3% on Vp/Vs into the RMS(z) minima rather than quoting a single depth.
* **The firn matters for near-vertical rays.** Test 3's ghost travels twice through the top
  300 m, where Vp goes 3.32 → 3.85 km/s. Use the layered model, not a single velocity.
* **Epicentres are robust; depths are what is in question.** Nothing here threatens the map-view
  results, which were identical across every configuration swept.
* **This supersedes, for T2, the earlier claim that the englacial clusters' depth spread is
  data-supported.** That was based on catalog picks.

## Order and cost

0 (30 min, gating) → 1 (~1 h) → 2 and 3 in parallel (~1–2 h) → 4, 5 (~30 min) → synthesis.
