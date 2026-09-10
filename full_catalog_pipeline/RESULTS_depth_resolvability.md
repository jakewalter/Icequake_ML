# Is T2's cluster depth structure real? — results

Answers `PLAN_depth_resolvability.md` (written 2026-08-27). Tests 0, 1 and 5 were run
2026-08-27; Tests 2, 3, 4 and the synthesis on 2026-08-28. All on the reflection velocity
model and the `T2_v5/hypodd_vels1d` catalog.

## Verdict

**Depth is not resolved at T2.** Report locations as epicentres; do not interpret depth
structure, including the between-cluster differences. This is the plan's bottom row, reached
by a route the plan's decision table did not anticipate: the RMS(z) curves are not flat, but
their minima are not reproducible — they move by up to 1.35 km depending on formulation and on
an assumed velocity scale that the data cannot pin down.

Epicentres are untouched by all of this and remain robust.

## What each test returned

| test | question | result |
|---|---|---|
| 0 · MCCC injection recovery | is the method valid? | **PASS** — recovers 84–91% of injected moveout; null returns ~0 |
| — · within-cluster spread | is it in the waveforms? | **no** (already established 08-27) |
| 2 · composite relocation, epicentre free | absolute depth | minima at 1.33 / 1.50 / 1.55 km (clusters 0/1/2) |
| 5 · station-differential S-P, epicentre fixed | absolute depth | minima at 1.60 / 2.00 / 2.30 km |
| 3 · depth phase | absolute depth | **no coherent phase** above the null, either candidate |
| 4 · between-cluster differences | relative depth | common depth rejected, but every fit has χ²_red 54–652 |

## The three findings that decide it

**1. Tests 2 and 5 disagree.** The plan says agreement between them is worth more than either
alone. They do not agree. Cluster 2's intervals are *disjoint* — [2.20, 2.60] km with the
epicentre fixed versus [1.22, 1.85] km with it free. Clusters 0 and 1 overlap only marginally.
Same data, same velocity model, same k-profiling; only the choice of which parameters stay free
differs. A depth that changes by 750 m under that choice is not a measurement.

**2. Depth trades off almost completely with the velocity scale.** Test 5 profiles out a scale
factor k at every trial depth so a mis-calibrated model cannot fake a minimum. That protects
the curve's shape but conceals the real problem: the *fitted* k differs per cluster — 0.84 to
1.02. These clusters sit under the same ice column and cannot each have their own Vp/Vs. Hold k
fixed instead and the depth minimum walks (`depth_velocity_tradeoff.py`):

| cluster | k=1.00 | k=0.95 | k=0.90 | k=0.85 | swing |
|---|---|---|---|---|---|
| 0 | 1.75 | 1.90 | 1.90 | 1.95 | 0.20 km |
| 1 | 1.10 | 1.58 | 1.98 | 2.05 | 0.95 km |
| 2 | 0.93 | 1.33 | 1.73 | 2.28 | **1.35 km** |
| 6 | 1.25 | 1.43 | 1.70 | 1.85 | 0.60 km |
| 7 | 1.15 | 1.55 | 1.95 | 2.25 | 1.10 km |

A 15% change in assumed velocity moves cluster 2 through the entire ice column. The plan
carried a ±3% Vp/Vs caveat; the actual sensitivity is far larger than that caveat implies.

**3. Under the physically required shared k, the depth ORDER inverts.** Impose one common k on
all five clusters — which is what a single ice column demands — and the best fit is k = 0.936
(χ²/dof = 743, i.e. the model does not fit at all). The depths shift by up to 825 m, and:

```
shallow -> deep, each cluster with its own k : [0, 6, 1, 7, 2]
shallow -> deep, one shared k                : [2, 6, 7, 1, 0]
```

Nearly a complete reversal. "Which cluster is shallower" — the entire content of the
englacial-versus-basal claim — is an artifact of letting each cluster carry its own velocity
error. Test 4's differential result inherits this and cannot be trusted either, consistent with
its own χ²_red of 54–652.

**Test 3 could have broken the degeneracy and does not.** A depth phase measures absolute depth
without reference to any velocity scale, so it was the one observable that could have cut this
knot. There isn't one: no cluster's migration peak clears a random-delay null for either the
ice-base reflection or the surface-bed multiple. Per the plan, absence is uninformative — it
does not refute any depth — but it leaves the degeneracy standing.

Note also that the test is *structurally* blind where it matters. The bed-reflection delay goes
to zero as the source approaches the bed, so once the honest per-station P-coda floor is applied
the test can only see sources shallower than ~0.6–1.3 km — above where the clusters are claimed
to be. Even a clean negative would say little about a basal population.

## What holds, and what this supersedes

* Epicentres and map-view structure: unaffected, still robust.
* Within-cluster depth spread: not waveform-supported (unchanged from 08-27).
* **Superseded:** the depth conclusions in `PLAN_englacial_bump.md`, and any T2 result that
  reads structure from hypoDD depths or from per-cluster S-P depth estimates.
* The velocity model itself is not on trial here — but a joint χ²/dof of 743 says the composite
  S-P data are not fit by *any* depth under it, so the misfit is not purely a depth problem.

## Scripts

| file | test |
|---|---|
| `test_mccc_moveout_recovery.py` | 0 |
| `build_cluster_stacks.py` | 1 (now also writes composite P/S offsets and their scatter) |
| `composite_relocation_depth_scan.py` | 2 |
| `cluster_depth_phase_stack.py`, `depth_phase_migration.py` | 3 |
| `between_cluster_depth_difference.py` | 4 |
| `composite_sp_depth_pin.py` | 5 |
| `depth_velocity_tradeoff.py` | synthesis |

Four traps that produced confident wrong answers on the way here — S coda mistaken for a depth
phase, a P-coda floor manufacturing a migration peak, a straw-man constant-offset null, and an
exactly-determined fit setting a confidence interval — are written up in the project memory
under `t2-depth-resolvability-test-traps`.

## Correction to the plan

Test 3 as written looks for a free-surface ghost delayed by `2z·cos(i)/Vp`. That is the
teleseismic pP geometry and does not reach a receiver sitting on the same free surface. The
local observables are the ice-base reflection (delay `2(H−z)cos(i)/Vp`, shrinking toward the
bed) and the surface-bed multiple. Both were implemented and tested.
