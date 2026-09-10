# Methods — Icequake detection, relocation, and tidal forcing at Thwaites Glacier

**Status:** working methods document, current as of **2026-09-10**. Supersedes the 2026-08-11
draft (kept at `docs/superseded/METHODS_DRAFT_2026-08-11.md`, untracked), which stopped at §6 and
documented a velocity model and event counts that have since been superseded. If you find an older
figure or paragraph whose numbers disagree with this file, it predates the velocity-model
correction in §7.1 and needs regenerating.

**This is the internal record, not manuscript prose.** Paper-facing versions
(`docs/METHODS_paper.md`, `docs/METHODS_summary.md`, and `.docx` conversions of all three) were
written from this document but live in `docs/`, which is **not tracked by git** — see the root
README for why and how to regenerate. This file is the source of record for every number.

Numbers below are traced to repository code and outputs. Items that are **not** settled are
marked and explained rather than smoothed over — §7, §8 and §9 in particular contain results
that constrain what can be claimed. Code layout is described in
[`full_catalog_pipeline/README.md`](full_catalog_pipeline/README.md).

**Reading guide for a new analyst.** The pipeline works and produces a catalogue whose
*epicentres* are robust. Its *depths* are not resolved, for reasons that are now quantified
(§7). The tidal/GPS work (§9–§10) is the most recent and the most self-contained. If you read
only two things, read §7.4 (why depth is not resolved) and §8.3 (what is staged but not
promoted).

---

## 1. Network and continuous archive

Two seven-station arrays (T1, T2) on the eastern shear margin of Thwaites Glacier, West
Antarctica, recording continuously at 200 Hz, three components (HHZ, HH1, HH2), network code
**7U**.

| array | stations | centroid | ice thickness |
|---|---|---|---|
| T1 | DEEJ, ELZA, LILA, LOUS, OTIS, SQIG, TJTJ | 77.299°S, 100.476°W | 3.24 km |
| T2 | BAUM, DRSC, EPJZ, FRST, JULA, OKGS, WICH | 76.570°S, 103.284°W | 2.02 km |

Ice thickness is the mean of BedMachine Antarctica v3 and Bedmap2 at each centroid, and is used
throughout as that array's ice–bed interface depth (`vels1d_model.ICE_THICKNESS_KM`).

> **Station naming trap.** Both arrays are network `7U` in all hypoDD inputs (`station.sel`,
> `dt.ct`, `dt.cc`). Network `2E` appears in some raw archive paths. Writing `2E` into a `dt.cc`
> causes hypoDD to **silently discard every cross-correlation observation** — it exits 0 and
> produces a plausible catalogue-only relocation. This happened once (§8.2). Always check
> `# cross corr P dtimes` in `hypoDD.log` is non-zero before reading any result.

Data availability is near-complete: every station has 707–728 days of HHZ out of ~730
(`artifacts/day_file_index.csv`). This matters for §10 — quiet periods in the catalogue are real,
not outages.

## 2. Reference catalogue (QuakeMigrate)

An initial catalogue per array from QuakeMigrate (waveform migration and stacking): 10,504 events
for T1, 14,671–14,700 for T2. Detection used a centred STA/LTA onset on bandpass-filtered data
(P: 20–90 Hz; S: 5–50 Hz, both 4th-order), STA/LTA windows 0.08/1.6 s (P) and 0.2/2.0 s (S),
arrival times from a 1-D Gaussian fit to the onset function (MAD multiplier 8.0). Triggering used
a dynamic MAD threshold (600 s window, multiplier 12.0), 60 s minimum inter-event interval, 0.1 s
triggering marginal window, 0.3 s at the location stage.

> **[VERIFY]** These parameters come from T2's run logs. No T1-specific QuakeMigrate log survives
> on this machine; identical configuration is likely but unconfirmed. QuakeMigrate's grid/lookup
> parameters (node spacing, extent) and its minimum pick count to locate are **not documented
> anywhere in this repository** and would need recovering from the original project.

## 3. Training-set construction

`full_catalog_pipeline/01_dataset/`

QuakeMigrate picks were pooled (T1: 73,491 P / 73,500 S deduplicated; T2: 100,187 P / 100,192 S)
and used to cut three-component windows. Per-window SNR was computed in dB on a 20–80 Hz
4th-order band-limited trace, signal window −0.25 to +0.25 s about the pick and noise window
−2.25 to −1.75 s.

Thresholds were chosen by a five-configuration ablation, scored as pyocto-catalogue agreement
against QuakeMigrate on a one-month (July 2020) T2 pilot (1,887 reference events):

| config | SNR floor | min stations | windows | precision | recall | F1 |
|---|---|---|---|---|---|---|
| A | ≥5 dB | ≥4 | 13,810 | 0.991 | 0.114 | 0.204 |
| B | ≥5 dB | ≥3 | 25,261 | 0.989 | 0.149 | 0.259 |
| C | ≥3 dB | ≥3 | 58,977 | 0.912 | 0.430 | 0.584 |
| **E (adopted)** | **≥2 dB** | **≥3** | **91,855** | **0.933** | **0.441** | **0.599** |
| D | ≥1 dB | ≥3 | 133,266 | 0.959 | 0.301 | 0.458 |

Model E gives 91,855 windows from 20,830 events (82.7% of pooled catalogue events), essentially
balanced (91,845 P vs 91,850 S). Events split 70/15/15 train/val/test, stratified so all windows
from one event stay in one split (seed 42), preventing leakage across an event's stations.

Recall and F1 peak at a 2–3 dB floor and degrade below 1 dB — examples below ~2 dB add net label
noise, not useful diversity.

## 4. Phase picker

`full_catalog_pipeline/02_picker/`

SeisBench PhaseNet, three-class (P, S, noise). Each example is a 1001-sample window (5.005 s at
200 Hz) cut at random offset from a 2000-sample buffer, detrended, demeaned, and normalised by
peak absolute amplitude. Labels are Gaussian peaks (σ = 30 samples, 0.15 s) on each arrival.

Adam, learning rate 0.01, batch size 32, up to 50 epochs, early stopping patience 10 on
validation loss, mixed precision. Loss is per-channel weighted cross-entropy (noise 0.05, P
0.475, S 0.475) summed over channels. The final model ran all 50 epochs without early stopping;
validation loss minimum 0.0616 at epoch 41.

> **[VERIFY]** Model E's held-out *test-set* precision/recall/F1 were never located — only the
> loss curve and the downstream pyocto-agreement numbers in §3. If the manuscript reports picker
> performance directly, regenerate it with `02_picker/evaluate.py` against the Model E checkpoint.

## 5. Deployment on continuous data

Each day-length trace was linearly detrended; **no bandpass, native 200 Hz**, matching training
preprocessing. This is a deliberate departure from easyQuake/SeisBench's default wrapper, which
applies a 3–20 Hz bandpass (tuned for crustal earthquakes, not these 20–80 Hz-dominant icequakes)
and silently resamples to 100 Hz. On one station-day with 347 hand-verified picks that default
produced 29,259–153,965 spurious picks (precision ~10⁻²). With the corrected preprocessing and
P/S trigger thresholds of 0.5: 536 picks/day, P precision/recall 0.47/0.72, S 0.18/0.27.

## 6. Association

`full_catalog_pipeline/03_association/`

pyocto v0.1.9 (octree associator), per array, against a two-layer 1-D model: ice over bedrock
half-space (Vp 5.1 km/s, Vp/Vs 1.8), transition at each array's own ice thickness (§1), smoothed
over a 0.15 km half-width to avoid a depth-location degeneracy at a sharp discontinuity.
Association required ≥6 picks total and ≥3 stations with both P and S, minimum pick fraction 0.2,
exponential-EDT loss, octree minimum node size 0.02 km, split depth 10, 3 refinement iterations —
tuned specifically to remove discrete depth banding present with pyocto's defaults here.

Matched to QuakeMigrate by origin-time proximity: **T1 2,400 events** (precision 0.75, recall
0.17, median epicentral offset 0.24 km); **T2 4,277 events** (0.76 / 0.22, 0.18 km).

## 7. Double-difference relocation

`full_catalog_pipeline/04_relocation/`, `05_crosscorrelation/`, `06_depth/`

hypoDD v2.1beta. Catalogue differential times from `ph2dt`; cross-correlation differential times
from a 20–90 Hz bandpass, ±0.1 s lag search, minimum coefficient 0.4, computed separately for P
(vertical) and S (all three components). `ph2dt`'s maximum pair separation was recomputed per
array as the 5th-percentile pairwise inter-event distance, geodetically correct (the ~4.5×
longitude-scale correction at ~77°S matters), giving 0.48 km for T1.

### 7.1 Velocity model — supersedes the draft's Table 2

Relocation uses a **1-D profile measured by reflection seismology over T2**
(`full_catalog_pipeline/vels1d/{depz,vp1d,vs1d}.mat`, 5 m sampling), harmonic-averaged
(traveltime-preserving) onto hypoDD layers by `vels1d_model.build_layers()`. T1 uses the same
profile depth-shifted so the ice–bed transition sits at its own 3.24 km thickness.

| | layer top | Vp (km/s) | Vp/Vs |
|---|---|---|---|
| surface (firn) | 0.000 | 3.3190 | 2.2143 |
| firn base | 0.300 | 3.8431 | 2.0367 |
| ice | 1.200 | 3.8503 | 2.0087 |
| ice–bed transition | T1 3.670 / T2 2.450 | 5.5696 | 2.0073 |
| bedrock | T1 4.720 / T2 3.500 | 5.6588 | 2.0078 |

T1 has 32 layers, T2 31. **The Vp/Vs of ~2.008 through the ice column and the bedrock is
measured, not assumed** — do not substitute a textbook rock value of ~1.73. (An attempt to do
exactly that during this work produced a large apparent improvement in one misfit and was wrong.)

> **Two corrected bugs are baked into this section.** (i) hypoDD's `imod=1` third line is the
> **Vp/Vs ratio, not Vs**; every run before 2026-08-25 used a wrong, near-constant Vs. (ii) The
> draft's Table 2 and its "fixed Vp/Vs of 1.73 throughout the ice" describe the superseded
> hand-assembled model. Both are fixed; any figure or text derived from them must be regenerated.

### 7.2 Cross-correlation data: two defects and their fixes

**Corrupt observations.** T2's original `dt.cc` had median |dt| 0.018 s and p99.9 0.19 s but a
maximum of **232 s**, and of the 241 observations above 0.5 s the median correlation coefficient
was **1.000** — spurious perfect correlations (cycle skips onto a near-identical waveform)
entering at full weight. These caused a 2.4 km depth flip in half the T2 catalogue. A coefficient
threshold provably cannot remove them (the bad data *has* coefficient 1.0); a cap on |dt| can.
Production uses **|dt| ≤ 0.2 s and coefficient ≥ 0.5** for T2.

**A near-total loss of P.** `05_crosscorrelation/diagnose_cc_p_deficit.py` found the root cause:
obspy's parabolic sub-sample fit needs 3 samples and gets 1–2 at 200 Hz, so P correlations were
silently dropped. `regenerate_dtcc.py` replaces hypoDDpy's CC stage, correlating at ×2 upsampling
and extracting each distinct (event, station, phase) window once rather than per pick pair — 27×
faster than hypoDDpy's 10 h, and `--verify` proves the preprocessing hoist is exact (worst
absolute difference 0.000e+00 over 40 real correlations).

| `dt.cc` in use | observations | P fraction |
|---|---|---|
| T2 original (filtered, authoritative) | 592,548 | **2.7%** |
| T2 regenerated (filtered) | 2,847,206 | **62.2%** |
| T1 original | 114,370 | **0.5%** |
| T1 regenerated (filtered) | 1,232,278 | **64.2%** |

Recovering P matters because with S-only CC data the differential times **cannot form S−P at
all**, so depth information came almost entirely from the catalogue picks (`dt.ct`).

### 7.3 The |dt| cap does not transfer between arrays

T2's 0.2 s cap must **not** be reused at T1. A physical-possibility test — an observation's |dt|
cannot exceed its pair separation divided by Vs — settles it:

| cap | T1 kept | T1 discarded | of discarded: physically impossible |
|---|---|---|---|
| 0.2 s (T2's) | 92.5% | 107,134 | **19.8%** — discards *legitimate* data |
| **0.5 s (adopted for T1)** | **99.88%** | 1,742 | **70.7%** |

T1's event pairs sit a median 0.75 km apart against T2's 0.23 km, so its genuine |dt| runs far
larger. At T2 the >0.2 s tail is 92% impossible; at T1 it is 80% legitimate. Caveat: ~12% of the
*kept* T1 data also fails this test at every cap, so the coefficient floor is still doing real
work and the filtered file is not artifact-free.

### 7.4 Depth is not resolved — read this before interpreting any depth

Multiple independent tests, all in `06_depth/`:

- **Depth trades off ~1:1 with velocity scale.** Holding the scale factor *k* fixed rather than
  profiling it out, one T2 cluster's depth moves **1.35 km across a 15% change in k**
  (0.93 km at k=1.00, 2.28 km at k=0.85).
- **Per-cluster fits demand different k (0.84–1.02), which one ice column cannot supply.**
  Imposing a single shared k shifts depths by up to 825 m and **inverts the shallow→deep cluster
  order** — so "which cluster is shallower", i.e. any englacial-vs-basal claim, is an artifact of
  per-cluster velocity error.
- **Two methods disagree by up to 750 m with disjoint intervals** (composite relocation vs
  station-differential S−P), on the same data and model, differing only in which parameters stay
  free.
- **Depth phases are absent.** A migration/semblance search for the ice-base reflection and the
  surface–bed multiple clears no null for any cluster; it is also structurally blind below
  ~1.3 km, where the bed-reflection delay goes to zero.
- **Station-specific S−P offsets span 142 ms** (DRSC +48, JULA −92, EPJZ −94 ms) after fitting
  each cluster its own best depth. A 1-D model cannot produce a station-dependent term, and the
  offsets are ~3× the 45–53 ms difference between competing depth hypotheses. **This is the
  quantitative reason absolute depth does not resolve.** Note double-difference *cancels* station
  terms by construction, so absolute S−P is the weaker instrument and should not be used to
  referee a hypoDD result.
- **Within-cluster spread is not waveform-supported.** For T2's dominant cluster (n=1732), JULA at
  0.22 km epicentral distance predicts an S−P slope of 260 ms/km if the 1126 m depth spread were
  real; measured is **12 ms/km (Z) and 11 (N)**, ~4–5%, on 1500 events with mean correlation
  0.76–0.88. Better alignment does not rescue it. An injection–recovery control
  (`test_mccc_moveout_recovery.py`) shows MCCC recovers 84–91% of an injected moveout, so the flat
  slopes are a property of the data.
- A **proper 1-D ray tracer** (`lib/raytrace1d.py`; direct, turning and head-wave branches;
  validated to <2 ms against an analytic homogeneous half-space and ~1 ms against a two-layer
  head-wave case) replaces straight rays. It shows depth sensitivity collapsing with offset:
  **243 ms/km at 0.21 km but 48 ms/km at 4.8 km**, so near stations carry ~5× the depth
  information.

**Consequence: report epicentres and map-view structure, which are robust. Do not interpret depth
structure, including between-cluster differences.**

One earlier exception is worth knowing: a T2 cluster (n=629 in the then-current catalogue) *did*
pass this class of test, but only when tested against its **own PCA long axis** rather than raw
depth, because it plunges ~19° from horizontal. A naive same-sign-at-every-station depth test
called it an artifact and was wrong. That verdict attaches to a catalogue that has since been
re-relocated twice, so the cluster needs re-identifying by event ID before the result is reused.

Also: formal SVD resolution is **not** physical reality. SVD reruns give EZ medians of 2.5–7.5 m
against within-cluster depth spreads of 53–133 m (9–28σ "resolved"), and the same class of number
previously supported a "23σ real column" at T1 that was later shown to be one noise realisation of
an unstable inversion.

## 8. Current relocations and what is staged

### 8.1 The one unambiguous improvement

hypoDD's LSQR returns no error estimate (`ez = 0`) for events where it stopped without
converging. This is the solver reporting on itself:

| relocation | events | `dt.cc` P | **LSQR non-convergence** |
|---|---|---|---|
| T1 authoritative (`hypodd_vels1d`) | 2,164 | 0.5% | **59.0%** |
| T1 candidate (`hypodd_vels1d_t1ccstrong`) | 2,287 | 64.2% | **0.1%** |
| T2 authoritative (`hypodd_vels1d`) | 3,293 | 2.7% | **41.5%** |
| T2 candidate (`hypodd_vels1d_ccstrong`) | 3,546 | 62.2% | **0.1%** |

Regenerating `dt.cc` takes both arrays from ~40–60% solver failure to 0.1%, with 123–253 more
events relocated. Everything else about the new catalogues — depth structure especially — is *not*
established as better.

### 8.2 Configuration selection

`04_relocation/hypodd_tune.py` (`run` / `sweep` / `report` / `promote`, `--dtcc` selects the CC
file, `--seed` the CV split). Ranking is by **held-out cross-validation**: `dt.cc` split 80/20 by
observation, relocate on the 80% plus all `dt.ct`, score against the withheld 20% at fixed
locations. Two hard-won rules:

- **Rank on median |residual| plus per-cluster depth shift, across two seeds.** Held-out RMS,
  correlation, and median-only stability each gave confident wrong answers. On T2's upsampled
  sweep, `ccmin_0.5` looked good at seed 42 and its stability collapsed from 48 m to 292 m at
  seed 7.
- **Never use CC weight = 0** (`ct_only`), whatever a compactness or stability metric says.

T2's six-config, two-seed sweep selected **`clean_ccstrong`** = the 0.2 s / 0.5 filters plus
`wt_cc = [0.99, 0.50, 0.25, 0.10, 0.05]` (more CC weight carried into the late stages). It wins
held-out median |residual| at both seeds (1.77 / 1.80 ms) and is the only seed-stable config
(28.6 → 29.2 m, where others degrade 3–6×). T1's `t1_ccstrong` is the *same weighting* with T1's
own 0.5 s cap (§7.3) — **one config, one seed only**, a transplant rather than a demonstrated
optimum.

Two metrics were added to the harness because the existing ones could not see real failures:
`worst_cluster_depth_shift_m` (every cluster, not just cluster 1 — `clean_ccstrong` scored 0 m on
cluster 1 while cluster 2 moved 500 m) and `cluster_consensus()` /
`consensus_dev_max_m` (how far each config sits from the across-config median, since the
withhold test cannot see a config that reproduces itself but disagrees with every other solution).
The second is a **dispersion** statistic — "explain before promoting", not automatic rejection.

### 8.3 Not promoted — the open decision

Both improved relocations are staged as **candidate run directories** and reachable via
`ICEQUAKE_RELOC`; the authoritative catalogues are untouched. Outstanding before promotion:

1. **`clean_ccstrong` moves one cluster (682 events, 21% of T2) 501 m shallower**, where five
   sibling configs move the same events 13–151 m. The waveforms cannot arbitrate (§7.4 station
   terms), so this is a 3–4× outlier with no independent check.
2. **T1 has one config and one seed**, and no cross-config dispersion check is possible.
3. **T1 has no `PROVENANCE.md`**, so whether its *authoritative* relocation ever used filtered CC
   data is unverified.
4. SVD succeeded on six T2 candidate clusters but **timed out on cluster 0** (n=1732, the dominant
   one) even subsampled to 150 events.

`promote()` rewrites `hypoDD.inp` line 2 to `dt.cc.authoritative` — necessary because hypoDD names
inputs relatively, and in `input_files/` the bare name `dt.cc` means the deliberately-unfiltered
raw file. Always read `PROVENANCE.md` to learn which CC file a run actually used.

> **Basal-band constants are catalogue-specific.** `07_cluster_structure/plot_hypodd_t2_basal_3d.py`
> defaults to a 1.1 km floor fitted to the authoritative catalogue; under the candidate it silently
> drops **20.8%** of events — precisely the population that distinguishes the two. Override with
> `T2_BASAL_MIN_KM` / `T2_BASAL_MAX_KM` (0.6 recovers it).

## 9. Ocean tide model

`full_catalog_pipeline/10_tides_gps/`

**CATS2008_v2023** (`CATS2008_v2023.nc`, 1.8 GB, [USAP-DC 601772](https://www.usap-dc.org/view/dataset/601772),
CC BY 4.0), 2 km circum-Antarctic grid including sub-ice-shelf cavities, constituents M2 S2 N2 K2
K1 O1 P1 Q1 Mf Mm. Synthesised with pyTMD 2.1.7's OTIS-convention predictor (`pyTMD.predict.time_series`).
Round-trip validated: re-fitting the synthesised series recovers input amplitudes to 0.99–1.12
(deviations are nodal, largest for K1/O1/Mf as expected over 2 years) with variance explained
0.99937.

**Both arrays lie outside the model domain** (`mask = 0`, grounded ice). Nearest wet cell is
**209 km for T1 and 140 km for T2**, and the model's own flexure field is zero throughout the
surrounding region — consistent with an elastic decay length of 10–20 km. So the tide used is the
**grounding-zone forcing**, and any transfer to the array must be treated as an ice-stream-scale
response with fitted gain and lag, not local flexure.

The local tide is **mixed, mainly diurnal** (form factor (K1+O1)/(M2+S2) = 2.7), and **M2 is the
weakest semidiurnal constituent**:

| constituent | T1 | T2 |
|---|---|---|
| K1 | 33.7 cm | 33.6 cm |
| O1 | 28.1 | 28.1 |
| S2 | 17.8 | 17.8 |
| P1 | 10.8 | 10.8 |
| **M2** | **5.4** | **4.7** |

> **Do not use `gps_data/thwaites_tidal_predictions.csv`.** It is not a model output. It is the silent fallback of
> `estimate_tides_pytmd.py` (on the GPS host, `jakewalter.mynetgear.com:~/gps/`, not in this repo) when the model grid is absent: four hardcoded
> amplitudes (M2 0.50, S2 0.15, K1 0.25, O1 0.20 m) with every other constituent exactly zero,
> recovered by harmonic fit with variance explained 1.0000. It is **M2-dominant where the real
> tide is diurnal-dominant** — M2 off by 8.6×. Ten scripts in `~/gps/` read it, so any prior
> conclusion about tidal phase, constituent identity, or which band drives the response is
> unreliable; amplitude-only results are less affected.

## 10. Tidal modulation of seismicity, and the GPS response

### 10.1 Do T1 and T2 switch on together?

`t1_t2_temporal_coherence.py`. 709 days of overlap covering ~100% of both catalogues.

| | Spearman | p (circular-shift null) |
|---|---|---|
| daily rate, raw | +0.450 | <0.0005 |
| daily, 29-day mean removed | +0.335 | <0.0005 |
| **hourly, days both active** | **+0.121** | 0.003 |

Cross-correlation peaks exactly at lag 0. Availability is not the driver (§1). Significance is
against a **circular-shift null**, which preserves each series' own burstiness — a textbook
p-value is meaningless here, since the null |r| p95 is 0.137–0.158.

Regional forcing is real but a minority of the variance (~20% daily, ~1% hourly), and the shared
signal is dominantly long-period. Most large bursts are one-array-only. **The two sites should not
be treated as one coherently-forced system at tidal periods.**

### 10.2 Tidal modulation of icequake timing

`tidal_modulation_test.py`, `tidal_forcing_response.py`. Two tests, both against a
burst-preserving null (each day's event count kept, time-of-day randomised):

| | T1 spec p / phase p | T2 spec p / phase p |
|---|---|---|
| **M2** | 0.514 / 0.422 — null | 0.634 / 0.706 — null |
| **K1** | **0.026 / 0.038** | **0.014 / 0.002** |
| O1 | **0.000 / 0.000** | 0.068 / 0.124 |
| K2 | 0.228 / 0.864 | **0.016 / 0.000** |

**K1 is the only constituent significant on both tests at both arrays; M2 is flatly null.** This
matches the forcing spectrum in §9 constituent-by-constituent — the M2 null is because M2 is tiny
here, not because the ice stream filters the semidiurnal band. Vector strengths are 0.04–0.07, so
a few percent modulation.

Three methodological constraints, each of which invalidated an earlier version of this analysis:

- **A 24.000 h solar/thermal cycle is present** (time-of-day rate varies 45–59% peak-to-trough).
  P1 (24.066 h) and K1 (23.934 h) lie within ~2 Rayleigh widths of it. All numbers above are after
  regressing out 24/12/8/6 h. O1 (25.82 h, well separated) is the most trustworthy detection.
- **Fortnightly constituents are untestable with this null.** The within-day shuffle preserves each
  day's count exactly, so the daily series is identical between observed and null and the spectral
  test has no power at multi-day periods. Mf/Msf/Mm are excluded.
- **The analytic Schuster p is void here.** exp(−R²/n) assumes independent events; these arrive in
  swarms. For Mf it returned 1.4e−38 against a shuffle-null p of 0.28 — inflation ~10³⁷.
- **S2 cannot be separated from solar forcing** at all: it sits at exactly 12.0000 h, identical to
  the second harmonic of the thermal cycle, so the solar regression removes it too. S2 is the third
  largest constituent (17.8 cm), so this is a real gap.

### 10.3 GPS: the glacier flows faster at low tide

Seven GPS stations (T01A–D, T02A–C), 5-min sampling, cleaned product
`gps_data/cache_disp_v3/*.parquet` (`vel_raw`, `vel_sm`, `mag`, `disp_e`, `disp_n`). Valid
**2020-01-02 → 2020-04-22**, 73–86% coverage — about 100 days of the 710-day seismic catalogue,
which is why velocity is predicted for the remainder (§10.4).

`vel_sm` is a 2 h per-segment Savitzky-Golay of `vel_raw`; its gain across 20–30 h is 0.985–1.002
(10–14 h: 0.974–0.994), and results agree to ±0.024 in correlation and ±8° in phase between the
two columns, so the smoothing does not affect anything here.

Band-limiting to 20–30 h is essential — the diurnal band carries 62 cm of the 78 cm tide while
>90% of velocity variance is non-tidal. **Artifact rejection matters more.** Using the project's
own validated `robust_outlier_mask` (8·MAD, cross-checked against
`noise_analysis/{STATION}_quality_flags_v2.csv`):

| station | km from ocean | raw band r | **cleaned band r** | dropped |
|---|---|---|---|---|
| T02A | 129 | −0.504 | **−0.614** | 0.9% |
| T02B | 135 | −0.471 | **−0.537** | 1.8% |
| T02C | 138 | −0.105 | **−0.437** | 2.3% |
| T01B | 213 | −0.210 | **−0.400** | 4.4% |
| T01C | 217 | −0.353 | −0.369 | 7.9% |
| T01D | 221 | −0.348 | −0.317 | 5.4% |
| T01A | 204 | −0.106 | **−0.024** | 5.6% |

**All stations negative — velocity peaks at low tide — with mean phase −140° to −164°** (180° would
be exactly at low tide). Additionally dropping every 5-min bin containing a `marked_bad` epoch
changes nothing beyond ±0.02, which validates the cheaper 8·MAD reject.

Cleaning also resolves what looked like spatial structure: **T02C was an artifact, not an outlier**
(−0.105 → −0.437 on removing 2.3% of samples). Excluding T01A, T02 sits at −0.44 to −0.61 and T01
at −0.32 to −0.40 — a consistent separation matching the 80 km difference in distance, where
uncleaned the two groups overlapped. **T01A is the genuine outlier** (−0.024, phase +94°, the only
station not near 180°) and needs investigating.

> **Propagation is not measurable from these phases.** At a ~24 h period a −171° lag and a +189°
> lead are the same number, so one diurnal cycle of ambiguity swamps any inland delay. A broadband
> lag scan gave 18 h between stations 10 km apart, which is noise. Measuring propagation needs a
> non-periodic marker — a spring-tide envelope maximum or a discrete speed-up event.

### 10.4 Predicting velocity where GPS is absent

`gps_tide_admittance.py` fits a **complex admittance per constituent**
(`Z_i = H_gps,i / H_tide,i`) rather than regressing on tide height, because the response is
frequency-dependent and lagged — a scalar regression forces one lag on all constituents and yields a
correlation of only |r| ≈ 0.01. Predicted hourly velocity over the full 710-day catalogue is written to
`artifacts/full_run/T2_v5/T0{1,2}_predicted_velocity.csv` (git-ignored; regenerate with the
script).

**This prediction is provisional and was fitted on uncleaned data.** The tidal fit explains only
3.2–5.5% of `vel_sm` at T01 and 3.6–12.1% at T02, and the admittance is not stable between
stations that must share it (T01 K1 gain spread 5.9×, O1 phase spread 259°; T02 better at 1.4–1.7×
and ~58°). It should be refitted on the cleaned series (§10.3) and validated by fitting one half of
the GPS window and predicting the other before use.

---

## Open items, ordered by how much they block

1. **Refit §10.4 on cleaned GPS**, and split-half validate. Cheap, and everything downstream of the
   prediction depends on it.
2. **Investigate T01A** (§10.3) — the one station with no tidal response.
3. **Decide on promotion** (§8.3): resolve or accept `clean_ccstrong`'s 500 m cluster shift; run
   T1's second seed; write T1 a `PROVENANCE.md`.
4. **SVD on T2 candidate cluster 0** — needs more than the 45-minute timeout, or a smaller subsample.
5. **Propagation test** with a non-periodic marker (§10.3).
6. **Recover QuakeMigrate grid parameters and T1's configuration** (§2), and the picker's test-set
   metrics (§4) — both needed for a manuscript, neither present in this repository.
7. `regenerate_dtcc.py`'s network default for T1 is still `2E` and will reproduce the §1 trap. It
   should assert its station tags against `station.sel` before writing.

## Software

| tool | version / note |
|---|---|
| QuakeMigrate | reference catalogue |
| SeisBench / PhaseNet | picker (Zhu & Beroza 2019; Woollam et al. 2022) |
| pyocto | v0.1.9 (Münchmeyer 2024) |
| hypoDD | v2.1beta (Waldhauser & Ellsworth 2000); separate `hypoDD_svd` binary for ISOLV=1 |
| GrowClust | independent relocation cross-check |
| pyTMD | **2.1.7 exactly** — 3.x needs Python ≥3.10 union syntax and fails on this Python 3.9 |
| CATS2008_v2023 | ocean tide model, USAP-DC 601772 |
| ITS_LIVE | surface velocity mosaic (RGI19A v02), 120 m |
| RPNet, SKHASH | polarity picking / focal mechanisms, separate `rpnet` conda env |

Citation keys/years are by common convention and were not verified against a bibliography.
