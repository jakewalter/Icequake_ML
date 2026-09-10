# `full_catalog_pipeline` — script index

Reorganized 2026-09-10 into stage directories. **Run everything from the repository root**, not
from inside a stage directory:

```bash
cd /home/jwalter/Icequake_ML
python full_catalog_pipeline/04_relocation/hypodd_tune.py report --array T2
```

Data paths inside the scripts are relative to the repository root. Each script carries a short
`sys.path` bootstrap so that shared modules (`common/catalog_paths.py`, `common/config.py`,
`04_relocation/vels1d_model.py`, `lib/*`) and sibling scripts import correctly from any stage
directory. If you add a new script, copy that bootstrap block from a neighbour.

For the scientific narrative and every parameter value, see **[`../METHODS.md`](../METHODS.md)**.
Manuscript prose lives in `../docs/` (`METHODS_paper.md`, `METHODS_summary.md`, plus `.docx`),
which is **not tracked by git** — see the root README. This file is only a map of the code.

---

## Two things to know before you run anything

**1. Which relocation is "authoritative" is a single switch.** `common/catalog_paths.py` names it
in one place; everything downstream imports from there. Override at runtime without editing
anything:

```bash
ICEQUAKE_RELOC=hypodd_vels1d_ccstrong python full_catalog_pipeline/09_margin_flow/plot_margin_oriented_sections.py --array T2
```

Derived artifacts (cluster event-id lists, stacks, figures) are written to that relocation's own
`work_dir`, deliberately — cluster membership belongs to the relocation that produced it and must
not be silently reused against another.

**2. Nothing is promoted yet.** The current authoritative relocations are `hypodd_vels1d` for both
arrays. The improved upsampled-`dt.cc` runs are staged as *candidates* (`hypodd_vels1d_ccstrong`
for T2, `hypodd_vels1d_t1ccstrong` for T1) and are **not** promoted. See METHODS.md §8 for what
promotion would involve and what is still unresolved.

---

## Stage directories

### `common/` — shared infrastructure
| script | role |
|---|---|
| `catalog_paths.py` | single source of truth for which relocation is authoritative; `ICEQUAKE_RELOC` override |
| `config.py` | archive paths, station lists, window/SNR parameters for the dataset build |

### `lib/` — importable helpers (not entry points)
`day_volume_index` (continuous-archive index), `windowing` (waveform extraction + rolling cache),
`deej_waveform_common` (shared CLI/loaders for waveform analyses), `snr`, `snr_analysis`,
`quakeml_fast`, `seisbench_pack`, `raytrace1d` (1-D ray tracer, see METHODS.md §7).

### `01_dataset/` — training-set construction
Run in numeric order. `01_extract_picks` → `02_index_continuous_data` → `03_extract_windows` →
`04_compute_snr_and_filter` → `05_pack_seisbench` → `06_verify_dataset`.
`04a`/`04b`/`04c` are SNR-threshold analyses; `plot_abc_summary` draws the ablation figure.

### `02_picker/` — PhaseNet training and deployment
`train` → `evaluate`; `classify_continuous` / `batch_classify` / `deploy_continuous_detection`
apply the model to the continuous archive; `score_continuous_picks` scores it against
hand-verified picks.

### `03_association/` — pyocto association
`associate_pyocto` (the associator), `compare_catalogs` (agreement vs QuakeMigrate),
`pyocto_to_quakeml`, `plot_full_archive_locations`, `plot_pilot_comparison`.

### `04_relocation/` — hypoDD, velocity models, GrowClust cross-check
| script | role |
|---|---|
| `hypodd_relocate` / `hypodd_relocate_vels1d` | build hypoDD inputs and run a relocation |
| `hypodd_tune` | **the main harness**: `run` / `sweep` / `report` / `rescore` / `promote`, `--dtcc` selects the CC file |
| `vels1d_model` | the reflection-seismology 1-D model; `build_layers()` is the authority for velocities |
| `hypodd_damp_sweep`, `compare_damp_sweep` | damping/iteration sensitivity |
| `hypodd_svd_cluster_errors[_t2]` | ISOLV=1 SVD reruns for real per-event errors (`--max-events` subsamples) |
| `convert_hypodd_to_growclust`, `analyze_growclust_t1` | independent GrowClust cross-check |
| `plot_tune_configs`, `plot_tune_depth_consensus`, `plot_svd_resolvability` | tuning diagnostics |
| `plot_t2_upsampled_improvement`, `plot_t2_upsampled_sections` | old vs upsampled `dt.cc` comparison |

### `05_crosscorrelation/` — the `dt.cc` problem and its fix
| script | role |
|---|---|
| `diagnose_cc_p_deficit` | finds the root cause: obspy's parabolic sub-sample fit needs 3 samples and gets 1–2 at 200 Hz |
| `regenerate_dtcc` | **replaces hypoDDpy's CC stage**; `--verify` proves the preprocessing hoist is exact |
| `plot_cc_upsampling_explainer` | figure explaining the fix |
| `cc_refine_component` | from-scratch single-component MCCC refinement (no envelope, no `dt.cc` reuse) |
| `hypodd_cc_depth_information` | diagonal Fisher information for depth, `dt.cc` vs `dt.ct` |

### `06_depth/` — is depth resolved? (answer: no — see METHODS.md §7)
`test_sp_absolute_depth`, `composite_sp_depth_pin`, `composite_relocation_depth_scan`,
`between_cluster_depth_difference`, `depth_velocity_tradeoff` (the ~1:1 depth/velocity trade-off),
`depth_phase_migration` + `cluster_depth_phase_stack` (depth-phase search, negative),
`fit_cluster_depths_raytrace` (proper ray tracing), `run_t2_sp_battery` (S-P battery driver),
`test_mccc_moveout_recovery` (injection–recovery control).

### `07_cluster_structure/` — clustering, stacks, per-cluster character
DBSCAN/Woodcock shape analysis (`plot_hypodd_t{1,2}_basal_3d`, `..._basal_clusters`,
`..._depth_section`), waveform stacks (`build_cluster_stacks`, `cluster_full_stack`,
`cluster_p_stack_polarity`), and the cluster3 firn/orientation investigations.

> The basal-band constants in `plot_hypodd_t2_basal_3d.py` are **catalog-specific**. Override with
> `T2_BASAL_MIN_KM` / `T2_BASAL_MAX_KM`; the default 1.1 km floor silently drops 20.8% of the
> `clean_ccstrong` candidate. See METHODS.md §8.

### `08_focal_mech/` — focal mechanisms (all unresolvable, see METHODS.md §10)
RPNet polarity picking (`rpnet_*`, **needs the `rpnet` conda env**), SKHASH fitting (`skhash_*`),
composite grid search (`focal_mech_*`), stack-based first motions.

### `09_margin_flow/` — ITS_LIVE flow context and margin-frame sections
`plot_hypodd_t{1,2}_shear_margin_map` / `_speed_map` (fetch + cache the ITS_LIVE window),
`plot_margin_oriented_sections` and `plot_margin_strain_profile` (both `--array T1|T2`) rotate a
catalog into the margin frame from the velocity-gradient structure tensor.

### `10_tides_gps/` — tidal modulation and the GPS response
| script | role |
|---|---|
| `tidal_modulation_test` | astronomy-only test: is icequake timing modulated at tidal periods? |
| `tidal_forcing_response` | adds the real CATS2008 tide; supplies the local forcing spectrum |
| `gps_tide_admittance` | per-constituent complex admittance, and velocity prediction beyond the GPS record |
| `plot_low_tide_speedup` | the low-tide speed-up result, all 7 GPS stations |
| `plot_gps_speed_all_stations` | GPS records, Lomb-Scargle spectra, diurnal band |
| `t1_t2_temporal_coherence` | do T1 and T2 switch on together? (partially, and it weakens with timescale) |

### `archive/` — superseded, kept for reference
`plot_t2_margin_oriented_sections.py`, `plot_t2_margin_strain_profile.py` — T2-only originals,
replaced by the `--array` versions in `09_margin_flow/`.

---

## Results committed to the repository

`figures/` (23 files, 6.4 MB) and `tables/` (14 CSVs) hold a curated handoff set, named by stage.
Everything else generated lives under `artifacts/` and is **git-ignored** (~31 GB: waveform
windows, per-run hypoDD trees, 883 figures, 372 CSVs). Regenerate from the scripts above.

## External inputs (not in the repository)

| what | where it came from |
|---|---|
| Continuous waveforms | see `common/config.py`; archived MiniSEED, per station per day |
| `CATS2008_v2023.nc` (1.7 GB) | [USAP-DC dataset 601772](https://www.usap-dc.org/view/dataset/601772), CC BY 4.0 — reCAPTCHA, manual download |
| `gps_data/` | `jakewalter.mynetgear.com:~/gps/` — `noise_analysis/cache_disp_v3/*.parquet` is the cleaned product |

> `gps_data/thwaites_tidal_predictions.csv` is **not** a tide model. It is the hardcoded fallback of `estimate_tides_pytmd.py`
> on the GPS host (not in this repo) (M2 0.50, S2 0.15, K1 0.25, O1 0.20 m, all other
> constituents zero) and is M2-dominant where the real local tide is diurnal-dominant. Do not use
> it. See METHODS.md §9.

## Python environment

Python 3.9. `pyTMD==2.1.7` specifically — 3.x requires Python ≥3.10 union syntax and fails to
import here. Also needs `pyarrow` (parquet), `netCDF4`, `rasterio`, `pyproj`, `obspy`, `scipy`,
`torch` + `seisbench` (picker only). `astropy` present but too old for this numpy
(`np.asscalar` removed) — use `scipy.signal.lombscargle`. RPNet scripts need the separate
`rpnet` conda env.
