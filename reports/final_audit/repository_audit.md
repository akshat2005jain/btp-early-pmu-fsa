# Phase 1 — Complete Repository and Scientific Audit

**Branch:** `btp-publication-audit` (branched from `e573837c249c7ef83a01f635db0d2e2ca2101a32`)  
**Audit Date:** 2026-11-17  
**Canonical Benchmark Dataset:** 750 scenarios (`270` ID + `480` OOD), SHA-256 `b543ffc51c757ddc7c0316549a6e86db3381df6531abaa61c85fb3df8aa47589`

---

## 1. End-to-End Pipeline Trace

Every production source file, CLI script, configuration file, dataset manifest, saved estimator, evaluation output, figure script, and test suite was inspected line by line across the complete path:

1. **Operating Conditions & Network Initialization (`src/pmu_fsa/simulation/ieee39_data.py`)**:
   - Builds the New England IEEE 39-bus, 10-machine benchmark in Pandapower (`S_base = 100 MVA`, nominal frequency `f0 = 50.0 Hz`, Bus 31 slack, regional Bus 39 plant `H_39 = 42.0 s` on `100 MVA` base by default [`macro_intertie: false`], with opt-in `--macro-intertie` `H_39 = 500.0 s`).
   - Base generation `6,140.81 MW` (Buses 30, 32–39 total `5,620.00 MW` + Bus 31 slack `~520.81 MW` at `load_scale = 1.0`), base load `6,097.10 MW` across 19 load buses.
   - `solve_power_flow()` converges full Newton-Raphson AC power flow (`max_iteration = 50`, `tolerance_mva = 1e-8`) at the scaled constant-power operating point before dynamic initialization.

2. **Renewable Integration Representation (`src/pmu_fsa/simulation/renewables.py`)**:
   - Implements **Approach B (Co-Located Constant-Power GFL Effective Inertia/Damping/Droop Derating)**: active power injections at generator terminal buses remain at their power-flow setpoints while synchronous inertia $H_i$, mechanical damping $D_i$, and governor droop participation $1/R_{\mathrm{sys},i}$ are scaled by $(1 - \alpha_{\mathrm{pen}})$ (`min_h_fraction = 0.05` safety floor, never triggered for $\alpha_{\mathrm{pen}} \in [0.10, 0.50]$).
   - Optional grid-forming (GFM) synthetic inertia (`h_ren`) and fast inverter droop (`r_ren`, `t_inv`) are implemented and active when `control_mode == "GFM"` and `enable_droop=True`.
   - **Audit Finding (`ISS-004`)**: Documentation previously described this as physical renewable active-power displacement across network buses rather than co-located effective-response derating, and `DynamicSimulator.rhs()` (only called when `enable_droop=True`) omitted zeroing the 4th inverter state on a complete unit trip.

3. **Disturbance & Dynamic Simulation (`src/pmu_fsa/simulation/dynamic_engine.py`)**:
   - Solves classical 3rd/4th-order synchronous machine + IEEEG1-type governor-turbine ODEs (`scipy.integrate.solve_ivp`, `RK45`, `rtol=1e-6`, `atol=1e-8`) coupled to Kron-reduced internal-node admittance $Y_{\mathrm{red}}$ and network algebraic bus voltage reconstruction $V_{\mathrm{bus}} = Y_{LL}^{-1}(-Y_{LG} E_{\mathrm{gen}})$.
   - Complete unit trip (`loss_fraction >= 0.999` at $t_0 = 1.0\text{ s}$): sets internal EMF $E_{\mathrm{eff}}[k] = 0$, removes electrical injection from $Y_{\mathrm{red}}$ and $V_{\mathrm{bus}}$, zeros mechanical power & state derivatives, and excludes the disconnected unit from surviving-machine COI ground-truth computation:
     $$f_{\mathrm{COI}}(t) = \frac{\sum_{i \in \mathcal{G}_{\mathrm{on}}(t)} H_{\mathrm{eff},i}\, f_i(t)}{\sum_{i \in \mathcal{G}_{\mathrm{on}}(t)} H_{\mathrm{eff},i}}, \qquad f_{\mathrm{nadir}} = \min_{t \ge t_0} f_{\mathrm{COI}}(t)$$
   - Partial generation loss (`loss_fraction < 0.999`): steps down mechanical input $P_m$ while keeping the unit synchronized and electrically connected.

4. **PMU Measurement Channel & Filtering (`src/pmu_fsa/simulation/pmu_recorder.py`)**:
   - Samples complex bus voltages at `50 fps` ($\Delta t = 20\text{ ms}$), injects independent zero-mean Gaussian phase-angle noise ($\sigma_\theta$) and magnitude noise ($\sigma_V$), unwraps bus phase angles, estimates bus frequency via causal backward finite differences $f_b[k] = f_0 + \frac{\theta_b[k] - \theta_b[k-1]}{2\pi \Delta t}$, applies a **strictly causal forward-only (`scipy.signal.lfilter`, never `filtfilt`)** 2nd-order low-pass Butterworth filter ($f_c = 10\text{ Hz}$), and computes causal backward-difference ROCOF.
   - Forms observable PMU system frequency proxy `pmu_sys_frequency` as uniform mean across observed buses (zero simulator inertia weights $H_i$ used).

5. **Observation Window Slicing & Canonical 14-Feature Schema (`src/pmu_fsa/features/window_slicer.py`, `src/pmu_fsa/features/extractor.py`)**:
   - Slices $[t_0, t_0 + T_{\mathrm{obs}}]$ (`idx_end = searchsorted(t, t0 + T_obs, side='right')`), strictly causal up to $t_0 + T_{\mathrm{obs}}$.
   - Extracts exactly 14 canonical PMU features (`FEATURE_NAMES`, `len == 14`), with zero internal simulator states (`coi_frequency`, rotor speeds, `H_eff`, `p_lost_mw`).
   - **Audit Finding (`ISS-003`)**: `extractor.py` and `window_slicer.py` retained a legacy fallback `traj.get("coi_frequency")` that would trigger if both `bus_frequencies` and `pmu_sys_frequency` were missing from a caller's dictionary. This fallback must be removed so missing PMU channels raise a hard error.

6. **Scenario Dataset & Scenario-Level Splits (`src/pmu_fsa/simulation/scenario_generator.py`, `dataset_validator.py`, `data/manifest.json`)**:
   - 750 unique scenarios (`270` ID + `480` OOD), zero legacy 450-scenario artifacts mixed into `data/processed/` or `models/`.
   - Strict scenario-level split of the 270 ID scenarios: `train = 148`, `val = 70`, `test = 52`, plus `ood_test = 480` (all disjoint at scenario level across all 4 observation windows `0.5, 1.0, 1.5, 2.0 s`).
   - Causality by window:
     - `0.5 s`, `1.0 s`, `1.5 s`: all `750 / 750` scenarios satisfy $t_0 + T_{\mathrm{obs}} < t_{\mathrm{nadir}}$ (`min lead time = +0.280 s` at `1.5 s`).
     - `2.0 s`: `270 / 270` ID scenarios are causal, while `8 / 480` OOD scenarios have $t_{\mathrm{nadir}} \le 3.00\text{ s}$ (`472` strictly causal OOD scenarios).
   - **Audit Finding (`ISS-001`)**: `WindowSensitivityBenchmark` and `InterpretabilityAnalyzer.compute_importance_across_windows` used `FeatureExtractor(strict_nadir=False)`, evaluating `2.0 s` OOD metrics on all `480` scenarios (`min lead time = -0.220 s`) instead of filtering to the `472` strictly causal scenarios.
   - **Audit Findings (`ISS-007`, `ISS-008`)**: Tracked manifests/reports contained local `/Users/akshat/...` paths, `setup.py` lacked `install_requires`, and `manifest.json` lacked per-scenario configuration/eligibility records and git/schema metadata.

7. **Model Training, Calibration, Evaluation & Statistical Testing (`scripts/train_models.py`, `scripts/evaluate_all.py`, `src/pmu_fsa/evaluation/*`)**:
   - All models fit exclusively on `train` (`N = 148`); classifier probability calibration (`Platt` logistic and `Isotonic`) fit exclusively on `val` (`N = 70` here, or `N_cal` validation partition, zero test leakage).
   - **Audit Finding (`ISS-005`)**: On validation (`N = 70`), `linear` (Ridge) achieved MAE `3.36 mHz` vs `gbdt` `10.60 mHz` (`best_regressor = "linear"`), yet `evaluate_all.py` evaluated only `gbdt` in downstream sensitivity/OOD/noise/statistical/latency modules. Both the validation-selected model (`linear` Ridge) and the proposed nonlinear model (`gbdt`) must be evaluated and reported transparently side by side.
   - **Audit Finding (`ISS-006`)**: Saved `.joblib` models lacked embedded `artifact_metadata_` (`feature_schema_version`, `feature_names`, `seed`, `git_commit`, `dataset_sha256`), and `models/` had duplicate `calibrated_gb_iso.joblib` and `calibrated_gb_isotonic.joblib`.
   - **Audit Finding (`ISS-002`)**: `statistical_tests.py` computed an uncentered bootstrap tail-sign quantity `p_boot`, combined Wilcoxon and bootstrap with an unadjusted `OR` rule (`p_wilcoxon < 0.05 or p_boot < 0.05`), and lacked Holm multiple-comparison correction—causing Brier comparisons whose 95% CIs crossed zero (`calibrated_gb_platt` vs `logistic_clf` and `dt_clf`) to be marked statistically significant due to tie/rank artifacts.
   - **Audit Finding (`ISS-009`)**: `fig01` claimed `"Full Rank 14/14"` whereas noiseless `1.0 s` pre-nadir ramps have rank `13/14` (`freq_drop_end == freq_drop_max` on monotonic pre-nadir drops, becoming `14/14` under PMU noise); physical threshold sensitivity (`49.20–49.80 Hz`), single-family feature drop ablations (`no_rocof`, `no_freq_level`, `no_curvature`), full 100-rep latency timing retention, and the statistical + latency distribution figure (`fig14`) needed to be added.

---

## 2. Disposition of All 18 Historical Issues

| Historical ID | Issue Description | Current Codebase Status | Verification Evidence |
| :--- | :--- | :--- | :--- |
| `HIST-001` | Generator-trip modelling reduced power without removing tripped generator's electrical contribution | **VERIFIED_ALREADY_FIXED** (with minor `rhs()` droop fix tracked in `ISS-004`) | `dynamic_engine.py` sets `e_eff[trip_idx] = 0.0` in both `Y_red` (`Pe`) and `V_bus` (`reconstruct_bus_voltages`), zeros mechanical/rotor state derivatives, and excludes disconnected machine from surviving-unit COI (`online_mask`). |
| `HIST-002` | Renewable integration represented through effective inertia/damping/droop derating rather than explicit bus active-power displacement | **ACTIVE (`ISS-004`) — Resolve via Phase 3.3 Approach B** | Equations in `renewables.py` are internally consistent for co-located constant-power GFL derating; documentation/README/reports updated to state Approach B explicitly without claiming bus-level dispatch displacement, plus 4th-state trip zeroing fixed in `rhs()`. |
| `HIST-003` | Internal COI frequency entering model features instead of measured PMU channels | **ACTIVE (`ISS-003`) — Hardening Residual Fallback** | Canonical 750-scenario features use `pmu_sys_frequency` / `bus_frequencies`, but `extractor.py` and `window_slicer.py` retained a fallback to `coi_frequency` if PMU keys were missing. Removing fallback completely. |
| `HIST-004` | Noise added to only part of measurement pipeline | **VERIFIED_ALREADY_FIXED** | `PMURecorder.record()` and `NoiseStressTest` inject complex phasor angle ($\sigma_\theta$) and magnitude ($\sigma_V$) noise onto raw bus voltages prior to angle unwrapping, causal Butterworth filtering, bus frequency differentiation, and ROCOF calculation. |
| `HIST-005` | Independently calculated ROCOF bypassing causal measurement filter | **VERIFIED_ALREADY_FIXED** | `PMURecorder.record()` computes `bus_rocof = np.diff(bus_freq_filtered, axis=1) / dt_pmu` strictly from the causally `lfilter`-smoothed bus frequencies. |
| `HIST-006` | Noise configuration arguments not matching dataclass interface | **VERIFIED_ALREADY_FIXED** | `PMUNoiseConfig(snr_db=..., angle_noise_std_rad=..., mag_noise_std_pu=..., seed=...)` matches all callers across `scenario_generator.py`, `noise_stress.py`, and tests. |
| `HIST-007` | Duplicate feature aliases or inconsistent feature counts | **VERIFIED_ALREADY_FIXED** (rank note tracked in `ISS-009`) | Canonical `FEATURE_NAMES` has 14 distinct features consistently across `extractor.py`, `data/processed/`, `models/`, and evaluation scripts. Note: noiseless monotonic `1.0 s` windows have `freq_drop_end == freq_drop_max` (rank `13/14`, `14/14` with PMU noise), documented in `ISS-009`. |
| `HIST-008` | Nadir labels based on different physical frequency definition from documented target | **VERIFIED_ALREADY_FIXED** | Ground-truth `f_nadir` in `dynamic_engine.py`, `scenario_generator.py`, `extractor.py`, and `dataset_validator.py` uniformly uses surviving-machine post-disturbance COI minimum (`SECURITY_THRESHOLD_HZ = 49.50 Hz`, secure iff $f_{\mathrm{nadir}} \ge 49.50\text{ Hz}$). |
| `HIST-009` | Observation-window slicing did not strictly enforce pre-nadir availability | **ACTIVE (`ISS-001`)** | Fixed in `extract_features.py --strict-nadir` (`X_2.0s.npy` has `742` rows), but `WindowSensitivityBenchmark` and `InterpretabilityAnalyzer` still ran `strict_nadir=False` (`480` OOD rows at `2.0 s` including `8` post-nadir cases). Fixing in Phase 5. |
| `HIST-010` | Old 450-scenario data and old model artifacts mixed with newer 750-scenario dataset | **VERIFIED_ALREADY_FIXED** (duplicate `.joblib` alias tracked in `ISS-006`) | All files in `data/raw/` (`750`), `data/raw_sim/` (`750`), `data/processed/` (`750` for `0.5–1.5 s`, `742` for `2.0 s`), and `models/` derive from the 750-scenario dataset. Redundant `calibrated_gb_isotonic.joblib` cleaned in `ISS-006`. |
| `HIST-011` | Old scripts using random row splits or evaluating on training data | **VERIFIED_ALREADY_FIXED** | `ScenarioGenerator.assign_splits()` partitions strictly by `scenario_id` (`train=148`, `val=70`, `test=52`, `ood_test=480`) with zero scenario overlap across splits or windows. |
| `HIST-012` | Artificial monotonicity smoothing or forced changes to reported metrics | **VERIFIED_ALREADY_FIXED** | Zero metric smoothing, clipping, or post-hoc overriding exists anywhere in `src/` or `scripts/`. |
| `HIST-013` | Dummy or incorrectly selected models in latency and evaluation scripts | **ACTIVE (`ISS-005`)** | `latency.py` loads real `.joblib` models (not dummy models), but `evaluate_all.py` defaulted to `gbdt` instead of evaluating validation-selected `best_regressor` (`linear` Ridge) alongside `gbdt`. Fixing in Phases 8–12. |
| `HIST-014` | Figures containing illustrative, random or hardcoded quantitative data | **ACTIVE (`ISS-009`)** | Quantitative figures `fig03`–`fig13` read real `.npz`/`.csv`/`.joblib` data, but `fig01` hardcoded `"Full Rank 14/14"` and Phase 13 requires data-driven statistical comparison + empirical latency distribution plots (`fig14`). Fixing in Phase 13. |
| `HIST-015` | Test suites validating reference implementations instead of production modules | **VERIFIED_ALREADY_FIXED** | All unit (`tests/test_*.py`) and E2E (`tests/e2e/test_*.py`) modules import and test `pmu_fsa.*` and `scripts/*` directly. |
| `HIST-016` | Inconsistent 50 Hz and 60 Hz assumptions | **VERIFIED_ALREADY_FIXED** | `NOMINAL_FREQ_HZ = 50.0` across `ieee39_data.py`, `configs/default_50hz.yaml`, `pmu_recorder.py`, `extractor.py`, and all test fixtures. |
| `HIST-017` | Inconsistent Bus 39 macro-intertie inertia settings or reproduction commands | **VERIFIED_ALREADY_FIXED** | `GENERATOR_DATA` default Bus 39 `H = 42.0 s` (`macro_intertie: false` in `configs/default_50hz.yaml`; optional `--macro-intertie` `H = 500.0 s` on `100 MVA` base) is consistent across code, YAML, and CLI scripts. |
| `HIST-018` | Documentation mismatches concerning model assumptions, features, thresholds, data and implementation | **ACTIVE (`ISS-004`, `ISS-005`, `ISS-007`, `ISS-009`, `ISS-010`)** | Updating `README.md`, `BTP_Complete_Project_Guide.pdf`, manifests, and audit reports so all claims, paths, models, statistics, limitations, and team attributions match verified code. |

---

## 3. Prioritized Active Issues (`ISS-001` – `ISS-010`)

1. **Scientific & Causal Validity (`CRITICAL`)**:
   - `ISS-001`: Enforce strict pre-nadir filtering ($t_0 + T_{\mathrm{obs}} < t_{\mathrm{nadir}}$) in `sensitivity.py` and `interpretability.py` (`472` causal OOD scenarios at `2.0 s`, `8` excluded) and report both `window_specific_causal` and `common_causal_intersection` populations.
   - `ISS-002`: Replace uncentered bootstrap `p_boot` and unadjusted Wilcoxon-OR-bootstrap rule in `statistical_tests.py` with null-centered paired sign-flip permutation test + paired bootstrap 95% CI + Holm multiple-comparison adjustment + separate Wilcoxon diagnostic.
   - `ISS-003`: Remove residual `coi_frequency` fallback in `extractor.py` and `window_slicer.py`.
   - `ISS-004`: Fix 4th-state complete-trip zeroing in `DynamicSimulator.rhs()` for `enable_droop=True` and align all renewable documentation with **Approach B** (co-located constant-power GFL effective-response derating).
2. **Experimental & Model Selection Validity (`HIGH` / `MEDIUM`)**:
   - `ISS-005`: Evaluate and report both validation-selected `linear` (Ridge) and proposed `gbdt` across all evaluation benchmarks, and select `calibrated_gb.joblib` via validation Brier.
   - `ISS-006`: Embed `artifact_metadata_` in all `.joblib` artifacts, validate schema on load, and remove duplicate `calibrated_gb_isotonic.joblib`.
   - `ISS-009`: Fix `fig01` rank label (`13/14` noiseless, `14/14` noisy), add physical security threshold sensitivity (`49.20–49.80 Hz`), add single-family feature drop ablations, keep all `100+` latency trial timings, and generate `fig14` (statistical comparisons + latency distributions).
3. **Reproducibility, Packaging & Attribution (`MEDIUM` / `LOW`)**:
   - `ISS-007`: Normalize tracked paths to portable repo-relative POSIX paths.
   - `ISS-008`: Add `pyproject.toml`, `requirements.txt`, full `setup.py` `install_requires`, and enriched `data/manifest.json` scenario-level audit records.
   - `ISS-010`: Add team members (**Aditya Jha, Akshat Jain, Arshdeep Singh, Raj Bathla**) to `README.md`, `setup.py`, `pyproject.toml`, and `scripts/generate_project_guide.py` (while strictly honoring the user's instruction not to create or edit any PPT or research paper).

---

## Checkpoint 1 Status

**CHECKPOINT 1: PASS** — Every historical issue (`HIST-001`–`HIST-018`) and every active repository defect (`ISS-001`–`ISS-010`) has been traced to exact source lines and recorded with a clear disposition and verification plan.
