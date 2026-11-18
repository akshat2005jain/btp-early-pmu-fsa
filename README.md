# Early PMU-Based Prediction of Frequency Nadir and Security Status in Renewable-Integrated Multi-Area Power Systems Under Limited Observation

An end-to-end simulation, synchrophasor emulation, strictly causal feature engineering, probability-calibrated machine learning, and statistical benchmarking framework (`pmu_fsa`) for early post-disturbance frequency-security assessment on the **IEEE 39-bus, 10-machine** multi-area power system standardized at **50.0 Hz** (`f_secure = 49.50 Hz`).

### Project Team & Supervision
- **Team Members**:
  - **Aditya Jha** (Member 1)
  - **Akshat Jain** (Member 2)
  - **Arshdeep Singh** (Member 3)
  - **Raj Bathla** (Member 4)
- **Faculty Mentor**: **Dr. Deepak Kumar Gupta**

---

## 1. Research Objective

Following a sudden generator outage in a low-inertia power grid, system frequency decays rapidly before turbine governors can arrest the excursion. Full numerical time-domain simulation of nonlinear swing and governor differential equations (`125.38 ms` median per `10 s` contingency) creates a computational bottleneck when screening hundreds of candidate contingencies online, whereas post-mortem machine learning classifiers trained on full trajectories provide zero early-warning lead time.

The primary objective of this study is to quantify **how early after a generation-loss disturbance (`T_obs in {0.5, 1.0, 1.5, 2.0} s` following fault inception at `t_0 = 1.0 s`) strictly causal wide-area PMU measurements suffice to**:
1. **Predict the continuous future Center-of-Inertia (COI) frequency nadir (`f_nadir` in Hz)** within physical bounds `[45.0, 50.5] Hz` before the nadir physically occurs (`t_nadir in [2.78, 3.70] s`).
2. **Deliver calibrated probability estimates `P(f_nadir >= 49.50 Hz)` and safety-asymmetric binary security decisions** (`1 = Secure`, `0 = Insecure`) with quantified epistemic/aleatoric prediction intervals across unseen high-renewable (`40%–50%`), unseen load (`0.80x`, `1.20x`), noisy PMU (`0–50 mHz`), sparse `13/39`-bus PMU (`33.3%`), and `25 fps` reporting regimes.

---

## 2. Scope & Assumptions

### Investigated Within Scope
- Post-contingency primary frequency response following complete and partial synchronous generation outages (`80.0 MW` to `1,200.0 MW`) across all 10 generator buses (`30–39`).
- Synthetic IEEE C37.118-style time-synchronized bus frequency, voltage magnitude, voltage phase angle, and causal ROCOF streams at `50 fps` (primary) and `25 fps` (sensitivity).
- Strictly causal pre-nadir observation windows (`[t_0, t_0 + T_obs]`) with zero access to internal simulator rotor speeds or post-window samples.
- Cross-regime generalization across disjoint scenario-level splits (`train`, `val`, `test`, and three out-of-distribution shifts: `ood_res`, `ood_load`, `ood_combined`).
- Post-hoc probability calibration (Platt scaling vs. Pool Adjacent Violators Algorithm [PAVA] isotonic regression), asymmetric false-secure error control (`FNR_insecure`), permutation interpretability, and hardware-timed CPU inference latency.

### Outside Initial Scope & Modelling Assumptions
- No closed-loop emergency control actuation (e.g., automated Under-Frequency Load Shedding [UFLS] relay tripping, fast frequency response [FFR] dispatch, or reinforcement learning control).
- No cyber-physical packet-injection, spoofing, or denial-of-service attack modelling.
- Classical third-order generator + first-order steam-reheat governor (`TGOV1`-equivalent) electro-mechanical dynamics with constant-impedance Kron-reduced passive loads rather than electromagnetic transient (`EMT`) converter-control loops or utility hardware-in-the-loop (`HIL`) recordings.

---

## 3. IEEE 39-Bus 50 Hz System Model

The benchmark network (`src/pmu_fsa/simulation/ieee39_data.py`, `dynamic_engine.py`) adapts the canonical IEEE 39-bus (New England) 10-machine, 46-branch multi-area test system to a **50.0 Hz** nominal frequency base (`S_base = 100.0 MVA`, `omega_0 = 2 * pi * 50.0 = 314.159 rad/s`, `f_secure = 49.50 Hz` corresponding to the `-0.50 Hz` emergency floor prior to UFLS Stage-1):

- **Three Coherent Operating Areas & Primary Tie-Lines**:
  - **Area 1 (South, 9 buses)**: Buses `[1, 2, 3, 4, 18, 25, 30, 31, 32]` hosting `G1–G3` (Buses `30–32`).
  - **Area 2 (Central, 15 buses)**: Buses `[5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 33, 34, 35, 36]` hosting `G4–G7` (Buses `33–36`).
  - **Area 3 (North, 15 buses)**: Buses `[9, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 37, 38, 39]` hosting `G8–G10` (Buses `37–39`).
  - **Primary Inter-Area Tie-Lines**: Branches `(4, 5)` (`Area 1 <-> Area 2`), `(1, 39)` (`Area 1 <-> Area 3`), and `(16, 17)` (`Area 2 <-> Area 3`).
- **Regional Islanded Bus-39 Equivalent (`macro_intertie = False` Default)**: Bus 39 (`G10`, `1,000 MW`, `1,200 MVA`) is modelled by default as a regional thermal plant (`H_39 = 42.0 s`, `X'_d = 0.0400 pu`), yielding total pre-fault synchronous inertia `H_tot = 324.7 s` on the `100 MVA` base so single-plant outages produce realistic regional frequency nadirs (`48.26–49.91 Hz`). An opt-in `--macro-intertie` flag restores the stiff external intertie equivalent (`H_39 = 500.0 s`, `X'_d = 0.0060 pu`, `H_tot = 782.7 s`).
- **Pre-Fault Newton-Raphson AC Power Flow**: For each load scaling factor (`0.80x–1.20x` of nominal `6,150.1 MW` / `1,408.9 MVAR` across 19 PQ load buses), `solve_power_flow()` converges in 4–5 iterations to `< 1e-11 pu` power mismatch, initializing internal generator electromotive forces `E'_i` and mechanical power `P_m0,i` with `< 1e-8 MW` pre-disturbance imbalance (`0.0000 Hz` flat-run drift).
- **Machine-Base 5% Droop & Coupled Swing/Governor ODEs**: Each unit `i in {1..10}` carries explicit MVA rating `S_mva,i in [350, 1200] MVA` and `5%` speed-droop `R_i = 0.05` on its own machine rating, converted to the `100 MVA` system base via `R_sys,i = R_i * (100.0 / S_mva,i)` with effective steam-reheat prime-mover time constant `T_g,i = 4.20 s` and damping `D_i = 2.0 pu`:
  - `d(delta_i)/dt = omega_0 * Delta_omega_i`
  - `2 * H_i * d(Delta_omega_i)/dt = P_m,i - P_e,i(delta) - D_i * Delta_omega_i`
  - `T_g,i * d(P_m,i)/dt = P_m0,i - (Delta_omega_i / R_sys,i) - P_m,i`

---

## 4. Generation-Loss Representation

Disturbances are applied at `t_0 = 1.0 s` over a `10.0 s` horizon using two-segment adaptive `RK45` integration (`[0, t_0]` with pre-fault Kron matrices `(Y_red,pre, K_pre)` and `[t_0, 10.0 s]` with post-fault `(Y_red,post, K_post)`):

1. **Complete Generator Disconnection (`loss_fraction = 1.0`)**: Removes the tripped unit's Norton shunt admittance `y_g[j] = 1 / (j * X'_d,j)` from the post-fault network admittance matrix `Y_NN`, zeros row/column `j` of `Y_red,post` and column `j` of bus voltage reconstruction operator `K_post`, enforces `P_e,post[j] = P_m,post[j] = Delta_omega_post[j] = 0`, and excludes unit `j` from post-fault COI inertia weighting `H_post`.
2. **Partial Multi-Unit Plant Loss (`0 < loss_fraction < 1`, specifically `0.70` and `0.40`)**: Represents tripping a subset of parallel generating units at a multi-unit power station by scaling `y_g,post[j]`, `H_post[j]`, `D_post[j]`, `P_m0,post[j]`, inverse droop gain `R_sys,post^-1[j]`, and initial turbine output `P_m(t_0^+)[j]` proportionally by `1 - loss_fraction`.

Ground-truth target `ground_truth_nadir` (`f_nadir`) and `ground_truth_t_nadir` (`t_nadir`) are defined strictly on the post-fault **system Center-of-Inertia (COI) frequency** `f_COI(t) = 50.0 * (1 + sum_{i in online} H_post,i * Delta_omega_i(t) / sum_{i in online} H_post,i)`, while `bus_min_nadir` and `nadir_bus_id` are archived in metadata for diagnostic comparison.

---

## 5. Renewable Integration Approximation (Approach B: Co-Located Effective-Response Derating)

Converter-interfaced renewable penetration (`src/pmu_fsa/simulation/renewables.py`) is represented via **Approach B (co-located constant-power grid-following effective-response derating)** across the 9 non-slack generator buses (`DEFAULT_DISPLACED_BUSES = [30, 32, 33, 34, 35, 36, 37, 38, 39]`, totaling `5,620.0 MW` nominal dispatch; Slack Bus 31 excluded):

- **Bounded Physical Representation**: Pre-disturbance terminal bus active-power setpoints in the AC power flow remain unchanged while rotational inertia, mechanical damping, and synchronous governor participation are derated homogeneously by `1 - alpha`. Explicit solar/wind park placement at separate load buses, pre-contingency line-flow redistribution, converter PLL dynamics, and current saturation are outside this model's scope.
- **Requested vs. Realized Penetration**: For requested load-referenced penetration `p_ren_requested in {0.10, 0.20, 0.30, 0.40, 0.50}`, required converter share is `P_ren = p_ren_requested * load_scale * P_load,base`, yielding uniform derating factor `alpha = P_ren / P_target,scaled`. Because total generation (`6,192.8 MW` nominal, including transmission losses) slightly exceeds PQ load (`6,150.1 MW`), generation-referenced `p_ren_realized = [0.09931, 0.19862, 0.29793, 0.39724, 0.49655]` (`9.93%–49.66%`), scaling strictly linearly with zero clipping.
- **Constant-PQ Grid-Following (GFL) Inertia/Damping/Droop Derating**: Under the default GFL converter assumption (`h_vir = 0.0 s`, `enable_droop = False`), displaced synchronous share contributes zero rotational inertia and zero primary converter droop, reducing effective synchronous inertia per displaced bus to `H_eff,i = (1 - alpha) * H_orig,i` (from `H_tot = 294.7 s` at `10%` RES down to `174.6 s` at `50%` RES before outage). Optional synthetic fast converter droop (`enable_droop = True`, `r_ren`, `t_inv`) is supported for grid-forming sensitivity studies.

---

## 6. PMU Measurement Model

`PMURecorder` (`src/pmu_fsa/simulation/pmu_recorder.py`) converts continuous bus trajectories into realistic synchrophasor streams without leaking internal generator states:

- **Reporting Rates & Wrap-Safe Interpolation**: Supports IEEE Std C37.118.1-2011 reporting rates including **`50 fps`** (`Delta t = 20 ms`, primary 50-Hz standard) and **`25 fps`** (`Delta t = 40 ms`, sensitivity case). Bus phase angles are unwrapped in radians prior to linear interpolation onto uniform `SOC`/`FRACSEC` timestamps and re-wrapped to `[-180, 180) deg`.
- **PMU-Only Wide-Area Frequency Reconstruction (`f_sys_pmu`)**: Reconstructs wide-area system frequency strictly from observed PMU `bus_frequencies` using area-weighted observed bus means `(9 * mean(Area_1) + 15 * mean(Area_2) + 15 * mean(Area_3)) / 39`, with zero access to internal rotor speeds `coi_frequency` or `H_eff`.
- **Strictly Causal 5-Point Backward OLS ROCOF**: Estimates bus and system ROCOF (`df/dt`) causally via closed-form backward Ordinary Least Squares slope over `[max(0, n-4) .. n]` (`0` at `n=0`, 2-pt backward difference at `n=1`, 3/4-pt causal OLS warmup at `n=2,3`, and `0.1 * fps * (2*(f[n]-f[n-4]) + f[n-1] - f[n-3])` for `n >= 4`). Compared against naive 2-point finite differences at `5 mHz` noise, causal OLS reduces ROCOF error variance by a factor of `0.0003` (`var_ols = 0.001616` vs. `var_2pt = 4.642054`).
- **Per-Scenario Seeded Measurement Noise**: Injects independent zero-mean Gaussian noise across `bus_frequencies` (`sigma_f`, default `5 mHz`), `voltage_magnitudes` (`sigma_v = 0.0005 pu`), and `voltage_angles` (`sigma_theta_deg = 0.05 deg`) seeded deterministically via `seed + MD5_offset(scenario_id)`, then recomputes causal 5-point ROCOF and `f_sys_pmu`.
- **Simplified Single-Station IEEE C37.118.2 Framing & Partial Observability**: Implements binary frame serialization/parsing (`SYNC = 0xAA01`, `FRAMESIZE`, `IDCODE`, `SOC`, `FRACSEC`, `STAT`, phasor, `FREQ`, `DFREQ`, `CRC-CCITT 0x1021`) and supports partial spatial PMU coverage (canonical 13-bus placement `[1, 2, 4, 5, 6, 9, 14, 16, 17, 19, 22, 25, 39]`, `33.3%` of buses) via `NaN` column masking.

---

## 7. Canonical 14-Feature Definitions & Units

`FeatureExtractor` (`src/pmu_fsa/features/extractor.py`) slices each PMU stream strictly on `W = [t_0, t_0 + T_obs]` via `extract_causal_window()` (stripping all forbidden internal simulator keys) and computes **14 canonical spatiotemporal synchrophasor features** (`CANONICAL_FEATURE_NAMES`):

| # | Canonical Feature Name | Formula on Causal Window `W = [t_0, t_0 + T_obs]` | Units | Physical Interpretation |
|---:|---|---|---|---|
| 1 | `rocof_init` | `OLS_slope(t, f_sys_pmu)` over `[t_0, min(t_0 + 0.10, t_end)]` | `Hz/s` | Initial post-fault ROCOF (`~ -Delta P * f_0 / (2 H_eff)`) |
| 2 | `rocof_peak_neg` | `min_{t in W} rocof_sys_pmu(t)` | `Hz/s` | Steepest downward causal 5-point wide-area ROCOF |
| 3 | `rocof_end` | `rocof_sys_pmu(t_0 + T_obs)` | `Hz/s` | Instantaneous wide-area ROCOF at decision time `t_0 + T_obs` |
| 4 | `rocof_mean` | `OLS_slope(t, f_sys_pmu)` over `[t_0, t_0 + T_obs]` | `Hz/s` | Noise-resilient window-wide average frequency decay slope |
| 5 | `freq_drop_end` | `f_sys_pmu(t_0 + T_obs) - f_sys_pmu(t_0)` | `Hz` | Net wide-area frequency excursion at observation boundary |
| 6 | `freq_drop_max` | `min_{t in W} f_sys_pmu(t) - f_0` | `Hz` | Worst wide-area frequency drop below `50.0 Hz` within `W` |
| 7 | `freq_dev_integral` | `integral_{t_0}^{t_0 + T_obs} (f_sys_pmu(t) - f_0) dt` | `Hz*s` | Cumulative rotor kinetic-energy deficit driving governor droop |
| 8 | `bus_drop_worst` | `nanmin_{t in W, b in B_obs} (f_bus(t, b) - f_0)` | `Hz` | Worst localized substation frequency drop across observed buses |
| 9 | `area_freq_spread_max` | `max_{t in W} max_{a,b in {1,2,3}} |mean_A_a(t) - mean_A_b(t)|` | `Hz` | Peak inter-area frequency separation across Areas 1, 2, and 3 |
| 10 | `bus_freq_std_mean` | `mean_{t in W} nanstd_{b in B_obs} f_bus(t, b)` | `Hz` | Window-mean spatial dispersion of observed bus frequencies |
| 11 | `angle_divergence_max` | `max_{t in W} max_{a,b in B_obs} |wrap_180(Delta theta_a(t) - Delta theta_b(t))|` | `deg` | Peak network-wide relative bus voltage phase-angle stress |
| 12 | `tie_line_angle_excursion` | `max_{t in W} max_{(a,b) in {(4,5),(1,39),(16,17)}} |wrap_180(Delta theta_ab(t))|` | `deg` | Peak synchronizing phase-angle swing across primary tie-lines |
| 13 | `freq_curvature_init` | `OLS_slope(t, rocof_sys_pmu)` over `[t_0, min(t_0 + 0.20, t_end)]` | `Hz/s^2` | Initial second-derivative frequency curvature `d^2f/dt^2` |
| 14 | `freq_curvature_mean` | `OLS_slope(t, rocof_sys_pmu)` over `[t_0, t_0 + T_obs]` | `Hz/s^2` | Window-wide arrest curvature `d^2f/dt^2` (governor valve ramp) |

---

## 8. 750-Scenario Dataset Generation & Separation (`raw_sim` vs `raw`)

The full combinatorial dataset (`configs/default_50hz.yaml`) spans **750 unique contingencies**:
- `10` generator buses (`30–39`) × `3` outage loss fractions (`1.0, 0.70, 0.40`) × `5` renewable penetrations (`0.10, 0.20, 0.30, 0.40, 0.50`) × `5` PQ load scales (`0.80, 0.90, 1.00, 1.10, 1.20`) = **750 scenarios** (`303` Secure [`40.4%`], `447` Insecure [`59.6%`]; COI nadir `[48.2556, 49.9109] Hz`, `t_nadir in [2.78, 3.70] s`).
- **Strict Simulator vs. PMU File Separation**:
  - `data/raw_sim/*.npz` (`750` files): Archives internal ODE trajectories (`generator_angles`, `generator_speeds`, `generator_powers`, `coi_frequency`, `coi_rocof`) exclusively for physics diagnostics and ground-truth verification.
  - `data/raw/*.npz` (`750` files): Contains **only** observable PMU channels (`timestamps`, `soc`, `fracsec`, `bus_frequencies`, `bus_rocof`, `voltage_magnitudes`, `voltage_angles`, `pmu_sys_frequency`) plus scenario provenance metadata. Zero internal ODE states exist in `data/raw/`.
- **Automated 14-Check Quality & Pre-Nadir Causality Audit (`results/dataset_quality_report.md`)**: Verifies `0` duplicate IDs, `0` missing files, `0` channel leaks, exact `1/fps` sampling, COI label consistency, and pre-nadir causality (`750/750 = 100.00%` strictly causal at `T_obs in {0.5, 1.0, 1.5} s`; `742/750 = 98.93%` at `T_obs = 2.0 s`, with the `8` heavy-load scenarios reaching `t_nadir <= 3.00 s` explicitly logged and excluded in `features_t2.0s.{npz,csv}`).

---

## 9. Disjoint `train` / `val` / `test` / `ood_test` Partitioning

Scenario partitioning (`assign_scenario_partition` in `scenario_generator.py`) is assigned ** strictly at the scenario level** via deterministic MD5 hashing of `scenario_id` (`scen_gen{bus}_loss{MW}_ren{pct}_ld{pct}_000`), guaranteeing zero scenario overlap across splits and identical partition membership across all observation windows `T_obs`:

| Partition | Regime (`ood_type`) | Operating Domain Definition | Scenarios (`N`) | Share | Secure (`>=49.50 Hz`) | Insecure (`<49.50 Hz`) | Mean / Min COI Nadir (Hz) |
|---|---|---|---:|---:|---:|---:|---|
| `train` | `id` | `RES in {10, 20, 30}%`, `Load in {0.90, 1.00, 1.10}x` | 148 | 19.73% | 71 (48.0%) | 77 (52.0%) | `49.4612` / `48.7188` |
| `val` | `id` | `RES in {10, 20, 30}%`, `Load in {0.90, 1.00, 1.10}x` | 70 | 9.33% | 34 (48.6%) | 36 (51.4%) | `49.4395` / `48.7358` |
| `test` | `id` | Held-Out ID (`RES in {10..30}%`, `Load in {0.90..1.10}x`) | 52 | 6.93% | 21 (40.4%) | 31 (59.6%) | `49.4271` / `48.8165` |
| `ood_test` | `ood_res` | Unseen High RES (`40–50%`), Seen Load (`0.90–1.10x`) | 180 | 24.00% | 49 (27.2%) | 131 (72.8%) | `49.2054` / `48.2689` |
| `ood_test` | `ood_load` | Seen RES (`10–30%`), Unseen Extreme Load (`{0.80, 1.20}x`) | 180 | 24.00% | 95 (52.8%) | 85 (47.2%) | `49.4874` / `48.7342` |
| `ood_test` | `ood_combined` | Simultaneous High RES (`40–50%`) + Extreme Load (`{0.80, 1.20}x`) | 120 | 16.00% | 33 (27.5%) | 87 (72.5%) | `49.2595` / `48.2556` |
| **Total** | **All** | **Full Combinatorial Grid** | **750** | **100.00%** | **303 (40.4%)** | **447 (59.6%)** | **`49.3694` / `48.2556`** |

---

## 10. Regression Methodology (7 Models)

Seven continuous nadir regressors (`src/pmu_fsa/models/regressors.py`) are trained under identical causal feature inputs, with predictions bounded to physical grid limits `[45.0, 50.5] Hz`:
1. **`MeanNadirRegressor`**: Constant training-mean baseline (`R^2 ~ 0` reference).
2. **`AnalyticalSwingNadirRegressor`**: Physics-informed baseline fitting an L2-regularized (`alpha = 0.01`) linear-quadratic + parabolic swing-vertex basis (`-rocof_init^2 / (2 * max(|freq_curvature_mean|, eps))`) on `[rocof_init, freq_drop_end, freq_curvature_mean]`.
3. **`LinearNadirRegressor` (Ridge)**: `StandardScaler` (fit strictly on training data) + Ridge regression (`alpha = 0.1`).
4. **`DecisionTreeNadirRegressor` (CART)**: Single recursive regression tree (`max_depth = 5`).
5. **`RandomForestNadirRegressor`**: Bagged ensemble (`n_estimators = 100`, `max_depth = 10`).
6. **`GradientBoostingNadirRegressor` (GBDT)**: Stagewise gradient-boosted regression trees (`n_estimators = 100`, `learning_rate = 0.08`, `max_depth = 4`).
7. **`LightGBMNadirRegressor`**: Histogram-based gradient boosting (`n_estimators = 100`, `learning_rate = 0.08`, `num_leaves = 15`; gracefully falls back to scikit-learn `HistGradientBoostingRegressor` with recorded `backend_ = "sklearn_hist_gbdt"` when `libomp.dylib` is absent on macOS).

Model selection is performed strictly on `val` (`N = 70` after fitting on `train`, `N = 148`); final saved estimators (`models/*.joblib`) use equalized `train + val` refitting (`N = 218`) with zero access to `test` (`N = 52`) or `ood_test` (`N = 480`).

---

## 11. Classification & Safety Methodology

Seven security classifiers (`src/pmu_fsa/models/classifiers.py`, `calibrated_gb.py`) predict binary status `y_sec = 1[f_nadir >= 49.50 Hz]`: Logistic Regression, Decision Tree, Random Forest, Uncalibrated GBDT, LightGBM Classifier, and the two proposed **Physics-Guided Calibrated GBDT** variants (`Platt` and `PAVA Isotonic`).

Because misclassifying an actual under-frequency emergency (`f_nadir < 49.50 Hz`, class `0`) as `"Secure"` (class `1`) leaves the grid unshielded against UFLS tripping, evaluation explicitly tracks **Missed-Insecure False-Negative Rate** alongside standard Precision, Recall, `F1`, `PR-AUC`, and `ROC-AUC`:
- **`FNR_insecure = FP_secure / (TN_insecure + FP_secure)`**: Fraction of truly insecure contingencies falsely cleared as secure (`Missed_Insecure` count).
- **`FPR_secure = FN_secure / (TP_secure + FN_secure)`**: Fraction of truly secure contingencies conservatively flagged as insecure (benign false alarms).
- **Decision-Threshold Sweep**: Sweeps probability cutoffs `tau in {0.30, 0.50, 0.60, 0.70, 0.80, 0.90}` (`y_hat = 1[P(Secure) >= tau]`) to identify conservative operating thresholds where `FNR_insecure -> 0.0000`.

---

## 12. Calibration Methodology

`CalibratedGBDTClassifier` (`src/pmu_fsa/models/calibrated_gb.py`) decouples continuous physical margin estimation from probability calibration:
1. **Stage 1 (Continuous Physical Score)**: Base `GradientBoostingRegressor` predicts `s(x) = f_hat_nadir(x)` in Hz, preserving exact physical distance from `49.50 Hz` (`49.49 Hz` vs. `48.26 Hz`).
2. **Stage 2 (Out-of-Fold Calibration)**: Using 5-fold stratified cross-fitting across `train + val` (`N_cal = 218`, zero `test` leakage), out-of-fold continuous nadir predictions `s_oof` are mapped to `[0, 1]` via:
   - **Platt Scaling (`calibrated_gb_platt`)**: L-BFGS logistic sigmoid `P(Secure | s) = 1 / (1 + exp(-(A * s + B)))` with physics monotonicity guardrail `A > 0` (learned `A = 71.8078`, `B = -3554.1001`, centered at `-B/A = 49.4947 Hz`).
   - **PAVA Isotonic Regression (`calibrated_gb_iso`)**: Non-parametric monotonic step-linear interpolation (`216` knots fitted natively with `fallback_used = False`; includes automatic fallback to Platt scaling only when `N_cal < 30`).
3. **Stage 3 (Quantile + Conformal + OOD Distance Intervals)**: Pairs auxiliary `alpha = 0.025` and `alpha = 0.975` quantile GBDT regressors with out-of-fold conformal residual floor (`q_95`) and standardized feature-space Mahalanobis/RMS distance inflation, widening prediction intervals automatically under `40%–50%` OOD domain shift.

---

## 13. Evaluation & Statistical Methodology

All evaluation modules (`src/pmu_fsa/evaluation/`) report **raw, unsmoothed, unclamped** empirical metrics strictly on causally eligible (`t_0 + T_obs < t_nadir`) held-out partitions, evaluating **both** the validation-selected best regressor (`linear` Ridge, Val MAE `3.4 mHz`) and the proposed nonlinear regressor (`gbdt`, Val MAE `10.6 mHz`) side by side:
- **Paired Sign-Flip Permutation & Holm-Adjusted Statistical Testing (`statistical_tests.py`)**: Uses a two-sided null-centered paired sign-flip permutation test on scenario-wise loss differences (`|y - y_hat|` for MAE; `(p - y)^2` for Brier) as the primary `p`-value, applies **Holm (1979) step-down FWER adjustment** across multiple model comparisons, reports `2,000`-resample paired bootstrap 95% confidence intervals, and reports Wilcoxon signed-rank separately as a rank-symmetry diagnostic (without combining tests via an unadjusted `OR` rule).
- **Physical Threshold & Probability Cutoff Sweeps**: Sweeps both physical grid-code nadir thresholds (`f_thresh in {49.20, 49.40, 49.50, 49.60, 49.70, 49.80} Hz`) and probability cutoffs (`tau in {0.30..0.90}`).
- **Multi-Seed & Sample-Complexity Audits**: Evaluates variance across 5 random seeds (`42–46`) and training-set fractions `[20%, 40%, 60%, 80%, 100%]` (`N = 30..148`).
- **6-Stage Desktop Latency Profiling (`latency.py`)**: Uses `time.perf_counter_ns()` (`10` warmup + `100` timed repetitions, retaining all per-repetition timings in `latency_trials.csv`) benchmarked relative to the `20.0 ms` 50-fps PMU frame interval and measured `10 s` `DynamicSimulator` runs on the same CPU (reported strictly as desktop algorithmic compute latency, not hard real-time RTOS/HIL certification).

---

## 14. Verified Experimental Results Summary

All numbers below match `results/tables/*.md`, `results/evaluation_report.md`, and `models/training_metrics.json` on the verified 750-scenario 50.0 Hz run (`T_obs = 1.0 s` unless noted):

### 14.1 Continuous Nadir Regression (Held-Out ID `test`, `N = 52`, `T_obs = 1.0 s`)

| Model | Val MAE (Hz) | Test MAE (Hz) | Test RMSE (Hz) | P95 \|e\| (Hz) | `R^2` | 95% Bootstrap CI (Hz) | Inference (ms) |
|---|---:|---:|---:|---:|---:|---|---:|
| Constant Mean Baseline | 0.2319 | 0.2104 | 0.2526 | 0.4053 | -0.0117 | `[0.1759, 0.2517]` | 0.0008 |
| Analytical Swing Baseline | 0.0035 | 0.0042 | 0.0056 | 0.0125 | 0.9995 | `[0.0032, 0.0052]` | 0.0127 |
| **Ridge Linear Regression** | **0.0034** | **0.0034** | **0.0046** | **0.0095** | **0.9997** | `[0.0026, 0.0043]` | 0.0111 |
| Decision Tree (CART) | 0.0151 | 0.0111 | 0.0148 | 0.0270 | 0.9965 | `[0.0087, 0.0137]` | 0.0086 |
| Random Forest (100 Trees) | 0.0112 | 0.0090 | 0.0140 | 0.0274 | 0.9969 | `[0.0064, 0.0122]` | 0.2905 |
| **Gradient Boosting (GBDT)** | 0.0106 | **0.0077** | **0.0113** | **0.0179** | **0.9980** | `[0.0058, 0.0101]` | 0.0182 |
| LightGBM (`sklearn_hist_gbdt`) | 0.0148 | 0.0086 | 0.0116 | 0.0232 | 0.9979 | `[0.0067, 0.0107]` | 0.0423 |

*(Note: Because classical 3rd-order swing + linear droop trajectories vary smoothly across the bounded ID domain, global L2 linear and analytical quadratic models avoid tree-leaf quantization steps [`3.4–4.2 mHz` vs. `7.7 mHz` GBDT MAE], while `CalibratedGBDTClassifier` achieves the lowest validation Brier score [`0.0056`] and provides non-parametric quantile/OOD uncertainty intervals.)*

### 14.2 Security Classification, Calibration & Safety Sweep (Held-Out ID `test`, `N = 52`)

| Classifier Model | Val Brier | Test F1 | PR-AUC | ROC-AUC | Test Brier | Test ECE | `FNR_insecure` (`tau=0.50`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0123 | **1.0000** | 1.0000 | 1.0000 | 0.0132 | 0.0501 | **0.0000** (`0/31`) |
| Decision Tree | 0.0143 | 0.9767 | 0.9773 | 0.9839 | 0.0192 | 0.0192 | 0.0323 (`1/31`) |
| Random Forest | 0.0091 | 0.9767 | 1.0000 | 1.0000 | 0.0093 | 0.0223 | 0.0323 (`1/31`) |
| Uncalibrated GBDT | 0.0143 | 0.9767 | 0.9773 | 0.9839 | 0.0192 | 0.0193 | 0.0323 (`1/31`) |
| LightGBM Classifier | 0.0143 | 0.9767 | 0.9989 | 0.9992 | 0.0181 | **0.0179** | 0.0323 (`1/31`) |
| **Proposed Calibrated GBDT (Platt)** | **0.0056** | 0.9767 | **1.0000** | **1.0000** | 0.0141 | 0.0267 | 0.0323 (`0.0000` at `tau=0.70`) |
| **Proposed Calibrated GBDT (Isotonic)** | 0.0063 | 0.9767 | **1.0000** | **1.0000** | **0.0080** | 0.0192 | 0.0323 (`0.0000` at `tau=0.70`) |

- **Safety Threshold Sweep**: Raising the decision threshold from `tau = 0.50` to conservative **`tau = 0.70`** eliminates all missed insecure emergencies (`FNR_insecure = 0.0000`, `0/31` missed, `TN=31, TP=20, FP=0, FN=1`) with `FPR_secure = 0.0476` and `F1 = 0.9756`.

### 14.3 Observation Window Sensitivity, OOD Generalization, Noise, Observability & Latency
- **Window Trade-off (`T_obs in {0.5, 1.0, 1.5, 2.0} s`, strictly pre-nadir `t_0 + T_obs < t_nadir`)**:
  - Held-Out ID `test` (`N=52`, 100% causal across all 4 windows): Proposed `GBDT` MAE is `0.0124 Hz` (`0.5 s`, mean `t_lead = 1.7942 s`) -> **`0.0077 Hz`** (`1.0 s`, `t_lead = 1.2942 s`) -> **`0.0069 Hz`** (`1.5 s`, `t_lead = 0.7942 s`, `F1 = 1.0000`) -> `0.0074 Hz` (`2.0 s`, `t_lead = 0.2942 s`); Validation-Selected `Ridge` MAE improves monotonically `0.0063 -> 0.0034 -> 0.0030 -> 0.0026 Hz`.
  - Held-Out `ood_test` (`N=480` at `0.5–1.5 s`; **`N=472` at `2.0 s` after excluding `8` non-causal heavy-load cases with `t_nadir <= 3.00 s`**, `t_lead_min = +0.0200 s`): `GBDT` MAE is `0.0279 Hz` (`0.5 s`) -> **`0.0184 Hz`** (`1.0 s`, `F1 = 0.9860`, `ECE = 0.0224`) -> `0.0218 Hz` (`1.5 s`, `F1 = 0.9943`, `FNR_insecure = 0.0000`) -> `0.0229 Hz` (`2.0 s`, `N=472`, `F1 = 0.9861`). On the `472`-scenario common causal intersection across all 4 windows (`window_sensitivity_ood_common.csv`), MAE is `0.0272 -> 0.0184 -> 0.0218 -> 0.0229 Hz`.
- **OOD Domain Shift (`T_obs = 1.0 s`, Validation-Selected `Ridge` vs. Proposed `GBDT`)**:
  - `OOD-Load` (`0.80x, 1.20x`, `N=180`): `Ridge MAE = 0.0046 Hz`, `GBDT MAE = 0.0078 Hz`, `Calibrated GBDT F1 = 0.9792`, `ECE = 0.0263`.
  - `OOD-RES` (`40–50%` RES, `N=180`): `Ridge MAE = 0.0054 Hz` vs. `GBDT MAE = 0.0267 Hz` (tree leaf support bounded at `48.72 Hz`), while `Calibrated GBDT F1 = 1.0000`, `FNR_insecure = 0.0000`, `ECE = 0.0216`, and 95% prediction interval widens from `0.6616 Hz` (ID) to **`0.7428 Hz`** (`+12.3%`).
  - `OOD-Combined` (`N=120`): `Ridge MAE = 0.0069 Hz`, `GBDT MAE = 0.0217 Hz`, `F1 = 0.9851`. Pooled `OOD-All` (`N=480`): `Ridge MAE = 0.0054 Hz`, `GBDT MAE = 0.0184 Hz`, `F1 = 0.9860`, `Brier = 0.0086`, `ECE = 0.0224`.
- **PMU Gaussian Noise (`sigma_f in {0, 5, 10, 20, 50} mHz` — Clean vs. Noisy Trade-off)**:
  - **Proposed `GBDT`**: Held-out ID `MAE` rises gently `0.0077 -> 0.0082 -> 0.0085 -> 0.0101 -> 0.0114 Hz` while `F1 = 0.9767` remains invariant across all noise levels (`prob_stability_mad <= 0.0131`).
  - **Validation-Selected `Ridge`**: Excels at `0–5 mHz` (`0.0034 -> 0.0050 Hz`), matches GBDT at `10 mHz` (`0.0087 Hz`), and degrades to `0.0183 Hz` (`20 mHz`) and `0.0571 Hz` (`50 mHz`) due to linear coefficient amplification of derivative noise—demonstrating why tree ensembles provide `5x` stronger high-noise resilience alongside calibrated intervals.
- **Physical Security-Threshold Sensitivity (`f_thresh in {49.20, 49.40, 49.50, 49.60, 49.70, 49.80} Hz`)**: Across `49.20–49.70 Hz`, held-out ID `F1` remains in `[0.9286, 1.0000]` and OOD `F1` remains in `[0.9860, 1.0000]` (`F1 = 1.0000` at the `49.40 Hz` UFLS Stage-1 threshold); only at `49.80 Hz` (`5.8%` secure minority) does `F1` drop due to extreme positive-class scarcity (`frequency_threshold_sensitivity.csv`).
- **Partial PMU, Single-Family Drops & 25-fps Ablations (`train`-only `N=148` baseline `ID MAE = 0.0093 Hz`)**:
  - **Single-Family Drops**: Dropping aggregate frequency level/integral (`no_freq_level`) increases `ID MAE` to `0.0144 Hz` (`OOD MAE = 0.0304 Hz`), confirming `freq_dev_integral` as the primary predictor; `no_curvature` gives `0.0100 Hz` and `no_rocof` gives `0.0084 Hz`.
  - **13/39 Sparse PMU (`33.3%` bus coverage)**: `ID MAE = 0.0108 Hz`, `ID F1 = 1.0000` (`FNR_insecure = 0.0000`), `OOD MAE = 0.0238 Hz`, `OOD F1 = 0.9858`.
  - **25 fps Reporting Rate**: `ID MAE = 0.0094 Hz` (`+0.0001 Hz` vs. `50 fps`), `ID F1 = 0.9767`, `OOD MAE = 0.0239 Hz`.
- **6-Stage Desktop Compute Latency (`Intel Core i7-9750H`, 100 reps, user-space Python)**: **`2.5306 ms` median total** (`3.2345 ms` P95, `3.9396 ms` P99; `2.2810 ms` median for `Ridge` + `Calibrated GBDT`) vs. `20.0 ms` 50-fps PMU frame interval and **`161.17 ms`** measured `10 s` ODE simulation (**`63.7x` desktop speedup**; desktop benchmark, not hard real-time RTOS/HIL certification).

---

## 15. Completed Functionality, Limitations & Future Work

### Completed & Verified Functionality
- Full 50.0 Hz Newton-Raphson power-flow-initialized IEEE 39-bus simulator (`H_39 = 42.0 s`), complete/partial generator disconnection physics, linear 9-bus RES displacement, causal 5-point OLS PMU recorder, 14-feature causal extractor, 7 regressors, 7 classifiers, cross-fit Platt & PAVA Isotonic calibration with quantile/conformal intervals, 26 CSV/Markdown benchmark tables, 13 publication PNG/PDF figures, and **232 automated production tests passing (`83` unit/integration + `149` E2E; `0` shadow mocks)**.

### Explicit Technical Limitations
1. **No Operational Relay Certification**: This codebase is a research screening prototype and is **not** certified for direct protective relay tripping without utility EMS/SCADA and HIL qualification.
2. **Classical 3rd-Order ODE & Constant-Admittance Load Fidelity**: Synchronous machines use classical constant-flux-linkage (`E'`) 3rd-order swing + `TGOV1` dynamics without subtransient/AVR/PSS flux decay, boiler pressure limits, or ZIP/induction-motor dynamic load recovery; converters are modelled as constant-PQ GFL injections (`H_vir = 0`).
3. **Simplified Synchrophasor Front-End**: `PMURecorder` applies linear interpolation + additive Gaussian channel noise + causal 5-point OLS differentiation rather than full sub-cycle DFT/phasor-estimation filter banks, instrument-transformer saturation, or GPS time-skew transients.
4. **Tree Extrapolation Ceiling Under Extreme Unseen Low Inertia**: Tree ensembles bound predictions within training leaf support (`>= 48.72 Hz`), increasing tail error (`P95 |e| = 0.1271 Hz`) on unseen `50%` RES contingencies that plunge to `48.26 Hz` (even though `49.50 Hz` classification remains `F1 = 1.0000`).

### Future Work
- Physics-residual hybrid learning (`AnalyticalSwingNadirRegressor` + GBDT residual boosting) to combine exact low-inertia tail extrapolation with nonlinear spatial tree corrections.
- Grid-forming (`GFM`) virtual-inertia converter control (`H_vir > 0`, fast droop saturation), `N-2` cascading lines + generation outages, and real-time digital simulator (`RTDS` / `OPAL-RT`) HIL testing under C37.118.2 UDP streams.

---

## 16. Exact Reproduction Instructions

### Environment Setup (Python 3.9+)
```bash
cd btp-early-pmu-fsa
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-build-isolation --no-deps
```

### Option A: Single Master Reproducible Command (`scripts/run_all.py`)
```bash
# Full 750-scenario simulation, feature extraction, validation, training, benchmarks & 14 figures:
.venv/bin/python3 scripts/run_all.py --config configs/default_50hz.yaml --with-figures

# Fast reproduction reusing existing raw .npz archives in data/raw/:
.venv/bin/python3 scripts/run_all.py --config configs/default_50hz.yaml --skip-simulation --with-figures
```

### Option B: Stage-by-Stage CLI Execution
```bash
# Stage 1: Simulate 750 contingencies (writes data/raw/*.npz, data/raw_sim/*.npz, data/manifest.json)
.venv/bin/python3 scripts/run_simulations.py --clean --n-scenarios 750 \
  --gen-buses 30 31 32 33 34 35 36 37 38 39 \
  --loss-fractions 1.0 0.70 0.40 \
  --penetrations 0.10 0.20 0.30 0.40 0.50 \
  --load-scales 0.80 0.90 1.00 1.10 1.20 \
  --fps 50 --f-nominal 50.0 --threshold 49.50 \
  --output-dir data/raw --raw-sim-dir data/raw_sim --validate

# Stage 2: Extract strictly causal 14-feature matrices across {0.5, 1.0, 1.5, 2.0} s
.venv/bin/python3 scripts/extract_features.py \
  --input-dir data/raw --t-obs 1.0 --f-nominal 50.0 \
  --all-windows --strict-nadir --save-csv \
  --output-file data/processed/features.npz

# Stage 3: Fit all 7 regressors & 7 classifiers on disjoint splits
.venv/bin/python3 scripts/train_models.py \
  --features data/processed/features.npz --model-type all \
  --calibration platt --threshold 49.50 \
  --output-dir models/ --tables-dir results/tables/

# Stage 4: Run window, OOD, noise, latency, statistical & ablation benchmarks
.venv/bin/python3 scripts/evaluate_all.py \
  --data-dir data/raw --models-dir models/ \
  --features data/processed/features.npz \
  --tables-dir results/tables/ --report-file results/evaluation_report.md

# Stage 5: Render all 14 publication figures (PNG @ 240 DPI + vector PDF)
.venv/bin/python3 scripts/generate_figures.py

# Stage 6: Run complete isolated production test suite (pytest + 5-tier E2E)
.venv/bin/pytest tests/ -q
.venv/bin/python3 tests/e2e/run_e2e.py --tier all --report-path reports/final_audit/reproduction_e2e_report.md
```

### Data Manifest & Git Tracking Policy
Bulk trajectory archives (`data/raw/*.npz` and `data/raw_sim/*.npz`) are excluded via `.gitignore` to keep the repository lightweight, while `data/manifest.json` (recording per-scenario SHA-256 digests and aggregate dataset hash `b543ffc51c757ddc36437391eced8df375de0ac793141e80a6e3b9be2aeaab11`), `configs/default_50hz.yaml`, `results/dataset_quality_report.md`, `results/tables/`, `results/figures/`, and `docs/` are tracked in version control for deterministic verification.

---

## 17. Project Structure

```text
btp-early-pmu-fsa/
├── configs/
│   └── default_50hz.yaml               # Canonical 750-scenario 50.0 Hz experiment configuration
├── src/pmu_fsa/
│   ├── simulation/
│   │   ├── ieee39_data.py              # 39-bus 50.0 Hz network, S_mva ratings, H_39 = 42.0 s default
│   │   ├── renewables.py               # 9-bus synchronous displacement & GFL/GFM inertia scaling
│   │   ├── dynamic_engine.py           # NR AC power flow, two-segment Kron ODEs, COI nadir labels
│   │   ├── pmu_recorder.py             # 50/25 fps resampling, area-weighted f_sys_pmu, causal 5-pt OLS ROCOF
│   │   ├── scenario_generator.py       # 750-scenario grid, MD5 partitioner, raw_sim vs raw separation
│   │   └── dataset_validator.py        # 14 automated dataset/causality checks & SHA-256 manifest
│   ├── features/
│   │   ├── window_slicer.py            # Pre-nadir causal windowing & forbidden-channel firewall
│   │   └── extractor.py                # 14 canonical spatiotemporal PMU features & NaN masking
│   ├── models/
│   │   ├── regressors.py               # Mean, Analytical Swing, Ridge, DT, RF, GBDT, LightGBM
│   │   ├── classifiers.py              # Logistic, DT, RF, GBDT, LightGBM, FNR_insecure & threshold sweep
│   │   ├── calibrated_gb.py            # Two-stage GBDT + Platt / PAVA Isotonic + quantile/conformal PIs
│   │   └── calibration_metrics.py      # 10-bin ECE, MCE, Brier score & reliability binning
│   ├── evaluation/
│   │   ├── sensitivity.py              # Unsmoothed T_obs in {0.5..2.0} s sweep on test & ood_test
│   │   ├── ood_generalization.py       # ID vs OOD-RES, OOD-Load & OOD-Combined evaluation
│   │   ├── noise_stress.py             # Seeded multi-channel PMU noise sweep (0-50 mHz)
│   │   ├── latency.py                  # 6-stage perf_counter_ns profiler vs measured 10s ODE sim
│   │   ├── statistical_tests.py        # Paired bootstrap CIs, Wilcoxon tests, learning & seed curves
│   │   └── interpretability.py         # Permutation importance, correlations, 13-bus & 25-fps ablations
│   └── pipeline/
│       └── run_experiments.py          # Programmatic ExperimentConfig & ExperimentPipeline API
├── scripts/
│   ├── run_all.py                      # Single master reproducible pipeline runner
│   ├── run_simulations.py              # CLI Stage 1: Trajectory generation & validation
│   ├── extract_features.py             # CLI Stage 2: Multi-window causal feature extraction
│   ├── train_models.py                 # CLI Stage 3: Disjoint model fitting & calibration
│   ├── evaluate_all.py                 # CLI Stage 4: Full benchmark, statistical & ablation suite
│   ├── generate_figures.py             # CLI Stage 5: 13 publication PNG + PDF figure generator
│   └── generate_project_guide.py       # HTML-to-PDF builder for BTP_Complete_Project_Guide.pdf
├── data/
│   ├── manifest.json                   # SHA-256 checksums & dataset provenance manifest
│   ├── raw/                            # 750 observable PMU .npz streams (git-ignored)
│   ├── raw_sim/                        # 750 internal ODE simulator .npz archives (git-ignored)
│   └── processed/                      # Causal 14-feature .npz/.csv matrices & extraction_report.json
├── models/                             # Serialized .joblib estimators & training_metrics.json
├── results/
│   ├── dataset_quality_report.md       # 14-check validation & pre-nadir causality audit
│   ├── evaluation_report.md            # Consolidated benchmark & statistical evaluation report
│   ├── tables/                         # 26 CSV & Markdown experimental result tables
│   └── figures/                        # fig01..fig13 publication figures (.png @ 240 DPI & .pdf)
├── docs/
│   ├── IEEE_BTP_Research_Paper.md      # 24-section IEEE-styled research manuscript (Markdown)
│   ├── IEEE_BTP_Research_Paper.pdf     # Rendered publication-ready IEEE research paper (PDF)
│   ├── BTP_Final_Presentation.html     # 16-slide 16:9 widescreen presentation source
│   └── BTP_Final_Presentation.pdf      # Rendered 16-slide widescreen presentation deck (PDF)
├── BTP_Complete_Project_Guide.pdf      # Zero-to-Hero 50-Hz visual project manual & Viva Q&A
└── tests/                              # 232 isolated 50-Hz production tests (83 unit + 149 E2E)
```
