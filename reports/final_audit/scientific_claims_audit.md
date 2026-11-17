# Phase 2 — Locked Scientific Definitions, Hypotheses, and Verified Literature Audit

**Status:** Locked prior to Phase 3–16 experimental execution  
**Scope:** Simulation-based early PMU-type frequency nadir regression and binary frequency-security assessment following under-frequency generation-loss contingencies on a 50 Hz IEEE 39-bus, 10-machine benchmark system.

---

## 2.1 Physical System Specification

| Parameter | Locked Value | Implementation Source | Engineering Justification |
| :--- | :--- | :--- | :--- |
| **Nominal Frequency ($f_0$)** | `50.0 Hz` ($\omega_s = 100\pi\text{ rad/s}$) | `ieee39_data.py` (`NOMINAL_FREQ_HZ = 50.0`) | Matches Indian/European 50 Hz synchronous grid standard (CERC IEGC 2023). |
| **Benchmark Network** | IEEE 39-Bus, 10-Machine (New England) | `ieee39_data.py` (`build_ieee39_net`) | Canonical multi-machine electromechanical benchmark (Athay et al., 1979): 39 buses, 10 synchronous generators (Buses 30–39), 19 load buses, 34 AC lines, 12 transformers, `S_base = 100 MVA`. |
| **Base Generation & Load** | `6,140.81 MW` gen / `6,097.10 MW` load (`43.71 MW` AC losses at `load_scale = 1.0`) | `solve_power_flow()` | Full Newton-Raphson AC power flow solved (`tol = 1e-8 MVA`) at each scenario's `load_scale` before initializing internal EMFs $E_i \angle \delta_{i0}$. |
| **Slack / Reference Bus** | Bus 31 (`Gen 2`, $V = 0.982\text{ pu}$, $\theta = 0^\circ$) | `GEN_DATA` (`is_slack=True`) | Absorbs initial pre-disturbance AC network loss variation across `load_scale` (`0.85–1.20`). |
| **Bus 39 Equivalent (`Gen 10`)** | Regional thermal plant $H_{39} = 42.0\text{ s}$ on `100 MVA` system base (`macro_intertie: false`, $H_{\mathrm{tot}} = 324.7\text{ s}$; optional `--macro-intertie` $H_{39} = 500.0\text{ s}$) | `ieee39_data.py` & `configs/default_50hz.yaml` | Produces realistic regional under-frequency excursions (`48.26–49.91 Hz`) across single-unit outages without stiff external infinite-bus masking. |
| **Machine & Governor Model** | Classical constant-$E'$ 3rd-order synchronous model ($X'_d$) + Kron-reduced $Y_{\mathrm{red}}$ + first-order steam-reheat governor-turbine (`R = 5%` on machine rating $S_{\mathrm{mva},i}$, $T_g = 4.20\text{ s}$, $D_i = 2.0\text{ pu}$) | `dynamic_engine.py` | Captures primary inertial, synchronizing, and governor arrest dynamics (`t_nadir` in `2.78–3.70 s`) over `[0, 10] s`; converter fast droop (`enable_droop=False`) is disabled under default GFL operation. |
| **Load Model** | Constant-power in pre-contingency AC power flow; constant-impedance shunt admittance in post-contingency Kron reduction | `ieee39_data.py` & `dynamic_engine.py` | Standard direct-method/classical transient stability network reduction; voltage-dependent ZIP/induction-motor load dynamics are noted as an explicit limitation. |
| **Renewable Representation (Approach B)** | Co-located constant-power GFL effective inertia, damping, and governor participation derating: $H_{\mathrm{eff},i} = (1-\alpha)H_i$, $D_{\mathrm{eff},i} = (1-\alpha)D_i$, $R_{\mathrm{eff},i} = R_i/(1-\alpha)$ | `renewables.py` | Models homogeneous displacement of synchronous rotational inertia and damping at constant terminal active-power dispatch without inventing arbitrary solar/wind bus locations; network power-flow redistribution, PLL dynamics, and current limiting are explicitly out of scope. |
| **Disturbance Model** | Single-unit partial or complete generation loss at $t_0 = 1.0\text{ s}$ (`loss_fraction` $\in [0.25, 1.00]$) | `dynamic_engine.py` & `scenario_generator.py` | Complete trip (`1.0`) sets $E'_k = 0$ (removing electrical injection from $Y_{\mathrm{red}}$ and $V_{\mathrm{bus}}$) and excludes unit $k$ from surviving COI; partial trip steps down $P_{m,k}$ while remaining synchronized. |
| **PMU Channel Model** | `50 fps` ($\Delta t = 20\text{ ms}$) across all 39 buses; complex phasor Gaussian angle/magnitude noise; causal `lfilter` 2nd-order Butterworth ($f_c = 10\text{ Hz}$); causal backward-difference $f_b$ and ROCOF | `pmu_recorder.py` | Conforms to 50-fps M-class reporting rate in IEEE Std C37.118.1-2011 (`TVE = 1%` stress point corresponds to $\sigma_\theta \approx 0.00577\text{ rad}$, $\sigma_V \approx 0.00577\text{ pu}$). |

---

## 2.2 Target Variable & Frequency-Security Threshold Definition

1. **Continuous Ground-Truth Target ($f_{\mathrm{nadir}}$)**:
   $$f_{\mathrm{COI}}(t) = \frac{\sum_{i \in \mathcal{G}_{\mathrm{on}}(t)} H_{\mathrm{eff},i}\, f_i(t)}{\sum_{i \in \mathcal{G}_{\mathrm{on}}(t)} H_{\mathrm{eff},i}}, \qquad f_{\mathrm{nadir}} = \min_{t \in [t_0,\, t_{\max}]} f_{\mathrm{COI}}(t), \qquad t_{\mathrm{nadir}} = \arg\min_{t \in [t_0,\, t_{\max}]} f_{\mathrm{COI}}(t)$$
   where $\mathcal{G}_{\mathrm{on}}(t)$ is the set of electrically connected synchronous machines at time $t$ (excluding any completely tripped unit for $t \ge t_0$), and $t_0 = 1.0\text{ s}$, $t_{\max} = 10.0\text{ s}$.
   - **Important distinction**: $f_{\mathrm{COI}}(t)$ uses simulator inertia weights $H_{\mathrm{eff},i}$ and future samples $t \le 10.0\text{ s}$ **exclusively to define the ground-truth label**. It never enters the ML feature vector.

2. **Binary Security Label ($y_{\mathrm{secure}}$) and Threshold Justification (`49.50 Hz`)**:
   $$y_{\mathrm{secure}} = \mathbb{I}\!\left(f_{\mathrm{nadir}} \ge f_{\mathrm{thresh}}\right), \qquad f_{\mathrm{thresh}} = 49.50\text{ Hz}$$
   - Under the **CERC Indian Electricity Grid Code (IEGC) Regulations, 2023**, the normal steady-state operating band is `49.90–50.05 Hz`, whereas during a reference generation-loss contingency transient frequency is expected not to fall below **`49.50 Hz`**, providing a `100 mHz` safety margin above Stage-1 automatic Under-Frequency Load Shedding (UFLS) initiation at **`49.40 Hz`** (`49.40, 49.20, 49.00, 48.80 Hz`).
   - Using `49.90 Hz` or `49.95 Hz` as a post-contingency nadir threshold would classify routine credible generator outages as insecure; `49.50 Hz` is the physically grounded contingency-nadir security boundary.
   - Because any single threshold induces a specific class balance across scenarios, we additionally run a **threshold sensitivity sweep** across $f_{\mathrm{thresh}} \in \{49.20, 49.40, 49.50, 49.60, 49.70, 49.80\}\text{ Hz}$.

3. **Strict Pre-Nadir Causal Eligibility Rule**:
   $$t_0 + T_{\mathrm{obs}} < t_{\mathrm{nadir}} \iff t_{\mathrm{lead}} = t_{\mathrm{nadir}} - (t_0 + T_{\mathrm{obs}}) > 0$$
   Every primary regression, classification, calibration, sensitivity, OOD, noise, and ablation metric is restricted to scenarios satisfying $t_{\mathrm{lead}} > 0$, with excluded scenario counts and IDs reported explicitly.

---

## 2.3 Falsifiable Research Hypotheses

- **H1 (Early Causal Predictability)**: A `1.0 s` post-disturbance window of PMU-only features ($t_{\mathrm{lead}} > 0$ for 100% of scenarios) predicts $f_{\mathrm{nadir}}$ on held-out ID scenarios with lower MAE and RMSE than an analytical single-equivalent-swing baseline and constant mean predictor (`null`: paired loss difference $\Delta \mathrm{MAE} \ge 0$, tested via paired sign-flip permutation test with Holm adjustment at $\alpha = 0.05$).
- **H2 (Linear vs Nonlinear Expressiveness under Smooth Inertial Swing Dynamics)**: Because early COI frequency drop and average ROCOF scale nearly linearly with $\Delta P / (2 H_{\mathrm{sys}})$ in classical inertial-response dynamics, regularized linear regression (`Ridge`) is expected to be competitive with or superior to tree ensembles (`GBDT`) on smooth noiseless features, whereas nonlinear models must justify their complexity empirically on validation and OOD/noise benchmarks rather than being assumed superior a priori.
- **H3 (Out-of-Distribution Generalization)**: Models trained solely on ID operating conditions (`ren 10–30%`, `Buses {30,33,34,37,38}`, `load_scale 0.85–1.10`) maintain bounded prediction error and high security recall across unseen tripped buses (`{32,35,36}`), higher renewable derating (`40–50%`), heavier loading (`1.15–1.20`), and combined OOD shifts, with predictable error growth under extreme low-inertia `40–50%` derating.
- **H4 (Window-Length vs Warning Lead-Time Trade-Off)**: Extending $T_{\mathrm{obs}}$ from `0.5 s` to `2.0 s` monotonically reduces causal lead time $t_{\mathrm{lead}}$ by `0.5 s` per step while improving nadir accuracy once curvature/deceleration information becomes observable—evaluated fairly on both window-specific causal scenarios and the `472`-scenario common causal intersection.
- **H5 (PMU Phasor Noise Sensitivity)**: Injecting realistic phasor angle and magnitude noise (`SNR = 45, 40, 34, 26 dB` and `IEEE C37.118 TVE = 1%`) degrades high-order derivative features (`ROCOF` and second-difference curvature) substantially more than integral/level features, quantifying the gap between clean-trained models and noisy PMU streams.
- **H6 (Post-Hoc Probability Calibration)**: Fitting Platt scaling or isotonic regression on a disjoint validation split preserves or improves probability calibration (Brier score / ECE) relative to uncalibrated ensemble scores without touching test scenarios, subject to finite-sample calibration variance at $N_{\mathrm{val}} = 70$, $N_{\mathrm{test}} = 52$.
- **H7 (Sub-Frame Computational Inference Latency)**: Feature extraction plus saved-model inference on a single CPU core executes in well under one `20 ms` 50-fps PMU frame interval, ensuring negligible algorithmic overhead on top of the `1.0 s` causal observation window (without claiming hard real-time hardware-in-the-loop protection deployment).

---

## 2.4 Scope Boundaries & Explicit Non-Claims

1. **Simulation-only PMU-type study**: No utility field PMU recordings, physical PMU hardware, PDC communication jitter, or RTDS/OPAL-RT hardware-in-the-loop (HIL) tests are claimed.
2. **Co-located effective-response renewable approximation (Approach B)**: Does not claim explicit solar/wind park placement, line congestion shifts, converter current saturation, fault ride-through blocking, or PLL loss of synchronism.
3. **Under-frequency generation loss only**: Over-frequency load-rejection events, transmission short-circuit clearing transients, automatic UFLS action, and AGC secondary control are outside scope.
4. **Desktop benchmark latency**: Measured with `time.perf_counter()` on macOS user-space Python; reported as algorithmic compute latency, never as deterministic RTOS/relay hard real-time certification.

---

## 2.5 Verified Literature & Standards Matrix

Every citation below has been bibliographically verified against primary IEEE / ACM / CERC sources:

| Ref ID | Verified Citation | Verified DOI / Official Identifier | Claim Supported in This Project |
| :--- | :--- | :--- | :--- |
| **[R1]** | IEEE Power and Energy Society, *IEEE Standard for Synchrophasor Measurements for Power Systems*, IEEE Std C37.118.1-2011, Dec. 2011 (and Amendment 1, IEEE Std C37.118.1a-2014). | `10.1109/IEEESTD.2011.6111219` / `10.1109/IEEESTD.2014.6830177` | Defines 50-fps synchrophasor reporting rate ($\Delta t = 20\text{ ms}$), Total Vector Error (`TVE = 1%` steady-state limit), frequency, and ROCOF measurement concepts. |
| **[R2]** | Central Electricity Regulatory Commission (CERC), *Indian Electricity Grid Code (IEGC) Regulations, 2023*, New Delhi, India, May 2023. | Official Gazette / CERC Regulation No. `L-1/265/2022/CERC` | Establishes `50.00 Hz` nominal frequency, `49.90–50.05 Hz` normal operating band, `>= 49.50 Hz` contingency nadir expectation, and `49.40 Hz` Stage-1 UFLS initiation. |
| **[R3]** | T. Athay, R. Podmore, and S. Virmani, "A Practical Method for the Direct Analysis of Transient Stability," *IEEE Trans. Power App. Syst.*, vol. PAS-98, no. 2, pp. 573–584, Mar./Apr. 1979. | `10.1109/TPAS.1979.319407` | Canonical source for the 10-machine, 39-bus New England test system topology, generator inertias (including Bus 39 equivalent), and classical Kron-reduced dynamic formulation. |
| **[R4]** | F. Milano and Á. Ortega, "Frequency Divider," *IEEE Trans. Power Syst.*, vol. 32, no. 2, pp. 1493–1501, Mar. 2017. | `10.1109/TPWRS.2016.2569563` | Establishes physical relationship between internal synchronous rotor speeds and network bus voltage phase-angle frequencies across transmission admittances. |
| **[R5]** | L. Liu, W. Li, Y. Ba, J. Shen, C. Jin, and K. Wen, "An Analytical Model for Frequency Nadir Prediction Following a Major Disturbance," *IEEE Trans. Power Syst.*, vol. 35, no. 4, pp. 2527–2536, Jul. 2020. | `10.1109/TPWRS.2019.2963706` | Supports analytical parabolic/swing nadir baseline structure and explains approximation error when multi-machine spatial modes and non-uniform damping are lumped. |
| **[R6]** | Dhanuja Lekshmi J., Z. H. Rather, and B. C. Pal, "Online Estimation of Disturbance Size and Frequency Nadir Prediction in Renewable Energy Integrated Power Systems," *IEEE Trans. Power Syst.*, vol. 39, no. 1, pp. 1126–1137, Jan. 2024. | `10.1109/TPWRS.2023.3267744` | Demonstrates post-disturbance PMU-based disturbance/nadir estimation in renewable-integrated 50 Hz grids and inertia-reduction impact on ROCOF and nadir. |
| **[R7]** | A. Niculescu-Mizil and R. Caruana, "Predicting Good Probabilities With Supervised Learning," *Proc. 22nd Int. Conf. Mach. Learn. (ICML)*, Bonn, Germany, 2005, pp. 625–632. | `10.1145/1102351.1102430` | Establishes post-hoc Platt sigmoid scaling vs. isotonic regression trade-offs for boosted trees and explains isotonic overfitting when calibration sample size is small. |
| **[R8]** | S. Holm, "A Simple Sequentially Rejective Multiple Test Procedure," *Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65–70, 1979. | JSTOR `4615733` | Justifies the step-down Holm family-wise error rate (FWER) adjustment applied across multiple paired model comparisons in Phase 11. |

---

## Checkpoint 2 Status

**CHECKPOINT 2: PASS** — System parameters, COI nadir target, `49.50 Hz` contingency security boundary (with threshold sensitivity sweep), strict pre-nadir rule, falsifiable hypotheses `H1`–`H7`, Approach B scope limits, and verified literature citations are locked consistently.
