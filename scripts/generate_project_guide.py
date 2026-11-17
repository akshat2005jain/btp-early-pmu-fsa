#!/usr/bin/env python3
"""Generate BTP_Complete_Project_Guide.pdf from verified 750-scenario 50.0 Hz results."""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "results" / "figures"
HTML_OUT = Path("/tmp/btp_complete_project_guide.html")
PDF_OUT = PROJECT_ROOT / "BTP_Complete_Project_Guide.pdf"
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _img_uri(filename: str) -> str:
    path = FIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing figure: {path}")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_html() -> str:
    fig01 = _img_uri("fig01_system_and_architecture.png")
    fig02 = _img_uri("fig02_generation_loss_response.png")
    fig03 = _img_uri("fig03_causal_window_illustration.png")
    fig06 = _img_uri("fig06_regression_model_comparison.png")
    fig07 = _img_uri("fig07_classification_and_safety.png")
    fig08 = _img_uri("fig08_calibration_reliability.png")
    fig09 = _img_uri("fig09_window_sensitivity.png")
    fig10 = _img_uri("fig10_ood_generalization.png")
    fig12 = _img_uri("fig12_feature_importance_and_ablation.png")
    fig13 = _img_uri("fig13_latency_and_learning_curve.png")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BTP Complete Zero-to-Hero Project Guide (50.0 Hz Standard)</title>
<style>
  @page {{
    size: A4;
    margin: 13mm 14mm 14mm 14mm;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 9.6pt;
    line-height: 1.46;
    color: #1e293b;
    margin: 0;
    padding: 0;
    background: #ffffff;
  }}
  h1, h2, h3 {{ color: #0f172a; margin-top: 0; }}
  h2 {{
    font-size: 13pt;
    border-bottom: 2px solid #cbd5e1;
    padding-bottom: 4px;
    margin-top: 16px;
    margin-bottom: 8px;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 10.6pt;
    color: #1d4ed8;
    margin-top: 11px;
    margin-bottom: 5px;
    page-break-after: avoid;
  }}
  p {{ margin: 0 0 7px 0; }}
  ul, ol {{ margin: 4px 0 8px 18px; padding: 0; }}
  li {{ margin-bottom: 4px; }}
  code {{
    font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    font-size: 8.5pt;
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    color: #0f172a;
  }}
  pre {{
    background: #0f172a;
    color: #e2e8f0;
    padding: 10px 12px;
    border-radius: 7px;
    font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    font-size: 7.9pt;
    line-height: 1.38;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
    margin: 7px 0 10px 0;
    break-inside: avoid;
  }}
  .hero {{
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #1d4ed8 100%);
    color: #ffffff;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
    break-inside: avoid;
  }}
  .badge {{
    display: inline-block;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 7.8pt;
    font-weight: 600;
    letter-spacing: 0.4px;
    margin-bottom: 7px;
  }}
  .hero h1 {{
    color: #ffffff;
    font-size: 16.5pt;
    line-height: 1.24;
    margin: 0 0 6px 0;
  }}
  .hero p.sub {{
    color: #dbeafe;
    font-size: 9.6pt;
    margin: 0 0 10px 0;
  }}
  .meta-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 16px;
    background: rgba(15, 23, 42, 0.42);
    border-radius: 7px;
    padding: 9px 12px;
    font-size: 8.6pt;
  }}
  .meta-grid strong {{ color: #93c5fd; }}
  .callout {{
    background: #f0fdf4;
    border-left: 4px solid #16a34a;
    padding: 9px 12px;
    border-radius: 6px;
    margin: 9px 0;
    break-inside: avoid;
  }}
  .callout-blue {{
    background: #eff6ff;
    border-left: 4px solid #2563eb;
    padding: 9px 12px;
    border-radius: 6px;
    margin: 9px 0;
    break-inside: avoid;
  }}
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin: 10px 0 12px 0;
    break-inside: avoid;
  }}
  .kpi {{
    border: 1px solid #bfdbfe;
    background: #f8fafc;
    border-radius: 8px;
    padding: 8px 8px;
    text-align: center;
  }}
  .kpi .val {{ font-size: 13pt; font-weight: 700; color: #1d4ed8; }}
  .kpi .lbl {{ font-size: 7.8pt; color: #475569; margin-top: 2px; }}
  .eq-box {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px 12px;
    text-align: center;
    margin: 7px 0;
    font-size: 9.4pt;
    break-inside: avoid;
  }}
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 8px 0;
    break-inside: avoid;
  }}
  .card {{
    border: 1px solid #cbd5e1;
    border-radius: 7px;
    padding: 9px 11px;
    background: #ffffff;
    break-inside: avoid;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 7px 0 10px 0;
    font-size: 8.3pt;
    break-inside: avoid;
  }}
  th {{
    background: #0f172a;
    color: #ffffff;
    text-align: left;
    padding: 5px 6px;
    font-weight: 600;
  }}
  td {{
    border-bottom: 1px solid #e2e8f0;
    padding: 4.5px 6px;
    vertical-align: top;
  }}
  tr:nth-child(even) td {{ background: #f8fafc; }}
  .fig-box {{
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 8px;
    margin: 9px 0 11px 0;
    background: #ffffff;
    break-inside: avoid;
  }}
  .fig-box img {{
    width: 100%;
    height: auto;
    display: block;
    border-radius: 4px;
  }}
  .fig-cap {{
    font-size: 8.1pt;
    color: #475569;
    margin-top: 5px;
    line-height: 1.35;
  }}
  .qa {{
    border: 1px solid #cbd5e1;
    border-left: 4px solid #1d4ed8;
    border-radius: 6px;
    padding: 8px 11px;
    margin-bottom: 8px;
    background: #f8fafc;
    break-inside: avoid;
  }}
  .qa .q {{ font-weight: 700; color: #0f172a; margin-bottom: 3px; }}
  .page-break {{ page-break-before: always; }}
</style>
</head>
<body>

<div class="hero">
  <div class="badge">B.TECH PROJECT (BTP) — COMPLETE ZERO-TO-HERO VISUAL MANUAL (50.0 HZ STANDARD)</div>
  <h1>Early PMU-Based Prediction of Frequency Nadir &amp; Security Status in Renewable-Integrated Multi-Area Power Systems Under Limited Observation</h1>
  <p class="sub">Plain-English Power Physics, 50.0 Hz Formulation (H<sub>39</sub> = 42.0 s), Verified 750-Scenario Results, Codebase Walkthrough &amp; Viva Q&amp;A</p>
  <div class="meta-grid">
    <div><strong>Mentor:</strong> Dr. Deepak Kumar Gupta</div>
    <div><strong>Team Members:</strong> Aditya Jha, Akshat Jain, Arshdeep Singh, Raj Bathla</div>
    <div><strong>Grid Standard:</strong> 50.0 Hz Nominal | 49.50 Hz Floor | 50 &amp; 25 fps PMU</div>
    <div><strong>Production Dataset:</strong> 750 Disjoint Scenarios (270 ID + 480 OOD)</div>
    <div><strong>Dataset &amp; Test Verification:</strong> 14/14 Data Checks &amp; Full Pytest + E2E Suites PASS</div>
    <div><strong>Desktop Compute Speedup:</strong> ~2.53 ms Median vs. 161.17 ms ODE (Sub-20 ms 50-fps Frame)</div>
  </div>
</div>

<div class="callout">
  <strong>How to Read This Guide (Zero Prior Power-Systems or ML Calibration Background Needed):</strong>
  Every concept begins with an intuitive physical explanation, connects to exact 50.0 Hz synchronous-grid equations (matching India's Central Electricity Authority / IEGC 50 Hz standard), maps directly to the Python modules inside <code>btp-early-pmu-fsa/</code>, and cites exact numbers from the verified 750-scenario experimental run.
</div>

<div class="kpi-row">
  <div class="kpi">
    <div class="val">50.0 Hz</div>
    <div class="lbl">Nominal Grid Base<br>(49.50 Hz Security Floor)</div>
  </div>
  <div class="kpi">
    <div class="val">750 Runs</div>
    <div class="lbl">148 Train / 70 Val / 52 Test<br>+ 480 OOD Scenarios</div>
  </div>
  <div class="kpi">
    <div class="val">7.7 mHz</div>
    <div class="lbl">GBDT Test MAE (R&sup2;=0.9980)<br>(Ridge: 3.4 mHz)</div>
  </div>
  <div class="kpi">
    <div class="val">52.7&times;</div>
    <div class="lbl">Hardware Speedup<br>(2.38 ms vs. 125.38 ms ODE)</div>
  </div>
</div>

<h2>1. The Engineering Problem &amp; Research Gap</h2>

<h3>1.1 Intuitive Physics: Why Renewable-Rich 50.0 Hz Grids Are Vulnerable</h3>
<ul>
  <li><strong>Grid Frequency (50.0 Hz = 314.16 rad/s):</strong> In a 50.0 Hz AC grid, all synchronous turbine-generator rotors spin in electro-mechanical lockstep. When active generation equals load plus losses, frequency remains steady at <code>50.0000 Hz</code>.</li>
  <li><strong>Generator Outages &amp; Rotational Inertia (H):</strong> When a generator trips at <code>t<sub>0</sub> = 1.0 s</code> (losing <code>80–1,200 MW</code>), the power deficit is immediately supplied from the kinetic energy of online synchronous rotors, causing frequency to drop. Converter-interfaced solar/wind plants configured in constant-PQ grid-following (GFL) mode contribute zero mechanical inertia (<code>H<sub>vir</sub> = 0</code>). Displacing <code>10%–50%</code> of synchronous dispatch shrinks total inertia <code>H<sub>tot</sub></code> from <code>324.7 s</code> down to <code>174.6 s</code>, producing steeper initial ROCOF and deeper frequency drops.</li>
  <li><strong>COI Frequency Nadir (f<sub>nadir</sub>) &amp; 49.50 Hz Security Criterion:</strong> Turbine governors (<code>T<sub>g</sub> = 4.20 s</code>, <code>5%</code> machine-base droop) gradually ramp mechanical power to arrest the fall at <code>t<sub>nadir</sub> &isin; [2.78, 3.70] s</code>. Following Indian Grid Code (IEGC) primary containment conventions, trajectories maintaining <code>f<sub>nadir</sub> &ge; 49.50 Hz</code> are <strong>Secure (1)</strong>; excursions below <code>49.50 Hz</code> risk Under-Frequency Load Shedding (UFLS) and are <strong>Insecure (0)</strong>.</li>
</ul>

<div class="fig-box">
  <img src="{fig01}" alt="IEEE 39-bus topology and causal screening architecture">
  <div class="fig-cap"><strong>Figure 1 (<code>results/figures/fig01_system_and_architecture.png</code>):</strong> (a) IEEE 39-bus, 10-machine, 46-branch network partitioned into 3 coherent areas with primary tie-lines (<code>1–39</code>, <code>4–5</code>, <code>16–17</code>) and sparse 13-bus PMU subset. (b) Five-stage strictly causal PMU-to-decision screening pipeline.</div>
</div>

<h3>1.2 Why Existing Approaches Fall Short (The Validated Research Gap)</h3>
<ol>
  <li><strong>Time-domain ODE integration is accurate but computationally heavy for bulk contingency screening:</strong> Integrating a <code>10 s</code> trajectory takes <code>125.38 ms</code> median per scenario on CPU—requiring over a minute for 500+ N-1/renewable operating combinations.</li>
  <li><strong>Post-mortem or non-causal ML pipelines leak post-nadir information:</strong> Using full trajectories or internal simulator rotor speeds (<code>&omega;<sub>i</sub></code>, <code>H<sub>eff</sub></code>) violates physical causality and eliminates operational warning lead time.</li>
  <li><strong>Uncalibrated classifiers ignore asymmetric blackout risk &amp; measurement imperfections:</strong> Clearing a truly insecure emergency (<code>f<sub>nadir</sub> &lt; 49.50 Hz</code>) as "Secure" (<code>FNR<sub>insecure</sub></code>) is safety-critical, while naive 2-point finite-difference ROCOF amplifies PMU noise by orders of magnitude.</li>
</ol>

<div class="callout-blue">
  <strong>Primary Research Question:</strong> <em>"How early after a 50.0 Hz generation-loss contingency (observing only <code>T<sub>obs</sub> &isin; {{0.5, 1.0, 1.5, 2.0}} s</code> of observable wide-area PMU streams) can a leakage-free, probability-calibrated ML framework predict continuous COI frequency nadir and safety-critical 49.50 Hz security status across unseen 40–50% renewable, extreme load, noisy, and sparse 13-bus PMU regimes?"</em>
</div>

<h2>2. Word-by-Word Mapping From Project Title to Codebase</h2>

<table>
  <thead>
    <tr>
      <th style="width:20%">Title Phrase</th>
      <th style="width:45%">Technical Role in Our 50.0 Hz Protocol</th>
      <th style="width:35%">Exact Python Module</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Early</strong></td>
      <td>Restricts inputs strictly to pre-nadir window <code>[t<sub>0</sub>, t<sub>0</sub> + T<sub>obs</sub>]</code>, preserving <code>1.29 s</code> mean lead time at <code>T<sub>obs</sub> = 1.0 s</code> (<code>100%</code> causal across <code>0.5–1.5 s</code>).</td>
      <td><code>src/pmu_fsa/features/window_slicer.py</code></td>
    </tr>
    <tr>
      <td><strong>PMU-Based</strong></td>
      <td>Uses only observable <code>50 fps</code> / <code>25 fps</code> bus synchrophasors, area-weighted <code>f<sub>sys,pmu</sub></code>, causal 5-pt OLS ROCOF, and C37.118.2 framing.</td>
      <td><code>src/pmu_fsa/simulation/pmu_recorder.py</code></td>
    </tr>
    <tr>
      <td><strong>Prediction of Nadir &amp; Security Status</strong></td>
      <td>7 continuous nadir regressors + 7 classifiers + cross-fit Platt &amp; PAVA Isotonic <code>CalibratedGBDTClassifier</code> with quantile/conformal intervals.</td>
      <td><code>src/pmu_fsa/models/regressors.py</code>,<br><code>classifiers.py</code>, <code>calibrated_gb.py</code></td>
    </tr>
    <tr>
      <td><strong>Renewable-Integrated</strong></td>
      <td>Linear synchronous inertia displacement across all 9 non-slack generator buses (<code>10%–50%</code> load-referenced RES; <code>9.93%–49.66%</code> realized).</td>
      <td><code>src/pmu_fsa/simulation/renewables.py</code></td>
    </tr>
    <tr>
      <td><strong>Multi-Area Power Systems</strong></td>
      <td>IEEE 39-bus 10-machine network at <code>50.0 Hz</code> (<code>H<sub>39</sub> = 42.0 s</code> regional default), NR AC power flow, and complete/partial Kron outage physics.</td>
      <td><code>src/pmu_fsa/simulation/ieee39_data.py</code>,<br><code>dynamic_engine.py</code></td>
    </tr>
    <tr>
      <td><strong>Under Limited Observation</strong></td>
      <td>Evaluates short windows (<code>0.5–2.0 s</code>), unseen OOD regimes (<code>N=480</code>), PMU Gaussian noise (<code>0–50 mHz</code>), sparse <code>13/39</code>-bus PMUs (<code>33.3%</code>), and <code>25 fps</code>.</td>
      <td><code>src/pmu_fsa/evaluation/sensitivity.py</code>,<br><code>ood_generalization.py</code>, <code>noise_stress.py</code>, <code>interpretability.py</code></td>
    </tr>
  </tbody>
</table>

<h2>3. Deep-Dive: 50.0 Hz Power System Physics &amp; PMU Layer</h2>

<h3>3.1 IEEE 39-Bus 50.0 Hz Model, NR Power Flow &amp; Machine-Base Droop</h3>
<ul>
  <li><strong>Regional Bus-39 Default (<code>macro_intertie = False</code>):</strong> Bus 39 (<code>G10</code>, <code>1,000 MW</code>, <code>1,200 MVA</code>) uses regional thermal inertia <code>H<sub>39</sub> = 42.0 s</code> (<code>X'<sub>d</sub> = 0.0400 pu</code>), giving pre-fault <code>H<sub>tot</sub> = 324.7 s</code> on <code>S<sub>base</sub> = 100 MVA</code>. (Opt-in <code>--macro-intertie</code> sets <code>H<sub>39</sub> = 500.0 s</code>.)</li>
  <li><strong>Exact Newton-Raphson AC Power Flow Initialization:</strong> Across all 5 load scales (<code>0.80, 0.90, 1.00, 1.10, 1.20&times;</code> of nominal <code>6,150.1 MW</code> / <code>1,408.9 MVAR</code>), <code>solve_power_flow()</code> converges to <code>&lt; 10<sup>-11</sup> pu</code> mismatch, eliminating pre-fault mechanical-electrical mismatch (<code>0.0000 Hz</code> flat-run drift).</li>
  <li><strong>5% Machine-Base Governor Droop &amp; Complete vs. Partial Outage Physics:</strong> Each generator <code>i</code> has rating <code>S<sub>mva,i</sub> &isin; [350, 1200] MVA</code>, <code>R<sub>i</sub> = 0.05</code> converted to system base via <code>R<sub>sys,i</sub> = 0.05 &middot; (100 / S<sub>mva,i</sub>)</code>, and steam-reheat lag <code>T<sub>g</sub> = 4.20 s</code>. At <code>t<sub>0</sub> = 1.0 s</code>, <code>loss_fraction = 1.0</code> completely disconnects unit <code>j</code> from post-fault Kron admittance <code>Y<sub>red,post</sub></code>, reconstruction matrix <code>K<sub>post</sub></code>, and COI inertia <code>H<sub>post</sub></code> (<code>P<sub>e,post,j</sub> = 0 MW</code>); partial plant trips (<code>0.70, 0.40</code>) scale <code>y<sub>g,j</sub>, H<sub>j</sub>, D<sub>j</sub>, P<sub>m0,j</sub>, R<sub>sys,j</sub><sup>-1</sup></code> by <code>1 - loss_fraction</code>.</li>
</ul>

<div class="eq-box">
  <strong>Coupled 30-State ODEs:</strong>&nbsp;&nbsp;
  <code>d&delta;<sub>i</sub>/dt = (2&pi;&middot;50)&Delta;&omega;<sub>i</sub></code> &nbsp;|&nbsp;
  <code>2H<sub>i</sub>(d&Delta;&omega;<sub>i</sub>/dt) = P<sub>m,i</sub> &minus; P<sub>e,i</sub>(&delta;) &minus; D<sub>i</sub>&Delta;&omega;<sub>i</sub></code> &nbsp;|&nbsp;
  <code>T<sub>g,i</sub>(dP<sub>m,i</sub>/dt) = P<sub>m0,i</sub> &minus; &Delta;&omega;<sub>i</sub>/R<sub>sys,i</sub> &minus; P<sub>m,i</sub></code>
</div>

<div class="fig-box">
  <img src="{fig02}" alt="Simulated 50 Hz generation loss and inertia displacement responses">
  <div class="fig-cap"><strong>Figure 2 (<code>results/figures/fig02_generation_loss_response.png</code>):</strong> Real 50.0 Hz trajectories showing (a) 40%, 70%, and 100% unit loss on Bus 38, (b) inertia displacement across 10%, 30%, and 50% RES on Bus 34 crossing the 49.50 Hz threshold, and (c) Area 1–3 spatial swings vs. true COI and PMU-reconstructed <code>f<sub>sys,pmu</sub>(t)</code>.</div>
</div>

<h3>3.2 Strict Simulator/PMU Separation &amp; Causal 5-Point OLS ROCOF</h3>
<ul>
  <li><strong>Zero Simulator Leakage (<code>data/raw_sim/</code> vs. <code>data/raw/</code>):</strong> Internal ODE states (<code>coi_frequency</code>, rotor speeds) are archived strictly in <code>data/raw_sim/*.npz</code>. Feature extraction reads only <code>data/raw/*.npz</code>, reconstructing <code>f<sub>sys,pmu</sub>(t)</code> from observed bus frequencies via area-weighted bus means <code>(9&mu;<sub>A1</sub> + 15&mu;<sub>A2</sub> + 15&mu;<sub>A3</sub>)/39</code>.</li>
  <li><strong>Causal 5-Point Backward OLS ROCOF:</strong> Computes closed-form backward OLS slope over past/present frames only (<code>0.1 &middot; fps &middot; [2(f<sub>n</sub> &minus; f<sub>n-4</sub>) + f<sub>n-1</sub> &minus; f<sub>n-3</sub>]</code>), cutting noise variance vs. 2-point finite differences to a ratio of <code>0.0003</code> (<code>0.001616</code> vs. <code>4.6421</code>).</li>
</ul>

<div class="fig-box">
  <img src="{fig03}" alt="Causal observation windows and pre-nadir lead time">
  <div class="fig-cap"><strong>Figure 3 (<code>results/figures/fig03_causal_window_illustration.png</code>):</strong> (a) Real contingency trajectory (<code>scen_gen32_loss650_ren10_ld100_000</code>) illustrating causal windows <code>T<sub>obs</sub> &isin; {{0.5, 1.0, 1.5, 2.0}} s</code> prior to <code>t<sub>nadir</sub> = 3.38 s</code>. (b) Noise suppression of wide-area causal 5-point OLS ROCOF vs. 2-point differences. (c) Dataset-wide warning lead time <code>t<sub>lead</sub></code> across all 750 scenarios.</div>
</div>

<h2>4. Canonical 14 Synchrophasor Features &amp; Disjoint 750-Scenario Splits</h2>

<table>
  <thead>
    <tr>
      <th style="width:4%">#</th>
      <th style="width:24%">Canonical Feature Name</th>
      <th style="width:36%">Causal Window Definition (<code>W = [t<sub>0</sub>, t<sub>0</sub>+T<sub>obs</sub>]</code>)</th>
      <th style="width:9%">Units</th>
      <th style="width:27%">Power-System Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td><td><code>rocof_init</code></td><td>OLS slope of <code>f<sub>sys,pmu</sub></code> on <code>[t<sub>0</sub>, t<sub>0</sub>+0.10 s]</code></td><td><code>Hz/s</code></td><td>Initial inertial decay rate (<code>&minus;&Delta;P&middot;f<sub>0</sub> / 2H<sub>eff</sub></code>)</td></tr>
    <tr><td>2</td><td><code>rocof_peak_neg</code></td><td><code>min<sub>t&isin;W</sub> rocof<sub>sys,pmu</sub>(t)</code></td><td><code>Hz/s</code></td><td>Worst downward 5-pt wide-area ROCOF</td></tr>
    <tr><td>3</td><td><code>rocof_end</code></td><td><code>rocof<sub>sys,pmu</sub>(t<sub>0</sub> + T<sub>obs</sub>)</code></td><td><code>Hz/s</code></td><td>Boundary ROCOF at decision instant</td></tr>
    <tr><td>4</td><td><code>rocof_mean</code></td><td>OLS slope of <code>f<sub>sys,pmu</sub></code> across full <code>W</code></td><td><code>Hz/s</code></td><td>Noise-resilient window-wide decay rate</td></tr>
    <tr><td>5</td><td><code>freq_drop_end</code></td><td><code>f<sub>sys,pmu</sub>(t<sub>0</sub>+T<sub>obs</sub>) &minus; f<sub>sys,pmu</sub>(t<sub>0</sub>)</code></td><td><code>Hz</code></td><td>Net wide-area frequency drop at <code>t<sub>0</sub>+T<sub>obs</sub></code></td></tr>
    <tr><td>6</td><td><code>freq_drop_max</code></td><td><code>min<sub>t&isin;W</sub> f<sub>sys,pmu</sub>(t) &minus; 50.0</code></td><td><code>Hz</code></td><td>Peak excursion below nominal inside <code>W</code></td></tr>
    <tr><td>7</td><td><code>freq_dev_integral</code></td><td><code>&int;<sub>W</sub> (f<sub>sys,pmu</sub>(t) &minus; 50.0) dt</code></td><td><code>Hz&middot;s</code></td><td>Cumulative kinetic-energy deficit (#1 feature)</td></tr>
    <tr><td>8</td><td><code>bus_drop_worst</code></td><td><code>nanmin<sub>t&isin;W, b&isin;B<sub>obs</sub></sub> (f<sub>bus</sub>(t,b) &minus; 50.0)</code></td><td><code>Hz</code></td><td>Deepest local substation frequency dip</td></tr>
    <tr><td>9</td><td><code>area_freq_spread_max</code></td><td><code>max<sub>t&isin;W, a,b</sub> |&mu;<sub>Area a</sub>(t) &minus; &mu;<sub>Area b</sub>(t)|</code></td><td><code>Hz</code></td><td>Peak inter-area frequency separation</td></tr>
    <tr><td>10</td><td><code>bus_freq_std_mean</code></td><td><code>mean<sub>t&isin;W</sub> nanstd<sub>b&isin;B<sub>obs</sub></sub> f<sub>bus</sub>(t,b)</code></td><td><code>Hz</code></td><td>Mean spatial bus frequency dispersion</td></tr>
    <tr><td>11</td><td><code>angle_divergence_max</code></td><td><code>max<sub>t&isin;W, a,b</sub> |wrap<sub>180</sub>(&Delta;&theta;<sub>a</sub>(t) &minus; &Delta;&theta;<sub>b</sub>(t))|</code></td><td><code>deg</code></td><td>Peak network-wide relative angle stress</td></tr>
    <tr><td>12</td><td><code>tie_line_angle_excursion</code></td><td>Peak wrap-safe swing on <code>(4,5), (1,39), (16,17)</code></td><td><code>deg</code></td><td>Inter-area synchronizing tie-line swing</td></tr>
    <tr><td>13</td><td><code>freq_curvature_init</code></td><td>OLS slope of <code>rocof<sub>sys,pmu</sub></code> on <code>[t<sub>0</sub>, t<sub>0</sub>+0.20 s]</code></td><td><code>Hz/s&sup2;</code></td><td>Initial 2nd-derivative deceleration curvature</td></tr>
    <tr><td>14</td><td><code>freq_curvature_mean</code></td><td>OLS slope of <code>rocof<sub>sys,pmu</sub></code> across full <code>W</code></td><td><code>Hz/s&sup2;</code></td><td>Governor arrest curvature (<code>d&sup2;f/dt&sup2; &gt; 0</code>)</td></tr>
  </tbody>
</table>

<p><strong>750-Scenario Disjoint Partitioning:</strong> <code>10</code> buses &times; <code>3</code> loss fractions (<code>1.0, 0.70, 0.40</code>) &times; <code>5</code> RES levels (<code>10%–50%</code>) &times; <code>5</code> load scales (<code>0.80–1.20&times;</code>) = <strong>750 scenarios</strong> (<code>303</code> Secure, <code>447</code> Insecure). Scenario-hash MD5 splitting yields <strong>270 ID scenarios</strong> (<code>148 train</code>, <code>70 val</code>, <code>52 held-out test</code>) and <strong>480 OOD scenarios</strong> (<code>180 ood_res</code> + <code>180 ood_load</code> + <code>120 ood_combined</code>).</p>

<h2>5. Verified 750-Scenario Experimental Results (50.0 Hz)</h2>

<h3>5.1 Continuous Nadir Regression (Held-Out ID Test, N = 52, T<sub>obs</sub> = 1.0 s)</h3>
<table>
  <thead>
    <tr>
      <th>Regressor Architecture</th>
      <th>Val MAE (Hz)</th>
      <th>Test MAE (Hz)</th>
      <th>RMSE (Hz)</th>
      <th>P95 |e| (Hz)</th>
      <th>R&sup2;</th>
      <th>95% Bootstrap CI</th>
      <th>Infer (ms)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Constant Mean Baseline</td><td>0.2319</td><td>0.2104</td><td>0.2526</td><td>0.4053</td><td>&minus;0.0117</td><td>[0.1759, 0.2517]</td><td>0.0008</td></tr>
    <tr><td>Analytical Swing Baseline</td><td>0.0035</td><td>0.0042</td><td>0.0056</td><td>0.0125</td><td>0.9995</td><td>[0.0032, 0.0052]</td><td>0.0127</td></tr>
    <tr><td><strong>Ridge Linear Regression</strong></td><td><strong>0.0034</strong></td><td><strong>0.0034</strong></td><td><strong>0.0046</strong></td><td><strong>0.0095</strong></td><td><strong>0.9997</strong></td><td><strong>[0.0026, 0.0043]</strong></td><td>0.0111</td></tr>
    <tr><td>Decision Tree (CART)</td><td>0.0151</td><td>0.0111</td><td>0.0148</td><td>0.0270</td><td>0.9965</td><td>[0.0087, 0.0137]</td><td>0.0086</td></tr>
    <tr><td>Random Forest (100 Trees)</td><td>0.0112</td><td>0.0090</td><td>0.0140</td><td>0.0274</td><td>0.9969</td><td>[0.0064, 0.0122]</td><td>0.2905</td></tr>
    <tr><td><strong>Gradient Boosting (GBDT)</strong></td><td>0.0106</td><td><strong>0.0077</strong></td><td><strong>0.0113</strong></td><td><strong>0.0179</strong></td><td><strong>0.9980</strong></td><td><strong>[0.0058, 0.0101]</strong></td><td>0.0182</td></tr>
    <tr><td>LightGBM (HistGBDT)</td><td>0.0148</td><td>0.0086</td><td>0.0116</td><td>0.0232</td><td>0.9979</td><td>[0.0067, 0.0107]</td><td>0.0423</td></tr>
  </tbody>
</table>

<div class="fig-box">
  <img src="{fig06}" alt="Regression model comparison across 7 models">
  <div class="fig-cap"><strong>Figure 4 (<code>results/figures/fig06_regression_model_comparison.png</code>):</strong> Held-out ID test error across all 7 regressors. Smooth global L2/analytical polynomials achieve <code>3.4–4.2 mHz</code> MAE on smooth classical ODE trajectories, while GBDT achieves <code>7.7 mHz</code> MAE (<code>R&sup2; = 0.9980</code>) and powers calibrated non-parametric quantile uncertainty intervals.</div>
</div>

<h3>5.2 Security Classification, Calibration &amp; Threshold Safety Sweep (Held-Out ID Test, N = 52)</h3>
<table>
  <thead>
    <tr>
      <th>Classifier Model</th>
      <th>Val Brier</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Test F1</th>
      <th>PR-AUC</th>
      <th>Test Brier</th>
      <th>Test ECE</th>
      <th>FNR<sub>insecure</sub> (&tau;=0.50)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Logistic Regression</td><td>0.0123</td><td>1.0000</td><td>1.0000</td><td><strong>1.0000</strong></td><td>1.0000</td><td>0.0132</td><td>0.0501</td><td><strong>0.0000</strong> (0/31)</td></tr>
    <tr><td>Decision Tree</td><td>0.0143</td><td>0.9545</td><td>1.0000</td><td>0.9767</td><td>0.9773</td><td>0.0192</td><td>0.0192</td><td>0.0323 (1/31)</td></tr>
    <tr><td>Random Forest</td><td>0.0091</td><td>0.9545</td><td>1.0000</td><td>0.9767</td><td>1.0000</td><td>0.0093</td><td>0.0223</td><td>0.0323 (1/31)</td></tr>
    <tr><td>Uncalibrated GBDT</td><td>0.0143</td><td>0.9545</td><td>1.0000</td><td>0.9767</td><td>0.9773</td><td>0.0192</td><td>0.0193</td><td>0.0323 (1/31)</td></tr>
    <tr><td>LightGBM Classifier</td><td>0.0143</td><td>0.9545</td><td>1.0000</td><td>0.9767</td><td>0.9989</td><td>0.0181</td><td><strong>0.0179</strong></td><td>0.0323 (1/31)</td></tr>
    <tr><td><strong>Calibrated GBDT (Platt)</strong></td><td><strong>0.0056</strong></td><td>0.9545</td><td>1.0000</td><td>0.9767</td><td><strong>1.0000</strong></td><td>0.0141</td><td>0.0267</td><td>0.0323 (<strong>0.0000 @ &tau;=0.70</strong>)</td></tr>
    <tr><td><strong>Calibrated GBDT (Isotonic)</strong></td><td>0.0063</td><td>0.9545</td><td>1.0000</td><td>0.9767</td><td><strong>1.0000</strong></td><td><strong>0.0080</strong></td><td>0.0192</td><td>0.0323 (<strong>0.0000 @ &tau;=0.70</strong>)</td></tr>
  </tbody>
</table>

<div class="fig-box">
  <img src="{fig07}" alt="Classification confusion matrix and safety threshold sweep">
  <div class="fig-cap"><strong>Figure 5 (<code>results/figures/fig07_classification_and_safety.png</code>):</strong> Held-out ID confusion matrix (<code>TN=30, FP=1, FN=0, TP=21</code> at <code>&tau;=0.50</code>), 7-classifier comparison, and safety threshold sweep showing <code>&tau; = 0.70</code> achieves <strong>0 missed insecure emergencies</strong> (<code>FNR<sub>insecure</sub> = 0.0000</code>, <code>FPR<sub>secure</sub> = 0.0476</code>).</div>
</div>

<div class="fig-box">
  <img src="{fig08}" alt="Calibration reliability curves">
  <div class="fig-cap"><strong>Figure 6 (<code>results/figures/fig08_calibration_reliability.png</code>):</strong> Learned 5-fold cross-fit Platt sigmoid (<code>A = 71.81</code>, <code>N<sub>cal</sub> = 218</code>) and PAVA Isotonic curve (<code>216</code> knots, <code>fallback_used = False</code>), 10-bin reliability diagram across <code>N = 532</code> held-out scenarios, and Brier/ECE improvements over uncalibrated GBDT.</div>
</div>

<h3>5.3 Window Sensitivity, OOD Domain Shift, Noise, Sparse 13-Bus PMU &amp; Latency</h3>

<div class="two-col">
  <div class="card">
    <h3>Window Trade-Off (Held-Out ID / OOD)</h3>
    <table>
      <thead><tr><th>T<sub>obs</sub></th><th>ID MAE</th><th>OOD MAE</th><th>OOD F1</th><th>Mean t<sub>lead</sub></th></tr></thead>
      <tbody>
        <tr><td><code>0.5 s</code></td><td>0.0124</td><td>0.0279</td><td>0.9831</td><td>1.79 s</td></tr>
        <tr><td><code>1.0 s</code></td><td><strong>0.0077</strong></td><td><strong>0.0184</strong></td><td>0.9860</td><td><strong>1.29 s</strong></td></tr>
        <tr><td><code>1.5 s</code></td><td><strong>0.0069</strong></td><td>0.0218</td><td><strong>0.9943</strong></td><td>0.79 s</td></tr>
        <tr><td><code>2.0 s</code></td><td>0.0074</td><td>0.0229</td><td>0.9861</td><td>0.29 s</td></tr>
      </tbody>
    </table>
    <p><code>T<sub>obs</sub> = 1.0 s</code> is the optimal early-warning operating point: <code>100%</code> pre-nadir causal (<code>750/750</code>), <code>7.7 mHz</code> ID MAE, <code>18.4 mHz</code> OOD MAE, and <code>1.29 s</code> mean warning lead time.</p>
  </div>
  <div class="card">
    <h3>OOD Regimes &amp; PMU Noise (T<sub>obs</sub> = 1.0 s)</h3>
    <table>
      <thead><tr><th>Regime / Stress</th><th>N</th><th>MAE (Hz)</th><th>F1</th><th>ECE</th></tr></thead>
      <tbody>
        <tr><td>ID Held-Out Test</td><td>52</td><td>0.0077</td><td>0.9767</td><td>0.0267</td></tr>
        <tr><td>OOD-RES (40–50%)</td><td>180</td><td>0.0267</td><td><strong>1.0000</strong></td><td>0.0216</td></tr>
        <tr><td>OOD-Load (0.8, 1.2&times;)</td><td>180</td><td>0.0078</td><td>0.9792</td><td>0.0263</td></tr>
        <tr><td>OOD-Combined</td><td>120</td><td>0.0217</td><td>0.9851</td><td>0.0250</td></tr>
        <tr><td>PMU Noise <code>10 mHz</code></td><td>52</td><td>0.0085</td><td>0.9767</td><td>0.0272</td></tr>
        <tr><td>PMU Noise <code>50 mHz</code></td><td>52</td><td>0.0114</td><td>0.9767</td><td>0.0418</td></tr>
      </tbody>
    </table>
  </div>
</div>

<div class="fig-box">
  <img src="{fig09}" alt="Observation window sensitivity">
  <div class="fig-cap"><strong>Figure 7 (<code>results/figures/fig09_window_sensitivity.png</code>):</strong> Unsmoothed empirical observation-window trade-off across <code>T<sub>obs</sub> &isin; {{0.5, 1.0, 1.5, 2.0}} s</code> on held-out ID and OOD partitions.</div>
</div>

<div class="fig-box">
  <img src="{fig10}" alt="OOD generalization breakdown">
  <div class="fig-cap"><strong>Figure 8 (<code>results/figures/fig10_ood_generalization.png</code>):</strong> Cross-regime generalization showing <code>100%</code> OOD-RES F1 (<code>0</code> missed emergencies) and automatic <code>+12.3%</code> widening of 95% quantile/OOD intervals (<code>0.662 &rarr; 0.743 Hz</code>).</div>
</div>

<div class="fig-box">
  <img src="{fig12}" alt="Feature importance and partial PMU ablations">
  <div class="fig-cap"><strong>Figure 9 (<code>results/figures/fig12_feature_importance_and_ablation.png</code>):</strong> Permutation importance across all 14 canonical features (<code>freq_dev_integral</code> #1 at <code>+169.5 mHz</code>) plus sparse <code>13/39</code>-bus PMU (<code>10.8 mHz</code> ID MAE, <code>100.0%</code> F1) and <code>25 fps</code> (<code>9.4 mHz</code> ID MAE) ablations.</div>
</div>

<div class="fig-box">
  <img src="{fig13}" alt="6-stage hardware latency and learning curve">
  <div class="fig-cap"><strong>Figure 10 (<code>results/figures/fig13_latency_and_learning_curve.png</code>):</strong> (a) Hardware-measured 6-stage online latency (<code>2.3805 ms</code> median, <code>3.3641 ms</code> P99) vs. <code>125.3759 ms</code> 10-s ODE simulation (<code>52.7&times;</code> speedup). (b) Disjoint scenario-level learning curve (<code>20%–100%</code> of <code>train</code>).</div>
</div>

<h2>6. Exact Reproduction Commands &amp; 4-Member Team Viva Split</h2>

<pre>cd btp-early-pmu-fsa

# Single Master Command (Runs Simulation -> Extraction -> Validation -> Training -> Benchmarks -> 14 Figures)
.venv/bin/python3 scripts/run_all.py --config configs/default_50hz.yaml --with-figures

# Fast Reproduction Reusing Existing Raw Trajectories + Running Full Isolated Production Test Suite
.venv/bin/python3 scripts/run_all.py --config configs/default_50hz.yaml --skip-simulation --with-figures
.venv/bin/pytest tests/ -q</pre>

<table>
  <thead>
    <tr>
      <th style="width:18%">Team Member</th>
      <th style="width:32%">Owned Codebase Modules</th>
      <th style="width:50%">Viva Presentation Focus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Aditya Jha</strong><br>Member 1 — Grid Physics Lead</td>
      <td><code>ieee39_data.py</code>, <code>renewables.py</code>, <code>dynamic_engine.py</code>, <code>run_simulations.py</code></td>
      <td>50.0 Hz IEEE 39-bus model (<code>H<sub>39</sub>=42 s</code>), NR AC power flow (<code>&lt;10<sup>-11</sup> pu</code>), machine-base 5% droop, complete vs. partial Kron outage physics, and Approach B co-located GFL inertia/damping/droop derating.</td>
    </tr>
    <tr>
      <td><strong>Akshat Jain</strong><br>Member 2 — PMU &amp; Features Lead</td>
      <td><code>pmu_recorder.py</code>, <code>window_slicer.py</code>, <code>extractor.py</code>, <code>dataset_validator.py</code></td>
      <td><code>raw_sim</code> vs. <code>raw</code> firewall, strict PMU-only enforcement, area-weighted <code>f<sub>sys,pmu</sub></code>, causal 5-pt backward OLS ROCOF (<code>0.0003</code> variance ratio), 14 canonical features, and 14-check SHA-256 dataset audit.</td>
    </tr>
    <tr>
      <td><strong>Arshdeep Singh</strong><br>Member 3 — ML &amp; Calibration Lead</td>
      <td><code>regressors.py</code>, <code>classifiers.py</code>, <code>calibrated_gb.py</code>, <code>train_models.py</code></td>
      <td>7 regressors (validation-selected Ridge <code>3.4 mHz</code> vs. proposed GBDT <code>7.7 mHz</code> &amp; noise robustness), 5-fold cross-fit Platt &amp; 216-knot PAVA Isotonic calibration, quantile/OOD intervals, and <code>&tau;=0.70</code> zero-missed safety sweep.</td>
    </tr>
    <tr>
      <td><strong>Raj Bathla</strong><br>Member 4 — Robustness &amp; Speed Lead</td>
      <td><code>sensitivity.py</code>, <code>ood_generalization.py</code>, <code>noise_stress.py</code>, <code>latency.py</code>, <code>interpretability.py</code></td>
      <td>Strictly pre-nadir <code>0.5–2.0 s</code> window trade-off (<code>472</code> causal OOD at <code>2.0 s</code>), <code>480</code>-scenario OOD breakdown, <code>0–50 mHz</code> noise stability, <code>13/39</code> sparse PMU &amp; <code>25 fps</code> ablations, Holm sign-flip permutation tests, and <code>2.53 ms</code> desktop latency.</td>
    </tr>
  </tbody>
</table>

<h2>7. Top 10 BTP Viva &amp; Mentor Defense Q&amp;As</h2>

<div class="qa">
  <div class="q">Q1: Why is Bus 39 configured with H<sub>39</sub> = 42.0 s instead of H<sub>39</sub> = 500.0 s in the 750-scenario benchmark?</div>
  <div>In Pai's classical New England test case, <code>H<sub>39</sub> = 500 s</code> represents a stiff external US–Canada intertie that clamps system frequency above <code>49.8 Hz</code> for almost every single-unit trip. Setting <code>macro_intertie = False</code> (<code>H<sub>39</sub> = 42.0 s</code>, <code>H<sub>tot</sub> = 324.7 s</code>) models an islanded regional multi-area grid where generator outages produce realistic 50-Hz frequency nadirs (<code>48.26–49.91 Hz</code>) straddling the <code>49.50 Hz</code> IEGC security floor.</div>
</div>

<div class="qa">
  <div class="q">Q2: How do you guarantee zero pre-disturbance drift and physically realistic generator disconnection?</div>
  <div>Before every simulation, a vectorized Newton-Raphson AC power flow solves exact bus voltages and slack/PV reactive dispatches to <code>&lt; 10<sup>-11</sup> pu</code> mismatch so <code>P<sub>m0</sub> = P<sub>e</sub>(t<sub>0</sub><sup>-</sup>)</code> (<code>0.0000 Hz</code> flat run). At <code>t<sub>0</sub> = 1.0 s</code>, complete outages (<code>loss_fraction = 1.0</code>) remove the tripped generator's Norton admittance from post-fault Kron matrix <code>Y<sub>red,post</sub></code> and COI inertia <code>H<sub>post</sub></code> so post-trip electrical output is identically <code>0.0 MW</code>.</div>
</div>

<div class="qa">
  <div class="q">Q3: How do you prevent internal simulator leakage and future-data leakage?</div>
  <div>Four layers enforce causality: (1) Internal ODE rotor states are isolated in <code>data/raw_sim/</code> while <code>data/raw/</code> stores only observable PMU channels; (2) <code>f<sub>sys,pmu</sub>(t)</code> is reconstructed solely from observed PMU bus frequencies; (3) <code>extract_causal_window</code> slices <code>[t<sub>0</sub>, t<sub>0</sub> + T<sub>obs</sub>]</code> and strips forbidden target keys (verified by bitwise zero-change post-window poisoning tests); and (4) MD5 scenario-hash splits keep <code>train</code> (148), <code>val</code> (70), <code>test</code> (52), and <code>ood_test</code> (480) 100% disjoint.</div>
</div>

<div class="qa">
  <div class="q">Q4: Why do Ridge Linear (3.4 mHz) and Analytical Swing (4.2 mHz) achieve lower ID MAE than GBDT (7.7 mHz)?</div>
  <div>Under classical 3rd-order swing + linear droop ODEs on clean PMU streams, the mapping from early kinetic-energy integral (<code>freq_dev_integral</code>), frequency drop, and arrest curvature (<code>d&sup2;f/dt&sup2;</code>) to <code>f<sub>nadir</sub></code> is smooth and nearly quadratic. Global L2-regularized models interpolate smoothly without piecewise-constant tree leaf quantization (<code>~5–8 mHz</code>), whereas <code>CalibratedGBDTClassifier</code> delivers the lowest validation Brier loss (<code>0.0056</code>), non-parametric tree splits, and quantile/OOD prediction intervals!</div>
</div>

<div class="qa">
  <div class="q">Q5: How do Platt Scaling and PAVA Isotonic Calibration compare on your 750-scenario dataset?</div>
  <div>Using 5-fold stratified cross-fitting across <code>train + val</code> (<code>N<sub>cal</sub> = 218</code>, well above the small-sample fallback threshold), both calibrators fit natively (<code>fallback_used = False</code>). Platt scaling achieves the lowest validation Brier score (<code>0.0056</code> vs. <code>0.0143</code> uncalibrated) with smooth sigmoid slope <code>A = 71.81</code>, while PAVA Isotonic (<code>216</code> knots) achieves the lowest held-out test Brier score (<code>0.0080</code>, <code>ECE = 0.0192</code>, <code>PR-AUC = 1.0000</code>).</div>
</div>

<div class="qa">
  <div class="q">Q6: How do you eliminate safety-critical missed emergencies (FNR<sub>insecure</sub>)?</div>
  <div>At the default symmetric cutoff <code>&tau; = 0.50</code>, <code>1</code> borderline insecure test scenario out of <code>31</code> (<code>f<sub>nadir</sub></code> just below <code>49.50 Hz</code>) is missed (<code>FNR<sub>insecure</sub> = 3.23%</code>). Raising the calibrated decision cutoff to conservative <code>&tau; = 0.70</code> (requiring <code>&ge; 70%</code> confidence to clear a contingency as safe) reduces <code>FNR<sub>insecure</sub></code> to <strong>0.0000 (0/31 missed)</strong> with only <code>1</code> benign false alarm (<code>FPR<sub>secure</sub> = 4.76%</code>, <code>F1 = 0.9756</code>).</div>
</div>

<div class="qa">
  <div class="q">Q7: Why does MAE increase to 26.7 mHz on OOD-RES (40–50% renewables), while Security F1 reaches 1.0000?</div>
  <div>At <code>40%–50%</code> RES, reduced inertia drives severe COI nadirs down to <code>48.26 Hz</code>—below the minimum nadir observed in <code>10%–30%</code> RES training (<code>48.72 Hz</code>). Because tree ensembles bound predictions at their lowest training leaf, extreme tail errors increase (<code>P95 |e| = 127.1 mHz</code>), and our uncertainty layer appropriately widens the 95% PI by <code>+12.3%</code> (<code>0.662 &rarr; 0.743 Hz</code>). Crucially, <code>49.50 Hz</code> lies well inside training support, yielding <code>F1 = 1.0000</code> and <code>0</code> missed emergencies across all 180 OOD-RES cases!</div>
</div>

<div class="qa">
  <div class="q">Q8: Why does your model survive 50 mHz PMU noise with F1 = 0.9767 unchanged?</div>
  <div>Two physical-statistical filters suppress zero-mean Gaussian sensor noise before inference: (1) area-weighted averaging across observed substations reduces variance by <code>1/N<sub>obs</sub></code>, and (2) closed-form 5-point backward OLS differentiation + trapezoidal integration (<code>freq_dev_integral</code>) attenuate high-frequency noise variance by <code>99.97%</code> relative to 2-point finite differences.</div>
</div>

<div class="qa">
  <div class="q">Q9: Does the method still work if utilities only have PMUs on one-third of substations or stream at 25 fps?</div>
  <div>Yes! Restricting observability to a canonical <code>13/39</code>-bus PMU placement (<code>33.3%</code> coverage across key generator step-up and tie-line boundary buses) yields <code>10.8 mHz</code> ID MAE and <code>100.0%</code> ID Security F1 (<code>23.8 mHz</code> OOD MAE). Halving the reporting rate from <code>50 fps</code> to <code>25 fps</code> changes ID MAE by only <code>+0.1 mHz</code> (<code>9.3 &rarr; 9.4 mHz</code>).</div>
</div>

<div class="qa">
  <div class="q">Q10: What are the honest limitations of this work before control-room relay deployment?</div>
  <div>(1) Classical 3rd-order synchronous generator ODEs and constant-impedance Kron loads omit subtransient AVR/PSS flux decay, boiler limits, and dynamic motor loads; (2) PMU channels use additive Gaussian noise + causal OLS rather than full hardware DFT phasor filters or cyber-attack packet loss; and (3) tree regressors under-extrapolate unseen <code>&lt;48.7 Hz</code> OOD nadirs—motivating a physics-residual hybrid (Analytical Swing + GBDT) and RTDS/OPAL-RT HIL validation in future work.</div>
</div>

</body>
</html>
"""


def main() -> int:
    html = build_html()
    HTML_OUT.write_text(html, encoding="utf-8")
    cmd = [
        CHROME_BIN,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--user-data-dir=/tmp/chrome_pdf_doc1",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_OUT}",
        f"file://{HTML_OUT}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not PDF_OUT.exists():
        sys.stderr.write(res.stderr + "\n")
        return 1
    print(f"Exported {PDF_OUT} ({PDF_OUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
