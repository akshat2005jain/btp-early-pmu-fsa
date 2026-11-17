| Parameter / Dimension | Specification | Notes |
| --- | --- | --- |
| Benchmark Power System | IEEE 39-Bus, 10-Machine, 46-Branch (3 Coherent Areas) | Base MVA = 100, Nominal Load = 6150.1 MW, Gen = 6192.8 MW |
| System Frequency & Threshold | 50.0 Hz nominal, 49.50 Hz security threshold | Regional Bus-39 inertia H_39 = 42.0 s (pre-fault H_tot = 324.7 s) |
| Outage Contingencies | 10 generators (Buses 30-39) x 3 loss fractions ([0.4, 0.7, 1.0]) | Active outage range: [80.0, 1200.0] MW |
| Renewable Penetration Grid | [10, 20, 30, 40, 50] % of system load | Realized RES range: [9.93%, 49.66%] |
| System Load Scaling Grid | [0.8, 0.9, 1.0, 1.1, 1.2] x nominal PQ load | Newton-Raphson AC power flow solved per operating point |
| Total Unique Scenarios | 750 trajectories (10.0 s horizon @ 50 fps, t_0 = 1.0 s) | Secure: 303 (40.4%), Insecure: 447 (59.6%) |
| COI Frequency Nadir Range | [48.2556, 49.9109] Hz | Median nadir = 49.4223 Hz, t_nadir in [2.78, 3.70] s |
| Causal Observation Windows | T_obs in {0.5, 1.0, 1.5, 2.0} s (primary T_obs = 1.0 s, 14 PMU features) | Pre-nadir causal at 1.0 s: 750/750 (100.00%) |