"""IEEE 39-bus dynamic simulation, PMU recording, scenario generation, and validation."""

from pmu_fsa.simulation.ieee39_data import (
    BASE_MVA,
    NOMINAL_FREQ,
    OMEGA_0,
    SECURITY_THRESHOLD_HZ,
    PHYSICAL_FREQ_BOUNDS,
    N_BUSES,
    N_GENERATORS,
    N_BRANCHES,
    AREA_1_BUSES,
    AREA_2_BUSES,
    AREA_3_BUSES,
    AREA_PARTITION,
    BUS_TO_AREA,
    GENERATOR_BUSES,
    GENERATOR_DATA,
    LOAD_DATA,
    BRANCH_DATA,
    get_ieee39_bus_data,
    get_ieee39_branch_data,
    get_ieee39_gen_data,
    get_total_system_inertia,
)
from pmu_fsa.simulation.renewables import (
    RenewablePenetrationConfig,
    compute_effective_inertias,
    apply_renewable_displacement,
)
from pmu_fsa.simulation.dynamic_engine import (
    build_ybus,
    compute_kron_reduction,
    DynamicSimulator,
)
from pmu_fsa.simulation.pmu_recorder import (
    PMUNoiseConfig,
    PMURecorder,
    apply_pmu_measurement_noise,
    compute_tve,
    compute_fir_rocof,
    compute_crc_ccitt,
)
from pmu_fsa.simulation.scenario_generator import (
    ScenarioGenerator,
    classify_ood_regime,
    assign_scenario_partition,
    create_scenario_id,
    export_raw_and_pmu_scenario,
    export_scenario_to_npz,
    load_scenario_from_npz,
)
from pmu_fsa.simulation.dataset_validator import (
    validate_dataset,
    write_dataset_quality_report,
)

__all__ = [
    "BASE_MVA",
    "NOMINAL_FREQ",
    "OMEGA_0",
    "SECURITY_THRESHOLD_HZ",
    "PHYSICAL_FREQ_BOUNDS",
    "N_BUSES",
    "N_GENERATORS",
    "N_BRANCHES",
    "AREA_1_BUSES",
    "AREA_2_BUSES",
    "AREA_3_BUSES",
    "AREA_PARTITION",
    "BUS_TO_AREA",
    "GENERATOR_BUSES",
    "GENERATOR_DATA",
    "LOAD_DATA",
    "BRANCH_DATA",
    "get_ieee39_bus_data",
    "get_ieee39_branch_data",
    "get_ieee39_gen_data",
    "get_total_system_inertia",
    "RenewablePenetrationConfig",
    "compute_effective_inertias",
    "apply_renewable_displacement",
    "build_ybus",
    "compute_kron_reduction",
    "DynamicSimulator",
    "PMUNoiseConfig",
    "PMURecorder",
    "apply_pmu_measurement_noise",
    "compute_tve",
    "compute_fir_rocof",
    "compute_crc_ccitt",
    "ScenarioGenerator",
    "classify_ood_regime",
    "assign_scenario_partition",
    "create_scenario_id",
    "export_raw_and_pmu_scenario",
    "export_scenario_to_npz",
    "load_scenario_from_npz",
    "validate_dataset",
    "write_dataset_quality_report",
]

try:
    from pmu_fsa.simulation.renewables import summarize_renewable_dispatch
    __all__.append("summarize_renewable_dispatch")
except ImportError:
    pass

try:
    from pmu_fsa.simulation.dynamic_engine import solve_power_flow
    __all__.append("solve_power_flow")
except ImportError:
    pass

try:
    from pmu_fsa.simulation.pmu_recorder import (
        CANONICAL_13BUS_PMU_PLACEMENT,
        compute_causal_rocof,
        estimate_pmu_system_frequency,
    )
    __all__.extend([
        "CANONICAL_13BUS_PMU_PLACEMENT",
        "compute_causal_rocof",
        "estimate_pmu_system_frequency",
    ])
except ImportError:
    pass
