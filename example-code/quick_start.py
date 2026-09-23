"""
Simple script for process simulation to test environment setup as found on https://github.com/franvinci/prosit.
Minor changes were done to get the code working.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore")

import pm4py
import pm4py.objects.log.importer.xes.importer as xes_importer
from prosit import SimulatorParameters, SimulatorEngine

if __name__ == "__main__":
    # 1. Load event log
    # download data at: https://github.com/franvinci/prosit-tool/blob/main/example_data/purchasing.xes
    log = xes_importer.apply("data/purchasing.xes")

    # 2. Discover Petri net
    net, im, fm = pm4py.discover_petri_net_inductive(log)

    # 3. Create and populate simulation parameters
    params = SimulatorParameters(net, im, fm)
    params.discover_from_eventlog(log, max_depth_tree=3, attribute_mode="empirical")

    # 4. Save for later reuse
    params.to_json("params_purchasing.json")

    # 5. Simulate 500 cases
    sim_engine = SimulatorEngine(params)
    sim_log = sim_engine.apply(n_traces=500)

    print(sim_log.head())