"""
Script for process simulation, including evaluation, to test environment setup as found on https://github.com/franvinci/prosit.
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
from pm4py.objects.log.obj import EventLog
from prosit import SimulatorParameters, SimulatorEngine
from prosit_compat import prepare_log_for_prosit
from datetime import datetime
import numpy as np

import random
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


if __name__ == "__main__":
    # Load log
    # download data at: https://github.com/franvinci/prosit-tool/blob/main/example_data/purchasing.xe
    log = xes_importer.apply("data/purchasing.xes")
    prepare_log_for_prosit(log)

    # Split: use 80% for discovery, compare simulation against remaining 20%
    n_cases = len(log)
    split_index = int(n_cases * 0.8)
    train_log = EventLog(log[:split_index])
    test_log = EventLog(log[split_index:])

    # Discover from training set
    net, im, fm = pm4py.discover_petri_net_inductive(train_log)
    params = SimulatorParameters(net, im, fm)
    params.discover_from_eventlog(
        train_log,
        max_depth_tree=3,
        min_samples_leaf_cv=[50, 100, 200],
        random_state=RANDOM_SEED,
        verbose=True
    )

    # Simulate the same number of cases as the test set
    engine = SimulatorEngine(params)
    sim_log = engine.apply(
        n_traces=len(test_log),
        t_start=datetime(2024, 1, 1, 8, 0, 0)
    )

    print(f"Simulated {len(sim_log)} events across {sim_log['case:concept:name'].nunique()} cases")