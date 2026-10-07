"""Run process simulation experiments for the available event logs.

The dataset-specific details are kept in :class:`DatasetConfig` so adding a
new log does not require another branch in the main simulation workflow.
"""

import random
from datetime import datetime, timezone
import pm4py
import pm4py.objects.log.importer.xes.importer as xes_importer
from prosit import SimulatorEngine, SimulatorParameters
from data_loader import *
from table1metrics import calculate_table1_metrics

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Specify which dataset to run the simulation on. Choose from the keys of `DATASETS`.
DATASET = "p2p"


def main():
    log = load_dataset(DATASET)
    train_log, test_log = split_log(log)

    # Discover from training set
    net, im, fm = pm4py.discover_petri_net_inductive(train_log)
    params = SimulatorParameters(net, im, fm)
    params.discover_from_eventlog(train_log)

    # Simulate the same number of cases as the test set
    engine = SimulatorEngine(params)

    
    sim_log = engine.apply(
        n_traces=len(test_log),
        t_start=datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    )

    print(f"Simulated {len(sim_log)} events across {sim_log['case:concept:name'].nunique()} cases")

    # calculate_table1_metrics(train_log, test_log, engine)


if __name__ == "__main__":
    main()
