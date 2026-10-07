"""Run process simulation experiments for the available event logs.

The dataset-specific details are kept in :class:`DatasetConfig` so adding a
new log does not require another branch in the main simulation workflow.
"""

import random
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description="Run a process simulation experiment.")
    parser.add_argument(
        "-d",
        "--dataset",
        choices=DATASETS,
        required=True,
        help="Dataset to simulate.",
    )
    parser.add_argument(
        "-cm",
        "--calculate-metrics",
        dest="calculate_metrics",
        action="store_true",
        help="Calculate metrics after the simulation.",
    )
    return parser.parse_args()


def simulate_process(engine, test_log, verbose=False):
    sim_log = engine.apply(
        n_traces=len(test_log),
        t_start=datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    )
    if verbose:
        print(f"Simulated {len(sim_log)} events across {sim_log['case:concept:name'].nunique()} cases")

    return sim_log


def main():
    args = parse_args()
    log = load_dataset(args.dataset)
    train_log, test_log = split_log(log)

    # Discover from training set
    net, im, fm = pm4py.discover_petri_net_inductive(train_log)
    params = SimulatorParameters(net, im, fm)
    params.discover_from_eventlog(train_log)

    # Simulate the same number of cases as the test set
    engine = SimulatorEngine(params)

    if args.calculate_metrics:
        calculate_table1_metrics(train_log, test_log, engine)
    else:
        # test a simulation anyways, if metrics are not being calculated
        sim_log = simulate_process(engine, test_log)


if __name__ == "__main__":
    main()
