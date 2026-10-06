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

def print_param_details(params):
    """Print the discovered simulator parameters for experiment inspection."""
    # Resources discovered from the log
    print(params.resources)

    # Working calendar per resource (weekday -> hour -> bool)
    print(params.calendars[params.resources[0]])

    # Per-resource maximum concurrency (1 = no multitasking, >1 = multitasking)
    print(params.max_concurrency)

    # Which resources can perform each activity
    print(params.act_to_resources)

    # Whether rules mode (Decision Trees) is active
    print(params.rules_mode)

    # Arrival time model (DecisionRules in rules mode, distribution tuple in no-rules mode)
    print(params.arrival_time_distribution)

    # Execution time model per activity (DecisionRules or distribution tuple)
    print(params.execution_time_distributions)

    # Waiting time model per resource (DecisionRules or distribution tuple)
    print(params.waiting_time_distributions)

    # Resource selection: flat dict {resource: DecisionRules|float}. One binary
    # classifier per resource, trained on the events where the resource was
    # eligible (activity's candidate pool). At simulation time, the engine first
    # filters resources via `act_to_resources[activity]`, scores each enabled
    # resource with its own tree, and samples proportionally.
    print(params.resource_weights)

    # Control flow model per transition (DecisionRules in rules mode, float frequency in no-rules)
    print(params.transition_weights)

    # Case-level data attribute distribution (None if no attributes in log)
    print(params.distribution_data_attributes)
    # {'mode': 'empirical', 'data': {(val1, val2): frequency, ...}}
    # or {'mode': 'distribution', 'data': {attr: {'type': 'categorical'|'continuous', ...}}}

def main():
    log = load_dataset(DATASET)
    train_log, test_log = split_log(log)

    # Discover from training set
    net, im, fm = pm4py.discover_petri_net_inductive(train_log)
    params = SimulatorParameters(net, im, fm)
    params.discover_from_eventlog(train_log)

    print_param_details(params)

    # Simulate the same number of cases as the test set
    engine = SimulatorEngine(params)
    sim_log = engine.apply(
        n_traces=len(test_log),
        t_start=datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    )

    print(f"Simulated {len(sim_log)} events across {sim_log['case:concept:name'].nunique()} cases")

if __name__ == "__main__":
    main()
