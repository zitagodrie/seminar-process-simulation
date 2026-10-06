
from datetime import datetime, timezone
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pm4py
import pm4py.objects.log.importer.xes.importer as xes_importer
from pm4py.objects.log.obj import EventLog
from prosit import SimulatorEngine, SimulatorParameters

from table1metrics import *
from data_loader import *

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings

warnings.filterwarnings("ignore")


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


# Main replication function
def do_everything(dataset_name, n_simulations=1, random_seed=42):

    # Load event log
    log = load_dataset(dataset_name)

    # Train/test split
    train_log, test_log = split_log(log)

    # Discover Petri net
    net, im, fm = pm4py.discover_petri_net_inductive(train_log)

    # Discover simulation parameters
    params = SimulatorParameters(net, im, fm)

    params.discover_from_eventlog(
        train_log,
        max_depth_tree=3,
        min_samples_leaf_cv=[50, 100, 200],
        random_state=random_seed,
        verbose=True
    )

    print_param_details(params)

    
    # Run simulations and calculate metrics
    engine = SimulatorEngine(params)

    results = []
    simulated_logs = []

    # Use the beginning of the TEST log as simulation start.
    # This is important for CAR.
    test_arrivals = case_arrival_times(test_log)

    if len(test_arrivals) == 0:
        raise ValueError("Could not determine test-log case arrival times.")

    simulation_start = test_arrivals.min()

    for run in range(n_simulations):

        print(f"\n========== Simulation {run + 1}/{n_simulations} ==========")

        # run simulation
        sim_log = engine.apply(n_traces=len(test_log), t_start=simulation_start)

        # Calculate the three paper metrics
        ngd_value = ngd(test_log, sim_log, n=3)
        print(f"NGD = {ngd_value:.4f}")
    
        ctd_value = ctd(test_log, sim_log)
        print(f"CTD = {ctd_value:.2f}")
        
        car_value = car(test_log, sim_log)
        print(f"CAR = {car_value:.2f}")

        result = {
            "run": run + 1,
            "NGD": ngd_value,
            "CTD": ctd_value,
            "CAR": car_value
        }
        results.append(result)
        simulated_logs.append(sim_log)

    # Store results
    metrics_df = pd.DataFrame(results)

    averages = {
        "NGD": metrics_df["NGD"].mean(),
        "CTD": metrics_df["CTD"].mean(),
        "CAR": metrics_df["CAR"].mean()
    }

    print("\n========== Average metrics ==========")
    print(f"NGD: {averages['NGD']:.4f}")
    print(f"CTD: {averages['CTD']:.2f}")
    print(f"CAR: {averages['CAR']:.2f}")

    # Return
    return {
        "params": params,
        "simulated_logs": simulated_logs,
        "metrics": metrics_df,
        "average_metrics": averages,
        "train_log": train_log,
        "test_log": test_log
    }