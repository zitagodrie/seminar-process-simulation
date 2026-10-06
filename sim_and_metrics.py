
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


# Main replication function
def main(dataset_name, n_simulations=1, random_seed=42):

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