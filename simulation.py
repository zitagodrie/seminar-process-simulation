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
from table1metrics import calculate_table1_metrics, case_arrival_times
from table4metrics import average_execution_time_entropy

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

N_SIM = {"none": 0, "fast": 1, "realistic": 5}  # number of simulation runs for each mode, realistic is the paper of the setting

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
        "-s",
        "--simulations",
        type=str,
        default="none",
        choices=N_SIM.keys(),
        help="Simulation mode for the experiment.",
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

def write_metrics_to_csv(metric_avg: dict, args, path: str = "metrics.csv") -> None:
    """Append averaged metrics, run arguments, and a timestamp to a CSV file.

    The result is one row per call: metrics, then args, then timestamp.
    """
    row = {
        "timestamp": datetime.now().isoformat(),
        **vars(args),
        **metric_avg,
    }
    file_exists = os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not file_exists, index=False)
    print(f"Metrics written to {path}")

def simulate_process_with_metrics(engine, test_log, args):
    results = []
    simulated_logs = []

    # Use the beginning of the TEST log as simulation start.
    # This is important for CAR.
    test_arrivals = case_arrival_times(test_log)

    if len(test_arrivals) == 0:
        raise ValueError("Could not determine test-log case arrival times.")

    for run in range(N_SIM[args.simulations]):

        print(f"\n========== Simulation {run + 1}/{N_SIM[args.simulations]} ==========")

        # run simulation
        sim_log = simulate_process(engine, test_log)

        # Calculate the metrics
        ngd_value, ctd_value, car_value = calculate_table1_metrics(test_log, sim_log)
        # add table 2/3 metrics here
        entropy_value = average_execution_time_entropy(sim_log, activity_col=DATASETS[args.dataset].activity_feature)

        result = {
            "run": run + 1,
            "NGD": ngd_value,
            "CTD": ctd_value,
            "CAR": car_value,
            # add table 2/3 metrics here
            "Execution Time Entropy": entropy_value
        }
        results.append(result)
        simulated_logs.append(sim_log)

    # Store results
    metrics_df = pd.DataFrame(results)

    averages = {
        "NGD": metrics_df["NGD"].mean(),
        "CTD": metrics_df["CTD"].mean(),
        "CAR": metrics_df["CAR"].mean(),
        #add table 2/3 metrics here
        "Execution Time Entropy": metrics_df["Execution Time Entropy"].mean()
    }

    write_metrics_to_csv(averages, args)


    print("\n========== Average metrics ==========")
    print(f"NGD: {averages['NGD']:.4f}")
    print(f"CTD: {averages['CTD']:.2f}")
    print(f"CAR: {averages['CAR']:.2f}")
    print(f"Execution Time Entropy: {averages['Execution Time Entropy']:.4f}")

    # Return
    return {
        "simulated_logs": simulated_logs,
        "metrics": metrics_df,
        "average_metrics": averages
    }

def main():
    args = parse_args()
    log = load_dataset(args.dataset)
    train_log, test_log = train_test_split(log, args.dataset)

    # Discover from training set
    net, im, fm = pm4py.discover_petri_net_inductive(train_log)
    params = SimulatorParameters(net, im, fm)
    params.discover_from_eventlog(train_log)

    # Simulate the same number of cases as the test set
    engine = SimulatorEngine(params)

    if N_SIM[args.simulations] > 0:
        simulate_process_with_metrics(engine, test_log, args)

if __name__ == "__main__":
    main()
