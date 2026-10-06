from collections import Counter
from datetime import datetime, timezone
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import pm4py
import pm4py.objects.log.importer.xes.importer as xes_importer
from pm4py.objects.log.obj import EventLog
from prosit import SimulatorEngine, SimulatorParameters

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings

warnings.filterwarnings("ignore")

DATASET = "p2p"

@dataclass(frozen=True)
class DatasetConfig:
    """Describe how one XES dataset should be loaded and normalized."""

    path: str
    resource_feature: str | None = None
    normalize_resource_names: bool = False


DATASETS = {
    "cvs": DatasetConfig(
        "data/CVS/cvs_pharmacy.xes", 
        resource_feature="org:resource"
    ),
    "bpi2012": DatasetConfig(
        "data/BPI Challenge 2012/BPI_Challenge_2012_W_Two_TS.xes",
        resource_feature="Resource"
    ),
    "bpi2017": DatasetConfig(
        "data/BPI Challenge 2017/BPI_Challenge_2017_W_Two_TS.xes",
        resource_feature="Resource"
    ),
    "acr": DatasetConfig(
        "data/reproducibility_package/event_logs/public_real_logs/ConsultaDataMining201618/ConsultaDataMining201618.xes",
        normalize_resource_names=True,
    ),
    "p2p": DatasetConfig("data/Purchase to Pay (P2P)/PurchasingExample.xes"),
}


def clean_timestamp_features(log):
    """Add start timestamps and remove event fields Prosit cannot discover."""
    # Prosit 1.0.3 treats unknown event columns as case attributes and later
    # reads them from the first event of every trace. These CVS fields are
    # event-level metadata and are missing from some first events, so keeping
    # them causes discovery to fail with a KeyError.
    event_attributes_to_ignore = {
        "elementId",
        "lifecycle:transition",
        "processId",
        "resourceCost",
        "resourceId",
    }

    for trace in log:
        for event in trace:
            if "start:timestamp" not in event:
                event["start:timestamp"] = event["time:timestamp"]
            for attribute in event_attributes_to_ignore:
                if attribute in event:
                    del event[attribute]
    # Keep org:resource: Prosit uses this standard XES attribute for resource
    # discovery and for mapping resources to activities.
    return log


def normalize_resource_names(log):
    """Prefix ACR resource names that are actually activity names."""
    activity_names = {
        event.get("concept:name")
        for trace in log
        for event in trace
        if event.get("concept:name") is not None
    }
    for trace in log:
        for event in trace:
            if event.get("org:resource") in activity_names:
                event["org:resource"] = f"resource:{event['org:resource']}"
    return log


def add_missing_values(log, feature):
    """Fill an absent event attribute with a stable categorical value."""

    # If 'org:resource'  not assigned, then add 'missing'
    for trace in log:
        for event in trace:
            if feature not in event:
                event[feature] = "missing"

    return log

def load_dataset(dataset_name):
    """Load and normalize a configured dataset by name.

    Args:
        dataset_name: Key from :data:`DATASETS`.

    Raises:
        KeyError: If ``dataset_name`` is not configured.
    """
    config = DATASETS[dataset_name]
    log = xes_importer.apply(config.path)
    log = clean_timestamp_features(log)
    if config.normalize_resource_names:
        log = normalize_resource_names(log)
    if config.resource_feature:
        log = add_missing_values(log, config.resource_feature)
    return log


def split_log(log, train_fraction=0.8):
    """Split an event log chronologically by trace into train and test logs."""
    split_index = int(len(log) * train_fraction)
    return EventLog(log[:split_index]), EventLog(log[split_index:])


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

# ============================================================
# Helper functions for NGD, CTD and CAR
# ============================================================

def _get_case_id(event):
    """Get case ID from a PM4Py event or pandas row."""
    if isinstance(event, dict):
        return event.get("case:concept:name")
    return event["case:concept:name"]


def _get_activity(event):
    """Get activity name from a PM4Py event or pandas row."""
    if isinstance(event, dict):
        return event.get("concept:name")
    return event["concept:name"]


def _get_timestamp(event, column):
    """Get timestamp from a PM4Py event or pandas row."""
    if isinstance(event, dict):
        return event.get(column)
    return event[column]


def _trace_activity_sequences(log):
    """
    Return activity sequences for all cases.

    Supports:
      - PM4Py EventLog
      - pandas DataFrame
    """

    if isinstance(log, pd.DataFrame):
        df = log.copy()

        case_col = "case:concept:name"
        activity_col = "concept:name"

        # Try to recover chronological ordering.
        if "time:timestamp" in df.columns:
            df = df.sort_values([case_col, "time:timestamp"])
        elif "start_timestamp" in df.columns:
            df = df.sort_values([case_col, "start_timestamp"])
        elif "start:timestamp" in df.columns:
            df = df.sort_values([case_col, "start:timestamp"])

        return [
            group[activity_col].tolist()
            for _, group in df.groupby(case_col, sort=False)
        ]

    # PM4Py EventLog
    sequences = []

    for trace in log:
        events = list(trace)

        # Sort by timestamp if available.
        if events and "time:timestamp" in events[0]:
            events = sorted(
                events,
                key=lambda e: pd.Timestamp(e["time:timestamp"])
            )

        sequences.append([
            e["concept:name"]
            for e in events
        ])

    return sequences


def ngd(log1, log2, n=3):
    """
    N-Gram Distance (NGD).

    Follows Chapela-Campa et al.:
      - convert every trace into an activity sequence
      - add n-1 dummy activities at start and end
      - count n-grams
      - sum absolute frequency differences
      - divide by total n-gram frequencies

    uses N=3.
    """

    traces1 = _trace_activity_sequences(log1)
    traces2 = _trace_activity_sequences(log2)

    # Dummy symbol must not collide with an actual activity.
    START = "__START__"
    END = "__END__"

    def make_histogram(traces):
        histogram = Counter()

        for trace in traces:
            padded = (
                [START] * (n - 1)
                + trace
                + [END] * (n - 1)
            )

            for i in range(len(padded) - n + 1):
                gram = tuple(padded[i:i+n])
                histogram[gram] += 1

        return histogram

    h1 = make_histogram(traces1)
    h2 = make_histogram(traces2)

    all_ngrams = set(h1) | set(h2)

    numerator = sum(
        abs(h1.get(gram, 0) - h2.get(gram, 0))
        for gram in all_ngrams
    )

    denominator = sum(h1.values()) + sum(h2.values())

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _case_cycle_times(log):
    """
    Extract end-to-end cycle times in seconds.

    For each case:
        cycle time = latest end timestamp - earliest start timestamp

    Works with:
      - PM4Py EventLog
      - pandas simulation DataFrame

    The simulator's column names can be adapted here if necessary.
    """

    cycle_times = []

    if isinstance(log, pd.DataFrame):
        df = log.copy()
        case_col = "case:concept:name"

        # Determine start/end timestamp columns.
        if "start:timestamp" in df.columns:
            start_col = "start:timestamp"
        elif "start_timestamp" in df.columns:
            start_col = "start_timestamp"
        elif "time:timestamp" in df.columns:
            start_col = "time:timestamp"
        else:
            raise ValueError(
                "Could not find start timestamp column."
            )

        if "end:timestamp" in df.columns:
            end_col = "end:timestamp"
        elif "end_timestamp" in df.columns:
            end_col = "end_timestamp"
        elif "time:timestamp" in df.columns:
            end_col = "time:timestamp"
        else:
            raise ValueError(
                "Could not find end timestamp column."
            )

        df[start_col] = pd.to_datetime(df[start_col])
        df[end_col] = pd.to_datetime(df[end_col])

        for _, group in df.groupby(case_col):
            start = group[start_col].min()
            end = group[end_col].max()

            if pd.notna(start) and pd.notna(end):
                cycle_times.append(
                    (end - start).total_seconds()
                )

    else:
        # PM4Py EventLog
        for trace in log:
            starts = []
            ends = []

            for event in trace:
                # Prefer explicit start timestamp when available.
                start = event.get(
                    "start:timestamp",
                    event.get("time:timestamp")
                )

                end = event.get(
                    "end:timestamp",
                    event.get("time:timestamp")
                )

                if start is not None:
                    starts.append(pd.Timestamp(start))

                if end is not None:
                    ends.append(pd.Timestamp(end))

            if starts and ends:
                cycle_times.append(
                    (max(ends) - min(starts)).total_seconds()
                )

    return np.asarray(cycle_times, dtype=float)


def ctd(log1, log2):
    """
    Cycle Time Distribution distance.

    The paper defines CTD as the 1-Wasserstein distance
    between the empirical distributions of case cycle times.
    """

    ct1 = _case_cycle_times(log1)
    ct2 = _case_cycle_times(log2)

    if len(ct1) == 0 or len(ct2) == 0:
        return np.nan

    return float(
        wasserstein_distance(ct1, ct2)
    )


def _case_arrival_times(log):
    """
    Return the earliest start timestamp of every case.
    """

    arrivals = []

    if isinstance(log, pd.DataFrame):
        df = log.copy()
        case_col = "case:concept:name"

        if "start:timestamp" in df.columns:
            timestamp_col = "start:timestamp"
        elif "start_timestamp" in df.columns:
            timestamp_col = "start_timestamp"
        elif "time:timestamp" in df.columns:
            timestamp_col = "time:timestamp"
        else:
            raise ValueError(
                "Could not find timestamp column for CAR."
            )

        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        arrivals = (
            df.groupby(case_col)[timestamp_col]
            .min()
            .dropna()
            .tolist()
        )

    else:
        for trace in log:
            timestamps = []

            for event in trace:
                ts = event.get(
                    "start:timestamp",
                    event.get("time:timestamp")
                )

                if ts is not None:
                    timestamps.append(pd.Timestamp(ts))

            if timestamps:
                arrivals.append(min(timestamps))

    return pd.to_datetime(arrivals)


def _hourly_arrival_histogram(arrivals, start, end):
    """
    Bin case arrivals into one-hour bins.

    The paper uses hourly bins for CAR.
    """

    start = pd.Timestamp(start).floor("h")
    end = pd.Timestamp(end).ceil("h")

    # Include the final hour.
    bins = pd.date_range(
        start=start,
        end=end + pd.Timedelta(hours=1),
        freq="h"
    )

    counts = (
        pd.Series(1, index=pd.to_datetime(arrivals))
        .groupby(pd.Grouper(freq="h"))
        .sum()
    )

    # Reindex onto common hourly timeline.
    counts = counts.reindex(
        bins[:-1],
        fill_value=0
    )

    return counts.to_numpy(dtype=float)


def _emd_1d_equal_bins(x, y):
    """
    1-D Earth Mover's Distance for equally spaced bins.

    For the Wildschut/Godrie replication, the test and simulated
    logs contain the same number of cases, so the total mass of
    the two CAR histograms is equal.

    For equal total mass, the EMD is equivalent to the
    1-Wasserstein distance on the histogram bins.

    The result is expressed in number of hourly bins.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("Histograms must have equal length.")

    total_x = x.sum()
    total_y = y.sum()

    if total_x == 0 and total_y == 0:
        return 0.0

    if not np.isclose(total_x, total_y):
        raise ValueError(
            "CAR histograms have different total mass. "
            "The replication assumes the same number of cases."
        )

    if total_x == 0:
        return 0.0

    # Earth mover's distance on a 1-D equally spaced grid:
    # sum of absolute cumulative mass differences.
    cumulative_difference = np.cumsum(
        x / total_x - y / total_y
    )

    return float(
        np.sum(np.abs(cumulative_difference))
    )


def car(log1, log2):
    """
    Case Arrival Rate (CAR).

    Cases are represented by their earliest start timestamp.
    Arrivals are binned hourly and compared using EMD.
    """

    arrivals1 = _case_arrival_times(log1)
    arrivals2 = _case_arrival_times(log2)

    if len(arrivals1) == 0 or len(arrivals2) == 0:
        return np.nan

    # Use one common timeline for both logs.
    start = min(arrivals1.min(), arrivals2.min())
    end = max(arrivals1.max(), arrivals2.max())

    h1 = _hourly_arrival_histogram(
        arrivals1,
        start,
        end
    )

    h2 = _hourly_arrival_histogram(
        arrivals2,
        start,
        end
    )

    return _emd_1d_equal_bins(h1, h2)


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
    test_arrivals = _case_arrival_times(test_log)

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