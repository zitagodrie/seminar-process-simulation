"""Run process simulation experiments for the available event logs.

The dataset-specific details are kept in :class:`DatasetConfig` so adding a
new log does not require another branch in the main simulation workflow.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings

warnings.filterwarnings("ignore")

import random
from dataclasses import dataclass
from datetime import datetime, timezone

import pm4py
import pm4py.objects.log.importer.xes.importer as xes_importer
from pm4py.objects.log.obj import EventLog
from prosit import SimulatorEngine, SimulatorParameters

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Specify which dataset to run the simulation on. Choose from the keys of `DATASETS`.
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
        "data/ACR/ConsultaDataMining201618.xes",
        normalize_resource_names=True,
    ),
    "p2p": DatasetConfig("data/P2P/PurchasingExample.xes"),
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

if __name__ == "__main__":
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