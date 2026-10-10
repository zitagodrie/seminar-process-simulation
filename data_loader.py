from dataclasses import dataclass
import os
import pandas as pd
import pm4py
import pm4py.objects.log.importer.xes.importer as xes_importer
from pm4py.objects.log.obj import EventLog
from prosit import SimulatorEngine, SimulatorParameters

@dataclass(frozen=True)
class DatasetConfig:
    """Describe how one XES dataset should be loaded and normalized."""

    path: str
    pm_path: str | None = None
    split_from_file: bool = False
    resource_feature: str | None = None
    activity_feature: str = "Activity"
    normalize_resource_names: bool = False

DATASETS = {
    "cvs": DatasetConfig(
        "data/CVS/cvs_pharmacy.xes",
        pm_path="bpmn/cvs_pharmacy/cvs_pharmacy.bpmn",
        resource_feature="org:resource",
        activity_feature="concept:name",
    ),
    "bpi2012": DatasetConfig(
        "data/BPI Challenge 2012/BPI_Challenge_2012_W_Two_TS.xes",
        pm_path="bpmn/BPI_Challenge_2012_W_Two_TS/BPI_Challenge_2012_W_Two_TS.bpmn",
        split_from_file=True,
        resource_feature="Resource"
    ),
    "bpi2017": DatasetConfig(
        "data/BPI Challenge 2017/BPI_Challenge_2017_W_Two_TS.xes",
        pm_path="bpmn/BPI_Challenge_2017_W_Two_TS/BPI_Challenge_2017_W_Two_TS.bpmn", 
        split_from_file=True,
        resource_feature="Resource"
    ),
    "acr": DatasetConfig(
        "data/ACR/ConsultaDataMining201618.xes",
        pm_path="bpmn/ConsultaDataMining201618/ConsultaDataMining201618.bpmn",
        split_from_file=True,
        normalize_resource_names=True,
    ),
    "p2p": DatasetConfig(
        "data/P2P/PurchasingExample.xes",
        pm_path="bpmn/PurchasingExample/PurchasingExample.bpmn",
        split_from_file=True,
    ),
}

def read_case_ids(csv_path):
    """Read the case IDs stored in a CSV split file."""
    csv_df = pd.read_csv(csv_path)
    return set(csv_df["Case ID"].astype(str).str.strip())


def filter_log_by_case_ids(log, case_ids):
    """Keep only traces whose case identifier is in the provided set."""
    return EventLog(
        [
            trace
            for trace in log
            if str(trace.attributes.get("concept:name")).strip() in case_ids
        ]
    )


def create_data_split(log, csv_path):
    """Build a log subset from the case IDs in a CSV split file."""
    case_ids = read_case_ids(csv_path)
    return filter_log_by_case_ids(log, case_ids)


def get_split_paths(data_set):
    """Return the train/test CSV paths for a dataset that stores splits on disk."""
    data_path = DATASETS[data_set].path
    directory_path = os.path.dirname(data_path)
    return (
        os.path.join(directory_path, "folds/train.csv"),
        os.path.join(directory_path, "folds/test.csv"),
    )


def split_log_from_file(log, data_set):
    """Create train/test logs from dataset split CSV files."""
    train_path, test_path = get_split_paths(data_set)
    return create_data_split(log, train_path), create_data_split(log, test_path)


def train_test_split(log, data_set):
    """Return train and test logs using either CSV-based or chronological splitting."""
    if DATASETS[data_set].split_from_file:
        print(f"Splitting {data_set} log using CSV files")
        return split_log_from_file(log, data_set)
    return split_log(log)

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
