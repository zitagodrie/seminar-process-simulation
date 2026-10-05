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
import pm4py.objects.conversion.log.variants.to_event_stream as to_event_stream

from prosit import SimulatorParameters, SimulatorEngine
from datetime import datetime
import numpy as np

import random
RANDOM_SEED = 42
random.seed(RANDOM_SEED)




# def petrinet_from_eventlog_with_IM(data_path):
#     log = xes_importer.apply(data_path)

#     n_cases = len(log)
#     split_index = int(n_cases * 0.8)
#     train_log = EventLog(log[:split_index])
#     test_log = EventLog(log[split_index:])

#     net, im, fm = pm4py.discover_petri_net_inductive(train_log)

#     return net, im, fm

def clean_timestamp_features(log):
    """
    Normalize CVS event attributes for compatibility with Prosit.
    """
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

def add_missing_values(log):

    # If 'org:resource'  not assigned, then add 'missing'
    for trace in log:
        for event in trace:
            if "org:resource" not in event:
                event["org:resource"] = "missing"

    return log

def print_param_details(params):
        # Resources discovered from the log
    print(params.resources)

    # Working calendar per resource (weekday -> hour -> bool)
    print(params.calendars["Technician-000002"])

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
    log = xes_importer.apply("data/CVS/cvs_pharmacy.xes")
    # take only the first 1000 traces for faster testing
    log = log[:100]
    log = clean_timestamp_features(log)
    log = add_missing_values(log)
    # # inspect the log by printing the first 5 traces
    # for trace in log[:5]:
    #     print(trace)

    # # print the attributes together with the types
    # print("Attributes and types:")
    # for attr in log[0][0].keys():
    #     print(f"{attr}: {type(log[0][0][attr])}")

    # Split: use 80% for discovery, compare simulation against remaining 20%
    n_cases = len(log)
    split_index = int(n_cases * 0.8)
    train_log = EventLog(log[:split_index])
    test_log = EventLog(log[split_index:])

    # Discover from training set
    net, im, fm = pm4py.discover_petri_net_inductive(train_log)
    params = SimulatorParameters(net, im, fm)
    params.discover_from_eventlog(train_log)

    print_param_details(params)

    # Simulate the same number of cases as the test set
    engine = SimulatorEngine(params)
    sim_log = engine.apply(
        n_traces=len(test_log),
        t_start=datetime(2024, 1, 1, 8, 0, 0)
    )

    print(f"Simulated {len(sim_log)} events across {sim_log['case:concept:name'].nunique()} cases")