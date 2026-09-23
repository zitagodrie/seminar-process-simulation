import warnings
warnings.filterwarnings("ignore")

import pm4py
import pm4py.objects.log.importer.xes.importer as xes_importer
from prosit import SimulatorParameters, SimulatorEngine
from datetime import datetime

# Load log
log = xes_importer.apply("data/logs/purchasing.xes")

# Split: use 80% for discovery, compare simulation against remaining 20%
n_cases = len(log)
train_log = log[:int(n_cases * 0.8)]
test_log  = log[int(n_cases * 0.8):]

# Discover from training set
net, im, fm = pm4py.discover_petri_net_inductive(train_log)
params = SimulatorParameters(net, im, fm)
params.discover_from_eventlog(
    train_log,
    max_depth_tree=3,
    min_samples_leaf_cv=[50, 100, 200],
    random_state=42,
    verbose=True
)

# Simulate the same number of cases as the test set
engine = SimulatorEngine(params)
sim_log = engine.apply(
    n_traces=len(test_log),
    t_start=datetime(2024, 1, 1, 8, 0, 0)
)

print(f"Simulated {len(sim_log)} events across {sim_log['case:concept:name'].nunique()} cases")