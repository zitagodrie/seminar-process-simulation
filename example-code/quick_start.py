import warnings
warnings.filterwarnings("ignore")

import pm4py
import pm4py.objects.log.importer.xes.importer as xes_importer
from prosit import SimulatorParameters, SimulatorEngine

# 1. Load event log
log = xes_importer.apply("data/logs/purchasing.xes")

# 2. Discover Petri net
net, im, fm = pm4py.discover_petri_net_inductive(log)

# 3. Create and populate simulation parameters
params = SimulatorParameters(net, im, fm)
params.discover_from_eventlog(log, max_depth_tree=3)

# 4. Save for later reuse
params.to_json("params_purchasing.json")

# 5. Simulate 500 cases
sim_engine = SimulatorEngine(params)
sim_log = sim_engine.apply(n_traces=500)

print(sim_log.head())