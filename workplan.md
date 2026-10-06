## 1. Pin the right version
- The README says **v0.1.0** matches the paper's implementation, while `main` has moved on. `main` adds gamma distributions, CRPS-based CV, per-resource selection classifiers and MAD outlier removal, none of which the paper describes. Install from the v0.1.0 tag for the main replication.
- Optionally run `main` as a secondary comparison.
- Python ≥ 3.10, pm4py, scikit-learn, and the system Graphviz binary.
- **AP** setup environment

## 2. Get the data and control-flow models
- Five logs: P2P (Fluxicon), CVS and ACR (Zenodo 4699983), BPIC12W and BPIC17W (4TU). CVS and P2P are synthetic, and BPIC12W and BPIC17W carry trace attributes. **AP** source all the datasets
- The paper does not specify the petri net mining algorithm used, so we will use the default implementation of the prosit package
    - "The simulator has been implemented as a Python package: the user can manually define each element of the DBPS or can discover one by giving an input event log and/or a Petri net model."
- Do a **temporal** 80/20 split by trace. Discover on the train set and evaluate against the test set.

## 3. Run DBPS with the paper's settings
- CART, 3-fold CV, `max_depth_tree` from 1 to 5. The README default of 5 already searches depths up to that value, but check the `min_samples_leaf_cv` default against the paper, which doesn't mention it.
- Set `use_workload_features` deliberately. The paper says waiting-time trees use resource workload, but the README default is `False`.
- Run `discover_from_eventlog`, then `SimulatorEngine.apply` with `n_traces` equal to the number of traces in the real log. Do this **5 times per log** and average, with varied seeds.

## 4. Build the baselines
- **Simod** [9] is a pip package. Give it the same Petri net if it allows that.
- **DSim** [5] and **RIMS** [17] both have public code. They need deep-learning dependencies and are the hardest to get running.
- **DBPS-det** (for Table 4) replaces the probabilistic trees with deterministic ones. Prosit's `deterministic_time=True` samples the distribution mean, which may not be identical, so check how the authors did it.

## 5. Implement the metrics
- **Table 2 (accuracy):** the three metrics from Chapela-Campa et al. [8], each computed between the simulated log and the real log: 3-gram distance (NGD), cycle time distribution distance (CTD), and case arrival rate distance (CAR). Look for their reference implementation, since the paper doesn't say whether they used it.
- **Table 3 (rule-level):**
  - Wasserstein distance per tree leaf, computed for arrival, execution and waiting times (Rule-ATD, Rule-ETD, Rule-WTD).
  - Absolute difference in transition firing frequencies (Rule-CFD).
  - Compared against Simod only.
  - The paper doesn't fully specify how simulated and real cases are matched to leaves, so we'll need to decide or ask.
- **Table 4 (entropy):** average per-activity entropy of execution times, binned with Sturges' rule (k = ⌈log₂ n⌉ + 1). The paper doesn't say whether the logarithm is natural or base 2, or whether the bin edges are shared with the real log.

## 6. Compare and sanity-check
- Regenerate Tables 2 to 4 and compare against the paper's numbers. Exact matches are unlikely because of stochasticity and version differences. The things to check are the ordering of methods and the rough magnitudes (for example, average NGD of about 0.28 for DBPS against 0.66 for Simod).
- Optionally reproduce the what-if interaction from Sect. 3.4 by editing leaf values in the JSON from `to_json` and re-simulating.

## Things to flag or ask the authors
- The exact control-flow models, the Chapela-Campa metric configuration, the rule-metric matching, and the seeds.
- Whether the "same number of traces as the real ones" means the test set or the full log.
- The README says training uses 80% of the traces, and I'd want to confirm that the evaluation reference is the 20% test set.

_Disclaimer: This workplan was made with the help of a LLM._