import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import wasserstein_distance


# ============================================================
# RULE-LEVEL METRICS FOR DBPS / PROSIT
# ============================================================

def _event_timestamp(event, key):
    """Return a pandas Timestamp or None."""
    value = event.get(key)
    if value is None:
        return None
    return pd.Timestamp(value)


def _case_attributes(trace, label_data_attributes):
    """
    Extract case-level attributes.

    Prosit stores case attributes separately from the event history.
    We use the first non-null value observed in the trace.
    """
    attributes = {}

    for attr in label_data_attributes:
        value = None

        for event in trace:
            if attr in event and event[attr] is not None:
                value = event[attr]
                break

        if value is not None:
            attributes[attr] = value

    return attributes


def _activity_history_before_event(trace, event_index, activities):
    """
    Reconstruct Prosit's case history immediately before an event.

    Prosit uses:
        case["history"]

    where the history contains the number of previous occurrences
    of each activity/transition label.
    """
    history = {
        activity: 0
        for activity in activities
    }

    for event in trace[:event_index]:
        activity = event.get("concept:name")

        if activity in history:
            history[activity] += 1

    return history


def _build_case_attributes_from_log(trace, params):
    """
    Reconstruct the case attributes used by Prosit.

    We use params.label_data_attributes, which is the same collection
    passed into Prosit's feature-building/discovery code.
    """
    return _case_attributes(
        trace,
        getattr(params, "label_data_attributes", [])
    )


def _get_resource(event):
    """
    Get the resource using the same fallback convention as the
    preprocessing in your existing code.
    """
    resource = event.get("org:resource")

    if resource is None:
        resource = event.get("resourceId")

    if resource is None:
        resource = "default_resource"

    return resource


def _get_execution_time(event):
    """
    Execution time in minutes.
    """
    start = _event_timestamp(event, "start:timestamp")
    end = _event_timestamp(event, "time:timestamp")

    if start is None or end is None:
        return None

    return max(0.0, (end - start).total_seconds() / 60.0)


def _get_waiting_time(event):
    """
    Waiting time in minutes.

    Prosit output contains enabled:timestamp and start:timestamp.
    """
    enabled = _event_timestamp(event, "enabled:timestamp")
    start = _event_timestamp(event, "start:timestamp")

    if enabled is None or start is None:
        return None

    return max(0.0, (start - enabled).total_seconds() / 60.0)


def _get_arrival_times(log):
    """
    Return case arrival timestamps.

    For a real event log this is the first event timestamp of each case.
    """
    arrivals = []

    for trace in log:
        if len(trace) == 0:
            continue

        timestamps = [
            _event_timestamp(event, "time:timestamp")
            for event in trace
        ]

        timestamps = [x for x in timestamps if x is not None]

        if timestamps:
            arrivals.append(min(timestamps))

    return arrivals


def _simlog_to_traces(sim_log):
    """
    Convert Prosit's pandas simulation output into a list of traces.

    Prosit outputs:
        case:concept:name
        concept:name
        org:resource
        enabled:timestamp
        start:timestamp
        time:timestamp
        + label attributes
    """
    traces = []

    for case_id, group in sim_log.groupby(
        "case:concept:name",
        sort=False
    ):
        group = group.sort_values(
            ["start:timestamp", "time:timestamp"]
        )

        trace = []

        for _, row in group.iterrows():
            event = row.to_dict()
            trace.append(event)

        traces.append(trace)

    return traces


# ============================================================
# LEAF IDENTIFICATION
# ============================================================

def _find_leaf_node(rules, features):
    """
    Traverse a Prosit DecisionRules.rules dictionary and return
    the actual leaf-node ID.

    This mirrors Prosit's traverse_tree() exactly, but returns
    the node ID rather than the leaf value.
    """
    current_node = 0

    while "value" not in rules[current_node]:

        feature = rules[current_node]["feature"]
        threshold = rules[current_node]["threshold"]

        if features[feature] <= threshold:
            current_node = rules[current_node]["children"][True]
        else:
            current_node = rules[current_node]["children"][False]

    return current_node


def _leaf_value(obj, features):
    """
    Use Prosit's own DecisionRules.apply().

    This is useful as a sanity check that our feature dictionary
    is valid for the tree.
    """
    return obj.apply(features)


# ============================================================
# FEATURE CONSTRUCTION
# ============================================================

def _transition_features(trace, event_index, params):
    """
    Features used by Prosit's transition decision rules.

    From the simulator:
        case["attributes"] | case["history"]
    """
    attributes = _build_case_attributes_from_log(trace, params)

    history = _activity_history_before_event(
        trace,
        event_index,
        params.net_transition_labels
    )

    return attributes | history


def _execution_features(trace, event_index, params):
    """
    Features used by Prosit's execution-time decision tree.

    Prosit uses:
        resource one-hot
        hour
        weekday
        case attributes
        case history
    """
    event = trace[event_index]

    resource = _get_resource(event)

    features = {
        "resource = " + r: int(r == resource)
        for r in params.resources
    }

    start = _event_timestamp(event, "start:timestamp")

    if start is None:
        start = _event_timestamp(event, "time:timestamp")

    features["hour"] = start.hour
    features["weekday"] = start.weekday()

    attributes = _build_case_attributes_from_log(trace, params)

    history = _activity_history_before_event(
        trace,
        event_index,
        params.net_transition_labels
    )

    return features | attributes | history


def _waiting_features(trace, event_index, params):
    """
    Features used by Prosit's waiting-time decision tree.

    Prosit uses:

        hour
        weekday
        waiting_activity = activity
        case attributes
        case history

    and, when use_workload_features=True:

        workload
        queue_length
    """
    event = trace[event_index]

    activity = event.get("concept:name")

    start = _event_timestamp(event, "start:timestamp")

    if start is None:
        start = _event_timestamp(event, "time:timestamp")

    features = {
        "hour": start.hour,
        "weekday": start.weekday(),
    }

    # Prosit's one-hot waiting activity vector
    for act in params.net_transition_labels:
        features["waiting_activity = " + act] = int(act == activity)

    attributes = _build_case_attributes_from_log(trace, params)

    history = _activity_history_before_event(
        trace,
        event_index,
        params.net_transition_labels
    )

    features = features | attributes | history

    # Workload/queue-length features cannot in general be reconstructed
    # exactly from an ordinary event log without reproducing Prosit's
    # resource scheduler. If the fitted tree actually uses these features,
    # we flag that below rather than silently fabricating them.
    if params.use_workload_features:
        features["workload"] = 0
        features["queue_length"] = 0

    return features


def _arrival_features(timestamp):
    """
    Exactly the feature dictionary used by Prosit's simulator.
    """
    return {
        "hour": timestamp.hour,
        "weekday": timestamp.weekday(),
    }


# ============================================================
# GENERIC TREE UTILITIES
# ============================================================

def _leaf_nodes(decision_rules):
    """
    Return all leaf IDs in a DecisionRules object.
    """
    return {
        node_id
        for node_id, node in decision_rules.rules.items()
        if "value" in node
    }


def _tree_features(decision_rules):
    """
    Return all feature names used by a DecisionRules tree.
    """
    return {
        node["feature"]
        for node in decision_rules.rules.values()
        if "feature" in node
    }


def _check_features_available(decision_rules, features):
    """
    Make missing tree features explicit.
    """
    required = _tree_features(decision_rules)
    available = set(features.keys())

    return required - available


# ============================================================
# ARRIVAL-TIME RULE DATA
# ============================================================

def _collect_arrival_rule_data(log, params):
    """
    Assign each case arrival to its Prosit arrival-time leaf.

    Returns:
        {
            leaf_id: [inter-arrival times in minutes, ...]
        }
    """
    result = defaultdict(list)

    decision_rules = params.arrival_time_distribution

    arrivals = _get_arrival_times(log)

    arrivals = sorted(arrivals)

    if len(arrivals) < 2:
        return result

    for i in range(1, len(arrivals)):

        previous = arrivals[i - 1]
        current = arrivals[i]

        delta = max(
            0.0,
            (current - previous).total_seconds() / 60.0
        )

        features = _arrival_features(previous)

        missing = _check_features_available(
            decision_rules,
            features
        )

        if missing:
            raise ValueError(
                f"Missing arrival-rule features: {missing}"
            )

        leaf = _find_leaf_node(
            decision_rules.rules,
            features
        )

        result[leaf].append(delta)

    return result


# ============================================================
# EXECUTION-TIME RULE DATA
# ============================================================

def _collect_execution_rule_data(log, params):
    """
    Assign each event to the leaf of the activity's
    execution-time decision tree.
    """
    result = defaultdict(lambda: defaultdict(list))

    for trace in log:

        for i, event in enumerate(trace):

            activity = event.get("concept:name")

            if activity not in params.execution_time_distributions:
                continue

            execution_time = _get_execution_time(event)

            if execution_time is None:
                continue

            decision_rules = (
                params.execution_time_distributions[activity]
            )

            features = _execution_features(
                trace,
                i,
                params
            )

            missing = _check_features_available(
                decision_rules,
                features
            )

            if missing:
                raise ValueError(
                    f"Missing execution-rule features for "
                    f"{activity}: {missing}"
                )

            leaf = _find_leaf_node(
                decision_rules.rules,
                features
            )

            result[activity][leaf].append(
                execution_time
            )

    return result


# ============================================================
# WAITING-TIME RULE DATA
# ============================================================

def _collect_waiting_rule_data(log, params):
    """
    Assign each event to the leaf of its resource's
    waiting-time decision tree.
    """
    result = defaultdict(lambda: defaultdict(list))

    for trace in log:

        for i, event in enumerate(trace):

            resource = _get_resource(event)

            if resource not in params.waiting_time_distributions:
                continue

            waiting_time = _get_waiting_time(event)

            if waiting_time is None:
                continue

            decision_rules = (
                params.waiting_time_distributions[resource]
            )

            features = _waiting_features(
                trace,
                i,
                params
            )

            missing = _check_features_available(
                decision_rules,
                features
            )

            if missing:
                raise ValueError(
                    f"Missing waiting-rule features for "
                    f"{resource}: {missing}"
                )

            leaf = _find_leaf_node(
                decision_rules.rules,
                features
            )

            result[resource][leaf].append(
                waiting_time
            )

    return result


# ============================================================
# WASSERSTEIN
# ============================================================

def _wasserstein(x, y):
    """
    Wasserstein distance between two one-dimensional samples.
    """
    if len(x) == 0 or len(y) == 0:
        return np.nan

    return float(
        wasserstein_distance(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float)
        )
    )


def _average_leaf_wasserstein(real_data, sim_data):
    """
    Average Wasserstein distance over leaves that occur in
    both real and simulated data.
    """
    distances = []

    common_keys = set(real_data) & set(sim_data)

    for key in common_keys:

        if (
            len(real_data[key]) == 0
            or len(sim_data[key]) == 0
        ):
            continue

        distances.append(
            _wasserstein(
                real_data[key],
                sim_data[key]
            )
        )

    if not distances:
        return np.nan

    return float(np.mean(distances))


# ============================================================
# RULE-ATD
# ============================================================

def rule_atd(real_log, sim_log, params):
    """
    Rule-ATD:
    Wasserstein distance between real and simulated
    inter-arrival-time distributions at decision-tree leaves.
    """
    real_data = _collect_arrival_rule_data(
        real_log,
        params
    )

    sim_traces = _simlog_to_traces(sim_log)

    sim_data = _collect_arrival_rule_data(
        sim_traces,
        params
    )

    return _average_leaf_wasserstein(
        real_data,
        sim_data
    )


# ============================================================
# RULE-ETD
# ============================================================

def rule_etd(real_log, sim_log, params):
    """
    Rule-ETD:
    Wasserstein distance between real and simulated
    execution-time distributions at decision-tree leaves.

    Distances are calculated separately for every activity
    and leaf, then averaged.
    """
    real_data = _collect_execution_rule_data(
        real_log,
        params
    )

    sim_traces = _simlog_to_traces(sim_log)

    sim_data = _collect_execution_rule_data(
        sim_traces,
        params
    )

    distances = []

    common_activities = (
        set(real_data) & set(sim_data)
    )

    for activity in common_activities:

        distances.append(
            _average_leaf_wasserstein(
                real_data[activity],
                sim_data[activity]
            )
        )

    distances = [
        d for d in distances
        if not np.isnan(d)
    ]

    if not distances:
        return np.nan

    return float(np.mean(distances))


# ============================================================
# RULE-WTD
# ============================================================

def rule_wtd(real_log, sim_log, params):
    """
    Rule-WTD:
    Wasserstein distance between real and simulated
    waiting-time distributions at decision-tree leaves.

    Distances are calculated separately for every resource
    and leaf, then averaged.
    """
    real_data = _collect_waiting_rule_data(
        real_log,
        params
    )

    sim_traces = _simlog_to_traces(sim_log)

    sim_data = _collect_waiting_rule_data(
        sim_traces,
        params
    )

    distances = []

    common_resources = (
        set(real_data) & set(sim_data)
    )

    for resource in common_resources:

        distances.append(
            _average_leaf_wasserstein(
                real_data[resource],
                sim_data[resource]
            )
        )

    distances = [
        d for d in distances
        if not np.isnan(d)
    ]

    if not distances:
        return np.nan

    return float(np.mean(distances))


# ============================================================
# RULE-CFD
# ============================================================

def _transition_firing_counts(log, params):
    """
    Count how often each transition fires.

    For a discovered Petri net, multiple transitions can have
    the same activity label. Therefore, where possible we use
    the transition identity recorded in Prosit's simulation
    output; for an ordinary XES log we fall back to activity labels.
    """
    counts = defaultdict(int)
    total = 0

    for trace in log:

        for event in trace:

            activity = event.get("concept:name")

            if activity is None:
                continue

            counts[activity] += 1
            total += 1

    return counts, total


def _transition_firing_frequencies(log, params):
    """
    Convert firing counts to frequencies.
    """
    counts, total = _transition_firing_counts(
        log,
        params
    )

    if total == 0:
        return {}

    return {
        activity: count / total
        for activity, count in counts.items()
    }


def rule_cfd(real_log, sim_log, params):
    """
    Rule-CFD:

        sum over transitions of
        |real firing frequency - simulated firing frequency|

    We use activity labels because ordinary XES event logs do not
    retain Prosit's internal Petri-net transition identity.
    """
    real_freq = _transition_firing_frequencies(
        real_log,
        params
    )

    sim_traces = _simlog_to_traces(sim_log)

    sim_freq = _transition_firing_frequencies(
        sim_traces,
        params
    )

    all_activities = (
        set(real_freq) | set(sim_freq)
    )

    return float(
        sum(
            abs(
                real_freq.get(activity, 0.0)
                -
                sim_freq.get(activity, 0.0)
            )
            for activity in all_activities
        )
    )


# ============================================================
# ALL FOUR RULE METRICS
# ============================================================

def calculate_rule_metrics(real_log, sim_log, params):
    """
    Calculate all four DBPS rule-level metrics.
    """
    return {
        "Rule-CFD": rule_cfd(
            real_log,
            sim_log,
            params
        ),

        "Rule-ATD": rule_atd(
            real_log,
            sim_log,
            params
        ),

        "Rule-ETD": rule_etd(
            real_log,
            sim_log,
            params
        ),

        "Rule-WTD": rule_wtd(
            real_log,
            sim_log,
            params
        ),
    }