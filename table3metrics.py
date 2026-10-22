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
    if isinstance(value, pd.Timestamp):
        return value
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

    Categorical attributes are one-hot encoded as "<attr> = <value>",
    exactly like Prosit's discovery and simulator.
    """
    attributes = _case_attributes(
        trace,
        getattr(params, "label_data_attributes", [])
    )

    categorical = getattr(params, "label_data_attributes_categorical", [])
    values = getattr(params, "attribute_values_label_categorical", {})

    for attr in categorical:
        value = attributes.pop(attr, None)

        for v in values.get(attr, []):
            attributes[attr + " = " + str(v)] = int(value == v)

    return attributes


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

    The arrival is the earliest moment a case's events became available:
    enabled:timestamp (Prosit output), else start:timestamp, else the
    completion timestamp. Using only time:timestamp would add the first
    activity's duration to the arrival.
    """
    arrivals = []

    for trace in log:
        if len(trace) == 0:
            continue

        timestamps = [
            _event_enabled(event)
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
    case_col = "case:concept:name"

    # Sort events by case (in order of first appearance) and then by
    # start/end time, so every case becomes one contiguous block.
    case_order = pd.factorize(sim_log[case_col])[0]
    order = (
        pd.DataFrame({
            "case": case_order,
            "start": sim_log["start:timestamp"].to_numpy(),
            "end": sim_log["time:timestamp"].to_numpy(),
        })
        .sort_values(["case", "start", "end"], kind="stable")
        .index
    )

    records = sim_log.iloc[order].to_dict("records")
    sizes = np.bincount(case_order)

    traces = []
    offset = 0
    for size in sizes:
        traces.append(records[offset:offset + size])
        offset += size

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


def _iter_event_contexts(trace, params):
    """
    Yield (event, case attributes, history before the event) for every
    event in a trace.

    The case attributes are computed once per trace and the history is
    updated incrementally, instead of rescanning the trace for every event.
    The yielded history dict is reused, so callers must not keep it.
    """
    attributes = _build_case_attributes_from_log(trace, params)

    history = {
        activity: 0
        for activity in params.net_transition_labels
    }

    for event in trace:
        yield event, attributes, history

        activity = event.get("concept:name")

        if activity in history:
            history[activity] += 1


def _one_hot(cache, value, labels, prefix):
    """Return a (cached) one-hot dict of ``value`` over ``labels``."""
    try:
        return cache[value]
    except KeyError:
        encoding = {
            prefix + label: int(label == value)
            for label in labels
        }
        cache[value] = encoding
        return encoding
    except TypeError:
        # Unhashable value: do not cache.
        return {
            prefix + label: int(label == value)
            for label in labels
        }


def _event_start(event):
    start = _event_timestamp(event, "start:timestamp")

    if start is None:
        start = _event_timestamp(event, "time:timestamp")

    return start


def _event_enabled(event):
    """Moment the event became enabled, falling back to its start."""
    enabled = _event_timestamp(event, "enabled:timestamp")

    if enabled is None:
        enabled = _event_start(event)

    return enabled


def _execution_features(event, attributes, history, params, cache):
    """
    Features used by Prosit's execution-time decision tree.

    Prosit uses:
        resource one-hot
        hour
        weekday
        case attributes
        case history
    """
    resource = _get_resource(event)

    features = dict(
        _one_hot(cache, resource, params.resources, "resource = ")
    )

    start = _event_start(event)

    features["hour"] = start.hour
    features["weekday"] = start.weekday()

    return features | attributes | history


def _waiting_features(event, attributes, history, params, cache):
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
    activity = event.get("concept:name")

    # The waiting time is sampled when the activity becomes enabled.
    enabled = _event_enabled(event)

    features = {
        "hour": enabled.hour,
        "weekday": enabled.weekday(),
    }

    # Prosit's one-hot waiting activity vector
    features.update(
        _one_hot(
            cache,
            activity,
            params.net_transition_labels,
            "waiting_activity = "
        )
    )

    features = features | attributes | history

    # Workload/queue-length features cannot be reconstructed from an
    # ordinary event log without reproducing Prosit's resource scheduler.
    # They are deliberately left out, so a tree that splits on them makes
    # _missing_features fail loudly instead of using fabricated values.
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


def _missing_features(required_cache, decision_rules, features):
    """
    Like _check_features_available, but caches the tree's required
    features so they are not recomputed for every event.
    """
    key = id(decision_rules)

    if key not in required_cache:
        required_cache[key] = _tree_features(decision_rules)

    return required_cache[key].difference(features)


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
    one_hot_cache = {}
    required_cache = {}

    for trace in log:

        for event, attributes, history in _iter_event_contexts(trace, params):

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
                event,
                attributes,
                history,
                params,
                one_hot_cache
            )

            missing = _missing_features(
                required_cache,
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
    one_hot_cache = {}
    required_cache = {}

    for trace in log:

        for event, attributes, history in _iter_event_contexts(trace, params):

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
                event,
                attributes,
                history,
                params,
                one_hot_cache
            )

            missing = _missing_features(
                required_cache,
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
# RULE DATA (collected once per log)
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


def _as_traces(log):
    """Prosit's pandas output is converted to traces; EventLogs are used as is."""
    if isinstance(log, pd.DataFrame):
        return _simlog_to_traces(log)
    return log


def _with_enabled_timestamps(traces, params):
    """
    Add enabled:timestamp to the events of a real event log.

    Only Prosit's simulation output has this column, so without it the real
    side of the waiting-time metric is empty. The enabled times are
    reconstructed with the same alignment-based replay that Prosit's own
    discovery uses (build_df_features). Traces are returned as copies; the
    input log is not modified. Events are matched to the replay rows by case,
    resource and start time; events the alignment dropped (log moves) are left
    without enabled:timestamp and are ignored by the waiting-time metric.
    """
    from prosit.utils.common_utils import build_df_features

    df = build_df_features(
        traces,
        params.net,
        params.initial_marking,
        params.final_marking,
        params.act_to_resources,
        params.net_transition_labels,
        params.resources,
        params.label_data_attributes,
    )
    df = df[df["resource"].notna() & df["start_t"].notna()]
    enabled_by_event = {
        (case, resource, start): enabled
        for case, resource, start, enabled in zip(
            df["case_id"], df["resource"], df["start_t"], df["enabled_t"]
        )
    }

    enriched = []

    for trace in traces:
        case = trace[0]["case:concept:name"]
        events = []

        for event in trace:
            event = dict(event)
            enabled = enabled_by_event.get(
                (case, event.get("org:resource"), event.get("start:timestamp"))
            )

            if enabled is not None:
                event["enabled:timestamp"] = enabled

            events.append(event)

        enriched.append(events)

    return enriched


def prepare_rule_data(log, params):
    """
    Collect everything the four rule metrics need from one log.

    The result for the test log is the same for every simulation run,
    so it can be computed once and passed to calculate_rule_metrics
    via real_data=...
    """
    traces = _as_traces(log)

    if not isinstance(log, pd.DataFrame):
        traces = _with_enabled_timestamps(traces, params)

    return {
        "CFD": _transition_firing_frequencies(traces, params),
        "ATD": _collect_arrival_rule_data(traces, params),
        "ETD": _collect_execution_rule_data(traces, params),
        "WTD": _collect_waiting_rule_data(traces, params),
    }


# ============================================================
# RULE-ATD
# ============================================================

def _rule_atd(real_data, sim_data):
    return _average_leaf_wasserstein(
        real_data["ATD"],
        sim_data["ATD"]
    )


def rule_atd(real_log, sim_log, params):
    """
    Rule-ATD:
    Wasserstein distance between real and simulated
    inter-arrival-time distributions at decision-tree leaves.
    """
    return _rule_atd(
        {"ATD": _collect_arrival_rule_data(_as_traces(real_log), params)},
        {"ATD": _collect_arrival_rule_data(_as_traces(sim_log), params)},
    )


# ============================================================
# RULE-ETD / RULE-WTD
# ============================================================

def _average_grouped_leaf_wasserstein(real_data, sim_data):
    """
    Average of the per-group (activity or resource) average
    leaf Wasserstein distances.
    """
    distances = []

    common_groups = (
        set(real_data) & set(sim_data)
    )

    for group in common_groups:

        distances.append(
            _average_leaf_wasserstein(
                real_data[group],
                sim_data[group]
            )
        )

    distances = [
        d for d in distances
        if not np.isnan(d)
    ]

    if not distances:
        return np.nan

    return float(np.mean(distances))


def _rule_etd(real_data, sim_data):
    return _average_grouped_leaf_wasserstein(
        real_data["ETD"],
        sim_data["ETD"]
    )


def rule_etd(real_log, sim_log, params):
    """
    Rule-ETD:
    Wasserstein distance between real and simulated
    execution-time distributions at decision-tree leaves.

    Distances are calculated separately for every activity
    and leaf, then averaged.
    """
    return _rule_etd(
        {"ETD": _collect_execution_rule_data(_as_traces(real_log), params)},
        {"ETD": _collect_execution_rule_data(_as_traces(sim_log), params)},
    )


def _rule_wtd(real_data, sim_data):
    return _average_grouped_leaf_wasserstein(
        real_data["WTD"],
        sim_data["WTD"]
    )


def rule_wtd(real_log, sim_log, params):
    """
    Rule-WTD:
    Wasserstein distance between real and simulated
    waiting-time distributions at decision-tree leaves.

    Distances are calculated separately for every resource
    and leaf, then averaged.
    """
    return _rule_wtd(
        {"WTD": _collect_waiting_rule_data(_as_traces(real_log), params)},
        {"WTD": _collect_waiting_rule_data(_as_traces(sim_log), params)},
    )


# ============================================================
# RULE-CFD
# ============================================================

def _rule_cfd(real_data, sim_data):
    real_freq = real_data["CFD"]
    sim_freq = sim_data["CFD"]

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


def rule_cfd(real_log, sim_log, params):
    """
    Rule-CFD:

        sum over transitions of
        |real firing frequency - simulated firing frequency|

    We use activity labels because ordinary XES event logs do not
    retain Prosit's internal Petri-net transition identity.
    """
    return _rule_cfd(
        {"CFD": _transition_firing_frequencies(_as_traces(real_log), params)},
        {"CFD": _transition_firing_frequencies(_as_traces(sim_log), params)},
    )


# ============================================================
# ALL FOUR RULE METRICS
# ============================================================

def calculate_rule_metrics(real_log, sim_log, params, real_data=None):
    """
    Calculate all four DBPS rule-level metrics.

    Pass real_data=prepare_rule_data(real_log, params) to reuse the
    test-log data across simulation runs.
    """
    if real_data is None:
        real_data = prepare_rule_data(real_log, params)

    sim_data = prepare_rule_data(sim_log, params)

    return {
        "Rule-CFD": _rule_cfd(real_data, sim_data),
        "Rule-ATD": _rule_atd(real_data, sim_data),
        "Rule-ETD": _rule_etd(real_data, sim_data),
        "Rule-WTD": _rule_wtd(real_data, sim_data),
    }
