import pandas as pd
import numpy as np

from scipy.stats import wasserstein_distance

CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"

# Internal column names of the normalised event frame.
_ORDER = "_order"
_START = "_start"
_END = "_end"


def calculate_table2_metrics(test_log, sim_log):
    # Normalise each log once and reuse it for all three metrics.
    test_frame = _event_frame(test_log)
    sim_frame = _event_frame(sim_log)

    ngd_value = _ngd(test_frame, sim_frame, n=3)
    print(f"NGD = {ngd_value:.4f}")

    ctd_value = _ctd(test_frame, sim_frame)
    print(f"CTD = {ctd_value:.2f}")

    car_value = _car(test_frame, sim_frame)
    print(f"CAR = {car_value:.2f}")
    return ngd_value, ctd_value, car_value


def _first_present(columns, candidates):
    return next((c for c in candidates if c in columns), None)


def _event_frame(log):
    """
    Convert a log into one flat DataFrame with the columns:

      - case:concept:name  case identifier
      - concept:name       activity
      - _order             timestamp used to order events within a case
      - _start / _end      start and end timestamp of the event

    Supports:
      - PM4Py EventLog
      - pandas DataFrame

    All metrics are computed with vectorised pandas/numpy operations on
    this frame instead of Python loops over cases and events.
    """

    if isinstance(log, pd.DataFrame):
        cols = log.columns
        order_col = _first_present(
            cols, ["time:timestamp", "start_timestamp", "start:timestamp"]
        )
        start_col = _first_present(
            cols, ["start:timestamp", "start_timestamp", "time:timestamp"]
        )
        end_col = _first_present(
            cols, ["end:timestamp", "end_timestamp", "time:timestamp"]
        )

        frame = log[[CASE_COL, ACTIVITY_COL]].copy()
        for target, source in [(_ORDER, order_col), (_START, start_col), (_END, end_col)]:
            if source is not None:
                frame[target] = pd.to_datetime(log[source])
        return frame

    # PM4Py EventLog: a single pass over all events.
    cases, activities, orders, starts, ends = [], [], [], [], []

    for case_index, trace in enumerate(log):
        for event in trace:
            ts = event.get("time:timestamp")
            cases.append(case_index)
            activities.append(event[ACTIVITY_COL])
            orders.append(ts)
            starts.append(event.get("start:timestamp", ts))
            ends.append(event.get("end:timestamp", ts))

    return pd.DataFrame({
        CASE_COL: cases,
        ACTIVITY_COL: activities,
        _ORDER: pd.to_datetime(orders),
        _START: pd.to_datetime(starts),
        _END: pd.to_datetime(ends),
    })


def _sorted_by_case(frame):
    """Sort events chronologically within each case (stable)."""
    keys = [CASE_COL, _ORDER] if _ORDER in frame.columns else [CASE_COL]
    return frame.sort_values(keys, kind="stable")


def _trace_activity_sequences(log):
    """
    Return activity sequences for all cases.

    Supports:
      - PM4Py EventLog
      - pandas DataFrame
    """

    frame = _sorted_by_case(_event_frame(log))

    return [
        group.tolist()
        for _, group in frame.groupby(CASE_COL, sort=False)[ACTIVITY_COL]
    ]


def _ngram_counts(frame, codes, n, start_code, end_code, base):
    """
    Count n-grams of all traces in ``frame`` using numpy.

    Every trace is padded with n-1 START and n-1 END symbols and all
    padded traces are laid out after each other in one integer array.
    Each n-gram is encoded as a single integer (base-``base`` number),
    and only windows that lie completely inside one padded trace count.
    """

    case_codes = frame[CASE_COL].to_numpy()

    # Length of each trace, in order of appearance (frame is sorted by case).
    change = np.flatnonzero(case_codes[1:] != case_codes[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(case_codes)]))
    lengths = np.diff(bounds) if len(case_codes) else np.array([], dtype=np.int64)

    padded_lengths = lengths + 2 * (n - 1)
    padded_offsets = np.concatenate(([0], np.cumsum(padded_lengths)[:-1]))
    total = int(padded_lengths.sum())

    if total == 0:
        return pd.Series(dtype=np.int64)

    padded = np.full(total, end_code, dtype=np.int64)
    for k in range(n - 1):
        padded[padded_offsets + k] = start_code

    # Position of every event inside its padded trace.
    event_offsets = np.repeat(padded_offsets + (n - 1), lengths)
    within_trace = np.arange(len(codes)) - np.repeat(bounds[:-1], lengths)
    padded[event_offsets + within_trace] = codes

    # Encode every window of length n as one integer.
    n_windows = total - n + 1
    keys = np.zeros(n_windows, dtype=np.int64)
    for k in range(n):
        keys = keys * base + padded[k:k + n_windows]

    # Keep only windows that do not cross a trace boundary.
    position = np.arange(total) - np.repeat(padded_offsets, padded_lengths)
    last_start = np.repeat(padded_lengths - n, padded_lengths)
    valid = (position <= last_start)[:n_windows]

    return pd.Series(keys[valid]).value_counts()


def _ngd(frame1, frame2, n=3):
    frame1 = _sorted_by_case(frame1)
    frame2 = _sorted_by_case(frame2)

    # Encode activities of both logs with one shared integer mapping.
    codes, uniques = pd.factorize(
        np.concatenate([
            frame1[ACTIVITY_COL].to_numpy(dtype=object),
            frame2[ACTIVITY_COL].to_numpy(dtype=object),
        ])
    )

    # Dummy symbols get codes that cannot collide with an actual activity.
    start_code = len(uniques)
    end_code = len(uniques) + 1
    base = len(uniques) + 2

    h1 = _ngram_counts(frame1, codes[:len(frame1)], n, start_code, end_code, base)
    h2 = _ngram_counts(frame2, codes[len(frame1):], n, start_code, end_code, base)

    numerator = h1.sub(h2, fill_value=0).abs().sum()
    denominator = h1.sum() + h2.sum()

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


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

    return _ngd(_event_frame(log1), _event_frame(log2), n=n)


def _require_column(frame, column, message):
    if column not in frame.columns:
        raise ValueError(message)


def _case_cycle_times_from_frame(frame):
    _require_column(frame, _START, "Could not find start timestamp column.")
    _require_column(frame, _END, "Could not find end timestamp column.")

    grouped = frame.groupby(CASE_COL)
    cycle_times = (grouped[_END].max() - grouped[_START].min()).dropna()

    return cycle_times.dt.total_seconds().to_numpy(dtype=float)


def _case_cycle_times(log):
    """
    Extract end-to-end cycle times in seconds.

    For each case:
        cycle time = latest end timestamp - earliest start timestamp

    Works with:
      - PM4Py EventLog
      - pandas simulation DataFrame

    The simulator's column names can be adapted in _event_frame if necessary.
    """

    return _case_cycle_times_from_frame(_event_frame(log))


def _ctd(frame1, frame2):
    ct1 = _case_cycle_times_from_frame(frame1)
    ct2 = _case_cycle_times_from_frame(frame2)

    if len(ct1) == 0 or len(ct2) == 0:
        return np.nan

    return float(
        wasserstein_distance(ct1, ct2)
    )


def ctd(log1, log2):
    """
    Cycle Time Distribution distance.

    The paper defines CTD as the 1-Wasserstein distance
    between the empirical distributions of case cycle times.
    """

    return _ctd(_event_frame(log1), _event_frame(log2))


def _case_arrival_times_from_frame(frame):
    _require_column(frame, _START, "Could not find timestamp column for CAR.")

    arrivals = frame.groupby(CASE_COL)[_START].min().dropna()

    return pd.DatetimeIndex(arrivals)


def case_arrival_times(log):
    """
    Return the earliest start timestamp of every case.
    """

    return _case_arrival_times_from_frame(_event_frame(log))


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


def _car(frame1, frame2):
    arrivals1 = _case_arrival_times_from_frame(frame1)
    arrivals2 = _case_arrival_times_from_frame(frame2)

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


def car(log1, log2):
    """
    Case Arrival Rate (CAR).

    Cases are represented by their earliest start timestamp.
    Arrivals are binned hourly and compared using EMD.
    """

    return _car(_event_frame(log1), _event_frame(log2))
