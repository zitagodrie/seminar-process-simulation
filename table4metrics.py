"""Calculate entropy of execution times."""

import numpy as np
import pandas as pd
from scipy.stats import entropy


def execution_time_entropy(durations, base: float | None = None) -> float:
    x = np.asarray(durations, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    counts, _ = np.histogram(x, bins="sturges")         # p.350 of paper says they use Sturges' method for binning
    return float(entropy(counts, base=base))


def average_execution_time_entropy(
    sim_log: pd.DataFrame,
    activity_col: str = "activity",
    start_col: str = "start_timestamp",
    end_col: str = "end_timestamp",
    base: float | None = None,
) -> float:
    """Mean over activities of the discretized entropy of execution times."""
    if activity_col not in sim_log.columns:
        for candidate in ("concept:name", "activity"):
            if candidate in sim_log.columns:
                activity_col = candidate
                break
        else:
            raise ValueError("Could not find activity column.")

    if start_col not in sim_log.columns:
        for candidate in ("start:timestamp", "start_timestamp", "time:timestamp"):
            if candidate in sim_log.columns:
                start_col = candidate
                break
        else:
            raise ValueError("Could not find start timestamp column.")

    if end_col not in sim_log.columns:
        for candidate in ("end:timestamp", "end_timestamp", "time:timestamp"):
            if candidate in sim_log.columns:
                end_col = candidate
                break
        else:
            raise ValueError("Could not find end timestamp column.")

    start = pd.to_datetime(sim_log[start_col])
    end = pd.to_datetime(sim_log[end_col])
    duration_seconds = (end - start).dt.total_seconds()

    entropy_per_activity = duration_seconds.groupby(sim_log[activity_col]).agg(
        execution_time_entropy, base=base
    )

    result = float(entropy_per_activity.mean())
    print(f"Average execution time entropy: {result:.4f}")
    return result