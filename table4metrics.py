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
    activity_col: str = "Activity",
    start_col: str = "start:timestamp",
    end_col: str = "time:timestamp",
    base: float | None = None,
) -> float:
    """Mean over activities of the discretized entropy of execution times."""

    start = pd.to_datetime(sim_log[start_col])
    end = pd.to_datetime(sim_log[end_col])
    duration_seconds = (end - start).dt.total_seconds()

    print(sim_log[activity_col].value_counts())
    entropy_per_activity = duration_seconds.groupby(sim_log[activity_col]).agg(
        execution_time_entropy, base=base
    )

    result = float(entropy_per_activity.mean())
    print(f"Average execution time entropy: {result:.4f}")
    return result