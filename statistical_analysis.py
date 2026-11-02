
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


# Transform dictionary: renames names from our file to the paper's naming, so
# both files can be compared. Names that are not listed are left unchanged.
TRANSFORM = {
    # dataset names
    "bpi2012": "BPI_Challenge_2012_W_Two_TS",
    # method names
    "prosit-replication": "prosit",
    # metric names
    "NGD": "2gd",
    "CTD": "ctd",
    "CAR": "car",
    "Rule-ATD": "arrival_tree",
    "Rule-WTD": "waiting_tree",
    "Rule-ETD": "execution_tree",
    "Rule-CFD": "transition_tree",
    "Execution Time Entropy": "etd_entropy",
}


def normalize(node):
    """Apply TRANSFORM to keys and unwrap {"values": [...]} nodes.
    Necessary for compatibility between our results and the paper's results,
    which use different names."""
    if not isinstance(node, dict):
        return node
    if isinstance(node.get("values"), list):
        return node["values"]
    return {TRANSFORM.get(key, key): normalize(value) for key, value in node.items()}


def compare(a, b, path=()):
    results = []

    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() & b.keys()):
            results += compare(a[key], b[key], path + (key,))

    elif isinstance(a, list) and isinstance(b, list):
        x = np.asarray(a, dtype=float)
        y = np.asarray(b, dtype=float)
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]

        if len(x) >= 2 and len(y) >= 2:
            t, p = ttest_ind(x, y, equal_var=False)
            results.append({
                "metric": "/".join(path),
                "n_A": len(x),
                "n_B": len(y),
                "mean_A": x.mean(),
                "mean_B": y.mean(),
                "difference_A_minus_B": x.mean() - y.mean(),
                "t_statistic": t,
                "p_value": p,
            })

    return results


def main():
    if len(sys.argv) < 3:
        print("Usage: python statistical_analysis.py samples_A.json samples_B.json")
        return

    with open(sys.argv[1], encoding="utf-8") as f:
        a = json.load(f)

    with open(sys.argv[2], encoding="utf-8") as f:
        b = json.load(f)

    results = compare(normalize(a), normalize(b))
    if not results:
        print("No matching metric lists with sufficient observations.")
        return

    reject, adjusted, _, _ = multipletests(
        [r["p_value"] for r in results],
        alpha=0.05,
        method="holm",
    )

    for row, p_adj, significant in zip(results, adjusted, reject):
        row["p_holm"] = p_adj
        row["significant"] = bool(significant)

    df = pd.DataFrame(results).sort_values("p_holm")
    df.to_csv("sample_comparison_results.csv", index=False)
    print(df.to_string(index=False))
    print("\nSaved to sample_comparison_results.csv")


if __name__ == "__main__":
    main()
