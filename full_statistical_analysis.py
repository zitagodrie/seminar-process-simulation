
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


def compare(a, b, path=()):
    results = []

    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() & b.keys()):
            results.extend(compare(a[key], b[key], path + (key,)))

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
    parser = argparse.ArgumentParser(
        description="Compare any number of JSON pairs with global Holm correction."
    )
    parser.add_argument(
        "files", nargs="+",
        help="JSON files in pairs: A1.json B1.json A2.json B2.json ..."
    )
    parser.add_argument(
        "--out", default="all_comparison_results.csv",
        help="Output CSV filename"
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if len(args.files) % 2:
        parser.error("Provide an even number of files: one A/B pair per comparison.")

    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1.")

    all_results = []

    # Collect every test before applying the correction.
    for i in range(0, len(args.files), 2):
        file_a, file_b = args.files[i:i + 2]

        with open(file_a, encoding="utf-8") as f:
            a = json.load(f)
        with open(file_b, encoding="utf-8") as f:
            b = json.load(f)

        pair_name = f"{Path(file_a).stem}_vs_{Path(file_b).stem}"
        results = compare(a, b)

        for row in results:
            row["pair"] = pair_name
            row["file_A"] = file_a
            row["file_B"] = file_b

        all_results.extend(results)

    if not all_results:
        raise SystemExit("No matching metric lists with at least 2 values per group.")

    # One correction across ALL tests from ALL file pairs.
    p_values = [r["p_value"] for r in all_results]
    reject, adjusted, _, _ = multipletests(
        p_values, alpha=args.alpha, method="holm"
    )

    for row, p_adj, significant in zip(all_results, adjusted, reject):
        row["p_holm_global"] = p_adj
        row["significant"] = bool(significant)
        row["alpha"] = args.alpha
        row["total_tests_corrected"] = len(all_results)

    df = pd.DataFrame(all_results).sort_values("p_holm_global")
    df.to_csv(args.out, index=False)

    print(df.to_string(index=False))
    print(f"\nTotal tests: {len(df)}")
    print(f"Significant after global Holm correction: {sum(reject)}")
    print(f"Results saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
