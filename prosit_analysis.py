
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from table_data import TABLES

DATASETS = {
    "BPIC12W": "BPI_Challenge_2012_W_Two_TS",
    "BPIC17W": "BPI_Challenge_2017_W_Two_TS",
    "P2P-1000": "confidential_1000",
    "P2P-2000": "confidential_2000",
    "ACR": "ConsultaDataMining201618",
    "CVS": "cvs_pharmacy",
    "Production": "Production",
    "PurchasingExample": "PurchasingExample",
    "SynLoan": "SynLoan",
}

METHODS = {
    "ProSiT": "prosit",
    "AgentSim": "agentsimulator",
    "SIMOD": "simod",
    "RIMS": "rims",
    "DSIM": "dsim",
}


def get_values(data, dataset, method, metric):
    raw = (
        data.get(DATASETS[dataset], {})
        .get(METHODS[method], {})
        .get(metric, {})
        .get("values")
    )
    if not isinstance(raw, list):
        return None

    x = np.asarray(raw, dtype=float)
    x = x[np.isfinite(x)]
    return x if len(x) >= 2 else None


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate(
        (len(p) - np.arange(len(p))) * p[order]
    )
    result = np.empty(len(p))
    result[order] = np.minimum(adjusted, 1.0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="Path to raw results JSON")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--out", default="welch_pairwise_results.csv"
    )
    parser.add_argument(
        "--audit-out", default="table_json_audit.csv"
    )
    args = parser.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    tests, audit = [], []

    for metric, datasets in TABLES.items():
        for dataset, cells in datasets.items():
            available = {}

            for method, (table_mean, table_sd) in cells.items():
                x = get_values(data, dataset, method, metric)

                if x is None:
                    audit.append({
                        "dataset": dataset,
                        "metric": metric,
                        "method": method,
                        "table_mean": table_mean,
                        "table_sd": table_sd,
                        "status": "Missing raw values or n < 2",
                    })
                    continue

                mean = float(x.mean())
                sd = float(x.std(ddof=0))

                agrees = (
                    np.isclose(mean, table_mean, atol=0.002, rtol=0.01)
                    and np.isclose(sd, table_sd, atol=0.002, rtol=0.05)
                )

                audit.append({
                    "dataset": dataset,
                    "metric": metric,
                    "method": method,
                    "table_mean": table_mean,
                    "table_sd": table_sd,
                    "json_n": len(x),
                    "json_mean": mean,
                    "json_sd": sd,
                    "mean_abs_diff": abs(mean - table_mean),
                    "sd_abs_diff": abs(sd - table_sd),
                    "status": "Agrees" if agrees else "DIFFERS",
                })

                available[method] = x

            for method_a, method_b in itertools.combinations(
                available, 2
            ):
                x, y = available[method_a], available[method_b]
                result = ttest_ind(
                    x, y, equal_var=False, alternative="two-sided"
                )

                tests.append({
                    "dataset": dataset,
                    "metric": metric,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_a": len(x),
                    "n_b": len(y),
                    "mean_a": x.mean(),
                    "sd_a": x.std(ddof=1),
                    "mean_b": y.mean(),
                    "sd_b": y.std(ddof=1),
                    "t_statistic": result.statistic,
                    "p_raw": result.pvalue,
                })

    if not tests:
        raise SystemExit(
            "No tests ran. Check JSON keys and table_data.py."
        )

    results = pd.DataFrame(tests)
    results["p_holm"] = np.nan

    # Correct across all dataset/method comparisons within each metric.
    for _, indices in results.groupby("metric").groups.items():
        results.loc[indices, "p_holm"] = holm(
            results.loc[indices, "p_raw"].to_numpy()
        )

    results["significant_holm"] = results["p_holm"] < args.alpha
    results.sort_values(
        ["metric", "dataset", "p_holm"]
    ).to_csv(args.out, index=False)

    pd.DataFrame(audit).to_csv(args.audit_out, index=False)

    print(f"Tests performed: {len(results)}")
    print(f"Test results: {args.out}")
    print(f"Summary audit: {args.audit_out}")

    significant = results[results["significant_holm"]]
    print("\nSignificant after Holm correction:")
    if significant.empty:
        print("None")
    else:
        print(significant[
            ["dataset", "metric", "method_a", "method_b",
             "p_raw", "p_holm"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
