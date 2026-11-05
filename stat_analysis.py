import json
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multitest as multitest

# same keys for both baseline and summary
KEY_CONVERT = {
    'BPI_Challenge_2012_W_Two_TS': 'BPIC12W',
    'BPI_Challenge_2017_W_Two_TS': 'BPIC17W',
    'cvs_pharmacy': 'CVS',
    'ConsultaDataMining201618': 'ACR',
    'confidential_2000': 'P2P',
    'BPIC12W': 'BPIC12W',
    'BPIC17W': 'BPIC17W',
    'CVS': 'CVS',
    'ACR': 'ACR',
    'P2P-2000': 'P2P',
    'prosit': 'prosit',
    'simod': 'simod',
    'rims': 'rims',
    'dsim': 'dsim',
    'ProSiT': 'prosit',
    'SIMOD': 'simod',
    'RIMS': 'rims',
    'DSIM': 'dsim',
}

def load_json_file(file_path):
    """
    Load a JSON file and return its content as a Python object.

    :param file_path: Path to the JSON file.
    :return: Python object representing the JSON data.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def print_json_structure(data):
    """
    Pretty print a Python object as JSON.

    :param data: Python object to be printed as JSON.
    """
    for dataset_key, dataset_value in data.items():
        print(f"Dataset: {dataset_key}")
        print(f"  Methods: {', '.join(dataset_value.keys())}")  # Print the remaining method keys
        print(f"  Metrics: {', '.join(dataset_value[next(iter(dataset_value))].keys())}")  # Print the metrics keys from the first method
        print()

def main():
    if len(sys.argv) < 3:
        print("Usage: python stat_analysis.py samples_our.json samples_github.json")
        return

    with open(sys.argv[1], encoding="utf-8") as f:
        summary = load_json_file(f)

    with open(sys.argv[2], encoding="utf-8") as f:
        baseline = load_json_file(f)

    
    # Load the JSON files
    print("Structure of the loaded JSON data:")
    print_json_structure(baseline)

    
    print("Structure of the loaded JSON data:")
    print_json_structure(summary)


    # drop unused methods
    for dataset_key, dataset_value in summary.items():
        del dataset_value['agentsimulator']  # Remove 'agentsimulator' key if it exists
        summary[dataset_key] = dataset_value  # Update the summary with the modified dataset_value

    for dataset_key, dataset_value in baseline.items():
        del dataset_value['agentsimulator']  # Remove 'agentsimulator' key if it exists
        baseline[dataset_key] = dataset_value  # Update the summary with the modified dataset_value

    # Remove datasets and methods that are not in the specified list
    dataset_keys, dataset_values = zip(*baseline.items())
    for dataset_key, dataset_value in zip(dataset_keys, dataset_values):
        if dataset_key not in ['BPI_Challenge_2012_W_Two_TS', 'BPI_Challenge_2017_W_Two_TS', 'confidential_2000', 'ConsultaDataMining201618', 'cvs_pharmacy']:
            del baseline[dataset_key]  # Remove datasets that are not in the specified list

    summary_renamed = {}
    for dataset_key, dataset_value in summary.items():
        summary_renamed[KEY_CONVERT[dataset_key]] = {}
        for method_key, method_value in dataset_value.items():
            summary_renamed[KEY_CONVERT[dataset_key]][KEY_CONVERT[method_key]] = {}
            for metric_key, metric_value in method_value.items():
                summary_renamed[KEY_CONVERT[dataset_key]][KEY_CONVERT[method_key]][metric_key] = metric_value

    print_json_structure(summary_renamed)  # Print the structure of the renamed summary dictionary

    baseline_renamed = {}
    for dataset_key, dataset_value in baseline.items():
        baseline_renamed[KEY_CONVERT[dataset_key]] = {}
        for method_key, method_value in dataset_value.items():
            # if method_key not in ['ProSiT', 'SIMOD', 'RIMS', 'DSIM']:
            baseline_renamed[KEY_CONVERT[dataset_key]][KEY_CONVERT[method_key]] = {}
            for metric_key, metric_value in method_value.items():
                baseline_renamed[KEY_CONVERT[dataset_key]][KEY_CONVERT[method_key]][metric_key] = metric_value

    print("Structure of the renamed github JSON:")
    print_json_structure(baseline_renamed)  # Print the structure of the renamed baseline dictionary



    summary = summary_renamed
    baseline = baseline_renamed

    #t-tests
    metrics = ['2gd', 'ctd', 'car', 'etd_entropy', 'arrival_tree', 'transition_tree', 'execution_tree', 'waiting_tree']
    datasets = ['BPIC12W', 'BPIC17W', 'ACR', 'CVS', 'P2P']
    methods = ['prosit', 'simod', 'rims', 'dsim']

    print("Upper limit number of tests:", len(metrics)*len(datasets)*len(methods))

    count_rejected = 0
    count_tested = 0
    p_values = []
    for dataset_key, dataset_value in summary.items():
        for method_key, method_value in dataset_value.items():
            for metric_key, metric_value in method_value.items():
                if metric_key in metrics:
                    print(f"Dataset: {dataset_key},     Method: {method_key},       Metric: {metric_key}")
                    t_statistic = stats.ttest_ind_from_stats(
                        mean1=metric_value['mean'],
                        std1=metric_value['std'],
                        nobs1=10,
                        mean2=baseline[dataset_key][method_key][metric_key]['mean'],
                        std2=baseline[dataset_key][method_key][metric_key]['std'],
                        nobs2=10,
                        equal_var=False
                    )
                    count_tested += 1
                    print(t_statistic)
                    p_values.append(t_statistic.pvalue)
                    if t_statistic.pvalue < 0.05:
                        print("Reject the null hypothesis: The means are significantly different.")
                        count_rejected += 1
                    print()

    print(f'Number of tests performed: {count_tested}')
    print(f'Number of rejections: {count_rejected}')

    holms_corrected = multitest.multipletests(p_values, alpha=0.05, method='holm')
    print(holms_corrected[0])
    num_rejections_holm = sum(holms_corrected[0])
    print(f'Number of rejections after Holm-Bonferroni correction: {num_rejections_holm}')

    #print as latex table in terminal
    print("""\\begin{table}[htbp]
    \\centering
    \\begin{tabular}{ll*{7}{>{\\centering\\arraybackslash}p{1.2cm}}>{\\centering\\arraybackslash}p{1.6cm}}
    \\toprule
    \\textbf{Method} & \\textbf{Experiment} & NGD & CTD & CAR & Rule-CFD & Rule-ATD & Rule-ETD & Rule-WTD & Exec. Time Entropy \\\\""")
    for dataset_key, dataset_value in summary.items():
        print("\\midrule")
        print(f"\\multicolumn{{9}}{{l}}{{{dataset_key}}}\\\\")
        for method_key, method_value in summary[dataset_key].items():
            lineprint_our = method_key + " & our"
            lineprint_baseline = method_key + " & paper"
            for metric_key in ['2gd', 'ctd', 'car', 'transition_tree', 'arrival_tree', 'execution_tree', 'waiting_tree', 'etd_entropy']:
                if metric_key in method_value and metric_key in baseline[dataset_key][method_key]:
                    lineprint_our += f" & {method_value[metric_key]['mean']:.2f}"
                    lineprint_baseline += f" & {baseline[dataset_key][method_key][metric_key]['mean']:.2f}"
                    if holms_corrected[0][count_tested - len(p_values) + p_values.index(stats.ttest_ind_from_stats(
                        mean1=method_value[metric_key]['mean'],
                        std1=method_value[metric_key]['std'],
                        nobs1=10,
                        mean2=baseline[dataset_key][method_key][metric_key]['mean'],
                        std2=baseline[dataset_key][method_key][metric_key]['std'],
                        nobs2=10,
                        equal_var=False
                    ).pvalue)]:
                        lineprint_our += " \\cellcolor{red!20}"
                        lineprint_baseline += " \\cellcolor{red!20}"
                else:
                    lineprint_our += " & N/A"
                    lineprint_baseline += " & N/A"
            print(lineprint_our + " \\\\")
            print(lineprint_baseline + " \\\\")
    print("""\\bottomrule
    \end{tabular}
    \caption{Metric results of the author's simulation compared to its posted results. Cells highlighted in red indicate results are significantly different from the baseline (p < 0.05) after Holm-Bonferroni correction. N/A's are placed where no data was collected by the author, or the author's replication code was unable to produce results.}
    \label{tab:results}
    \end{table}""")

if __name__ == "__main__":
    main()




