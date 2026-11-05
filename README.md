# seminar-process-simulation
Replication of paper  "Reliable and Configurable Process Simulations via Probabilistic White-Box Models" by Vinci et al. (2026) for the Seminar Process Analytics course at TU/e..

## Install dependencies

Install the pinned Python dependencies before running the simulation:

```bash
python -m pip install -r requirements.txt
```

## Run a simulation using our code

Specify the dataset with the required `--dataset` option (or `-d` for short):

```bash
python simulation.py --dataset p2p
```

The short form is also supported:

```bash
python simulation.py -d p2p
```

For example, run a different dataset with:

```bash
python simulation.py --dataset bpi2017
```

Available dataset keys are `p2p`, `cvs`, `acr`, `bpi2012`, and `bpi2017`.

The `--simulations` option (or `-s`) controls how many simulation runs are
performed. It accepts `none` (the default, no simulation), `fast` (one run),
`paper-5` (five runs), or `paper-10` (ten runs):

```bash
python simulation.py --dataset p2p --simulations paper-5
```

The short form is also supported:

```bash
python simulation.py -d p2p -s fast
```

`cvs` is split chronologically into 80% training and 20% test data. The
`p2p`, `acr`, `bpi2012`, and `bpi2017` datasets use their corresponding
`data/<dataset>/folds/train.csv` and `test.csv` files. When simulations are
run, metrics are written to `metrics.json` (per dataset: mean, std and the values of each run for every metric).


## Analyse and compare results

The following command runs t-tests on the summary files produced by our code, and the code by the author. The summary of us running the code by the author on the github branch is found in `summary_our_expB.json`, and the summary from running that code as given by the author is found in `summary_github_expB.json`.

```bash
python stat_analysis.py summary_a.json summary_b.json
```
