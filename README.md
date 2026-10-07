# seminar-process-simulation
Replication of paper  "Reliable and Configurable Process Simulations via Probabilistic White-Box Models" by Vinci et al. (2026) for the Seminar Process Analytics course at TU/e..

## Run a simulation

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

Metrics are not calculated by default. Enable them with `--calculate-metrics`:

```bash
python simulation.py --dataset p2p --calculate-metrics
```
