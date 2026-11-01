#!/usr/bin/env bash
# Run simulation.py in the "paper-5" setting (5 runs) for all 5 datasets.
# Usage: ./run_all.sh [dataset ...]   (no arguments = all datasets)

set -u
cd "$(dirname "$0")"

# Activate the virtual environment (Windows layout first, then Linux/macOS)
if [ -f .venv/Scripts/activate ]; then
    source .venv/Scripts/activate
elif [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "No .venv found in $(pwd)" >&2
    exit 1
fi

if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=(p2p cvs acr bpi2012 bpi2017)
fi

failed=()
for ds in "${DATASETS[@]}"; do
    echo "=== Running $ds (paper-10) ==="
    if ! python simulation.py --dataset "$ds" --simulations paper-10; then
        echo "!!! $ds failed" >&2
        failed+=("$ds")
    fi
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo "Failed datasets: ${failed[*]}" >&2
    exit 1
fi
echo "All done. Metrics are in metrics.json"
