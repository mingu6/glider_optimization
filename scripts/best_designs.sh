#!/bin/bash
# Single-objective optimum for each task, for the paper's design table.
#
# Run sequentially: each uses 8 IPOPT workers and ~19 GB, so two at once starves
# both (measured earlier: 16 workers halved throughput, 32 sent the box into swap).
#
# Landing carries the |alpha| <= 25 deg bound. Without it the glider stalls and
# parachutes down at ~107 deg incidence, where the aerodynamic model falls back to
# an analytic 2*sin^2(alpha) with no airfoil dependence at all -- roughly 73% of the
# flight becomes blind to the design and the optimiser thins the section.
set -u
cd /home/affinita/PhD/glider_optimization

run_task () {
    local name=$1
    echo "=== $(date +%H:%M) $name starting"
    rm -f traj_*.npy
    /home/affinita/PhD/venv/bin/python -m glider_optimization.main \
        --config "conf/best_${name}.yaml" --run-name "best_${name}" \
        > "runs/best_${name}/train.log" 2>&1
    echo "--- $(date +%H:%M) $name done: $(grep -a 'Objective (total)' "runs/best_${name}/train.log" | tail -1 | sed 's/.*| //')"
    echo "    failures: $(grep -ac 'solves failed' "runs/best_${name}/train.log")"
}

mkdir -p runs/best_glide runs/best_landing runs/best_perching
run_task landing      # cheapest (horizon 111), gives an early result
run_task perching     # also horizon 111
run_task glide        # horizon 500, slowest
echo "=== $(date +%H:%M) all three done"
