#!/bin/bash
# Cold, sequential single-task co-design runs for the experiments/ deliverable.
#
# best_landing already finished (249/250 iters) in an earlier session and is left
# untouched. best_glide and best_perching crashed partway (iter 25/150 and 56/250)
# under a solver bug that has since been fixed in ocp.py; this reruns both cold
# under the fixed code. perching_3D reruns cold too since the last completed run
# predates the current physics/OCP code by weeks.
#
# Sequential on purpose -- see scripts/best_designs.sh: two IPOPT-heavy runs at once
# roughly halve each other's throughput on this box (measured on 16 workers), so
# running concurrently would slow every run down, not shorten the wall-clock total.
set -u
cd /home/affinita/PhD/glider_optimization
PY=/home/affinita/PhD/venv/bin/python
LOG=experiments/run_experiments.log

run_task () {
    local conf=$1
    local name=$2
    echo "=== $(date '+%F %H:%M:%S') $name starting" | tee -a "$LOG"
    "$PY" -m glider_optimization.main --config "$conf" --run-name "$name" \
        > "runs/${name}/train.log" 2>&1
    status=$?
    obj=$(grep -a 'Objective (total)' "runs/${name}/train.log" | tail -1 | sed 's/.*| //')
    fails=$(grep -ac 'solves failed' "runs/${name}/train.log")
    echo "--- $(date '+%F %H:%M:%S') $name done (exit $status): $obj  [iters with a failed solve: $fails]" | tee -a "$LOG"
}

run_task conf/best_perching.yaml best_perching
run_task conf/best_glide.yaml    best_glide
run_task conf/perching_3D.yaml   perching3D

echo "=== $(date '+%F %H:%M:%S') all remaining experiments done" | tee -a "$LOG"
