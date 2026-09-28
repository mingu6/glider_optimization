#!/bin/bash
# Drive each front point's margin to zero.
#
# A point with a slack margin was never shaped by its bound: the optimiser found a
# local unconstrained optimum that happens to sit at that Phi_2. Re-running with
# eps set to the point's OWN achieved Phi_2 makes the constraint active from
# iteration 0, so the optimiser must push glide until it cannot do so without
# violating -- which is the actual solution of the epsilon-subproblem.
#
# Each point is seeded from its own design npz (not a wandb run name: several runs
# share names, and picking the wrong one silently changes the experiment).
set -u
cd /home/affinita/PhD/glider_optimization
ITERS=${1:-40}

run_point () {
    local tag=$1 eps=$2 npz=$3
    local out="runs/tight_${tag}"
    mkdir -p "$out"
    cp runs/pareto_eps8/warm_start.pkl "$out/warm_start.pkl" 2>/dev/null
    echo "=== $tag: eps=$eps seeded from $npz"
    /home/affinita/PhD/venv/bin/python scripts/pareto_chain.py \
        --epsilons "$eps" --iters "$ITERS" --warmup-iters 0 \
        --link-lr 0.0015 --link-gamma 0.999 \
        --seed-npz "$npz" --out "$out" > "$out/chain.log" 2>&1
    echo "--- $tag result: $(cat $out/front.json 2>/dev/null | tr -d '\n ')"
}

# eps = each point's own achieved Phi_2, so the bound starts exactly active
run_point p1 201.67 results/pareto_2026-08-26/pareto_eps_fromthick/design_eps215.npz
run_point p2 206.69 results/pareto_2026-08-26/pareto_eps9/design_eps216.npz
run_point p3 212.11 results/pareto_2026-08-26/pareto_eps8/design_eps216.npz
run_point p4 217.81 results/pareto_2026-08-26/pareto_eps9/design_eps217.npz

echo "=== all tight-margin points done"
for t in p1 p2 p3 p4; do
    echo "  $t: $(cat runs/tight_$t/front.json 2>/dev/null | tr -d '\n ')"
done
