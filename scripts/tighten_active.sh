#!/bin/bash
# Close every front point's margin by making its bound an EQUALITY.
#
# eps is set to each point's own achieved Phi_2, and epsilon_active makes the AL
# drive Phi_2 *onto* that value rather than merely below it. At the optimum of an
# epsilon-subproblem the bound is active anyway, so this is the same problem stated
# in the form the solver can actually converge to: waiting for the inequality
# multiplier to settle onto the boundary left margins of +6 to +14 after 40
# iterations, because the penalty drives into the feasible interior and the dual
# only slowly brings it back.
set -u
cd /home/affinita/PhD/glider_optimization
export GLIDER_EPS_ACTIVE=1
ITERS=${1:-150}

run_point () {
    local tag=$1 eps=$2 seed=$3 mode=$4
    local out="runs/active_${tag}"
    mkdir -p "$out"
    cp runs/pareto_eps8/warm_start.pkl "$out/warm_start.pkl" 2>/dev/null
    echo "=== $tag: eps=$eps (equality) from $seed"
    if [ "$mode" = npz ]; then
        /home/affinita/PhD/venv/bin/python scripts/pareto_chain.py --epsilons "$eps" \
            --iters "$ITERS" --warmup-iters 0 --link-lr 0.0015 --link-gamma 0.999 \
            --seed-npz "$seed" --out "$out" > "$out/chain.log" 2>&1
    else
        /home/affinita/PhD/venv/bin/python scripts/pareto_chain.py --epsilons "$eps" \
            --iters "$ITERS" --warmup-iters 0 --link-lr 0.0015 --link-gamma 0.999 \
            --seed-run "$seed" --out "$out" > "$out/chain.log" 2>&1
    fi
    echo "--- $tag: $(cat "$out/front.json" 2>/dev/null | tr -d '\n ')"
}

run_point a1 195.50 runs/tight_p1/design_eps201.npz npz
run_point a2 201.67 results/pareto_2026-08-26/pareto_eps_fromthick/design_eps215.npz npz
run_point a3 206.69 results/pareto_2026-08-26/pareto_eps9/design_eps216.npz npz
run_point a4 207.14 results/pareto_2026-08-26/pareto_eps9/design_eps217.npz npz
run_point a5 217.81 pchain_eps262 run

echo "=== all active-bound points done"
for t in a1 a2 a3 a4 a5; do
    echo "  $t: $(cat "runs/active_$t/front.json" 2>/dev/null | tr -d '\n ')"
done
