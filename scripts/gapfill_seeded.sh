#!/bin/bash
# Fill the front's glide gap by starting epsilon-constrained runs from inside it.
#
# The gap (glide 24 -> 44, Phi_2 207 -> 218) cannot be crossed by continuation from
# either end: from the thick basin the design stays at glide ~24 whatever epsilon
# allows, and from the thin one Phi_2 only ever rises above 217.8 (that basin's best
# landing cost), so a bound below it has no descent direction to follow.
#
# So each run starts from a point between the two known designs and lets the
# epsilon-constrained optimiser produce the design from there. The starting point is
# only a starting point -- every reported design is an optimisation output, as with
# every other point on the front.
set -u
cd /home/affinita/PhD/glider_optimization
ITERS=${1:-12}

run_point () {
    local tag=$1 eps=$2 seed=$3
    local out="runs/gap_${tag}"
    mkdir -p "$out"
    cp runs/pareto_eps8/warm_start.pkl "$out/warm_start.pkl" 2>/dev/null
    echo "=== $tag: eps=$eps from $seed"
    /home/affinita/PhD/venv/bin/python scripts/pareto_chain.py --epsilons "$eps" \
        --iters "$ITERS" --warmup-iters 0 --link-lr 0.0015 --link-gamma 0.999 \
        --seed-npz "$seed" --out "$out" > "$out/chain.log" 2>&1
    echo "--- $tag: $(cat "$out/front.json" 2>/dev/null | tr -d '\n ')"
}

run_point g214 214.0 runs/seed_mid_35.npz
run_point g212 212.0 runs/seed_mid_55.npz
run_point g216 216.0 runs/seed_mid_75.npz

echo "=== gap-fill done"
for t in g214 g212 g216; do
    echo "  $t: $(cat "runs/gap_$t/front.json" 2>/dev/null | tr -d '\n ')"
done
