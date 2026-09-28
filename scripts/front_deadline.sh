#!/bin/bash
# Stop the epsilon-constraint chain at the cutoff and report the front from the
# optimisation's own outputs. No post-hoc re-measurement: every number printed here
# is what the epsilon-constrained run itself reported at the end of its link.
CUTOFF=${1:-18:00}
cd /home/affinita/PhD/glider_optimization

while [ "$(date +%H:%M)" \< "$CUTOFF" ]; do
    if ! ps -eo args | grep -q '[p]areto_chain.py'; then
        echo "$(date +%H:%M) chain finished on its own"
        break
    fi
    sleep 60
done

if ps -eo args | grep -q '[p]areto_chain.py'; then
    echo "$(date +%H:%M) cutoff reached; stopping chain"
    ps -eo pid,args | grep -E "[p]areto_chain.py|pchain_eps" | awk '{print $1}' | xargs -r kill -9
    sleep 5
fi

echo "$(date +%H:%M) FRONT (from the epsilon-constraint runs):"
cat runs/pareto_eps7/front.json 2>/dev/null || echo "no completed links"
echo "$(date +%H:%M) front reported"
