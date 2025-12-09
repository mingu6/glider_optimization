import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import numpy as np
from casadi import pi, vertcat
from tqdm import tqdm
import matplotlib.pyplot as plt

idoc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "Implicit-Diff-Optimal-Control"))
if idoc_path not in sys.path:
    sys.path.append(idoc_path)
import examples.setup_path
from go_safe_pdp import COCsys
import IDOC_ineq as idoc_ineq
from glider_jinenv import GliderPerching

# -----------------------------  Load environment -----------------------------------------
env = GliderPerching()
env.initDyn()
env_dyn = env.X + env.X[-1] * env.f
env.initCost(state_weights=[10., 10., 10., 0.1, 1., 1, 1, 0.1], wu=0.1)
env.initConstraints(-pi/3, pi/8)

# ----------------------------create tunable coc object-----------------------
coc = COCsys()
coc.setAuxvarVariable(vertcat(env.dyn_auxvar, env.constraint_auxvar))
coc.setStateVariable(env.X)
coc.setControlVariable(env.U)
coc.setDyn(env_dyn)
coc.setPathCost(env.path_cost)
coc.setFinalCost(env.final_cost)
coc.setPathInequCstr(env.path_inequ)
coc.diffCPMP()

# ---------------------------- Grid of initial positions ----------------------
init_state = [-3.5, 0.1, 0., 0., 7., 0., 0., 0.01]
x_vals = np.linspace(init_state[0]-3, init_state[0]+3, 50)   
z_vals = np.linspace(init_state[1]-1.5, init_state[1]+1.5, 50)
cost_grid = np.zeros((len(x_vals), len(z_vals)))

init_parameter = [0.158, 0.017, 13]

#traj = coc.ocSolver(horizon=111, init_state=[-5, -0.9, 0., 0., 7., 0., 0., 0.01], auxvar_value=init_parameter, timeVarying=True)
#env.play_animation(traj['state_traj_opt'], traj['control_traj_opt'], save_option=False, title="perching_initial")
# Outer loop: x, inner loop: z
for i, x0 in enumerate(tqdm(x_vals, desc="Evaluating x positions")):
    for j, z0 in enumerate(z_vals):
        curr_init_state = [x0, z0, 0., 0., 7., 0., 0., 0.01]
        traj = coc.ocSolver(horizon=111, init_state=curr_init_state, auxvar_value=init_parameter, timeVarying=True)
        cost_grid[i, j] = min(traj["cost"][0][0], 100)



# ---------------------------- Plot heatmap ----------------------
plt.figure(figsize=(6,5))
plt.imshow(cost_grid.T, origin='lower', extent=[x_vals[0], x_vals[-1], z_vals[0], z_vals[-1]],
           aspect='auto', cmap='viridis')
plt.colorbar(label='Cost')
plt.xlabel('x0')
plt.ylabel('z0')
plt.title('Robustness Heatmap of Glider Perching')
plt.plot(init_state[0], init_state[1], 'r*', markersize=12, label='Original init')
plt.legend()
plt.show()


binary_grid = (cost_grid < 50).astype(int)

plt.figure(figsize=(6,5))
plt.imshow(binary_grid.T, origin='lower',
           extent=[x_vals[0], x_vals[-1], z_vals[0], z_vals[-1]],
           aspect='auto', cmap='gray_r')
plt.colorbar(label='Feasible (1) / Infeasible (0)')
plt.xlabel('x0')
plt.ylabel('z0')
plt.title('Feasibility Map (cost < 50)')
plt.plot(init_state[0], init_state[1], 'r*', markersize=12)
plt.show()
