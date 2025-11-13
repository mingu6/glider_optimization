import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import time
import sys

import numpy as np
import matplotlib.pyplot as plt
from casadi import pi, vertcat
from tqdm import trange
import asciichartpy as ac

idoc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "Implicit-Diff-Optimal-Control"))
if idoc_path not in sys.path:
    sys.path.append(idoc_path)
import examples.setup_path
from go_safe_pdp import COCsys
import IDOC_ineq as idoc_ineq
from glider_jinenv import GliderPerching
import numpy


def gradient(env, traj, traj_deriv):
    state_traj = traj['state_traj_opt']
    dldx_traj = env.dfinal_cost_dx_fn(state_traj[-1])
    dxdp_traj = traj_deriv['state_traj_opt']
    grad = numpy.matmul(dldx_traj.T, dxdp_traj[-1])
    
    return grad.T

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', '-m', type=str, required=True, choices=['full', 'vjp'], help='method used to differentiate through inner problem')
    parser.add_argument('--delta', '-d', default=0., type=float)
    args = parser.parse_args()

    # -----------------------------  Load environment -----------------------------------------
    env = GliderPerching()
    env.initDyn()
    env_dyn = env.X + env.X[-1] * env.f
    env.initCost(state_weights=[10., 10., 10., 0.1, 1., 1, 1, 0.1], wu=0.1)
    env.initConstraints(-pi/3, pi/8)

    # ----------------------------create tunable coc object-----------------------
    coc = COCsys()
    # pass the system to coc
    coc.setAuxvarVariable(vertcat(env.dyn_auxvar, env.constraint_auxvar))
    coc.setStateVariable(env.X)
    coc.setControlVariable(env.U)
    coc.setDyn(env_dyn)

    coc.setPathCost(env.path_cost)
    coc.setFinalCost(env.final_cost)

    coc.setPathInequCstr(env.path_inequ)
    # differentiating CPMP
    coc.diffCPMP()

    # ----------------------------main learning procedure ----------------------
    # initial guess
    nn_seed = 100
    np.random.seed(nn_seed)
    init_parameter = [0.086, 0.022, 13]

    # learning rate and maximum iteration
    lr = 1e-7
    max_iter = 20000

    loss = 0
    grad = 0
    
    best_loss_idx = 0
    best_loss = float("inf")
    
    current_parameter_COC = init_parameter
    save_results = False

    init_state = [-3.5, 0.1 , 0. , 0., 7., 0. , 0., 0.01]

    loss_history = []
    param_history = []
    
    
    traj_COC = coc.ocSolver(horizon=111, init_state=init_state, auxvar_value=current_parameter_COC, timeVarying=True)
    env.play_animation(traj_COC['state_traj_opt'], traj_COC['control_traj_opt'], save_option=save_results, title="perching_initial")
    exit(0)    

    pbar = trange(max_iter, desc="Training", ncols=100)
    for k in pbar:
        traj_COC = coc.ocSolver(horizon=111, init_state=init_state, auxvar_value=current_parameter_COC, timeVarying=True)
        auxsys_COC = coc.getAuxSys(opt_sol=traj_COC, threshold=1e-5)
        idoc_blocks = idoc_ineq.build_blocks_idoc(auxsys_COC, args.delta)
        if args.method == 'full':
            start = time.time()
            traj_deriv_COC = idoc_ineq.idoc_full(*idoc_blocks)
            loss = traj_COC["cost"][0][0]
            grad = gradient(env, traj_COC, traj_deriv_COC)
            time_grad = time.time() - start
        else:
            start = time.time()
            loss = traj_COC["cost"][0][0]
            grad = 0
            time_grad = time.time() - start

        loss_history.append(min(float(loss), best_loss))
        param_history.append(current_parameter_COC)
        
        if loss_history[-1] < loss_history[best_loss_idx]:
            best_loss_idx = k
            best_loss = float(loss)

        if k % 50 == 0:
            os.system('clear') 
            print(f"Iter {k+1}/{max_iter} | Loss: {loss_history[-1]:.6f}")
            print(ac.plot(loss_history[-100:], {'height': 10}))

        pbar.set_description(f"Iter {k+1}/{max_iter} | Loss: {loss_history[-1]:.6f}")
        current_parameter_COC = current_parameter_COC - lr * grad
    
    
    param_history = [
        np.array(x.full().squeeze(), dtype=float).tolist() if hasattr(x, "full") else x
        for x in param_history
    ]
    param_history = np.array(param_history, dtype=np.float32)
    
    def moving_average(data, window_size=20):
        data = np.array(data, dtype=float)
        avg = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window_size + 1)
            avg[i] = np.mean(data[start:i+1])
        return avg


    traj_COC = coc.ocSolver(horizon=111, init_state=init_state, auxvar_value=param_history[best_loss_idx], timeVarying=True)
    env.play_animation(traj_COC['state_traj_opt'], traj_COC['control_traj_opt'], save_option=save_results, title="perching_final")

    # Plot loss
    plt.figure(figsize=(6,4))
    line, = plt.plot(loss_history, label='Loss', alpha=1)
    #plt.plot(moving_average(loss_history, 20), color=line.get_color(), linewidth=2, label='Moving Avg')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss evolution during training')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot first two parameters
    plt.figure(figsize=(6,4))
    plt.plot(param_history[:,0], label='Surface Wing')
    plt.plot(param_history[:,1], label='Surface Elevator')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title('First two parameters evolution')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    print(f"Best loss obtained {loss_history[best_loss_idx]}")
    print(f"Best param {param_history[best_loss_idx]}")
    