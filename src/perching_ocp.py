import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import time
import sys

import numpy as np
import matplotlib.pyplot as plt
from casadi import pi, vertcat


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
    dt = 0.01

    # -----------------------------  Load environment -----------------------------------------
    env = GliderPerching()
    env.initDyn()
    env_dyn = env.X + dt * env.f
    env.initCost(state_weights=[10., 10., 10., 0.1, 1., 1, 1], wu=0.1)
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
    gamma = 1e-2
    coc.convert2BarrierOC(gamma=gamma)

    # ----------------------------main learning procedure ----------------------
    # initial guess
    sigma = 0.05
    nn_seed = 100
    np.random.seed(nn_seed)
    init_parameter = [0.086, 0.022, 13]

    # learning rate and maximum iteration
    lr = 5e-7
    max_iter = 170

    loss = 0
    grad = 0
    
    current_parameter_COC = init_parameter

    init_state = [-3.5, 0.1 , 0. , 0., 7., 0. , 0.]

    loss_history = []
    param1_history = []
    param2_history = []

    for k in range(max_iter):
        traj_COC = coc.ocSolver(horizon=111, init_state=init_state, auxvar_value=current_parameter_COC)
        auxsys_COC = coc.getAuxSys(opt_sol=traj_COC, threshold=1e-5)
        idoc_blocks = idoc_ineq.build_blocks_idoc(auxsys_COC, args.delta)

        if args.method == 'full':
            start = time.time()
            traj_deriv_COC = idoc_ineq.idoc_full(*idoc_blocks)
            loss = traj_COC["cost"]
            grad = gradient(env, traj_COC, traj_deriv_COC)
            time_grad = time.time() - start
        else:
            start = time.time()
            loss = traj_COC["cost"]
            grad = 0
            time_grad = time.time() - start

        loss_history.append(float(loss))
        param1_history.append(float(current_parameter_COC[0]))
        param2_history.append(float(current_parameter_COC[1]))

        print('iter #:', k, ' loss_COC:', loss)
        current_parameter_COC = current_parameter_COC - lr * grad

    # Plot loss
    plt.figure(figsize=(6,4))
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss evolution during training')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot first two parameters
    plt.figure(figsize=(6,4))
    plt.plot(param1_history, label='Surface Wing')
    plt.plot(param2_history, label='Surface Elevator')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title('First two parameters evolution')
    plt.grid(True)
    plt.legend()
    plt.show()