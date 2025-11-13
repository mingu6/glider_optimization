import casadi
import numpy as np
from casadi import *
import numpy


# This class contains the following  utilities:
# 1) define an parameterized constraint optimal control system
# 2) provide a constraint optimal control solver
# 3) differentiate the Pontryagin maximum principle (differential PMP)
# 4) generate the auxiliary control system
# The variable styles used in this class is the same as the notations in the constraint PDP paper.
class COCsys:

    def __init__(self, project_name='my cpdp object'):
        self.project_name = project_name
        self.inf = 1e20

    # define the tunable parameter (if exists)
    def setAuxvarVariable(self, auxvar=None):
        if auxvar is None or auxvar.numel() == 0:
            self.auxvar = SX.sym('auxvar')
        else:
            self.auxvar = auxvar
        self.n_auxvar = self.auxvar.numel()

    # define the state variable and its allowable upper and lower bounds
    def setStateVariable(self, state, state_lb=None, state_ub=None):
        self.state = state
        self.n_state = self.state.numel()

        if state_lb is not None:
            self.state_lb = state_lb
        else:
            self.state_lb = self.n_state * [-self.inf]

        if state_ub is not None:
            self.state_ub = state_ub
        else:
            self.state_ub = self.n_state * [self.inf]

    # define the control input variable and its allowable upper and lower bounds
    def setControlVariable(self, control, control_lb=None, control_ub=None):
        self.control = control
        self.n_control = self.control.numel()

        if control_lb is not None:
            self.control_lb = control_lb
        else:
            self.control_lb = self.n_control * [-self.inf]

        if control_ub is not None:
            self.control_ub = control_ub
        else:
            self.control_ub = self.n_control * [self.inf]

    # set the dynamics model
    def setDyn(self, ode):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        self.dyn = ode
        self.dyn_fn = casadi.Function('dynamics', [self.state, self.control, self.auxvar], [self.dyn])

    # set the path cost
    def setPathCost(self, path_cost):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        assert path_cost.numel() == 1, "path_cost must be a scalar function"

        self.path_cost = path_cost
        self.path_cost_fn = casadi.Function('path_cost', [self.state, self.control, self.auxvar], [self.path_cost])

    # set the final cost
    def setFinalCost(self, final_cost=None):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        if final_cost is not None:
            self.final_cost = final_cost
        else:
            self.final_cost = 0

        self.final_cost_fn = casadi.Function('final_cost', [self.state, self.auxvar], [self.final_cost])

    # set the path inequality constraints (if exists)
    def setPathInequCstr(self, path_inequality_cstr=None):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        self.path_inequ_cstr = path_inequality_cstr

        if self.path_inequ_cstr is not None:
            self.path_inequ_cstr_fn = casadi.Function('path_inequality_constraint',
                                                      [self.state, self.control, self.auxvar],
                                                      [self.path_inequ_cstr])
            self.n_path_inequ_cstr = self.path_inequ_cstr_fn.numel_out()
        else:
            self.n_path_inequ_cstr = 0

    # set the path equality constraints (if exists)
    def setPathEquCstr(self, path_equality_cstr=None):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        self.path_equ_cstr = path_equality_cstr

        if self.path_equ_cstr is not None:
            self.path_equ_cstr_fn = casadi.Function('path_equality_constraint', [self.state, self.control, self.auxvar],
                                                    [self.path_equ_cstr])
            self.n_path_equ_cstr = self.path_equ_cstr_fn.numel_out()
        else:
            self.n_path_equ_cstr = 0

    # set the final inequality constraints (if exists)
    def setFinalInequCstr(self, final_inequality_cstr=None):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        self.final_inequ_cstr = final_inequality_cstr

        if self.final_inequ_cstr is not None:
            self.final_inequ_cstr_fn = casadi.Function('final_inequality_constraint', [self.state, self.auxvar],
                                                       [self.final_inequ_cstr])
            self.n_final_inequ_cstr = self.final_inequ_cstr_fn.numel_out()
        else:
            self.n_final_inequ_cstr = 0

    # set the final equality constraints (if exists)
    def setFinalEquCstr(self, final_equality_cstr=None):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        self.final_equ_cstr = final_equality_cstr

        if self.final_equ_cstr is not None:
            self.final_equ_cstr_fn = casadi.Function('path_equality_constraint', [self.state, self.auxvar],
                                                     [self.final_equ_cstr])
            self.n_final_equ_cstr = self.final_equ_cstr_fn.numel_out()
        else:
            self.n_final_equ_cstr = 0

    # set the initial condition function (if exists)
    def setInitCondition(self, initial_condition=None):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        self.init_condition = initial_condition

        if self.init_condition is not None:
            self.init_condition_fn = casadi.Function('initial_condition', [self.auxvar], [self.init_condition])

    # optimal control solver (using direct methods)
    def ocSolver(self, horizon, init_state=None, auxvar_value=1, print_level=0, timeVarying=False):

        if not hasattr(self, 'final_equ_cstr'):
            self.setFinalEquCstr()
        if not hasattr(self, 'final_inequ_cstr'):
            self.setFinalInequCstr()
        if not hasattr(self, 'path_equ_cstr'):
            self.setPathEquCstr()
        if not hasattr(self, 'path_inequ_cstr'):
            self.setPathInequCstr()
        if not hasattr(self, 'final_cost'):
            self.setFinalCost()

        if init_state is None:
            init_state = self.init_condition_fn(auxvar_value).full().flatten().tolist()
        else:
            init_state = casadi.DM(init_state).full().flatten().tolist()

        # Start with an empty NLP 
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = MX.sym('X0', self.n_state)
        w += [Xk]
        
        lbw += init_state
        ubw += init_state
        w0 += init_state        
        x0 = init_state
        
        if timeVarying:
            lbw[-1] = 0
            ubw[-1] = self.inf

        # formulate the NLP
        for k in range(horizon):
            # New NLP variable for the control
            Uk = MX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.control_lb
            ubw += self.control_ub
            u0 = [0.5 * (x + y) for x, y in zip(self.control_lb, self.control_ub)]
            w0 += u0

            # Add constraint for the path inequality if exist
            if self.path_inequ_cstr is not None:
                g += [self.path_inequ_cstr_fn(Xk, Uk, auxvar_value)]
                lbg += self.n_path_inequ_cstr * [- self.inf]
                ubg += self.n_path_inequ_cstr * [0]

            # Add constraint for the path equality if exist
            if self.path_equ_cstr is not None:
                g += [self.path_equ_cstr_fn(Xk, Uk, auxvar_value)]
                lbg += self.n_path_equ_cstr * [0]
                ubg += self.n_path_equ_cstr * [0]

            # Integrate till the end of the interval
            Xnext = self.dyn_fn(Xk, Uk, auxvar_value)
            Ck = self.path_cost_fn(Xk, Uk, auxvar_value)
            J = J + Ck

            # New NLP variable for state at end of interval
            Xk = MX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.state_lb
            ubw += self.state_ub
            x0 = self.dyn_fn(x0, u0, auxvar_value).full().ravel().tolist()
            w0 += x0

            # Add constraint for the dynamics
            g += [Xnext - Xk]
            lbg += self.n_state * [0]
            ubg += self.n_state * [0]

        # Add final inequality constraint if exist
        if self.final_inequ_cstr is not None:
            g += [self.final_inequ_cstr_fn(Xk, auxvar_value)]
            lbg += self.n_final_inequ_cstr * [- self.inf]
            ubg += self.n_final_inequ_cstr * [0]

        # Add final equality constraint if exist
        if self.final_equ_cstr is not None:
            g += [self.final_equ_cstr_fn(Xk, auxvar_value)]
            lbg += self.n_final_equ_cstr * [0]
            ubg += self.n_final_equ_cstr * [0]

        sl = MX.sym('Sl', self.n_state)
        w += [sl]
        w0 += [0]*self.n_state
        lbw += self.n_state*[0]
        ubw += self.n_state*[self.inf]

        g += [(Xk - [0,0,0.5,0,0,0,0,0]) - sl]
        g += [-(Xk - [0,0,0.5,0,0,0,0,0]) - sl]
        lbg += 2*self.n_state * [- self.inf]
        ubg += 2*self.n_state * [0]

        # Add the final cost
        J = J + sum1([10., 20., 10., 0, 14, 7, 0, 0.] * sl)

        # Create an NLP solver and solve
        opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)
        # Solve the NLP
        #assert len(lbw) == self.n_state * (horizon+1) + self.n_control * (horizon)
        #assert len(ubw) == self.n_state * (horizon+1) + self.n_control * (horizon)
                
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()
        lam_g = sol['lam_g'].full().flatten()
        g = sol['g'].full().flatten()

        # extract the optimal control and state
        print(w_opt[-self.n_state:])
        w_opt = w_opt[:-self.n_state]
        lam_g = lam_g[:-2*self.n_state]
        g = g[:-2*self.n_state]

        sol_traj = numpy.concatenate((w_opt, self.n_control * [0]))
        sol_traj = numpy.reshape(sol_traj, (-1, self.n_state + self.n_control))
        state_traj_opt = sol_traj[:, 0:self.n_state]
        control_traj_opt = numpy.delete(sol_traj[:, self.n_state:], -1, 0)
        time = numpy.array([k for k in range(horizon + 1)])

        # compute the costate trajectory, mu trajectory and nu trajectory (the latter two is Lagrangian multipliers)
        v_w_lam_path = numpy.reshape(lam_g[:lam_g.size - self.n_final_inequ_cstr - self.n_final_equ_cstr],
                                     (-1, self.n_state + self.n_path_equ_cstr + self.n_path_inequ_cstr))
        v_path = v_w_lam_path[:, 0:self.n_path_inequ_cstr]
        w_path = v_w_lam_path[:, self.n_path_inequ_cstr:self.n_path_inequ_cstr + self.n_path_equ_cstr]
        costate_traj = v_w_lam_path[:, self.n_path_inequ_cstr + self.n_path_equ_cstr:]
        v_final = lam_g[
                  lam_g.size - self.n_final_inequ_cstr - self.n_final_equ_cstr:lam_g.size - self.n_final_equ_cstr]
        w_final = lam_g[lam_g.size - self.n_final_equ_cstr:]

        # compute the inequality constraint value
        g_h_f_traj = numpy.reshape(g[:g.size - self.n_final_inequ_cstr - self.n_final_equ_cstr],
                                   (-1, self.n_state + self.n_path_equ_cstr + self.n_path_inequ_cstr))
        inequ_path = g_h_f_traj[:, 0:self.n_path_inequ_cstr]
        inequ_final = g[lam_g.size - self.n_final_inequ_cstr - self.n_final_equ_cstr:lam_g.size - self.n_final_equ_cstr]

        # output
        opt_sol = {"state_traj_opt": state_traj_opt,
                   "control_traj_opt": control_traj_opt,
                   "costate_traj_opt": costate_traj,
                   "inequ_path": inequ_path,
                   "inequ_final": inequ_final,
                   "v_path": v_path,
                   "v_final": v_final,
                   "w_path": w_path,
                   "w_final": w_final,
                   'auxvar_value': auxvar_value,
                   "time": time,
                   "horizon": horizon,
                   "cost": sol['f'].full()}

        return opt_sol

    # get differential CPMP
    def diffCPMP(self, ):

        if not hasattr(self, 'final_equ_cstr'):
            self.setFinalEquCstr()
        if not hasattr(self, 'final_inequ_cstr'):
            self.setFinalInequCstr()
        if not hasattr(self, 'path_equ_cstr'):
            self.setPathEquCstr()
        if not hasattr(self, 'path_inequ_cstr'):
            self.setPathInequCstr()
        if not hasattr(self, 'init_condition'):
            self.setInitCondition()

        # Differentiating dynamics; notations here are consistent with the PDP paper
        self.dfx = jacobian(self.dyn, self.state)
        self.dfx_fn = casadi.Function('dfx', [self.state, self.control, self.auxvar], [self.dfx])
        self.dfu = jacobian(self.dyn, self.control)
        self.dfu_fn = casadi.Function('dfu', [self.state, self.control, self.auxvar], [self.dfu])
        self.dfe = jacobian(self.dyn, self.auxvar)
        self.dfe_fn = casadi.Function('dfe', [self.state, self.control, self.auxvar], [self.dfe])

        # Define the path Hamiltonian-Lagrangian function (path_LP)
        costate = casadi.SX.sym('lambda', self.n_state)
        v_path = casadi.SX.sym('v_path', self.n_path_inequ_cstr)
        w_path = casadi.SX.sym('nu_path', self.n_path_equ_cstr)
        # specifically handle the none inequality and equality cases
        self.L_path = self.path_cost + dot(self.dyn, costate)
        if self.path_inequ_cstr is not None:
            self.L_path = self.L_path + dot(self.path_inequ_cstr, v_path)
        if self.path_equ_cstr is not None:
            self.L_path = self.L_path + dot(self.path_equ_cstr, w_path)

        # First-order derivative of path Hamiltonian-Lagrangian
        self.dLx_path = jacobian(self.L_path, self.state).T
        self.dLx_path_fn = casadi.Function('dLx_path',
                                           [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                           [self.dLx_path])
        self.dLu_path = jacobian(self.L_path, self.control).T
        self.dLu_path_fn = casadi.Function('dLu_path',
                                           [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                           [self.dLu_path])
        self.dLe_path = jacobian(self.L_path, self.auxvar).T
        self.dLe_path_fn = casadi.Function('dLe_path',
                                           [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                           [self.dLe_path])

        # Second-order derivative of Hamiltonian-Lagrangian
        self.ddLxx_path = jacobian(self.dLx_path, self.state)
        self.ddLxx_path_fn = casadi.Function('ddLxx_path',
                                             [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                             [self.ddLxx_path])
        self.ddLxu_path = jacobian(self.dLx_path, self.control)
        self.ddLxu_path_fn = casadi.Function('ddLxu_path',
                                             [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                             [self.ddLxu_path])
        self.ddLxe_path = jacobian(self.dLx_path, self.auxvar)
        self.ddLxe_path_fn = casadi.Function('ddLxe_path',
                                             [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                             [self.ddLxe_path])
        self.ddLux_path = jacobian(self.dLu_path, self.state)
        self.ddLux_path_fn = casadi.Function('ddLux_path',
                                             [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                             [self.ddLux_path])
        self.ddLuu_path = jacobian(self.dLu_path, self.control)
        self.ddLuu_path_fn = casadi.Function('ddLuu_path',
                                             [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                             [self.ddLuu_path])
        self.ddLue_path = jacobian(self.dLu_path, self.auxvar)
        self.ddLue_path_fn = casadi.Function('ddHue_path',
                                             [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                             [self.ddLue_path])

        # Define the final Hamiltonian-Lagrangian function (final_H)
        v_final = casadi.SX.sym('v_final', self.n_final_inequ_cstr)
        w_final = casadi.SX.sym('nu_final', self.n_final_equ_cstr)
        # specifically handle the none inequality and equality cases
        self.L_final = self.final_cost
        if self.final_inequ_cstr is not None:
            self.L_final = self.L_final + dot(self.final_inequ_cstr, v_final)
        if self.final_equ_cstr is not None:
            self.L_final = self.L_final + dot(self.final_equ_cstr, w_final)

        # First-order derivative of final Hamiltonian-Lagrangian
        self.dLx_final = jacobian(self.L_final, self.state).T
        self.dLx_final_fn = casadi.Function('dLx_final',
                                            [self.state, v_final, w_final, self.auxvar],
                                            [self.dLx_final])

        # Second order differential of final Hamiltonian-Lagrangian
        self.ddLxx_final = jacobian(self.dLx_final, self.state)
        self.ddLxx_final_fn = casadi.Function('ddLxx_final', [self.state, v_final, w_final, self.auxvar],
                                              [self.ddLxx_final])
        self.ddLxe_final = jacobian(self.dLx_final, self.auxvar)
        self.ddLxe_final_fn = casadi.Function('ddLxe_final', [self.state, v_final, w_final, self.auxvar],
                                              [self.ddLxe_final])

        # differentiate the path equality constraint if exist
        if self.path_equ_cstr is not None:
            self.dHx_path = jacobian(self.path_equ_cstr, self.state)
            self.dHx_path_fn = Function('dHx_path_fn', [self.state, self.control, self.auxvar],
                                        [self.dHx_path])
            self.dHu_path = jacobian(self.path_equ_cstr, self.control)
            self.dHu_path_fn = Function('dHu_path_fn', [self.state, self.control, self.auxvar],
                                        [self.dHu_path])
            self.dHe_path = jacobian(self.path_equ_cstr, self.auxvar)
            self.dHe_path_fn = Function('dHe_path_fn', [self.state, self.control, self.auxvar],
                                        [self.dHe_path])

        # differentiate the final equality constraint if exist
        if self.final_equ_cstr is not None:
            self.dHx_final = jacobian(self.final_equ_cstr, self.state)
            self.dHx_final_fn = Function('dHx_final_fn', [self.state, self.auxvar],
                                         [self.dHx_final])
            self.dHe_final = jacobian(self.final_equ_cstr, self.auxvar)
            self.dHe_final_fn = Function('dHe_final_fn', [self.state, self.auxvar],
                                         [self.dHe_final])

        # differentiate the path inequality constraint if exist
        if self.path_inequ_cstr is not None:
            self.dGx_path = jacobian(self.path_inequ_cstr, self.state)
            self.dGx_path_fn = Function('dGx_path_fn', [self.state, self.control, self.auxvar],
                                        [self.dGx_path])
            self.dGu_path = jacobian(self.path_inequ_cstr, self.control)
            self.dGu_path_fn = Function('dGu_path_fn', [self.state, self.control, self.auxvar],
                                        [self.dGu_path])
            self.dGe_path = jacobian(self.path_inequ_cstr, self.auxvar)
            self.dGe_path_fn = Function('dGe_path_fn', [self.state, self.control, self.auxvar],
                                        [self.dGe_path])

        # differentiate the final inequality constraint if exist
        if self.final_inequ_cstr is not None:
            self.dGx_final = jacobian(self.final_inequ_cstr, self.state)
            self.dGx_final_fn = Function('dHx_final_fn', [self.state, self.auxvar],
                                         [self.dGx_final])
            self.dGe_final = jacobian(self.final_inequ_cstr, self.auxvar)
            self.dGe_final_fn = Function('dHe_final_fn', [self.state, self.auxvar],
                                         [self.dGe_final])

        # differentiate the initial condition if parameterized
        if self.init_condition is not None:
            self.dX0 = jacobian(self.init_condition, self.state)
            self.dx0_fn = Function('dx0_fn', [self.auxvar], [self.dX0])
        else:
            self.dX0_fn = Function('dx0_fn', [self.auxvar], [SX.zeros(self.n_state, self.n_auxvar)])

    # get the auxiliary control system (here the threshold is to determine the active inequality constraints
    def getAuxSys(self, opt_sol, threshold=1e-2):

        # parse the optimal solution argument opt_sol
        state_traj_opt = opt_sol['state_traj_opt']
        control_traj_opt = opt_sol['control_traj_opt']
        costate_traj = opt_sol['costate_traj_opt']
        auxvar_value = opt_sol['auxvar_value']

        v_path = opt_sol['v_path']
        g_path = opt_sol['inequ_path']
        v_final = opt_sol['v_final']
        g_final = opt_sol['inequ_final']
        w_path = opt_sol['w_path']
        w_final = opt_sol['w_final']

        # in case of not differentiating the PMP
        if not hasattr(self, 'dLx_path'):
            self.diffCPMP()

        # Initialize the coefficient matrices of the auxiliary control system: note that all the notations used here are
        # consistent with the notations defined in the constraint PDP paper.
        dynFx_t, dynFu_t, dynFe_t = [], [], []
        Lxx_t, Lxu_t, Lxe_t, Lux_t, Luu_t, Lue_t = [], [], [], [], [], []
        GbarHx_t, GbarHu_t, GbarHe_t = [], [], []  # this is the concatenated matrix G_bar_t and H_t in the constraint PDP paper
        GbarHx_T, GbarHe_T, = [], []  # this is the concatenated matrix G_bar_T and H_T in the constraint PDP paper

        horizon = numpy.size(control_traj_opt, 0)
        for t in range(horizon):
            curr_x = state_traj_opt[t, :]
            curr_u = control_traj_opt[t, :]
            next_lambda = costate_traj[t, :]
            curr_v = v_path[t, :]
            curr_w = w_path[t, :]
            curr_g = g_path[t, :]

            dynFx_t += [self.dfx_fn(curr_x, curr_u, auxvar_value).full()]
            dynFu_t += [self.dfu_fn(curr_x, curr_u, auxvar_value).full()]
            dynFe_t += [self.dfe_fn(curr_x, curr_u, auxvar_value).full()]

            Lxx_t += [self.ddLxx_path_fn(curr_x, curr_u, next_lambda, curr_v, curr_w, auxvar_value).full()]
            Lxu_t += [self.ddLxu_path_fn(curr_x, curr_u, next_lambda, curr_v, curr_w, auxvar_value).full()]
            Lxe_t += [self.ddLxe_path_fn(curr_x, curr_u, next_lambda, curr_v, curr_w, auxvar_value).full()]

            Lux_t += [self.ddLux_path_fn(curr_x, curr_u, next_lambda, curr_v, curr_w, auxvar_value).full()]
            Luu_t += [self.ddLuu_path_fn(curr_x, curr_u, next_lambda, curr_v, curr_w, auxvar_value).full()]
            Lue_t += [self.ddLue_path_fn(curr_x, curr_u, next_lambda, curr_v, curr_w, auxvar_value).full()]

            # generate the G_bar_t and H_t, where B_bar_t is identified using the threshold
            if self.path_inequ_cstr is not None:
                Gbarx_t = self.dGx_path_fn(curr_x, curr_u, auxvar_value).full()[(curr_g > -threshold)]
                Gbaru_t = self.dGu_path_fn(curr_x, curr_u, auxvar_value).full()[(curr_g > -threshold)]
                Gbare_t = self.dGe_path_fn(curr_x, curr_u, auxvar_value).full()[(curr_g > -threshold)]
            else:
                Gbarx_t = np.empty((0, self.n_state))
                Gbaru_t = np.empty((0, self.n_control))
                Gbare_t = np.empty((0, self.n_auxvar))

            if self.path_equ_cstr is not None:
                Hx_t = self.dHx_path_fn(curr_x, curr_u, auxvar_value).full()
                Hu_t = self.dHu_path_fn(curr_x, curr_u, auxvar_value).full()
                He_t = self.dHe_path_fn(curr_x, curr_u, auxvar_value).full()
            else:
                Hx_t = np.empty((0, self.n_state))
                Hu_t = np.empty((0, self.n_control))
                He_t = np.empty((0, self.n_auxvar))

            GbarHx_t += [np.vstack((Gbarx_t, Hx_t))]
            GbarHu_t += [np.vstack((Gbaru_t, Hu_t))]
            GbarHe_t += [np.vstack((Gbare_t, He_t))]

        # handle the final cost, inequality, equality constraints
        Lxx_T = [self.ddLxx_final_fn(state_traj_opt[-1, :], v_final, w_final, auxvar_value).full()]
        Lxe_T = [self.ddLxe_final_fn(state_traj_opt[-1, :], v_final, w_final, auxvar_value).full()]

        if self.final_inequ_cstr is not None:
            Gbarx_T = self.dGx_final_fn(state_traj_opt[-1, :], auxvar_value).full()[(v_final > -threshold)]
            Gbare_T = self.dGe_final_fn(state_traj_opt[-1, :], auxvar_value).full()[(g_final > -threshold)]
        else:
            Gbarx_T = np.empty((0, self.n_state))
            Gbare_T = np.empty((0, self.n_auxvar))

        if self.final_equ_cstr is not None:
            Hx_T = self.dHx_final_fn(state_traj_opt[-1, :], auxvar_value).full()
            He_T = self.dHe_final_fn(state_traj_opt[-1, :], auxvar_value).full()
        else:
            Hx_T = np.empty((0, self.n_state))
            He_T = np.empty((0, self.n_auxvar))

        GbarHx_T += [np.vstack((Gbarx_T, Hx_T))]
        GbarHe_T += [np.vstack((Gbare_T, He_T))]

        # print(GbarHx_T, GbarHe_T)

        # return the axuliary control system
        X0 = self.dX0_fn(auxvar_value).full()
        auxSys = {"dynFx_t": dynFx_t,
                  "dynFu_t": dynFu_t,
                  "dynFe_t": dynFe_t,
                  "Lxx_t": Lxx_t,
                  "Lxu_t": Lxu_t,
                  "Lxe_t": Lxe_t,
                  "Lux_t": Lux_t,
                  "Luu_t": Luu_t,
                  "Lue_t": Lue_t,
                  "Lxx_T": Lxx_T,
                  "Lxe_T": Lxe_T,
                  "GbarHx_t": GbarHx_t,
                  "GbarHu_t": GbarHu_t,
                  "GbarHe_t": GbarHe_t,
                  "GbarHx_T": GbarHx_T,
                  "GbarHe_T": GbarHe_T,
                  "X0": X0,
                  "horizon": horizon
                  }

        return auxSys
