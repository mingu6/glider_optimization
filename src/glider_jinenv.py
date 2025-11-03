from casadi import sin, cos, SX, vcat, vertcat, atan2, sqrt, diag, gradient, Function
import numpy as np

class GliderPerching :
    def __init__(self, project_name='glider-perching'):
        self.project_name = project_name

    def C_L(self, alpha):
        return 2 * sin(alpha) * cos(alpha)

    def C_D(self, alpha):
        return 2 * sin(alpha) * sin(alpha)

    def C_M(self, alpha):
        return -self.C_L(alpha) * 0.25

    def mc_to_wcom(self, l_w):
        return l_w+0.003

    def initDyn(self):
        # set the global parameters
        m = 0.065
        l_w_i = -0.005                                   # vector from CoM to centroid of wing (positive means wing is in front of CoM)
        l_w_f = -0.015                                   # vector from CoM to centroid of wing (positive means wing is in front of CoM)
        l = 0.26                                        # vector from CoM to start of elevator (attachment point to body)
        l_e = 0.02                                      # distance to centroid of elevator from start (attachment point to body)
        rho = 1.2041                                    # assume 20 degrees C
        m_f = 0.4 * m                                   # mass of fuselage
        l_w = 0.5*(l_w_i+l_w_f)
        g = 9.81

        # declare system parameters
        self.S_w = SX.sym('S_w')
        self.S_e = SX.sym('S_e')
                
        parameter = [self.S_w, self.S_e]
        self.dyn_auxvar = vcat(parameter)

        m_w = 0.6 * m * self.S_w / (self.S_w + self.S_e)
        m_e = 0.6 * m * self.S_e / (self.S_w + self.S_e)
        l_f = -(l_w * m_w + (l - l_e) * m_e) / m_f      # vector to fuselage CoM
        I = m_w * l_w ** 2 + m_e * (l + l_e) ** 2 + m_f * l_f ** 2
        
        # Declare system variables
        x = SX.sym("x")
        z = SX.sym("z")
        theta = SX.sym("theta")
        phi = SX.sym("phi")
        xdot = SX.sym("xdot")
        zdot = SX.sym("zdot")
        thetadot = SX.sym("thetadot")
        phidot = SX.sym("phidot")

        self.X = vertcat(x, z, theta, phi, xdot, zdot, thetadot)
        self.U = phidot

        # wing mean chord 
        l_w_m = (l_w_i + l_w_f) / 2

        com_w = l_w_m + self.mc_to_wcom(l_w_m)
        com_e = l + l_e # simplifying assumption, the elevator's com doesn't depend on the angle (quasi static assumption)                
        com_f = l_f
        com_a = (com_w*m_w + com_e*m_e + com_f*m_f) / (m_w + m_e + m_f)

        # geometric centroid of aerodynamic surfaces (mean chord for flat plate)
        x_wdot = xdot + l_w_m * thetadot * sin(theta)
        z_wdot = zdot - l_w_m * thetadot * cos(theta)
        x_edot = xdot + l * thetadot * sin(theta) + l_e * (thetadot + phidot) * sin(theta + phi)
        z_edot = zdot - l * thetadot * cos(theta) - l_e * (thetadot + phidot) * cos(theta + phi)
        
        # force vectors for aerodynamic surfaces (lift, drag, gravity)

        c = np.abs(l_w_f - l_w_i)
        alpha_w = theta - atan2(z_wdot, x_wdot)
        v_w = sqrt(x_wdot * x_wdot + z_wdot * z_wdot + 1e-8) # flow/air speed
        F_Lw = self.C_L(alpha_w) * vertcat(-z_wdot, x_wdot)  # lift force vector (proportional to)
        F_Dw = self.C_D(alpha_w) * vertcat(-x_wdot, -z_wdot) # drag force vector (proportional to)
        F_w = 0.5 * rho * v_w * self.S_w * (F_Lw + F_Dw)
        M_w = 0.5 * rho * v_w**2 * self.S_w * c * self.C_M(alpha_w)

        alpha_e = theta + phi - atan2(z_edot, x_edot)
        v_e = sqrt(x_edot * x_edot + z_edot * z_edot + 1e-8)    # flow/air speed
        F_Le = self.C_L(alpha_e) * vertcat(-z_edot, x_edot)          # lift force vector (proportional to)
        F_De = self.C_D(alpha_e) * vertcat(-x_edot, -z_edot)         # drag force vector (proportional to)
        F_e = 0.5 * rho * v_e * self.S_e * (F_Le + F_De)
        M_e = 0.5 * rho * v_e**2 * self.S_e * c * self.C_M(alpha_e)

        # compute torques with respect to fixed reference point induced by forces

        # moment arms (vector from reference point of state to wing/elevator/fuselage)
        r_w = [ (- com_w + com_a) * cos(theta), (- com_w + com_a) * sin(theta) ]
        r_e = [ (- com_e + com_a) * cos(theta), (- com_e + com_a) * sin(theta)]

        τ_w = r_w[1] * F_w[0] - r_w[0] * F_w[1] + M_w
        τ_e = r_e[1] * F_e[0] - r_e[0] * F_e[1] + M_e
        thetaddot = -1. / I * (τ_w + τ_e)

        # linear accelerations (F = ma)
        xddot = 1. / m * (F_w[0] + F_e[0])
        zddot = 1. / m * (F_w[1] + F_e[1]) - g
        
        self.f = vertcat(xdot, zdot, thetadot, phidot, xddot, zddot, thetaddot)

    def initCost(self, state_weights, wu=0.001):
        X_goal = [0., 0., 0., 0., 0., 0., 0.]

        self.cost_auxvar = vcat([])

        err = self.X - X_goal
        self.path_cost = wu * (self.U * self.U) + err.T @ diag(state_weights)*0.001 @ err 
        self.dpath_cost_dx = gradient(self.path_cost, self.X)
        self.dpath_cost_du = gradient(self.path_cost, self.U)
        
        self.final_cost = err.T @ diag(state_weights) @ err

        self.dfinal_cost_dx = gradient(self.final_cost, self.X)
        self.dfinal_cost_dx_fn = Function("dfinal_cost_dx_fn", [self.X], [self.dfinal_cost_dx])
        self.dfinal_cost_du_fn = Function("dfinal_cost_du_fn", [self.X], [self.dfinal_cost_dx])
        
    def initConstraints(self, min_phi, max_phi, max_u=None):
        # set path constraint h_final(x)
        constraint_auxvar = []
        if max_u is None:
            max_u = SX.sym('max_u')
            constraint_auxvar += [max_u]

        self.constraint_auxvar = vcat(constraint_auxvar)

        path_inequ_Uub = self.U - max_u
        path_inequ_Ulb = -self.U - max_u
        path_inequ_Xub = self.X[3] - max_phi
        path_inequ_Xlb = -self.X[3] + min_phi
        self.path_inequ = vcat([path_inequ_Uub, path_inequ_Ulb, path_inequ_Xub, path_inequ_Xlb])

    def play_animation(self, pole_len, dt, state_traj, state_traj_ref=None, save_option=0, title='glider-perching'):
        pass