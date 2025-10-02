import casadi as cs
import numpy as np
from utils.live_plot import live_plot_landing

def C_L(alpha):
    return 2 * cs.sin(alpha) * cs.cos(alpha)

def C_D(alpha):
    return 2 * cs.sin(alpha) * cs.sin(alpha)

def C_M(alpha):
    return -C_L(alpha) * 0.25

# Mean Chrod to Wing Center Of Mass
def mc_to_wcom(l_w):
    # should depend on the airfoil's shape
    return l_w+0.003

def construct_dyn_state(t):
    x = cs.SX.sym("x_" + str(t))
    z = cs.SX.sym("z_" + str(t))
    theta = cs.SX.sym("theta_" + str(t))
    phi = cs.SX.sym("phi_" + str(t))
    xdot = cs.SX.sym("xdot_" + str(t))
    zdot = cs.SX.sym("zdot_" + str(t))
    thetadot = cs.SX.sym("thetadot_" + str(t))
    return cs.vertcat(x, z, theta, phi, xdot, zdot, thetadot)

def construct_forces(t):
    fn1 = cs.SX.sym(f"fn1_{t}")
    fn2 = cs.SX.sym(f"fn2_{t}")

    sl1 = cs.SX.sym(f"sl1_{t}")
    sl2 = cs.SX.sym(f"sl2_{t}")

    return cs.vertcat(fn1, fn2), cs.vertcat(sl1, sl2)

def build_dynamic_model(params):
    # The state vector is expressed wrt a static point on the aircraft called Body Reference Point (BRP)
    x = cs.SX.sym("x")
    z = cs.SX.sym("z")
    theta = cs.SX.sym("theta")
    phi = cs.SX.sym("phi")
    xdot = cs.SX.sym("xdot")
    zdot = cs.SX.sym("zdot")
    thetadot = cs.SX.sym("thetadot")
    fn1 = cs.SX.sym("Fn1")
    fn2 = cs.SX.sym("Fn2")
    phidot = cs.SX.sym("phidot")

    states = cs.vertcat(x, z, theta, phi, xdot, zdot, thetadot)
    forces = cs.vertcat(fn1, fn2)
    controls = phidot

    # the l* quantities are expressed along the x of the BRP RF.
    l_w_i, l_w_f , l, l_e, l_f, m_w, m_e, m_f, rho, S_w, S_e, m, g, I = params

    # wing mean chord 
    l_w_m = (l_w_i + l_w_f) / 2

    com_w = l_w_m + mc_to_wcom(l_w_m)
    com_e = l + l_e # simplifying assumption, the elevator's com doesn't depend on the angle (quasi static assumption)                
    com_f = l_f
    com_a = (com_w*m_w + com_e*m_e + com_f*m_f) / (m_w + m_e + m_f)

    # geometric centroid of aerodynamic surfaces (mean chord for flat plate)
    x_wdot = xdot + l_w_m * thetadot * cs.sin(theta)
    z_wdot = zdot - l_w_m * thetadot * cs.cos(theta)
    x_edot = xdot + l * thetadot * cs.sin(theta) + l_e * (thetadot + phidot) * cs.sin(theta + phi)
    z_edot = zdot - l * thetadot * cs.cos(theta) - l_e * (thetadot + phidot) * cs.cos(theta + phi)
    
    # force vectors for aerodynamic surfaces (lift, drag, gravity)

    c = np.abs(l_w_f - l_w_i)
    alpha_w = theta - cs.atan2(z_wdot, x_wdot)
    v_w = cs.sqrt(x_wdot * x_wdot + z_wdot * z_wdot + 1e-8)             # flow/air speed
    F_Lw = C_L(alpha_w) * cs.vertcat(-z_wdot, x_wdot)                   # lift force vector (proportional to)
    F_Dw = C_D(alpha_w) * cs.vertcat(-x_wdot, -z_wdot)                  # drag force vector (proportional to)
    F_w = 0.5 * rho * v_w * S_w * (F_Lw + F_Dw)
    M_w = 0.5 * rho * v_w**2 * S_w * c * C_M(alpha_w)

    alpha_e = theta + phi - cs.atan2(z_edot, x_edot)
    v_e = cs.sqrt(x_edot * x_edot + z_edot * z_edot + 1e-8)    # flow/air speed
    F_Le = C_L(alpha_e) * cs.vertcat(-z_edot, x_edot)          # lift force vector (proportional to)
    F_De = C_D(alpha_e) * cs.vertcat(-x_edot, -z_edot)         # drag force vector (proportional to)
    F_e = 0.5 * rho * v_e * S_e * (F_Le + F_De)
    M_e = 0.5 * rho * v_e**2 * S_e * c * C_M(alpha_e)

    # compute torques with respect to fixed reference point induced by forces

    # moment arms (vector from reference point of state to wing/elevator/fuselage)
    r_w = [ (- com_w + com_a) * cs.cos(theta), (- com_w + com_a) * cs.sin(theta) ]
    r_e = [ (- com_e + com_a) * cs.cos(theta), (- com_e + com_a) * cs.sin(theta)]
    r_p1 = [ ( -l + com_a) * cs.cos(theta), ( -l + com_a) * cs.sin(theta)]
    r_p2 = [ ( +l + com_a) * cs.cos(theta), ( +l + com_a) * cs.sin(theta)]

    τ_w = r_w[1] * F_w[0] - r_w[0] * F_w[1] + M_w
    τ_e = r_e[1] * F_e[0] - r_e[0] * F_e[1] + M_e
    τ_c1 = r_p1[1] * 0 - r_p1[0] * fn1
    τ_c2 = r_p2[1] * 0 - r_p2[0] * fn2
    thetaddot = -1. / I * (τ_w + τ_e + τ_c1 + τ_c2)

    # linear accelerations (F = ma)
    xddot = 1. / m * (F_w[0] + F_e[0])
    zddot = 1. / m * (F_w[1] + F_e[1] + fn1 + fn2) - g
    
    return cs.Function('f', [states, controls, forces],
                       [cs.vertcat(xdot, zdot, thetadot, phidot, xddot, zddot, thetaddot)])
    
def main():
    # plane parameters
    m = 0.065
    l_w_i = -0.005                                  # vector from CoM to centroid of wing (positive means wing is in front of CoM)
    l_w_f = -0.015                                  # vector from CoM to centroid of wing (positive means wing is in front of CoM)
    l = 0.26                                        # vector from CoM to start of elevator (attachment point to body)
    l_e = 0.02                                      # distance to centroid of elevator from start (attachment point to body)
    rho = 1.2041                                    # assume 20 degrees C
    S_w = 0.086                                     # surface area of wing
    S_e = 0.022                                     # surface area of elevator
    m_w = 0.6 * m * S_w / (S_w + S_e)               # mass of wing
    m_e = 0.6 * m * S_e / (S_w + S_e)               # mass of elevator
    m_f = 0.4 * m                                   # mass of fuselage
    l_w = 0.5*(l_w_i+l_w_f)
    l_f = -(l_w * m_w + (l - l_e) * m_e) / m_f      # vector to fuselage CoM
    gravity = 9.81
    I = m_w * l_w ** 2 + m_e * (l + l_e) ** 2 + m_f * l_f ** 2

    h = 0.001
    N = 1000

    params = [l_w_i, l_w_f, l, l_e, l_f, m_w, m_e, m_f, rho, S_w, S_e, m, gravity, I]

    f = build_dynamic_model(params)

    R = 5. 
    Sl_w = 100
    F_w = 1
    T_w = 100
    Landing_w = 100
    Distance_w = 0
    Quick_landing_w = 100
    Final_velocity_w = 10

    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0.
    g = []
    lbg = []
    ubg = []

    # "Lift" initial conditions
    Xk = construct_dyn_state(0)
    w += [Xk]
    lbw += [-1.5, 0.5 , 0. , 0., 6., -2. , 0.]
    ubw += [-1.5, 0.5 , 0. , 0., 6., -2. , 0.]
    w0 = [-1.5, 0.5 , 0. , 0., 6., -2. , 0.]
    wk = [-1.5, 0.5 , 0. , 0., 6., -2. , 0.]

    for k in range(N):
        Uk = cs.SX.sym("U_" + str(k))
        w   += [Uk]
        lbw += [-13.]
        ubw += [13.]
        if k < N / 3:
            uk = -2.
        elif k < 2 * N / 3:
            uk = 1.
        else:
            uk = 0.
        w0  += [uk]
        J += h * R * Uk ** 2
        J += h * T_w * Xk[2]**2
        J += h * Quick_landing_w * Xk[1]**2 

        Fk, Sl = construct_forces(k)
        w += [Fk]
        lbw += [0,0]
        ubw += [cs.inf,cs.inf]
        fk = [m*gravity/2, m*gravity/2] if wk[1] <= 0 else [0,0]
        w0 += fk

        J += F_w * (Fk[0] - m*gravity/2)**2
        J += F_w * (Fk[1] - m*gravity/2)**2

        w += [Sl]
        lbw += [0,0]
        ubw += [cs.inf,cs.inf]
        sl = [0,0]
        w0 += sl

        J += Sl_w * (Sl[0] + Sl[1])

        # implicit Euler step
        Xnext = construct_dyn_state(k+1)
        w += [Xnext]
        if k < N-1:
            lbw += [-cs.inf, -cs.inf, -cs.inf, -cs.pi / 3, -cs.inf, -cs.inf, -cs.inf]
            ubw += [ cs.inf,  cs.inf,  cs.inf,  cs.pi / 8,  cs.inf,  cs.inf,  cs.inf]
            wk = wk + h * f(wk, uk, fk).toarray()[:, 0]
            w0 += wk.tolist()

        # dynamics constraint: implicit Euler
        g   += [Xnext - Xk - h * f(Xnext, Uk, Fk)]
        lbg += [0]*7
        ubg += [0]*7

        # move forward
        Xk = Xnext

        # Linear Complementarity Problem
        phi_1 = Xk[1] - l*cs.sin(Xk[2])
        phi_2 = Xk[1] + l*cs.sin(Xk[2])
        g   += [phi_1, phi_2]
        lbg += [0, 0]
        ubg += [cs.inf, cs.inf]

        g   += [phi_1*Fk[0] - Sl[0], phi_2*Fk[1] - Sl[1]]
        lbg += [0, 0]
        ubg += [0, 0]


    # final state constraints
    lbw += [-cs.inf, 0, -cs.inf, -cs.inf, -cs.inf, -cs.inf, -cs.inf]
    ubw += [ cs.inf,  cs.inf,  cs.inf,  cs.inf,  cs.inf,  cs.inf,  cs.inf]
    wk = f(wk, uk, fk).toarray()[:, 0]
    w0 += wk.tolist()

    #J_error = Xk-np.array(x_N)
    #J += 40. * cs.dot(J_error * Q_N, J_error)
    J += Landing_w * Xk[1]**2 + Distance_w * (Xk[0] - w0[0])**2 + Final_velocity_w * cs.norm_1(Xk[4:6])

    # Create an NLP solver
    print("solving")
    prob = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
    solver = cs.nlpsol('solver', 'ipopt', prob)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full()

    print("Final objective:", sol['f'].full().item())
    live_plot_landing(w_opt, state_len=12, stop_frame=5, save=False)

main()