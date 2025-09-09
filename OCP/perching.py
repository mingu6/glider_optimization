import casadi as cs
import random
import numpy as np


# flat plate model lift and drag

def C_L(alpha):
    return 2 * cs.sin(alpha) * cs.cos(alpha)


def C_D(alpha):
    return 2 * cs.sin(alpha) * cs.sin(alpha)


def construct_dyn_state(t):
    x = cs.SX.sym("x_" + str(t))
    z = cs.SX.sym("z_" + str(t))
    theta = cs.SX.sym("theta_" + str(t))
    phi = cs.SX.sym("phi_" + str(t))
    xdot = cs.SX.sym("xdot_" + str(t))
    zdot = cs.SX.sym("zdot_" + str(t))
    thetadot = cs.SX.sym("thetadot_" + str(t))
    return cs.vertcat(x, z, theta, phi, xdot, zdot, thetadot)


def build_f(params):
    x = cs.SX.sym("x")
    z = cs.SX.sym("z")
    theta = cs.SX.sym("theta")
    phi = cs.SX.sym("phi")
    xdot = cs.SX.sym("xdot")
    zdot = cs.SX.sym("zdot")
    thetadot = cs.SX.sym("thetadot")
    phidot = cs.SX.sym("phidot")

    states = cs.vertcat(x, z, theta, phi, xdot, zdot, thetadot)
    controls = phidot

    l_w, l, l_e, rho, S_w, S_e, m, g, I = params

    # geometric centroid of aerodynamic surfaces (mean chord for flat plate)
    x_wdot = xdot + l_w * thetadot * cs.sin(theta)
    z_wdot = zdot - l_w * thetadot * cs.cos(theta)
    x_edot = xdot + l * thetadot * cs.sin(theta) + l_e * (thetadot + phidot) * cs.sin(theta + phi)
    z_edot = zdot - l * thetadot * cs.cos(theta) - l_e * (thetadot + phidot) * cs.cos(theta + phi)
    
    # force vectors for aerodynamic surfaces (lift, drag, gravity)

    alpha_w = theta - cs.atan2(z_wdot, x_wdot)
    v_w = cs.sqrt(x_wdot * x_wdot + z_wdot * z_wdot + 1e-8)             # flow/air speed
    F_Lw = C_L(alpha_w) * cs.vertcat(-z_wdot, x_wdot)                   # lift force vector (proportional to)
    F_Dw = C_D(alpha_w) * cs.vertcat(-x_wdot, -z_wdot)                  # drag force vector (proportional to)
    F_w = 0.5 * rho * v_w * S_w * (F_Lw + F_Dw)

    alpha_e = theta + phi - cs.atan2(z_edot, x_edot)
    v_e = cs.sqrt(x_edot * x_edot + z_edot * z_edot + 1e-8)    # flow/air speed
    F_Le = C_L(alpha_e) * cs.vertcat(-z_edot, x_edot)          # lift force vector (proportional to)
    F_De = C_D(alpha_e) * cs.vertcat(-x_edot, -z_edot)         # drag force vector (proportional to)
    F_e = 0.5 * rho * v_e * S_e * (F_Le + F_De)

    # compute torques with respect to fixed reference point induced by forces

    # moment arms (vector from reference point of state to wing/elevator/fuselage)

    r_w = [-l_w * cs.cos(theta), -l_w * cs.sin(theta)]
    r_e = [-l * cs.cos(theta) - l_e * cs.cos(theta + phi), -l * cs.sin(theta) - l_e * cs.sin(theta + phi)]

    τ_w = r_w[1] * F_w[0] - r_w[0] * F_w[1]
    τ_e = r_e[1] * F_e[0] - r_e[0] * F_e[1]
    thetaddot = -1. / I * (τ_w + τ_e)

    # linear accelerations (F = ma)
    xddot = 1. / m * (F_w[0] + F_e[0])
    zddot = 1. / m * (F_w[1] + F_e[1]) - g
    
    return cs.Function('f', [states, controls],
                       [cs.vertcat(xdot, zdot, thetadot, phidot, xddot, zddot, thetaddot)])


def main():
    # plane parameters
    m = 0.065
    l_w = 0.01                                      # vector from CoM to centroid of wing (positive means wing is in front of CoM)
    l = -0.26                                       # vector from CoM to start of elevator (attachment point to body)
    l_e = 0.02                                      # distance to centroid of elevator from start (attachment point to body)
    rho = 1.2041                                    # assume 20 degrees C
    S_w = 0.086                                     # surface area of wing
    S_e = 0.022                                     # surface area of elevator
    m_w = 0.6 * m * S_w / (S_w + S_e)               # mass of wing
    m_e = 0.6 * m * S_e / (S_w + S_e)               # mass of elevator
    m_f = 0.4 * m                                   # mass of fuselage
    l_f = -(l_w * m_w + (l - l_e) * m_e) / m_f      # vector to fuselage CoM
    g = 9.81
    I = m_w * l_w ** 2 + m_e * (l + l_e) ** 2 + m_f * l_f ** 2

    h = 0.01
    N = 101

    params = [l_w, l, l_e, rho, S_w, S_e, m, g, I]

    f = build_f(params)

    # objective function weights
    x_N = [0., 0., cs.pi / 4, 0., 0., 0., 0.]
    Q_N = [2., 50., 0., 0., 4., 4., 0.]
    R = 100.

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
    lbw += [-3.5, 0.1 , 0. , 0., 7., 0. , 0.]
    ubw += [-3.5, 0.1 , 0. , 0., 7., 0. , 0.]
    w0 = [-3.5, 0.1 , 0. , 0., 7., 0. , 0.]

    # initialisation for X
    X0 = np.linspace(w0, np.zeros_like(w0), N+1)
    wk = [-3.5, 0.1 , 0. , 0., 7., 0. , 0.]

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

        Xnext = Xk + h * f(Xk, Uk)
        Xk = construct_dyn_state(k+1)
        w   += [Xk]
        if k < N-1:
            lbw += [-cs.inf, -cs.inf, -cs.inf, -cs.pi / 3, -cs.inf, -cs.inf, -cs.inf]
            ubw += [ cs.inf,  cs.inf,  cs.inf,  cs.pi / 8,  cs.inf,  cs.inf,  cs.inf]
            wk = wk + h * f(wk, uk).toarray()[:, 0]
            w0 += wk.tolist()

        # Add equality constraint (dynamics)
        g   += [Xnext - Xk]
        lbg += [0., 0., 0., 0., 0., 0., 0.,]
        ubg += [0., 0., 0., 0., 0., 0., 0.,]

    # final state constraints
    lbw += [-cs.inf, -cs.inf, -cs.inf, -cs.inf, -cs.inf, -cs.inf, -cs.inf]
    ubw += [ cs.inf,  cs.inf,  cs.inf,  cs.inf,  cs.inf,  cs.inf,  cs.inf]
    wk = f(wk, uk).toarray()[:, 0]
    w0 += wk.tolist()

    J += 40. * cs.dot(Xk * Q_N, Xk)

    # Create an NLP solver
    print("solving")
    prob = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
    solver = cs.nlpsol('solver', 'ipopt', prob)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full()

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 6, figsize=(20, 4))
    axs[0].plot(w_opt[0::8], w_opt[1::8])
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("z")
    axs[0].set_title("Positions")
    axs[0].set_xlim(-4., 1.)
    axs[0].set_ylim(-1., 1.)

    axs[1].plot(w_opt[4::8], w_opt[5::8])
    axs[1].set_xlabel("xdot")
    axs[1].set_ylabel("zdot")
    axs[1].set_title("Velocities")

    axs[2].plot(w_opt[2::8])
    axs[2].set_title("theta")
    axs[3].plot(w_opt[3::8])
    axs[3].set_title("phi")

    axs[4].plot(w_opt[0::8])
    axs[4].set_ylabel("x")
    axs[4].set_ylim(-4., 3.)

    axs[5].plot(w_opt[1::8])
    axs[5].set_ylabel("z")
    axs[5].set_ylim(-1., 1.)
    plt.show()

main()