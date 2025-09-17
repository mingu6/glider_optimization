import casadi as cs
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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


def plot(w_opt, target, penalty_weights):
    L = 1.0
    L_lift = 0.3    
    f = 0.6
    eps = 0.4
    extra = 60

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.5)

    ax_sim = fig.add_subplot(gs[0:2, 0])
    ax_sim.set_xlim(-3.5, 2)
    ax_sim.set_ylim(-3, 2.5)
    ax_sim.set_aspect('equal', adjustable='box')
    ax_sim.set_title("Glider perching simulation")

    ax_vel = fig.add_subplot(gs[2, 0])
    ax_vel.set_xlim(0, len(w_opt)//8 - 1)
    dot_concat = np.concatenate([w_opt[4::8], w_opt[5::8]])
    ax_vel.set_ylim(np.min(dot_concat)-eps, np.max(dot_concat)+eps)
    xdot_line, = ax_vel.plot([], [], lw=2, color="r", label="xdot")
    zdot_line, = ax_vel.plot([], [], lw=2, color="g", label="zdot")
    ax_vel.legend(loc="upper right")
    ax_vel.set_title("Linear velocities")

    ax_ang = fig.add_subplot(gs[2, 1])
    ax_ang.set_xlim(0, len(w_opt)//8 - 1)
    ang_concat = np.concatenate([w_opt[6::8], w_opt[7::8]])
    ax_ang.set_ylim(np.min(ang_concat)-eps, np.max(ang_concat)+eps)
    thetadot_line, = ax_ang.plot([], [], lw=2, color="b", label="thetadot")
    phidot_line, = ax_ang.plot([], [], lw=2, color="m", label="phidot")
    ax_ang.legend(loc="upper right")
    ax_ang.set_title("Angular velocities")

    x0_target = target[0] - f*L*np.cos(target[2])
    z0_target = target[1] - f*L*np.sin(target[2])
    x1_target = x0_target + L*np.cos(target[2])
    z1_target = z0_target + L*np.sin(target[2])

    glider, = ax_sim.plot([], [], 'o-', lw=3)
    target_pose, = ax_sim.plot([x0_target, x1_target], [z0_target, z1_target], 'o-', lw=3, color="red", alpha=0.2)
    trail, = ax_sim.plot([], [], '-', lw=3, color='k', alpha=0.3)

    ax_attack = fig.add_subplot(gs[0, 1])
    ax_attack.set_xlim(0, len(w_opt)//8 - 1)
    attack_angles = w_opt[2::8] - np.atan2(w_opt[5::8], w_opt[4::8])
    ax_attack.set_ylim(np.min(attack_angles)-eps, np.max(attack_angles)+eps)
    attack_line, = ax_attack.plot([], [], lw=2, color="orange")
    ax_attack.set_title("Angle of attack")

    ax_error = fig.add_subplot(gs[1, 1])
    ax_error.set_xlim(0, len(w_opt)//8 - 1)

    Js = w_opt[:7] - target
    weighted_errors = np.dot(Js*penalty_weights, Js)
    ax_error.set_ylim(0, np.max(weighted_errors))
    error_line, = ax_error.plot([], [], lw=2, color="purple")
    ax_error.set_title("Target weighted squared error")

    def init():
        global trail_x, trail_z, xdot_data, zdot_data, thetadot_data, phidot_data, attack_angle_data, error_data

        glider.set_data([], [])
        trail.set_data([], [])
        xdot_line.set_data([], [])
        zdot_line.set_data([], [])
        thetadot_line.set_data([], [])
        phidot_line.set_data([], [])
        attack_line.set_data([], [])
        error_line.set_data([], [])

        trail_x = []
        trail_z = []
        xdot_data = []
        zdot_data = []
        thetadot_data = []
        phidot_data = []
        attack_angle_data = []
        error_data = []
        return glider, trail, xdot_line, zdot_line, thetadot_line, phidot_line, attack_line, error_line

    def update(i):
        global trail_x, trail_z, xdot_data, zdot_data, thetadot_data, phidot_data, attack_angle_data, error_data

        if(len(w_opt) < (i+1)*8):
            return
        else:
            x, z, theta, phi, xdot, zdot, thetadot, phi_dot = w_opt[i*8 : (i+1)*8]

        x0 = x - f*L*np.cos(theta)
        z0 = z - f*L*np.sin(theta)
        x1 = x0 + L*np.cos(theta)
        z1 = z0 + L*np.sin(theta)
        xl = x - f*L*np.cos(theta) - L_lift*np.cos(theta+phi)
        zl = z - f*L*np.sin(theta) - L_lift*np.sin(theta+phi)
        trail_x.append(x)
        trail_z.append(z)
        glider.set_data([xl, x0, x1], [zl, z0, z1])
        trail.set_data(trail_x, trail_z)

        xdot_data.append(xdot)
        zdot_data.append(zdot)
        x_axis = range(len(xdot_data))
        xdot_line.set_data(x_axis, xdot_data)
        zdot_line.set_data(x_axis, zdot_data)

        thetadot_data.append(thetadot)
        phidot_data.append(phi_dot)
        thetadot_line.set_data(x_axis, thetadot_data)
        phidot_line.set_data(x_axis, phidot_data)

        attack_angle_data.append(theta - np.atan2(zdot, xdot))
        attack_line.set_data(x_axis, attack_angle_data)

        J_error = w_opt[i*8 : i*8+7].flatten() - target
        weighted_squared_error = np.dot(J_error*penalty_weights, J_error)
        error_data.append(weighted_squared_error)
        error_line.set_data(x_axis, error_data)

        return glider, trail, xdot_line, zdot_line, thetadot_line, phidot_line, attack_line, error_line

    ani = animation.FuncAnimation(fig, update, frames=len(w_opt)//8 + extra,
                                init_func=init, blit=False, interval=1000 / 30, repeat=True)
    fig.canvas.manager.set_window_title("Perching OCP")
    plt.show()

def main():
    # plane parameters
    m = 0.065
    l_w = 0.01                                      # vector from CoM to centroid of wing (positive means wing is in front of CoM)
    l = 0.26                                       # vector from CoM to start of elevator (attachment point to body)
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
    N = 111 

    params = [l_w, l, l_e, rho, S_w, S_e, m, g, I]

    f = build_f(params)

    # objective function weights
    x_N = [0.,  0,  0. , 0.,       0.,    0.,    0.]
    #      x    z    theta       phidot    xdot   ydot   thetadot
    Q_N = [10.,  10., 10.,        0.,       1.,    1,    1] 
    R = 5. 

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
        lbg += [0]*7
        ubg += [0]*7

    # final state constraints
    lbw += [-cs.inf, -cs.inf, -cs.inf, -cs.inf, -cs.inf, -cs.inf, -cs.inf]
    ubw += [ cs.inf,  cs.inf,  cs.inf,  cs.inf,  cs.inf,  cs.inf,  cs.inf]
    wk = f(wk, uk).toarray()[:, 0]
    w0 += wk.tolist()

    J_error = Xk-np.array(x_N)
    J += 40. * cs.dot(J_error * Q_N, J_error)

    # Create an NLP solver
    print("solving")
    prob = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
    solver = cs.nlpsol('solver', 'ipopt', prob)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full()

    print("Final objective:", sol['f'].full().item())

    plot(w_opt, x_N, Q_N)

main()