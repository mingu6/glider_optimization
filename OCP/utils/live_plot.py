import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def live_plot(w_opt, target, penalty_weights):
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
    ax_vel.axhline(0, color='black', linestyle='--')
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

    Js = np.array([w_opt[i:i+7].flatten() - target for i in range(0,len(w_opt),8)])
    weighted_errors = np.sum(Js*penalty_weights*Js, axis=1)
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