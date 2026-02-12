from casadi import (
    SX, Function,
    sin, cos, tanh, atan2, sqrt,
    dot, gradient,
    vertcat, vcat, diag,
    pi, fmax, fmin
)
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from ..config import Config
class GliderPerching :
    def __init__(self, config: Config, project_name='glider-perching'):
        self.project_name = project_name
        self.config = config

    def C_L(self, alpha):
        return 2 * sin(alpha) * cos(alpha)

    def C_D(self, alpha):
        return 2 * sin(alpha) * sin(alpha)

    def C_M(self, alpha):
        return -self.C_L(alpha) * 0.25

    def mc_to_wcom(self, l_w):      # distance from leading edge to wing center of mass, constant 
        return l_w+0.003
    
    def scale(self, x, min, max): # scale x in [min, max] to [-1, 1] - Only for the Chebyshev basis
        return 2*(x - min)/(max - min) - 1
    
    def cheb_basis_2d(self, alpha_s, Re_s, deg):
        T_a = [1, alpha_s]
        T_r = [1, Re_s]

        for k in range(2, deg+1):
            T_a.append(2*alpha_s*T_a[k-1] - T_a[k-2])
            T_r.append(2*Re_s*T_r[k-1] - T_r[k-2])

        B = []
        for i in range(deg+1):
            for j in range(deg+1):
                B.append(T_a[i]*T_r[j])

        return vertcat(*B)

    # Smooth, differentiable gate that is ~1 for xmin < x < xmax and ~0 outside, with transition sharpness controlled by k
    def smooth_gate(self, x, xmin, xmax, k):
        return 0.5*(tanh(k*(x - xmin)) - tanh(k*(x - xmax)))

    def initDyn(self):
        # set the global parameters
        m = 0.065
        l_w_i = -0.005                                # vector to leading edge from total body center of mass
        l_w_f = -0.015                                # vector to trailing edge from total body center of mass
        l = 0.26                                      # vector from CoM to start of elevator (attachment point to body / hinge point)
        l_e = 0.02                                    # distance from the hinge to the mean aerodynamic chord of the elevator
        rho = 1.225                                   # assume Standard sea-level air density
        m_f = 0.4 * m                                 # mass of fuselage
        l_w = 0.5*(l_w_i+l_w_f)                       # mean aerodynamic chord location
        g = 9.81
        S_w = 0.158
        S_e = 0.017
        mu_air = 1.789e-5                               # assume Standard sea-level air dynamic viscosity      
        
        chebyshev_deg = self.config.reducedModel.chebyshev_degree

        phi_CL = SX.sym("phi_CL", (chebyshev_deg+1)**2, 1)
        phi_CD = SX.sym("phi_CD", (chebyshev_deg+1)**2, 1)
        phi_CM = SX.sym("phi_CM", (chebyshev_deg+1)**2, 1)
                
        #parameter = [phi_CL, phi_CD, phi_CM]

        nfConfig = self.config.neuralFoilSampling

        # In 3D mode we optionally add a separate (fixed) elevator surrogate.
        if getattr(nfConfig, "use_3d_llt", False):
            phi_CL_e = SX.sym("phi_CL_e", (chebyshev_deg+1)**2, 1)
            phi_CD_e = SX.sym("phi_CD_e", (chebyshev_deg+1)**2, 1)
            phi_CM_e = SX.sym("phi_CM_e", (chebyshev_deg+1)**2, 1)
            parameter = [phi_CL, phi_CD, phi_CM, phi_CL_e, phi_CD_e, phi_CM_e]
        else:
            phi_CL_e = phi_CD_e = phi_CM_e = None
            parameter = [phi_CL, phi_CD, phi_CM]

        self.dyn_auxvar = vcat(parameter)

        m_w = 0.6 * m * S_w / (S_w + S_e)
        m_e = 0.6 * m * S_e / (S_w + S_e)
        l_f = -(l_w * m_w + (l - l_e) * m_e) / m_f      # vector to fuselage CoM
        I = m_w * l_w ** 2 + m_e * (l + l_e) ** 2 + m_f * l_f ** 2
        chord = np.abs(l_w_f - l_w_i) # mean aerodynamic chord length
        
        # Declare system variables
        x = SX.sym("x")
        z = SX.sym("z")
        theta = SX.sym("theta")         # pitch angle
        phi = SX.sym("phi")             # elevator angle ( colinear with the wing at 0 rad )
        xdot = SX.sym("xdot")           # velocity in x
        zdot = SX.sym("zdot")           # velocity in z
        thetadot = SX.sym("thetadot")
        t = SX.sym("t")
        
        phidot = SX.sym("phidot") # elevator angular velocity

        self.X = vertcat(x, z, theta, phi, xdot, zdot, thetadot, t)
        self.U = phidot

        # wing mean chord 
        l_w_m = (l_w_i + l_w_f) / 2

        com_w = l_w_m + self.mc_to_wcom(l_w_m)
        com_e = l + l_e # simplifying assumption, the elevator's com doesn't depend on the angle (quasi static assumption) - aligned with the fuselage               
        
        # Defaults: preserve old 2D behavior
        com_w_x = float(com_w)     # existing scalar
        com_w_z = 0.0
        com_e_x = float(com_e)
        com_e_z = 0.0
        # 3D override: use chordwise centroids saved in aero_rom 3d_blocks.pt
        if getattr(nfConfig, "use_3d_llt", False):
            try:
                import torch
                ckpt = torch.load(getattr(nfConfig, "llt_ckpt_path", ""), map_location="cpu")
                cent = ckpt.get("centroid", {}) if isinstance(ckpt, dict) else {}

                # Wing centroid
                if "wing_x" in cent:
                    com_w_x = float(cent["wing_x"])
                if "wing_z" in cent:
                    com_w_z = float(cent["wing_z"])

                # Elevator centroid
                if "elevator_x" in cent:
                    com_e_x = float(cent["elevator_x"])
                if "elevator_z" in cent:
                    com_e_z = float(cent["elevator_z"])

            except Exception:
                pass
        else:
            # 2D mode
            com_e_z = com_w_z
        
        com_f = l_f
        com_a = (com_w*m_w + com_e*m_e + com_f*m_f) / (m_w + m_e + m_f)
        if getattr(nfConfig, "use_3d_llt", False):
            com_a_x = (com_w_x*m_w + com_e_x*m_e + com_f*m_f) / (m_w + m_e + m_f)
        else:
            com_a_x = com_a   # keep old behavior

        # geometric centroid of aerodynamic surfaces (mean chord for flat plate)
        if getattr(nfConfig, "use_3d_llt", False):
            # body-frame lever arms from reference point to aero centroid
            r_w_bx = com_w_x
            r_w_bz = com_w_z
            r_e_bx = com_e_x
            r_e_bz = com_e_z

            # rotate to world
            r_w_x = r_w_bx * cos(theta) - r_w_bz * sin(theta)
            r_w_z = r_w_bx * sin(theta) + r_w_bz * cos(theta)
            r_e_x = r_e_bx * cos(theta) - r_e_bz * sin(theta)
            r_e_z = r_e_bx * sin(theta) + r_e_bz * cos(theta)

            # rigid-body point velocities at those centroids
            x_wdot = xdot + thetadot * r_w_z
            z_wdot = zdot - thetadot * r_w_x

            x_edot = xdot + thetadot * r_e_z + l_e * (thetadot + phidot) * sin(theta + phi)
            z_edot = zdot - thetadot * r_e_x - l_e * (thetadot + phidot) * cos(theta + phi)
        else:
            # --- 2D mode: unchanged original code ---
            x_wdot = xdot + l_w_m * thetadot * sin(theta)
            z_wdot = zdot - l_w_m * thetadot * cos(theta)
            x_edot = xdot + l * thetadot * sin(theta) + l_e * (thetadot + phidot) * sin(theta + phi)
            z_edot = zdot - l * thetadot * cos(theta) - l_e * (thetadot + phidot) * cos(theta + phi)
        v_w = sqrt(x_wdot * x_wdot + z_wdot * z_wdot + 1e-8) # flow/air speed
        
        alpha_w = theta - atan2(z_wdot, x_wdot)
        Re = rho * v_w * chord / mu_air
        
        #nfConfig = self.config.neuralFoilSampling
        
        a0_min = nfConfig.AoA_min*pi/180
        a0_max = nfConfig.AoA_max*pi/180
        sharpness = 20
        w = self.smooth_gate(alpha_w, a0_min, a0_max, sharpness)
        alpha_scaled = self.scale(alpha_w, a0_min, a0_max)
        Re_scaled = self.scale(Re, nfConfig.Re_min, nfConfig.Re_max)
        
        # Clamp scaled inputs to [-1, 1] to prevent Chebyshev basis explosion outside the envelope
        alpha_scaled_clamped = fmax(-1.0, fmin(1.0, alpha_scaled))
        Re_scaled_clamped = fmax(-1.0, fmin(1.0, Re_scaled))
        
        X = self.cheb_basis_2d(alpha_scaled_clamped, Re_scaled_clamped, chebyshev_deg)

        CL_w = w*dot(X, phi_CL) + (1-w)*self.C_L(alpha_w)
        CD_w = w*dot(X, phi_CD) + (1-w)*self.C_D(alpha_w)
        CM_w = w*dot(X, phi_CM) + (1-w)*self.C_M(alpha_w)
        
        # force vectors for aerodynamic surfaces (lift, drag, gravity)
        F_Lw = CL_w * vertcat(-z_wdot, x_wdot)  # lift force vector (proportional to)
        F_Dw = CD_w * vertcat(-x_wdot, -z_wdot) # drag force vector (proportional to)
        F_w = 0.5 * rho * v_w * S_w * (F_Lw + F_Dw)
        M_w = 0.5 * rho * v_w**2 * S_w * chord * CM_w

        alpha_e = theta + phi - atan2(z_edot, x_edot)
        v_e = sqrt(x_edot * x_edot + z_edot * z_edot + 1e-8)   # flow/air speed

        Re_e = rho * v_e * chord / mu_air

        # Elevator coefficients: default analytic model; in 3D mode, use the
        # fixed-elevator Chebyshev surrogate inside the same envelope.
        if phi_CL_e is not None:
            w_e = self.smooth_gate(alpha_e, a0_min, a0_max, sharpness)
            alpha_e_scaled = self.scale(alpha_e, a0_min, a0_max)
            Re_e_scaled = self.scale(Re_e, nfConfig.Re_min, nfConfig.Re_max)
            alpha_e_scaled_clamped = fmax(-1.0, fmin(1.0, alpha_e_scaled))
            Re_e_scaled_clamped = fmax(-1.0, fmin(1.0, Re_e_scaled))
            X_e = self.cheb_basis_2d(alpha_e_scaled_clamped, Re_e_scaled_clamped, chebyshev_deg)

            CL_e = w_e * dot(X_e, phi_CL_e) + (1 - w_e) * self.C_L(alpha_e)
            CD_e = w_e * dot(X_e, phi_CD_e) + (1 - w_e) * self.C_D(alpha_e)
            CM_e = w_e * dot(X_e, phi_CM_e) + (1 - w_e) * self.C_M(alpha_e)
        else:
            CL_e = self.C_L(alpha_e)
            CD_e = self.C_D(alpha_e)
            CM_e = self.C_M(alpha_e)

        #F_Le = self.C_L(alpha_e) * vertcat(-z_edot, x_edot)    # lift force vector (proportional to)
        F_Le = CL_e * vertcat(-z_edot, x_edot)
        #F_De = self.C_D(alpha_e) * vertcat(-x_edot, -z_edot)   # drag force vector (proportional to)
        F_De = CD_e * vertcat(-x_edot, -z_edot)
        F_e = 0.5 * rho * v_e * S_e * (F_Le + F_De)
        #M_e = 0.5 * rho * v_e**2 * S_e * chord * self.C_M(alpha_e)
        M_e = 0.5 * rho * v_e**2 * S_e * chord * CM_e

        # compute torques with respect to fixed reference point induced by forces
        # moment arms (vector from reference point of state to wing/elevator/fuselage)
        if getattr(nfConfig, "use_3d_llt", False):
            # --- 3D mode: lever arms from reference point (com_a_x) to aero centroids ---
            r_w_bx = -com_w_x + com_a_x
            r_w_bz = -com_w_z              # com_a_z assumed 0 in planar model
            r_e_bx = -com_e_x + com_a_x
            r_e_bz = -com_e_z

            # rotate lever arms to world
            r_w_x = r_w_bx * cos(theta) - r_w_bz * sin(theta)
            r_w_z = r_w_bx * sin(theta) + r_w_bz * cos(theta)
            r_e_x = r_e_bx * cos(theta) - r_e_bz * sin(theta)
            r_e_z = r_e_bx * sin(theta) + r_e_bz * cos(theta)

            # torque about y: tau = r_x*F_z - r_z*F_x
            τ_w = r_w_x * F_w[1] - r_w_z * F_w[0] + M_w
            τ_e = r_e_x * F_e[1] - r_e_z * F_e[0] + M_e
            thetaddot = -1. / I * (τ_w + τ_e)

        else:
            # --- 2D mode: unchanged original code ---
            r_w = [ (- com_w + com_a) * cos(theta), (- com_w + com_a) * sin(theta) ]
            r_e = [ (- com_e + com_a) * cos(theta), (- com_e + com_a) * sin(theta)]

            τ_w = r_w[1] * F_w[0] - r_w[0] * F_w[1] + M_w
            τ_e = r_e[1] * F_e[0] - r_e[0] * F_e[1] + M_e
            thetaddot = -1. / I * (τ_w + τ_e)

        # linear accelerations (F = ma)
        xddot = 1. / m * (F_w[0] + F_e[0])
        zddot = 1. / m * (F_w[1] + F_e[1]) - g
        
        self.f = vertcat(xdot, zdot, thetadot, phidot, xddot, zddot, thetaddot, 0)

    def initCost(self, state_weights, wu=0.001, stage_scale = 0.0001):
        self.goal = [0., 0., 0., 0., 0., 0., 0., 0.]
        self.state_weights = state_weights
        self.cost_auxvar = vcat([])

        err = self.X - self.goal
        self.path_cost = wu * (self.U * self.U) + err.T @ diag(state_weights)*stage_scale @ err 
        self.dpath_cost_dx = gradient(self.path_cost, self.X)
        self.dpath_cost_du = gradient(self.path_cost, self.U)
        
        self.final_cost = err.T @ diag(state_weights) @ err
        
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

    def play_animation(self, state_traj, control_traj, 
                    save_option=False, title='glider-perching', fps=30):
        """
        Create stunning glider perching animation with all metrics.
        
        Args:
            state_traj: State trajectory (N x 8) - [x, z, theta, phi, xdot, zdot, thetadot]
            control_traj: Control trajectory (N x 1) - [phidot]
            goal: Goal state (8,)
            state_weights: State weights for error computation (7,)
            save_option: Whether to save animation as GIF
            title: Filename for saved animation
            fps: Frames per second
        """
        # ==================== PRE-COMPUTE ALL METRICS ====================
        n_frames = len(state_traj)
        
        # Attack angles: theta - arctan2(zdot, xdot)
        attack_angles = state_traj[:, 2] - np.arctan2(state_traj[:, 5], state_traj[:, 4])
        
        # Weighted tracking errors
        errors = state_traj - self.goal
        weighted_errors = np.sum(errors * self.state_weights * errors, axis=1)
        
        # Data ranges for plot limits
        eps = 0.4
        vel_range = (np.concatenate([state_traj[:, 4], state_traj[:, 5]]).min() - eps,
                    np.concatenate([state_traj[:, 4], state_traj[:, 5]]).max() + eps)
        ang_range = (np.concatenate([state_traj[:, 6], control_traj.squeeze()]).min() - eps,
                    np.concatenate([state_traj[:, 6], control_traj.squeeze()]).max() + eps)
        attack_range = (attack_angles.min() - eps, attack_angles.max() + eps)
        error_range = (0, weighted_errors.max() * 1.1)
        
        # ==================== GEOMETRY CONSTANTS ====================
        L = 1.0              # Glider length
        L_lift = 0.3         # Lift surface length
        f = 0.6              # Center offset fraction
        
        # Target pose geometry
        x_target, z_target, theta_target = self.goal[0], self.goal[1], self.goal[2]
        x0_target = x_target - f * L * np.cos(theta_target)
        z0_target = z_target - f * L * np.sin(theta_target)
        x1_target = x0_target + L * np.cos(theta_target)
        z1_target = z0_target + L * np.sin(theta_target)
        
        # ==================== FIGURE SETUP ====================
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(14, 10), facecolor='#F5F5F5')
        fig.suptitle('Glider Perching Trajectory Optimization', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3,
                            left=0.08, right=0.95, top=0.93, bottom=0.06)
        
        # ==================== SIMULATION PANEL (Main) ====================
        ax_sim = fig.add_subplot(gs[0:2, 0])
        
        # Dynamic limits based on trajectory
        x_all = np.concatenate([state_traj[:, 0], [x_target]])
        z_all = np.concatenate([state_traj[:, 1], [z_target]])
        
        x_min, x_max = x_all.min(), x_all.max()
        z_min, z_max = z_all.min(), z_all.max()
        
        # Add padding (at least 1m or 10% of range)
        pad_x = max(1.0, (x_max - x_min) * 0.1)
        pad_z = max(1.0, (z_max - z_min) * 0.1)
        
        ax_sim.set_xlim(x_min - pad_x, x_max + pad_x)
        ax_sim.set_ylim(z_min - pad_z, z_max + pad_z)
        
        ax_sim.set_aspect('equal', adjustable='box')
        ax_sim.set_title("Glider Perching Simulation", fontsize=12, fontweight='bold', pad=10)
        ax_sim.set_xlabel("X Position (m)", fontsize=10)
        ax_sim.set_ylabel("Z Position (m)", fontsize=10)
        ax_sim.grid(True, alpha=0.2, linestyle=':')
        
        # Start position marker
        ax_sim.plot(state_traj[0, 0], state_traj[0, 1], 'x', color='black', markersize=8, markeredgewidth=2, label='Start', zorder=4)

        # Target visualization with glow
        target_circle = Circle((x_target, z_target), 0.1, fill=False, 
                            edgecolor='#FF5252', linestyle='-', linewidth=3, zorder=4, label='Target')
        ax_sim.add_patch(target_circle)
        
        # Glider artists
        glider_body, = ax_sim.plot([], [], 'o-', lw=4, color='#2E86AB', 
                                markersize=8, markerfacecolor='#A23B72',
                                markeredgewidth=2, markeredgecolor='white',
                                label='Glider', zorder=5)
        com_marker, = ax_sim.plot([], [], 'o', markersize=6, color='red',
                                markerfacecolor='yellow', markeredgewidth=1.5,
                                markeredgecolor='red', alpha=0.7, zorder=6)
        trail_collection = LineCollection([], linewidths=2, alpha=0.6, cmap='viridis')
        ax_sim.add_collection(trail_collection)
        ax_sim.legend(loc='upper right', fontsize=9)
        
        # ==================== LINEAR VELOCITIES ====================
        ax_vel = fig.add_subplot(gs[2, 0])
        ax_vel.set_xlim(0, n_frames - 1)
        ax_vel.set_ylim(vel_range)
        ax_vel.set_title("Linear Velocities", fontsize=11, fontweight='bold', pad=10)
        ax_vel.set_ylabel("Velocity (m/s)", fontsize=9)
        ax_vel.grid(True, alpha=0.3, linestyle='--')
        ax_vel.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        xdot_line, = ax_vel.plot([], [], lw=2.5, color="#E63946", label="$\\dot{x}$", alpha=0.9)
        zdot_line, = ax_vel.plot([], [], lw=2.5, color="#06A77D", label="$\\dot{z}$", alpha=0.9)
        ax_vel.legend(loc="upper right", fontsize=9)
        
        # ==================== ANGLE OF ATTACK ====================
        ax_attack = fig.add_subplot(gs[0, 1])
        ax_attack.set_xlim(0, n_frames - 1)
        ax_attack.set_ylim(attack_range)
        ax_attack.set_title("Angle of Attack", fontsize=11, fontweight='bold', pad=10)
        ax_attack.set_ylabel("Angle (rad)", fontsize=9)
        ax_attack.grid(True, alpha=0.3, linestyle='--')
        ax_attack.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        attack_line, = ax_attack.plot([], [], lw=2.5, color="#F77F00", alpha=0.9)
        
        # ==================== TRACKING ERROR ====================
        ax_error = fig.add_subplot(gs[1, 1])
        ax_error.set_xlim(0, n_frames - 1)
        ax_error.set_ylim(error_range)
        ax_error.set_title("Tracking Error", fontsize=11, fontweight='bold', pad=10)
        ax_error.set_ylabel("Weighted Error", fontsize=9)
        ax_error.grid(True, alpha=0.3, linestyle='--')
        error_line, = ax_error.plot([], [], lw=2.5, color="#9D4EDD", alpha=0.9)
        
        # ==================== ANGULAR VELOCITIES ====================
        ax_ang = fig.add_subplot(gs[2, 1])
        ax_ang.set_xlim(0, n_frames - 1)
        ax_ang.set_ylim(ang_range)
        ax_ang.set_title("Angular Velocities", fontsize=11, fontweight='bold', pad=10)
        ax_ang.set_ylabel("Velocity (rad/s)", fontsize=9)
        ax_ang.grid(True, alpha=0.3, linestyle='--')
        ax_ang.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        thetadot_line, = ax_ang.plot([], [], lw=2.5, color="#118AB2", label="$\\dot{\\theta}$", alpha=0.9)
        phidot_line, = ax_ang.plot([], [], lw=2.5, color="#D81159", label="$\\dot{\\phi}$", alpha=0.9)
        ax_ang.legend(loc="upper right", fontsize=9)
        
        # ==================== ANIMATION DATA BUFFERS ====================
        trail_points = []
        xdot_data, zdot_data = [], []
        thetadot_data, phidot_data = [], []
        attack_data, error_data = [], []
        
        # ==================== ANIMATION FUNCTIONS ====================
        def init():
            """Initialize all artists."""
            glider_body.set_data([], [])
            com_marker.set_data([], [])
            trail_collection.set_segments([])
            xdot_line.set_data([], [])
            zdot_line.set_data([], [])
            thetadot_line.set_data([], [])
            phidot_line.set_data([], [])
            attack_line.set_data([], [])
            error_line.set_data([], [])
            
            trail_points.clear()
            xdot_data.clear()
            zdot_data.clear()
            thetadot_data.clear()
            phidot_data.clear()
            attack_data.clear()
            error_data.clear()
            
            return (glider_body, com_marker, trail_collection, xdot_line, zdot_line,
                    thetadot_line, phidot_line, attack_line, error_line)
        
        def update(frame):
            """Update all artists for current frame."""
            if frame >= n_frames:
                return (glider_body, com_marker, trail_collection, xdot_line, zdot_line,
                    thetadot_line, phidot_line, attack_line, error_line)
            
            # Extract current state
            x, z, theta, phi, xdot, zdot, thetadot, time = state_traj[frame]
            
            # ============ UPDATE SIMULATION ============
            # Compute glider body points
            x0 = x - f * L * np.cos(theta)
            z0 = z - f * L * np.sin(theta)
            x1 = x0 + L * np.cos(theta)
            z1 = z0 + L * np.sin(theta)
            xl = x - f * L * np.cos(theta) - L_lift * np.cos(theta + phi)
            zl = z - f * L * np.sin(theta) - L_lift * np.sin(theta + phi)
            
            glider_body.set_data([xl, x0, x1], [zl, z0, z1])
            com_marker.set_data([x], [z])
            
            # Update trail with gradient
            trail_points.append([x, z])
            if len(trail_points) > 1:
                segments = [[trail_points[i], trail_points[i + 1]] 
                        for i in range(len(trail_points) - 1)]
                colors = np.linspace(0, 1, len(segments))
                trail_collection.set_segments(segments)
                trail_collection.set_array(colors)
            
            # ============ UPDATE TIME SERIES PLOTS ============
            x_axis = range(frame + 1)
            
            # Linear velocities
            xdot_data.append(xdot)
            zdot_data.append(zdot)
            xdot_line.set_data(x_axis, xdot_data)
            zdot_line.set_data(x_axis, zdot_data)
            
            # Angular velocities
            thetadot_data.append(thetadot)
            if frame < len(control_traj):
                phidot_data.append(control_traj[frame, 0])
                phidot_line.set_data(x_axis, phidot_data)
            thetadot_line.set_data(x_axis, thetadot_data)
            
            # Attack angle (pre-computed)
            attack_data.append(attack_angles[frame])
            attack_line.set_data(x_axis, attack_data)
            
            # Tracking error (pre-computed)
            error_data.append(weighted_errors[frame])
            error_line.set_data(x_axis, error_data)
            
            return (glider_body, com_marker, trail_collection, xdot_line, zdot_line,
                    thetadot_line, phidot_line, attack_line, error_line)
        
        # ==================== CREATE ANIMATION ====================
        ani = animation.FuncAnimation(
            fig, update, frames=n_frames + 50,
            init_func=init, blit=False, 
            interval=1000 / fps, repeat=False
        )
        
        if save_option:
            save_path = f"{title}.gif"
            print(f"Saving animation to {save_path}...")
            ani.save(save_path, writer='pillow', fps=fps, dpi=100)
            print("Animation saved!")
            plt.close(fig)
            return ani

        try:
            fig.canvas.manager.set_window_title("Glider Perching OCP")
        except Exception:
            pass
        plt.show()
        return ani