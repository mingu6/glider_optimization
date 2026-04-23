from casadi import (
    SX, Function,
    sin, cos, tanh, atan2, sqrt,
    dot, gradient, substitute,
    vertcat, vcat, diag,
    pi, fmax, fmin
)
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
from ..config import Config, EvaluationMode
class GliderPerching :
    def __init__(self, config: Config, project_name='glider-perching', wing_reference_geometry=None):
        self.project_name = project_name
        self.config = config
        self.wing_reference_geometry = wing_reference_geometry

    def C_L(self, alpha):
        return 2 * sin(alpha) * cos(alpha)

    def C_D(self, alpha):
        return 2 * sin(alpha) * sin(alpha)

    def C_M(self, alpha):
        return -self.C_L(alpha) * 0.25

    def mc_to_wcom(self, l_w):
        return l_w+0.003
    
    def scale(self, x, min, max):
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
        # 2D fallback parameters (kept explicit)
        m_2d = 0.1
        l_w_i_2d = -0.005
        l_w_f_2d = -0.015
        #l_2d = 0.26
        l_2d = 0.344
        l_e_2d = 0.02
        rho_2d = 1.225
        g_2d = 9.81
        S_w_2d = 0.158
        S_e_2d = 0.017
        chord_e=0.06
        mu_air_2d = 1.789e-5

        plane_cfg = getattr(self.config, "plane", {}) or {}
        dyn_cfg = plane_cfg.get("dyn", {}) if isinstance(plane_cfg, dict) else {}
        flow_cfg = plane_cfg.get("flow", {}) if isinstance(plane_cfg, dict) else {}
        wing_cfg = plane_cfg.get("wing", {}) if isinstance(plane_cfg, dict) else {}

        m = float(dyn_cfg.get("mass", m_2d))
        l_w_i = float(dyn_cfg.get("l_w_i", l_w_i_2d))
        l_w_f = float(dyn_cfg.get("l_w_f", l_w_f_2d))
        l_body = float(dyn_cfg.get("l", l_2d))
        self.l = l_body
        l_e = float(dyn_cfg.get("l_e", l_e_2d))
        rho = float(flow_cfg.get("rho", rho_2d))
        g = float(g_2d)
        S_w = float(dyn_cfg.get("S_w", S_w_2d))
        S_e = float(dyn_cfg.get("S_e", S_e_2d))
        mu_air = float(flow_cfg.get("mu", mu_air_2d))

        chord_2d = float(np.abs(l_w_f - l_w_i))
        #l_w_m_2d= 0.5 * (l_w_i + l_w_f) 
        l_w_cg = 0.5 * (l_w_i + l_w_f)           # structural CoM fallback (midchord)
        l_w_ac = l_w_i + 0.25 * (l_w_f - l_w_i)  # aerodynamic center fallback (c/4)
        chord = chord_2d
        l_w_z = 0.0                                 # z-height of structural CoM; nonzero with dihedral
        #l_w_m = l_w_m_2d

        dihedral_deg = float(wing_cfg.get("dihedral", 0.0)) if isinstance(wing_cfg, dict) else 0.0

        y_half = wing_cfg.get("y_half", None) if isinstance(wing_cfg, dict) else None
        c_half = wing_cfg.get("c_half", None) if isinstance(wing_cfg, dict) else None
        xle_half = wing_cfg.get("xle_half", None) if isinstance(wing_cfg, dict) else None

        if isinstance(y_half, list) and isinstance(c_half, list) and len(y_half) >= 2 and len(c_half) >= 2:
            y = np.asarray(y_half, dtype=float)
            c_raw = np.asarray(c_half, dtype=float)
            # c_half may supply only root/tip (2 values); interpolate onto y_half grid if needed
            if len(c_raw) != len(y):
                y_c = np.linspace(y[0], y[-1], len(c_raw))
                c = np.interp(y, y_c, c_raw)
            else:
                c = c_raw
            if np.all(np.diff(y) >= 0):
                S_half = float(np.trapezoid(c, y))
                span = float(2.0 * y[-1]) if y[-1] > 0 else 0.0
                if S_half > 0 and span > 0:
                    S_w = float(2.0 * S_half)
                    chord = float(S_w / span)

                    den_cg = float(np.trapezoid(c, y))
                    if den_cg > 0:
                        # chord-weighted mean z-height for dihedral inertia arm
                        z_vals = y * np.tan(np.deg2rad(dihedral_deg))
                        l_w_z = float(np.trapezoid(c * z_vals, y) / den_cg)

                    if isinstance(xle_half, list) and len(xle_half) == len(y_half):
                        xle = np.asarray(xle_half, dtype=float)
                        den = float(np.trapezoid(c, y))
                        if den > 0:
                            l_w_ac = float(np.trapezoid(c * (xle + 0.25 * c), y) / den)
                            l_w_cg = float(np.trapezoid(c * (xle + 0.50 * c), y) / den)

        dynamic_centroid_enabled = bool(wing_cfg.get("dynamic_centroid", False)) if isinstance(wing_cfg, dict) else False
        if dynamic_centroid_enabled and isinstance(self.wing_reference_geometry, dict):
            l_override = self.wing_reference_geometry.get("l_w_m", None)
            if l_override is not None:
                try:
                    # l_w_ac stays at c/4 (NeuralFoil convention — do NOT override)
                    # chord stays fixed (planform geometry, not airfoil section shape)
                    l_w_cg = float(l_override)
                except Exception:
                    pass

        m_f = 0.4 * m
        # Role 1 — pitch inertia: structural CoM arm (c/2), sets rotational inertia of the wing
        l_w = l_w_cg
        
        chebyshev_deg = self.config.reducedModel.chebyshev_degree

        phi_CL = SX.sym("phi_CL", (chebyshev_deg+1)**2, 1)
        phi_CD = SX.sym("phi_CD", (chebyshev_deg+1)**2, 1)
        phi_CM = SX.sym("phi_CM", (chebyshev_deg+1)**2, 1)
                
        parameter = [phi_CL, phi_CD, phi_CM]
        self.dyn_auxvar = vcat(parameter)

        m_w = 0.6 * m * S_w / (S_w + S_e)
        m_e = 0.6 * m * S_e / (S_w + S_e)
        #l_f = -(l_w * m_w + (l + l_e) * m_e) / m_f      # vector to fuselage CoM
        l_f= -0.025
        inertia = m_w * (l_w ** 2 + l_w_z ** 2) + m_e * (l_body + l_e) ** 2 + m_f * l_f ** 2
        
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

        # Role 3 — torque arm: F_w applied at c/4 (l_w_ac), consistent with NeuralFoil CM convention;
        #           system CoM (com_a) uses structural CoM (l_w_cg) for mass distribution
        com_e = l_body + l_e # simplifying assumption, the elevator's com doesn't depend on the angle (quasi static assumption)                
        com_f = l_f
        com_a = (l_w_cg*m_w + com_e*m_e + com_f*m_f) / (m_w + m_e + m_f)

        # Role 2 — velocity at aero center: alpha, v_w, Re evaluated at c/4 (matches NeuralFoil convention)
        x_wdot = xdot + l_w_ac * thetadot * sin(theta)
        z_wdot = zdot - l_w_ac * thetadot * cos(theta)
        x_edot = xdot + l_body * thetadot * sin(theta) + l_e * (thetadot + phidot) * sin(theta + phi)
        z_edot = zdot - l_body * thetadot * cos(theta) - l_e * (thetadot + phidot) * cos(theta + phi)
        
        v_w = sqrt(x_wdot * x_wdot + z_wdot * z_wdot + 1e-8) # flow/air speed
        
        alpha_w = theta - atan2(z_wdot, x_wdot)
        Re = rho * v_w * chord / mu_air
        
        nfConfig = self.config.neuralFoilSampling
        
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

        # Sugar-Gabor (2018) Eq. 13 — quasi-unsteady corrections (Terms 2 & 3)
        # Term 2: ½ρ·c·S·Clα·θ̇ in lift direction  (∂Γ/∂t ≈ ½v_w·c·Clα·θ̇, dS=c·dy)
        # Term 3: ½ρ·c·S·CL·θ̇  in drag direction  (dn̂/dt = -θ̇·ĉ)
        # Both terms are O(v_w⁰): no singularity at low speed.
        # Compatible with use_3d_llt: true or false — only depends on CL_w and geometry.
        if self.config.neuralFoilSampling.unsteady:
            # Exact dCL/dalpha: introduce a fresh pure symbol for alpha so that CasADi can
            # differentiate w.r.t. it directly.  Re_scaled_clamped is kept as-is (treated as
            # a constant in this derivative, which is physically correct).
            # After differentiation, substitute the actual alpha_w derived expression back.
            alpha_sym                = SX.sym('alpha_sym')
            alpha_scaled_sym         = self.scale(alpha_sym, a0_min, a0_max)
            alpha_scaled_clamped_sym = fmax(-1.0, fmin(1.0, alpha_scaled_sym))
            w_sym  = self.smooth_gate(alpha_sym, a0_min, a0_max, sharpness)
            X_sym  = self.cheb_basis_2d(alpha_scaled_clamped_sym, Re_scaled_clamped, chebyshev_deg)
            CL_sym = w_sym * dot(X_sym, phi_CL) + (1 - w_sym) * self.C_L(alpha_sym)
            dCL_dalpha = substitute(gradient(CL_sym, alpha_sym), alpha_sym, alpha_w)
            F_w2 = 0.5 * rho * chord * S_w * dCL_dalpha * thetadot * vertcat(-z_wdot,  x_wdot)
            F_w3 = 0.5 * rho * chord * S_w * CL_w         * thetadot * vertcat(-x_wdot, -z_wdot)
        else:
            F_w2 = SX.zeros(2, 1)
            F_w3 = SX.zeros(2, 1)

        # force vectors for aerodynamic surfaces (lift, drag, gravity)
        F_Lw = CL_w * vertcat(-z_wdot, x_wdot)  # lift force vector (proportional to)
        F_Dw = CD_w * vertcat(-x_wdot, -z_wdot) # drag force vector (proportional to)
        F_w = 0.5 * rho * v_w * S_w * (F_Lw + F_Dw) + F_w2 + F_w3
        M_w = 0.5 * rho * v_w**2 * S_w * chord * CM_w

        alpha_e = theta + phi - atan2(z_edot, x_edot)
        v_e = sqrt(x_edot * x_edot + z_edot * z_edot + 1e-8)   # flow/air speed
        F_Le = self.C_L(alpha_e) * vertcat(-z_edot, x_edot)    # lift force vector (proportional to)
        F_De = self.C_D(alpha_e) * vertcat(-x_edot, -z_edot)   # drag force vector (proportional to)
        F_e = 0.5 * rho * v_e * S_e * (F_Le + F_De)
        M_e = 0.5 * rho * v_e**2 * S_e * chord_e * self.C_M(alpha_e)

        # compute torques with respect to fixed reference point induced by forces

        # moment arms (vector from reference point of state to wing/elevator/fuselage)
        r_w = [ (- l_w_ac + com_a) * cos(theta), (- l_w_ac + com_a) * sin(theta) ]
        r_e = [ (- com_e + com_a) * cos(theta), (- com_e + com_a) * sin(theta)]

        τ_w = r_w[1] * F_w[0] - r_w[0] * F_w[1] + M_w
        τ_e = r_e[1] * F_e[0] - r_e[0] * F_e[1] + M_e
        thetaddot = -1. / inertia * (τ_w + τ_e)

        # linear accelerations (F = ma)
        xddot = 1. / m * (F_w[0] + F_e[0])
        zddot = 1. / m * (F_w[1] + F_e[1]) - g
        
        self.debug_f = Function(
            'debug_f',
            [self.X, self.U, self.dyn_auxvar],
            [alpha_w, v_w, CL_w, CD_w, CM_w, F_w, F_e, τ_w, τ_e, xddot, zddot, thetaddot]
        )
        
        self.f = vertcat(xdot, zdot, thetadot, phidot, xddot, zddot, thetaddot, 0)

    def initCost(self, state_weights, wu=0.001, stage_scale = 0.0001, init_state = None):
        # [x   z   theta   phi   xdot   zdot   thetadot]
        self.goal = [0., 0., 0., 0., 0., 0., 0., 0.]
        
        if self.config.evaluation.mode == EvaluationMode.SoftLanding:
            if init_state is None:
                raise RuntimeError("[initCost] Expected an initial state for SoftLanding")
            self.goal[0] = init_state[0]    
        
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

        # min_phi_dot <= phi_dot <= max_phi_dot
        path_inequ_Uub = self.U - max_u
        path_inequ_Ulb = -self.U - max_u
        
        # min_phi <= phi <= max_phi
        path_inequ_Xub = self.X[3] - max_phi
        path_inequ_Xlb = -self.X[3] + min_phi 
        
        const = [path_inequ_Uub, path_inequ_Ulb, path_inequ_Xub, path_inequ_Xlb]
        
        if self.config.evaluation.mode == EvaluationMode.SoftLanding:
            const.append(-self.X[1]) # z >= 0
            const.append(self.X[2] - pi/2 ) # -pi/2 >= theta >= pi/2
            const.append(- self.X[2] - pi/2 ) 
            
            const.append(- (self.X[1] - self.l * sin(self.X[2]) ) ) # back-z >= 0
            const.append(- (self.X[1] + self.l * sin(self.X[2]) ) ) # front-z >= 0
                
        self.path_inequ = vcat(const)
        self.final_inequ = vcat(const[2:])

    def play_animation(self, state_traj, control_traj, 
                    save_option=False, title='glider-perching', fps=30):
        """
        Create stunning glider perching animation with mode-specific visualizations.
        
        For SoftLanding: Shows ground, dynamic distance measurement, altitude display
        For Perching: Shows target circle at goal position
        
        Args:
            state_traj: State trajectory (N x 8) - [x, z, theta, phi, xdot, zdot, thetadot, t]
            control_traj: Control trajectory (N x 1) - [phidot]
            save_option: Whether to save animation as GIF
            title: Filename for saved animation
            fps: Frames per second
        """
        # ==================== PRE-COMPUTE ALL METRICS ====================
        n_frames = len(state_traj)
        is_soft_landing = self.config.evaluation.mode == EvaluationMode.SoftLanding
        
        # Attack angles: theta - arctan2(zdot, xdot)
        attack_angles = state_traj[:, 2] - np.arctan2(state_traj[:, 5], state_traj[:, 4])
        
        # Weighted tracking errors
        if is_soft_landing:
            goal = np.array([state_traj[0][0], 0., 0., 0., 0., 0., 0., 0.])
        else:
            goal = self.goal
        errors = state_traj - goal
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
        x_target, z_target, theta_target = goal[0], goal[1], goal[2]
        x0_target = x_target - f * L * np.cos(theta_target)
        z0_target = z_target - f * L * np.sin(theta_target)
        x1_target = x0_target + L * np.cos(theta_target)
        z1_target = z0_target + L * np.sin(theta_target)
                
        # ==================== FIGURE SETUP ====================
        with plt.style.context('seaborn-v0_8-darkgrid'):
            fig = plt.figure(figsize=(14, 10), facecolor='#F5F5F5')
            
            mode_title = "Soft Landing" if is_soft_landing else "Perching"
            fig.suptitle(f'Glider {mode_title} Trajectory Optimization', 
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
            ax_sim.set_title(f"Glider {mode_title} Simulation", fontsize=12, fontweight='bold', pad=10)
            ax_sim.set_xlabel("X Position (m)", fontsize=10)
            ax_sim.set_ylabel("Z Position (m)", fontsize=10)
            ax_sim.grid(True, alpha=0.2, linestyle=':')
            
            # Start position marker
            ax_sim.plot(state_traj[0, 0], state_traj[0, 1], 'x', color='black', 
                       markersize=8, markeredgewidth=2, label='Start', zorder=4)

            # ==================== MODE-SPECIFIC VISUALIZATION ====================
            if is_soft_landing:
                # === SOFT LANDING MODE ===
                
                # Ground line
                ground_x_min = x_min - pad_x
                ground_x_max = x_max + pad_x
                ax_sim.axhline(0, color='#8B4513', linewidth=3, linestyle='-', alpha=0.8, label='Ground', zorder=2)
                
                # Add ground fill below z=0 for visual effect
                ax_sim.fill_between([ground_x_min, ground_x_max], [z_min - pad_z, z_min - pad_z], 
                                   [0, 0], color='#D2691E', alpha=0.15, zorder=1)
                
                # Landing target zone (vertical dashed line)
                landing_zone_line = ax_sim.axvline(x_target, color='#FF5252', linewidth=2.5, 
                                                   linestyle='--', alpha=0.7, label='Landing Target', zorder=3)
                
                # Landing zone rectangle (highlight area)
                landing_zone_width = 0.3  # meters
                landing_zone_rect = Rectangle((x_target - landing_zone_width/2, z_min - pad_z), 
                                             landing_zone_width, pad_z,
                                             facecolor='#FF5252', alpha=0.1, edgecolor='none', zorder=1)
                ax_sim.add_patch(landing_zone_rect)
                
                # Distance measurement arrow (will be updated dynamically)
                distance_arrow = ax_sim.annotate('', xy=(x_target, -0.5), xytext=(state_traj[0, 0], -0.5),
                    arrowprops=dict(arrowstyle='<->', color='#FF5252', lw=3, shrinkA=0, shrinkB=0),
                    zorder=10)
                
                # Distance text (will be updated dynamically)
                distance_text = ax_sim.text((state_traj[0, 0] + x_target) / 2, -0.7, '',
                    fontsize=12, fontweight='bold', color='#FF5252', ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#FF5252', linewidth=2),
                    zorder=11)
                
                # Altitude display (top-left corner, will be updated)
                altitude_text = ax_sim.text(0.02, 0.98, '', transform=ax_sim.transAxes,
                    fontsize=12, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.7', facecolor='white', alpha=0.95, 
                             edgecolor='#2E86AB', linewidth=2.5),
                    zorder=15)
                
                # Velocity magnitude display (top-left corner, below altitude)
                velocity_text = ax_sim.text(0.02, 0.88, '', transform=ax_sim.transAxes,
                    fontsize=11, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95, 
                             edgecolor='#E63946', linewidth=2.5),
                    zorder=15)
                
            else:
                # === PERCHING MODE ===
                target_circle = Circle((x_target, z_target), 0.1, fill=False, 
                                    edgecolor='#FF5252', linestyle='-', linewidth=3, 
                                    zorder=4, label='Target')
                ax_sim.add_patch(target_circle)
                
                for radius_mult in [1.5, 2.0, 2.5]:
                    glow_circle = Circle((x_target, z_target), 0.1 * radius_mult, fill=False,
                                       edgecolor='#FF5252', linestyle=':', linewidth=1,
                                       alpha=0.3 / radius_mult, zorder=3)
                    ax_sim.add_patch(glow_circle)
                
                distance_arrow = None
                distance_text = None
                altitude_text = None
                velocity_text = None
            
            # Glider artists (common to both modes)
            glider_body, = ax_sim.plot([], [], 'o-', lw=4, color='#2E86AB', 
                                    markersize=8, markerfacecolor='#A23B72',
                                    markeredgewidth=2, markeredgecolor='white',
                                    label='Glider', zorder=5)
            com_marker, = ax_sim.plot([], [], 'o', markersize=6, color='red',
                                    markerfacecolor='yellow', markeredgewidth=1.5,
                                    markeredgecolor='red', alpha=0.7, zorder=6)
            trail_collection = LineCollection([], linewidths=2, alpha=0.6, cmap='viridis')
            ax_sim.add_collection(trail_collection)
            ax_sim.legend(loc='upper right', fontsize=9, framealpha=0.9)
            
            # ==================== LINEAR VELOCITIES ====================
            ax_vel = fig.add_subplot(gs[2, 0])
            ax_vel.set_xlim(0, n_frames - 1)
            ax_vel.set_ylim(vel_range)
            ax_vel.set_title("Linear Velocities", fontsize=11, fontweight='bold', pad=10)
            ax_vel.set_ylabel("Velocity (m/s)", fontsize=9)
            ax_vel.set_xlabel("Frame", fontsize=9)
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
            ax_ang.set_xlabel("Frame", fontsize=9)
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
                
                if is_soft_landing:
                    distance_text.set_text('')
                    altitude_text.set_text('')
                    velocity_text.set_text('')
                
                trail_points.clear()
                xdot_data.clear()
                zdot_data.clear()
                thetadot_data.clear()
                phidot_data.clear()
                attack_data.clear()
                error_data.clear()
                
                artists = [glider_body, com_marker, trail_collection, xdot_line, zdot_line,
                          thetadot_line, phidot_line, attack_line, error_line]
                
                if is_soft_landing:
                    artists.extend([distance_arrow, distance_text, altitude_text, velocity_text])
                
                return tuple(artists)
            
            def update(frame):
                """Update all artists for current frame."""
                if frame >= n_frames:
                    artists = [glider_body, com_marker, trail_collection, xdot_line, zdot_line,
                          thetadot_line, phidot_line, attack_line, error_line]
                
                    if is_soft_landing:
                        artists.extend([distance_arrow, distance_text, altitude_text, velocity_text])
                    
                    return tuple(artists) 
                
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
                
                # ============ SOFT LANDING SPECIFIC UPDATES ============
                if is_soft_landing:
                    # Update distance arrow and text
                    horizontal_distance = abs(x - x_target)
                    arrow_height = -0.5  # Fixed height for arrow
                    
                    distance_arrow.xy = (x_target, arrow_height)
                    distance_arrow.set_position((x, arrow_height))
                    
                    distance_text.set_position(((x + x_target) / 2, arrow_height - 0.2))
                    distance_text.set_text(f'{horizontal_distance:.2f} m')
                    
                    distance_arrow.arrowprops['color'] = '#FF5252'
                    distance_text.set_color('#FF5252')
                    distance_text.get_bbox_patch().set_edgecolor('#FF5252')
                    
                    altitude = max(0, z) 
                    altitude_color = '#00FF00' if altitude < 0.5 else '#2E86AB' if altitude < 2.0 else '#FFA500'
                    altitude_text.set_text(f'Altitude: {altitude:.2f} m')
                    altitude_text.get_bbox_patch().set_edgecolor(altitude_color)
                    
                    velocity_magnitude = np.sqrt(xdot**2 + zdot**2)
                    vel_color = '#00FF00' if velocity_magnitude < 2.0 else '#FFA500' if velocity_magnitude < 4.0 else '#FF5252'
                    velocity_text.set_text(f'Speed: {velocity_magnitude:.2f} m/s')
                    velocity_text.get_bbox_patch().set_edgecolor(vel_color)
                
                # ============ UPDATE TIME SERIES PLOTS ============
                x_axis = range(frame + 1)
                
                xdot_data.append(xdot)
                zdot_data.append(zdot)
                xdot_line.set_data(x_axis, xdot_data)
                zdot_line.set_data(x_axis, zdot_data)
                
                thetadot_data.append(thetadot)
                if frame < len(control_traj):
                    phidot_data.append(control_traj[frame, 0])
                    phidot_line.set_data(x_axis, phidot_data)
                thetadot_line.set_data(x_axis, thetadot_data)
                
                attack_data.append(attack_angles[frame])
                attack_line.set_data(x_axis, attack_data)
                
                error_data.append(weighted_errors[frame])
                error_line.set_data(x_axis, error_data)
                
                artists = [glider_body, com_marker, trail_collection, xdot_line, zdot_line,
                          thetadot_line, phidot_line, attack_line, error_line]
                
                if is_soft_landing:
                    artists.extend([distance_arrow, distance_text, altitude_text, velocity_text])
                
                return tuple(artists)
            
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
                fig.canvas.manager.set_window_title(f"Glider {mode_title} OCP")
            except Exception:
                pass
            plt.show()
            return ani