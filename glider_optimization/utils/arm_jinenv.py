
import numpy as np
from casadi import (SX, Function, cos, diag, dot, exp, fmax, gradient,
                    horzcat, jacobian, mtimes, pi, sin, solve, sqrt, sum1,
                    vcat, vertcat)

G = 9.81

def Rz(a):
    return vertcat(horzcat(cos(a), -sin(a), SX(0)),
                   horzcat(sin(a), cos(a), SX(0)),
                   horzcat(SX(0), SX(0), SX(1)))

def Ry(a):
    return vertcat(horzcat(cos(a), SX(0), sin(a)),
                   horzcat(SX(0), SX(1), SX(0)),
                   horzcat(-sin(a), SX(0), cos(a)))

class ArmThrowing:

    N_DOF = 4
    N_SEG = 3
    JOINT_SEG = [0, 0, 1, 2]          # which segment carries each motor

    def __init__(self, config=None, project_name='arm-throwing',
                 arm_geometry=None):
        self.project_name = project_name
        self.config = config
        g = dict(arm_geometry or {})
        if config is not None:
            a = config.arm
            for key, val in (('shoulder_h', a.shoulder_h),
                             ('M_STRUCT', a.struct_mass),
                             ('C_TOTAL', a.torque_budget),
                             ('K_MOTOR', a.motor_mass_per_Nm),
                             ('M_BALL', a.ball_mass),
                             ('R0', list(a.proximal_radii)),
                             ('z_target', a.z_target)):
                g.setdefault(key, val)
        self.shoulder_h = g.get('shoulder_h', 1.30)
        self.M_STRUCT = g.get('M_STRUCT', 3.60)      # kg, structural budget
        self.C_TOTAL = g.get('C_TOTAL', 170.0)       # N m, torque budget
        self.K_MOTOR = g.get('K_MOTOR', 0.010)       # kg per N m of capacity
        self.M_BALL = g.get('M_BALL', 0.145)         # kg
        self.RHO = g.get('RHO', 1050.0)              # kg/m3 limb density
        self.R0 = g.get('R0', [0.045, 0.038, 0.028])  # proximal radii
        self.z_target = g.get('z_target', 1.00)
        self._build_design_map()

    # ------------------------------------------------------------------
    # design psi -> auxvar   (the expensive-ish map, outside the OCP)
    # ------------------------------------------------------------------
    def _frustum(self, L, tp, mass, r0, n=20):
        xs, ws = np.polynomial.legendre.leggauss(n)
        a0 = SX(0); s1 = SX(0); sax = SX(0); str_ = SX(0)
        for i in range(n):
            xi = 0.5 * L * (xs[i] + 1.0)
            wi = 0.5 * L * ws[i]
            r = r0 * (1.0 + (tp - 1.0) * xi / L)
            A = pi * r ** 2
            a0 = a0 + wi * A
            s1 = s1 + wi * A * xi
            sax = sax + wi * A * 0.5 * r ** 2
            str_ = str_ + wi * A * (0.25 * r ** 2 + xi ** 2)
        dens = mass / a0
        com = s1 / a0
        Iax = dens * sax
        Itr = dens * str_ - mass * com ** 2
        return com, Iax, Itr

    def _build_design_map(self):
        psi = SX.sym('psi', 13)
        L = psi[0:3]
        tp = psi[3:6]
        e_f = exp(psi[6:9]); f = e_f / sum1(e_f) * self.M_STRUCT
        e_c = exp(psi[9:13]); c = e_c / sum1(e_c) * self.C_TOTAL

        m_mot = self.K_MOTOR * c
        mm, cc, Ia, It = [], [], [], []
        for b in range(self.N_SEG):
            com_s, Iax_s, Itr_s = self._frustum(L[b], tp[b], f[b], self.R0[b])
            m_m = SX(0)
            for j in range(self.N_DOF):
                if self.JOINT_SEG[j] == b:
                    m_m = m_m + m_mot[j]
            m_tip = self.M_BALL if b == self.N_SEG - 1 else SX(0)
            m_tot = f[b] + m_m + m_tip
            com = (f[b] * com_s + m_tip * L[b]) / m_tot     # motor at origin
            Itr = (Itr_s + f[b] * (com_s - com) ** 2
                   + m_m * com ** 2 + m_tip * (L[b] - com) ** 2)
            mm.append(m_tot); cc.append(com); Ia.append(Iax_s); It.append(Itr)

        aux = vertcat(L, vcat(mm), vcat(cc), vcat(Ia), vcat(It))
        self.psi_to_auxvar = Function('psi_to_auxvar', [psi], [aux])
        self.psi_to_caps = Function('psi_to_caps', [psi], [c])
        self.psi_to_mass = Function('psi_to_mass', [psi], [sum1(vcat(mm))])
        self.n_psi = 13

    def nominal_psi(self):
        if self.config is not None:
            return np.asarray(self.config.arm.initial_psi, dtype=float)
        return np.array([0.32, 0.27, 0.11,
                         1.0, 1.0, 1.0,
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    def initDyn(self):
        """State X = [q(4), qd(4), tau(4), h], control U = tau_dot(4).

        Torque is a STATE and its rate is the control. A real motor cannot
        reverse its torque instantly, and the transcription's constraints are
        evaluated one node at a time -- there is no way to write |tau_{k+1} -
        tau_k| <= rate * h against a control that only exists at a single node.
        Carrying tau in the state turns that coupling into an ordinary bound on
        the control, and has two further consequences that matter: the
        torque-speed curve becomes a STATE constraint, so it also binds at the
        release node where there is no control, and tau(0) = 0 forces the swing
        to build up load instead of starting at full torque.

        With tau as a control instead, the optimiser ambles through the swing
        and does the throw in the last one or two intervals, driving three
        joints onto their no-load speed inside a single 20 ms step.
        """
        nd, ns = self.N_DOF, self.N_SEG
        q = SX.sym('q', nd)
        qd = SX.sym('qd', nd)
        tau = SX.sym('tau', nd)
        taudot = SX.sym('taudot', nd)
        h = SX.sym('h')

        L = SX.sym('L', ns); m = SX.sym('m', ns); com = SX.sym('com', ns)
        Iax = SX.sym('Iax', ns); Itr = SX.sym('Itr', ns)
        self.dyn_auxvar = vertcat(L, m, com, Iax, Itr)

        self.X = vertcat(q, qd, tau, h)
        self.U = taudot
        self.tau = tau

        ex = SX([1, 0, 0]); ey = SX([0, 1, 0]); ez = SX([0, 0, 1])
        R0 = Rz(q[0])
        R1 = mtimes(R0, Ry(q[1]))
        R2 = mtimes(R1, Ry(q[2]))
        R3 = mtimes(R2, Ry(q[3]))
        Rs = [R1, R2, R3]

        p = [vertcat(SX(0), SX(0), SX(self.shoulder_h))]
        for b in range(ns):
            p.append(p[b] + mtimes(Rs[b], L[b] * ex))
        self.p_tip = p[-1]
        c = [p[b] + mtimes(Rs[b], com[b] * ex) for b in range(ns)]

        w = [qd[0] * ez + qd[1] * mtimes(R0, ey)]
        w.append(w[0] + qd[2] * mtimes(R1, ey))
        w.append(w[1] + qd[3] * mtimes(R2, ey))

        T = SX(0); V = SX(0)
        for b in range(ns):
            v = mtimes(jacobian(c[b], q), qd)
            Iw = mtimes(mtimes(Rs[b], diag(vertcat(Iax[b], Itr[b], Itr[b]))),
                        Rs[b].T)
            T = T + 0.5 * m[b] * dot(v, v) + 0.5 * dot(w[b], mtimes(Iw, w[b]))
            V = V + m[b] * G * c[b][2]
        Lag = T - V

        p_gen = gradient(T, qd)                    # V independent of qd
        M = jacobian(p_gen, qd)
        n_vec = mtimes(jacobian(p_gen, q), qd) - gradient(Lag, q)
        qdd = solve(M, tau - n_vec)

        self.f = vertcat(qd, qdd, taudot, SX(0))   # h is constant

        self.v_tip = mtimes(jacobian(self.p_tip, q), qd)
        self.dyn_fn = Function('dyn', [self.X, self.U, self.dyn_auxvar],
                               [self.f])
        # Inverse-dynamics form: M(q) qdd + n(q,qd) = tau.  Exposing M and n
        # separately lets the transcription treat qdd as a decision variable,
        # so the NLP contains no matrix inverse. The forward form keeps a
        # symbolic 4x4 solve that RK4 calls four times per interval, which is
        # what made the NLP take tens of minutes.
        self.mass_fn = Function('Mmat', [q, self.dyn_auxvar], [M])
        self.bias_fn = Function('nbias', [q, qd, self.dyn_auxvar], [n_vec])
        self.tip_fn = Function('tip', [self.X, self.dyn_auxvar],
                               [self.p_tip, self.v_tip])
        self.energy_fn = Function('energy', [self.X, self.dyn_auxvar], [T + V])

        # Target-parameterised landing/miss, so the outer objective can be
        # evaluated and differentiated for any target without rebuilding the
        # cost. initCost() below still binds a single fixed target, which is
        # what the OCP transcription needs.
        tgt = SX.sym('tgt', 2)
        land_t, tfly_t = self.landing_point(self.p_tip, self.v_tip)
        miss_t = dot(land_t - tgt, land_t - tgt)
        self.landing_fn = Function('landing', [self.X, self.dyn_auxvar],
                                   [land_t, tfly_t])
        self.miss_of_fn = Function('miss_of', [self.X, self.dyn_auxvar, tgt],
                                   [miss_t])
        self.dmiss_dx_fn = Function('dmiss_dx', [self.X, self.dyn_auxvar, tgt],
                                    [gradient(miss_t, self.X)])
        # The miss also depends on the design DIRECTLY (longer segments move the
        # release point and scale the tip velocity), not only through the
        # trajectory the OCP returns. The outer gradient needs both halves.
        self.dmiss_daux_fn = Function('dmiss_daux', [self.X, self.dyn_auxvar, tgt],
                                      [gradient(miss_t, self.dyn_auxvar)])

    # ------------------------------------------------------------------
    def landing_point(self, p_tip, v_tip):
        """Ballistic landing (x, y) of the ball released at p with velocity v."""
        dz = p_tip[2] - self.z_target
        disc = fmax(v_tip[2] ** 2 + 2.0 * G * dz, 1e-9)
        t_fly = (v_tip[2] + sqrt(disc)) / G
        return vertcat(p_tip[0] + v_tip[0] * t_fly,
                       p_tip[1] + v_tip[1] * t_fly), t_fly

    def initCost(self, target, w_miss=200.0, wu=2e-5, w_time=1.0,
                 stage_scale=1e-4, w_taudot=1e-8):
        """target: (x, y) landing point. Release happens at the final node."""
        self.target = np.asarray(target, dtype=float)
        self.cost_auxvar = vcat([])
        nd = self.N_DOF

        # h is a state, so the per-stage w_time * h sums to w_time * N * h over
        # the horizon: this is what makes the throw minimum-time. Without it the
        # timestep is a free variable that nothing pushes down, and the solver
        # happily stretches the swing to several seconds.
        #
        # The effort term is on the torque, which is now a state. The control
        # (its rate) needs a weight of its own for a second reason: it is the
        # only thing putting the control block of the Hessian on the diagonal,
        # and IDOC has to invert that block.
        h = self.X[-1]
        self.path_cost = (wu * dot(self.tau, self.tau)
                          + w_taudot * dot(self.U, self.U)
                          + stage_scale * dot(self.X[nd:2 * nd],
                                              self.X[nd:2 * nd])
                          + w_time * h)
        self.dpath_cost_dx = gradient(self.path_cost, self.X)
        self.dpath_cost_du = gradient(self.path_cost, self.U)

        land, _ = self.landing_point(self.p_tip, self.v_tip)
        err = land - SX(self.target)
        self.miss_sq = dot(err, err)
        self.final_cost = w_miss * self.miss_sq
        self.w_time = w_time
        self.miss_fn = Function('miss', [self.X, self.dyn_auxvar],
                                [self.miss_sq])
        self.land_fn = Function('land', [self.X, self.dyn_auxvar], [land])

    def state_bounds(self, h_min, h_max):
        """Variable bounds for the state, i.e. the timestep and nothing else.

        The timestep must NOT be bounded with a path inequality. h is constant
        along the trajectory (f_h = 0), so h_{k+1} = h_k is already one of the
        transcription's equalities and a row saying "h is at its bound" lies in
        their span. As soon as the throw becomes range-limited, h sits on h_max
        and that redundant row is active at every node at once, which makes the
        active-set Jacobian rank deficient and A H^-1 A^T singular in IDOC. As a
        variable bound it never enters the active set at all, and nothing is
        lost: the sensitivity treats the initial state as design-independent, so
        dh/dpsi is identically zero either way.
        """
        n = self.N_DOF * 3 + 1
        lb = [-1e20] * n
        ub = [1e20] * n
        lb[-1] = h_min
        ub[-1] = h_max
        return lb, ub

    def initConstraints(self, q_min, q_max, caps=None, p_joint=900.0,
                        du_max=900.0):
        """Joint limits plus a REAL actuator model.

        Each joint has a linear torque-speed curve, as a geared DC motor does:

            |tau_j| <= c_j (1 -+ qd_j / w_max_j),    w_max_j = P_joint / c_j

        Two consequences that matter. A joint cannot produce torque at its
        no-load speed, so the arm cannot reach arbitrary joint rates; and
        because w_max = P / c, allocating MORE torque capacity to a joint
        buys LESS speed there. That is the gear-ratio choice, and it turns
        the torque budget into a genuine design trade rather than a scalar
        anyone would max out.

        Without this, the optimiser flicks the wrist to ~140 rad/s and
        "throws" while the hand travels four centimetres.

        The torque-speed curve bounds torque, not speed: adding its two sides
        gives 0 <= 2 c_j, so a joint spinning far past its no-load speed is
        still feasible as long as it is being braked. The explicit rate bound
        |qd_j| <= w_max_j is what actually caps the joint rates.

        Torque being a state (see initDyn) makes the torque-speed curve a
        state-only constraint, so together with the joint and rate limits it is
        imposed at the release node too, through final_inequ -- that node
        carries no control and is otherwise untouched by the path constraints.
        The only path-only constraint left is the torque slew limit, which is a
        plain bound on the control.

        The timestep is deliberately not constrained here: see state_bounds().
        """
        nd = self.N_DOF
        constraint_auxvar = []
        if caps is None:
            caps = SX.sym('caps', nd)
            constraint_auxvar += [caps]
        self.constraint_auxvar = vcat(constraint_auxvar)
        self.q_min = np.asarray(q_min, float)
        self.q_max = np.asarray(q_max, float)
        self.caps = caps
        self.p_joint = p_joint
        self.du_max = du_max

        state_only = []
        for j in range(nd):
            w_max = p_joint / caps[j]
            qd = self.X[nd + j]
            tau_j = self.tau[j]
            state_only += [self.X[j] - self.q_max[j], -self.X[j] + self.q_min[j]]
            state_only += [qd - w_max, -qd - w_max]
            state_only += [tau_j - caps[j] * (1.0 - qd / w_max),
                           -tau_j - caps[j] * (1.0 + qd / w_max)]
        # the timestep is bounded through state_bounds(), never here

        slew = []
        for j in range(nd):
            slew += [self.U[j] - du_max, -self.U[j] - du_max]

        self.path_inequ = vcat(slew + state_only)
        self.final_inequ = vcat(state_only)

    # ------------------------------------------------------------------
    def joint_positions(self, q, aux):
        """Numeric world positions of shoulder, elbow, wrist, tip."""
        aux = np.asarray(aux).ravel()
        L = aux[0:3]
        cz, sz = np.cos(q[0]), np.sin(q[0])
        Rz_ = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        pts = [np.array([0.0, 0.0, self.shoulder_h])]
        ang = 0.0
        R = Rz_
        for b in range(3):
            ang = ang + q[b + 1]
            ca, sa = np.cos(ang), np.sin(ang)
            Ry_ = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
            Rb = Rz_ @ Ry_
            pts.append(pts[-1] + Rb @ np.array([L[b], 0.0, 0.0]))
        return np.array(pts)

    def play_animation(self, state_traj, control_traj, auxvar,
                       save_option=False, title='arm-throwing', fps=30):
        """3D throw animation: arm, ball flight, target, and time series."""
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        nd = self.N_DOF
        aux = np.asarray(auxvar).ravel()
        n_arm = len(state_traj)
        h = float(state_traj[0, -1])

        Xf = state_traj[-1]
        p_rel, v_rel = self.tip_fn(Xf, aux)
        p_rel = np.asarray(p_rel).ravel(); v_rel = np.asarray(v_rel).ravel()
        land = np.asarray(self.land_fn(Xf, aux)).ravel()
        dz = p_rel[2] - self.z_target
        t_fly = (v_rel[2] + np.sqrt(max(v_rel[2] ** 2 + 2 * G * dz, 1e-9))) / G
        n_ball = max(int(t_fly / max(h, 1e-3)), 12)
        tb = np.linspace(0, t_fly, n_ball)
        ball = np.stack([p_rel[0] + v_rel[0] * tb,
                         p_rel[1] + v_rel[1] * tb,
                         p_rel[2] + v_rel[2] * tb - 0.5 * G * tb ** 2], 1)

        joints = np.array([self.joint_positions(state_traj[k, :nd], aux)
                           for k in range(n_arm)])
        tip_speed = np.array([np.linalg.norm(
            np.asarray(self.tip_fn(state_traj[k], aux)[1]).ravel())
            for k in range(n_arm)])
        n_frames = n_arm + n_ball

        with plt.style.context('seaborn-v0_8-darkgrid'):
            fig = plt.figure(figsize=(14, 8), facecolor='#F5F5F5')
            fig.suptitle('3D Throwing Arm — Trajectory Optimisation',
                         fontsize=14, fontweight='bold', y=0.97)
            gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.25,
                                  left=0.04, right=0.95, top=0.9, bottom=0.07,
                                  width_ratios=[1.35, 1.0])
            ax = fig.add_subplot(gs[:, 0], projection='3d')

            xs = np.concatenate([joints[:, :, 0].ravel(), ball[:, 0],
                                 [self.target[0]]])
            ys = np.concatenate([joints[:, :, 1].ravel(), ball[:, 1],
                                 [self.target[1]]])
            zs = np.concatenate([joints[:, :, 2].ravel(), ball[:, 2], [0.0]])
            ax.set_xlim(xs.min() - 0.3, xs.max() + 0.3)
            ax.set_ylim(min(ys.min() - 0.5, -1.0), max(ys.max() + 0.5, 1.0))
            ax.set_zlim(0, max(zs.max() + 0.3, 2.0))
            ax.set_box_aspect((2.6, 1.3, 1.0))
            ax.view_init(elev=16, azim=-62)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
            ax.set_title('Throw', fontsize=12, fontweight='bold')

            gx = np.linspace(xs.min() - 0.3, xs.max() + 0.3, 2)
            gy = np.linspace(min(ys.min() - 0.5, -1.0),
                             max(ys.max() + 0.5, 1.0), 2)
            GX, GY = np.meshgrid(gx, gy)
            ax.plot_surface(GX, GY, np.zeros_like(GX), color='#D2B48C',
                            alpha=0.18, zorder=0)
            ax.plot([0, 0], [0, 0], [0, self.shoulder_h], color='#555',
                    lw=3, alpha=0.7)
            ax.scatter([self.target[0]], [self.target[1]], [self.z_target],
                       s=140, c='#FF5252', marker='v', depthshade=False,
                       label='Target', zorder=10)
            th = np.linspace(0, 2 * np.pi, 40)
            for r in (0.15, 0.30):
                ax.plot(self.target[0] + r * np.cos(th),
                        self.target[1] + r * np.sin(th),
                        np.full_like(th, self.z_target), color='#FF5252',
                        lw=1, alpha=0.5)

            arm_line, = ax.plot([], [], [], 'o-', lw=5, color='#2E86AB',
                                markersize=7, markerfacecolor='#A23B72',
                                markeredgecolor='white', markeredgewidth=1.5,
                                label='Arm', zorder=6)
            tip_trail, = ax.plot([], [], [], '-', lw=2, color='#F77F00',
                                 alpha=0.8, label='Tip path', zorder=5)
            ball_line, = ax.plot([], [], [], '--', lw=2, color='#E63946',
                                 alpha=0.9, label='Ball', zorder=5)
            ball_dot, = ax.plot([], [], [], 'o', markersize=9, color='#E63946',
                                markeredgecolor='white', zorder=11)
            ax.legend(loc='upper left', fontsize=9)
            hud = ax.text2D(0.02, 0.02, '', transform=ax.transAxes, fontsize=10,
                            fontweight='bold', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.5',
                                      facecolor='white', alpha=0.9,
                                      edgecolor='#2E86AB', lw=2))

            ax_q = fig.add_subplot(gs[0, 1])
            ax_q.set_xlim(0, n_arm - 1); ax_q.set_title('Joint angles', fontsize=11,
                                                        fontweight='bold')
            ax_q.set_ylabel('rad', fontsize=9)
            lo = state_traj[:, :nd].min() - 0.3; hi = state_traj[:, :nd].max() + 0.3
            ax_q.set_ylim(lo, hi)
            names = ['sh-az', 'sh-el', 'elbow', 'wrist']
            cols = ['#E63946', '#06A77D', '#118AB2', '#9D4EDD']
            q_lines = [ax_q.plot([], [], lw=2.2, color=cols[j],
                                 label=names[j])[0] for j in range(nd)]
            ax_q.legend(fontsize=7, ncol=4, loc='upper left')

            # torque is a state now; control_traj holds its rate
            torque_traj = state_traj[:, 2 * nd:3 * nd]
            ax_u = fig.add_subplot(gs[1, 1])
            ax_u.set_xlim(0, max(len(torque_traj) - 1, 1))
            ax_u.set_title('Joint torques', fontsize=11, fontweight='bold')
            ax_u.set_ylabel('N m', fontsize=9)
            um = np.abs(torque_traj).max() * 1.15 + 1e-6
            ax_u.set_ylim(-um, um)
            u_lines = [ax_u.plot([], [], lw=2.2, color=cols[j])[0]
                       for j in range(nd)]

            ax_s = fig.add_subplot(gs[2, 1])
            ax_s.set_xlim(0, n_arm - 1)
            ax_s.set_ylim(0, tip_speed.max() * 1.2 + 1e-6)
            ax_s.set_title('Ball speed at hand', fontsize=11, fontweight='bold')
            ax_s.set_ylabel('m/s', fontsize=9); ax_s.set_xlabel('step', fontsize=9)
            s_line, = ax_s.plot([], [], lw=2.5, color='#F77F00')

            def init():
                for ln in [arm_line, tip_trail, ball_line, ball_dot,
                           s_line] + q_lines + u_lines:
                    ln.set_data([], [])
                    if hasattr(ln, 'set_3d_properties'):
                        try:
                            ln.set_3d_properties([])
                        except Exception:
                            pass
                hud.set_text('')
                return ()

            def update(k):
                if k < n_arm:
                    P = joints[k]
                    arm_line.set_data(P[:, 0], P[:, 1])
                    arm_line.set_3d_properties(P[:, 2])
                    tr = joints[:k + 1, -1]
                    tip_trail.set_data(tr[:, 0], tr[:, 1])
                    tip_trail.set_3d_properties(tr[:, 2])
                    ball_dot.set_data([P[-1, 0]], [P[-1, 1]])
                    ball_dot.set_3d_properties([P[-1, 2]])
                    ax_ = range(k + 1)
                    for j in range(nd):
                        q_lines[j].set_data(ax_, state_traj[:k + 1, j])
                    ku = min(k + 1, len(torque_traj))
                    for j in range(nd):
                        u_lines[j].set_data(range(ku), torque_traj[:ku, j])
                    s_line.set_data(ax_, tip_speed[:k + 1])
                    hud.set_text(f'wind-up   t = {k*h:5.3f} s\n'
                                 f'hand speed {tip_speed[k]:5.2f} m/s')
                else:
                    kb = min(k - n_arm, n_ball - 1)   # hold on the last frame
                    ball_line.set_data(ball[:kb + 1, 0], ball[:kb + 1, 1])
                    ball_line.set_3d_properties(ball[:kb + 1, 2])
                    ball_dot.set_data([ball[kb, 0]], [ball[kb, 1]])
                    ball_dot.set_3d_properties([ball[kb, 2]])
                    hud.set_text(f'release {np.linalg.norm(v_rel):5.2f} m/s\n'
                                 f'flight  t = {tb[kb]:5.3f} s\n'
                                 f'miss    {np.linalg.norm(land-self.target):5.3f} m')
                return ()

            ani = animation.FuncAnimation(fig, update, frames=n_frames + 25,
                                          init_func=init, blit=False,
                                          interval=1000 / fps, repeat=False)
            if save_option:
                path = f'{title}.gif'
                print(f'saving {path} ...')
                ani.save(path, writer='pillow', fps=fps, dpi=90)
                print('saved')
                plt.close(fig)
                return ani
            plt.show()
            return ani