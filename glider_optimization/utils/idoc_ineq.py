import numpy as np
from dataclasses import dataclass

@dataclass
class IDOC_Context:
    H_t: np.ndarray
    H_T: np.ndarray
    A_t: list
    A_T: np.ndarray
    B_t: np.ndarray
    B_T: np.ndarray
    C_t: list
    C_T: np.ndarray
    ns: int
    nc: int
    T: int


def build_blocks_idoc(auxsys_COC, delta=None) -> IDOC_Context:
    """
    This function takes the building blocks of the auxiliary COC system defined in the Safe-PDP paper
    and uses them to construct the blocks in our IDOC identities. 

    Inputs:

    auxsys_COC object: Dictionary with values being Jacobian/Hessian blocks of the constraints/cost.

    Outputs: - H_t: List of first T blocks in H
             - H_T: Final state block in H
             - A_t: First T+1 blocks in A (lower diagonal) corresponding to init. state, dynamics + eq. + ineq. (r_{-1}, r_{0}, ... r{T-1})
             - A_T: Final equality + inequality constraints block (no dynamics) (r_T)
             - B_t: First T blocks of B
             - B_T: Final block of B
             - C_t: First T blocks of C except C_0 (C_1, ..., C_T). C_0 is just zeros
             - C_T: Final block of C
             - ns: number of states
             - nc: number of controls
             - T: Horizon

    """
    T = auxsys_COC['horizon']
    ns = auxsys_COC['Lxx_t'][0].shape[0]
    nc = auxsys_COC['Luu_t'][0].shape[0]

    # H blocks
    Lxx_t = np.stack(auxsys_COC['Lxx_t'], axis=0)
    Lxu_t = np.stack(auxsys_COC['Lxu_t'], axis=0)
    Luu_t = np.stack(auxsys_COC['Luu_t'], axis=0)
    H_t = np.block([[Lxx_t, Lxu_t], [Lxu_t.transpose(0, 2, 1), Luu_t]])
    H_T = auxsys_COC['Lxx_T'][0]
    if delta is not None:
        H_t += delta * np.eye(ns+nc)[None, ...]
        H_T += delta * np.eye(ns)

    # A blocks
    GbarHx_t = auxsys_COC['GbarHx_t']
    GbarHu_t = auxsys_COC['GbarHu_t']
    GbarHx_T = auxsys_COC['GbarHx_T'][0]
    dynFx_t = auxsys_COC['dynFx_t']
    dynFu_t = auxsys_COC['dynFu_t']
    
    A_t = [np.block([[GbarHx_t[t], GbarHu_t[t]], [-dynFx_t[t], -dynFu_t[t]]])  for t in range(T)]
    A_T = GbarHx_T

    # B blocks
    Lxe_t = np.stack(auxsys_COC['Lxe_t'], axis=0)
    Lue_t = np.stack(auxsys_COC['Lue_t'], axis=0)
    B_t = np.concatenate((Lxe_t, Lue_t), axis=1)
    B_T = auxsys_COC['Lxe_T'][0]

    # C blocks
    GbarHe_t = auxsys_COC['GbarHe_t']
    dynFe_t = auxsys_COC['dynFe_t']
    C_t = [np.concatenate((GbarHe_t[t], -dynFe_t[t]), axis=0) for t in range(T)]
    C_T = auxsys_COC['GbarHe_T'][0]

    return IDOC_Context(H_t=H_t, H_T=H_T, A_t=A_t, A_T=A_T, B_t=B_t, B_T=B_T,
                        C_t=C_t, C_T=C_T, ns=ns, nc=nc, T=T)


def idoc_full(ctx: IDOC_Context):
    H_t, H_T = ctx.H_t, ctx.H_T
    A_t, A_T = ctx.A_t, ctx.A_T
    B_t, B_T = ctx.B_t, ctx.B_T
    C_t, C_T = ctx.C_t, ctx.C_T
    ns, nc, T = ctx.ns, ctx.nc, ctx.T
    nz = B_T.shape[1]
    
    inv = np.linalg.inv
    
    Hinv_t = inv(H_t)
    Hinv_T = inv(H_T)
    
    # ================== (H^-1 A^T) ======================    
    assert len(Hinv_t) == len(A_t) == T
    
    # Right block
    HinvAT_upper_t_blocks = [Hinv_t[i] @ A_t[i].T for i in range(T)]
    HinvAT_upper_T_block = Hinv_T @ A_T.T 
    
    # Left block
    HinvAT_lower_t_blocks = Hinv_t[..., :ns] 
    # No need to restrict to ns blocks, since the terminal trajectory does not depend on the control
    HinvAT_lower_T_block = Hinv_T 

    # =====================================================
    
    # ================== (A H^-1 B - C) ======================    

    # T + 2 blocks. 1 for the initial state constraint + T for the stage constraints + 1 for the terminal constraint 
    AHinvB_C = (T + 2) * [None]
    
    # First block. Dimension state x theta
    AHinvB_C[0] = HinvAT_lower_t_blocks[0].T @ B_t[0]
    
    for i in range(T):
        # (HinvA.T).T = A Hinv.T = A Hinv [hessian is symmetric]   
        AHinv = HinvAT_upper_t_blocks[i].T
        B = B_t[i]
        C = C_t[i]
        B_next = B_t[i+1] if i < T - 1 else B_T
        HT_inv_next = HinvAT_lower_t_blocks[i+1].T if i < T - 1 else HinvAT_lower_T_block.T
        AHinvB_e = np.matmul(AHinv, B)
        
        AHinvB_e[-ns:] += np.matmul(HT_inv_next, B_next)
        AHinvB_C[i+1] = AHinvB_e - C
        
    AHinvB_C[T+1] = np.matmul(HinvAT_upper_T_block.T, B_T) - C_T
    
    # =====================================================
    
    # ================== (A H^-1 AT) ======================    
    
    # AHinvAT is symmetric, it's convenient to store only one side band 
    AHinvAT_diag = (T + 2) * [None]
    AHinvAT_lower = (T + 1) * [None]
    
    # HinvAT cropped by the identity
    AHinvAT_diag[0] = HinvAT_lower_t_blocks[0][:ns]
    
    for i in range(T):
        AHinv_up = HinvAT_upper_t_blocks[i].T
        HinvAT_low = HinvAT_lower_t_blocks[i+1] if i < T - 1 else HinvAT_lower_T_block
        
        AT = A_t[i].T
        AHinvAT_lower[i] = AHinv_up[:, :ns]
         
        diag = np.matmul(AHinv_up, AT)
        diag[-ns:, -ns:] += HinvAT_low[:ns]
        AHinvAT_diag[i+1] = diag
        
    AHinvAT_lower[T] = HinvAT_upper_T_block.T
    AHinvAT_diag[T+1] = np.matmul(HinvAT_upper_T_block.T, A_T.T)
    
    # =====================================================
    
    
    # =============== (AH^-1A^T)^-1(AH^-1B - C) using Thomas's algorithm ===============
    
    # solve Ax = B where A is tridiagonal
    # AH^-1A^T      := A
    # AH^-1B - C    := B
    
    AHinvAT_upper = [AHinvAT_lower[0].T.copy()]
    for t in range(1, T+1):
        sz = AHinvAT_diag[t].shape[0]
        padding = np.zeros((AHinvAT_lower[t].shape[0], sz - ns))
        AHinvAT_lower[t] = np.concatenate((padding, AHinvAT_lower[t]), axis=1)
        AHinvAT_upper.append(AHinvAT_lower[t].T.copy())

    scratch = [None] * (T+1)
    AHinvAT_AHinvB_C = [None] * (T+2)
    scratch[0] = np.linalg.solve(AHinvAT_diag[0], AHinvAT_upper[0])
    AHinvAT_AHinvB_C[0] = np.linalg.solve(AHinvAT_diag[0], AHinvB_C[0])
    
    for i in range(1, T+2):
        lhs = AHinvAT_diag[i] - np.matmul(AHinvAT_lower[i-1], scratch[i-1])
        if i < T + 1:
            scratch[i] = np.linalg.solve(lhs, AHinvAT_upper[i])

        rhs = AHinvB_C[i] - np.matmul(AHinvAT_lower[i-1], AHinvAT_AHinvB_C[i-1])
        AHinvAT_AHinvB_C[i] = np.linalg.solve(lhs, rhs)
        
    for i in range(T, -1, -1):
        AHinvAT_AHinvB_C[i] -= np.matmul(scratch[i], AHinvAT_AHinvB_C[i+1])
    # =====================================================
    
    # ================== (H^-1 B) ======================    

    HinvB_t = [np.matmul(Hinv_t[i], B_t[i]) for i in range(T)]
    HinvB_T = np.matmul(Hinv_T, B_T)
    
    HinvB = np.vstack(HinvB_t+[HinvB_T])
    
    # =====================================================
    
    # =============== H^-1 A^T (AH^-1A^T)^-1(AH^-1B - C) ===============

    AHinvAT_AHinvB_C_v = np.vstack(AHinvAT_AHinvB_C)

    left_side = [None] * (T+1)

    slide_idx = 0
    for i in range(T + 1):
        l = HinvAT_lower_t_blocks[i] if i < T else HinvAT_lower_T_block
        r = HinvAT_upper_t_blocks[i] if i < T else HinvAT_upper_T_block
        n_cstr = r.shape[1]
        
        x1 = AHinvAT_AHinvB_C_v[slide_idx : slide_idx+ns]
        x2 = AHinvAT_AHinvB_C_v[slide_idx+ns : slide_idx+ns+n_cstr]
        
        left_side[i] =  np.matmul(l, x1)
        left_side[i] += np.matmul(r, x2)
        
        slide_idx += n_cstr

    left_side = np.vstack(left_side)
    
    # =====================================================
    
    # =============== full expression ===============
    
    idoc = left_side - HinvB
    
    dxu_dp_t = idoc[:-ns, :].reshape(T, ns+nc, nz)
    dx_dp = np.concatenate((dxu_dp_t[:, :ns, :], idoc[None, -ns:, :]), axis=0)
    du_dp = dxu_dp_t[:, ns:, :]
    time_ = [k for k in range(T + 1)]
    sol_full = {'state_traj_opt': dx_dp,
                'control_traj_opt': du_dp,
                'time': time_}
    
    return sol_full
