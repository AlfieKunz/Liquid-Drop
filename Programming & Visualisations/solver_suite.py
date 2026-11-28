# Generated in Collaberation with Google Gemini 4 Neon (Extra High). Collection of validatino methods
import numpy as np
from scipy.special import ellipkm1, j0, j1, jn_zeros
from scipy.integrate import quad, cumulative_trapezoid
from scipy.linalg import solve_banded, lu_factor, lu_solve
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.fft import dctn, idctn

TWO_OVER_PI = 2.0 / np.pi

# =============================================================================
# 0. Shared Grid, Kernel, and Lubrication Utilities
# =============================================================================

def kernel_axisym(r_prime, r_i):
    r_sum = r_i + r_prime
    k1 = (r_i - r_prime) / r_sum
    return TWO_OVER_PI * ellipkm1(k1 * k1) / r_sum

def make_mesh(Nr, r_max=1.0, uniform=False, r_0=0.18075432):
    if uniform:
        rHalf = np.linspace(0.0, r_max, Nr + 1)
    else:
        rFine = np.linspace(0.0, r_max, Nr * 40 + 1)
        w = np.exp(-2.0 * rFine / r_max) + 2.0 * np.exp(-(25.0 * (rFine - r_0) / r_max)**2)
        cdf = cumulative_trapezoid(w, rFine, initial=0.0)
        cdf /= cdf[-1]
        rHalf = np.interp(np.linspace(0.0, 1.0, Nr + 1), cdf, rFine)
    CellWidth = np.diff(rHalf)
    r = rHalf[:-1] + 0.5 * CellWidth
    dr = np.diff(r)
    return r, rHalf, CellWidth, dr

def build_lubrication_banded(h, rHalf, CellWidth, dr, FVMMult):
    n = len(h)
    idx = np.arange(1, n)
    h_half = (h[idx - 1] * CellWidth[idx] + h[idx] * CellWidth[idx - 1]) / (CellWidth[idx - 1] + CellWidth[idx])
    flux = (h_half**3 * rHalf[1:-1]) / dr
    AB = np.zeros((3, n))
    AB[0, 1:] = flux * FVMMult[:-1]
    AB[1, 0] = -flux[0] * FVMMult[0]
    AB[1, 1:-1] = -(flux[:-1] + flux[1:]) * FVMMult[1:-1]
    hR = 2.0 * h[-1] - (h[-2] * CellWidth[-1] + h[-1] * CellWidth[-2]) / (CellWidth[-2] + CellWidth[-1])
    AB[1, -1] = -(flux[-1] + 2.0 * (hR**3 * rHalf[-1] / CellWidth[-1])) * FVMMult[-1]
    AB[2, :-1] = flux * FVMMult[1:]
    return AB

def apply_banded_to_matrix(AB, X):
    """Fast O(N^2) tridiagonal-times-matrix (or vector) product for scipy banded AB."""
    out = AB[1, :][:, None] * X if X.ndim == 2 else AB[1, :] * X
    if X.ndim == 2:
        out[:-1, :] += AB[0, 1:][:, None] * X[1:, :]
        out[1:, :]  += AB[2, :-1][:, None] * X[:-1, :]
    else:
        out[:-1] += AB[0, 1:] * X[1:]
        out[1:]  += AB[2, :-1] * X[:-1]
    return out

def stiff_dt(h, v, CellWidth, St, dt_max, c_stiff=1.5, c_h=0.01):
    dt_h = c_h * np.min(np.where(v < 0, h / np.maximum(-v, 1e-300), np.inf))
    h_min = np.min(h)
    imin = np.argmin(h)
    w_half = max(CellWidth[imin], np.sum(CellWidth[h <= 2.0 * h_min]))
    lam_est = 12.0 * w_half / (St * h_min**3)
    dt_stiff = c_stiff / max(lam_est, 1e-12)
    return min(dt_max, dt_h, dt_stiff)

def build_dtn_kernels(r, CellWidth):
    """Your K_P, K_S matrices (with trailing comma fixed in K_S args)."""
    n = len(r)
    K_P = np.zeros((n, n))
    K_S = np.zeros((n, n))
    pts, wts = np.polynomial.legendre.leggauss(20)
    hs = 0.5 * CellWidth
    rj = r[:, None]
    eval_pts = np.outer(hs, pts) + rj

    def k_sym(s, ri):
        return kernel_axisym(ri + s, ri) + kernel_axisym(ri - s, ri)
    def ks_sym(s, ri):
        return s * (kernel_axisym(ri + s, ri) - kernel_axisym(ri - s, ri))

    for i in range(n):
        kp = kernel_axisym(eval_pts, r[i])
        K_P[i, :] = hs * (kp @ wts)
        K_S[i, :] = hs * (((eval_pts - rj) * kp) @ wts)
        K_P[i, i], _ = quad(k_sym, 0.0, hs[i], args=(r[i],), epsabs=0, epsrel=1e-13)
        K_S[i, i], _ = quad(ks_sym, 0.0, hs[i], args=(r[i],), epsabs=0, epsrel=1e-13)
    return K_P, K_S

def build_ntd_matrix(r, CellWidth):
    """Neumann-to-Dirichlet matrix M such that p = M @ d2hdt2."""
    n = len(r)
    M = np.zeros((n, n))
    pts, wts = np.polynomial.legendre.leggauss(20)
    hs = 0.5 * CellWidth
    rj = r[:, None]
    eval_pts = np.outer(hs, pts) + rj

    def m_sym(s, ri):
        return (ri + s) * kernel_axisym(ri + s, ri) + (ri - s) * kernel_axisym(ri - s, ri)

    for i in range(n):
        M[i, :] = hs * ((eval_pts * kernel_axisym(eval_pts, r[i])) @ wts)
        M[i, i], _ = quad(m_sym, 0.0, hs[i], args=(r[i],), epsabs=0, epsrel=1e-13)
    return M

def build_fourier_bessel_dtn(r, rHalf, CellWidth, n_modes=None):
    """Exact cell-averaged Fourier-Bessel spectral DtN operator: d2hdt2 = D_fb @ p."""
    R_max = rHalf[-1]
    n = len(r)
    if n_modes is None:
        n_modes = min(n // 2, 140)
    alphas = jn_zeros(0, n_modes)
    k_m = alphas / R_max
    # Exact cell integral: int_{r_{j-1/2}}^{r_{j+1/2}} r J0(k_m r) dr = [r J1(k_m r)] / k_m
    rJ1 = rHalf[:, None] * j1(np.outer(rHalf, k_m))
    I_jm = (rJ1[1:, :] - rJ1[:-1, :]) / k_m[None, :]
    J0_bar = I_jm / (r * CellWidth)[:, None]
    norm_m = 0.5 * (R_max**2) * (j1(alphas)**2)
    return (J0_bar * (k_m / norm_m)[None, :]) @ I_jm.T


# =============================================================================
# 1. Analytic Unit Test for Any Axisymmetric DtN Operator
# =============================================================================

def verify_dtn_operator(r, apply_dtn_fn, R0=0.12, a=0.02):
    """
    Tests d2hdt2 = -dp/dz|_{z=0} against an exact 3D harmonic ring of dipoles
    at radius R0 and depth z = -a (decays as 1/r^3, mimicking a sharp ring pressure peak).
    """
    p_exact = np.zeros_like(r)
    a_exact = np.zeros_like(r)
    for i, rv in enumerate(r):
        p_exact[i] = quad(lambda th: a / (rv**2 + R0**2 - 2*rv*R0*np.cos(th) + a**2)**1.5,
                          0, np.pi, epsrel=1e-12)[0] / np.pi
        a_exact[i] = quad(lambda th: (3*a**2 - (rv**2 + R0**2 - 2*rv*R0*np.cos(th) + a**2))
                                     / (rv**2 + R0**2 - 2*rv*R0*np.cos(th) + a**2)**2.5,
                          0, np.pi, epsrel=1e-12)[0] / np.pi
    a_num = apply_dtn_fn(p_exact)
    mask = r < 0.4
    rel_err = np.max(np.abs(a_num[mask] - a_exact[mask])) / np.max(np.abs(a_exact[mask]))
    print(f"[DtN Ring-Dipole Test] max rel error (r<0.4): {rel_err:.4e} "
          f"| peak exact={a_exact[mask].max():.3f}, num={a_num[mask].max():.3f}")
    return rel_err


# =============================================================================
# 2. Solver 1: Your DtN (K_P, K_S) + RK4 + h_min^3-aware dt
# =============================================================================

def run_dtn_rk4(St, h_0, U=1.0, Nr=300, r_max=1.0, t_end=None, dt_max=None,
                uniform=False, save_every=4):
    r, rHalf, cw, dr = make_mesh(Nr, r_max, uniform=uniform)
    FVM = St / (12.0 * r * cw)
    K_P, K_S = build_dtn_kernels(r, cw)

    def rhs(h_, v_):
        AB = build_lubrication_banded(h_, rHalf, cw, dr, FVM)
        p_ = solve_banded((1, 1), AB, v_, check_finite=False)
        dpdr = np.zeros(Nr + 1)
        dpdr[1:-1] = np.diff(p_) / dr
        dpdr[-1] = -2.0 * p_[-1] / cw[-1]
        f = rHalf * dpdr
        a_ = (K_P @ ((f[:-1] - f[1:]) / cw)) + 2.0 * (K_S @ ((dpdr[:-1] - dpdr[1:]) / cw))
        return p_, a_

    return _integrate_rk4_1d(r, cw, St, h_0, U, rhs, t_end, dt_max, save_every)


# =============================================================================
# 3. Solver 2: Neumann-to-Dirichlet (M @ a = p, no p-derivatives) + RK4
# =============================================================================

def run_ntd_rk4(St, h_0, U=1.0, Nr=250, r_max=1.0, t_end=None, dt_max=None,
                uniform=True, save_every=4):
    r, rHalf, cw, dr = make_mesh(Nr, r_max, uniform=uniform)
    FVM = St / (12.0 * r * cw)
    M = build_ntd_matrix(r, cw)
    M_lu = lu_factor(M)

    def rhs(h_, v_):
        AB = build_lubrication_banded(h_, rHalf, cw, dr, FVM)
        p_ = solve_banded((1, 1), AB, v_, check_finite=False)
        a_ = lu_solve(M_lu, p_, check_finite=False)
        return p_, a_

    return _integrate_rk4_1d(r, cw, St, h_0, U, rhs, t_end, dt_max, save_every)


# =============================================================================
# 4. Solver 3: Implicit NtD (theta=0.5 Crank-Nicolson, theta=1.0 Backward Euler)
# =============================================================================

def run_ntd_implicit(St, h_0, U=1.0, Nr=250, r_max=1.0, t_end=None, dt_max=None,
                     theta=0.5, uniform=False, save_every=2, picard_tol=1e-11):
    """
    Unconditionally stable against the h_min^{-3} pinch stiffness!
    Use theta=0.5 for 2nd-order Crank-Nicolson, theta=1.0 for 1st-order Backward Euler.
    """
    r, rHalf, cw, dr = make_mesh(Nr, r_max, uniform=uniform)
    FVM = St / (12.0 * r * cw)
    M = build_ntd_matrix(r, cw)
    I_mat = np.eye(Nr)

    if t_end is None:
        t_end = 2.2 * (h_0 / U)
    if dt_max is None:
        dt_max = (h_0 / U) / 400.0  # ~0.02 in HP time units

    h = h_0 + 0.5 * r**2
    v = np.full(Nr, -U)
    t = 0.0
    AB = build_lubrication_banded(h, rHalf, cw, dr, FVM)
    p = solve_banded((1, 1), AB, v, check_finite=False)

    h_hist, p_hist, t_hist, dt_hist = [], [], [], []
    step = 0
    while t < t_end - 1e-12:
        dt_h = 0.05 * np.min(np.where(v < 0, h / np.maximum(-v, 1e-300), np.inf))
        dt = min(dt_max, dt_h, t_end - t)

        if step % save_every == 0:
            h_hist.append(h.copy()); p_hist.append(p.copy())
            t_hist.append(t); dt_hist.append(dt)

        # (L(h^{n+1}) M - theta*dt*I) v^{n+1} = L(h^{n+1}) (M v^n + (1-theta)*dt*p^n)
        rhs_inner = (M @ v) + dt * (1.0 - theta) * p
        h_new = h + dt * v
        for _ in range(15):
            AB_new = build_lubrication_banded(h_new, rHalf, cw, dr, FVM)
            LM = apply_banded_to_matrix(AB_new, M)
            LM[np.diag_indices(Nr)] -= dt * theta
            rhs_vec = apply_banded_to_matrix(AB_new, rhs_inner)
            v_new = np.linalg.solve(LM, rhs_vec)
            h_next = h + dt * ((1.0 - theta) * v + theta * v_new)
            if np.max(np.abs(h_next - h_new)) < picard_tol * max(1.0, np.max(h)):
                h_new = h_next
                break
            h_new = h_next

        h, v = h_new, v_new
        AB = build_lubrication_banded(h, rHalf, cw, dr, FVM)
        p = solve_banded((1, 1), AB, v, check_finite=False)
        t += dt
        step += 1
        if np.min(h) < h_0 * 1e-3:
            break

    h_hist.append(h.copy()); p_hist.append(p.copy()); t_hist.append(t); dt_hist.append(dt)
    return dict(r=r, h=np.array(h_hist), p=np.array(p_hist), t=np.array(t_hist), dt=np.array(dt_hist))


# =============================================================================
# 5. Solver 4: Fourier-Bessel Spectral DtN + RK4 (Zero Elliptic Integrals)
# =============================================================================

def run_fourier_bessel_rk4(St, h_0, U=1.0, Nr=240, n_modes=120, r_max=1.0,
                           t_end=None, dt_max=None, save_every=4):
    r, rHalf, cw, dr = make_mesh(Nr, r_max, uniform=True)
    FVM = St / (12.0 * r * cw)
    D_fb = build_fourier_bessel_dtn(r, rHalf, cw, n_modes=n_modes)

    def rhs(h_, v_):
        AB = build_lubrication_banded(h_, rHalf, cw, dr, FVM)
        p_ = solve_banded((1, 1), AB, v_, check_finite=False)
        return p_, D_fb @ p_

    return _integrate_rk4_1d(r, cw, St, h_0, U, rhs, t_end, dt_max, save_every)


# =============================================================================
# 6. Solver 5: 2D Cartesian DCT/FFT + RK4 (Zero Cylindrical Coords/Elliptics)
# =============================================================================

def run_2d_cartesian_dct_rk4(St, h_0, U=1.0, N=48, L=0.52, t_end=None, dt_max=None, save_every=4):
    """
    Solves the full 3D axisymmetric problem on the quarter-plane [0, L]^2 using
    2D DCT-II (a_hat = |k| * p_hat). Default L=0.52 corresponds to r_HP ~ 10.
    """
    dx = L / N
    x = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    R2 = X**2 + Y**2

    k1 = np.pi * np.arange(N) / L
    KX, KY = np.meshgrid(k1, k1, indexing='ij')
    Kmag = np.sqrt(KX**2 + KY**2)
    idx = np.arange(N * N).reshape(N, N)

    def rhs_2d(h_, v_):
        rows, cols, vals = [], [], []
        diag = np.zeros((N, N))

        hx = 0.5 * (h_[:-1, :] + h_[1:, :])
        cx = St * (hx**3) / (12.0 * dx**2)
        rows.extend([idx[:-1, :].ravel(), idx[1:, :].ravel()])
        cols.extend([idx[1:, :].ravel(), idx[:-1, :].ravel()])
        vals.extend([cx.ravel(), cx.ravel()])
        diag[:-1, :] -= cx; diag[1:, :] -= cx
        hR = 1.5 * h_[-1, :] - 0.5 * h_[-2, :]
        diag[-1, :] -= 2.0 * St * (hR**3) / (12.0 * dx**2)

        hy = 0.5 * (h_[:, :-1] + h_[:, 1:])
        cy = St * (hy**3) / (12.0 * dx**2)
        rows.extend([idx[:, :-1].ravel(), idx[:, 1:].ravel()])
        cols.extend([idx[:, 1:].ravel(), idx[:, :-1].ravel()])
        vals.extend([cy.ravel(), cy.ravel()])
        diag[:, :-1] -= cy; diag[:, 1:] -= cy
        hT = 1.5 * h_[:, -1] - 0.5 * h_[:, -2]
        diag[:, -1] -= 2.0 * St * (hT**3) / (12.0 * dx**2)

        rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(diag.ravel())
        A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(N * N, N * N))
        p_ = spla.spsolve(A, v_.ravel()).reshape(N, N)
        a_ = idctn(Kmag * dctn(p_, type=2, norm='ortho'), type=2, norm='ortho')
        return p_, a_

    if t_end is None:
        t_end = 1.75 * (h_0 / U)  # ~t_HP = 6
    if dt_max is None:
        dt_max = (h_0 / U) / 400.0

    h = h_0 + 0.5 * R2
    v = np.full((N, N), -U)
    t = 0.0
    h_hist, p_hist, t_hist, dt_hist = [], [], [], []
    step = 0
    while t < t_end - 1e-12:
        p, k1_v = rhs_2d(h, v); k1_h = v
        h_min = np.min(h)
        w_half = max(dx, np.sum(h[:, 0] <= 2.0 * h_min) * dx)
        lam_est = 12.0 * w_half / (St * h_min**3)
        dt_h = 0.02 * np.min(np.where(v < 0, h / np.maximum(-v, 1e-300), np.inf))
        dt = min(dt_max, dt_h, 1.0 / max(lam_est, 1e-12), t_end - t)

        if step % save_every == 0:
            h_hist.append(h[:, 0].copy()); p_hist.append(p[:, 0].copy())
            t_hist.append(t); dt_hist.append(dt)

        _, k2_v = rhs_2d(h + 0.5*dt*k1_h, v + 0.5*dt*k1_v); k2_h = v + 0.5*dt*k1_v
        _, k3_v = rhs_2d(h + 0.5*dt*k2_h, v + 0.5*dt*k2_v); k3_h = v + 0.5*dt*k2_v
        _, k4_v = rhs_2d(h + dt*k3_h, v + dt*k3_v);         k4_h = v + dt*k3_v
        h += (dt / 6.0) * (k1_h + 2*k2_h + 2*k3_h + k4_h)
        v += (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        t += dt; step += 1
        if np.min(h) < h_0 * 1e-3:
            break

    p, _ = rhs_2d(h, v)
    h_hist.append(h[:, 0].copy()); p_hist.append(p[:, 0].copy()); t_hist.append(t); dt_hist.append(dt)
    return dict(r=x, h=np.array(h_hist), p=np.array(p_hist), t=np.array(t_hist), dt=np.array(dt_hist))


# =============================================================================
# Shared 1D RK4 Driver
# =============================================================================

def _integrate_rk4_1d(r, cw, St, h_0, U, rhs_fn, t_end, dt_max, save_every):
    if t_end is None:
        t_end = 1.75 * (h_0 / U)  # t_HP = 6.0 when h_0 corresponds to t_HP = -8
    if dt_max is None:
        dt_max = (h_0 / U) / 800.0  # ~0.01 in HP time units

    h = h_0 + 0.5 * r**2
    v = np.full(len(r), -U)
    t = 0.0
    h_hist, p_hist, t_hist, dt_hist = [], [], [], []
    step = 0
    while t < t_end - 1e-12:
        p, k1_v = rhs_fn(h, v); k1_h = v
        dt = min(stiff_dt(h, v, cw, St, dt_max), t_end - t)

        if step % save_every == 0:
            h_hist.append(h.copy()); p_hist.append(p.copy())
            t_hist.append(t); dt_hist.append(dt)

        _, k2_v = rhs_fn(h + 0.5*dt*k1_h, v + 0.5*dt*k1_v); k2_h = v + 0.5*dt*k1_v
        _, k3_v = rhs_fn(h + 0.5*dt*k2_h, v + 0.5*dt*k2_v); k3_h = v + 0.5*dt*k2_v
        _, k4_v = rhs_fn(h + dt*k3_h, v + dt*k3_v);         k4_h = v + dt*k3_v
        h += (dt / 6.0) * (k1_h + 2*k2_h + 2*k3_h + k4_h)
        v += (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        t += dt; step += 1
        if np.min(h) < h_0 * 1e-3:
            break

    p, _ = rhs_fn(h, v)
    h_hist.append(h.copy()); p_hist.append(p.copy()); t_hist.append(t); dt_hist.append(dt)
    return dict(r=r, h=np.array(h_hist), p=np.array(p_hist), t=np.array(t_hist), dt=np.array(dt_hist))



from scipy.special import ellipk, ellipe

def test_kernel_z_to_0(r, rHalf, CellWidth, dr, K_P, K_S, R0=0.148, a=0.013):
    """Tests DtN as z -> 0+ on a sharp off-axis ring peak (R0 ~ r_hp=2.82, a ~ 0.25)."""
    def P_ring(r_arr, z):
        ApB = (r_arr + R0)**2 + (z + a)**2
        m = 4.0 * r_arr * R0 / ApB
        return (2.0 / np.pi) * ellipk(m) / np.sqrt(ApB)

    ApB = (r + R0)**2 + a**2
    AmB = (r - R0)**2 + a**2
    m = 4.0 * r * R0 / ApB
    p_test = (2.0 * a / np.pi) * ellipe(m) / (AmB * np.sqrt(ApB))

    # 4th-order exact d^2 P_ring / dz^2 at z = 0^+
    eps = 1e-6
    dtn_exact = (-P_ring(r, 2*eps) + 16*P_ring(r, eps) - 30*P_ring(r, 0.0)
                 + 16*P_ring(r, -eps) - P_ring(r, -2*eps)) / (12 * eps**2)

    dpdr = np.zeros(len(r) + 1)
    dpdr[1:-1] = np.diff(p_test) / dr
    dpdr[-1] = -2.0 * p_test[-1] / CellWidth[-1]
    f = rHalf * dpdr
    dtn_num = K_P @ ((f[:-1] - f[1:]) / CellWidth) + 2.0 * (K_S @ ((dpdr[:-1] - dpdr[1:]) / CellWidth))

    mask = r < 0.5
    rel_err = np.max(np.abs(dtn_num[mask] - dtn_exact[mask])) / np.max(np.abs(dtn_exact[mask]))
    print(f"Ring-peak z->0+ max relative error (r < 0.5): {rel_err:.3e}")