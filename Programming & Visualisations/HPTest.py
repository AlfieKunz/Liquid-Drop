import numpy as np
from scipy.special import ellipkm1
from scipy.integrate import quad
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
from HPFig3ExtractedData import HPFig3aResults, HPFig3bResults

# ==================== H&P ABLATION TOGGLES ====================
# Try these presets:
#   Clean physics (matches your current sim): R_MAX_HP = 19.08, USE_HP_TIME_SCHEME = False, HP_A2_TRUNC_L = None
#   Only (1) domain truncation:               R_MAX_HP = 6.0,   USE_HP_TIME_SCHEME = False, HP_A2_TRUNC_L = None
#   (1) + (2) domain + H&P time scheme:       R_MAX_HP = 6.0,   USE_HP_TIME_SCHEME = True,  HP_A2_TRUNC_L = None
#   (2) + (4) closest match to H&P Fig. 3:    R_MAX_HP = 13.0,  USE_HP_TIME_SCHEME = True,  HP_A2_TRUNC_L = 13.0
#   (1) + (2) + (4) literal L=6 Eq. (A6):     R_MAX_HP = 6.0,   USE_HP_TIME_SCHEME = True,  HP_A2_TRUNC_L = 6.0

Nr                 = 256 * 2      # Fast (~1.5s) and plenty accurate for this test (use 128 to match H&P's I=128)
R_MAX_HP           = 6.0      # <<< H&P MOD (1): domain radius in H&P units (your full code has ~19.08)
USE_HP_TIME_SCHEME = True     # <<< H&P MOD (2): Eq. (A5)-(A6) iterative backward-diff at dt_hp = 0.02
HP_A2_TRUNC_L      = 13.0     # <<< H&P MOD (4): set to e.g. 6.0 or 13.0 to include Eq. (A2) box bias, or None
# ==============================================================

# Physical / Stokes scaling (identical to your code)
rho, R, U, mu_g = 1e3, 5e-4, 1.0, 1.8e-5
St  = rho * R * U / mu_g
L_h = 2**(1/3) * St**(-2/3)
L_r = 2**(2/3) * St**(-1/3)
L_t = L_h
L_p = L_r / L_h
h_0 = 8.0 * L_h  # = 0.0109892567285 (matches H&P t = -8)

# Uniform grid on [0, rMax]  <<< H&P MOD (1): replace 1.0 with R_MAX_HP * L_r
rMax = R_MAX_HP * L_r
rHalf = np.linspace(0.0, rMax, Nr + 1)
CellWidth = np.diff(rHalf)
r = rHalf[:-1] + CellWidth / 2.0
dr = np.diff(r)
FVMMultiplier = St / (12.0 * r * CellWidth)

# --- Your exact Kernel & ConstructKernel (fixed 2.0*ri syntax & removed dead returns) ---
twopi = 2.0 / np.pi
def Kernel(rp, ri):
    rSum = ri + rp
    k1 = (ri - rp) / rSum
    return twopi * ellipkm1(k1 * k1) / rSum

def KernelSym(s, ri):
    r1Sum, r2Sum = 2.0 * ri + s, 2.0 * ri - s
    k11, k12 = s / r1Sum, s / r2Sum
    return twopi * ((ellipkm1(k11 * k11) / r1Sum) + (ellipkm1(k12 * k12) / r2Sum))

def KSSym(s, ri):
    r1Sum, r2Sum = 2.0 * ri + s, 2.0 * ri - s
    k11, k12 = s / r1Sum, s / r2Sum
    return twopi * s * ((ellipkm1(k11 * k11) / r1Sum) - (ellipkm1(k12 * k12) / r2Sum))

def ConstructKernel():
    K_P = np.zeros((Nr, Nr))
    K_S = np.zeros((Nr, Nr))
    Points, Weights = np.polynomial.legendre.leggauss(20)
    HalfSpacing = CellWidth / 2.0
    r_j = r[:, None]
    KernelEvalPoints = np.outer(HalfSpacing, Points) + r_j
    for i in range(Nr):
        KPoints = Kernel(KernelEvalPoints, r[i])
        KSPoints = (KernelEvalPoints - r_j) * KPoints
        K_P[i, :] = HalfSpacing * (KPoints @ Weights)
        K_S[i, :] = HalfSpacing * (KSPoints @ Weights)
        K_P[i, i], _ = quad(KernelSym, 0, HalfSpacing[i], args=(r[i],), epsabs=0, epsrel=1e-13)
        K_S[i, i], _ = quad(KSSym,     0, HalfSpacing[i], args=(r[i],), epsabs=0, epsrel=1e-13)
    return K_P, K_S

K_P, K_S = ConstructKernel()

# <<< H&P MOD (4): Analytical truncated Eq. (A2) weight W_x(r; L) on [-L, L]^2
if HP_A2_TRUNC_L is not None:
    L_phys = HP_A2_TRUNC_L * L_r
    assert L_phys >= rMax, "HP_A2_TRUNC_L must be >= R_MAX_HP"
    Wx = (1.0 / np.pi) * (np.arcsinh(L_phys / (L_phys - r)) - np.arcsinh(L_phys / (L_phys + r)))
else:
    Wx = None

# --- Your exact Lubrication Matrix (parameterised by h_arr so H&P iterations can call it) ---
def solve_lubrication(h_arr, dhdt_arr):
    h_half = (h_arr[:-1] * CellWidth[1:] + h_arr[1:] * CellWidth[:-1]) / (CellWidth[:-1] + CellWidth[1:])
    FluxCoeff = (h_half**3 * rHalf[1:-1]) / dr
    AB = np.zeros((3, Nr))
    AB[0, 1:]   = FluxCoeff * FVMMultiplier[:-1]
    AB[1, 0]    = -FluxCoeff[0] * FVMMultiplier[0]
    AB[1, 1:-1] = -(FluxCoeff[:-1] + FluxCoeff[1:]) * FVMMultiplier[1:-1]
    hRightEdge  = 2.0 * h_arr[-1] - h_half[-1]
    AB[1, -1]   = -(FluxCoeff[-1] + 2.0 * (hRightEdge**3 * rHalf[-1] / CellWidth[-1])) * FVMMultiplier[-1]
    AB[2, :-1]  = FluxCoeff * FVMMultiplier[1:]
    return solve_banded((1, 1), AB, dhdt_arr, check_finite=False)

def compute_d2hdt2(p_arr):
    dpdr = np.zeros(Nr + 1)
    dpdr[1:-1] = np.diff(p_arr) / dr
    dpdr[-1]   = -2.0 * p_arr[-1] / CellWidth[-1]
    f = rHalf * dpdr
    fPrime = (f[:-1] - f[1:]) / CellWidth
    gPrime = (dpdr[:-1] - dpdr[1:]) / CellWidth
    acc = (K_P @ fPrime) + 2.0 * (K_S @ gPrime)
    if Wx is not None:  # <<< H&P MOD (4)
        dpdr_center = 0.5 * (dpdr[:-1] + dpdr[1:])
        acc -= dpdr_center * Wx
    return acc

# --- Time integration from t_hp = -8 to +6 ---
dt_hp = 0.02 if USE_HP_TIME_SCHEME else 0.005
dt = dt_hp * L_t
n_steps = int(round(14.0 / dt_hp))
save_times_hp = np.arange(-6, 7, 1)
snapshots = {}

h = h_0 + 0.5 * r**2
t = 0.0

if not USE_HP_TIME_SCHEME:
    # Your original symplectic Euler loop
    dhdt = np.full(Nr, -U)
    for n in range(n_steps + 1):
        t_hp = -8.0 + t / L_t
        p = solve_lubrication(h, dhdt)
        for ts in save_times_hp:
            if abs(t_hp - ts) < 0.5 * dt_hp and int(ts) not in snapshots:
                snapshots[int(ts)] = (h / L_h, p / L_p)
        if n == n_steps:
            break
        d2hdt2 = compute_d2hdt2(p)
        dhdt += d2hdt2 * dt
        h    += dhdt * dt
        t    += dt
else:
    # <<< H&P MOD (2): H&P Appendix Eq. (A5)-(A8) iterative scheme
    h_prev = (h_0 + U * dt) + 0.5 * r**2   # undisturbed drop one step earlier
    p_curr = np.zeros(Nr)                  # H&P starts with p = 0 at t = -8
    for n in range(n_steps):
        t_next = (n + 1) * dt
        h_star = h.copy()
        h_star[-1] = (h_0 - U * t_next) + 0.5 * r[-1]**2  # updated far-field BC
        for it in range(500):
            dhdt_est = (h_star - h) / dt                  # backward difference
            p_star   = solve_lubrication(h_star, dhdt_est)
            p_avg    = 0.5 * (p_star + p_curr)            # Eq. (A5)
            h_new    = 2.0 * h - h_prev + (dt**2) * compute_d2hdt2(p_avg)  # Eq. (A6)
            h_new[-1] = (h_0 - U * t_next) + 0.5 * r[-1]**2
            rms = np.sqrt(np.mean(((h_new - h_star) / L_h)**2))
            h_star = 0.5 * h_new + 0.5 * h_star if it > 15 else h_new
            if rms < 1e-8:
                h_star = h_new
                break
        h_prev, h, p_curr = h, h_star, p_star
        t = t_next
        t_hp = -8.0 + t / L_t
        for ts in save_times_hp:
            if abs(t_hp - ts) < 0.5 * dt_hp and int(ts) not in snapshots:
                snapshots[int(ts)] = (h / L_h, p_curr / L_p)






# --- Summary & Quick Plot in H&P dimensionless units ---
r_hp = r / L_r
h6, p6 = snapshots[6]
i6 = np.argmax(p6)
print(f"At t=6 | r_peak = {r_hp[i6]:.3f}, p_max = {p6[i6]:.4f}, h_min = {h6.min():.4f}, p(0) = {p6[0]:.4f}")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
Styles = {-6:(1.5,(0,(4,1.6))), -5:(0.7,(0,(5,4))), -4:(0.8,(0,(1,2))), -3:(0.7,(0,(11,3,1.5,3))),
         -2:(0.7,(0,(7,4,1.5,4))), -1:(0.7,(0,(4,7))), 0:(1.5,'-'), 1:(0.7,(0,(10,3,2,3))),
          2:(0.7,(0,(5,3))), 3:(0.7,'-'), 4:(0.7,(0,(9,3))), 5:(0.7,(0,(4,4))), 6:(1.2,(0,(1,1)))}
cmap = plt.cm.viridis(np.linspace(0, 0.95, len(save_times_hp)))
for c, ts in zip(cmap, save_times_hp):
    hh, pp = snapshots[int(ts)]
    r_ref_f, f_ref = HPFig3aResults[ts]
    r_ref_p, p_ref = HPFig3bResults[ts]
    lw,ls=Styles[ts]
    ax1.plot(r_ref_f, f_ref, color=c, lw=lw, ls=ls, solid_capstyle='butt')
    ax2.plot(r_ref_p, p_ref, color=c, lw=lw, ls=ls, solid_capstyle='butt')
    
    ax1.plot(r_hp, hh, color=c, lw=1.6)
    ax2.plot(r_hp, pp, color=c, lw=1.6)
ax1.set_ylim(0, 10); ax1.set_ylabel("Free-surface height $h$")
ax2.set_ylim(0, 0.5); ax2.set_xlim(0, 5)
ax2.set_ylabel("Pressure $p$"); ax2.set_xlabel("Radius $r$")
plt.tight_layout()
plt.show()