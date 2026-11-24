"""
Droplet impact on a thin gas film: gas lubrication (finite volume) + liquid boundary integral.
Axisymmetric and non-dimensional:  h/R,  r/R,  t/(R/U),  p/(rho U^2).

=====================  INDEXING CHEAT-SHEET  (read this first!)  =====================
 Nr cells.  Cell j sits between FACE j and FACE j+1.   Face k sits between CELL k-1 and CELL k.

   faces    k = 0 .. Nr     (Nr+1)  rHalf, hFace, g = dp/dr, f = r*dp/dr
   centres  j = 0 .. Nr-1   (Nr)    r, CellWidth, h, dhdt, p, accel
   centre-to-centre gaps    (Nr-1)  dr   <- lives on the INTERIOR faces 1..Nr-1 only

   * Differences of face arrays give cell arrays:   f[:-1] - f[1:]   ->  (Nr)
   * Differences of cell arrays give interior-face arrays:  np.diff(p)  ->  (Nr-1)
   * K_P[i, j], K_S[i, j]:  field point = CENTRE i,  source = CELL j   ->  (Nr, Nr)
   * Face 0 is the axis (r = 0: symmetry), face Nr is the outer edge (p = 0 there).
=======================================================================================
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad
from scipy.linalg import solve_banded
from scipy.special import ellipkm1

# ============================ Parameters ============================
rho, R, U, mu_g = 1e3, 5e-4, 1.0, 1.8e-5

# NOTE: was 1e-4.  That makes St ~ 2.8 (air ~1e4x too viscous): the drop is stopped within ~one time step
# (damping rate ~3e3) and nothing interesting happens.  1.0 is the real water drop (St ~ 2.8e4).
ScalingFactor = 1.0
St = ScalingFactor * rho * R * U / mu_g
print(f"Stokes Number: {St:.2e}")

# NOTE: was 0.25 and doubled as the plot zoom.  The pressure length scale is ell = sqrt(2 y) ~ 0.32 at
# t = 0, so a box of 0.25 with p(R_max) = 0 gives a 38% error in p.  Simulate big, zoom only the plot.
R_max = 1.0
PlotR = 0.25
Nr = 500

h_0 = 5e-2                       # initial gap
T_end = 3.0 * h_0
dt_max = T_end / 200
CFL_h = 0.05                     # an interface point may close at most 5% of its local gap per step
CFL_u = 0.05                     # velocities may change by at most 5% of max|u| per step (stiffness guard)
h_stop = 5e-3 * h_0              # declare touchdown when the neck gets this thin
SaveEvery = 1                    # store every n-th step (steps are adaptive; raise for very stiff runs)
RUN_SELF_TESTS = True

# ============================ Mesh ============================
class Grid:
    """Finite-volume mesh from ANY monotone array of face positions (so graded meshes just work)."""
    def __init__(self, rHalf):
        self.rHalf = np.asarray(rHalf, dtype=float)       # (Nr+1) faces
        self.Nr = len(self.rHalf) - 1
        self.CellWidth = np.diff(self.rHalf)                # (Nr)   cell volumes (width)
        self.r = 0.5 * (self.rHalf[:-1] + self.rHalf[1:])   # (Nr)   cell centres
        self.dr = np.diff(self.r)                           # (Nr-1) centre-to-centre distance


# ============================ Boundary-integral kernels ============================
twopi = 2.0 / np.pi
gl_x, gl_w = np.polynomial.legendre.leggauss(20)


def Kernel(rp, ri):
    """curly-K(ri, r') = 2 K(m) / (pi (ri + r')),  m = 4 ri r' / (ri + r')^2.
    ellipkm1(p) = K(1 - p); here p = 1 - m = ((ri - r')/(ri + r'))^2, formed directly (no cancellation)."""
    rSum = ri + rp
    k1 = (ri - rp) / rSum
    return twopi * ellipkm1(k1 * k1) / rSum


def BuildKernelMatrices(grid, epsrel=1e-12):
    """K_P[i,j] = int_{cell j} curlyK(r_i, r') dr',   K_S[i,j] = int_{cell j} (r' - r_j) curlyK(r_i, r') dr'."""
    Nr, r, rHalf, CellWidth = grid.Nr, grid.r, grid.rHalf, grid.CellWidth
    K_P = np.zeros((Nr, Nr))
    K_S = np.zeros((Nr, Nr))
    half = 0.5 * CellWidth
    nodes = r[:, None] + half[:, None] * gl_x[None, :]      # (Nr, 20): Gauss nodes in EVERY cell
    offs = nodes - r[:, None]                                # (r' - r_j) at those nodes
    for i in range(Nr):
        kern = Kernel(nodes, r[i])                           # (Nr, 20), vectorised over all source cells
        K_P[i, :] = half * (kern @ gl_w)
        K_S[i, :] = half * ((offs * kern) @ gl_w)
        # Diagonal (j = i): logarithmic singularity at the cell centre -> adaptive QUADPACK split there.
        K_P[i, i], _ = quad(Kernel, rHalf[i], rHalf[i + 1], args=(r[i],), points=[r[i]],
                            epsabs=0, epsrel=epsrel, limit=200)
        # (r' - r_i) is odd about the centre, so only the antisymmetric part of the kernel survives: the
        # two log spikes cancel analytically and the integrand is benign (no roundoff warnings).
        K_S[i, i], _ = quad(lambda s, ri=r[i]: s * (Kernel(ri + s, ri) - Kernel(ri - s, ri)),
                            0.0, half[i], epsabs=0, epsrel=epsrel, limit=200)
    return K_P, K_S


def BoundaryIntegralAccel(g, grid, K_P, K_S):
    """d2h/dt2 at the cell centres from g = dp/dr on the faces (report eq. BIAQuadFull).
    The 1/CellWidth belongs to the SOURCE cell j, so it is applied BEFORE the matrix product."""
    f = grid.rHalf * g                                       # (Nr+1)  f = r dp/dr on faces
    fPrime = (f[:-1] - f[1:]) / grid.CellWidth               # (Nr)    -f'(r_j)
    gPrime = (g[:-1] - g[1:]) / grid.CellWidth               # (Nr)    -g'(r_j)
    return K_P @ fPrime + 2.0 * (K_S @ gPrime)               # (Nr)


# ============================ Lubrication solve ============================
def FaceHeights(h, grid):
    """h linearly interpolated onto the faces (correct weights for non-uniform meshes)."""
    rHalf, r, dr = grid.rHalf, grid.r, grid.dr
    hFace = np.empty(grid.Nr + 1)
    w = (rHalf[1:-1] - r[:-1]) / dr                          # 0.5 on a uniform mesh
    hFace[1:-1] = h[:-1] + w * (h[1:] - h[:-1])
    hFace[0] = h[0]                                          # axis: multiplied by r = 0 anyway
    hFace[-1] = h[-1] + (rHalf[-1] - r[-1]) / dr[-1] * (h[-1] - h[-2])   # linear extrapolation
    return hFace


def SolvePressure(h, dhdt, St, grid):
    """Solve (1/r) d/dr( St/12 r h^3 dp/dr ) = dhdt  with  dp/dr = 0 at r = 0 and p = 0 at r = R_max.
    Returns p at centres (Nr) and g = dp/dr at faces (Nr+1)."""
    Nr, rHalf, r, dr, CellWidth = grid.Nr, grid.rHalf, grid.r, grid.dr, grid.CellWidth
    hFace = FaceHeights(h, grid)
    F = (St / 12.0) * rHalf * hFace**3                       # (Nr+1) flux coefficient: r_face * h_face^3

    C = np.zeros(Nr + 1)                                     # face "conductances"; C[0] = 0 -> no flux at axis
    C[1:-1] = F[1:-1] / dr                                   # interior face: gradient over centre-to-centre gap
    C[-1] = F[-1] / (0.5 * CellWidth[-1])                    # outer face: p = 0 sits HALF a cell from last centre
    # (the p_N = -p_{N-1} ghost cell is exactly this, folded into the diagonal - no ghost row needed)

    pref = 1.0 / (r * CellWidth)                             # 1 / (r_i * Delta_i): the FV divergence
    AB = np.zeros((3, Nr))
    AB[0, 1:] = pref[:-1] * C[1:-1]                          # A[i, i+1]
    AB[1, :] = -pref * (C[:-1] + C[1:])                      # A[i, i]
    AB[2, :-1] = pref[1:] * C[1:-1]                          # A[i+1, i]
    p = solve_banded((1, 1), AB, dhdt, check_finite=False)

    g = np.zeros(Nr + 1)                                     # dp/dr on faces, same stencil as the matrix
    g[1:-1] = np.diff(p) / dr
    g[-1] = -p[-1] / (0.5 * CellWidth[-1])                   # NOTE: factor 2 / half-cell, matching C[-1]
    return p, g


# ============================ Self tests ============================
def RunSelfTests():
    print("\n--- Self-test 1: boundary integral vs exact harmonic dipole (expect order -> 2) ---")
    a, prev = 1.0, None
    for N in (100, 200, 400):
        grid = Grid(np.linspace(0.0, 40.0, N + 1))
        K_P, K_S = BuildKernelMatrices(grid)
        g = -3 * a * grid.rHalf / (grid.rHalf**2 + a**2) ** 2.5            # p = a/(r^2+a^2)^{3/2}
        exact = (2 * a**2 - grid.r**2) / (grid.r**2 + a**2) ** 2.5          # -dp/dz at z = 0
        err = np.max(np.abs(BoundaryIntegralAccel(g, grid, K_P, K_S) - exact))
        print(f"   Nr = {N:4d}   max error = {err:.3e}   observed order = "
              f"{'  -' if prev is None else f'{np.log2(prev / err):.2f}'}")
        prev = err
    print("--- Self-test 2: lubrication solve vs exact rigid-parabola pressure 3/(St (y + r^2/2)^2) ---")
    y = 0.05
    grid = Grid(np.linspace(0.0, 4.0, 1001))
    p, _ = SolvePressure(y + 0.5 * grid.r**2, np.full(grid.Nr, -1.0), St, grid)
    exact = 3.0 / (St * (y + 0.5 * grid.r**2) ** 2)
    print(f"   max |p - exact| / p(0) = {np.max(np.abs(p - exact)) / exact[0]:.2e}   (domain-truncation limited)\n")


if RUN_SELF_TESTS:
    RunSelfTests()

# ============================ Setup ============================
grid = Grid(np.linspace(0.0, R_max, Nr + 1))   # <- swap in a graded array of faces here later
r, CellWidth = grid.r, grid.CellWidth

st = time.time()
K_P, K_S = BuildKernelMatrices(grid)
print(f"Precomputed Kernel Data in {time.time() - st:.2e}s!")

h = h_0 + 0.5 * r**2             # (Nr) initial parabola
dhdt = np.full(Nr, -1.0)         # (Nr) rigid downward translation

hResults, pResults = [], []

# ============================ Time loop ============================
t, n = 0.0, 0
st = time.time()
while t < T_end:
    # 1) gas film: pressure from the current interface velocity (h is never touched in here)
    p, g = SolvePressure(h, dhdt, St, grid)

    # 2) liquid: interface acceleration from that pressure via the boundary integral
    accel = BoundaryIntegralAccel(g, grid, K_P, K_S)

    # 3) adaptive step: stiffness grows like h^-3, so a fixed dt cannot survive dimple formation
    aMax = np.max(np.abs(accel))
    dt_h = CFL_h * np.min(np.where(dhdt < 0, h / np.maximum(-dhdt, 1e-300), np.inf))
    dt_u = CFL_u * np.max(np.abs(dhdt)) / aMax if aMax > 0 else np.inf
    dt = max(min(dt_max, dt_h, dt_u, T_end - t), 1e-12)

    if n % SaveEvery == 0:
        hResults.append(np.concatenate((h[::-1], h)))        # mirror -> (2 Nr)
        pResults.append(np.concatenate((p[::-1], p)))

    # 4) semi-implicit (symplectic) Euler: velocity first, then position
    dhdt += accel * dt
    h += dhdt * dt
    t += dt
    n += 1

    if n % 10 == 0:
        j = np.argmin(h)
        print(f"step {n:6d}  t = {t:.5f}  dt = {dt:.1e}  h_min = {h[j]:.3e} (r = {r[j]:.3f})  "
              f"p_max = {np.max(p):.3g}", end="\r", flush=True)
    if not np.all(np.isfinite(h)):
        print("\nNon-finite h - aborting."); break
    if np.min(h) < h_stop:
        j = np.argmin(h)
        print(f"\nTouchdown ring: h_neck = {h[j]:.2e} at r = {r[j]:.3f}, t = {t:.5f} "
              f"(no surface tension in this model, so this is a finite-time singularity).")
        break

print("\n" + f"Simulation Complete in {time.time() - st:.2e}s ({n} steps, {len(hResults)} frames)!")
print("Plotting...")

# ============================ Plotting ============================
rPlot = np.concatenate((-r[::-1], r))                        # (2 Nr)
hArr, pArr = np.array(hResults), np.array(pResults)
visible = np.abs(rPlot) <= PlotR
hTop = hArr[:, visible].max() * 1.05                         # y-range from the visible window only
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios': [1.3, 1]})

# The problem is 1D - stretch all x pressure data in z
IMPlot = ax1.imshow(pArr[0][np.newaxis, :], extent=[rPlot[0], rPlot[-1], 0, hTop], aspect='auto',
                    origin='lower', cmap='plasma', vmin=0, vmax=pArr.max(), interpolation='bilinear')

# Clip out all data above the drop - it is not important!
DropletXVertices = np.concatenate(([rPlot[0]], rPlot, [rPlot[-1]]))
DropletMesh = ax1.fill(DropletXVertices, np.concatenate(([hTop], hArr[0], [hTop])),
                       color='paleturquoise', zorder=2)[0]
DropletBoundary, = ax1.plot(rPlot, hArr[0], color='dodgerblue', lw=2.5, zorder=3)
PressureMap, = ax2.plot(rPlot, pArr[0], color='crimson', lw=2.5)

ax1.set_xlim(-PlotR, PlotR)
ax1.set_ylim(0, hTop)
ax1.set_ylabel(r'Dimensionless Height $h/R$')
ax1.set_title('Interaction of a Droplet Colliding with a Thin Film')
ax2.set_ylim(0, pArr[:, visible].max() * 1.1)
ax2.set_xlabel(r'Position ($x/R$)')
ax2.set_ylabel(r'Pressure ($p / (\rho U^2)$)')
ax2.grid(True, alpha=0.3)
cbar = fig.colorbar(IMPlot, ax=ax1, pad=0.02)
cbar.set_label(r'Pressure ($p / (\rho U^2)$)')
plt.tight_layout()


def DrawFrame(frame):
    h_f, p_f = hArr[frame], pArr[frame]
    IMPlot.set_data(p_f[np.newaxis, :])
    IMPlot.set_clim(0, np.max(p_f) * 1.2)
    DropletMesh.set_xy(np.column_stack((DropletXVertices, np.concatenate(([hTop], h_f, [hTop])))))
    DropletBoundary.set_ydata(h_f)
    PressureMap.set_ydata(p_f)
    return IMPlot, DropletMesh, DropletBoundary, PressureMap


Animation = FuncAnimation(fig, DrawFrame, frames=len(hArr), interval=20, blit=True)
plt.show()