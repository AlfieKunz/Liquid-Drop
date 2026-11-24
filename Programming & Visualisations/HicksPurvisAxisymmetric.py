import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded
from scipy.special import ellipkm1

# Note: use numba-scipy for scipy function! :)

twopi = 2 / np.pi
def Kernel(r, ri):
    rSum = ri + r
    k1 = (ri - r) / rSum # Do I do the min / max formula? NO!!!!
    return twopi * ellipkm1(k1 * k1) / rSum
def KS(r, ri, rj):
    return (r - rj) * Kernel(r, ri)
def ConstructKernel():
    #TODO: Saw something about writing s = |r' - r_i| to make the K_S integral a lot easier... chuck s * (Kernel(ri + s, ri) - Kernel(ri - s, ri)) into quad, from 0 to delta/2
    K_P = np.zeros((Nr, Nr))
    K_S = np.zeros((Nr, Nr))
    t, w = np.polynomial.legendre.leggauss(20)
    HalfSpacing = CellWidth / 2
    
    # Use legendre leggauss for all elements.
    print(f"Precomputing Kernel Data: Legengre Leggauss", end="\r", flush=True)
    r_i = r[:, None, None] 
    r_j = r[None, :, None]
    KernelEvalPoints = HalfSpacing[None, :, None] * t[None, None, :] + r_j
    KPoints = Kernel(KernelEvalPoints, r_i)
    KSPoints = (KernelEvalPoints - r_j) * KPoints
    K_P = HalfSpacing[None, :] * np.sum(KPoints * w, axis=-1)
    K_S = HalfSpacing[None, :] * np.sum(KSPoints * w, axis=-1)
    
    # Fill in the i = j areas to deal with the Logarithmic Singularity
    for i in range(Nr):
        print(f"Precomputing Kernel Data: Scipy Quad Row {i+1} / {Nr}...", end="\r", flush=True)
        K_P[i,i], _ = quad(Kernel, rHalf[i], rHalf[i+1], args=(r[i],), points=[r[i]], epsabs=0, epsrel=1e-12)
        K_S[i,i], _ = quad(KS, rHalf[i], rHalf[i+1], args=(r[i], r[i]), points=[r[i]], epsabs=1e-16, epsrel=1e-12)
    return K_P, K_S


def ConstructBandedMatrix():
    AB = np.zeros((3, Nr))
    AB[0, 1:] = [InitMatrix(j - 1, j) for j in range(1, Nr)]
    AB[1, :] = [InitMatrix(i, i) for i in range(Nr)]
    AB[2, :-1] = [InitMatrix(j + 1, j) for j in range(Nr - 1)]
    return AB
def InitMatrix(i, j):
    MatrixElement = 0.0
    match i - j:
        case -1:
            MatrixElement = FluxCoefficientMinusHalf(i + 1)
        case 0:
            MatrixElement -= 0.0 if i == 0 else FluxCoefficientMinusHalf(i)
            # Right face (face i + 1): p = 0 at r = RMax (half a cell away -> factor of 2).
            if i == Nr - 1:
                hN = 1.5 * h[-1] - 0.5 * h[-2]
                MatrixElement -= 2.0 * (hN**3 * rHalf[Nr])
            else:
                MatrixElement -= FluxCoefficientMinusHalf(i + 1)
        case 1:
            MatrixElement = FluxCoefficientMinusHalf(i)
    return St * MatrixElement / (12 * r[i] * CellWidth[i] * CellWidth[i])
def FluxCoefficientMinusHalf(i):
    hMinusHalf = 0.5 * (h[i - 1] + h[i]) # 0.5 * (h[i - 1]**3 + h[i]**3)???
    return hMinusHalf**3 * rHalf[i]



# Sets up water-like configuration into a non-dimensionless constant. Afer this, we can remove all dimensions from further points.
rho = 1e3
R = 5e-4
U = 1.0
mu_g = 1.8e-5
m = 1.0 # rho * np.pi * R**3 * 4/3

ScalingFactor = 1.0
St = ScalingFactor * rho * R * U / mu_g
print(f"Stokes Number: {St:.2e}")


ZoomFactor = 1/4
RMax = 1.0
Nr = 2000

rHalf = np.linspace(0, RMax, Nr + 1) # Values of r, defined at the edges (eg: r_{i - 1/2})
CellWidth = np.diff(rHalf) # The width of each cell.
r = rHalf[:-1] + CellWidth / 2 # Size Nr, the cell centers
dr = np.diff(r) # The distance from the centre of each cell to the next.
dpdr = np.zeros(Nr + 1) # dp/dr, held at cell edges.

h_0 = 2e-2 # h_0 << R for weak parabolic effect.
h = h_0 + 0.5 * r**2
dhdt = np.full(Nr, -1.0)
d2hdt2 = np.zeros(Nr)

fPrime = np.zeros(Nr)
gPrime = np.zeros(Nr)

NtMax = 2500
dt0 = (3.0 * h_0 / 1.0) / NtMax  # Time for a rigid drop to descent 3x its height.
GlobalSpeedupMultiplier = 5

hResults = []
pResults = []
                
st = time.time()
K_P, K_S = ConstructKernel()          
ft = time.time()
print("\n" + f"Precomputed Kernel Data in {ft - st:.2e}s!")


st = time.time()
for n in range(NtMax):
    print(f"Running Time Iteration {n+1} / {NtMax}...", end="\r", flush=True)
    
    # Lubrication Method.
    AB = ConstructBandedMatrix()
    p = solve_banded((1, 1), AB, dhdt)

    # Calculates dp/dr (note: why can't we just embed this inside the matrix construction? Van der Walls??).
    # Boundary conditions (as in InitMatrix: no-flux at r = 0, p(r=N) = 0).
    dpdr[1:-1] = np.diff(p) / dr
    dpdr[0] = 0.0 
    dpdr[-1] = -2 * p[-1] / dr[-1]
    
    # Plugs into Boundary Integral.
    f = rHalf * dpdr
    fPrime = (f[:-1] - f[1:]) / CellWidth
    gPrime = (dpdr[:-1] - dpdr[1:]) / CellWidth
    d2hdt2 = (np.dot(K_P, fPrime) + 2 * np.dot(K_S, gPrime))
    
    # # a = v/t. If the drop begins to accelerate
    # MaxAccel = np.max(np.abs(d2hdt2))
    # if MaxAccel > 1e-12:
    #     Acceldt = np.max(np.abs(dhdt)) / MaxAccel
    #     dt = min(dt, Acceldt * 0.1)
        
    dt_h = 0.01 * np.min(np.where(dhdt < 0, h / np.maximum(-dhdt, 1e-300), np.inf))
    MaxAccel = np.max(np.abs(d2hdt2))
    dt_u = 0.01 * np.max(np.abs(dhdt)) / MaxAccel if MaxAccel > 1e-12 else np.inf
    dt = min(dt0, dt_h, dt_u)
    
    dhdt += d2hdt2 * dt
    h += dhdt * dt
    
    # Saves results, so that they can be outputted. Note that we make everything symmetric also! :)
    if (n % GlobalSpeedupMultiplier == 0):
        hResults.append(np.concatenate((h[::-1], h)))
        pResults.append(np.concatenate((p[::-1], p)))
    
    if np.min(h) < h_0 * 0.005:
        print("\n" + "Breaking Early Due to Droplet Touching Solid.")
        break
    
ft = time.time()

print("\n" + f"Simulation Complete in {ft - st:.2e}s!")
print("Plotting...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios': [1.3, 1]})
r = np.concatenate((-r[::-1], r))

# The problem is 1D - stretch all x pressure data in z
IMPlot = ax1.imshow(pResults[0][np.newaxis, :], extent=[r[0], r[-1], 0, np.max(hResults)], aspect='auto', origin='lower', cmap='plasma', vmin=0, vmax=np.max(pResults), interpolation='bilinear')

# Clip out all data above the drop - it is not important!
DropletXVertices = np.concatenate(([r[0]], r, [r[-1]]))
DropletYVertices = np.concatenate(([np.max(hResults)], hResults[0], [np.max(hResults)]))
DropletMesh = ax1.fill(DropletXVertices, DropletYVertices, color='paleturquoise', zorder=2)[0]

DropletBoundary, = ax1.plot(r, hResults[0], color='dodgerblue', lw=2.5, zorder=3)
PressureMap, = ax2.plot(r, pResults[0], color='crimson', lw=2.5)

ax1.set_xlim(r[0] * ZoomFactor, r[-1] * ZoomFactor)
ax1.set_xlabel(r'Position ($x/R$)')
VisibleRegion = np.abs(r) <= RMax * ZoomFactor
ax1.set_ylim(0, np.max(hResults[0][VisibleRegion]) * 0.8)
ax1.set_ylabel(r'Dimensionless Height $h/R$')
ax1.set_title('Interaction of a Droplet Colliding with a Thin Film')

ax1.set_xlim(r[0] * ZoomFactor, r[-1] * ZoomFactor)
ax2.set_ylim(0, np.max(pResults) * 1.1)
ax2.set_xlabel(r'Position ($x/R$)')
ax2.set_ylabel(r'Pressure ($p / (\rho U^2)$')
ax2.grid(True, alpha=0.3)

cbar = fig.colorbar(IMPlot, ax=ax1, pad=0.02)
cbar.set_label(r'Pressure ($p / (\rho U^2)$')
plt.tight_layout()

def DrawFrame(frame):
    h_f = hResults[frame]
    p_f = pResults[frame]
    IMPlot.set_data(p_f[np.newaxis, :])
    IMPlot.set_clim(0, np.max(p_f) * 1.2)
    DropletMesh.set_xy(np.column_stack((DropletXVertices, np.concatenate(([DropletYVertices[0]], h_f, [DropletYVertices[-1]])))))
    DropletBoundary.set_ydata(h_f)
    PressureMap.set_ydata(p_f)
    return IMPlot, DropletMesh, DropletBoundary, PressureMap

Animation = FuncAnimation(fig, DrawFrame, frames=len(hResults), interval=20, blit=True)
plt.show()