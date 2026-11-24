import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter, LogLocator, StrMethodFormatter
from scipy.integrate import quad
from scipy.linalg import solve_banded
from scipy.special import ellipkm1
from tqdm import tqdm

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
    Points, Weights = np.polynomial.legendre.leggauss(20)
    HalfSpacing = CellWidth / 2
    # Precomputes all the evaluation points r = r', using Legendre Leggauss Points. This works by calculating the
    # outer product of HalfSpacing and Points (local delta evaluation points within each r_{j-1/2} -> r_{j+1/2} cell)
    # Then, we add the central r evaluation points to centre them ([:, None] adds to each row, reflecting HalfSpacing).
    r_j = r[:, None] # And r_i = r.
    KernelEvalPoints = np.outer(HalfSpacing, Points) + r_j    
    for i in tqdm(range(Nr), desc="Precomputing Kernel Data"):        
        # Use legendre leggauss for all elements, via the dot product to make things vectorised.
        KPoints = Kernel(KernelEvalPoints, r[i])
        KSPoints = (KernelEvalPoints - r_j) * KPoints
        K_P[i, :] = HalfSpacing * (KPoints @ Weights)
        K_S[i, :] = HalfSpacing * (KSPoints @ Weights)
    
        # Fill in the i = j areas to deal with the Logarithmic Singularity, using Scipy Quad.
        # Note that I use epsabs=1e-16 in the second of these as there is lots of symmetry around the pole.
        K_P[i,i], _ = quad(Kernel, rHalf[i], rHalf[i+1], args=(r[i],), points=[r[i]], epsabs=0, epsrel=1e-12)
        K_S[i,i], _ = quad(KS, rHalf[i], rHalf[i+1], args=(r[i], r[i]), points=[r[i]], epsabs=1e-16, epsrel=1e-12)
    return K_P, K_S


def ConstructBandedMatrix():
    AB = np.zeros((3, Nr))
    FluxCoefficientMap = FluxCoefficientMinusHalf(np.arange(1, Nr)) # Calculates the half-cell index matrix, from a matrix of indices.
    FVMMultiplier = St / (12.0 * r * CellWidth * CellWidth)
    
    AB[0, 1:] = FluxCoefficientMap * FVMMultiplier[:-1]
    # Boundary conditions: q_{-1/2} = 0 - no pressure gradient along the central line (symmetry).
    AB[1, 0] = -FluxCoefficientMap[0] * FVMMultiplier[0]
    AB[1, 1:-1] = -(FluxCoefficientMap[:-1] + FluxCoefficientMap[1:]) * FVMMultiplier[1:-1]
    # Boundary conditions: q_{n + 1/2} = 0. We then enforce p_N = -p_{N-1} s.t the pressure at the edge of the final cell is 0.
    # To do this, we must interpolate h_N to h_{N+1/2} = h[N] + 0.5*(h[N] - h[N-1]), or h[N] + (h[N] - h[N-1/2]). 
    hRightEdge = 2 * h[-1] - hIndexMinusHalf(-1)
    AB[1, -1] = -(FluxCoefficientMap[-1] + (2.0 * (hRightEdge**3 * rHalf[-1]))) * FVMMultiplier[-1]
    AB[2, :-1] = FluxCoefficientMap * FVMMultiplier[1:]
    # # Sanity Check: compare against old InitMatrix Function.
    # for i in range(1, Nr):
    #     if not np.isclose(AB[0, i], InitMatrix(i - 1, i)):
    #         print("Ahhhh")
    # for i in range(Nr - 1):
    #         if not np.isclose(AB[2, i], InitMatrix(i + 1, i)):
    #             print("Ahhhh")
    # for i in range(Nr):
    #     if not np.isclose(AB[1, i], InitMatrix(i, i)):
    #         print("Ahhhh")
    return AB

def hIndexMinusHalf(i):
    hMinusInterp = h[i - 1] + 0.5 * CellWidth[i - 1]
    hPlusInterp = h[i] - 0.5 * CellWidth[i]
    return 0.5 * (hMinusInterp + hPlusInterp)
def FluxCoefficientMinusHalf(i):
    # 0.5 * (h[i - 1]**3 + h[i]**3)???
    return hIndexMinusHalf(i)**3 * rHalf[i]



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
Nr = 500

rHalf = np.linspace(0, 1, Nr + 1) # Values of r, defined at the edges (eg: r_{i - 1/2}). Non-dimensionalisation make r from 0 to 1.
CellWidth = np.diff(rHalf) # The width of each cell.
r = rHalf[:-1] + CellWidth / 2 # Size Nr, the cell centers
dr = np.diff(r) # The distance from the centre of each cell to the next.
dpdr = np.zeros(Nr + 1) # dp/dr, held at cell edges.

h_0 = 5e-2 # h_0 << R for weak parabolic effect.
h = h_0 + 0.5 * r**2
dhdt = np.full(Nr, -1.0)
d2hdt2 = np.zeros(Nr)

fPrime = np.zeros(Nr)
gPrime = np.zeros(Nr)

NtMax = 2000
t = 0
dt0 = (3.0 * h_0 / 1.0) / (NtMax / 5)  # Time for a rigid drop to descent 3x its height. TODO: replace 5 with 1
GlobalSpeedupMultiplier = 4

hResults = []
pResults = []
tResults = []
dtResults = []
EXITCode = 0 # 0 for all good, 1 for exit early due to droplet touching solid.
                
st = time.time()
K_P, K_S = ConstructKernel()          
ft = time.time()
print(f"Precomputed Kernel Data in {ft - st:.2f}s!")


st = time.time()
for n in tqdm(range(NtMax), desc="Running Lubrication + Boundary Integral Time Iteration"):     
    # Lubrication Method.
    AB = ConstructBandedMatrix()
    p = solve_banded((1, 1), AB, dhdt, check_finite=False)

    # Calculates dp/dr (note: why can't we just embed this inside the matrix construction? Van der Walls??).
    # Boundary conditions (as in InitMatrix: no-flux at r = 0, p(r=N) = 0).
    dpdr[0] = 0.0 
    dpdr[1:-1] = np.diff(p) / dr
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
    
    # Saves results for the p[t = n] and h[t = n] update, so that they can be outputted. Note that we make everything symmetric also! :)
    if (n % GlobalSpeedupMultiplier == 0):
        hResults.append(np.concatenate((h[::-1], h)))
        pResults.append(np.concatenate((p[::-1], p)))
        dtResults.append(dt)
        tResults.append(t)
    
    # Calculates h[t = n+1]
    dhdt += d2hdt2 * dt
    h += dhdt * dt
    t += dt
    
    if np.min(h) < h_0 * 0.005:
        EXITCode = 1
        break
    
ft = time.time()
match EXITCode:
    case 0: print("Simulation Completed Normally.")
    case 1: print("Simulation Exited Early Due to Droplet Touching Solid.")
        
print(f"Simulation Completed in {ft - st:.2f}s! Simulation Time: {tResults[-1]:.4f}")
print(f"Predicted Pressure Asymptote at r = {(np.where(pResults[-1] == np.max(pResults[-1]))[-1][-1] - Nr) / Nr}")










print("Plotting...")
rPlot = np.concatenate((-r[::-1], r))
hResults, pResults, tResults, dtResults = np.array(hResults), np.array(pResults), np.array(tResults), np.array(dtResults)
hMax = hResults.max()
pMiddlePlot = np.array([result[len(result) // 2] for result in pResults])
pMaxPlot = np.array([np.max(result) for result in pResults])

VisibleRegion = np.abs(rPlot) <= ZoomFactor
# The rigid drop only exists until it hits the film. As the rigid film descends with velocity U = -1, this happens at t = h_0.
RigidTime = tResults[tResults < h_0]
Rigidp = lambda t, r: 3.0 / (St * (h_0 - t + 0.5 * r**2)**2)  #TODO: Analytic rigid-drop pressure: CHECK!!!!!
Box = lambda colour: dict(boxstyle='round,pad=0.3', fc='white', ec=colour, alpha=0.85)
fig, ax = plt.subplot_mosaic([['ax1', 'ax3'], ['ax1', 'axT'], ['ax2', 'ax4']], figsize=(12, 7.5), layout='constrained', gridspec_kw={'width_ratios': [1.35, 1], 'height_ratios': [1.3, 0.1, 1]})
ax1, ax2, ax3, ax4, axT = map(ax.get, ('ax1', 'ax2', 'ax3', 'ax4', 'axT'))
ax2.sharex(ax1); ax4.sharex(ax3)  # r shared down the left, t down the right


# Top-Left: Live Interface Simulation.
IMPlot = ax1.imshow(pResults[0][np.newaxis, :], extent=[rPlot[0], rPlot[-1], 0, hMax], aspect='auto', origin='lower',
                    cmap='plasma', norm=LogNorm(vmin=0.05, vmax=np.max(pResults)), interpolation='nearest')
DropletXVertices = np.concatenate(([rPlot[0]], rPlot, [rPlot[-1]]))
DropletMesh = ax1.fill(DropletXVertices, np.concatenate(([hMax], hResults[0], [hMax])), color='paleturquoise', zorder=2)[0]
DropletBoundary, = ax1.plot(rPlot, hResults[0], color='dodgerblue', lw=2.5, zorder=3)

cbar = fig.colorbar(IMPlot, ax=ax1, fraction=0.035, pad=0.01)
fig.colorbar(IMPlot, ax=ax2, fraction=0.035, pad=0.01).ax.set_visible(False) # Places an identical colourbar on the second axis, such that the two figures align.
# Makes the scale for the pressure heatmap logarithmic and global, so that we can visually see pressure updating.
cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
cbar.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))

ax1.set_xlim(-ZoomFactor, ZoomFactor)
ax1.set_ylim(0, np.max(hResults[0][VisibleRegion]) * 0.75)
ax1.set_xlabel(r'Position ($r/R$)')
ax1.set_ylabel(r'Dimensionless Height ($h/R$)')
ax1.set_title('Droplet Interface Evolution')



# Bottom-Left: Global Pressure Map (with analytic solution).
PressureMap, = ax2.plot(rPlot, pResults[0], color='darkorange', lw=2.5, label='Simulation')
RigidPressure, = ax2.plot(rPlot, Rigidp(tResults[0], rPlot), 'k--', lw=1.4, alpha=0.7, label='Rigid Interface')

ax2.set_ylim(0, np.max(pResults) * 1.05)
ax2.set_xlabel(r'Position ($r/R$)')
ax2.set_ylabel(r'Pressure ($p / \rho U^2$)')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=8)
ax2.set_title('Local Pressure Across the Interface')



# Top-right: Profile evolution, and comparison to analytic solution. simulation vs analytic rigid-drop pressure
ax3.axvline(h_0, color='grey', ls='-.', lw=1.5, label=r'Rigid Collapse ($t = h_0$)')
ax3.semilogy(RigidTime, Rigidp(RigidTime, r[0]), 'k--', lw=1.5, label=r'$p_\text{rigid}(0, t)$')
ax3.semilogy(tResults, pMiddlePlot, color='crimson', lw=2.2, label=r'$p(0, t)$')
ax3.semilogy(tResults, pMaxPlot, color='royalblue', lw=1.6, alpha=0.85, label=r'$\max p(r, t)$')
TRTracker = ax3.axvline(tResults[0], color='darkorange', ls='--', lw=1.5)
TRMarker, = ax3.plot(tResults[0], pMiddlePlot[0], 'o', color='crimson', ms=6, zorder=4)
TRLabel = ax3.text(0.04, 0.95, '', transform=ax3.transAxes, va='top', fontsize=8.5, bbox=Box('purple'))

ax3.set_xlim(tResults[0], tResults[-1])
ax3.set_xlabel(r'Time ($t U / R$)')
ax3.set_ylim(min(pMaxPlot) * 0.5, max(pMaxPlot) * 2.0)
ax3.set_ylabel(r'Pressure ($p / \rho U^2$)')
ax3.yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
ax3.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
ax3.set_title('Evolution of Pressure')
ax3.grid(True, which='both', alpha=0.3)
ax3.legend(loc='lower right', fontsize=8)
ax3.tick_params(labelbottom=True)



# Bottom-right: Evolution of the time-step.
ax4.semilogy(tResults, dtResults, color='teal', lw=2.0)
BRTracker = ax4.axvline(tResults[0], color='darkorange', ls='--', lw=1.5)
BRMarker, = ax4.plot(tResults[0], dtResults[0], 'o', color='teal', ms=6, zorder=4)
BRLabel = ax4.text(0.04, 0.08, '', transform=ax4.transAxes, fontsize=9, bbox=Box('teal'))
ax4.set_xlim(tResults[0], tResults[-1])
ax4.set_xlabel(r'Time ($t U / R$)')
ax4.set_ylim(min(dtResults) * 0.5, max(dtResults) * 2.0)
ax4.set_ylabel(r'Timestep ($\Delta t$)')
ax4.set_title(r'Evolution of Timestep $\Delta t$')
ax4.grid(True, which='both', alpha=0.3)



# Middle-right: linear time bar.
TimeBar = axT.barh(0.5, 0, height=1, color='darkorange')[0]
axT.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], facecolor='0.93')
Frames = range(0, len(hResults))



Artists = (IMPlot, DropletMesh, DropletBoundary, PressureMap, RigidPressure, BRTracker, BRMarker, BRLabel, TRTracker, TRMarker, TRLabel, TimeBar)
def DrawFrame(frame):
    h_f, p_f, t_f = hResults[frame], pResults[frame], tResults[frame]
    IMPlot.set_data(p_f[np.newaxis, :])
    DropletMesh.set_xy(np.column_stack((DropletXVertices, np.concatenate(([hMax], h_f, [hMax])))))
    DropletBoundary.set_ydata(h_f)
    PressureMap.set_ydata(p_f)

    TRLabel.set_text(f'Simulation p(r=0) = {pMiddlePlot[frame]:.4g}' + (f'\nRigid p(r=0) = {Rigidp(t_f, r[0]):.4g}' if t_f < h_0 else ''))
    RigidPressure.set_ydata(Rigidp(t_f, rPlot) if t_f < h_0 else np.full_like(rPlot, np.nan))
    TimeBar.set_width(frame / Frames[-1])

    BRLabel.set_text(f't = {t_f:.5f}\nΔt = {dtResults[frame]:.3e}')
    BRTracker.set_xdata([t_f, t_f])
    BRMarker.set_data([t_f], [dtResults[frame]])
    TRTracker.set_xdata([t_f, t_f])
    TRMarker.set_data([t_f], [pMiddlePlot[frame]])
    return Artists

Animation = FuncAnimation(fig, DrawFrame, frames=Frames, interval=20, blit=True)
# Suppresses some errors that appear during playback (exiting last frame early).
_Step = Animation._step
def QuietStep(*args):
    return _Step(*args) if fig.canvas.manager is not None else False
Animation.event_source.remove_callback(_Step)
Animation.event_source.add_callback(QuietStep)
plt.show()