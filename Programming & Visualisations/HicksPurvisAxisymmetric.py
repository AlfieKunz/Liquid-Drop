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
import datetime
from scipy.integrate import cumulative_trapezoid
from matplotlib import cm
from matplotlib.lines import Line2D

from HPFig3ExtractedData import HPFig3aResults, HPFig3bResults

# import warnings
# from scipy.integrate import IntegrationWarning
# warnings.filterwarnings("ignore", category=IntegrationWarning)

# Note: use numba-scipy for scipy function! :)

twopi = 2 / np.pi
def Kernel(r, ri):
    rSum = ri + r
    k1 = (ri - r) / rSum # Do I do the min / max formula? NO!!!!
    return twopi * ellipkm1(k1 * k1) / rSum
def KS(r, ri, rj):
    return (r - rj) * Kernel(r, ri)
def KernelSym(s, ri):
    r1Sum, r2Sum = 2.0*ri + s, 2.0*ri - s
    k11, k12 = s / r1Sum, s / r2Sum # k11 should be -..., but this gets squares so we're fine :)
    return twopi * ((ellipkm1(k11 * k11) / r1Sum) + (ellipkm1(k12 * k12) / r2Sum))
    return Kernel(ri + s, ri) + Kernel(ri - s, ri)
def KSSym(s, ri):
    r1Sum, r2Sum = 2.0*ri + s, 2.0*ri - s
    k11, k12 = s / r1Sum, s / r2Sum # k11 should be -..., but this gets squares so we're fine :)
    return twopi * s * ((ellipkm1(k11 * k11) / r1Sum) - (ellipkm1(k12 * k12) / r2Sum))
    return s * (Kernel(ri + s, ri) - Kernel(ri - s, ri))

def ConstructKernel():
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
        K_P[i,i], _ = quad(KernelSym, 0, HalfSpacing[i], args=(r[i],), epsabs=0, epsrel=1e-13)
        K_S[i,i], _ = quad(KSSym, 0, HalfSpacing[i], args=(r[i],), epsabs=0, epsrel=1e-13)
        # K_P[i,i], _ = quad(Kernel, rHalf[i], rHalf[i+1], args=(r[i],), points=[r[i]], epsabs=1e-14, epsrel=1e-12)
        # K_S[i,i], _ = quad(KS, rHalf[i], rHalf[i+1], args=(r[i], r[i]), points=[r[i]], epsabs=1e-14, epsrel=1e-12)
    return K_P, K_S


def ConstructBandedMatrix():
    AB = np.zeros((3, Nr))
    FluxCoefficientMap = FluxCoefficientMinusHalf(np.arange(1, Nr)) / dr # Calculates the half-cell index matrix, from a matrix of indices.
    
    AB[0, 1:] = FluxCoefficientMap * FVMMultiplier[:-1]
    # Boundary conditions: q_{-1/2} = 0 - no pressure gradient along the central line (symmetry).
    AB[1, 0] = -FluxCoefficientMap[0] * FVMMultiplier[0]
    AB[1, 1:-1] = -(FluxCoefficientMap[:-1] + FluxCoefficientMap[1:]) * FVMMultiplier[1:-1]
    # Boundary conditions: q_{n + 1/2} = 0. We then enforce p_N = -p_{N-1} s.t the pressure at the edge of the final cell is 0.
    # To do this, we must interpolate h_N to h_{N+1/2} = h[N] + 0.5*(h[N] - h[N-1]), or h[N] + (h[N] - h[N-1/2]). 
    hRightEdge = 2 * h[-1] - hIndexMinusHalf(-1)
    AB[1, -1] = -(FluxCoefficientMap[-1] + (2.0 * (hRightEdge**3 * rHalf[-1] / CellWidth[-1]))) * FVMMultiplier[-1]
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

def hIndexMinusHalf(i): # Uses a level design to interpolate h to the edges. This should all be CellWidth / 2 but they all cancel out :)
    hMinusInterp = h[i - 1] * CellWidth[i]
    hPlusInterp = h[i] * CellWidth[i - 1]
    return (hMinusInterp + hPlusInterp) / (CellWidth[i - 1] + CellWidth[i])
def FluxCoefficientMinusHalf(i):
    # 0.5 * (h[i - 1]**3 + h[i]**3)???
    return hIndexMinusHalf(i)**3 * rHalf[i]


def CellWidthWeightingFunction(r, r_0 = 0.18075432): #r_0 is the location of the pressure asymptote (where the droplet touches the surface).
    return np.exp(-2.0 * r) + 2.0 * np.exp(-(25.0 * (r - r_0))**2) #TODO: Replace 2.0 with 10.0 - 20.0 when we find out exactly where the asymptote is!!
    return 1


#! ~~~~~~~~~~~~~~~~~~~~ PROGRAM SETTINGS ~~~~~~~~~~~~~~~~~~~~
# (Relative) file path to retieve data from. Set to '' for full simulation mode.
SAVEFILE = 'SimulationResults27112026-232818,Nr=10000,Nt=10000'
SAVESIMULATION = True
PlottingMode = 0 # 0 for both, 1 for only interface evolution, 2 for only HP comparison.
GlobalSpeedupMultiplier = 5
ZoomFactor = 1/4

Nr = 10000
NtMax = 10000
#! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Sets up water-like configuration into a non-dimensionless constant. Afer this, we can remove all dimensions from further points.
rho = 1e3
R = 5e-4
U = 1.0
mu_g = 1.8e-5
mu_l = 1e-3
m = 1.0 # rho * np.pi * R**3 * 4/3

ScalingFactor = 1.0
St = ScalingFactor * rho * R * U / mu_g
print(f"Stokes Number: {St:.2e}")




# Use the equidistribution principle to distribute CellWidth, using CellWidthWeightingFunction(r) for r from 0 to 1.
# Get a really good mesh of the CellWidthWeightingFunction function, rho(r).
rFine = np.linspace(0, 1, (Nr * 100) + 1)
CellMeshrho = cumulative_trapezoid(CellWidthWeightingFunction(rFine), rFine, initial=0.0)
CellMeshrho /= CellMeshrho[-1] # Normalises the mesh density function.
rHalf = np.interp(np.linspace(0, 1, Nr + 1), CellMeshrho, rFine) 
CellWidth = np.diff(rHalf) # The width of each cell.

r = rHalf[:-1] + CellWidth / 2 # Size Nr, the cell centers
dr = np.diff(r) # The distance from the centre of each cell to the next.
dpdr = np.zeros(Nr + 1) # dp/dr, held at cell edges.
FVMMultiplier = St / (12.0 * r * CellWidth)

#TODO: changing h_0 has massive variance! :()
h_0 = 0.01098925673 # 4e-2 # h_0 << R for weak parabolic effect. Note: HP has t = -8, corresponding to h_0 = 0.01098925673
h = h_0 + 0.5 * r**2 #TODO: HP uses h_0 + r**2, I use h_0 + 0.5 * r**2?
dhdt = np.full(Nr, -U)
d2hdt2 = np.zeros(Nr)

fPrime = np.zeros(Nr)
gPrime = np.zeros(Nr)

t = 0
TouchdownTime = h_0 / U
dt0 = 3.0 * TouchdownTime / NtMax  # Time for a rigid drop to descent 3x its height.

hResults = []
pResults = []
tResults = []
dtResults = []
EXITCode = 0 # 0 for all good, 1 for exit early due to droplet touching solid.
     
if SAVEFILE == "":           
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
        dpdr[1:-1] = np.diff(p) / dr #* This only works if the cell size doesn't change much??
        dpdr[-1] = -2 * p[-1] / CellWidth[-1]
        
        # Plugs into Boundary Integral. #TODO: verify this :).
        f = rHalf * dpdr
        fPrime = (f[:-1] - f[1:]) / CellWidth
        gPrime = (dpdr[:-1] - dpdr[1:]) / CellWidth
        d2hdt2 = (np.dot(K_P, fPrime) + 2 * np.dot(K_S, gPrime))
        
        # # a = v/t. If the drop begins to accelerate
        # Max1ccel = np.max(np.abs(d2hdt2))
        # if Max1ccel > 1e-12:
        #     Acceldt = np.max(np.abs(dhdt)) / Max1ccel
        #     dt = min(dt, Acceldt * 0.1)
        
        dt_h = 0.01 * np.min(np.where(dhdt < 0, h / np.maximum(-dhdt, 1e-300), np.inf))
        Max1ccel = np.max(np.abs(d2hdt2))
        dt_u = 0.01 * np.max(np.abs(dhdt)) / Max1ccel if Max1ccel > 1e-12 else np.inf
        dt = min(dt0, dt_h, dt_u)
        
        # Saves results for the p[t = n] and h[t = n] update, so that they can be outputted. Note that we make everything symmetric also! :)
        if (n % GlobalSpeedupMultiplier == 0):
            hResults.append(np.concatenate((h[::-1], h)))
            pResults.append(np.concatenate((p[::-1], p)))
            dtResults.append(dt)
            tResults.append(t)
        
        # Calculates h[t = n+1]
        #TODO: yeah make this loads better pls :)
        dhdt += d2hdt2 * dt
        h += dhdt * dt
        t += dt
        
        if np.min(h) < h_0 * 1e-3:
            EXITCode = 1
            break
        
    ft = time.time()
    match EXITCode:
        case 0: print("Simulation Completed Normally.")
        case 1: print("Simulation Exited Early Due to Droplet Touching Solid.")
            
    print(f"Simulation Completed in {ft - st:.2f}s! Simulation Time: {tResults[-1]:.4f}")

    # Saves data to a new file.
    rPlot = np.concatenate((-r[::-1], r))
    hResults, pResults, tResults, dtResults = np.array(hResults), np.array(pResults), np.array(tResults), np.array(dtResults)
    if SAVESIMULATION:
        NewSaveFile = f'SimulationResults{datetime.datetime.now().strftime("%d%m%Y-%H%M%S")},Nr={Nr},Nt={NtMax}.npz'
        np.savez_compressed('./Programming & Visualisations/SavedSimulations/' + NewSaveFile, rPlot=rPlot, hResults=hResults, pResults=pResults, tResults=tResults, dtResults=dtResults)
        print(f"Successfully Saved Simulation Results into file: {NewSaveFile}")
    
    print(f"Predicted Pressure Asymptote at r = {rPlot[np.where(pResults[-1] == np.max(pResults[-1]))[-1]]}.")

else: # Retrieve data from save file.
    SaveData = np.load(f'./Programming & Visualisations/SavedSimulations/{SAVEFILE}.npz')
    rPlot = SaveData['rPlot']
    hResults = SaveData['hResults']
    pResults = SaveData['pResults']
    tResults = SaveData['tResults']
    dtResults = SaveData['dtResults']
    print(f"Successfully Loaded Simulation Results from file: {SAVEFILE}.npz")
        
        

#TODO: research analytical solution, check dt conditions, study St^1/3 scaling and put on graph
#* Notes from Gemini about things to improve:
#* 1) H&P use implicit finite-difference discretization with backward Euler on a much coarser grid than mine - is that why my corners are much rougher? Is mine better??
#* 2) Relacing Implicit Euler with RK4 doesn't actually help much. But switching to this, plus a stricter h^3_min aware timestep should stop stiffness near touchdown.
#* 3) My peaks are approx 5% bigger than H&P consistently... what's going on? Am I not getting the right data or conditions from the paper?
#* 









print(f"{"Plotting" if PlottingMode != 2 else "Skipping Plotting for"} Interface Evolution...")
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

cbar = fig.colorbar(IMPlot, ax=ax1, fraction=0.035, pad=-0.05)
fig.colorbar(IMPlot, ax=ax2, fraction=0.035, pad=-0.05).ax.set_visible(False) # Places an identical colourbar on the second axis, such that the two figures align.
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
ax2Twin = ax2.twinx()
CellWidthLine, = ax2Twin.plot(rPlot, 1 / (Nr * np.concatenate((CellWidth[::-1], CellWidth))), color='grey', ls=':', lw=1.4, alpha=0.6, label=r'Domain Cell Distribution $(1 / \Delta r)$')

ax2.set_ylim(0, np.max(pResults) * 1.05)
ax2.set_xlabel(r'Position ($r/R$)')
ax2.set_ylabel(r'Pressure ($p / \rho U^2$)')
ax2Twin.set_ylabel(r'Cells per $N_r^{-1}$ Region')
ax2.grid(True, alpha=0.3)
lines = [PressureMap, RigidPressure, CellWidthLine]
ax2.legend(lines, [l.get_label() for l in lines], loc='upper right', fontsize=8)
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
#axT.set_title(r"Linear Progression of Simulation time $t$")
axT.set_xlabel(r"Linear Progression of Simulation time $t$")
Frames = range(0, len(hResults), 1 if SAVEFILE == '' else GlobalSpeedupMultiplier)



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

if (PlottingMode != 2):
    Animation = FuncAnimation(fig, DrawFrame, frames=Frames, interval=20, blit=True)
    # Suppresses some errors that appear during playback (exiting last frame early).
    _Step = Animation._step
    def QuietStep(*args):
        return _Step(*args) if fig.canvas.manager is not None else False
    Animation.event_source.remove_callback(_Step)
    Animation.event_source.add_callback(QuietStep)
    plt.show()
plt.close('all')






print(f"{"Plotting" if PlottingMode != 1 else "Skipping Plotting for"} Hicks & Purvis Comparison...")
# Hicks & Purvis (2010), Figs. 3(a), 3(b): Free-surface height f(r, t) and Interfacial pressure p(r, t).
# Converts our time domain to Hicks & Purvis time: shifts by TouchdownTime, and apply dimensions to the variables:
# HP gives f = ~f * cbrt{2} / delta^2  ,  t = ~t * cbrt{2} / delta^2  ,  p = delta * cbrt{2} ~p  ,  x = cbrt{2}^2 x~ / delta           , where delta = cbrt{mu_g / mu_l Re}
Re = rho * U * R / mu_l #? Is this mu_g or mu_l?
delta = np.cbrt(mu_g / (mu_l * Re))
rPlot = rPlot[len(rPlot)//2:] / (np.cbrt(2)**2 * delta)
hResults = hResults[:, hResults.shape[1]//2:] / (np.cbrt(2) * delta**2)
pResults = pResults[:, pResults.shape[1]//2:] * delta / np.cbrt(2)
tResults = (tResults - TouchdownTime) / (np.cbrt(2) * delta**2)
tPlotValues = np.arange(-6, 7)
print(f"Translated Time Domain: [{tResults[0]:.2f}, {tResults[-1]:.2f}]s.")



# import solver_suite as ss
# c2 = np.cbrt(2.0)
# t_end_hp6 = h_0 + 6.0 * (c2 * delta**2)  # stop exactly at t_HP = 6
# runs = {
#     "DtN (K_P+K_S) RK4":     ss.run_dtn_rk4(St, h_0, Nr=250, t_end=t_end_hp6),
#     "NtD (M^-1) RK4":        ss.run_ntd_rk4(St, h_0, Nr=250, t_end=t_end_hp6),
#     "NtD Crank-Nicolson":    ss.run_ntd_implicit(St, h_0, Nr=250, theta=0.5, t_end=t_end_hp6),
#     "NtD Backward Euler":    ss.run_ntd_implicit(St, h_0, Nr=250, theta=1.0, dt_max=0.1*(c2*delta**2), t_end=t_end_hp6),
#     "Fourier-Bessel RK4":    ss.run_fourier_bessel_rk4(St, h_0, Nr=240, n_modes=120, t_end=t_end_hp6),
#     "2D Cartesian DCT RK4":  ss.run_2d_cartesian_dct_rk4(St, h_0, N=48, t_end=t_end_hp6),
# }
# print(f"{'Solver':24s} | {'f(0)':>7s} | {'f_min':>7s} @ {'r_min':>5s} | {'p_max':>7s} @ {'r_pk':>5s}")
# print("-" * 68)
# for name, res in runs.items():
#     r_hp = res['r'] / (c2**2 * delta)
#     f_hp = res['h'][-1] / (c2 * delta**2)
#     p_hp = res['p'][-1] * delta / c2
#     i_min, i_max = np.argmin(f_hp), np.argmax(p_hp)
#     print(f"{name:24s} | {f_hp[0]:7.4f} | {f_hp[i_min]:7.4f} @ {r_hp[i_min]:5.3f} | {p_hp[i_max]:7.4f} @ {r_hp[i_max]:5.3f}")
# res = ss.run_ntd_implicit(St, h_0, Nr=300, theta=0.5, t_end=h_0 + 8.5 * (c2 * delta**2))
# rPlot    = np.concatenate((-res['r'][::-1], res['r']))
# hResults = np.hstack((res['h'][:, ::-1], res['h']))
# pResults = np.hstack((res['p'][:, ::-1], res['p']))
# tResults = res['t']
# dtResults = res['dt']



Styles = {-6:(1.5,(0,(4,1.6))), -5:(0.7,(0,(5,4))), -4:(0.8,(0,(1,2))), -3:(0.7,(0,(11,3,1.5,3))),
         -2:(0.7,(0,(7,4,1.5,4))), -1:(0.7,(0,(4,7))), 0:(1.5,'-'), 1:(0.7,(0,(10,3,2,3))),
          2:(0.7,(0,(5,3))), 3:(0.7,'-'), 4:(0.7,(0,(9,3))), 5:(0.7,(0,(4,4))), 6:(1.2,(0,(1,1)))}
Colors = cm.viridis(np.linspace(0.05, 0.92, len(tPlotValues)))
fig,(ax1, ax2)=plt.subplots(2,1,figsize=(9.2,7.0),sharex=True,gridspec_kw=dict(height_ratios=[418,446],hspace=0.045))

for t, col in zip(tPlotValues, Colors):
    # Hicks & Purvis (2010) reference points
    r_ref_f, f_ref = HPFig3aResults[t]
    r_ref_p, p_ref = HPFig3bResults[t]
    lw,ls=Styles[t]
    ax1.plot(r_ref_f, f_ref, color=col, lw=lw, ls=ls, solid_capstyle='butt')
    ax2.plot(r_ref_p, p_ref, color=col, lw=lw, ls=ls, solid_capstyle='butt')
    
    # Finds closest datapoint in simulation and attempts to draw it.
    SimTimeIndex = np.abs(tResults - t).argmin()
    ax1.plot(rPlot, hResults[SimTimeIndex], "-", color=col, lw=1.6)
    ax2.plot(rPlot, pResults[SimTimeIndex], "-", color=col, lw=1.6)

# Match Hicks & Purvis Fig. 3 formatting
ax1.tick_params(direction='in',top=True,right=True,length=5)
ax1.set_xlim(0,5)
ax2.tick_params(direction='in',top=True,right=True,length=5)
ax2.set_xlim(0,5)
ax1.set_ylim(0,10); ax2.set_ylim(0,0.5)
ax1.set_yticks(range(0,11,2)); ax2.set_yticks([0.1,0.2,0.3,0.4,0.5])
ax1.set_ylabel('Free-surface height $h$'); ax2.set_ylabel('Pressure $p$'); ax2.set_xlabel('Radius $r$')
ax1.annotate('$t=-6$',xy=(1.5,8.3),xytext=(0.15,9.0),va='center',arrowprops=dict(arrowstyle='-',lw=1.5,ls='--'))
ax1.annotate('$t=0$',xy=(2.38,5.72),xytext=(4.45,6.15),va='center',arrowprops=dict(arrowstyle='-',lw=1.5))
ax1.annotate('$t=6$',xy=(3.0,1.97),xytext=(4.45,1.0),va='center',arrowprops=dict(arrowstyle='-',lw=1.2,ls=':'))

legend_elements = [ Line2D([0], [0], color='black', lw=1.2, ls='--', label=r'H\&P'), Line2D([0], [0], color='black', lw=2.0, ls='-', label=r'Simulation') ]
ax2.legend(handles=legend_elements, loc='upper right', frameon=True)

if (PlottingMode != 1):
    plt.show()