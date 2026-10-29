import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_banded
from scipy.integrate import solve_ivp
import time

# Sets up water-like configuration into a non-dimensionless constant. Afer this, we can remove all dimensions from further points.
rho = 1e3
R = 5e-4
U = 1.0
mu_g = 1.8e-5
m = 1.0 # rho * np.pi * R**3 * 4/3

ScalingFactor = 1e-4
St = ScalingFactor * rho * R * U / mu_g
print(f"Stokes Number: {St:.2e}")

def MatrixPreElement(i):
    h_iMinus1 = 0.5 * (h[i - 1]**3 + h[i]**3)
    r_iMinus1 = i * dr
    return h_iMinus1 * r_iMinus1

def InitMatrix(i, j):
    MatrixElement = 0.0
    #TODO: Okay wtf, inverting this line seems logical but then we can't feed it into the scipy solver. What is
    #TODO: the right way to handle this BC?
    #if i == N - 1: return -InitMatrix(i - 1, j) # Boundary conditions: sets p_N = -p_{N-1} s.t the pressure at the edge of the final cell is 0.
    match i - j:
        case -1:
            MatrixElement = MatrixPreElement(i + 1)
        case 0:
            # Boundary conditions: q_{n + 1/2} = 0. We then enforce p_N = -p_{N-1} s.t the pressure at the edge of the final cell is 0.
            if (i == Nr - 1):
                MatrixElement = -2.0 * MatrixPreElement(i)
            elif (i == Nr - 2):
                MatrixElement = -2.0 * MatrixPreElement(i + 1)
            else:
                MatrixElement = -MatrixPreElement(i + 1)
            MatrixElement -= (0.0 if i == 0 else MatrixPreElement(i)) # Boundary conditions: q_{-1/2} = 0 - no pressure gradient along the central line (symmetry).
        case 1:
            MatrixElement = MatrixPreElement(i)
    return St * MatrixElement / (12 * r[i] * dr * dr)

# def ODEForce(t, dhdt, h0_val):
#     return -dhdt * 6 * np.pi / (St * h0_val * m) #TODO: +??

ZoomFactor = 1/4
Nr = 1000
dr = ZoomFactor / Nr
r = (np.arange(Nr) + 0.5) * dr # Data taken at cell centres.
h_0 = 5e-2 # h_0 << R for weak parabolic effect.
h = h_0 + 0.5 * r**2
dhdt_0 = np.full(Nr, -1.0)
dhdt = dhdt_0.copy()

Nt = 500
dt = (3.0 * h_0 / 1.0) / Nt

hResults = []
pResults = []
st = time.time()
for n in range(Nt):
    print(f"Running Time Iteration {n+1} / {Nt}...", end="\r", flush=True)
    AB = np.zeros((3,Nr))
    AB[0, 1:] = [InitMatrix(j - 1, j) for j in range(1, Nr)]
    AB[1, :] = [InitMatrix(i, i) for i in range(Nr)]
    AB[2, :-1] = [InitMatrix(j + 1, j) for j in range(Nr - 1)]
    p = solve_banded((1, 1), AB, dhdt)
    # Saves results, so that they can be outputted. Note that we make everything symmetric also! :)
    hResults.append(np.concatenate((h[::-1], h)))
    pResults.append(np.concatenate((p[::-1], p)))
    
    # Adjusts the film via the force acting on the film. I derived this to be F = ma = -dh/dt * 6pi / St h. Force points in the opposite way, so this becomes a +.
    # sol = solve_ivp(ODEForce, t_span=(0.0, dt), y0=dhdt, args=(h[0],))
    # h += sol.y[:,-1] * dt
    
    accel = np.sum((3/2) * p * r * dr) #? F = \int^\infty_0 p 2\pi r dr, but non-dimensionalised. My thoughts are that we therefore just multiply by each area element? :O
    #print(accel)
    
    # r_faces = np.arange(1, Nr) * dr
    # dpdr = (p[1:] - p[:-1]) / dr
    # kernel = (2.0 * r_faces[None, :]) / (r_faces[None, :]**2 - r[:, None]**2)
    # accel = -(1.0 / np.pi) * (kernel @ dpdr) * dr

    dhdt += accel * dt
    h += dhdt * dt
    
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

ax1.set_xlim(r[0], r[-1])
ax1.set_ylim(0, np.max(hResults))
ax1.set_ylabel(r'Dimensionless Height $h/R$')
ax1.set_title('Interaction of a Droplet Colliding with a Thin Film')

ax2.set_ylim(0, np.max(pResults) * 1.1)
ax2.set_xlabel(r'Position ($x/R$)')
ax2.set_ylabel(r'Pressure ($p / (\rho U^2)$)')
ax2.grid(True, alpha=0.3)

cbar = fig.colorbar(IMPlot, ax=ax1, pad=0.02)
cbar.set_label(r'Pressure ($p / (\rho U^2)$)')
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

Animation = FuncAnimation(fig, DrawFrame, frames=Nt, interval=20, blit=True)
plt.show()