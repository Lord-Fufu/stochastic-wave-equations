
import numpy as np
import matplotlib.pyplot as plt
import imageio as iio

from fipy import numerix as nx
import pandas as pd
import os

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from scipy.stats import truncnorm


#--------------------------------------------------------------------------------------------------------------
#---------------------------------------------SIMULATION PARAMETERS-----------------------------------------------
#--------------------------------------------------------------------------------------------------------------

# maximum number of timesteps
itmax = float('infinity')

# grid setup
n_x = 52
dx = 1e3
l_x = n_x * dx

n_y = 52
dy = 1e3
l_y = n_y * dy

x, y = (
    np.arange(n_x) * dx,
    np.arange(n_y) * dy
)
Y, X = np.meshgrid(y, x, indexing='ij')

l_ref = np.sqrt(l_x * l_y)

# physical parameters
gravity = 9.81
depth = 100.
coriolis_f = 0 #2e-4
coriolis_beta = 0 #2e-11
coriolis_param = coriolis_f + Y * coriolis_beta

# Poisson problem setup
n_z = 12
depthDiscrete = depth
kx = np.fft.fftfreq(n_x, dx)
ky = np.fft.fftfreq(n_y, dy)
Kx,Ky = np.meshgrid(kx,ky,indexing='xy')

#epsilon only impacts the wave height in this model --> maybe write it with adimensioned variables
eps = 0.1
beta = 1 #depth/np.sqrt(l_x*l_y)
nu = 1 #1/(1+beta) #scaling parameter (cf Vincent Duchêne)
amplitude = eps*depth

# type of boundary conditions parameters (periodic or Dirichlet)
periodic_boundary_x = True
periodic_boundary_y = True

# adams-bashforth parameters
adams_bashforth_a = 1.5
adams_bashforth_b = -0.5

# timestep
dt = 0.1*min(dx, dy) / np.sqrt(depth)
print( "\n timestep", dt, "\n")

# matrix for implicit temporal scheme
diffCoeff = 1e-3
nRavelled = (n_x-2)*(n_y-2)
diffusionMatrix = 2*(1/dx**2 + 1/dy**2)*np.eye(nRavelled)
for i in range(n_x-2):
    for j in range(n_y-2):
        diffusionMatrix[i%(n_x-3)*(n_y-3) + j%(n_y-3), (i+1)%(n_x-3)*(n_y-3) + j%(n_y-3)] = -1/dx**2
        diffusionMatrix[i%(n_x-3)*(n_y-3) + j%(n_y-3), (i-1)%(n_x-3)*(n_y-3) + j%(n_y-3)] = -1/dx**2
        diffusionMatrix[i%(n_x-3)*(n_y-3) + j%(n_y-3), i%(n_x-3)*(n_y-3) + (j+1)%(n_y-3)] = -1/dy**2
        diffusionMatrix[i%(n_x-3)*(n_y-3) + j%(n_y-3), i%(n_x-3)*(n_y-3) + (j-1)%(n_y-3)] = -1/dy**2
diffusionMatrix *= diffCoeff
diffusionMatrix = np.eye(nRavelled) + diffusionMatrix
diffusionMatrix = csr_matrix(diffusionMatrix)

# other parameters
phase_speed = np.sqrt(gravity * depth)
rossby_radius = np.sqrt(gravity * depth) / coriolis_param.mean()

coeffAlias = 1
antiAlias = np.exp(-coeffAlias*(Kx**2 + Ky**2))[1:-1,1:-1]

#--------------------------------------------------------------------------------------------------------------
#------------------------------------------SAVE & PLOT PARAMETERS----------------------------------------------
#--------------------------------------------------------------------------------------------------------------

# save parameters
save_data = False

if save_data:
    dir_to_store = 'data_deterministic'
    path_to_store=dir_to_store+'//'

    try:
        os.makedirs(dir_to_store)
        print(f"Directory '{dir_to_store}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_to_store}' already exists.")

    os.makedirs(path_to_store + 'height')
    os.makedirs(path_to_store + 'xmom')
    os.makedirs(path_to_store + 'ymom')
    print(f"Subdirectories created successfully.")

# plot parameters
plot_range = 0.5*amplitude
plot_every = 1

max_quivers = 41

make_plots = True
save_plots = False
makegif = False

plotNoiseDivergence = False
plotIS = False
plotISDIV = False

ratio = (1+np.sqrt(5))/2

#--------------------------------------------------------------------------------------------------------------
#---------------------------------------------INITIAL CONDITIONS-----------------------------------------------
#--------------------------------------------------------------------------------------------------------------

'''temp = 10 * np.exp(-(Y - y[n_y // 2])**2 / (0.02 * l_x)**2)
spotential0 = np.zeros_like(temp)'''

h0 = (
    depth
    # small perturbation
    #+ 1 * np.sin(X / l_x * 10 * np.pi) * np.cos(Y / l_y * 8 * np.pi)
    + amplitude*np.exp(-1e-15*((X-l_x/4)**2 + (Y-l_y/4)**2)**2)
    #+ amplitude*np.exp(-100 * (0.9*abs(Y / l_y - 0.5)**2+ ((Y/l_y - 0.5) - abs(X / l_x - 0.5)**(2/3))**2))
)
spotential0 = np.zeros_like(h0)


#--------------------------------------------------------------------------------------------------------------
#----------------------------------------------NOISE PARAMETERS------------------------------------------------
#---------------------------------------FOR DOUBLE PERIODIC BOUNDARIES-----------------------------------------
#--------------------------------------------------------------------------------------------------------------
nmodes = 8
phi = np.zeros((n_y, n_x, nmodes))
chix = np.zeros((n_y, n_x, nmodes))
chiy = np.zeros((n_y, n_x, nmodes))
chiz = np.zeros((n_y, n_x, nmodes))

if periodic_boundary_x and periodic_boundary_y and nmodes > 0:
    ups = 0 #1e1
    wavenumber = 2*np.pi*20

    upsRot = 1e4
    coeffChiZ = 0 #0.01
    wavenumberRot = 2*np.pi*100

    coeff=-2
    l_waveX = (l_x + coeff*dx)
    l_waveY = (l_y + coeff*dy)
    l_wave = (l_waveX*l_waveY)**0.5

    phi1 = np.sqrt(ups)*l_wave/l_ref*np.sin(wavenumber*(X/l_waveX+Y/l_waveY))
    phi2 = np.sqrt(ups)*l_wave/l_ref*np.sin(wavenumber*(X/l_waveX-Y/l_waveY))
    phi3 = np.sqrt(ups)*l_wave/l_ref*np.cos(wavenumber*(X/l_waveX+Y/l_waveY))
    phi4 = np.sqrt(ups)*l_wave/l_ref*np.cos(wavenumber*(X/l_waveX-Y/l_waveY))

    phi[1:-1,1:-1,0] = phi1[1:-1,1:-1]
    phi[1:-1,1:-1,1] = phi2[1:-1,1:-1]
    phi[1:-1,1:-1,2] = phi3[1:-1,1:-1]
    phi[1:-1,1:-1,3] = phi4[1:-1,1:-1]

    chix1 = - np.sqrt(upsRot)*l_wave/l_ref*np.cos(wavenumberRot*(X/l_waveX+Y/l_waveY))
    chix2 =   np.sqrt(upsRot)*l_wave/l_ref*np.cos(wavenumberRot*(X/l_waveX-Y/l_waveY))
    chix3 =   np.sqrt(upsRot)*l_wave/l_ref*np.sin(wavenumberRot*(X/l_waveX+Y/l_waveY))
    chix4 = - np.sqrt(upsRot)*l_wave/l_ref*np.sin(wavenumberRot*(X/l_waveX-Y/l_waveY))

    chix[1:-1,1:-1,4] = chix1[1:-1,1:-1]
    chix[1:-1,1:-1,5] = chix2[1:-1,1:-1]
    chix[1:-1,1:-1,6] = chix3[1:-1,1:-1]
    chix[1:-1,1:-1,7] = chix4[1:-1,1:-1]

    chiy1 = np.sqrt(upsRot)*l_wave/l_ref*np.cos(wavenumberRot*(X/l_waveX+Y/l_waveY))
    chiy2 = np.sqrt(upsRot)*l_wave/l_ref*np.cos(wavenumberRot*(X/l_waveX-Y/l_waveY))
    chiy3 = np.sqrt(upsRot)*l_wave/l_ref*np.sin(wavenumberRot*(X/l_waveX+Y/l_waveY))
    chiy4 = np.sqrt(upsRot)*l_wave/l_ref*np.sin(wavenumberRot*(X/l_waveX-Y/l_waveY))

    chiy[1:-1,1:-1,4] = chiy1[1:-1,1:-1]
    chiy[1:-1,1:-1,5] = chiy2[1:-1,1:-1]
    chiy[1:-1,1:-1,6] = chiy3[1:-1,1:-1]
    chiy[1:-1,1:-1,7] = chiy4[1:-1,1:-1]

    chiz1 = coeffChiZ*np.sqrt(upsRot)*l_wave/l_ref*np.cos(wavenumberRot*(X/l_waveX+Y/l_waveY))
    chiz2 = coeffChiZ*np.sqrt(upsRot)*l_wave/l_ref*np.cos(wavenumberRot*(X/l_waveX-Y/l_waveY))
    chiz3 = coeffChiZ*np.sqrt(upsRot)*l_wave/l_ref*np.sin(wavenumberRot*(X/l_waveX+Y/l_waveY))
    chiz4 = coeffChiZ*np.sqrt(upsRot)*l_wave/l_ref*np.sin(wavenumberRot*(X/l_waveX-Y/l_waveY))

    chiz[1:-1,1:-1,4] = chiz1[1:-1,1:-1]
    chiz[1:-1,1:-1,5] = chiz2[1:-1,1:-1]
    chiz[1:-1,1:-1,6] = chiz3[1:-1,1:-1]
    chiz[1:-1,1:-1,7] = chiz4[1:-1,1:-1]
    

#--------------------------------------------------------------------------------------------------------------
#---------------------------------------------AUXILIARY FUNCTIONS----------------------------------------------
#--------------------------------------------------------------------------------------------------------------
@np.vectorize
def vectorizedMax(a,b):
    return max(a,b)

def prepare_plot():
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    cs = update_plot(0, h0, spotential0, ax, count=0, draw=False)
    plt.colorbar(cs, label='$\\eta$ (m)')
    return fig, ax

def update_plot(t, h, phi0, ax, count, draw=True):
    eta = h - depth

    quiver_stride = (
        slice(1, -1, n_y // max_quivers),
        slice(1, -1, n_x // max_quivers)
    )

    ax.clear()
    cs = ax.pcolormesh(
        x[1:-1] / 1e3,
        y[1:-1] / 1e3,
        eta[1:-1, 1:-1],
        vmin=-plot_range, vmax=plot_range, cmap='RdBu_r'
    )

    ax.set_aspect('equal')
    ax.set_xlabel('$x$ (km)')
    ax.set_ylabel('$y$ (km)')
    ax.set_xlim(x[1] / 1e3, x[-2] / 1e3)
    ax.set_ylim(y[1] / 1e3, y[-2] / 1e3)
    ax.set_title(
        't=%5.2fdays, c=%5.1f m/s '
        % (t/86400, phase_speed)
    )
    if save_plots:
        # Enregistrement dans le dossier "images"
        plt.savefig(f"images/{count}.jpg")

    if draw:
        plt.pause(0.1)

    return cs


def enforce_boundaries(arr, grid):
    assert grid in ('h', 'u', 'v')
    if periodic_boundary_x:
        arr[:, 0]  = arr[:, -2]
        arr[:, -1] = arr[:, 1]
    elif grid == 'u':
        arr[:, -2] = 0.
    if periodic_boundary_y:
        arr[0 , :] = arr[-2, :]
        arr[-1, :] = arr[1 , :]
    elif grid == 'v':
        arr[-2, :] = 0.
    if periodic_boundary_x and periodic_boundary_y:
        arr[0, 0]   = arr[-2, -2]
        arr[-1, 0]  = arr[1, -2]
        arr[0, -1]  = arr[-2, 1]
        arr[-1, -1] = arr[1, 1]
    return arr


for k in range(nmodes):
    phi[:,:,k] = enforce_boundaries(phi[:,:,k], 'h')
    chix[:,:,k] = enforce_boundaries(chix[:,:,k], 'h')
    chiy[:,:,k] = enforce_boundaries(chiy[:,:,k], 'h')



vectx = np.zeros_like(phi)
vecty = np.zeros_like(phi)
for k in range(nmodes):
    vectx[1:-1,1:-1,k] = (phi[1:-1,2:,k] - phi[1:-1,:-2,k])/(2*dx)
    vecty[1:-1,1:-1,k] = (phi[2:,1:-1,k] - phi[:-2,1:-1,k])/(2*dy)
    vectx[1:-1,1:-1,k] += chix[1:-1,1:-1,k]
    vecty[1:-1,1:-1,k] += chiy[1:-1,1:-1,k]
    vectx[:,:,k] = enforce_boundaries(vectx[:,:,k], 'u')
    vecty[:,:,k] = enforce_boundaries(vecty[:,:,k], 'v')

#if ups == 0:
#    nmodes = 0

a_xx = np.zeros_like(spotential0)
a_xy = np.zeros_like(spotential0)
a_yy = np.zeros_like(spotential0)

us = np.zeros_like(spotential0)
vs = np.zeros_like(spotential0)

a_xx = vectx**2
a_xy = vectx*vecty
a_yy = vecty**2
for k in range(nmodes):
    a_xx[:,:,k] = enforce_boundaries(a_xx[:,:,k], 'u')
    a_xx[:,:,k] = enforce_boundaries(a_xx[:,:,k], 'v')
    a_xy[:,:,k] = enforce_boundaries(a_xy[:,:,k], 'u')
    a_xy[:,:,k] = enforce_boundaries(a_xy[:,:,k], 'v')
    a_yy[:,:,k] = enforce_boundaries(a_yy[:,:,k], 'u')
    a_yy[:,:,k] = enforce_boundaries(a_yy[:,:,k], 'v')

us[1:-1,1:-1] = np.sum((a_xx[1:-1,2:,:] - a_xx[1:-1,:-2,:])/dx + (a_xy[2:,1:-1,:] - a_xy[:-2,1:-1,:])/dy, axis=2)/2
vs[1:-1,1:-1] = np.sum((a_xy[1:-1,2:,:] - a_xy[1:-1,:-2,:])/dx + (a_yy[2:,1:-1,:] - a_yy[:-2,1:-1,:])/dy, axis=2)/2
us[:,:] = enforce_boundaries(us[:,:], 'u')
vs[:,:] = enforce_boundaries(vs[:,:], 'v')

div_us = np.zeros_like(us)
div_us[1:-1,1:-1] = (us[1:-1,1:-1] - us[1:-1,:-2])/dx + (vs[1:-1,1:-1] - vs[:-2,1:-1])/dy
div_us = enforce_boundaries(div_us, 'h')


nplots = plotNoiseDivergence + plotIS + plotISDIV

if nplots>0:
    fig, ax = plt.subplots(2, 2, figsize=(14, 7))
    cs = ax[0,0].pcolormesh(
        x[1:-1] / 1e3,
        y[1:-1] / 1e3,
        (vectx[1:-1, 1:-1, 1] - vectx[1:-1, :-2, 1])/dx + (vecty[1:-1, 1:-1, 1] - vecty[:-2, 1:-1, 1])/dy,
        vmin=-1e-5, vmax=1e-5, cmap='RdBu_r'
    )
    ax[0,0].set_title('Noise divergence of 2nd mode along x.')
    fig.colorbar(cs, label='$phi2x', ax=ax[0,0])

    cs = ax[1,0].pcolormesh(
        x[1:-1] / 1e3,
        y[1:-1] / 1e3,
        us[1:-1,1:-1],
        vmin=-1e-3, vmax=1e-3, cmap='RdBu_r'
    )
    ax[1,0].set_title('Itô-Stokes drift along x.')
    fig.colorbar(cs, label='$us along x', ax=ax[1,0])

    cs = ax[0,1].pcolormesh(
        x[1:-1] / 1e3,
        y[1:-1] / 1e3,
        div_us[1:-1,1:-1],
        vmin=-1e-7, vmax=1e-7, cmap='RdBu_r'
    )
    ax[0,1].set_title('Divergence of the ISD.')
    fig.colorbar(cs, label='$div us', ax=ax[0,1])

    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])
    ax[1,1].spines['top'].set_visible(False)
    ax[1,1].spines['right'].set_visible(False)
    ax[1,1].spines['bottom'].set_visible(False)
    ax[1,1].spines['left'].set_visible(False)

print("Maximal value of norm us: ", np.max(np.sqrt(us[1:-1,1:-1]**2 + vs[1:-1,1:-1]**2)))
print("Maximal value of div us: ", np.max(abs((us[1:-1,1:-1] - us[1:-1,:-2])/dx + (vs[1:-1,1:-1] - vs[:-2,1:-1])/dy)))
'''print("Maximal value of norm chi: ", np.max(np.sqrt(chix[1:-1,1:-1,:]**2 + chiy[1:-1,1:-1,:]**2)))
print("Maximal value of div chi: ", np.max(abs((chix[1:-1,1:-1,:] - chix[1:-1,:-2,:])/dx + (chiy[1:-1,1:-1,:] - chiy[:-2,1:-1,:])/dy)))
print("Periodicity default of chix: ", np.max(abs(chix[1,:,7] - chix[-2,:,7]) + abs(chix[:,1,7] - chix[:,-2,7])))'''

def export_to_csv(field, name):
    df = pd.DataFrame(field)
    df.to_csv(name)

#--------------------------------------------------------------------------------------------------------------
#-----------------------------------------DIRICHLET TO NEUMANN OPERATOR----------------------------------------------
#--------------------------------------------------------------------------------------------------------------
def solvePoisson(Phi, dz): # Solve the Poisson linear system
    fftPhi = np.fft.fft2(Phi[1:-1,1:-1])
    fftFullPotentialInternal = np.zeros((n_y-2,n_x-2,n_z), dtype='complex')
    b = np.zeros(n_z, dtype='complex')
    
    for i in range(0,n_x-2):
        for j in range(0,n_y-2):
            A = (1/dz[j,i]**2)*(2*np.eye(n_z) - np.eye(n_z, k=1) - np.eye(n_z, k=-1))
            A[0,0] -= 1/dz[j,i]**2
            A += (kx[i]**2 + ky[j]**2)*np.eye(n_z)
            #A = csr_matrix(A)
            b[-1] = fftPhi[j,i]/dz[j,i]**2
            fftFullPotentialInternal[j,i,:] = np.linalg.solve(A,b)
            #fftFullPotentialInternal[j,i,:] = spsolve(A,b)

    fullPotentialInternal = np.zeros((n_y-2,n_x-2,n_z))
    for k in range(n_z):
        fullPotentialInternal[:,:,k] = np.real(np.fft.ifft2(fftFullPotentialInternal[:,:,k])) #np.real(np.fft.ifft2(antiAlias*fftFullPotentialInternal[:,:,k])) #
    
    fullPotential = np.zeros((n_y, n_x, n_z))
    for k in range(n_z):
        fullPotential[:,:,k] = np.pad(fullPotentialInternal[:,:,k], 1, 'edge')
        fullPotential[:,:,k] = enforce_boundaries(fullPotential[:,:,k], 'h')
    return fullPotential

z = np.arange(n_z) * depthDiscrete/n_z
Y2, X2, Z2 = np.meshgrid(y, x, z, indexing='ij')
alpha=2
etaForTest = np.zeros((n_y, n_x))
PhiForTest = np.sin(alpha*np.pi*X/l_x)*np.sin(alpha*np.pi*Y/l_y)
dzForTest = depthDiscrete/n_z * np.ones_like(etaForTest)

solForTest = solvePoisson(PhiForTest, dzForTest)
solReference = (np.cosh(alpha*np.sqrt(1/l_x**2 + 1/l_y**2)*np.pi*Z2)/np.cosh(alpha*np.sqrt(1/l_x**2 + 1/l_y**2)*np.pi*depthDiscrete)
                    *np.sin(alpha*np.pi*X2/l_x)*np.sin(alpha*np.pi*Y2/l_y))
    
err = np.max(abs(solForTest - solReference)[1:-1,1:-1,:])
if err >=1e-3:
    plt.plot(X2[6,1:-1,-1],solForTest[6,1:-1,-1])
    plt.plot(X2[6,1:-1,-1],solReference[6,1:-1,-1])
    plt.show()
    raise ValueError("Poisson equation isn't solved properly.")
else:
    print('Poisson equation solved OK.')

@np.vectorize
def computeSign(c):
    if abs(c) < 1e-6:
        return 0
    else:
        return np.sign(c)

def computeDerivativeZ(phiSurface, phiFluid, eta, dz):
    output = np.zeros((n_y, n_x))
    sgnEta = computeSign(eta)
    #zeroth order method (with constant dz) 
    '''dzZeroth = depthDiscrete/n_z
    output[1:-1,1:-1] = (phiSurface[1:-1,1:-1] - phiFluid[1:-1,1:-1,-1])/dzZeroth*sgnEta[1:-1,1:-1]'''
    #first order method
    '''for i in range(n_x):
        for j in range(n_y):
            if phiSurface[j,i] - phiFluid[j,i,-1]>=0:
                output[j,i] = (3*phiSurface[j,i] - 4*phiFluid[j,i,-1] + phiFluid[j,i,-2])/dz[j,i]/2*sgnEta[j,i]
            else:
                output[j,i] = (- phiSurface[j,i] + 4*phiFluid[j,i,-1] - 3*phiFluid[j,i,-2])/dz[j,i]/2*sgnEta[j,i]'''
    #multiple layers method
    '''for i in range(n_x):
        for j in range(n_y):
            layer = int(eta[j,i]//dz[j,i])
            if layer > 0:
                output[j,i] = (phiSurface[j,i] - phiFluid[j,i,-1])/dz[j,i]
            else:
                output[j,i] = (phiFluid[j,i,layer-1] - phiFluid[j,i,layer-2])/dz[j,i]'''
    output[1:-1,1:-1] = (phiSurface[1:-1,1:-1] - phiFluid[1:-1,1:-1,-1])/dz[1:-1,1:-1]*sgnEta[1:-1,1:-1]
    return output
        
def dirichletToNeumannForPhi(eta, Phi): # Compute Dirichlet to Neumann from the solution of the Poisson equation
    hDiscrete = depthDiscrete + eta
    dz= hDiscrete/n_z #depthDiscrete/n_z * np.ones_like(hDiscrete) #
    fullPotential = solvePoisson(Phi, dz)
    
    gxFullPotentialRight = np.zeros_like(Phi)
    gxFullPotentialLeft = np.zeros_like(Phi)
    gyFullPotentialUp = np.zeros_like(Phi)
    gyFullPotentialDown = np.zeros_like(Phi)
    gxFullPotentialRight[1:-1,1:-1] = (fullPotential[1:-1,2:,  -1] - fullPotential[1:-1,1:-1,-1])/dx
    gxFullPotentialLeft[1:-1,1:-1]  = (fullPotential[1:-1,1:-1,-1] - fullPotential[1:-1,:-2, -1])/dx
    gyFullPotentialUp[1:-1,1:-1]    = (fullPotential[2:,1:-1,  -1] - fullPotential[1:-1,1:-1,-1])/dy
    gyFullPotentialDown[1:-1,1:-1]  = (fullPotential[1:-1,1:-1,-1] - fullPotential[:-2,1:-1, -1])/dy

    velocityX = (gxFullPotentialRight + gxFullPotentialLeft)/2
    velocityY = (gyFullPotentialUp + gyFullPotentialDown)/2
    velocityLeft, velocityRight = np.maximum(velocityX,0), np.minimum(velocityX,0)
    velocityDown, velocityUp  = np.maximum(velocityY,0), np.minimum(velocityY,0)

    gxEtaRight = np.zeros_like(eta)
    gxEtaLeft = np.zeros_like(eta)
    gyEtaUp = np.zeros_like(eta)
    gyEtaDown = np.zeros_like(eta)
    gxEtaRight[1:-1,1:-1] = (eta[1:-1,  2:] - eta[1:-1,1:-1])/dx
    gxEtaLeft[1:-1,1:-1]  = (eta[1:-1,1:-1] - eta[1:-1, :-2])/dx
    gyEtaUp[1:-1,1:-1]    = (eta[2:,  1:-1] - eta[1:-1,1:-1])/dy
    gyEtaDown[1:-1,1:-1]  = (eta[1:-1,1:-1] - eta[:-2, 1:-1])/dy

    derivativeZ = computeDerivativeZ(Phi, fullPotential, eta, dz)
    output = np.zeros_like(Phi)
    output[1:-1,1:-1] =(derivativeZ - beta**2*(velocityRight*gxEtaRight + velocityLeft*gxEtaLeft 
                                             + velocityUp*gyEtaUp + velocityDown*gyEtaDown))[1:-1,1:-1]
    #output[1:-1,1:-1] = derivativeZ[1:-1,1:-1]
    output = enforce_boundaries(output, 'h')

    '''err = np.max(abs(Phi[1:-1,1:-1] - fullPotential[1:-1,1:-1,-1])/dz[1:-1,1:-1])
    if err >= 1e-3:
        print(err)
        raise ValueError("Dirichlet to Neumann operator isn't computed precisely.")'''
    
    return output

def dirichletToNeumannForEta(eta, Phi): # Compute Dirichlet to Neumann from the solution of the Poisson equation
    hDiscrete = depthDiscrete + eta
    dz= depthDiscrete/n_z * np.ones_like(hDiscrete) #hDiscrete/n_z #
    fullPotential = solvePoisson(Phi, dz)
    
    intGxFullPotential = np.zeros_like(Phi)
    intGyFullPotential = np.zeros_like(Phi)
    intGxFullPotential[1:-1,1:-1] = np.sum((fullPotential[1:-1,2:,:] - fullPotential[1:-1,1:-1,:])/dx, axis = 2)*dz[1:-1,1:-1]
    intGyFullPotential[1:-1,1:-1] = np.sum((fullPotential[2:,1:-1,:] - fullPotential[1:-1,1:-1,:])/dy, axis = 2)*dz[1:-1,1:-1]
    intGxFullPotential[1:-1,1:-1] += (Phi[1:-1,2:] - Phi[1:-1,1:-1])/dx*dz[1:-1,1:-1]
    intGyFullPotential[1:-1,1:-1] += (Phi[2:,1:-1] - Phi[1:-1,1:-1])/dy*dz[1:-1,1:-1]
    
    intGxFullPotential = enforce_boundaries(intGxFullPotential, 'h')
    intGyFullPotential = enforce_boundaries(intGyFullPotential, 'h')

    output = np.zeros_like(Phi)
    output[1:-1,1:-1] = -((intGxFullPotential[1:-1,1:-1] - intGxFullPotential[1:-1,:-2])/dx
                         +(intGyFullPotential[1:-1,1:-1] - intGyFullPotential[:-2,1:-1])/dy)
    output = enforce_boundaries(output, 'h')

    '''err = np.max(abs(Phi[1:-1,1:-1] - fullPotential[1:-1,1:-1,-1])/dz[1:-1,1:-1])
    if err >= 1e-3:
        print(err)
        raise ValueError("Dirichlet to Neumann operator isn't computed precisely.")'''
    
    '''hDiscrete = depthDiscrete + eta
    intGxFullPotential = np.zeros_like(Phi)
    intGyFullPotential = np.zeros_like(Phi)
    intGxFullPotential[1:-1,1:-1] += (Phi[1:-1,2:] - Phi[1:-1,1:-1])/dx
    intGyFullPotential[1:-1,1:-1] += (Phi[2:,1:-1] - Phi[1:-1,1:-1])/dy
    intGxFullPotential = enforce_boundaries(intGxFullPotential, 'h')
    intGyFullPotential = enforce_boundaries(intGyFullPotential, 'h')

    output = np.zeros_like(Phi)
    output[1:-1,1:-1] = ((intGxFullPotential[1:-1,1:-1] - intGxFullPotential[1:-1,:-2])/dx
                        +(intGyFullPotential[1:-1,1:-1] - intGyFullPotential[:-2,1:-1])/dy)*hDiscrete[1:-1,1:-1]
    output = enforce_boundaries(output, 'h')'''

    return output

def dirichletToNeumannForPhi(eta, Phi):
    return dirichletToNeumannForEta(eta, Phi)

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------WAVE MODEL ITERATOR-----------------------------------------------
#--------------------------------------------------------------------------------------------------------------

def oneStepIncr(eta, spotential):
    newH = np.zeros_like(eta)
    newPot = np.zeros_like(spotential)

    dToNForEta = dirichletToNeumannForEta(eta, spotential)
    dToNForPhi = dirichletToNeumannForPhi(eta, spotential)
    newH[1:-1, 1:-1] = dToNForEta[1:-1, 1:-1]/beta**2/nu
    newH = enforce_boundaries(newH, 'h')

    gradxPhiRight = np.zeros_like(spotential)
    gradxPhiLeft  = np.zeros_like(spotential)
    gradyPhiUp    = np.zeros_like(spotential)
    gradyPhiDown  = np.zeros_like(spotential)
    gradxPhiRight[1:-1,1:-1] = (spotential[1:-1,  2:] - spotential[1:-1,1:-1])/dx
    gradxPhiLeft[1:-1,1:-1]  = (spotential[1:-1,1:-1] - spotential[1:-1, :-2])/dx
    gradyPhiUp[1:-1,1:-1]    = (spotential[2:,  1:-1] - spotential[1:-1,1:-1])/dy
    gradyPhiDown[1:-1,1:-1]  = (spotential[1:-1,1:-1] - spotential[:-2, 1:-1])/dy

    velocityX = (gradxPhiRight + gradxPhiLeft)/2
    velocityY = (gradyPhiUp + gradyPhiDown)/2
    velocityLeft, velocityRight = np.maximum(velocityX,0), np.minimum(velocityX,0)
    velocityDown, velocityUp  = np.maximum(velocityY,0), np.minimum(velocityY,0)

    gradxEtaRight = np.zeros_like(eta)
    gradxEtaLeft  = np.zeros_like(eta)
    gradyEtaUp    = np.zeros_like(eta)
    gradyEtaDown   = np.zeros_like(eta)
    gradxEtaRight[1:-1,1:-1] = (eta[1:-1,  2:] - eta[1:-1,1:-1])/dx
    gradxEtaLeft[1:-1,1:-1]  = (eta[1:-1,1:-1] - eta[1:-1, :-2])/dx
    gradyEtaUp[1:-1,1:-1]    = (eta[2:,  1:-1] - eta[1:-1,1:-1])/dy
    gradyEtaDown[1:-1,1:-1]  = (eta[1:-1,1:-1] - eta[:-2, 1:-1])/dy

    normGradEta2 = np.zeros_like(eta)
    normGradPhi2 = np.zeros_like(spotential)
    
    """gradxEta[1:-1, 1:-1] = (eta[1:-1, 2:] - eta[1:-1, :-2])/(2*dx)
    gradyEta[1:-1, 1:-1] = (eta[2:, 1:-1] - eta[:-2, 1:-1])/(2*dy)
    normGradEta2[1:-1, 1:-1] = gradxEta[1:-1, 1:-1]**2 + gradyEta[1:-1, 1:-1]**2"""
    #normGradEta2[1:-1, 1:-1] = (gradxEtaRight*gradxEtaLeft + gradyEtaUp*gradyEtaDown)[1:-1,1:-1]
    normGradEta2[1:-1, 1:-1] = ((gradxEtaRight**2 + gradxEtaLeft**2)/2 + (gradyEtaUp**2 + gradyEtaDown**2)/2)[1:-1,1:-1]

    """gradxPhi[1:-1, 1:-1] = (spotential[1:-1, 2:] - spotential[1:-1, :-2])/(2*dx)
    gradyPhi[1:-1, 1:-1] = (spotential[2:, 1:-1] - spotential[:-2, 1:-1])/(2*dy)
    normGradPhi2[1:-1, 1:-1] = gradxPhi[1:-1, 1:-1]**2 + gradyPhi[1:-1, 1:-1]**2"""
    #normGradPhi2[1:-1, 1:-1] = ((gradxPhiRight**2 + gradxPhiLeft**2)/2 + (gradyPhiUp**2 + gradyPhiDown**2)/2)[1:-1,1:-1]
    normGradPhi2[1:-1, 1:-1] = ((gradxPhiRight**2 + gradxPhiLeft**2)/2 + (gradyPhiUp**2 + gradyPhiDown**2)/2)[1:-1,1:-1]
    
    newPot[1:-1, 1:-1] = (- gravity*eta[1:-1, 1:-1]  - 0.5/nu*normGradPhi2[1:-1, 1:-1]
        + 0.5*beta**2/nu*((gradxPhiRight[1:-1, 1:-1]*gradxEtaRight[1:-1, 1:-1] + gradxPhiLeft[1:-1, 1:-1]*gradxEtaLeft[1:-1, 1:-1])/2
                         +(gradyPhiUp[1:-1, 1:-1]*gradyEtaUp[1:-1, 1:-1]       + gradyPhiDown[1:-1, 1:-1]*gradyEtaDown[1:-1, 1:-1])/2
                         + dToNForPhi[1:-1, 1:-1]/beta**2)**2 / (1 + beta**2*normGradEta2[1:-1, 1:-1])
    ) #+ diffCoeff*((1/dx**2+1/dy**2)*spotential[1:-1, 1:-1] - spotential[1:-1, 2:]/dx**2 - spotential[1:-1, :-2]/dx**2
                                                           # - spotential[2:, 1:-1]/dy**2 - spotential[:-2, 1:-1]/dy**2)

    '''ravelledNewPot = np.ravel(newPot[1:-1, 1:-1])
    ravelledNewH = np.ravel(newH[1:-1, 1:-1])
    ravelledNewPot = spsolve(diffusionMatrix,ravelledNewPot)
    ravelledNewH = spsolve(diffusionMatrix,ravelledNewH)

    newPot[1:-1,1:-1] = np.reshape(ravelledNewPot, newshape = (n_y-2,n_x-2))
    newH[1:-1,1:-1] = np.reshape(ravelledNewH, newshape = (n_y-2,n_x-2))
    newPot = enforce_boundaries(newPot, 'h')
    newH = enforce_boundaries(newH, 'h')

    newH[1:-1,1:-1] = np.real(np.fft.ifft2(antiAlias*np.fft.fft2(newH[1:-1,1:-1])))
    newPot[1:-1,1:-1] = np.real(np.fft.ifft2(antiAlias*np.fft.fft2(newPot[1:-1,1:-1])))
    newH = enforce_boundaries(newH, 'h')
    newPot = enforce_boundaries(newPot, 'h')'''
    
    return newH, newPot

def iterate_shallow_water():
    # allocate arrays
    spotential, h = np.empty((n_y, n_x)), np.empty((n_y, n_x))
    spotentialTemp, htemp = np.empty((n_y, n_x)), np.empty((n_y, n_x))
    dspotential, dh = np.empty((n_y, n_x)), np.empty((n_y, n_x))
    dspotential_new, dh_new = np.empty((n_y, n_x)), np.empty((n_y, n_x))
    dspotential_new_stocha, dh_new_stocha = np.empty((n_y, n_x, nmodes)), np.empty((n_y, n_x, nmodes))
    spotential_stocha, spotential_stocha_temp = np.empty((n_y, n_x, nmodes)), np.empty((n_y, n_x, nmodes))

    # initial conditions
    h[...] = h0
    spotential[...] = spotential0

    # boundary values of h must not be used
    h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = np.nan
    h = enforce_boundaries(h, 'h')
    spotential = enforce_boundaries(spotential, 'h')

    first_step = True

    # time step equations
    while True:
        hc = np.pad(h[1:-1, 1:-1], 1, 'edge')
        hc = enforce_boundaries(hc, 'h')
        eta = np.pad(h[1:-1, 1:-1] - depth, 1, 'edge')
        eta = enforce_boundaries(eta, 'h')
    
        #martingale terms
        gradxPhiRight = np.zeros_like(spotential)
        gradxPhiLeft  = np.zeros_like(spotential)
        gradyPhiUp    = np.zeros_like(spotential)
        gradyPhiDown  = np.zeros_like(spotential)
        gradxPhiRight[1:-1,1:-1] = (spotential[1:-1,  2:] - spotential[1:-1,1:-1])/dx
        gradxPhiLeft[1:-1,1:-1]  = (spotential[1:-1,1:-1] - spotential[1:-1, :-2])/dx
        gradyPhiUp[1:-1,1:-1]    = (spotential[2:,  1:-1] - spotential[1:-1,1:-1])/dy
        gradyPhiDown[1:-1,1:-1]  = (spotential[1:-1,1:-1] - spotential[:-2, 1:-1])/dy

        gradxEtaRight = np.zeros_like(eta)
        gradxEtaLeft  = np.zeros_like(eta)
        gradyEtaUp    = np.zeros_like(eta)
        gradyEtaDown   = np.zeros_like(eta)
        gradxEtaRight[1:-1,1:-1] = (eta[1:-1,  2:] - eta[1:-1,1:-1])/dx
        gradxEtaLeft[1:-1,1:-1]  = (eta[1:-1,1:-1] - eta[1:-1, :-2])/dx
        gradyEtaUp[1:-1,1:-1]    = (eta[2:,  1:-1] - eta[1:-1,1:-1])/dy
        gradyEtaDown[1:-1,1:-1]  = (eta[1:-1,1:-1] - eta[:-2, 1:-1])/dy

        velocityX = (gradxPhiRight + gradxPhiLeft)/2
        velocityY = (gradyPhiUp + gradyPhiDown)/2
        velocityLeft, velocityRight = np.maximum(velocityX,0), np.minimum(velocityX,0)
        velocityDown, velocityUp  = np.maximum(velocityY,0), np.minimum(velocityY,0)

        normGradEta2 = np.zeros_like(eta)
        normGradPhi2 = np.zeros_like(spotential)
        normGradEta2[1:-1, 1:-1] = ((gradxEtaRight**2 + gradxEtaLeft**2)/2 + (gradyEtaUp**2 + gradyEtaDown**2)/2)[1:-1,1:-1]
        normGradPhi2[1:-1, 1:-1] = ((gradxPhiRight**2 + gradxPhiLeft**2)/2 + (gradyPhiUp**2 + gradyPhiDown**2)/2)[1:-1,1:-1]

        for k in range(nmodes):
            dh_new_stocha[1:-1, 1:-1, k] = (
                   dirichletToNeumannForEta(eta, phi[:, :, k])[1:-1, 1:-1]/beta**2/nu
                - (chix[1:-1,2:, k]*eta[1:-1,2:] - chix[1:-1,:-2, k]*eta[1:-1,:-2])/(2*dx)
                - (chiy[2:,1:-1, k]*eta[2:,1:-1] - chiy[:-2,1:-1, k]*eta[:-2,1:-1])/(2*dy)
                +  chiz[1:-1,1:-1,k]
            )
            dh_new_stocha[:,:,k] = enforce_boundaries(dh_new_stocha[:,:,k], 'h')

            dspotential_new_stocha[1:-1, 1:-1, k] = (
                (dirichletToNeumannForPhi(eta, spotential)[1:-1, 1:-1]/beta**2
                + (gradxPhiRight[1:-1, 1:-1]*gradxEtaRight[1:-1, 1:-1] + gradxPhiLeft[1:-1, 1:-1]*gradxEtaLeft[1:-1, 1:-1])/2
                + (gradyPhiUp[1:-1, 1:-1]*gradyEtaUp[1:-1, 1:-1] + gradyPhiDown[1:-1, 1:-1]*gradyEtaDown[1:-1, 1:-1])/2)
                / (1 + normGradEta2[1:-1, 1:-1]) * dh_new_stocha[1:-1, 1:-1, k]
            )
            dspotential_new_stocha[:, :, k] = enforce_boundaries(dspotential_new_stocha[:, :, k], 'h')
        
        #Euler-Heun method for stochastic terms
        dBt = np.random.normal(0, np.sqrt(dt), nmodes)
        #dBt = np.sqrt(dt)*truncnorm.rvs(-5, 5, size = nmodes)
        
        htemp[1:-1, 1:-1] = hc[1:-1, 1:-1]
        spotentialTemp[1:-1, 1:-1] = spotential[1:-1, 1:-1]
        htemp = enforce_boundaries(htemp, 'h')        
        spotentialTemp = enforce_boundaries(spotentialTemp, 'h')
        for k in range(nmodes):
            htemp[1:-1, 1:-1]  += dBt[k]*dh_new_stocha[1:-1, 1:-1, k]
            spotentialTemp[1:-1, 1:-1] += dBt[k]*dspotential_new_stocha[1:-1, 1:-1, k]
            htemp = enforce_boundaries(htemp, 'h')            
            spotentialTemp = enforce_boundaries(spotentialTemp, 'h')

        etatemp = np.pad(htemp[1:-1, 1:-1] - depth, 1, 'edge')
        etatemp = enforce_boundaries(etatemp, 'h')

        gradxPhiRight[1:-1,1:-1] = (spotentialTemp[1:-1,  2:] - spotentialTemp[1:-1,1:-1])/dx
        gradxPhiLeft[1:-1,1:-1]  = (spotentialTemp[1:-1,1:-1] - spotentialTemp[1:-1, :-2])/dx
        gradyPhiUp[1:-1,1:-1]    = (spotentialTemp[2:,  1:-1] - spotentialTemp[1:-1,1:-1])/dy
        gradyPhiDown[1:-1,1:-1]  = (spotentialTemp[1:-1,1:-1] - spotentialTemp[:-2, 1:-1])/dy

        gradxEtaRight[1:-1,1:-1] = (etatemp[1:-1,  2:] - etatemp[1:-1,1:-1])/dx
        gradxEtaLeft[1:-1,1:-1]  = (etatemp[1:-1,1:-1] - etatemp[1:-1, :-2])/dx
        gradyEtaUp[1:-1,1:-1]    = (etatemp[2:,  1:-1] - etatemp[1:-1,1:-1])/dy
        gradyEtaDown[1:-1,1:-1]  = (etatemp[1:-1,1:-1] - etatemp[:-2, 1:-1])/dy

        velocityX = (gradxPhiRight + gradxPhiLeft)/2
        velocityY = (gradyPhiUp + gradyPhiDown)/2
        velocityLeft, velocityRight = np.maximum(velocityX,0), np.minimum(velocityX,0)
        velocityDown, velocityUp  = np.maximum(velocityY,0), np.minimum(velocityY,0)

        normGradEta2[1:-1, 1:-1] = ((gradxEtaRight**2 + gradxEtaLeft**2)/2 + (gradyEtaUp**2 + gradyEtaDown**2)/2)[1:-1,1:-1]
        normGradPhi2[1:-1, 1:-1] = ((gradxPhiRight**2 + gradxPhiLeft**2)/2 + (gradyPhiUp**2 + gradyPhiDown**2)/2)[1:-1,1:-1]

        for k in range(nmodes):
            dh_new_stocha[1:-1, 1:-1, k] = (
                   dirichletToNeumannForEta(eta, phi[:, :, k])[1:-1, 1:-1]/beta**2/nu
                - (chix[1:-1,2:, k]*eta[1:-1,2:] - chix[1:-1,:-2, k]*eta[1:-1,:-2])/(2*dx)
                - (chiy[2:,1:-1, k]*eta[2:,1:-1] - chiy[:-2,1:-1, k]*eta[:-2,1:-1])/(2*dy)
                +  chiz[1:-1,1:-1,k]
            )
            dh_new_stocha[:,:,k] = enforce_boundaries(dh_new_stocha[:,:,k], 'h')

            dspotential_new_stocha[1:-1, 1:-1, k] = (
                (dirichletToNeumannForPhi(eta, spotential)[1:-1, 1:-1] 
                + (gradxPhiRight[1:-1, 1:-1]*gradxEtaRight[1:-1, 1:-1] + gradxPhiLeft[1:-1, 1:-1]*gradxEtaLeft[1:-1, 1:-1])
                + (gradyPhiUp[1:-1, 1:-1]*gradyEtaUp[1:-1, 1:-1] + gradyPhiDown[1:-1, 1:-1]*gradyEtaDown[1:-1, 1:-1]))
                / (1 + normGradEta2[1:-1, 1:-1]) * dh_new_stocha[1:-1, 1:-1, k]
            )
            dspotential_new_stocha[:, :, k] = enforce_boundaries(dspotential_new_stocha[:, :, k], 'h')
        
        dh_new_stocha *= 0.5
        dspotential_new_stocha *= 0.5

        timeIncrementMethod = 'RK4'

        #RK4 method for time increment
        if timeIncrementMethod == 'RK4':
            deta0, dSpotential0 = oneStepIncr(eta, spotential)
            eta_inter1, spotential_inter1 = eta+deta0*dt/2, spotential + dSpotential0*dt/2
            deta_inter1, dSpotential_inter1 = oneStepIncr(eta_inter1, spotential_inter1)
            eta_inter2, spotential_inter2 = eta+deta_inter1*dt/2, spotential + dSpotential_inter1*dt/2
            deta_inter2, dSpotential_inter2 = oneStepIncr(eta_inter2, spotential_inter2)
            eta_inter3, spotential_inter3 = eta+deta_inter2*dt/2, spotential + dSpotential_inter2*dt/2
            deta_inter3, dSpotential_inter3 = oneStepIncr(eta_inter3, spotential_inter3)

            #simple RK4
            dspotential_new[1:-1, 1:-1] = 1/6*( dSpotential_inter3 + 2*dSpotential_inter2 + 2*dSpotential_inter1 + dSpotential0)[1:-1, 1:-1]
            dh_new[1:-1, 1:-1] = 1/6*( deta_inter3 + 2*deta_inter2 + 2*deta_inter1 + deta0)[1:-1, 1:-1]
            spotential[1:-1, 1:-1] += dt*dspotential_new[1:-1, 1:-1]
            h[1:-1, 1:-1] += dt*dh_new[1:-1, 1:-1]
            for k in range(nmodes):
                spotential[1:-1, 1:-1] += dBt[k]*dspotential_new_stocha[1:-1, 1:-1, k]
                h[1:-1, 1:-1] += dBt[k]*dh_new_stocha[1:-1, 1:-1, k]

            #RK4 + adams-bashforth
            '''if first_step:
                dspotential_new[1:-1, 1:-1] = 1/6*( dSpotential_inter3 + 2*dSpotential_inter2 + 2*dSpotential_inter1 + dSpotential0)[1:-1, 1:-1]
                dh_new[1:-1, 1:-1] = 1/6*( deta_inter3 + 2*deta_inter2 + 2*deta_inter1 + deta0)[1:-1, 1:-1]
                spotential[1:-1, 1:-1] += dt*dspotential_new[1:-1, 1:-1]
                h[1:-1, 1:-1] += dt*dh_new[1:-1, 1:-1]
                for k in range(nmodes):
                    spotential[1:-1, 1:-1] += dBt[k]*dspotential_new_stocha[1:-1, 1:-1, k]
                    h[1:-1, 1:-1] += dBt[k]*dh_new_stocha[1:-1, 1:-1, k]
                first_step = False
            else:
                dspotential_new[1:-1, 1:-1] = 1/6*( dSpotential_inter3 + 2*dSpotential_inter2 + 2*dSpotential_inter1 + dSpotential0)[1:-1, 1:-1]
                dh_new[1:-1, 1:-1] = 1/6*( deta_inter3 + 2*deta_inter2 + 2*deta_inter1 + deta0)[1:-1, 1:-1]
                spotential[1:-1, 1:-1] += dt*(adams_bashforth_a*dspotential_new[1:-1, 1:-1]+adams_bashforth_b*dspotential[1:-1, 1:-1])
                h[1:-1, 1:-1] += dt*(adams_bashforth_a*dh_new[1:-1, 1:-1]+adams_bashforth_b*dh[1:-1, 1:-1])
                for k in range(nmodes):
                    spotential[1:-1, 1:-1] += dBt[k]*dspotential_new_stocha[1:-1, 1:-1, k]
                    h[1:-1, 1:-1] += dBt[k]*dh_new_stocha[1:-1, 1:-1, k]
        '''
        #fix point method for time increment
        '''elif timeIncrementMethod == 'fix point':
            spotentialTemp = np.pad(spotential[1:-1, 1:-1], 1, 'edge')
            spotentialTemp = enforce_boundaries(spotentialTemp, 'h') 
            hTemp = np.pad(h[1:-1, 1:-1], 1, 'edge')
            hTemp = enforce_boundaries(hTemp, 'h')

            hTemp2, spotentialTemp2 = oneStepIncr(hTemp - depth, spotential)
            error = np.max(abs(hTemp2 - hTemp) + abs(spotentialTemp2 - spotentialTemp))

            countIter=10
            for _ in range(countIter):
            #while error > tol:
                hTemp = hTemp2.copy()
                spotentialTemp = spotentialTemp2.copy()

                hTemp2, spotentialTemp2 = oneStepIncr(hTemp - depth, spotentialTemp)
                error = np.max(abs(hTemp2 - hTemp) + abs(spotentialTemp2 - spotentialTemp))
                #print(error)
            dh_new, dspotential_new = oneStepIncr(hTemp - depth, spotentialTemp)
            spotential[1:-1, 1:-1] += dt*dspotential_new[1:-1, 1:-1]
            h[1:-1, 1:-1] += dt*dh_new[1:-1, 1:-1]
            #print(np.max(abs(dh_new)), np.max(abs(dspotential_new)))'''
            
        spotential = enforce_boundaries(spotential, 'h')
        h = enforce_boundaries(h, 'h')
        #h = np.maximum(h, depth-amplitude/2)
        # rotate quantities
        dspotential[...] = dspotential_new
        dh[...] = dh_new

        yield h, spotential


#--------------------------------------------------------------------------------------------------------------
#-------------------------------------------GENERATE AND SAVE PLOTS--------------------------------------------
#--------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    if make_plots:
        fig, ax = prepare_plot()
    model = iterate_shallow_water()
    count=0
    totalMass = []
    totalMomX = []
    totalMomY = []
    totalEnerg = []

    if save_data:
        spec = [n_x, n_y, dx, dy]
        export_to_csv(spec, path_to_store+'//spec.csv')

    for iteration, (h, spotential) in enumerate(model):
        if iteration % plot_every == 0:
            t = iteration * dt
            #print('Current time: ', t,'s')
            
            totalMass.append(np.sum(h[1:-1,1:-1])*dx/l_x*dy/l_y)
            '''totalMomX.append(dx/l_x*dy/l_y*np.sum(hu[1:-1,1:-1]))
            totalMomY.append(dx/l_x*dy/l_y*np.sum(hv[1:-1,1:-1]))
            
            local_hu = np.zeros_like(hu)
            local_hv = np.zeros_like(hv)
            local_hu[1:-1, 1:-1] = 0.5 * (hu[1:-1, 1:-1] + hu[:-2, 1:-1])
            local_hv[1:-1, 1:-1] = 0.5 * (hv[1:-1, 1:-1] + hv[1:-1, :-2])
            local_hu = enforce_boundaries(local_hu, 'u')
            local_hv = enforce_boundaries(local_hv, 'v')
            totalEnerg.append(np.sum((local_hu**2/h)[1:-1,1:-1]/2 + (local_hv**2/h)[1:-1,1:-1]/2 + (gravity*h**2)[1:-1,1:-1]/2)*dx/l_x*dy/l_y)
            '''
            if make_plots:
                update_plot(t, h, spotential, ax, count)

            if save_data:
                export_to_csv(h, path_to_store+'height//'+str(count)+'.csv')
                export_to_csv(spotential, path_to_store+'potential//'+str(count)+'.csv')
            count+=1

            if iteration > itmax:
                break

        # stop if user closes plot window
        if make_plots and not plt.fignum_exists(fig.number):
            break

    if make_plots:
        fig, ax = plt.subplots(2, 1, figsize=(12, 5))
        timespace = np.linspace(0, count*dt, count)
        cmass = ax[0].plot(timespace, totalMass/totalMass[0] -1, color='darkmagenta')
        ax[0].set_ylabel('Mass error', color='darkmagenta')
        ax[0].tick_params(axis='y', labelcolor='darkmagenta')
        ax[0].legend(['Mass'], loc='upper left')

        '''ax_bis = ax[0].twinx()
        cmomx = ax_bis.plot(timespace, np.array(totalMomX), color='royalblue')
        cmomy = ax_bis.plot(timespace, np.array(totalMomY), color='midnightblue')
        ax_bis.set_ylabel('Momentum error', color='blue')
        ax_bis.tick_params(axis='y', labelcolor='blue')
        ax_bis.legend(['X Momentum', 'Y Momentum'], loc='lower left')
        
        cenerg = ax[1].plot(timespace, totalEnerg/totalEnerg[0] -1, color='red')
        ax[1].set_ylabel('Energy error', color='red')
        ax[1].tick_params(axis='y', labelcolor='red')
        ax[1].legend(['Energy'], loc='best')

        energ_amplitude_stat = max(totalEnerg[count//2:]/totalEnerg[0]) - min(totalEnerg[count//2:]/totalEnerg[0])
        print("Maximal amplitude of energy variation over the second half of the simulation (to be interpreted as stationary regime if the simulation in long enough):",
                                energ_amplitude_stat)'
        '''
        plt.show()

    if makegif:
        frames = np.stack([iio.imread(f"images/{i}.jpg") for i in range(count)] + [iio.imread(f"images/{i}.jpg") for i in range(count-2, 0, -1)], axis = 0)
        iio.mimwrite('test1000.gif', frames, loop=0)
