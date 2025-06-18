
import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
import time 

from fipy import numerix as nx
import pandas as pd
import os

from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse.linalg import spsolve

from scipy.stats import truncnorm
from numba import njit

import warnings


#--------------------------------------------------------------------------------------------------------------
#---------------------------------------------SIMULATION PARAMETERS-----------------------------------------------
#--------------------------------------------------------------------------------------------------------------

# maximum number of timesteps
itmax = float('infinity')

# grid setup
n_x = 31
dx = 1e3
l_x = n_x * dx

n_y = 31
dy = 1e3
l_y = n_y * dy

l_ref = np.sqrt(l_x * l_y)

x, y = (
    np.arange(n_x) * dx,
    np.arange(n_y) * dy
)
Y, X = np.meshgrid(y, x, indexing='ij')

# physical parameters
gravity = 9.81
depth = 100.
coriolis_f = 0 #2e-4
coriolis_beta = 0 #2e-11
coriolis_param = coriolis_f + Y * coriolis_beta

phase_speed = np.sqrt(gravity * depth)
rossby_radius = np.sqrt(gravity * depth) / coriolis_param.mean()

# timestep
dt = 0.05*min(dx, dy) / np.sqrt(depth)
print( "\n timestep", dt, "\n")

# Standard scaling parameters
eps = 0.1
beta = 1 #depth/np.sqrt(l_x*l_y)
nu = 1 #1/(1+beta) #scaling parameter (cf Vincent Duchêne)
amplitude = eps*depth
#epsilon only impacts the wave height in this model --> maybe write it with adimensioned variables

# type of boundary conditions parameters (periodic or Dirichlet)
periodic_boundary_x = True
periodic_boundary_y = True

stochasticPotentialAtFixedHeight = True

#--------------------------------------------------------------------------------------------------------------
#---------------------------------------------SOLVERS PARAMETERS-----------------------------------------------
#--------------------------------------------------------------------------------------------------------------

# Poisson problem setup (Fourier)
n_z = 11
kx = np.fft.fftfreq(n_x, dx)
ky = np.fft.fftfreq(n_y, dy)
Kx,Ky = np.meshgrid(kx,ky,indexing='xy')

kxBIS = np.concatenate((kx[:kx.size//2],kx[(kx.size//2+2):]), axis=None)
kyBIS = np.concatenate((ky[:ky.size//2],ky[(ky.size//2+2):]), axis=None)
KxBIS, KyBIS = np.meshgrid(kxBIS, kyBIS,indexing='xy')
tolPoisson = 1e-1

# Approximation of the water waves system
#wwApprox = "full system" #true water waves system
wwApprox = "Shallow-water-order0" #nth order shallow water (zeroth order = Saint-Venant, first order contains Boussinesq and SGN)
#wwApprox = "Small-waves-order2" #nth order small amplitude (zeroth order = Whitham-Boussinesq)

if wwApprox == "Shallow-water-order0":
    warnings.simplefilter("ignore")

#Approximations on the noise
smallNoiseAssumption = False

# Method to be used for the Laplace problem
#solveLaplaceMethod = "small wave amplitude"
#solveLaplaceMethod = "small wave slope"
solveLaplaceMethod = "full model"

# Time increment method
timeIncrementMethod = 'RK4'
#timeIncrementMethod = 'Adams-Bashforth'

# Parameters of the Adams-Bashforth method
if timeIncrementMethod == 'Adams-Bashforth':
    adams_bashforth_a = 1.5
    adams_bashforth_b = -0.5

# Diffusion matrix for implicit temporal scheme
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


#--------------------------------------------------------------------------------------------------------------
#------------------------------------------SAVE & PLOT PARAMETERS----------------------------------------------
#--------------------------------------------------------------------------------------------------------------

# Save parameters
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

# Plot parameters
plot_range = 0.5*amplitude
plot_every = 1

max_quivers = 41

make_plots = True
save_plots = False
makegif = False

plotStochasticParameters = False

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
    + amplitude*np.exp(-1e4*((X/l_x - 1/2 + 1/n_x)**2 + (Y/l_y - 1/2 + 1/n_y)**2)**2)
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
    ups = 0 #2e2
    wavenumber = 2*np.pi

    upsRot = 1e3
    coeffChiZ = 0 #0.01
    wavenumberRot = 2*np.pi

    coeff=-3
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

    coeff=-3
    l_waveX = (l_x + coeff*dx)
    l_waveY = (l_y + coeff*dy)
    l_wave = (l_waveX*l_waveY)**0.5

    chix1 = - np.sqrt(upsRot)*l_wave/l_ref*np.cos(wavenumberRot*(X/l_waveX+Y/l_waveY))
    chix2 =   np.sqrt(upsRot)*l_wave/l_ref*np.cos(wavenumberRot*(X/l_waveX-Y/l_waveY))
    chix3 = - np.sqrt(upsRot)*l_wave/l_ref*np.sin(wavenumberRot*(X/l_waveX+Y/l_waveY))
    chix4 =   np.sqrt(upsRot)*l_wave/l_ref*np.sin(wavenumberRot*(X/l_waveX-Y/l_waveY))

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
        't=%5.1fs, c=%5.1f m/s '
        % (t, phase_speed)
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
div_us[1:-1,1:-1] = (us[1:-1,2:] - us[1:-1,:-2])/dx/2 + (vs[2:,1:-1] - vs[:-2,1:-1])/dy/2
div_us = enforce_boundaries(div_us, 'h')


if nmodes>0:
    maxNoiseNorm = np.zeros_like(div_us)
    maxNoiseDivg = np.zeros_like(div_us)
    ISDNorm = np.zeros_like(div_us)

    maxNoiseNorm[1:-1,1:-1] = np.max(np.sqrt(vectx[1:-1,1:-1,:]**2 + vecty[1:-1,1:-1,:]**2), axis=-1)
    maxNoiseDivg[1:-1,1:-1] = np.max(abs((chix[1:-1,2:,:] - chix[1:-1,:-2,:])/dx/2
                                        +(chiy[2:,1:-1,:] - chiy[:-2,1:-1,:])/dy/2), axis=-1)
    ISDNorm[1:-1,1:-1] = np.sqrt(us[1:-1,1:-1]**2 + vs[1:-1,1:-1]**2)

    maxNoiseNorm = enforce_boundaries(maxNoiseNorm, 'h')
    maxNoiseDivg = enforce_boundaries(maxNoiseDivg, 'h')
    ISDNorm = enforce_boundaries(ISDNorm, 'h')

if plotStochasticParameters:
    fig, ax = plt.subplots(2, 2, figsize=(14, 7))

    cs = ax[0,0].pcolormesh(
        x[1:-1] / 1e3,
        y[1:-1] / 1e3,
        maxNoiseNorm[1:-1,1:-1],
        vmin=-1e2, vmax=1e2, cmap='RdBu_r'
    )
    ax[0,0].set_title('Noise norm.')
    fig.colorbar(cs, label='Max of the R2 norm', ax=ax[0,0])

    cs = ax[0,1].pcolormesh(
        x[1:-1] / 1e3,
        y[1:-1] / 1e3,
        maxNoiseDivg[1:-1,1:-1],
        vmin=-1e-5, vmax=1e-5, cmap='RdBu_r'
    )
    ax[0,1].set_title('Noise divergence.')
    fig.colorbar(cs, label='Max of the divergence', ax=ax[0,1])

    cs = ax[1,0].pcolormesh(
        x[1:-1] / 1e3,
        y[1:-1] / 1e3,
        ISDNorm[1:-1,1:-1],
        vmin=-1e-7, vmax=1e-7, cmap='RdBu_r'
    )
    ax[1,0].set_title('Itô-Stokes drift norm.')
    fig.colorbar(cs, label='ISD norm', ax=ax[1,0])

    cs = ax[1,1].pcolormesh(
        x[1:-1] / 1e3,
        y[1:-1] / 1e3,
        div_us[1:-1,1:-1],
        vmin=-1e-7, vmax=1e-7, cmap='RdBu_r'
    )
    ax[1,1].set_title('Itô-Stokes drift divergence.')
    fig.colorbar(cs, label='ISD div', ax=ax[1,1])

if nmodes > 0:
    maxNormUs = np.max(np.sqrt(us[1:-1,1:-1]**2 + vs[1:-1,1:-1]**2))
    maxDivUs = np.max(abs((us[1:-1,1:-1] - us[1:-1,:-2])/dx + (vs[1:-1,1:-1] - vs[:-2,1:-1])/dy))
    maxNormChi = np.max(np.sqrt(chix[1:-1,1:-1,:]**2 + chiy[1:-1,1:-1,:]**2))
    maxDivChi = np.max(abs((chix[1:-1,2:,:] - chix[1:-1,:-2,:])/dx/2 + (chiy[2:,1:-1,:] - chiy[:-2,1:-1,:])/dy/2))

    print("Maximal value of norm us: ", maxNormUs)
    print("Maximal value of div us: ", maxDivUs)
    print("Maximal value of norm chi: ", maxNormChi)
    print("Maximal value of div chi: ", maxDivChi)
    print("Ratio of maximal values: ", maxDivChi/maxNormChi)
    print("Periodicity default of chix: ", np.max(abs(chix[1,:,:] - chix[-2,:,:]) + abs(chix[:,1,:] - chix[:,-2,:])))
    print("Periodicity default of chiy: ", np.max(abs(chiy[1,:,:] - chiy[-2,:,:]) + abs(chiy[:,1,:] - chiy[:,-2,:])))
    print("Periodicity default of phi: ", np.max(abs(phi[1,:,:] - phi[-2,:,:]) + abs(phi[:,1,:] - phi[:,-2,:])))

def export_to_csv(field, name):
    df = pd.DataFrame(field)
    df.to_csv(name)

#--------------------------------------------------------------------------------------------------------------
#-----------------------------------------DIRICHLET TO NEUMANN OPERATOR----------------------------------------------
#--------------------------------------------------------------------------------------------------------------

@njit
def index(j,i,k):
    return (j%(n_y-2))*(n_x-2)*n_z + (i%(n_x-2))*n_z + k

@njit
def assignCoeff(Phi, eta, method = solveLaplaceMethod):
    trueDepth = (depth + eta)[1:-1,1:-1]
    dz= trueDepth/n_z
    n=(n_x-2)*(n_y-2)*n_z
    b = np.zeros(n)
    A = np.zeros((n,n))

    gxEta = (eta[1:-1, 2:] - eta[1:-1, :-2])/dx/2
    gyEta = (eta[2:, 1:-1] - eta[:-2, 1:-1])/dy/2
    lapEta = ((eta[1:-1, 2:] - 2*eta[1:-1, 1:-1] + eta[1:-1, :-2])/dx**2
            + (eta[2: ,1:-1] - 2*eta[1:-1, 1:-1] + eta[:-2, 1:-1])/dy**2)
    gEta2 = gxEta**2 + gyEta**2

    for i in range(0,n_x-2):
        for j in range(0,n_y-2):
        #Scaled Laplace operator
            #Upper boundary (Dirichlet)
            A[index(j,i,n_z-1), index(j,i,n_z-1)] += 2/dx**2 + 2/dy**2 + 2/dz[j,i]**2*(1 + (method == 'full model')*(trueDepth*gEta2)[j,i]**2)
            A[index(j,i,n_z-1), index(j,i,n_z-2)] -= 1/dz[j,i]**2*(1 + (method == 'full model')*(trueDepth*gEta2)[j,i]**2)
            A[index(j,i,n_z-1), index(j,i+1,n_z-1)] -= 1/dx**2
            A[index(j,i,n_z-1), index(j,i-1,n_z-1)] -= 1/dx**2
            A[index(j,i,n_z-1), index(j+1,i,n_z-1)] -= 1/dy**2
            A[index(j,i,n_z-1), index(j-1,i,n_z-1)] -= 1/dy**2
            #Lower boundary (Neumann)
            A[index(j,i,0), index(j,i,0)] += 2/dx**2 + 2/dy**2 + 2/dz[j,i]**2
            A[index(j,i,0), index(j,i,1)] -= 2/dz[j,i]**2
            A[index(j,i,0), index(j,i+1,0)] -= 1/dx**2
            A[index(j,i,0), index(j,i-1,0)] -= 1/dx**2
            A[index(j,i,0), index(j+1,i,0)] -= 1/dy**2
            A[index(j,i,0), index(j-1,i,0)] -= 1/dy**2
            #Internal points
            for k in range(1,n_z-1):
                A[index(j,i,k), index(j, i, k)] += 2/dx**2 + 2/dy**2 + 2/dz[j,i]**2*(1 + (method == 'full model')*(k*dz*gEta2)[j,i]**2)
                A[index(j,i,k), index(j,i+1,k)] -= 1/dx**2
                A[index(j,i,k), index(j,i-1,k)] -= 1/dx**2
                A[index(j,i,k), index(j+1,i,k)] -= 1/dy**2
                A[index(j,i,k), index(j-1,i,k)] -= 1/dy**2
                A[index(j,i,k), index(j,i,k+1)] -= 1/dz[j,i]**2*(1 + (method == 'full model')*(k*dz*gEta2)[j,i]**2)
                A[index(j,i,k), index(j,i,k-1)] -= 1/dz[j,i]**2*(1 + (method == 'full model')*(k*dz*gEta2)[j,i]**2)
            #Dirichlet boundary condition on b
            b[index(j,i,n_z-1)] = Phi[j+1,i+1]/dz[j,i]**2*(1 + (method == 'full model')*(trueDepth*gEta2)[j,i]**2)
            
        #Extra terms in the full resolution
            if method == 'full model':     
            #Term proportional to d_z nabla_h phi
                #Upper boundary
                A[index(j,i,n_z-1), index(j,i-1,n_z-2)] += 2*(k*dz/trueDepth**2)[j,i]*gxEta[j,i]/4/dx/dz[j,i]
                A[index(j,i,n_z-1), index(j,i+1,n_z-2)] -= 2*(k*dz/trueDepth**2)[j,i]*gxEta[j,i]/4/dx/dz[j,i]
                
                A[index(j,i,n_z-1), index(j-1,i,n_z-2)] += 2*(k*dz/trueDepth**2)[j,i]*gyEta[j,i]/4/dx/dz[j,i]
                A[index(j,i,n_z-1), index(j+1,i,n_z-2)] -= 2*(k*dz/trueDepth**2)[j,i]*gyEta[j,i]/4/dx/dz[j,i]
                #Internal points
                for k in range(1,n_z-1):
                    A[index(j,i,k), index(j,i+1,k+1)] += 2*(k*dz/trueDepth**2)[j,i]*gxEta[j,i]/4/dx/dz[j,i]
                    A[index(j,i,k), index(j,i-1,k-1)] += 2*(k*dz/trueDepth**2)[j,i]*gxEta[j,i]/4/dx/dz[j,i]
                    A[index(j,i,k), index(j,i+1,k-1)] -= 2*(k*dz/trueDepth**2)[j,i]*gxEta[j,i]/4/dx/dz[j,i]
                    A[index(j,i,k), index(j,i-1,k+1)] -= 2*(k*dz/trueDepth**2)[j,i]*gxEta[j,i]/4/dx/dz[j,i]
                    
                    A[index(j,i,k), index(j+1,i,k+1)] += 2*(k*dz/trueDepth**2)[j,i]*gyEta[j,i]/4/dx/dz[j,i]
                    A[index(j,i,k), index(j-1,i,k-1)] += 2*(k*dz/trueDepth**2)[j,i]*gyEta[j,i]/4/dx/dz[j,i]
                    A[index(j,i,k), index(j+1,i,k-1)] -= 2*(k*dz/trueDepth**2)[j,i]*gyEta[j,i]/4/dx/dz[j,i]
                    A[index(j,i,k), index(j-1,i,k+1)] -= 2*(k*dz/trueDepth**2)[j,i]*gyEta[j,i]/4/dx/dz[j,i]
                #Dirichlet boundary condition on b
                b[index(j,i,n_z-1)] -= ((Phi[j,i+1] - Phi[j,i-1])*2*(k*dz/trueDepth**2)[j,i]*gxEta[j,i]/4/dx/dz[j,i]
                                        +(Phi[j+1,i] - Phi[j-1,i])*2*(k*dz/trueDepth**2)[j,i]*gyEta[j,i]/4/dx/dz[j,i])
            
            #Term proportional to d_z phi
                #Upper boundary
                A[index(j,i,n_z-1), index(j,i,n_z-2)] -= (k*dz/trueDepth**2*lapEta)[j,i]/2/dz[j,i] - 2*(k*dz/trueDepth**3*gEta2)[j,i]/2/dz[j,i]
                #Internal points
                for k in range(1,n_z-1):
                    A[index(j,i,k), index(j,i,k+1)] += (k*dz/trueDepth**2*lapEta)[j,i]/2/dz[j,i] - 2*(k*dz/trueDepth**3*gEta2)[j,i]/2/dz[j,i]
                    A[index(j,i,k), index(j,i,k-1)] -= (k*dz/trueDepth**2*lapEta)[j,i]/2/dz[j,i] - 2*(k*dz/trueDepth**3*gEta2)[j,i]/2/dz[j,i]
                #Dirichlet boundary condition on b
                b[index(j,i,n_z-1)] -= ((k*dz/trueDepth**2*lapEta)[j,i]/2/dz[j,i]
                                    - 2*(k*dz/trueDepth**3*gEta2 )[j,i]/2/dz[j,i])*Phi[j,i]
    return A,b

def solveLaplace(Phi, eta, method=solveLaplaceMethod): # Solve the Laplace equation
    fullPotentialInternal = np.zeros((n_y-2,n_x-2,n_z))
    fullPotential = np.zeros((n_y, n_x, n_z))

    if method == 'small wave amplitude':
        dz= depth/n_z
        fftPhi = np.fft.fft2(Phi[1:-1,1:-1])
        fftFullPotentialInternal = np.zeros((n_y-2,n_x-2,n_z), dtype='complex')
        b = np.zeros(n_z, dtype='complex')
        for i in range(0,n_x-2):
            for j in range(0,n_y-2):
                A = (1/dz**2)*(2*np.eye(n_z) - np.eye(n_z, k=1) - np.eye(n_z, k=-1))
                A[0,0] -= 1/dz**2
                A += (kx[i]**2 + ky[j]**2)*np.eye(n_z)
                #A = csr_matrix(A)
                b[-1] = fftPhi[j,i]/dz**2
                fftFullPotentialInternal[j,i,:] = np.linalg.solve(A,b)
                #fftFullPotentialInternal[j,i,:] = spsolve(A,b)

        for k in range(n_z):
            fullPotentialInternal[:,:,k] = np.real(np.fft.ifft2(fftFullPotentialInternal[:,:,k])) #np.real(np.fft.ifft2(antiAlias*fftFullPotentialInternal[:,:,k])) #
        
        for k in range(n_z):
            fullPotential[:,:,k] = np.pad(fullPotentialInternal[:,:,k], 1, 'edge')
            fullPotential[:,:,k] = enforce_boundaries(fullPotential[:,:,k], 'h')
    
    if method == 'small wave slope' or method == 'full model':
        #t0=time.time()
        A,b = assignCoeff(Phi, eta)
        #t1=time.time()
        A = csr_matrix(A)
        #t2=time.time()
        fullPotentialInternal = np.reshape(spsolve(A,b), shape=(n_y-2,n_x-2,n_z))
        #t3=time.time()

        #print("Time to fill matrices:", t1-t0)
        #print("Time to make CSR matrix:", t2-t1)
        #print("Time to invert linear system:", t3-t2)
        #fullPotentialInternal = np.transpose(fullPotentialInternal, (1, 0, 2))
        #print(fullPotentialInternal.shape)
        fullPotential = np.zeros((n_y, n_x, n_z))
        for k in range(n_z):
            fullPotential[:,:,k] = np.pad(fullPotentialInternal[:,:,k], 1, 'edge')
            fullPotential[:,:,k] = enforce_boundaries(fullPotential[:,:,k], 'h')
    return fullPotential

def dirichletToNeumann(eta, Phi, method = wwApprox): # Compute Dirichlet to Neumann from the solution of the Poisson equation
    
    output = np.zeros_like(Phi, dtype='complex')

    if method == "full system":
        fullPotential = solveLaplace(Phi, eta)
        if solveLaplaceMethod == "simpleLaplace":
            dz = depth/n_z*np.ones_like(eta)
        else:
            dz = (depth+eta)/n_z
        intGxFullPotential = np.zeros_like(Phi)
        intGyFullPotential = np.zeros_like(Phi)
        intGxFullPotential[1:-1,1:-1] = np.sum((fullPotential[1:-1,2:,:] - fullPotential[1:-1,1:-1,:])/dx, axis = 2)*dz[1:-1,1:-1]
        intGyFullPotential[1:-1,1:-1] = np.sum((fullPotential[2:,1:-1,:] - fullPotential[1:-1,1:-1,:])/dy, axis = 2)*dz[1:-1,1:-1]
        intGxFullPotential[1:-1,1:-1] += (Phi[1:-1,2:] - Phi[1:-1,1:-1])/dx*dz[1:-1,1:-1]
        intGyFullPotential[1:-1,1:-1] += (Phi[2:,1:-1] - Phi[1:-1,1:-1])/dy*dz[1:-1,1:-1]
        
        intGxFullPotential = enforce_boundaries(intGxFullPotential, 'h')
        intGyFullPotential = enforce_boundaries(intGyFullPotential, 'h')

        output[1:-1,1:-1] = -((intGxFullPotential[1:-1,1:-1] - intGxFullPotential[1:-1,:-2])/dx
                            +(intGyFullPotential[1:-1,1:-1] - intGyFullPotential[:-2,1:-1])/dy)
            
    else:
        fftPhi = np.fft.fft2(Phi[1:-1,1:-1])
        h = depth + eta

        if method[0:13] == "Shallow-water":
            methodOrder =float(method[-1])
            output[1:-1,1:-1] =  (np.fft.ifft2(KxBIS * np.fft.fft2(h[1:-1,1:-1] * np.fft.ifft2(KxBIS * fftPhi)))
                                + np.fft.ifft2(KyBIS * np.fft.fft2(h[1:-1,1:-1] * np.fft.ifft2(KyBIS * fftPhi))))
            if methodOrder >= 1:
                operatorHT = -1/3*(np.fft.ifft2(KxBIS * np.fft.fft2(h[1:-1,1:-1]**3 * np.fft.ifft2(KxBIS * fftPhi)))
                                 + np.fft.ifft2(KyBIS * np.fft.fft2(h[1:-1,1:-1]**3 * np.fft.ifft2(KyBIS * fftPhi))))
                output[1:-1,1:-1] -= (np.fft.ifft2(KxBIS * np.fft.fft2(operatorHT * np.fft.ifft2(KxBIS * fftPhi)))
                                    + np.fft.ifft2(KyBIS * np.fft.fft2(operatorHT * np.fft.ifft2(KyBIS * fftPhi))))
            if methodOrder >= 2:
                warnings.warn("Shallow water method of order higher than 1 not implemented yet. Will fall back to order 1.")
    
        elif method[0:11] == "Small-waves":
            methodOrder = float(method[-1])
            def G0(fftField):
                normK = np.sqrt(KxBIS**2 + KyBIS**2)
                return np.fft.ifft2( normK * np.tanh(normK*depth) * fftField)
            zerothOrderTerm = G0(fftPhi)
            output[1:-1,1:-1] = zerothOrderTerm 
            if methodOrder >= 1:
                G0G0 = G0(eta[1:-1,1:-1]*zerothOrderTerm)
                output[1:-1,1:-1] -= G0G0 + (np.fft.ifft2(KxBIS * np.fft.fft2(eta[1:-1,1:-1] * np.fft.ifft2(KxBIS * fftPhi)))
                               +  np.fft.ifft2(KyBIS * np.fft.fft2(eta[1:-1,1:-1] * np.fft.ifft2(KyBIS * fftPhi))))
            if methodOrder >= 2:
                output[1:-1,1:-1] += G0(eta[1:-1,1:-1]*G0G0) + 0.5 * G0(eta[1:-1,1:-1]**2 *np.fft.ifft2((KxBIS**2+KyBIS**2)*fftPhi))
                output[1:-1,1:-1] += 0.5 * np.fft.ifft2((KxBIS**2+KxBIS**2)*np.fft.fft2(eta[1:-1,1:-1]**2 *zerothOrderTerm))
            if methodOrder >= 3:
                warnings.warn("Small waves method of order higher than 2 not implemented yet. Will fall back to order 2.")
    
    output = np.real(output)
    output = enforce_boundaries(output, 'h')
    return output


#----------------------------------------TEST OF THE LAPLACE SOLVER----------------------------------------

if solveLaplaceMethod == 'simpleLaplace': #True: #
    z = np.arange(n_z) * depth/n_z
    Y2, X2, Z2 = np.meshgrid(y, x, z, indexing='ij')
    alpha=2
    etaForTest = np.zeros((n_y, n_x))
    PhiForTest = np.sin(alpha*np.pi*X/l_x)*np.sin(alpha*np.pi*Y/l_y)

    solForTest = solveLaplace(PhiForTest, etaForTest)
    solReference = (np.cosh(alpha*np.sqrt(1/l_x**2 + 1/l_y**2)*np.pi*Z2)/np.cosh(alpha*np.sqrt(1/l_x**2 + 1/l_y**2)*np.pi*depth)
                        *np.sin(alpha*np.pi*X2/l_x)*np.sin(alpha*np.pi*Y2/l_y))
        
    err = np.max(abs(solForTest - solReference)[1:-1,1:-1,:])
    if err >=tolPoisson:
        plt.plot(X2[6,1:-1,-1],solForTest[6,1:-1,-1])
        plt.plot(X2[6,1:-1,-1],solReference[6,1:-1,-1])
        plt.show()
        raise ValueError("Laplace equation isn't solved properly.")
    else:
        print('Laplace equation solved OK.')

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------INCREMENT FUNCTIONS-----------------------------------------------
#--------------------------------------------------------------------------------------------------------------

def oneStepIncr(eta, spotential):
    newH = np.zeros_like(eta)
    newPot = np.zeros_like(spotential)

    dToN = dirichletToNeumann(eta, spotential)
    newH[1:-1, 1:-1] = dToN[1:-1, 1:-1]/beta**2/nu
    newH = enforce_boundaries(newH, 'h')

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
                         + dToN[1:-1, 1:-1]/beta**2)**2 / (1 + beta**2*normGradEta2[1:-1, 1:-1])
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

def oneStepIncrStocha(eta, spotential):
    dh_stocha, dspotential_stocha = np.zeros_like(chix), np.zeros_like(chix)

    gradxPhiRight = np.zeros_like(spotential)
    gradxPhiLeft  = np.zeros_like(spotential)
    gradyPhiUp    = np.zeros_like(spotential)
    gradyPhiDown  = np.zeros_like(spotential)
    gradxPhiRight[1:-1,1:-1] = (spotential[1:-1,  2:] - spotential[1:-1,1:-1])/dx
    gradxPhiLeft[1:-1,1:-1]  = (spotential[1:-1,1:-1] - spotential[1:-1, :-2])/dx
    gradyPhiUp[1:-1,1:-1]    = (spotential[2:,  1:-1] - spotential[1:-1,1:-1])/dy
    gradyPhiDown[1:-1,1:-1]  = (spotential[1:-1,1:-1] - spotential[:-2, 1:-1])/dy
    gradxPhiRight = enforce_boundaries(gradxPhiRight, 'h')
    gradxPhiLeft = enforce_boundaries(gradxPhiLeft, 'h')
    gradyPhiUp = enforce_boundaries(gradyPhiUp, 'h')
    gradyPhiDown = enforce_boundaries(gradyPhiDown, 'h')

    gradxEtaRight = np.zeros_like(eta)
    gradxEtaLeft  = np.zeros_like(eta)
    gradyEtaUp    = np.zeros_like(eta)
    gradyEtaDown   = np.zeros_like(eta)
    gradxEtaRight[1:-1,1:-1] = (eta[1:-1,  2:] - eta[1:-1,1:-1])/dx
    gradxEtaLeft[1:-1,1:-1]  = (eta[1:-1,1:-1] - eta[1:-1, :-2])/dx
    gradyEtaUp[1:-1,1:-1]    = (eta[2:,  1:-1] - eta[1:-1,1:-1])/dy
    gradyEtaDown[1:-1,1:-1]  = (eta[1:-1,1:-1] - eta[:-2, 1:-1])/dy
    gradxEtaRight = enforce_boundaries(gradxEtaRight, 'h')
    gradxEtaLeft = enforce_boundaries(gradxEtaLeft, 'h')
    gradyEtaUp = enforce_boundaries(gradyEtaUp, 'h')
    gradyEtaDown = enforce_boundaries(gradyEtaDown, 'h')

    velocityX = (gradxPhiRight + gradxPhiLeft)/2
    velocityY = (gradyPhiUp + gradyPhiDown)/2
    gradEtaX = (gradxEtaRight + gradxEtaLeft)/2
    gradEtaY = (gradyEtaUp + gradyEtaDown)/2

    normGradEta2 = np.zeros_like(eta)
    normGradPhi2 = np.zeros_like(spotential)
    normGradEta2[1:-1, 1:-1] = ((gradxEtaRight**2 + gradxEtaLeft**2)/2 + (gradyEtaUp**2 + gradyEtaDown**2)/2)[1:-1,1:-1]
    normGradPhi2[1:-1, 1:-1] = ((gradxPhiRight**2 + gradxPhiLeft**2)/2 + (gradyPhiUp**2 + gradyPhiDown**2)/2)[1:-1,1:-1]

    #used for Saint-Venant without small noise assumption
    hu, hv = np.zeros_like(eta), np.zeros_like(eta)
    hu[1:-1, 1:-1] = (depth + eta[1:-1, 1:-1])*velocityX[1:-1, 1:-1]
    hv[1:-1, 1:-1] = (depth + eta[1:-1, 1:-1])*velocityY[1:-1, 1:-1]
    hu = enforce_boundaries(hu, 'h')
    hv = enforce_boundaries(hv, 'h')
    divCoeff = np.zeros_like(eta)
    divCoeff[1:-1, 1:-1] = (hu[1:-1, 2:] - hu[1:-1, :-2])/dx/2 + (hv[2:, 1:-1] - hv[:-2, 1:-1])/dy/2
    divCoeff = enforce_boundaries(divCoeff, 'h')
    #-----

    for k in range(nmodes):
        if stochasticPotentialAtFixedHeight:
            gradxPhiStochaRight,gradxPhiStochaLeft = np.zeros_like(spotential),np.zeros_like(spotential)
            gradyPhiStochaUp,gradyPhiStochaDown = np.zeros_like(spotential),np.zeros_like(spotential)
            gradxPhiStochaRight[1:-1, 1:-1] = (phi[1:-1, 2:, k] - phi[1:-1, 1:-1, k])/dx
            gradxPhiStochaLeft[1:-1, 1:-1]  = (phi[1:-1, 1:-1, k] - phi[1:-1, :-2, k])/dx
            gradyPhiStochaUp[1:-1, 1:-1]    = (phi[2:, 1:-1, k] - phi[1:-1, 1:-1, k])/dy
            gradyPhiStochaDown[1:-1, 1:-1]  = (phi[1:-1, 1:-1, k] - phi[:-2, 1:-1, k])/dy
            gradxPhiStochaRight = enforce_boundaries(gradxPhiStochaRight, 'h')
            gradxPhiStochaLeft = enforce_boundaries(gradxPhiStochaLeft, 'h')
            gradyPhiStochaUp = enforce_boundaries(gradyPhiStochaUp, 'h')
            gradyPhiStochaDown = enforce_boundaries(gradyPhiStochaDown, 'h')

            dToN_stocha = (1/depth * np.sinh(1+eta/depth)/np.cosh(1)*phi[:, :, k]
                            -np.cosh(1+eta/depth)/np.cosh(1)*(gradxPhiStochaRight*gradxEtaRight + gradxPhiStochaLeft*gradxEtaLeft)/2
                            -np.cosh(1+eta/depth)/np.cosh(1)*(gradyPhiStochaUp*gradyEtaUp + gradyPhiStochaDown*gradyEtaDown)/2
                            )
        else:
            if ups != 0:
                dToN_stocha = dirichletToNeumann(eta, phi[:, :, k])
            else:
                dToN_stocha = np.zeros_like(eta)
        
        dh_stocha[1:-1, 1:-1, k] = (
                dToN_stocha[1:-1, 1:-1]/beta**2/nu
            - (chix[1:-1,2:, k]*eta[1:-1,2:] - chix[1:-1,:-2, k]*eta[1:-1,:-2])/(2*dx)
            - (chiy[2:,1:-1, k]*eta[2:,1:-1] - chiy[:-2,1:-1, k]*eta[:-2,1:-1])/(2*dy)
            +  chiz[1:-1,1:-1,k]
        )
        dh_stocha[:,:,k] = enforce_boundaries(dh_stocha[:,:,k], 'h')

        if smallNoiseAssumption:
            dspotential_stocha[1:-1, 1:-1, k] = (
                (dToN_stocha[1:-1, 1:-1]/beta**2
                + (gradxPhiRight[1:-1, 1:-1]*gradxEtaRight[1:-1, 1:-1] + gradxPhiLeft[1:-1, 1:-1]*gradxEtaLeft[1:-1, 1:-1])/2
                + (gradyPhiUp[1:-1, 1:-1]*gradyEtaUp[1:-1, 1:-1] + gradyPhiDown[1:-1, 1:-1]*gradyEtaDown[1:-1, 1:-1])/2)
                / (1 + normGradEta2[1:-1, 1:-1]) * dh_stocha[1:-1, 1:-1, k]
            )
        else:
            chixR, chixL, chiyU, chiyD = np.zeros_like(eta),np.zeros_like(eta),np.zeros_like(eta),np.zeros_like(eta)
            chixR = (chix[1:-1, 2:, k]   - chix[1:-1, 1:-1, k])/2
            chixL = (chix[1:-1, 1:-1, k] - chix[1:-1, :-2, k])/2
            chiyU = (chiy[2:, 1:-1, k]   - chiy[1:-1, 1:-1, k])/2
            chiyD = (chiy[1:-1, 1:-1, k] - chiy[:-2, 1:-1, k])/2

            firstTerm  = ((chixR*gradxPhiRight[1:-1, 1:-1] + chixL*gradxPhiLeft[1:-1, 1:-1])/2
                            +(chiyU*gradyPhiUp[1:-1, 1:-1]    + chiyD*gradyPhiDown[1:-1, 1:-1])/2)
            secondTerm = (
                ((chixR*gradxEtaRight[1:-1, 1:-1] + chixL*gradxEtaLeft[1:-1, 1:-1])/2
                +(chiyU*gradyEtaUp[1:-1, 1:-1] + chiyD*gradyEtaDown[1:-1, 1:-1])/2)
                *((gradxPhiRight[1:-1, 1:-1]*gradxEtaRight[1:-1, 1:-1] + gradxPhiLeft[1:-1, 1:-1]*gradxEtaLeft[1:-1, 1:-1])/2
                + (gradyPhiUp[1:-1, 1:-1]*gradyEtaUp[1:-1, 1:-1] + gradyPhiDown[1:-1, 1:-1]*gradyEtaDown[1:-1, 1:-1])/2)
                )
            dspotential_stocha[1:-1, 1:-1, k] = (
                (firstTerm + secondTerm)
                / (1 + normGradEta2[1:-1, 1:-1]) * dh_stocha[1:-1, 1:-1, k]
            )
        if wwApprox == "Shallow-water-order0":
            gxChix = (chix[1:-1, 2:, k] - chix[1:-1, :-2, k])/dx/2
            gxChiy = (chiy[1:-1, 2:, k] - chiy[1:-1, :-2, k])/dx/2
            gyChix = (chix[2:, 1:-1, k] - chix[:-2, 1:-1, k])/dy/2
            gyChiy = (chiy[2:, 1:-1, k] - chiy[:-2, 1:-1, k])/dy/2
            toBeProjectedX, toBeProjectedY = np.zeros_like(eta), np.zeros_like(eta)
            toBeProjectedX[1:-1,1:-1] = gxChix * (velocityX + divCoeff*gradEtaX)[1:-1,1:-1] + gxChiy * (velocityY + divCoeff*gradEtaY)[1:-1,1:-1]
            toBeProjectedY[1:-1,1:-1] = gyChix * (velocityX + divCoeff*gradEtaX)[1:-1,1:-1] + gyChiy * (velocityY + divCoeff*gradEtaY)[1:-1,1:-1]
            toBeProjectedX = enforce_boundaries(toBeProjectedX, 'h')
            toBeProjectedY = enforce_boundaries(toBeProjectedY, 'h')

            meanToBeProjX = np.mean(toBeProjectedX)
            meanToBeProjY = np.mean(toBeProjectedY)
            fftNewTerm = (Kx*np.fft.fft(toBeProjectedX - meanToBeProjX) + Ky*np.fft.fft(toBeProjectedY - meanToBeProjY))/(Kx**2 + Ky**2)
            fftNewTerm[0,0] = 0
            newTerm = np.fft.ifft(fftNewTerm)
            dspotential_stocha[1:-1, 1:-1, k] -= np.real(newTerm)[1:-1, 1:-1]
        
        dspotential_stocha[:, :, k] = enforce_boundaries(dspotential_stocha[:, :, k], 'h')

    return dh_stocha, dspotential_stocha

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------WAVE MODEL ITERATOR-----------------------------------------------
#--------------------------------------------------------------------------------------------------------------
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
        dh_new_stocha, dspotential_new_stocha = oneStepIncrStocha(eta, spotential)
        
        #Euler-Heun method for stochastic terms
        dBt = np.random.normal(0, np.sqrt(dt), nmodes)
        '''dBt = np.sqrt(dt)*truncnorm.rvs(-5, 5, size = nmodes)'''
        
        htemp[1:-1, 1:-1] = hc[1:-1, 1:-1] 
        spotentialTemp[1:-1, 1:-1] = spotential[1:-1, 1:-1]
        for k in range(nmodes):
            htemp[1:-1, 1:-1]  += dBt[k]*dh_new_stocha[1:-1, 1:-1, k]
            spotentialTemp[1:-1, 1:-1] += dBt[k]*dspotential_new_stocha[1:-1, 1:-1, k]
        htemp = enforce_boundaries(htemp, 'h')            
        spotentialTemp = enforce_boundaries(spotentialTemp, 'h')

        etatemp = np.pad(htemp[1:-1, 1:-1] - depth, 1, 'edge')
        etatemp = enforce_boundaries(etatemp, 'h')

        dh_new_stocha_bis, dspotential_new_stocha_bis = oneStepIncrStocha(etatemp, spotentialTemp)
        dh_new_stocha = (dh_new_stocha + dh_new_stocha_bis)/2
        dspotential_new_stocha = (dspotential_new_stocha + dspotential_new_stocha_bis)/2

        #RK4 method for BV time increment
        if timeIncrementMethod == 'RK4':
            deta0, dSpotential0 = oneStepIncr(eta, spotential)
            eta_inter1, spotential_inter1 = eta+deta0*dt/2, spotential + dSpotential0*dt/2
            deta_inter1, dSpotential_inter1 = oneStepIncr(eta_inter1, spotential_inter1)
            eta_inter2, spotential_inter2 = eta+deta_inter1*dt/2, spotential + dSpotential_inter1*dt/2
            deta_inter2, dSpotential_inter2 = oneStepIncr(eta_inter2, spotential_inter2)
            eta_inter3, spotential_inter3 = eta+deta_inter2*dt/2, spotential + dSpotential_inter2*dt/2
            deta_inter3, dSpotential_inter3 = oneStepIncr(eta_inter3, spotential_inter3)

            dspotential_new[1:-1, 1:-1] = 1/6*( dSpotential_inter3 + 2*dSpotential_inter2 + 2*dSpotential_inter1 + dSpotential0)[1:-1, 1:-1]
            dh_new[1:-1, 1:-1] = 1/6*( deta_inter3 + 2*deta_inter2 + 2*deta_inter1 + deta0)[1:-1, 1:-1]
            spotential[1:-1, 1:-1] += dt*dspotential_new[1:-1, 1:-1]
            h[1:-1, 1:-1] += dt*dh_new[1:-1, 1:-1]
            for k in range(nmodes):
                spotential[1:-1, 1:-1] += dBt[k]*dspotential_new_stocha[1:-1, 1:-1, k]
                h[1:-1, 1:-1] += dBt[k]*dh_new_stocha[1:-1, 1:-1, k]
        
        #Adams-Bashforth method for BV time increment
        elif timeIncrementMethod == 'Adams-Bashforth':
            deta0, dSpotential0 = oneStepIncr(eta, spotential)
            if first_step:
                dspotential_new[1:-1, 1:-1] = dSpotential0[1:-1, 1:-1]
                dh_new[1:-1, 1:-1] = deta0[1:-1, 1:-1]
                spotential[1:-1, 1:-1] += dt*dspotential_new[1:-1, 1:-1]
                h[1:-1, 1:-1] += dt*dh_new[1:-1, 1:-1]
                for k in range(nmodes):
                    spotential[1:-1, 1:-1] += dBt[k]*dspotential_new_stocha[1:-1, 1:-1, k]
                    h[1:-1, 1:-1] += dBt[k]*dh_new_stocha[1:-1, 1:-1, k]
                first_step = False
            else:
                dspotential_new[1:-1, 1:-1] = dSpotential0[1:-1, 1:-1]
                dh_new[1:-1, 1:-1] = deta0[1:-1, 1:-1]
                spotential[1:-1, 1:-1] += dt*(adams_bashforth_a*dspotential_new[1:-1, 1:-1]+adams_bashforth_b*dspotential[1:-1, 1:-1])
                h[1:-1, 1:-1] += dt*(adams_bashforth_a*dh_new[1:-1, 1:-1]+adams_bashforth_b*dh[1:-1, 1:-1])
                for k in range(nmodes):
                    spotential[1:-1, 1:-1] += dBt[k]*dspotential_new_stocha[1:-1, 1:-1, k]
                    h[1:-1, 1:-1] += dBt[k]*dh_new_stocha[1:-1, 1:-1, k]

        #RK4 + Adams-Bashforth
        """if first_step:
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
        """
            
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
    totalMomZ = []
    totalEnerg = []

    if save_data:
        spec = [n_x, n_y, dx, dy]
        export_to_csv(spec, path_to_store+'//spec.csv')

    for iteration, (h, spotential) in enumerate(model):
        if iteration % plot_every == 0:
            t = iteration * dt
            #print('Current time: ', t,'s')
            
            totalMass.append(np.sum(h[1:-1,1:-1])*dx/l_x*dy/l_y)

            dz= h/n_z
            fullPotential = solveLaplace(spotential, h-depth)
    
            intGxFullPotential = np.zeros_like(spotential)
            intGyFullPotential = np.zeros_like(spotential)
            intGxFullPotential[1:-1,1:-1] = np.sum((fullPotential[1:-1,2:,:] - fullPotential[1:-1,1:-1,:])/dx, axis = 2)*dz[1:-1,1:-1]
            intGyFullPotential[1:-1,1:-1] = np.sum((fullPotential[2:,1:-1,:] - fullPotential[1:-1,1:-1,:])/dy, axis = 2)*dz[1:-1,1:-1]
            intGxFullPotential[1:-1,1:-1] += (spotential[1:-1,2:] - spotential[1:-1,1:-1])/dx*dz[1:-1,1:-1]
            intGyFullPotential[1:-1,1:-1] += (spotential[2:,1:-1] - spotential[1:-1,1:-1])/dy*dz[1:-1,1:-1]
            
            intGxFullPotential = enforce_boundaries(intGxFullPotential, 'h')
            intGyFullPotential = enforce_boundaries(intGyFullPotential, 'h')

            totalMomX.append(dx/l_x*dy/l_y*np.sum(intGxFullPotential[1:-1,1:-1]))
            totalMomY.append(dx/l_x*dy/l_y*np.sum(intGyFullPotential[1:-1,1:-1]))
            totalMomZ.append(dx/l_x*dy/l_y*np.sum(spotential[1:-1,1:-1] - fullPotential[1:-1,1:-1,0]))
            
            intKinEnerg = np.zeros_like(spotential)
            intKinEnerg[1:-1,1:-1] = np.sum((fullPotential[1:-1,2:,:] - fullPotential[1:-1,1:-1,:])**2/dx**2
                             +(fullPotential[2:,1:-1,:] - fullPotential[1:-1,1:-1,:])**2/dy**2, axis = 2)*dz[1:-1,1:-1]
            totalEnerg.append(dx/l_x*dy/l_y*np.sum(intKinEnerg[1:-1,1:-1]) + gravity*np.sum((h-depth)[1:-1,1:-1]**2))
            
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

        ax_bis = ax[0].twinx()
        cmomx = ax_bis.plot(timespace, np.array(totalMomX), color='royalblue')
        cmomy = ax_bis.plot(timespace, np.array(totalMomY), color='midnightblue')
        #cmomz = ax_bis.plot(timespace, np.array(totalMomZ), color='lightblue')
        ax_bis.set_ylabel('Momentum error', color='blue')
        ax_bis.tick_params(axis='y', labelcolor='blue')
        ax_bis.legend(['X Momentum', 'Y Momentum', 'Z momentum'], loc='lower left')
        
        cenerg = ax[1].plot(timespace, totalEnerg/totalEnerg[0] -1, color='red')
        ax[1].set_ylabel('Energy error', color='red')
        ax[1].tick_params(axis='y', labelcolor='red')
        ax[1].legend(['Energy'], loc='best')

        energ_amplitude_stat = max(totalEnerg[count//2:]/totalEnerg[0]) - min(totalEnerg[count//2:]/totalEnerg[0])
        print("Maximal amplitude of energy variation over the second half of the simulation (to be interpreted as stationary regime if the simulation in long enough):",
                                energ_amplitude_stat)
        
        plt.show()

    if makegif:
        frames = np.stack([iio.imread(f"images/{i}.jpg") for i in range(count)] + [iio.imread(f"images/{i}.jpg") for i in range(count-2, 0, -1)], axis = 0)
        iio.mimwrite('test1000.gif', frames, loop=0)
