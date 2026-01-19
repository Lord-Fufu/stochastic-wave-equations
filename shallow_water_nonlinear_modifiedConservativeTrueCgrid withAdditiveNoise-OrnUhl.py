""" model2d_nonlinear.py

2D shallow water model with:

- varying Coriolis force
- nonlinear terms
- lateral friction
- periodic boundary conditions

"""

import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
import scipy.sparse as scsp
from scipy.stats import truncnorm, truncexpon
from fipy import numerix as nx
import pandas as pd
import os

from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse.linalg import spsolve

#Choose simulation type. Can be "Saint-Venant", "Boussinesq" or "Serre-Green-Naghdi".
simulationType = "Saint-Venant"


n_data=1
for i_trajectory in range(n_data):
    
    #--------------------------------------------------------------------------------------------------------------
    #---------------------------------------------SIMULATION PARAMETERS-----------------------------------------------
    #--------------------------------------------------------------------------------------------------------------

    # maximum number of timesteps
    itmax = float('infinity') #1400

    # grid setup
    n_x = 128
    dx = 2.5e1
    l_x = n_x * dx

    n_y = 128
    dy = 2.5e1
    l_y = n_y * dy

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
    lateral_viscosity = 1e-3 * 2e-4 * dx ** 2

    #epsilon only impacts the wave height in this model --> maybe write it with adimensioned variables
    eps=0.1
    beta= depth/np.sqrt(l_x*l_y)
    amplitude = eps*depth

    # type of boundary conditions parameters (periodic or Dirichlet)
    periodic_boundary_x = True
    periodic_boundary_y = True

    # adams-bashforth parameters
    adams_bashforth_a = 1.5 + 0.1
    adams_bashforth_b = -(0.5 + 0.1)

    # timestep
    dt = 0.1 * min(dx, dy) / np.sqrt(gravity * depth)
    print( "\n timestep", dt, "\n")

    # other parameters
    phase_speed = np.sqrt(gravity * depth)
    rossby_radius = np.sqrt(gravity * depth) / coriolis_param.mean()
    energySGN = (simulationType == "Serre-Green-Naghdi") #the energy to be computed is different for SV adn SGN

    # Diffusion matrix for implicit temporal scheme
    diffCoeff = 0 #1e0
    coeffForcing = 0 #1e-6
    addForcingX = 0 #1e5
    addForcingY = 0 #-1e0
    
    Kx = np.fft.fftfreq(n_x, d=dx)
    Ky = np.fft.fftfreq(n_y, d=dy)
    k_x, k_y = np.meshgrid(Kx, Ky, indexing='ij')
    normk = np.sqrt(k_x**2 + k_y**2)

    antiAlias = (normk <= np.max(normk)*0.49)


    #--------------------------------------------------------------------------------------------------------------
    #------------------------------------------SAVE & PLOT PARAMETERS----------------------------------------------
    #--------------------------------------------------------------------------------------------------------------

    # save parameters
    save_data = False

    if save_data:
        dir_to_store = 'data_deterministic'
        if n_data > 1:
            dir_to_store = 'data_looped//data'+str(i_trajectory)
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
    plot_range = 1*amplitude
    plot_every = 10

    withArrows = False
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

    u0 = 10 * np.exp(-(Y - y[n_y // 2])**2 / (0.02 * l_x)**2)
    v0 = np.zeros_like(u0)
    u0 = 0*np.ones_like(u0)

    h_geostrophy = np.cumsum(-dy * u0 * coriolis_param / gravity, axis=0)
    '''h0 = (
        depth
        + h_geostrophy
        # make sure h0 is centered around depth
        - h_geostrophy.mean()
        # small perturbation
        #+ 1 * np.sin(X / l_x * 10 * np.pi) * np.cos(Y / l_y * 8 * np.pi)
        + amplitude*np.exp(-1e-10*((X-l_x/2)**2 + (Y-l_y/2)**2)**2)
        #+ amplitude*np.exp(-100 * (0.9*abs(Y / l_y - 0.5)**2+ ((Y/l_y - 0.5) - abs(X / l_x - 0.5)**(2/3))**2))
    )'''
    h0 = depth * np.ones_like(u0) #+ amplitude*np.exp(-1e-10*((X-l_x/2)**2 + 1e-0*(Y-l_y/2)**2)**2)
                                  #+  2*amplitude*np.exp(-1e-12*((X-l_x*3/4)**2 + 1e1*(Y-l_y*3/4)**2)**2))

    hu0 = h0 * u0
    hv0 = h0 * v0

    #--------------------------------------------------------------------------------------------------------------
    #----------------------------------------------NOISE PARAMETERS------------------------------------------------
    #---------------------------------------FOR DOUBLE PERIODIC BOUNDARIES-----------------------------------------
    #--------------------------------------------------------------------------------------------------------------
    locFFTCoeff = 1e7
    locFFtCoeffAdd = 1e7
    
    @np.vectorize
    def functionFor5on2Decrease(x,w1,w2,w3):
        output = np.exp(-locFFTCoeff*(x-w1)**2) #+np.exp(-locFFTCoeff*(x-w2)**2)+np.exp(-locFFTCoeff*(x-w3)**2)#+5*x/(1+x)**(5/2)
        return output/np.sqrt(locFFTCoeff*np.pi) #/3
    def functionForAdd(x, w):
        return np.exp(-locFFtCoeffAdd*(x-w)**2)/np.sqrt(locFFtCoeffAdd*np.pi)

    if periodic_boundary_x and periodic_boundary_y:

        ups = 5e7
        wavenumber1 = np.pi*2 #np.pi*1
        wavenumber2 = np.pi*3
        wavenumber3 = np.pi*5
        upsAdd = 1e4       
        wavenumberAdd = np.pi*2
        dampingCharTime = float('infinity') #90*60

        coeff=-2
        coeffx=0
        coeffy=0

        l_ref = np.sqrt(l_x * l_y)
        l_waveX = (l_x + coeff*dx)
        l_waveY = (l_y + coeff*dy)
        l_wave = (l_waveX*l_waveY)**0.5

        #plt.plot(k_x, functionForAdd(k_x,wavenumberAdd/l_wave))


        #TRANSPORT NOISE
        normk = np.sqrt(k_x**2+k_y**2)
        baseFFTNoiseX = (functionFor5on2Decrease(normk, wavenumber1/l_wave, wavenumber2/l_wave, wavenumber3/l_wave)*(-k_y))[1:-1,1:-1]
        baseFFTNoiseY = (functionFor5on2Decrease(normk, wavenumber1/l_wave, wavenumber2/l_wave, wavenumber3/l_wave)*(k_x))[1:-1,1:-1]

        baseFFTNoiseX*=ups/np.sqrt(np.sum(baseFFTNoiseX**2))/(dx*dy)
        baseFFTNoiseY*=ups/np.sqrt(np.sum(baseFFTNoiseY**2))/(dx*dy)

        #ADDITIVE NOISE
        baseFFTAddNoiseX = functionForAdd(normk, wavenumberAdd/l_wave)[1:-1,1:-1]
        baseFFTAddNoiseY = baseFFTAddNoiseX.copy()
        baseFFTAddNoiseX*=upsAdd/np.sqrt(np.sum(baseFFTAddNoiseX**2))/(dx*dy)**(1/2)
        baseFFTAddNoiseY*=upsAdd/np.sqrt(np.sum(baseFFTAddNoiseY**2))/(dx*dy)**(1/2)

        nCorrSteps = 10
        memWeightCoeff = 0.9
        weightArr = memWeightCoeff**(np.arange(nCorrSteps))

        nCorrStepsAdd = 1
        memWeightCoeffAdd = 0.9
        weightArrAdd = memWeightCoeffAdd**(np.arange(nCorrStepsAdd))

        weightArrScaleCoeff = np.log(memWeightCoeff)/(1-memWeightCoeff**nCorrSteps)
        weightArrScaleCoeffAdd = np.log(memWeightCoeffAdd)/(1-memWeightCoeffAdd**nCorrStepsAdd)


    #--------------------------------------------------------------------------------------------------------------
    #---------------------------------------------AUXILIARY FUNCTIONS----------------------------------------------
    #--------------------------------------------------------------------------------------------------------------

    def prepare_plot():
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        #cs, cs2, cs3 = update_plot(0, h0, hu0, hv0, ax, count=0, draw=False)
        cs = update_plot(0, h0, hu0, hv0, ax, count=0, draw=False)
        #cs, cs2 = update_plot(0, h0, hu0, hv0, ax, count=0, draw=False)
        plt.colorbar(cs, label='$\\eta$ (m)')
        #plt.colorbar(cs2, label='vorticity$ (NU)')
        #plt.colorbar(cs2, label='grad $\\eta$ (NU)')
        #plt.colorbar(cs3, label='fft$\\eta$ (NU)')
        return fig, ax


    def update_plot(t, h, hu, hv, ax, count, draw=True):
        
        eta = h - depth

        quiver_stride = (
            slice(1, -1, n_y // max_quivers),
            slice(1, -1, n_x // max_quivers)
        )
        ax.clear()
        '''ax[0].clear()
        ax[1].clear()
        ax[2].clear()'''
        cs = ax.pcolormesh(
            x[1:-1] / 1e3,
            y[1:-1] / 1e3,
            eta[1:-1, 1:-1],
            vmin=-plot_range, vmax=plot_range, cmap='RdBu_r'
        )
        
        '''cs = ax[0].pcolormesh(
            x[1:-1] / 1e3,
            y[1:-1] / 1e3,
            eta[1:-1, 1:-1],
            vmin=-plot_range, vmax=plot_range, cmap='RdBu_r'
        )'''

        '''
        uTEMP = np.zeros_like(hu)
        vTEMP = np.zeros_like(hv)
        uTEMP[1:-1, 1:-1] = (hu[1:-1, 1:-1] + hu[1:-1, 2:])/2/h[1:-1,1:-1]
        vTEMP[1:-1, 1:-1] = (hv[1:-1, 1:-1] + hu[2:, 1:-1])/2/h[1:-1,1:-1]
        uTEMP = enforce_boundaries(uTEMP, 'u')
        vTEMP = enforce_boundaries(vTEMP, 'v')

        vorticity = np.zeros_like(hv)
        vorticity[1:-1, 1:-1] = (vTEMP[1:-1,1:-1] - vTEMP[1:-1,:-2])/dx - (uTEMP[1:-1,1:-1] - uTEMP[:-2,1:-1])/dy

        print(np.max(vorticity))
        cs2 = ax[1].pcolormesh(
            x[1:-1] / 1e3,
            y[1:-1] / 1e3,
            vorticity[1:-1, 1:-1],
            vmin=-3e-3, vmax=3e-3, cmap='RdBu_r'
        )'''

        #print(np.max(normk))

        fftETA = np.fft.fft2(eta[1:-1, 1:-1])
        '''fftHU = np.fft.fft2(hu[1:-1, 1:-1]/(depth + (eta[1:-1,0:-2]+eta[1:-1,1:-1])/2))
        fftHV = np.fft.fft2(hv[1:-1, 1:-1]/(depth + (eta[0:-2,1:-1]+eta[1:-1,1:-1])/2))
        fftOMEGA =  k_x[1:-1, 1:-1]*fftHV - k_y[1:-1, 1:-1]*fftHU'''

        '''        
        cs2 = ax[1].pcolormesh(
            x[1:-1] / 1e3,
            y[1:-1] / 1e3,
            np.fft.ifft2((normk >= 0.0025)[1:-1, 1:-1]*fftETA).real,
            #np.fft.ifft2(fftOMEGA).real,
            vmin=-plot_range, vmax=plot_range, cmap='Greys'
        )

        cs3 = ax[2].pcolormesh(
            Kx[0:n_x//2]*1e3,
            Ky[0:n_y//2]*1e3,
            abs(fftETA)[0:n_y//2,0:n_x//2],
            vmin=0, vmax=100, cmap='Reds'
        )
        '''
        if withArrows:
            if np.any((hu[quiver_stride] != 0) | (hv[quiver_stride] != 0)):
                ax.quiver(
                    x[quiver_stride[1]] / 1e3,
                    y[quiver_stride[0]] / 1e3,
                    hu[quiver_stride]/h[quiver_stride],
                    hv[quiver_stride]/h[quiver_stride],
                    clip_on=False
                )
        
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ (km)')
        ax.set_ylabel('$y$ (km)')
        ax.set_xlim(x[1] / 1e3, x[-2] / 1e3)
        ax.set_ylim(y[1] / 1e3, y[-2] / 1e3)
        ax.set_title(
            't=%5.0f min %5.2f s' #, R=%5.1f km, c=%5.1f m/s '
            % (t // 60, t%60) #, rossby_radius / 1e3, phase_speed)
        )
        
        '''
        ax[0].set_aspect('equal')
        ax[0].set_xlabel('$x$ (km)')
        ax[0].set_ylabel('$y$ (km)')
        ax[0].set_xlim(x[1] / 1e3, x[-2] / 1e3)
        ax[0].set_ylim(y[1] / 1e3, y[-2] / 1e3)
        ax[0].set_title(
            't=%5.0f min %5.2f s' #, R=%5.1f km, c=%5.1f m/s '
            % (t // 60, t%60) #, rossby_radius / 1e3, phase_speed)
        )
        
        ax[1].set_aspect('equal')
        ax[1].set_xlim(x[1] / 1e3, x[-2] / 1e3)
        ax[1].set_ylim(y[1] / 1e3, y[-2] / 1e3)
        ax[1].set_title(
            't=%5.0f min %5.2f s' #, R=%5.1f km, c=%5.1f m/s '
            % (t // 60, t%60) #, rossby_radius / 1e3, phase_speed)
        )

        ax[2].set_aspect('equal')
        ax[2].set_xlabel('$kx$ (km^-1)')
        ax[2].set_ylabel('$ky$ (km^-1)')
        ax[2].set_title(
            't=%5.0f min %5.2f s' #, R=%5.1f km, c=%5.1f m/s '
            % (t // 60, t%60) #, rossby_radius / 1e3, phase_speed)
        )'''
        if save_plots:
            # Enregistrement dans le dossier "images"
            plt.savefig(f"images/{count}_SV.pdf")

        if draw:
            plt.pause(0.1)

        return cs #,cs2,cs3


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


    phix = np.zeros_like(h0)
    phiy = np.zeros_like(h0)
    phix[1:-1,1:-1] = np.fft.ifft2(baseFFTNoiseX).real
    phiy[1:-1,1:-1] = np.fft.ifft2(baseFFTNoiseY).real
    phix = enforce_boundaries(phix, 'u')
    phiy = enforce_boundaries(phiy, 'v')

    a_xx = np.zeros_like(phix)
    a_xy = np.zeros_like(phix)
    a_yy = np.zeros_like(phix)

    us = np.zeros_like(u0)
    vs = np.zeros_like(v0)

    a_xx = phix**2
    a_xy = phix*phiy
    a_yy = phiy**2

    us[1:-1,1:-1] = (a_xx[1:-1,2:] - a_xx[1:-1,:-2])/dx/2 + (a_xy[2:,1:-1] - a_xy[:-2,1:-1])/dy/2
    vs[1:-1,1:-1] = (a_xy[1:-1,2:] - a_xy[1:-1,:-2])/dx/2 + (a_yy[2:,1:-1] - a_yy[:-2,1:-1])/dy/2
    us[:,:] = enforce_boundaries(us, 'u')
    vs[:,:] = enforce_boundaries(vs, 'v')

    div_us = np.zeros_like(us)
    div_us[1:-1,1:-1] = (us[1:-1,1:-1] - us[1:-1,:-2])/dx + (vs[1:-1,1:-1] - vs[:-2,1:-1])/dy
    div_us = enforce_boundaries(div_us, 'h')

    nplots = plotNoiseDivergence + plotIS + plotISDIV

    if nplots>0:
        fig, ax = plt.subplots(2, 2, figsize=(14, 7))
        cs = ax[0,0].pcolormesh(
            x[1:-1] / 1e3,
            y[1:-1] / 1e3,
            (phix[1:-1, 1:-1, 1] - phix[1:-1, :-2, 1])/dx + (phiy[1:-1, 1:-1, 1] - phiy[:-2, 1:-1, 1])/dy,
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


    print("Maximal value of div us: ", np.max(abs((us[1:-1,1:-1] - us[1:-1,:-2])/dx + (vs[1:-1,1:-1] - vs[:-2,1:-1])/dy)))

    maxNoiseDivg = np.zeros_like(div_us)
    maxNoiseDivg[1:-1,1:-1] = abs((phix[1:-1,2:] - phix[1:-1,:-2])/dx/2
                                 +(phiy[2:,1:-1] - phiy[:-2,1:-1])/dy/2)
    print("Maximal value of div phi: ", np.max(maxNoiseDivg))

    print("Periodicity default of phix: ", np.max(abs(phix[1,:] - phix[-2,:]) + abs(phix[:,1] - phix[:,-2])))
    print("Periodicity default of phiy: ", np.max(abs(phiy[1,:] - phiy[-2,:]) + abs(phiy[:,1] - phiy[:,-2])))

    def export_to_csv(field, name):
        df = pd.DataFrame(field)
        df.to_csv(name)

    #--------------------------------------------------------------------------------------------------------------
    #--------------------------------------------WAVE MODEL ITERATOR-----------------------------------------------
    #--------------------------------------------------------------------------------------------------------------

    def iterate_shallow_water():
        # allocate arrays
        hu, hv, h = np.empty((n_y, n_x)), np.empty((n_y, n_x)), np.empty((n_y, n_x))
        hutemp, hvtemp, htemp = np.empty((n_y, n_x)), np.empty((n_y, n_x)), np.empty((n_y, n_x))
        dhu, dhv, dh = np.empty((n_y, n_x)), np.empty((n_y, n_x)), np.empty((n_y, n_x))
        dhu_new, dhv_new, dh_new = np.empty((n_y, n_x)), np.empty((n_y, n_x)), np.empty((n_y, n_x))
        dhu_new_stocha, dhv_new_stocha, dh_new_stocha = np.empty((n_y, n_x)), np.empty((n_y, n_x)), np.empty((n_y, n_x))
        hu_stocha, hv_stocha = np.empty((n_y, n_x)), np.empty((n_y, n_x))
        hu_stocha_temp, hv_stocha_temp = np.empty((n_y, n_x)), np.empty((n_y, n_x))
        fue, fun = np.empty((n_y, n_x)), np.empty((n_y, n_x))
        fve, fvn = np.empty((n_y, n_x)), np.empty((n_y, n_x))

        det_mat, diag_term_11, diag_term_22 = np.empty((n_y, n_x)), np.empty((n_y, n_x)), np.empty((n_y, n_x))
        non_diag_term_12, non_diag_term_21 = np.empty((n_y, n_x)), np.empty((n_y, n_x))
        
        dh_new_fourier, dhu_new_fourier, dhv_new_fourier = np.empty((n_y, n_x), dtype=complex), np.empty((n_y, n_x), dtype=complex), np.empty((n_y, n_x), dtype=complex)
        dh_newStocha_fourier, dhu_newStocha_fourier, dhv_newStocha_fourier = np.empty((n_y, n_x), dtype=complex), np.empty((n_y, n_x), dtype=complex), np.empty((n_y, n_x), dtype=complex)


        # initial conditions
        h[...] = h0
        hu[...] = hu0
        hv[...] = hv0

        # boundary values of h must not be used
        h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = np.nan
        h = enforce_boundaries(h, 'h')
        hu = enforce_boundaries(hu, 'u')
        hv = enforce_boundaries(hv, 'v')

        first_step = True
        currentTime = 0

        memBrownian = np.zeros(baseFFTNoiseY.shape+(nCorrSteps,))
        memBrownianAddX = np.zeros(baseFFTNoiseY.shape+(nCorrStepsAdd,))
        memBrownianAddY = np.zeros(baseFFTNoiseY.shape+(nCorrStepsAdd,))

        # time step equations
        while True:
            hc = np.pad(h[1:-1, 1:-1], 1, 'edge')
            hc = enforce_boundaries(hc, 'h')
            eta = np.pad(h[1:-1, 1:-1] - depth, 1, 'edge')
            eta = enforce_boundaries(eta, 'h')

            #print('Min height:', np.min(h),'\n')

            localx_h = np.zeros_like(hc) + depth
            localy_h = np.zeros_like(hc) + depth
            localx_h[1:-1, 1:-1] = 0.5 * (hc[1:-1, 1:-1] + hc[1:-1, 2:])
            localy_h[1:-1, 1:-1] = 0.5 * (hc[1:-1, 1:-1] + hc[2:, 1:-1])
            localx_h = enforce_boundaries(localx_h, 'h')
            localy_h = enforce_boundaries(localy_h, 'h')

            dh_new[1:-1, 1:-1] = -(
                (hu[1:-1, 1:-1] - hu[1:-1, :-2]) / dx - ( (localx_h*us)[1:-1, 1:-1] - (localx_h*us)[1:-1, :-2]) / dx
                + (hv[1:-1, 1:-1] - hv[:-2, 1:-1]) / dy - ( (localy_h*vs)[1:-1, 1:-1] - (localy_h*vs)[:-2, 1:-1]) / dy
            )

            if lateral_viscosity > 0:
                # lateral friction
                fue[1:-1, 1:-1] = 0.5*lateral_viscosity * (hu[1:-1, 2:] - 2*hu[1:-1, 1:-1] + hu[1:-1, :-2]) / dx**2
                fun[1:-1, 1:-1] = 0.5*lateral_viscosity * (hu[2:, 1:-1] - 2*hu[1:-1, 1:-1] + hu[:-2, 1:-1]) / dy**2
                fue = enforce_boundaries(fue, 'u')
                fun = enforce_boundaries(fun, 'v')

                fve[1:-1, 1:-1] = 0.5*lateral_viscosity * (hv[1:-1, 2:] - 2*hv[1:-1, 1:-1] + hv[1:-1, :-2]) / dx**2
                fvn[1:-1, 1:-1] = 0.5*lateral_viscosity * (hv[2:, 1:-1] - 2*hv[1:-1, 1:-1] + hv[:-2, 1:-1]) / dy**2
                fve = enforce_boundaries(fve, 'u')
                fvn = enforce_boundaries(fvn, 'v')

            # nonlinear momentum equation
            local_uStar_Forhu = np.zeros_like(hu)
            local_vStar_Forhu = np.zeros_like(hv)
            local_uStar_Forhu[1:-1, 1:-1] = (hu/localx_h)[1:-1, 1:-1] - us[1:-1, 1:-1]
            local_vStar_Forhu[1:-1, 1:-1] = (
                (hv[1:-1, 1:-1] + hv[1:-1, 2:] + hv[:-2, 2:] + hv[:-2, 1:-1])/4/localy_h[1:-1,1:-1]
                - (vs[1:-1, 1:-1] + vs[1:-1, 2:] + vs[:-2, 2:] + vs[:-2, 1:-1])/4
            )
            local_uStar_Forhu=enforce_boundaries(local_uStar_Forhu, 'u')
            local_vStar_Forhu=enforce_boundaries(local_vStar_Forhu, 'v')

            dhu_new[1:-1, 1:-1] = - (
                        ((hu*local_uStar_Forhu)[1:-1, 2:] - (hu*local_uStar_Forhu)[1:-1, :-2])/dx/2
                      + ((hu*local_vStar_Forhu)[2:, 1:-1] - (hu*local_vStar_Forhu)[:-2, 1:-1])/dy/2
                    + 1/2*(hc[1:-1, 2:]**2 - hc[1:-1, 1:-1]**2)/dx
            ) + coriolis_param[1:-1, 1:-1] * hv[1:-1, 1:-1] + coeffForcing*(localx_h*addForcingX - hu)[1:-1, 1:-1]
            dhu_new = enforce_boundaries(dhu_new, 'u')

            local_uStar_Forhv = np.zeros_like(hv)
            local_vStar_Forhv = np.zeros_like(hv)
            local_uStar_Forhv[1:-1, 1:-1] = (
                (hu[1:-1, 1:-1] + hu[1:-1, :-2] + hu[2:, :-2] + hu[2:, 1:-1])/4/localx_h[1:-1,1:-1]
                - ((localx_h*us)[1:-1, 1:-1] + (localx_h*us)[1:-1, :-2] + (localx_h*us)[2:, :-2] + (localx_h*us)[2:, 1:-1])/4
            )
            local_vStar_Forhv[1:-1, 1:-1] = (hv/localy_h)[1:-1, 1:-1] - vs[1:-1, 1:-1]
            local_uStar_Forhv=enforce_boundaries(local_uStar_Forhv, 'u')
            local_vStar_Forhv=enforce_boundaries(local_vStar_Forhv, 'v')

            dhv_new[1:-1, 1:-1] = - (
                        ((hv*local_vStar_Forhv)[2:, 1:-1] - (hv*local_vStar_Forhv)[:-2, 1:-1])/dy/2
                      + ((hv*local_uStar_Forhv)[1:-1, 2:] - (hv*local_uStar_Forhv)[1:-1, :-2])/dx/2
                    + 1/2*(hc[2:, 1:-1]**2 - hc[1:-1, 1:-1]**2)/dy
            ) - coriolis_param[1:-1, 1:-1] * hu[1:-1, 1:-1] + coeffForcing*(localy_h*addForcingY - hv)[1:-1, 1:-1]
            dhv_new = enforce_boundaries(dhv_new, 'v')

            
            #SGN bounded variation term (no Ito-Stokes drift)
            if simulationType == "Serre-Green-Naghdi":
                div_vit, div_mom = np.empty((n_y, n_x)), np.empty((n_y, n_x))
                sgn_term_stocha_dxX, sgn_term_stocha_dyX = np.empty((n_y, n_x)), np.empty((n_y, n_x))
                sgn_term_stocha_dxY, sgn_term_stocha_dyY = np.empty((n_y, n_x)), np.empty((n_y, n_x))
                sgn_term_x, sgn_term_y = np.empty((n_y, n_x)), np.empty((n_y, n_x))
                h3on3_div2, adv_div_vit = np.empty((n_y, n_x)), np.empty((n_y, n_x))

                div_vit[1:-1, 1:-1] = (   ((hu/localx_h)[1:-1, 1:-1] - (hu/localx_h)[1:-1, :-2]) / dx
                                        + ((hv/localy_h)[1:-1, 1:-1] - (hv/localy_h)[:-2, 1:-1]) / dy    )
                div_vit = enforce_boundaries(div_vit, 'h')
                        
                # SGN BV terms: advection of the divergence & product of divergences
                hustar = np.zeros_like(hu)
                hvstar = np.zeros_like(hu)
                hustar[1:-1, 1:-1] = 0.5*(hu[1:-1, 1:-1] + hu[1:-1, :-2]) - 0.5*hc[1:-1, 1:-1]*(us[1:-1, 1:-1] + us[1:-1, :-2])
                hvstar[1:-1, 1:-1] = 0.5*(hv[1:-1, 1:-1] + hv[:-2, 1:-1]) - 0.5*hc[1:-1, 1:-1]*(vs[1:-1, 1:-1] + vs[:-2, 1:-1])
                hustar = enforce_boundaries(hustar, 'u')
                hvstar = enforce_boundaries(hvstar, 'v')

                div_mom[1:-1, 1:-1] = (   ((hustar)[1:-1, 1:-1] - (hustar)[1:-1, :-2]) / dx
                                        + ((hvstar)[1:-1, 1:-1] - (hvstar)[:-2, 1:-1]) / dy    )
                div_mom = enforce_boundaries(div_mom, 'h')

                adv_div_vit[1:-1, 1:-1] = (
                    hc[1:-1, 1:-1]**2 / 3 * hustar[1:-1, 1:-1] * (div_vit[1:-1, 2:] - div_vit[1:-1, :-2])/(2*dx)
                + hc[1:-1, 1:-1]**2 / 3 * hvstar[1:-1, 1:-1] * (div_vit[2:, 1:-1] - div_vit[:-2, 1:-1])/(2*dy)
                )
                h3on3_div2[1:-1, 1:-1] = (hc[1:-1, 1:-1]**3)/3 * div_vit[1:-1, 1:-1] * (div_vit[1:-1, 1:-1] - div_us[1:-1, 1:-1])
                adv_div_vit = enforce_boundaries(adv_div_vit, 'h')
                h3on3_div2  = enforce_boundaries(h3on3_div2, 'h')

                sgn_term_x[1:-1, 1:-1] = (adv_div_vit[1:-1, 2:] - adv_div_vit[1:-1, 1:-1]) / dx - (h3on3_div2[1:-1, 2:] - h3on3_div2[1:-1, 1:-1]) / dx
                sgn_term_y[1:-1, 1:-1] = (adv_div_vit[2:, 1:-1] - adv_div_vit[1:-1, 1:-1]) / dy - (h3on3_div2[2:, 1:-1] - h3on3_div2[1:-1, 1:-1]) / dy
                
                
                # SGN BV terms: term arising from conservative form derivation
                sgn_term_x[1:-1, 1:-1] += (hc[1:-1, 2:]**3 * div_vit[1:-1, 2:]**2 - hc[1:-1, 1:-1]**3 * div_vit[1:-1, 1:-1]**2)/dx
                sgn_term_y[1:-1, 1:-1] += (hc[2:, 1:-1]**3 * div_vit[2:, 1:-1]**2 - hc[1:-1, 1:-1]**3 * div_vit[1:-1, 1:-1]**2)/dy

                sgn_term_x = enforce_boundaries(sgn_term_x, 'u')
                sgn_term_y = enforce_boundaries(sgn_term_y, 'v')

                dhu_new[1:-1, 1:-1] += (beta ** 2) * sgn_term_x[1:-1, 1:-1]
                dhv_new[1:-1, 1:-1] += (beta ** 2) * sgn_term_y[1:-1, 1:-1]
                dhu_new = enforce_boundaries(dhu_new, 'u')
                dhv_new = enforce_boundaries(dhv_new, 'v')
            
                
            #martingale terms
            signSelect = 2*np.random.randint(0,1, size = baseFFTNoiseY.shape)-1
            signSelectAddX = 2*np.random.randint(0,1, size = baseFFTNoiseY.shape)-1
            signSelectAddY = 2*np.random.randint(0,1, size = baseFFTNoiseY.shape)-1

            gap = 1000 #0.5
            gapAdd = 0 #0.5
            #dBt     = signSelect     * truncnorm.rvs(gap,    float('infinity'), loc=0, scale=np.sqrt(dt), size=baseFFTNoiseY.shape) 
            #dBtAddX = signSelectAddX * truncnorm.rvs(gapAdd, float('infinity'), loc=0, scale=np.sqrt(dt), size=baseFFTNoiseY.shape)
            #dBtAddY = signSelectAddY * truncnorm.rvs(gapAdd, float('infinity'), loc=0, scale=np.sqrt(dt), size=baseFFTNoiseY.shape)
            
            dBt = np.random.normal(0, np.sqrt(dt), baseFFTNoiseY.shape)
            dBtAddX = np.random.normal(0, np.sqrt(dt), baseFFTNoiseY.shape)
            dBtAddY = np.random.normal(0, np.sqrt(dt), baseFFTNoiseY.shape)
            
            if currentTime < nCorrSteps*dt:
                k = int(currentTime//dt)
                locCoeff = nCorrSteps/(1+k)
                memBrownian[:,:,k] = dBt
                dBt = weightArrScaleCoeff*locCoeff * np.mean(memBrownian*weightArr, axis=-1)
            else:
                for k in range(nCorrSteps-1):
                    memBrownian[:,:,k] = memBrownian[:,:,k+1]
                memBrownian[:,:,-1] = dBt
                dBt = weightArrScaleCoeff*np.mean(memBrownian*weightArr, axis=-1)

            if currentTime < nCorrStepsAdd*dt:
                k = int(currentTime//dt)
                locCoeff = nCorrStepsAdd/(1+k)
                memBrownianAddX[:,:,k] = dBtAddX
                memBrownianAddY[:,:,k] = dBtAddY
                dBtAddX = weightArrScaleCoeffAdd*locCoeff * np.mean(memBrownianAddX*weightArrAdd, axis=-1)
                dBtAddY = weightArrScaleCoeffAdd*locCoeff * np.mean(memBrownianAddY*weightArrAdd, axis=-1)
            else:
                for k in range(nCorrStepsAdd-1):
                    memBrownianAddX[:,:,k] = memBrownianAddX[:,:,k+1]
                    memBrownianAddY[:,:,k] = memBrownianAddY[:,:,k+1]
                memBrownianAddX[:,:,-1] = dBtAddX
                memBrownianAddY[:,:,-1] = dBtAddY
                dBtAddX = weightArrScaleCoeffAdd*np.mean(memBrownianAddX*weightArrAdd, axis=-1)
                dBtAddY = weightArrScaleCoeffAdd*np.mean(memBrownianAddY*weightArrAdd, axis=-1)
            
            noiseX = np.zeros_like(h0)
            noiseY = np.zeros_like(h0)
            noiseX[1:-1,1:-1] = np.fft.ifft2(baseFFTNoiseX*dBt).real #+ 0.1*ups*np.random.normal(0, np.sqrt(dt))
            noiseY[1:-1,1:-1] = np.fft.ifft2(baseFFTNoiseY*dBt).real #+ 0.1*ups*np.random.normal(0, np.sqrt(dt))
            noiseX = enforce_boundaries(noiseX, 'u')
            noiseY = enforce_boundaries(noiseY, 'v')

            addNoiseX = np.zeros_like(h0)
            addNoiseY = np.zeros_like(h0)
            addNoiseX[1:-1,1:-1] = np.fft.ifft2(baseFFTAddNoiseX*dBtAddX).real
            addNoiseY[1:-1,1:-1] = np.fft.ifft2(baseFFTAddNoiseY*dBtAddY).real
            addNoiseX = enforce_boundaries(addNoiseX, 'u')
            addNoiseY = enforce_boundaries(addNoiseY, 'v')

            noiseX = (noiseX + noiseX[::-1,::-1])/2
            noiseY = (noiseY + noiseY[::-1,::-1])/2
            addNoiseX = (addNoiseX + addNoiseX[::-1,::-1])/2 * (currentTime <= dampingCharTime)
            addNoiseY = (addNoiseY + addNoiseY[::-1,::-1])/2 * (currentTime <= dampingCharTime)

            hu_stocha[1:-1, 1:-1] = 0.5 * (eta[1:-1, 1:-1] + eta[1:-1, 2:]) * noiseX[1:-1, 1:-1]
            hv_stocha[1:-1, 1:-1] = 0.5 * (eta[1:-1, 1:-1] + eta[2:, 1:-1]) * noiseY[1:-1, 1:-1]
            hu_stocha = enforce_boundaries(hu_stocha, 'u')
            hv_stocha = enforce_boundaries(hv_stocha, 'v')

            dh_new_stocha[1:-1, 1:-1] = -(
                  (hu_stocha[1:-1, 1:-1] - hu_stocha[1:-1, :-2]) / dx
                + (hv_stocha[1:-1, 1:-1] - hv_stocha[:-2, 1:-1]) / dy
            )
            dh_new_stocha = enforce_boundaries(dh_new_stocha, 'h')

            dhu_new_stocha[1:-1, 1:-1] = - (
                  (hu[1:-1, 2:]*noiseX[1:-1, 2:] - hu[1:-1, :-2]*noiseX[1:-1, :-2]) /(2*dx)
                + (hu[2:, 1:-1]*noiseY[2:, 1:-1] - hu[:-2, 1:-1]*noiseY[:-2, 1:-1]) /(2*dy)
            ) + coriolis_param[1:-1, 1:-1] * hc[1:-1, 1:-1] * noiseY[1:-1, 1:-1] + (localx_h*addNoiseX)[1:-1, 1:-1]
            dhu_new_stocha = enforce_boundaries(dhu_new_stocha, 'u')

            dhv_new_stocha[1:-1, 1:-1] =  - (
                  (hv[1:-1, 2:]*noiseX[1:-1, 2:] - hv[1:-1, :-2]*noiseX[1:-1, :-2]) /(2*dx)
                + (hv[2:, 1:-1]*noiseY[2:, 1:-1] - hv[:-2, 1:-1]*noiseY[:-2, 1:-1]) /(2*dy)
            ) - coriolis_param[1:-1, 1:-1] * hc[1:-1, 1:-1] * noiseX[1:-1, 1:-1] + (localy_h*addNoiseY)[1:-1, 1:-1]
            dhv_new_stocha[:, :] = enforce_boundaries(dhv_new_stocha, 'v')

            
            # SGN martingale term
            if simulationType == "Serre-Green-Naghdi":
                toBeInterp_dyX = np.zeros_like(hu)
                toBeInterp_dxY = np.zeros_like(hv)

                loc_phix = np.zeros_like(noiseX)
                loc_phiy = np.zeros_like(noiseY)
                loc_phix[1:-1, 1:-1] = (noiseX[1:-1, :-2]+noiseX[1:-1, 1:-1])/2
                loc_phiy[1:-1, 1:-1] = (noiseY[:-2, 1:-1]+noiseY[1:-1, 1:-1])/2
                loc_phix = enforce_boundaries(loc_phix, 'h')
                loc_phiy = enforce_boundaries(loc_phiy, 'h')
                
                sgn_term_stocha_dxX[1:-1, 1:-1] = (hc[1:-1, 2:]**3/3*div_vit[1:-1, 2:]*loc_phix[1:-1, 2:]
                                                -  hc[1:-1, 1:-1]**3/3*div_vit[1:-1, 1:-1]*loc_phix[1:-1, 1:-1])/dx
                sgn_term_stocha_dyY[1:-1, 1:-1] = (hc[2:, 1:-1]**3/3*div_vit[2:, 1:-1]*loc_phiy[2:, 1:-1]
                                                -  hc[1:-1, 1:-1]**3/3*div_vit[1:-1, 1:-1]*loc_phiy[1:-1, 1:-1])/dy
                
                #cross terms must be interpolated
                toBeInterp_dyX[1:-1, 1:-1] = (hc[2:, 1:-1]**3/3*div_vit[2:, 1:-1]*loc_phix[2:, 1:-1]
                                            -  hc[1:-1, 1:-1]**3/3*div_vit[1:-1, 1:-1]*loc_phix[1:-1, 1:-1])/dy
                toBeInterp_dyX = enforce_boundaries(toBeInterp_dyX, 'v')
                sgn_term_stocha_dyX[1:-1, 1:-1] = 0.25*(toBeInterp_dyX[1:-1, 1:-1] + toBeInterp_dyX[1:-1, 2:]
                                                        +toBeInterp_dyX[:-2, 1:-1] + toBeInterp_dyX[:-2, 2:])
                
                toBeInterp_dxY[1:-1, 1:-1] = (hc[1:-1, 2:]**3/3*div_vit[1:-1, 2:]*loc_phiy[1:-1, 2:]
                                            -  hc[1:-1, 1:-1]**3/3*div_vit[1:-1, 1:-1]*loc_phiy[1:-1, 1:-1])/dx
                toBeInterp_dxY = enforce_boundaries(toBeInterp_dxY, 'u')
                sgn_term_stocha_dxY[1:-1, 1:-1] = 0.25*(toBeInterp_dxY[1:-1, 1:-1] + toBeInterp_dxY[2:, 1:-1]
                                                        +toBeInterp_dxY[1:-1, :-2] + toBeInterp_dxY[2:, :-2])
                
                sgn_term_stocha_dxX = enforce_boundaries(sgn_term_stocha_dxX, 'u')
                sgn_term_stocha_dyX = enforce_boundaries(sgn_term_stocha_dyX, 'u')    
                sgn_term_stocha_dxY = enforce_boundaries(sgn_term_stocha_dxY, 'v')    
                sgn_term_stocha_dyY = enforce_boundaries(sgn_term_stocha_dyY, 'v')    
    
                dhu_new_stocha[1:-1, 1:-1] += (beta ** 2) * ((sgn_term_stocha_dxX[1:-1, 2:] - sgn_term_stocha_dxX[1:-1, :-2]) / (2*dx)
                                                          +  (sgn_term_stocha_dxY[2:, 1:-1] - sgn_term_stocha_dxY[:-2, 1:-1]) / (2*dy))
                dhv_new_stocha[1:-1, 1:-1] += (beta ** 2) * ((sgn_term_stocha_dyX[1:-1, 2:] - sgn_term_stocha_dyX[1:-1, :-2]) / (2*dx)
                                                          +  (sgn_term_stocha_dyY[2:, 1:-1] - sgn_term_stocha_dyY[:-2, 1:-1]) / (2*dy))
                dhu_new_stocha = enforce_boundaries(dhu_new_stocha, 'u')
                dhv_new_stocha = enforce_boundaries(dhv_new_stocha, 'v')
        
            #Euler-Heun method for stochastic terms
            
            hutemp[1:-1, 1:-1] = hu[1:-1, 1:-1]
            hvtemp[1:-1, 1:-1] = hv[1:-1, 1:-1]
            htemp[1:-1, 1:-1] = hc[1:-1, 1:-1]
            
            hutemp = enforce_boundaries(hutemp, 'u')
            hvtemp = enforce_boundaries(hvtemp, 'v')
            htemp = enforce_boundaries(htemp, 'h')
            
            hutemp[1:-1, 1:-1] += dhu_new_stocha[1:-1, 1:-1]
            hvtemp[1:-1, 1:-1] += dhv_new_stocha[1:-1, 1:-1]
            htemp[1:-1, 1:-1]  += dh_new_stocha[1:-1, 1:-1]
            hutemp = enforce_boundaries(hutemp, 'u')
            hvtemp = enforce_boundaries(hvtemp, 'v')
            htemp = enforce_boundaries(htemp, 'h')

            localx_htemp = np.zeros_like(hc) + depth
            localy_htemp = np.zeros_like(hc) + depth
            localx_htemp[1:-1, 1:-1] = 0.5 * (htemp[1:-1, 1:-1] + htemp[1:-1, 2:])
            localy_htemp[1:-1, 1:-1] = 0.5 * (htemp[1:-1, 1:-1] + htemp[2:, 1:-1])
            localx_htemp = enforce_boundaries(localx_htemp, 'h')
            localy_htemp = enforce_boundaries(localy_htemp, 'h')

            etatemp = np.pad(htemp[1:-1, 1:-1] - depth, 1, 'edge')
            etatemp = enforce_boundaries(etatemp, 'h')
            
            hu_stocha_temp[1:-1, 1:-1] = 0.5 * (etatemp[1:-1, 1:-1] + etatemp[1:-1, 2:]) * noiseX[1:-1, 1:-1]
            hv_stocha_temp[1:-1, 1:-1] = 0.5 * (etatemp[1:-1, 1:-1] + etatemp[2:, 1:-1]) * noiseY[1:-1, 1:-1]
            hu_stocha_temp = enforce_boundaries(hu_stocha_temp, 'u')
            hv_stocha_temp = enforce_boundaries(hv_stocha_temp, 'v')

            dh_new_stocha[1:-1, 1:-1] += - (
                (hu_stocha_temp[1:-1, 1:-1] - hu_stocha_temp[1:-1, :-2]) / dx
                + (hv_stocha_temp[1:-1, 1:-1] - hv_stocha_temp[:-2, 1:-1]) / dy
            )
            dh_new_stocha = enforce_boundaries(dh_new_stocha, 'h')

            dhu_new_stocha[1:-1, 1:-1] += (
                    (hutemp[1:-1, 2:]*noiseX[1:-1, 2:] - hutemp[1:-1, :-2]*noiseX[1:-1, :-2]) /(2*dx)
                  + (hutemp[2:, 1:-1]*noiseY[2:, 1:-1] - hutemp[:-2, 1:-1]*noiseY[:-2, 1:-1]) /(2*dy)
            ) + coriolis_param[1:-1, 1:-1] * hc[1:-1, 1:-1] * noiseY[1:-1, 1:-1] #+ (localx_htemp*addNoiseX)[1:-1, 1:-1]
            dhu_new_stocha = enforce_boundaries(dhu_new_stocha, 'u')
        
            dhv_new_stocha[1:-1, 1:-1] += (
                    (hvtemp[1:-1, 2:]*noiseX[1:-1, 2:] - hvtemp[1:-1, :-2]*noiseX[1:-1, :-2]) /(2*dx)
                  + (hvtemp[2:, 1:-1]*noiseY[2:, 1:-1] - hvtemp[:-2, 1:-1]*noiseY[:-2, 1:-1]) /(2*dy)
            ) - coriolis_param[1:-1, 1:-1] * hc[1:-1, 1:-1] * noiseX[1:-1, 1:-1] #+ (localy_htemp*addNoiseY)[1:-1, 1:-1]
            dhv_new_stocha = enforce_boundaries(dhv_new_stocha, 'v')

            
            # SGN stochastic term (Euler-Heun)
            if simulationType == "Serre-Green-Naghdi":
                
                localx_h_temp = np.zeros_like(htemp) + depth
                localy_h_temp = np.zeros_like(htemp) + depth
                localx_h_temp[1:-1, 1:-1] = 0.5 * (htemp[1:-1, 1:-1] + htemp[1:-1, 2:])
                localy_h_temp[1:-1, 1:-1] = 0.5 * (htemp[1:-1, 1:-1] + htemp[2:, 1:-1])
                localx_h_temp = enforce_boundaries(localx_h_temp, 'h')
                localy_h_temp = enforce_boundaries(localy_h_temp, 'h')

                div_vit_temp = np.empty((n_y, n_x))
                div_vit_temp[1:-1, 1:-1] = (((hutemp/localx_h_temp)[1:-1, 1:-1] - (hutemp/localx_h_temp)[1:-1, :-2]) / dx
                                         +  ((hvtemp/localy_h_temp)[1:-1, 1:-1] - (hvtemp/localy_h_temp)[:-2, 1:-1]) / dy    )
                div_vit_temp = enforce_boundaries(div_vit_temp, 'h')

                toBeInterp_dyX = np.zeros_like(hu)
                toBeInterp_dxY = np.zeros_like(hv)

                loc_phix = np.zeros_like(noiseX)
                loc_phiy = np.zeros_like(noiseY)
                loc_phix[1:-1, 1:-1] = (noiseX[1:-1, :-2]+noiseX[1:-1, 1:-1])/2
                loc_phiy[1:-1, 1:-1] = (noiseY[:-2, 1:-1]+noiseY[1:-1, 1:-1])/2
                loc_phix = enforce_boundaries(loc_phix, 'h')
                loc_phiy = enforce_boundaries(loc_phiy, 'h')
                
                sgn_term_stocha_dxX[1:-1, 1:-1] = (hc[1:-1, 2:]**3/3*div_vit_temp[1:-1, 2:]*loc_phix[1:-1, 2:]
                                                -  hc[1:-1, 1:-1]**3/3*div_vit_temp[1:-1, 1:-1]*loc_phix[1:-1, 1:-1])/dx
                sgn_term_stocha_dyY[1:-1, 1:-1] = (hc[2:, 1:-1]**3/3*div_vit_temp[2:, 1:-1]*loc_phiy[2:, 1:-1]
                                                -  hc[1:-1, 1:-1]**3/3*div_vit_temp[1:-1, 1:-1]*loc_phiy[1:-1, 1:-1])/dy
                
                #cross terms must be interpolated
                toBeInterp_dyX[1:-1, 1:-1] = (hc[2:, 1:-1]**3/3*div_vit_temp[2:, 1:-1]*loc_phix[2:, 1:-1]
                                            - hc[1:-1, 1:-1]**3/3*div_vit_temp[1:-1, 1:-1]*loc_phix[1:-1, 1:-1])/dy
                toBeInterp_dyX = enforce_boundaries(toBeInterp_dyX, 'v')
                sgn_term_stocha_dyX[1:-1, 1:-1] = 0.25*(toBeInterp_dyX[1:-1, 1:-1] + toBeInterp_dyX[1:-1, 2:]
                                                        +toBeInterp_dyX[:-2, 1:-1] + toBeInterp_dyX[:-2, 2:])
                
                toBeInterp_dxY[1:-1, 1:-1] = (hc[1:-1, 2:]**3/3*div_vit_temp[1:-1, 2:]*loc_phiy[1:-1, 2:]
                                            - hc[1:-1, 1:-1]**3/3*div_vit_temp[1:-1, 1:-1]*loc_phiy[1:-1, 1:-1])/dx
                toBeInterp_dxY = enforce_boundaries(toBeInterp_dxY, 'u')
                sgn_term_stocha_dxY[1:-1, 1:-1] = 0.25*(toBeInterp_dxY[1:-1, 1:-1] + toBeInterp_dxY[2:, 1:-1]
                                                        +toBeInterp_dxY[1:-1, :-2] + toBeInterp_dxY[2:, :-2])
                
                sgn_term_stocha_dxX = enforce_boundaries(sgn_term_stocha_dxX, 'u')
                sgn_term_stocha_dyX = enforce_boundaries(sgn_term_stocha_dyX, 'u')    
                sgn_term_stocha_dxY = enforce_boundaries(sgn_term_stocha_dxY, 'v')    
                sgn_term_stocha_dyY = enforce_boundaries(sgn_term_stocha_dyY, 'v')    
    
                dhu_new_stocha[1:-1, 1:-1] += (beta ** 2) * ((sgn_term_stocha_dxX[1:-1, 2:] - sgn_term_stocha_dxX[1:-1, :-2]) / (2*dx)
                                                          +  (sgn_term_stocha_dxY[2:, 1:-1] - sgn_term_stocha_dxY[:-2, 1:-1]) / (2*dy))
                dhv_new_stocha[1:-1, 1:-1] += (beta ** 2) * ((sgn_term_stocha_dyX[1:-1, 2:] - sgn_term_stocha_dyX[1:-1, :-2]) / (2*dx)
                                                          +  (sgn_term_stocha_dyY[2:, 1:-1] - sgn_term_stocha_dyY[:-2, 1:-1]) / (2*dy))
                dhu_new_stocha = enforce_boundaries(dhu_new_stocha, 'u')
                dhv_new_stocha = enforce_boundaries(dhv_new_stocha, 'v')
            
            dh_new_stocha  *= 0.5
            dhu_new_stocha *= 0.5
            dhv_new_stocha *= 0.5

            # no inversion in SV model, but we add some numerical diffusion 
            if simulationType == "Saint-Venant" and (periodic_boundary_x and periodic_boundary_y):
                k_x = np.fft.fftfreq(n_x, d=dx)
                k_y = np.fft.fftfreq(n_y, d=dy)
                k_x, k_y = np.meshgrid(k_x, k_y, indexing='ij')
                norm2k = (k_x**2+k_y**2)[1:-1, 1:-1]

                dh_new_fourier[1:-1, 1:-1] = np.fft.fft2(dh_new[1:-1, 1:-1])
                dhu_new_fourier[1:-1, 1:-1] = np.fft.fft2(dhu_new[1:-1, 1:-1])
                dhv_new_fourier[1:-1, 1:-1] = np.fft.fft2(dhv_new[1:-1, 1:-1])
                dh_new[1:-1, 1:-1] = np.fft.ifft2(dh_new_fourier[1:-1, 1:-1]/(1+diffCoeff*norm2k)).real
                dhu_new[1:-1, 1:-1] = np.fft.ifft2(dhu_new_fourier[1:-1, 1:-1]/(1+diffCoeff*norm2k)).real
                dhv_new[1:-1, 1:-1] = np.fft.ifft2(dhv_new_fourier[1:-1, 1:-1]/(1+diffCoeff*norm2k)).real
                dh_new = enforce_boundaries(dh_new, 'h')
                dhu_new = enforce_boundaries(dhu_new, 'u')
                dhv_new = enforce_boundaries(dhv_new, 'v')

                dh_newStocha_fourier[1:-1, 1:-1] = np.fft.fft2(dh_new_stocha[1:-1, 1:-1])
                dhu_newStocha_fourier[1:-1, 1:-1] = np.fft.fft2(dhu_new_stocha[1:-1, 1:-1])
                dhv_newStocha_fourier[1:-1, 1:-1] = np.fft.fft2(dhv_new_stocha[1:-1, 1:-1])
                dh_new_stocha[1:-1, 1:-1] = np.fft.ifft2(dh_newStocha_fourier[1:-1, 1:-1]/(1+diffCoeff*norm2k)).real
                dhu_new_stocha[1:-1, 1:-1] = np.fft.ifft2(dhu_newStocha_fourier[1:-1, 1:-1]/(1+diffCoeff*norm2k)).real
                dhv_new_stocha[1:-1, 1:-1] = np.fft.ifft2(dhv_newStocha_fourier[1:-1, 1:-1]/(1+diffCoeff*norm2k)).real
                dh_new_stocha = enforce_boundaries(dh_new_stocha, 'h')
                dhu_new_stocha = enforce_boundaries(dhu_new_stocha, 'u')
                dhv_new_stocha = enforce_boundaries(dhv_new_stocha, 'v')
            '''
            ravelled_hu_new_stocha = np.ravel(dhu_new_stocha[1:-1, 1:-1])
            ravelled_hv_new_stocha = np.ravel(dhv_new_stocha[1:-1, 1:-1])
            ravelledNewPot = spsolve(diffusionMatrix,ravelledNewPot)
            ravelledNewH = spsolve(diffusionMatrix,ravelledNewH)

            newPot[1:-1,1:-1] = np.reshape(ravelledNewPot, newshape = (n_y-2,n_x-2))
            newH[1:-1,1:-1] = np.reshape(ravelledNewH, newshape = (n_y-2,n_x-2))
            newPot = enforce_boundaries(newPot, 'h')
            newH = enforce_boundaries(newH, 'h')
            '''

            #inversion in Boussinesq model (computed with Fourier transforms)    
            if simulationType == "Boussinesq" and (periodic_boundary_x and periodic_boundary_y):
                # creating matrices for inversion computation
                k_x = np.fft.fftfreq(n_x, d=dx)
                k_y = np.fft.fftfreq(n_y, d=dy)

                k_x, k_y = np.meshgrid(k_x, k_y, indexing='ij')

                cst = (beta ** 2) * depth**2 / 3
                det_mat[:, :] = 1 - cst * (k_x[:, :] ** 2 + k_y[:, :] ** 2)

                diag_term_11[:, :] = (1 - cst * (k_y[:, :] ** 2)) / det_mat[:, :]
                diag_term_22[:, :] = (1 - cst * (k_x[:, :] ** 2)) / det_mat[:, :]
                non_diag_term_12[:, :] = (cst * k_x[:, :] * k_y[:, :]) / det_mat[:, :]
                non_diag_term_21[:, :] = (cst * k_x[:, :] * k_y[:, :]) / det_mat[:, :]

                #inversion
                dhu_new_fourier[1:-1, 1:-1] = np.fft.fft2(dhu_new[1:-1, 1:-1])
                dhv_new_fourier[1:-1, 1:-1] = np.fft.fft2(dhv_new[1:-1, 1:-1]) 
                dhu_new_complex = diag_term_11[1:-1, 1:-1] * dhu_new_fourier[1:-1, 1:-1] + non_diag_term_12[1:-1, 1:-1] * dhv_new_fourier[1:-1, 1:-1]
                dhv_new_complex = non_diag_term_21[1:-1, 1:-1] * dhu_new_fourier[1:-1, 1:-1] + diag_term_22[1:-1, 1:-1] * dhv_new_fourier[1:-1, 1:-1]
                
                dhu_new[1:-1, 1:-1] = np.fft.ifft2(dhu_new_complex).real
                dhv_new[1:-1, 1:-1] = np.fft.ifft2(dhv_new_complex).real
                dhu_new = enforce_boundaries(dhu_new, 'u')
                dhv_new = enforce_boundaries(dhv_new, 'v')

                dhu_newStocha_fourier[1:-1, 1:-1] = np.fft.fft2(dhu_new_stocha[1:-1, 1:-1])
                dhv_newStocha_fourier[1:-1, 1:-1] = np.fft.fft2(dhv_new_stocha[1:-1, 1:-1])
                dhu_newStocha_complex = diag_term_11[1:-1, 1:-1] * dhu_newStocha_fourier[1:-1, 1:-1] + non_diag_term_12[1:-1, 1:-1] * dhv_newStocha_fourier[1:-1, 1:-1]
                dhv_newStocha_complex = non_diag_term_21[1:-1, 1:-1] * dhu_newStocha_fourier[1:-1, 1:-1] + diag_term_22[1:-1, 1:-1] * dhv_newStocha_fourier[1:-1, 1:-1]
    
                dhu_new_stocha[1:-1, 1:-1] = np.fft.ifft2(dhu_newStocha_complex).real
                dhv_new_stocha[1:-1, 1:-1] = np.fft.ifft2(dhv_newStocha_complex).real
                dhu_new_stocha = enforce_boundaries(dhu_new_stocha, 'u')
                dhv_new_stocha = enforce_boundaries(dhv_new_stocha, 'v')
            
            # matrix inversion in SGN model (not computed with Fourier transforms)    
            if simulationType == "Serre-Green-Naghdi":
                nPoints = n_x*n_y
                nInternalPoints = (n_x-2)*(n_y-2)
                internalH = hc[1:-1,1:-1]
                generalMatrix = np.eye(2*nPoints)
                cst = beta**2/3
                
                dmom = np.concatenate((nx.reshape(dhu_new[1:-1,1:-1], (nInternalPoints,)), nx.reshape(dhv_new[1:-1,1:-1], (nInternalPoints,))))
                internalH = hc[1:-1,1:-1]
                internalLocalHx = localx_h[1:-1,1:-1]
                internalLocalHy = localy_h[1:-1,1:-1]
                generalMatrix = np.zeros((2*nInternalPoints, 2*nInternalPoints))
                for i in range(n_x-2):
                    for j in range(n_y-2):
                        #matrix terms associated to (hu)_ij
                        generalMatrix[j*(n_x-2) + i,j*(n_x-2) + i]             = 1 + cst*(internalH[j,i]**3 + internalH[j,(i+1)%(n_x-2)]**3)/dx**2   / internalLocalHx[j,i]
                        generalMatrix[j*(n_x-2) + i,j*(n_x-2) + (i+1)%(n_x-2)] =   - cst* internalH[j,(i+1)%(n_x-2)]**3/dx**2                        / internalLocalHx[j,(i+1)%(n_x-2)]
                        generalMatrix[j*(n_x-2) + i,j*(n_x-2) + (i-1)%(n_x-2)] =   - cst* internalH[j,i]**3/dx**2                                    / internalLocalHx[j,(i-1)%(n_x-2)]
                        
                        generalMatrix[j*(n_x-2) + i,nInternalPoints+j*(n_x-2) + i]                         =  cst*internalH[j,i]**3/dx/dy               / internalLocalHy[j,i]
                        generalMatrix[j*(n_x-2) + i,nInternalPoints+j*(n_x-2) + (i+1)%(n_x-2)]             = -cst*internalH[j,(i+1)%(n_x-2)]**3/dx/dy   / internalLocalHy[j,(i+1)%(n_x-2)]
                        generalMatrix[j*(n_x-2) + i,nInternalPoints+(j-1)%(n_y-2)*(n_x-2) + (i+1)%(n_x-2)] =  cst*internalH[j,(i+1)%(n_x-2)]**3/dx/dy   / internalLocalHy[(j-1)%(n_y-2),(i+1)%(n_x-2)]
                        generalMatrix[j*(n_x-2) + i,nInternalPoints+(j-1)%(n_y-2)*(n_x-2) + i]             = -cst*internalH[j,i]**3/dx/dy               / internalLocalHy[(j-1)%(n_y-2),i]

                        #matrix terms associated to (hv)_ij
                        generalMatrix[nInternalPoints+j*(n_x-2) + i,nInternalPoints+j*(n_x-2) + i]                 = 1 + cst*(hc[j,i]**3 + hc[(j+1)%(n_y-2),i]**3)/dy**2   / internalLocalHy[j,i]
                        generalMatrix[nInternalPoints+j*(n_x-2) + i,nInternalPoints+(j+1)%(n_y-2)*(n_x-2) + i]     =   - cst* hc[(j+1)%(n_y-2),i]**3/dy**2                 / internalLocalHy[(j+1)%(n_y-2),i]
                        generalMatrix[nInternalPoints+j*(n_x-2) + i,nInternalPoints+(j-1)%(n_y-2)*(n_x-2) + i]     =   - cst* hc[j,i]**3/dy**2                             / internalLocalHy[(j-1)%(n_y-2),i]
                        
                        generalMatrix[nInternalPoints+j*(n_x-2) + i,j*(n_x-2) + i]                         =  cst*hc[j,i]**3/dx/dy              / internalLocalHx[j,i]
                        generalMatrix[nInternalPoints+j*(n_x-2) + i,(j+1)%(n_y-2)*(n_x-2) + i]             = -cst*hc[(j+1)%(n_y-2),i]**3/dx/dy  / internalLocalHx[(j+1)%(n_y-2),i]
                        generalMatrix[nInternalPoints+j*(n_x-2) + i,(j+1)%(n_y-2)*(n_x-2) + (i-1)%(n_x-2)] =  cst*hc[(j+1)%(n_y-2),i]**3/dx/dy  / internalLocalHx[(j+1)%(n_y-2),(i-1)%(n_x-2)]
                        generalMatrix[nInternalPoints+j*(n_x-2) + i,j*(n_x-2) + (i-1)%(n_x-2)]             = -cst*hc[j,i]**3/dx/dy              / internalLocalHx[j,(i-1)%(n_x-2)]
                
                matrixToInvert = scsp.csr_matrix(generalMatrix)

                #direct method
                dmom = scsp.linalg.spsolve(matrixToInvert, dmom)

                dhu_new[1:-1,1:-1] = nx.reshape(dmom[:nInternalPoints ], (n_y-2, n_x-2))
                dhv_new[1:-1,1:-1] = nx.reshape(dmom[ nInternalPoints:], (n_y-2, n_x-2))
                dhu_new = enforce_boundaries(dhu_new, 'u')
                dhv_new = enforce_boundaries(dhv_new, 'v')
                
                #direct method
                dmomStocha = np.concatenate((nx.reshape(dhu_new_stocha[1:-1,1:-1], (nInternalPoints,)), nx.reshape(dhv_new_stocha[1:-1,1:-1], (nInternalPoints,)))).copy()
                dmomStocha = scsp.linalg.spsolve(matrixToInvert, dmomStocha)
                
                dhu_new_stocha[1:-1,1:-1] = nx.reshape(dmomStocha[:nInternalPoints ], (n_y-2, n_x-2))
                dhv_new_stocha[1:-1,1:-1] = nx.reshape(dmomStocha[ nInternalPoints:], (n_y-2, n_x-2))
                dhu_new_stocha = enforce_boundaries(dhu_new_stocha, 'u')
                dhv_new_stocha = enforce_boundaries(dhv_new_stocha, 'v')
            
            #time increment
            if first_step:
                hu[1:-1, 1:-1] += dt * dhu_new[1:-1, 1:-1]
                hv[1:-1, 1:-1] += dt * dhv_new[1:-1, 1:-1]
                h[1:-1, 1:-1]  += dt * dh_new[1:-1, 1:-1]

                hu[1:-1, 1:-1] += dhu_new_stocha[1:-1, 1:-1]
                hv[1:-1, 1:-1] += dhv_new_stocha[1:-1, 1:-1]
                h[1:-1, 1:-1]  += dh_new_stocha[1:-1, 1:-1]

                hu = enforce_boundaries(hu, 'u')
                hv = enforce_boundaries(hv, 'v')
                h = enforce_boundaries(h, 'h')
                first_step = False
            else:
                hu[1:-1, 1:-1] += dt * (
                    adams_bashforth_a * dhu_new[1:-1, 1:-1]
                    + adams_bashforth_b * dhu[1:-1, 1:-1]
                )
                hv[1:-1, 1:-1] += dt * (
                    adams_bashforth_a * dhv_new[1:-1, 1:-1]
                    + adams_bashforth_b * dhv[1:-1, 1:-1]
                )
                h[1:-1, 1:-1] += dt * (
                    adams_bashforth_a * dh_new[1:-1, 1:-1]
                    + adams_bashforth_b * dh[1:-1, 1:-1]
                )

                hu[1:-1, 1:-1] += dhu_new_stocha[1:-1, 1:-1]
                hv[1:-1, 1:-1] += dhv_new_stocha[1:-1, 1:-1]
                h[1:-1, 1:-1]  += dh_new_stocha[1:-1, 1:-1]
                hu = enforce_boundaries(hu, 'u')
                hv = enforce_boundaries(hv, 'v')
                h = enforce_boundaries(h, 'h')
        
            if lateral_viscosity > 0:    
                hu[1:-1, 1:-1] += dt * (fue[1:-1, 1:-1] + fun[1:-1, 1:-1])
                hv[1:-1, 1:-1] += dt * (fve[1:-1, 1:-1] + fvn[1:-1, 1:-1])
            hu = enforce_boundaries(hu, 'u')
            hv = enforce_boundaries(hv, 'v')

            #h[1:-1,1:-1]  = np.real(np.fft.ifft2(antiAlias[1:-1,1:-1]*np.fft.fft2(h[1:-1,1:-1])))
            #hu[1:-1,1:-1] = np.real(np.fft.ifft2(antiAlias[1:-1,1:-1]*np.fft.fft2(hu[1:-1,1:-1])))
            #hv[1:-1,1:-1] = np.real(np.fft.ifft2(antiAlias[1:-1,1:-1]*np.fft.fft2(hv[1:-1,1:-1])))
            h = enforce_boundaries(h, 'h')
            hu = enforce_boundaries(hu, 'u')
            hv = enforce_boundaries(hv, 'v')

            # rotate quantities
            dhu[...] = dhu_new
            dhv[...] = dhv_new
            dh[...] = dh_new

            currentTime += dt

            yield h, hu, hv


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

        typicalFreqX = []
        typicalFreqY = []
        typicalFreqX2 = []
        typicalFreqY2 = []
        
        if save_data:
            spec = [n_x, n_y, dx, dy]
            export_to_csv(spec, path_to_store+'//spec.csv')

        for iteration, (h, hu, hv) in enumerate(model):
            if iteration % plot_every == 0:
                t = iteration * dt

                a=abs(np.fft.fft2(h[1:-1, 1:-1]))
                a[0,0] = 0
                couple = (0,0) #np.argwhere(a==a.max())[0]
                wx,wy = couple[0], couple[1]
                a[wx,wy] = 0
                couple = (0,0) #np.argwhere(a==a.max())[0]
                wx2,wy2 = couple[0], couple[1]

                typicalFreqX.append(wx)
                typicalFreqY.append(wy)
                typicalFreqX2.append(wx2)
                typicalFreqY2.append(wy2)

                totalMass.append(np.sum(h[1:-1,1:-1])*dx/l_x*dy/l_y)
                totalMomX.append(dx/l_x*dy/l_y*np.sum(hu[1:-1,1:-1]))
                totalMomY.append(dx/l_x*dy/l_y*np.sum(hv[1:-1,1:-1]))

                if not(energySGN):
                    local_hu = np.zeros_like(hu)
                    local_hv = np.zeros_like(hv)
                    local_hu[1:-1, 1:-1] = 0.5 * (hu[1:-1, 1:-1] + hu[:-2, 1:-1])
                    local_hv[1:-1, 1:-1] = 0.5 * (hv[1:-1, 1:-1] + hv[1:-1, :-2])
                    local_hu = enforce_boundaries(local_hu, 'u')
                    local_hv = enforce_boundaries(local_hv, 'v')
                    totalEnerg.append(np.sum((local_hu**2/h)[1:-1,1:-1]/2 + (local_hv**2/h)[1:-1,1:-1]/2 + (gravity*h**2)[1:-1,1:-1]/2)*dx/l_x*dy/l_y)
                
                else:
                    local_hu = np.zeros_like(hu)
                    local_hv = np.zeros_like(hv)
                    local_hu[1:-1, 1:-1] = 0.5 * (hu[1:-1, 1:-1] + hu[:-2, 1:-1])
                    local_hv[1:-1, 1:-1] = 0.5 * (hv[1:-1, 1:-1] + hv[1:-1, :-2])
                    local_hu = enforce_boundaries(local_hu, 'u')
                    local_hv = enforce_boundaries(local_hv, 'v')

                    divu = np.zeros_like(hu)
                    localx_h = np.zeros_like(h) + depth
                    localy_h = np.zeros_like(h) + depth
                    localx_h[1:-1, 1:-1] = 0.5 * (h[1:-1, 1:-1] + h[1:-1, 2:])
                    localy_h[1:-1, 1:-1] = 0.5 * (h[1:-1, 1:-1] + h[2:, 1:-1])
                    localx_h = enforce_boundaries(localx_h, 'h')
                    localy_h = enforce_boundaries(localy_h, 'h')

                    divu[1:-1, 1:-1] = (((hu/localx_h)[1:-1, 1:-1] - (hu/localx_h)[:-2, 1:-1])/dx
                                        + ((hv/localy_h)[1:-1, 1:-1] - (hv/localy_h)[1:-1,:-2])/dy)
                    
                    totalEnerg.append( np.sum((local_hu**2/h)[1:-1,1:-1]/2 + (local_hv**2/h)[1:-1,1:-1]/2 + (gravity*h**2)[1:-1,1:-1] + beta**2 /6 * (h**3)[1:-1,1:-1] * (divu**2)[1:-1,1:-1] )*dx/l_x*dy/l_y )
                
                if make_plots:
                    update_plot(t, h, hu, hv, ax, count)

                if save_data:
                    export_to_csv(h, path_to_store+'height//'+str(count)+'.csv')
                    export_to_csv(hu, path_to_store+'xmom//'+str(count)+'.csv')
                    export_to_csv(hv, path_to_store+'ymom//'+str(count)+'.csv')

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
            ax_bis.set_ylabel('Momentum error', color='blue')
            ax_bis.tick_params(axis='y', labelcolor='blue')
            ax_bis.legend(['X Momentum', 'Y Momentum'], loc='lower left')
            
            cenerg = ax[1].plot(timespace, totalEnerg/totalEnerg[0] -1, color='red')
            ax[1].set_ylabel('Energy error', color='red')
            ax[1].tick_params(axis='y', labelcolor='red')
            ax[1].legend(['Energy'], loc='best')

            energ_amplitude_stat = max(totalEnerg[count//2:]/totalEnerg[0]) - min(totalEnerg[count//2:]/totalEnerg[0])
            print("Maximal amplitude of energy variation over the second half of the simulation (to be interpreted as stationary regime if the simulation in long enough):",
                                    energ_amplitude_stat)
            plt.show()

        if makegif:
            frames = np.stack([iio.imread(f"images/{i}.pdf") for i in range(count)] ) #+ [iio.imread(f"images/{i}.jpg") for i in range(count-2, 0, -1)], axis = 0)
            iio.mimwrite('testTest.gif', frames, format='GIF', fps = 24, loop=0)
