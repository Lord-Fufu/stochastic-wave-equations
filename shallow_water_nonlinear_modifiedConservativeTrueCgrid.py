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
from fipy import numerix as nx
import pandas as pd
import os

#Choose simulation type. Can be "Saint-Venant", "Boussinesq" or "Serre-Green-Naghdi".
simulationType = "Serre-Green-Naghdi"

n_data=1
for i_trajectory in range(n_data):

    #--------------------------------------------------------------------------------------------------------------
    #---------------------------------------------SIMULATION PARAMETERS-----------------------------------------------
    #--------------------------------------------------------------------------------------------------------------

    # maximum number of timesteps
    itmax = float('infinity')

    # grid setup
    n_x = 51
    dx = 1e4
    l_x = n_x * dx

    n_y = 51
    dy = 1e4
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
    eps=0.5
    beta=0.1
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
    plot_range = 0.1*amplitude
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
    u0 = np.zeros_like(u0)

    h_geostrophy = np.cumsum(-dy * u0 * coriolis_param / gravity, axis=0)
    h0 = (
        depth
        + h_geostrophy
        # make sure h0 is centered around depth
        - h_geostrophy.mean()
        # small perturbation
        #+ 1 * np.sin(X / l_x * 10 * np.pi) * np.cos(Y / l_y * 8 * np.pi)
        + amplitude*np.exp(-(10**(-9)*(X-l_x/3)**2 + 10**(-9)*(Y-l_y/3)**2)**2)
        #+ amplitude*np.exp(-100 * (0.9*abs(Y / l_y - 0.5)**2+ ((Y/l_y - 0.5) - abs(X / l_x - 0.5)**(2/3))**2))
    )

    hu0 = h0 * u0
    hv0 = h0 * v0

    #--------------------------------------------------------------------------------------------------------------
    #----------------------------------------------NOISE PARAMETERS------------------------------------------------
    #---------------------------------------FOR DOUBLE PERIODIC BOUNDARIES-----------------------------------------
    #--------------------------------------------------------------------------------------------------------------

    if periodic_boundary_x and periodic_boundary_y:
        nmodes = 4
        ups = 2e3
        wavenumber = 2*np.pi*2

        coeff=-2
        coeffx=0
        coeffy=0

        l_ref = np.sqrt(l_x * l_y)
        l_waveX = (l_x + coeff*dx)
        l_waveY = (l_y + coeff*dy)

        phi1x = np.sqrt(ups)*l_waveX/l_ref*np.sin(wavenumber*((X+dx/2)/l_waveX+Y/l_waveY))
        phi1y = -np.sqrt(ups)*l_waveY/l_ref*np.sin(wavenumber*(X/l_waveX+(Y+dy/2)/l_waveY))
        phi2x = np.sqrt(ups)*l_waveX/l_ref*np.sin(wavenumber*((X+dx/2)/l_waveX-Y/l_waveY))
        phi2y = np.sqrt(ups)*l_waveY/l_ref*np.sin(wavenumber*(X/l_waveX-(Y+dy/2)/l_waveY))
        phi3x = np.sqrt(ups)*l_waveX/l_ref*np.cos(wavenumber*((X+dx/2)/l_waveX+Y/l_waveY))
        phi3y = -np.sqrt(ups)*l_waveY/l_ref*np.cos(wavenumber*(X/l_waveX+(Y+dy/2)/l_waveY))
        phi4x = np.sqrt(ups)*l_waveX/l_ref*np.cos(wavenumber*((X+dx/2)/l_waveX-Y/l_waveY))
        phi4y = np.sqrt(ups)*l_waveY/l_ref*np.cos(wavenumber*(X/l_waveX-(Y+dy/2)/l_waveY))

        phix = np.zeros((n_y, n_x, nmodes))
        phix[1:-1,1:-1,0] = phi1x[1:-1,1:-1]
        phix[1:-1,1:-1,1] = phi2x[1:-1,1:-1]
        phix[1:-1,1:-1,2] = phi3x[1:-1,1:-1]
        phix[1:-1,1:-1,3] = phi4x[1:-1,1:-1]

        phiy = np.zeros((n_y, n_x, nmodes))
        phiy[1:-1,1:-1,0] = phi1y[1:-1,1:-1]
        phiy[1:-1,1:-1,1] = phi2y[1:-1,1:-1]
        phiy[1:-1,1:-1,2] = phi3y[1:-1,1:-1]
        phiy[1:-1,1:-1,3] = phi4y[1:-1,1:-1]


    #--------------------------------------------------------------------------------------------------------------
    #----------------------------------------------NOISE PARAMETERS------------------------------------------------
    #--------------------------------------FOR DOUBLE DIRICHLET BOUNDARIES-----------------------------------------
    #--------------------------------------------------------------------------------------------------------------

    if not(periodic_boundary_x) and not(periodic_boundary_y):

        nmodes = 2
        ups = 5e3
        wavenumber = 2*np.pi
        
        K=1
        R=min(l_x/dx,l_y/dy)/2
        xcenter=l_x/2
        ycenter=l_y/2

        @np.vectorize
        def radialBoundaryFunction(xval,yval):
            locRadius = np.sqrt((xval/dx)**2 + (yval/dy)**2)
            dist = 1 - locRadius/R
            if dist > 0:
                return np.exp(K*(1-1/dist**2))
            else:
                return .0
            
        @np.vectorize
        def orthoradialBoundaryFunction(xval,yval):
            locRadius = np.sqrt((xval/dx)**2 + (yval/dy)**2)
            dist = 1 - locRadius/R
            if dist > 0:
                return ( 1  -  K*locRadius/R*2/(dist**3)  ) * np.exp(K*(1-1/dist**2))
            else:
                return .0

        @np.vectorize
        def thetaFromXY(x,y):
            if x>0:
                theta = np.arctan(y/x)
            elif x<0 and y>0:
                theta = np.arctan(y/x) + np.pi
            elif x<0 and y<0:
                theta = np.arctan(y/x) - np.pi
            elif x<0 and y==0:
                theta = np.pi
            elif x==0 and y>0:
                theta = np.pi/2
            elif x==0 and y<0:
                theta = - np.pi/2
            else:
                theta = .0
            return theta

        rbfx = radialBoundaryFunction(X-xcenter+dx/2,Y-ycenter)
        rbfy = radialBoundaryFunction(X-xcenter,Y-ycenter+dy/2)
        obfx = orthoradialBoundaryFunction(X-xcenter+dx/2,Y-ycenter)
        obfy = orthoradialBoundaryFunction(X-xcenter,Y-ycenter+dy/2)
        thetax = thetaFromXY(X-xcenter+dx/2,Y-ycenter)
        thetay = thetaFromXY(X-xcenter,Y-ycenter+dy/2)

        intWave = wavenumber/(2*np.pi)

        #MODE =            RADIAL COORDINATES              *u_r vector      +             ORTHORADIAL COORDINATES                   *u_theta vector
        phi1x = np.sqrt(ups)*rbfx*np.cos(intWave*thetax)   *np.cos(thetax)  +  np.sqrt(ups)*obfx*(-np.sin(intWave*thetax))/intWave  *(-np.sin(thetax))
        phi1y = np.sqrt(ups)*rbfy*np.cos(intWave*thetay)   *np.sin(thetay)  +  np.sqrt(ups)*obfy*(-np.sin(intWave*thetay))/intWave  *np.cos(thetay)
        phi2x = np.sqrt(ups)*rbfx*np.sin(intWave*thetax)   *np.cos(thetax)  +  np.sqrt(ups)*obfx*(np.cos(intWave*thetax))/intWave   *(-np.sin(thetax))
        phi2y = np.sqrt(ups)*rbfy*np.sin(intWave*thetay)   *np.sin(thetay)  +  np.sqrt(ups)*obfy*(np.cos(intWave*thetay))/intWave   *np.cos(thetay)

        phix = np.zeros((n_y, n_x, nmodes))
        phix[1:-1,1:-1,0] = phi1x[1:-1,1:-1]
        phix[1:-1,1:-1,1] = phi2x[1:-1,1:-1]

        phiy = np.zeros((n_y, n_x, nmodes))
        phiy[1:-1,1:-1,0] = phi1y[1:-1,1:-1]
        phiy[1:-1,1:-1,1] = phi2y[1:-1,1:-1]

    #--------------------------------------------------------------------------------------------------------------
    #---------------------------------------------AUXILIARY FUNCTIONS----------------------------------------------
    #--------------------------------------------------------------------------------------------------------------

    def prepare_plot():
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        cs = update_plot(0, h0, hu0, hv0, ax, count=0, draw=False)
        plt.colorbar(cs, label='$\\eta$ (m)')
        return fig, ax


    def update_plot(t, h, hu, hv, ax, count, draw=True):
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
            't=%5.2f days, R=%5.1f km, c=%5.1f m/s '
            % (t / 86400, rossby_radius / 1e3, phase_speed)
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
        phix[:,:,k] = enforce_boundaries(phix[:,:,k], 'u')
        phiy[:,:,k] = enforce_boundaries(phiy[:,:,k], 'v')

    if ups == 0:
        nmodes = 0

    a_xx = np.zeros_like(phix)
    a_xy = np.zeros_like(phix)
    a_yy = np.zeros_like(phix)

    us = np.zeros_like(u0)
    vs = np.zeros_like(v0)

    a_xx = phix**2
    a_xy = phix*phiy
    a_yy = phiy**2
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
        dhu_new_stocha, dhv_new_stocha, dh_new_stocha = np.empty((n_y, n_x, nmodes)), np.empty((n_y, n_x, nmodes)), np.empty((n_y, n_x, nmodes))
        hu_stocha, hv_stocha = np.empty((n_y, n_x, nmodes)), np.empty((n_y, n_x, nmodes))
        hu_stocha_temp, hv_stocha_temp = np.empty((n_y, n_x, nmodes)), np.empty((n_y, n_x, nmodes))
        fue, fun = np.empty((n_y, n_x)), np.empty((n_y, n_x))
        fve, fvn = np.empty((n_y, n_x)), np.empty((n_y, n_x))

        det_mat, diag_term_11, diag_term_22 = np.empty((n_y, n_x)), np.empty((n_y, n_x)), np.empty((n_y, n_x))
        non_diag_term_12, non_diag_term_21 = np.empty((n_y, n_x)), np.empty((n_y, n_x))
        dhu_new_fourier, dhv_new_fourier = np.empty((n_y, n_x), dtype=complex), np.empty((n_y, n_x), dtype=complex)
        dhu_newStocha_fourier, dhv_newStocha_fourier = np.empty((n_y, n_x), dtype=complex), np.empty((n_y, n_x), dtype=complex)


        # initial conditions
        h[...] = h0
        hu[...] = hu0
        hv[...] = hv0

        # boundary values of h must not be used
        h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = np.nan
        h = enforce_boundaries(h, 'h')
        hu = enforce_boundaries(hu, 'u')
        hv = enforce_boundaries(hv, 'v')

        #--------------------------------------------------------------------------------------------------------------
        #-------------------------OFFLINE COMPUTATION FOR BOUSSINESQ WITH NON-PERIODIC BOUNDARIES----------------------
        if simulationType == "Boussinesq" and (not(periodic_boundary_x) and not(periodic_boundary_y)):
            nPoints = n_x*n_y
            nInternalPoints = (n_x-2)*(n_y-2)
            matrixToInvert = np.zeros((2*nInternalPoints, 2*nInternalPoints))
            generalMatrix = np.diag(np.ones(2*nPoints))
            cst = beta**2/3
            for i in range(1,n_x-1):
                for j in range(1,n_y-1):
                    #matrix terms associated to (hu)_ij
                    generalMatrix[j*n_x + i,j*n_x + i]   = 1 + 2*cst*depth**2/dx**2
                    generalMatrix[j*n_x + i,j*n_x + i+1] =   -   cst*depth**2/dx**2
                    generalMatrix[j*n_x + i,j*n_x + i-1] =   -   cst*depth**2/dx**2
                    
                    generalMatrix[j*n_x + i,nPoints+j*n_x + i]       =  cst*depth**2/dx/dy
                    generalMatrix[j*n_x + i,nPoints+j*n_x + i+1]     = -cst*depth**2/dx/dy         
                    generalMatrix[j*n_x + i,nPoints+(j-1)*n_x + i+1] =  cst*depth**2/dx/dy        
                    generalMatrix[j*n_x + i,nPoints+(j-1)*n_x + i]   = -cst*depth**2/dx/dy

                    #matrix terms associated to (hv)_ij
                    generalMatrix[nPoints+j*n_x + i,nPoints+j*n_x + i]     = 1 + cst*depth**2/dy**2
                    generalMatrix[nPoints+j*n_x + i,nPoints+(j+1)*n_x + i] =   - cst*depth**2/dy**2
                    generalMatrix[nPoints+j*n_x + i,nPoints+(j-1)*n_x + i] =   - cst*depth**2/dy**2
                    
                    generalMatrix[nPoints+j*n_x + i,j*n_x + i]       =  cst*depth**2/dx/dy           
                    generalMatrix[nPoints+j*n_x + i,(j+1)*n_x + i]   = -cst*depth**2/dx/dy         
                    generalMatrix[nPoints+j*n_x + i,(j+1)*n_x + i-1] =  cst*depth**2/dx/dy        
                    generalMatrix[nPoints+j*n_x + i,j*n_x + i-1]     = -cst*depth**2/dx/dy          

            for i in range(0,n_x-2):
                for j in range(0,n_y-2):
                    matrixToInvert[i*(n_y-2) + j, i*(n_y-2) + j] = generalMatrix[(i+1)*n_y + j+1, (i+1)*n_y + j+1]
                    matrixToInvert[nInternalPoints + i*(n_y-2) + j, i*(n_y-2) + j] = generalMatrix[nPoints + (i+1)*n_y + j+1, (i+1)*n_y + j+1]
                    matrixToInvert[i*(n_y-2) + j, nInternalPoints + i*(n_y-2) + j] = generalMatrix[(i+1)*n_y + j+1, nPoints + (i+1)*n_y + j+1]
                    matrixToInvert[nInternalPoints + i*(n_y-2) + j, nInternalPoints + i*(n_y-2) + j] = generalMatrix[nPoints + (i+1)*n_y + j+1, nPoints + (i+1)*n_y + j+1]
            matrixToInvert = scsp.csr_matrix(matrixToInvert)
        #------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------

        first_step = True

        # time step equations
        while True:
            hc = np.pad(h[1:-1, 1:-1], 1, 'edge')
            hc = enforce_boundaries(hc, 'h')
            eta = np.pad(h[1:-1, 1:-1] - depth, 1, 'edge')
            eta = enforce_boundaries(eta, 'h')

            for k in range(nmodes):
                hu_stocha[1:-1, 1:-1, k] = 0.5 * (eta[1:-1, 1:-1] + eta[1:-1, 2:]) * phix[1:-1, 1:-1, k]
                hv_stocha[1:-1, 1:-1, k] = 0.5 * (eta[1:-1, 1:-1] + eta[2:, 1:-1]) * phiy[1:-1, 1:-1, k]
                hu_stocha[:,:,k] = enforce_boundaries(hu_stocha[:,:,k], 'u')
                hv_stocha[:,:,k] = enforce_boundaries(hv_stocha[:,:,k], 'v')

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

            dh_new_stocha[1:-1, 1:-1, :] = -(
                (hu_stocha[1:-1, 1:-1, :] - hu_stocha[1:-1, :-2, :]) / dx
                + (hv_stocha[1:-1, 1:-1, :] - hv_stocha[:-2, 1:-1, :]) / dy
            )
            for k in range(nmodes):
                dh_new_stocha[:,:,k] = enforce_boundaries(dh_new_stocha[:,:,k], 'h')

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
            ) + coriolis_param[1:-1, 1:-1] * hv[1:-1, 1:-1]
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
            ) - coriolis_param[1:-1, 1:-1] * hu[1:-1, 1:-1]
            dhv_new = enforce_boundaries(dhv_new, 'v')

            #SGN bounded variation term (no Ito-Stokes drift)
            if simulationType == "Serre-Green-Naghdi":
                div_vit, div_mom = np.empty((n_y, n_x)), np.empty((n_y, n_x))
                sgn_term_stocha_dxX, sgn_term_stocha_dyX = np.empty((n_y, n_x, nmodes)), np.empty((n_y, n_x, nmodes))
                sgn_term_stocha_dxY, sgn_term_stocha_dyY = np.empty((n_y, n_x, nmodes)), np.empty((n_y, n_x, nmodes))
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
            for k in range(nmodes):
                dhu_new_stocha[1:-1, 1:-1, k] = - (
                        (hu[1:-1, 2:]*phix[1:-1, 2:, k] - hu[1:-1, :-2]*phix[1:-1, :-2, k]) /(2*dx)
                    + (hu[2:, 1:-1]*phiy[2:, 1:-1, k] - hu[:-2, 1:-1]*phiy[:-2, 1:-1, k]) /(2*dy)
                ) + coriolis_param[1:-1, 1:-1] * hc[1:-1, 1:-1] * phiy[1:-1, 1:-1, k]
                dhu_new_stocha[:, :, k] = enforce_boundaries(dhu_new_stocha[:, :, k], 'u')

                dhv_new_stocha[1:-1, 1:-1, k] =  - (
                        (hv[1:-1, 2:]*phix[1:-1, 2:, k] - hv[1:-1, :-2]*phix[1:-1, :-2, k]) /(2*dx)
                    + (hv[2:, 1:-1]*phiy[2:, 1:-1, k] - hv[:-2, 1:-1]*phiy[:-2, 1:-1, k]) /(2*dy)
                ) - coriolis_param[1:-1, 1:-1] * hc[1:-1, 1:-1] * phix[1:-1, 1:-1, k]
                dhv_new_stocha[:, :, k] = enforce_boundaries(dhv_new_stocha[:, :, k], 'v')

            # SGN martingale term
            if simulationType == "Serre-Green-Naghdi":
                toBeInterp_dyX = np.zeros_like(hu)
                toBeInterp_dxY = np.zeros_like(hv)
                for k in range(nmodes):
                    loc_phix = np.zeros_like(phix)
                    loc_phiy = np.zeros_like(phix)
                    loc_phix[1:-1, 1:-1, k] = (phix[1:-1, :-2, k]+phix[1:-1, 1:-1, k])/2
                    loc_phiy[1:-1, 1:-1, k] = (phiy[:-2, 1:-1, k]+phiy[1:-1, 1:-1, k])/2
                    loc_phix = enforce_boundaries(loc_phix, 'h')
                    loc_phiy = enforce_boundaries(loc_phiy, 'h')
                    
                    sgn_term_stocha_dxX[1:-1, 1:-1, k] = (hc[1:-1, 2:]**3/3*div_vit[1:-1, 2:]*loc_phix[1:-1, 2:, k]
                                                    -  hc[1:-1, 1:-1]**3/3*div_vit[1:-1, 1:-1]*loc_phix[1:-1, 1:-1, k])/dx
                    sgn_term_stocha_dyY[1:-1, 1:-1, k] = (hc[2:, 1:-1]**3/3*div_vit[2:, 1:-1]*loc_phiy[2:, 1:-1, k]
                                                    -  hc[1:-1, 1:-1]**3/3*div_vit[1:-1, 1:-1]*loc_phiy[1:-1, 1:-1, k])/dy
                    
                    #cross terms must be interpolated
                    toBeInterp_dyX[1:-1, 1:-1] = (hc[2:, 1:-1]**3/3*div_vit[2:, 1:-1]*loc_phix[2:, 1:-1, k]
                                                -  hc[1:-1, 1:-1]**3/3*div_vit[1:-1, 1:-1]*loc_phix[1:-1, 1:-1, k])/dy
                    toBeInterp_dyX = enforce_boundaries(toBeInterp_dyX, 'v')
                    sgn_term_stocha_dyX[1:-1, 1:-1, k] = 0.25*(toBeInterp_dyX[1:-1, 1:-1] + toBeInterp_dyX[1:-1, 2:]
                                                            +toBeInterp_dyX[:-2, 1:-1] + toBeInterp_dyX[:-2, 2:])
                    
                    toBeInterp_dxY[1:-1, 1:-1] = (hc[1:-1, 2:]**3/3*div_vit[1:-1, 2:]*loc_phiy[1:-1, 2:, k]
                                                -  hc[1:-1, 1:-1]**3/3*div_vit[1:-1, 1:-1]*loc_phiy[1:-1, 1:-1, k])/dx
                    toBeInterp_dxY = enforce_boundaries(toBeInterp_dxY, 'u')
                    sgn_term_stocha_dxY[1:-1, 1:-1, k] = 0.25*(toBeInterp_dxY[1:-1, 1:-1] + toBeInterp_dxY[2:, 1:-1]
                                                            +toBeInterp_dxY[1:-1, :-2] + toBeInterp_dxY[2:, :-2])
                    
                    sgn_term_stocha_dxX[:, :, k] = enforce_boundaries(sgn_term_stocha_dxX[:, :, k], 'u')
                    sgn_term_stocha_dyX[:, :, k] = enforce_boundaries(sgn_term_stocha_dyX[:, :, k], 'u')    
                    sgn_term_stocha_dxY[:, :, k] = enforce_boundaries(sgn_term_stocha_dxY[:, :, k], 'v')    
                    sgn_term_stocha_dyY[:, :, k] = enforce_boundaries(sgn_term_stocha_dyY[:, :, k], 'v')    
        
                    dhu_new_stocha[1:-1, 1:-1, k] += (beta ** 2) * ((sgn_term_stocha_dxX[1:-1, 2:, k] - sgn_term_stocha_dxX[1:-1, :-2, k]) / (2*dx)
                                                                + (sgn_term_stocha_dxY[2:, 1:-1, k] - sgn_term_stocha_dxY[:-2, 1:-1, k]) / (2*dy))
                    dhv_new_stocha[1:-1, 1:-1, k] += (beta ** 2) * ((sgn_term_stocha_dyX[1:-1, 2:, k] - sgn_term_stocha_dyX[1:-1, :-2, k]) / (2*dx)
                                                                + (sgn_term_stocha_dyY[2:, 1:-1, k] - sgn_term_stocha_dyY[:-2, 1:-1, k]) / (2*dy))
                    dhu_new_stocha[:, :, k] = enforce_boundaries(dhu_new_stocha[:, :, k], 'u')
                    dhv_new_stocha[:, :, k] = enforce_boundaries(dhv_new_stocha[:, :, k], 'v')
            
            #Euler-Heun method for stochastic terms
            dBt = np.random.normal(0, np.sqrt(dt), nmodes)
            hutemp[1:-1, 1:-1] = hu[1:-1, 1:-1]
            hvtemp[1:-1, 1:-1] = hv[1:-1, 1:-1]
            htemp[1:-1, 1:-1] = hc[1:-1, 1:-1]
            
            hutemp = enforce_boundaries(hutemp, 'u')
            hvtemp = enforce_boundaries(hvtemp, 'v')
            htemp = enforce_boundaries(htemp, 'h')
            for k in range(nmodes):
                hutemp[1:-1, 1:-1] += dBt[k]*dhu_new_stocha[1:-1, 1:-1, k]
                hvtemp[1:-1, 1:-1] += dBt[k]*dhv_new_stocha[1:-1, 1:-1, k]
                htemp[1:-1, 1:-1]  += dBt[k]*dh_new_stocha[1:-1, 1:-1, k]
                hutemp = enforce_boundaries(hutemp, 'u')
                hvtemp = enforce_boundaries(hvtemp, 'v')
                htemp = enforce_boundaries(htemp, 'h')

            etatemp = np.pad(htemp[1:-1, 1:-1] - depth, 1, 'edge')
            etatemp = enforce_boundaries(etatemp, 'h')
            
            for k in range(nmodes):
                hu_stocha_temp[1:-1, 1:-1, k] = 0.5 * (etatemp[1:-1, 1:-1] + etatemp[1:-1, 2:]) * phix[1:-1, 1:-1, k]
                hv_stocha_temp[1:-1, 1:-1, k] = 0.5 * (etatemp[1:-1, 1:-1] + etatemp[2:, 1:-1]) * phiy[1:-1, 1:-1, k]
                hu_stocha_temp[:,:,k] = enforce_boundaries(hu_stocha_temp[:,:,k], 'u')
                hv_stocha_temp[:,:,k] = enforce_boundaries(hv_stocha_temp[:,:,k], 'v')

            for k in range(nmodes):
                dh_new_stocha[1:-1, 1:-1, k] += - (
                    (hu_stocha_temp[1:-1, 1:-1, k] - hu_stocha_temp[1:-1, :-2, k]) / dx
                    + (hv_stocha_temp[1:-1, 1:-1, k] - hv_stocha_temp[:-2, 1:-1, k]) / dy
                )
                dh_new_stocha[:,:,k] = enforce_boundaries(dh_new_stocha[:,:,k], 'h')

                dhu_new_stocha[1:-1, 1:-1, k] += (
                        (hutemp[1:-1, 2:]*phix[1:-1, 2:, k] - hutemp[1:-1, :-2]*phix[1:-1, :-2, k]) /(2*dx)
                    + (hutemp[2:, 1:-1]*phiy[2:, 1:-1, k] - hutemp[:-2, 1:-1]*phiy[:-2, 1:-1, k]) /(2*dy)
                ) - coriolis_param[1:-1, 1:-1] * hc[1:-1, 1:-1] * phiy[1:-1, 1:-1, k]
                dhu_new_stocha[:,:,k] = enforce_boundaries(dhu_new_stocha[:,:,k], 'u')
            
                dhv_new_stocha[1:-1, 1:-1, k] += (
                        (hvtemp[1:-1, 2:]*phix[1:-1, 2:, k] - hvtemp[1:-1, :-2]*phix[1:-1, :-2, k]) /(2*dx)
                    + (hvtemp[2:, 1:-1]*phiy[2:, 1:-1, k] - hvtemp[:-2, 1:-1]*phiy[:-2, 1:-1, k]) /(2*dy)
                ) + coriolis_param[1:-1, 1:-1] * hc[1:-1, 1:-1] * phix[1:-1, 1:-1, k]
                dhv_new_stocha[:,:,k] = enforce_boundaries(dhv_new_stocha[:,:,k], 'v')

            # SGN stochastic term (Euler-Heun)
            if simulationType == "Serre-Green-Naghdi":
                
                localx_h_temp = np.zeros_like(htemp) + depth
                localy_h_temp = np.zeros_like(htemp) + depth
                localx_h_temp[1:-1, 1:-1] = 0.5 * (htemp[1:-1, 1:-1] + htemp[1:-1, 2:])
                localy_h_temp[1:-1, 1:-1] = 0.5 * (htemp[1:-1, 1:-1] + htemp[2:, 1:-1])
                localx_h_temp = enforce_boundaries(localx_h_temp, 'h')
                localy_h_temp = enforce_boundaries(localy_h_temp, 'h')

                div_vit_temp = np.empty((n_y, n_x))
                div_vit_temp[1:-1, 1:-1] = (   ((hutemp/localx_h_temp)[1:-1, 1:-1] - (hutemp/localx_h_temp)[1:-1, :-2]) / dx
                                        + ((hvtemp/localy_h_temp)[1:-1, 1:-1] - (hvtemp/localy_h_temp)[:-2, 1:-1]) / dy    )
                div_vit_temp = enforce_boundaries(div_vit_temp, 'h')

                toBeInterp_dyX = np.zeros_like(hu)
                toBeInterp_dxY = np.zeros_like(hv)

                for k in range(nmodes):
                    loc_phix = np.zeros_like(phix)
                    loc_phiy = np.zeros_like(phix)
                    loc_phix[1:-1, 1:-1, k] = (phix[1:-1, :-2, k]+phix[1:-1, 1:-1, k])/2
                    loc_phiy[1:-1, 1:-1, k] = (phiy[:-2, 1:-1, k]+phiy[1:-1, 1:-1, k])/2
                    loc_phix = enforce_boundaries(loc_phix, 'h')
                    loc_phiy = enforce_boundaries(loc_phiy, 'h')
                    
                    sgn_term_stocha_dxX[1:-1, 1:-1, k] = (hc[1:-1, 2:]**3/3*div_vit_temp[1:-1, 2:]*loc_phix[1:-1, 2:, k]
                                                    -  hc[1:-1, 1:-1]**3/3*div_vit_temp[1:-1, 1:-1]*loc_phix[1:-1, 1:-1, k])/dx
                    sgn_term_stocha_dyY[1:-1, 1:-1, k] = (hc[2:, 1:-1]**3/3*div_vit_temp[2:, 1:-1]*loc_phiy[2:, 1:-1, k]
                                                    -  hc[1:-1, 1:-1]**3/3*div_vit_temp[1:-1, 1:-1]*loc_phiy[1:-1, 1:-1, k])/dy
                    
                    #cross terms must be interpolated
                    toBeInterp_dyX[1:-1, 1:-1] = (hc[2:, 1:-1]**3/3*div_vit_temp[2:, 1:-1]*loc_phix[2:, 1:-1, k]
                                                -  hc[1:-1, 1:-1]**3/3*div_vit_temp[1:-1, 1:-1]*loc_phix[1:-1, 1:-1, k])/dy
                    toBeInterp_dyX = enforce_boundaries(toBeInterp_dyX, 'v')
                    sgn_term_stocha_dyX[1:-1, 1:-1, k] = 0.25*(toBeInterp_dyX[1:-1, 1:-1] + toBeInterp_dyX[1:-1, 2:]
                                                            +toBeInterp_dyX[:-2, 1:-1] + toBeInterp_dyX[:-2, 2:])
                    
                    toBeInterp_dxY[1:-1, 1:-1] = (hc[1:-1, 2:]**3/3*div_vit_temp[1:-1, 2:]*loc_phiy[1:-1, 2:, k]
                                                -  hc[1:-1, 1:-1]**3/3*div_vit_temp[1:-1, 1:-1]*loc_phiy[1:-1, 1:-1, k])/dx
                    toBeInterp_dxY = enforce_boundaries(toBeInterp_dxY, 'u')
                    sgn_term_stocha_dxY[1:-1, 1:-1, k] = 0.25*(toBeInterp_dxY[1:-1, 1:-1] + toBeInterp_dxY[2:, 1:-1]
                                                            +toBeInterp_dxY[1:-1, :-2] + toBeInterp_dxY[2:, :-2])
                    
                    sgn_term_stocha_dxX[:, :, k] = enforce_boundaries(sgn_term_stocha_dxX[:, :, k], 'u')
                    sgn_term_stocha_dyX[:, :, k] = enforce_boundaries(sgn_term_stocha_dyX[:, :, k], 'u')    
                    sgn_term_stocha_dxY[:, :, k] = enforce_boundaries(sgn_term_stocha_dxY[:, :, k], 'v')    
                    sgn_term_stocha_dyY[:, :, k] = enforce_boundaries(sgn_term_stocha_dyY[:, :, k], 'v')    
        
                    dhu_new_stocha[1:-1, 1:-1, k] += (beta ** 2) * ((sgn_term_stocha_dxX[1:-1, 2:, k] - sgn_term_stocha_dxX[1:-1, :-2, k]) / (2*dx)
                                                                + (sgn_term_stocha_dxY[2:, 1:-1, k] - sgn_term_stocha_dxY[:-2, 1:-1, k]) / (2*dy))
                    dhv_new_stocha[1:-1, 1:-1, k] += (beta ** 2) * ((sgn_term_stocha_dyX[1:-1, 2:, k] - sgn_term_stocha_dyX[1:-1, :-2, k]) / (2*dx)
                                                                + (sgn_term_stocha_dyY[2:, 1:-1, k] - sgn_term_stocha_dyY[:-2, 1:-1, k]) / (2*dy))
                    dhu_new_stocha[:, :, k] = enforce_boundaries(dhu_new_stocha[:, :, k], 'u')
                    dhv_new_stocha[:, :, k] = enforce_boundaries(dhv_new_stocha[:, :, k], 'v')
            
            dh_new_stocha  *= 0.5
            dhu_new_stocha *= 0.5
            dhv_new_stocha *= 0.5

            # no inversion in SV model 
            # direct inversion in Boussinesq model (computed with Fourier transforms)    
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

                for k in range(nmodes):
                    dhu_newStocha_fourier[1:-1, 1:-1] = np.fft.fft2(dhu_new_stocha[1:-1, 1:-1, k])
                    dhv_newStocha_fourier[1:-1, 1:-1] = np.fft.fft2(dhv_new_stocha[1:-1, 1:-1, k])

                    dhu_newStocha_complex = diag_term_11[1:-1, 1:-1] * dhu_newStocha_fourier[1:-1, 1:-1] + non_diag_term_12[1:-1, 1:-1] * dhv_newStocha_fourier[1:-1, 1:-1]
                    dhv_newStocha_complex = non_diag_term_21[1:-1, 1:-1] * dhu_newStocha_fourier[1:-1, 1:-1] + diag_term_22[1:-1, 1:-1] * dhv_newStocha_fourier[1:-1, 1:-1]
        
                    dhu_new_stocha[1:-1, 1:-1, k] = np.fft.ifft2(dhu_newStocha_complex).real
                    dhv_new_stocha[1:-1, 1:-1, k] = np.fft.ifft2(dhv_newStocha_complex).real
                    dhu_new_stocha[:, :, k] = enforce_boundaries(dhu_new_stocha[:,:,k], 'u')
                    dhv_new_stocha[:, :, k] = enforce_boundaries(dhv_new_stocha[:,:,k], 'v')

            if simulationType == "Boussinesq" and (not(periodic_boundary_x) and not(periodic_boundary_y)):
                nInternalPoints = (n_x-2)*(n_y-2)
                dmom = np.concatenate((nx.reshape(dhu_new[1:-1,1:-1], (nInternalPoints,)), nx.reshape(dhv_new[1:-1,1:-1], (nInternalPoints,))))
                dmom = scsp.linalg.spsolve(matrixToInvert, dmom)

                dhu_new[1:-1,1:-1] = nx.reshape(dmom[:nInternalPoints ], (n_y-2, n_x-2))
                dhv_new[1:-1,1:-1] = nx.reshape(dmom[ nInternalPoints:], (n_y-2, n_x-2))
                dhu_new = enforce_boundaries(dhu_new, 'u')
                dhv_new = enforce_boundaries(dhv_new, 'v')

                for k in range(nmodes):
                    dmomStocha = np.concatenate((nx.reshape(dhu_new_stocha[1:-1,1:-1, k], (nInternalPoints,)), nx.reshape(dhv_new_stocha[1:-1,1:-1, k], (nInternalPoints,)))).copy()
                    dmomStocha = scsp.linalg.spsolve(matrixToInvert, dmomStocha)
                    dhu_new_stocha[1:-1,1:-1, k] = nx.reshape(dmomStocha[:nInternalPoints ], (n_y-2, n_x-2))
                    dhv_new_stocha[1:-1,1:-1, k] = nx.reshape(dmomStocha[ nInternalPoints:], (n_y-2, n_x-2))
                    dhu_new_stocha[:, :, k] = enforce_boundaries(dhu_new_stocha[:,:,k], 'u')
                    dhv_new_stocha[:, :, k] = enforce_boundaries(dhv_new_stocha[:,:,k], 'v')
            
            if simulationType == "Serre-Green-Naghdi":
                nPoints = n_x*n_y
                nInternalPoints = (n_x-2)*(n_y-2)
                generalMatrix = np.diag(np.ones(2*nPoints))
                cst = beta**2/3
                
                for i in range(1,n_x-1):
                    for j in range(1,n_y-1):
                        #matrix terms associated to (hu)_ij
                        generalMatrix[j*n_x + i,j*n_x + i]   = 1 + cst*(hc[j,i]**3 + hc[j,i+1]**3)/dx**2   / localx_h[j,i]
                        generalMatrix[j*n_x + i,j*n_x + i+1] =   - cst* hc[j,i+1]**3/dx**2                 / localx_h[j,i+1]
                        generalMatrix[j*n_x + i,j*n_x + i-1] =   - cst* hc[j,i]**3/dx**2                   / localx_h[j,i-1]
                        
                        generalMatrix[j*n_x + i,nPoints+j*n_x + i]       =  cst*hc[j,i]**3/dx/dy           / localy_h[j,i]
                        generalMatrix[j*n_x + i,nPoints+j*n_x + i+1]     = -cst*hc[j,i+1]**3/dx/dy         / localy_h[j,i+1]
                        generalMatrix[j*n_x + i,nPoints+(j-1)*n_x + i+1] =  cst*hc[j,i+1]**3/dx/dy         / localy_h[j-1,i+1]
                        generalMatrix[j*n_x + i,nPoints+(j-1)*n_x + i]   = -cst*hc[j,i]**3/dx/dy           / localy_h[j-1,i]

                        #matrix terms associated to (hv)_ij
                        generalMatrix[nPoints+j*n_x + i,nPoints+j*n_x + i]     = 1 + cst*(hc[j,i]**3 + hc[j+1,i]**3)/dy**2   / localy_h[j,i]
                        generalMatrix[nPoints+j*n_x + i,nPoints+(j+1)*n_x + i] =   - cst* hc[j+1,i]**3/dy**2                 / localy_h[j+1,i]
                        generalMatrix[nPoints+j*n_x + i,nPoints+(j-1)*n_x + i] =   - cst* hc[j,i]**3/dy**2                   / localy_h[j-1,i]
                        
                        generalMatrix[nPoints+j*n_x + i,j*n_x + i]       =  cst*hc[j,i]**3/dx/dy           / localx_h[j,i]
                        generalMatrix[nPoints+j*n_x + i,(j+1)*n_x + i]   = -cst*hc[j+1,i]**3/dx/dy         / localx_h[j+1,i]
                        generalMatrix[nPoints+j*n_x + i,(j+1)*n_x + i-1] =  cst*hc[j+1,i]**3/dx/dy         / localx_h[j+1,i-1]
                        generalMatrix[nPoints+j*n_x + i,j*n_x + i-1]     = -cst*hc[j,i]**3/dx/dy           / localx_h[j,i-1]

                if not(periodic_boundary_x) and not(periodic_boundary_y):
                    dmom = np.concatenate((nx.reshape(dhu_new[1:-1,1:-1], (nInternalPoints,)), nx.reshape(dhv_new[1:-1,1:-1], (nInternalPoints,))))
                    matrixToInvert = np.zeros((2*nInternalPoints, 2*nInternalPoints))
                    for i in range(0,n_x-2):
                        for j in range(0,n_y-2):
                            matrixToInvert[i*(n_y-2) + j, i*(n_y-2) + j] = generalMatrix[(i+1)*n_y + j+1, (i+1)*n_y + j+1]
                            matrixToInvert[nInternalPoints + i*(n_y-2) + j, i*(n_y-2) + j] = generalMatrix[nPoints + (i+1)*n_y + j+1, (i+1)*n_y + j+1]
                            matrixToInvert[i*(n_y-2) + j, nInternalPoints + i*(n_y-2) + j] = generalMatrix[(i+1)*n_y + j+1, nPoints + (i+1)*n_y + j+1]
                            matrixToInvert[nInternalPoints + i*(n_y-2) + j, nInternalPoints + i*(n_y-2) + j] = generalMatrix[nPoints + (i+1)*n_y + j+1, nPoints + (i+1)*n_y + j+1]
                    matrixToInvert = scsp.csr_matrix(matrixToInvert)
                    dmom = scsp.linalg.spsolve(matrixToInvert, dmom)

                    dhu_new[1:-1,1:-1] = nx.reshape(dmom[:nInternalPoints ], (n_y-2, n_x-2))
                    dhv_new[1:-1,1:-1] = nx.reshape(dmom[ nInternalPoints:], (n_y-2, n_x-2))
                    dhu_new = enforce_boundaries(dhu_new, 'u')
                    dhv_new = enforce_boundaries(dhv_new, 'v')

                    for k in range(nmodes):
                        dmomStocha = np.concatenate((nx.reshape(dhu_new_stocha[1:-1,1:-1, k], (nInternalPoints,)), nx.reshape(dhv_new_stocha[1:-1,1:-1, k], (nInternalPoints,)))).copy()
                        dmomStocha = scsp.linalg.spsolve(matrixToInvert, dmomStocha)

                        dhu_new_stocha[1:-1,1:-1, k] = nx.reshape(dmomStocha[:nInternalPoints ], (n_y-2, n_x-2))
                        dhv_new_stocha[1:-1,1:-1, k] = nx.reshape(dmomStocha[ nInternalPoints:], (n_y-2, n_x-2))
                        dhu_new_stocha[:, :, k] = enforce_boundaries(dhu_new_stocha[:,:,k], 'u')
                        dhv_new_stocha[:, :, k] = enforce_boundaries(dhv_new_stocha[:,:,k], 'v')

                elif periodic_boundary_x and periodic_boundary_y:
                    dmom = np.concatenate((nx.reshape(dhu_new, (nPoints,)), nx.reshape(dhv_new, (nPoints,))))
                    matrixToInvert = scsp.csr_matrix(generalMatrix)

                    #direct method
                    dmom = scsp.linalg.spsolve(matrixToInvert, dmom)

                    """#Jacobi matrices
                    vecDiag = np.diag(generalMatrix)
                    invDiagPart  = scsp.csr_matrix(np.diag(  1/ vecDiag ) )
                    residualPart = scsp.csr_matrix(generalMatrix) - scsp.csr_matrix(np.diag(vecDiag))
                    #Jacobi method
                    precision = 0.1
                    vector = dmom
                    normalizationCoeff = max(np.linalg.norm(dmom),1)
                    while np.linalg.norm(matrixToInvert @ vector - dmom)/normalizationCoeff > precision :
                        vector = invDiagPart @ dmom - invDiagPart @ residualPart @ vector
                    dmom = vector.copy()"""

                    """#Gauss-Seidel matrices
                    upperPart = scsp.csr_matrix(np.triu(generalMatrix, k=1))
                    lowerPart = scsp.csr_matrix(np.tril(generalMatrix, k=0))
                    #Gauss-Seidel method
                    vector = dmom
                    normalizationCoeff = max(np.linalg.norm(dmom),1)
                    while np.linalg.norm(matrixToInvert @ vector - dmom)/normalizationCoeff > 1e-1 :
                        vector = scsp.linalg.spsolve(lowerPart, dmom - upperPart @ vector)
                    dmom = vector.copy()"""

                    dhu_new = nx.reshape(dmom[:nPoints ], (n_y, n_x))
                    dhv_new = nx.reshape(dmom[ nPoints:], (n_y, n_x))
                    dhu_new = enforce_boundaries(dhu_new, 'u')
                    dhv_new = enforce_boundaries(dhv_new, 'v')

                    for k in range(nmodes):
                        dmomStocha = np.concatenate((nx.reshape(dhu_new_stocha[:,:, k], (nPoints,)), nx.reshape(dhv_new_stocha[:,:, k], (nPoints,)))).copy()
                        
                        #direct method
                        dmomStocha = scsp.linalg.spsolve(matrixToInvert, dmomStocha)
                        
                        """#Jacobi method
                        vector = dmomStocha
                        normalizationCoeff = max(np.linalg.norm(dmomStocha),1)
                        while np.linalg.norm(matrixToInvert @ vector - dmomStocha)/normalizationCoeff > precision :
                            vector = invDiagPart @ dmomStocha - invDiagPart @ residualPart @ vector
                        dmomStocha = vector.copy()"""

                        """#Gauss-Seidel method
                        vector = dmomStocha
                        normalizationCoeff = max(np.linalg.norm(dmomStocha),1)
                        while np.linalg.norm(matrixToInvert @ vector - dmomStocha)/normalizationCoeff > 5e-1 :
                            vector = scsp.linalg.spsolve(lowerPart, dmomStocha - upperPart @ vector)
                        dmomStocha = vector.copy()"""
                        
                        dhu_new_stocha[:, :, k] = nx.reshape(dmomStocha[:nPoints ], (n_y, n_x))
                        dhv_new_stocha[:, :, k] = nx.reshape(dmomStocha[ nPoints:], (n_y, n_x))
                        dhu_new_stocha[:, :, k] = enforce_boundaries(dhu_new_stocha[:,:,k], 'u')
                        dhv_new_stocha[:, :, k] = enforce_boundaries(dhv_new_stocha[:,:,k], 'v')

            #time increment
            if first_step:
                hu[1:-1, 1:-1] += dt * dhu_new[1:-1, 1:-1]
                hv[1:-1, 1:-1] += dt * dhv_new[1:-1, 1:-1]
                h[1:-1, 1:-1]  += dt * dh_new[1:-1, 1:-1]
                for k in range(nmodes):
                    hu[1:-1, 1:-1] += dBt[k]*dhu_new_stocha[1:-1, 1:-1, k]
                    hv[1:-1, 1:-1] += dBt[k]*dhv_new_stocha[1:-1, 1:-1, k]
                    h[1:-1, 1:-1] += dBt[k]*dh_new_stocha[1:-1, 1:-1, k]
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
                for k in range(nmodes):
                    hu[1:-1, 1:-1] += dBt[k]*dhu_new_stocha[1:-1, 1:-1, k]
                    hv[1:-1, 1:-1] += dBt[k]*dhv_new_stocha[1:-1, 1:-1, k]
                    h[1:-1, 1:-1] += dBt[k]*dh_new_stocha[1:-1, 1:-1, k]
                hu = enforce_boundaries(hu, 'u')
                hv = enforce_boundaries(hv, 'v')
                h = enforce_boundaries(h, 'h')
        
            if lateral_viscosity > 0:    
                hu[1:-1, 1:-1] += dt * (fue[1:-1, 1:-1] + fun[1:-1, 1:-1])
                hv[1:-1, 1:-1] += dt * (fve[1:-1, 1:-1] + fvn[1:-1, 1:-1])
            hu = enforce_boundaries(hu, 'u')
            hv = enforce_boundaries(hv, 'v')

            # rotate quantities
            dhu[...] = dhu_new
            dhv[...] = dhv_new
            dh[...] = dh_new

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

        if save_data:
            spec = [n_x, n_y, dx, dy]
            export_to_csv(spec, path_to_store+'//spec.csv')

        for iteration, (h, hu, hv) in enumerate(model):
            if iteration % plot_every == 0:
                t = iteration * dt
                
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
            frames = np.stack([iio.imread(f"images/{i}.jpg") for i in range(count)] + [iio.imread(f"images/{i}.jpg") for i in range(count-2, 0, -1)], axis = 0)
            iio.mimwrite('test1000.gif', frames, loop=0)
