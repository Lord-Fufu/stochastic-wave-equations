import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dir_deter='data_deterministic'
specDeter = pd.read_csv(dir_deter + '//spec.csv').to_numpy()
itmax = 500
nstep = 50
itstep = itmax//nstep
dir_to_read = 'data_stats'

n_x = specDeter[0,1]
n_y = specDeter[1,1]
dx = specDeter[2,1]
dy = specDeter[3,1]

x, y = (
    np.arange(n_x) * dx,
    np.arange(n_y) * dy
)
gravity = 9.81
depth = 100.
dt = 0.1 * min(dx, dy) / np.sqrt(gravity * depth)

df_heightDet = pd.read_csv(dir_deter + '//height//0.csv')
df_xmomDet = pd.read_csv(dir_deter + '//xmom//0.csv')
df_ymomDet = pd.read_csv(dir_deter + '//ymom//0.csv')
heightDeter = df_heightDet.to_numpy()[:,1:]
xmomDeter = df_xmomDet.to_numpy()[:,1:]
ymomDeter = df_ymomDet.to_numpy()[:,1:]
m,n = np.shape(heightDeter)

meanHeightArr = np.zeros((nstep,m,n))
varHeightArr = np.zeros((nstep,m,n))
for k in range(0,itmax,itstep):
    df_mean = pd.read_csv(dir_to_read + '//mean'+str(k)+'.csv')
    df_var = pd.read_csv(dir_to_read + '//var'+str(k)+'.csv')
    meanHeight = df_mean.to_numpy()[:,1:]
    varHeight = df_var.to_numpy()[:,1:]
    meanHeightArr[k//itstep,:,:] = meanHeight
    varHeightArr[k//itstep,:,:] = varHeight
def prepare_plot(rmean,rvar):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    cs0 = update_plot(0, meanHeightArr[0,:,:], ax[0], -rmean, rmean, draw=False)
    cs1 = update_plot(0, varHeightArr[0,:,:], ax[1], 0, rvar, draw=False)
    plt.colorbar(cs0, label='mean of $\\eta$ (m)')
    plt.colorbar(cs1, label='std of $\\eta$ (m)')
    return fig, ax

def update_plot(t, h, ax0, valmin, valmax, draw=True):
    ax0.clear()
    if valmin == 0:
        colormap = 'Reds'
    else:
        colormap = 'RdBu_r'
    cs = ax0.pcolormesh(
        x[1:-1] / 1e3,
        y[1:-1] / 1e3,
        h[1:-1, 1:-1],
        vmin=valmin, vmax=valmax, cmap=colormap
    )

    ax0.set_aspect('equal')
    ax0.set_xlabel('$x$ (km)')
    ax0.set_ylabel('$y$ (km)')
    ax0.set_xlim(x[1] / 1e3, x[-2] / 1e3)
    ax0.set_ylim(y[1] / 1e3, y[-2] / 1e3)

    return cs

plot_range_mean = np.max(abs(meanHeightArr))
plot_range_var = np.sqrt(np.max(abs(varHeightArr)))

fig, ax = prepare_plot(plot_range_mean,plot_range_var)

for i in range(nstep):
    update_plot(i*dt, meanHeightArr[i,:,:], ax[0], -plot_range_mean, plot_range_mean, draw=True)
    update_plot(i*dt, np.sqrt(varHeightArr[i,:,:]), ax[1], 0, plot_range_var, draw=True)
    plt.pause(0.5)
    if not plt.fignum_exists(fig.number):
        break
