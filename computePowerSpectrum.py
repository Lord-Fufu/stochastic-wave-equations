import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

itmax = 30000
nStart = 1500
nStop = 3000
dir_to_store = 'data_stats_powerSpectrum'

n_x=128
n_y=128
n_t=nStop - nStart

try:
    os.makedirs(dir_to_store)
    print(f"Directory '{dir_to_store}' created successfully.")
except FileExistsError:
    print(f"Directory '{dir_to_store}' already exists.")

def export_to_csv(field, name):
    df = pd.DataFrame(field)
    df.to_csv(name)

df_heightRef = pd.read_csv('data_looped//data0//height//0.csv')
heightRef = df_heightRef.to_numpy()[:,1:]

n_data=50
npoints = 32

powerSpectrum = np.zeros(n_t)
for i_trajectory in range(n_data):
    sampledTrajectory = np.zeros(npoints, npoints, n_t)
    for k in range(n_t):
        dir_stocha='data_looped//data'+str(i_trajectory)
        df_heightStocha = pd.read_csv(dir_stocha + '//height//'+str(k+nStart)+'.csv')

        heightStocha = df_heightStocha.to_numpy()[:,1:]
        sampledTrajectory[:,:,k] = heightStocha[::(n_x//npoints), ::(n_y//npoints)]
    fftTraj = np.fft.fft(sampledTrajectory)
    meanPowTraj = np.mean(fftTraj**2)
    powerSpectrum += meanPowTraj**2/n_data


    export_to_csv(powerSpectrum, dir_to_store + '//meanPowerSpectrum.csv')
