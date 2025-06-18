import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

dir_deter='data_deterministic'
specDeter = pd.read_csv(dir_deter + '//spec.csv').to_numpy()
itmax = 700
nstep = 71
dir_to_store = 'data_stats'

try:
    os.makedirs(dir_to_store)
    print(f"Directory '{dir_to_store}' created successfully.")
except FileExistsError:
    print(f"Directory '{dir_to_store}' already exists.")

def export_to_csv(field, name):
    df = pd.DataFrame(field)
    df.to_csv(name)

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

n_data=100
for k in range(0,nstep):
    print(k)
    df_heightDet = pd.read_csv(dir_deter + '//height//'+str(k)+'.csv')
    df_xmomDet = pd.read_csv(dir_deter + '//xmom//'+str(k)+'.csv')
    df_ymomDet = pd.read_csv(dir_deter + '//ymom//'+str(k)+'.csv')
    heightDeter = df_heightDet.to_numpy()[:,1:]
    xmomDeter = df_xmomDet.to_numpy()[:,1:]
    ymomDeter = df_ymomDet.to_numpy()[:,1:]

    meanHeight=np.zeros_like(heightDeter)
    for i_trajectory in range(n_data):

        dir_stocha='data_looped//data'+str(i_trajectory)
        specStocha = pd.read_csv(dir_stocha + '//spec.csv').to_numpy()
        if (specDeter != specStocha).any():
            raise Exception("Specs are different!")

        df_heightStocha = pd.read_csv(dir_stocha + '//height//'+str(k)+'.csv')
        df_xmomStocha = pd.read_csv(dir_stocha + '//xmom//'+str(k)+'.csv')
        df_ymomStocha = pd.read_csv(dir_stocha + '//ymom//'+str(k)+'.csv')

        heightStocha = df_heightStocha.to_numpy()[:,1:]
        xmomStocha = df_xmomStocha.to_numpy()[:,1:]
        ymomStocha = df_ymomStocha.to_numpy()[:,1:]

        heightDiff = heightDeter - heightStocha
        xmomDiff = xmomDeter - xmomStocha
        ymomDiff = ymomDeter - ymomStocha

        meanHeight += heightDiff/n_data

    varHeight=np.zeros_like(heightDeter)
    for i_trajectory in range(n_data):

        dir_stocha='data_looped//data'+str(i_trajectory)
        specStocha = pd.read_csv(dir_stocha + '//spec.csv').to_numpy()
        if (specDeter != specStocha).any():
            raise Exception("Specs are different!")

        df_heightStocha = pd.read_csv(dir_stocha + '//height//'+str(k)+'.csv')
        df_xmomStocha = pd.read_csv(dir_stocha + '//xmom//'+str(k)+'.csv')
        df_ymomStocha = pd.read_csv(dir_stocha + '//ymom//'+str(k)+'.csv')

        heightStocha = df_heightStocha.to_numpy()[:,1:]
        xmomStocha = df_xmomStocha.to_numpy()[:,1:]
        ymomStocha = df_ymomStocha.to_numpy()[:,1:]

        heightDiff = heightDeter - heightStocha
        xmomDiff = xmomDeter - xmomStocha
        ymomDiff = ymomDeter - ymomStocha

        varHeight += (heightDiff - meanHeight)**2/n_data

    export_to_csv(meanHeight, dir_to_store + '//mean'+str(k)+'.csv')
    export_to_csv(varHeight, dir_to_store + '//var'+str(k)+'.csv')
