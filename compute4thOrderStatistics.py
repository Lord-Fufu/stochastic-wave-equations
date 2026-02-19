import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

itmax = 30000
nStart = 1500
nStop = 3000
dir_to_store = 'data_stats_4thOrder'

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
for k in range(nStart,nStop):
    print(k)
    meanHeight=np.zeros_like(heightRef)
    for i_trajectory in range(n_data):
        dir_stocha='data_looped//data'+str(i_trajectory)
        df_heightStocha = pd.read_csv(dir_stocha + '//height//'+str(k)+'.csv')
        #df_xmomStocha = pd.read_csv(dir_stocha + '//xmom//'+str(k)+'.csv')
        #df_ymomStocha = pd.read_csv(dir_stocha + '//ymom//'+str(k)+'.csv')

        heightStocha = df_heightStocha.to_numpy()[:,1:]
        #xmomStocha = df_xmomStocha.to_numpy()[:,1:]
        #ymomStocha = df_ymomStocha.to_numpy()[:,1:]
        if (np.isnan(meanHeight)).any():
            print('->',i_trajectory)
            pass
        else:
            meanHeight += heightStocha/n_data

    varHeight=np.zeros_like(heightRef)
    for i_trajectory in range(n_data):
        dir_stocha='data_looped//data'+str(i_trajectory)
        df_heightStocha = pd.read_csv(dir_stocha + '//height//'+str(k)+'.csv')

        heightStocha = df_heightStocha.to_numpy()[:,1:]
        varHeight += (heightStocha - meanHeight)**2/n_data
    
    stdHeight=np.sqrt(varHeight)
    skewHeight=np.zeros_like(heightRef)
    kurtHeight=np.zeros_like(heightRef)
    for i_trajectory in range(n_data):
        dir_stocha='data_looped//data'+str(i_trajectory)
        df_heightStocha = pd.read_csv(dir_stocha + '//height//'+str(k)+'.csv')

        heightStocha = df_heightStocha.to_numpy()[:,1:]
        normalisedHeight = (heightStocha - meanHeight)/stdHeight

        skewHeight += normalisedHeight**3/n_data
        kurtHeight += normalisedHeight**4/n_data

    export_to_csv(meanHeight, dir_to_store + '//mean'+str(k)+'.csv')
    export_to_csv(varHeight, dir_to_store + '//var'+str(k)+'.csv')
    export_to_csv(skewHeight, dir_to_store + '//skew'+str(k)+'.csv')
    export_to_csv(kurtHeight, dir_to_store + '//kurt'+str(k)+'.csv')
