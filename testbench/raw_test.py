import numpy as np
import setigen as stg
from blimpy import Waterfall
import matplotlib.pyplot as plt
import random
from astropy import units as u
from tqdm import tqdm
import os
from sklearn.metrics import silhouette_score
import pandas as pd
import tqdm

def painting(data):
    all_data = []
    labels = []
    for c in tqdm(range(num_classes)):
        drift = 2*random.random()*(-1)**random.randint(0,2)
        snr = random.randint(100, 150)
        width = random.randint(20, 50)
        for s in range(num_samples_per_class):
            index = random.randint(0, data.shape[0]-1)
            window = data[index, :,:]
            
            start = random.randint(50, 180)
            
            frame = stg.Frame.from_data(df=2.7939677238464355*u.Hz,
                                        dt=18.253611008*u.s,
                                        fch1=1289*u.MHz,
                                        ascending=True,
                                        data=window)
            frame.add_signal(stg.constant_path(
                                        f_start=frame.get_frequency(index=start),
                                       drift_rate=drift*u.Hz/u.s),
                                      stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
                                      stg.gaussian_f_profile(width=width*u.Hz),
                                      stg.constant_bp_profile(level=1))
            all_data.append(frame.data)
            labels.append(c)
    all_data = np.array(all_data)
    labels = np.vstack(labels)
    return all_data, labels


for i in tqdm(range(10)):

    num_classes = 100
    num_samples_per_class = 1000


    # Open dataset

    directory = os.fsencode( "../../../../../datax/scratch/pma/reverse_search/test/")
    count = 0
    data = []
    for folder in os.listdir(directory):
        print(folder)
        for subfolder in os.listdir(directory+folder):
            back = os.fsencode( "/")
            for file in os.listdir(directory+folder+back+subfolder):
                file_directory = str(os.path.join(directory+folder+back+subfolder, file)).replace('b', '').replace("'","")
                if 'filtered.npy' in file_directory:
                    data.append(np.load(str(file_directory)))
                    count += 1
    data = np.vstack(data)
    print(data.shape)

    injected, labels = painting(data)


    features = []
    for i in range(injected.shape[0]):
        features.append(injected[i,:,:].flatten())    
    features = np.array(features)

    print(features.shape)
    print(labels[:,0].shape)
    score = silhouette_score(X = feautres[:50000,:], labels = labels[:50000,0])
    print(score)


    f = open('output1.txt','a')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
    f.write('{}'.format(score))
    f.close()