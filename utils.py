import numpy as np
import random
from scipy.signal import savgol_filter

anxiliary_folder = './auxiliary'
def normalise_wlc(train, valid) :

    wlc_train_min = train.min()
    wlc_train_max = train.max()
    train_norm = (train - wlc_train_min) / (wlc_train_max - wlc_train_min)
    valid_norm = (valid - wlc_train_min) / (wlc_train_max - wlc_train_min)
    
    return train_norm, valid_norm

def normalize (train, valid) : 
    max_train = train.max()
    min_train = train.min()
    train_norm = (train - min_train) / (max_train - min_train)
    valid_norm = (valid - min_train) / (max_train - min_train)
    return train_norm, valid_norm, min_train, max_train

def split(data, N_train) : 
    list_planets = random.sample(range(0, data.shape[0]), N_train)
    list_index_1 = np.zeros(data.shape[0], dtype = bool)
    for planet in list_planets : 
        list_index_1[planet] = True
    data_1 = data[list_index_1]
    data_2 = data[~list_index_1]
    return data_1, data_2, list_index_1

def create_dataset_norm(dataset1, dataset2) :
    dataset_norm1 = np.zeros(dataset1.shape)
    dataset_norm2 = np.zeros(dataset1.shape)
    dataset_min = dataset1.min()
    dataset_max = dataset1.max()
    dataset_norm1 = (dataset1 - dataset_min) / (dataset_max - dataset_min)
    dataset_norm2 = (dataset2 - dataset_min) / (dataset_max - dataset_min)
    return dataset_norm1, dataset_norm2


def norm_star_spectrum(signal):
    break_point_csv = np.loadtxt(f'{anxiliary_folder}/breakpoints.csv', delimiter=',', skiprows=1, usecols=(-4, -1)).astype(np.int16)
    
    for i in range(signal.shape[0]):
        # break_point = min(break_point_csv[i][0], 187 - break_point_csv[i][1])
        break_point_left = break_point_csv[i, 0]
        break_point_right = break_point_csv[i, 1]
        out_of_transit = np.concatenate([signal[i, :break_point_left], signal[i, break_point_right:]], axis=0)
        img_star = out_of_transit.mean(axis=0)
        signal[i] = signal[i] / img_star[np.newaxis, :]
    
    return signal
    # img_star = signal[:,:50].mean(axis = 1) + signal[:,-50:].mean(axis = 1)
    # print(img_star.shape)
    # # return signal/img_star[:,np.newaxis,:]
    # exit()
    
def denoise1(train, valid):
    # shape: (673, 187, 282)
    train = savgol_filter(train, 5, 3, axis=1)
    valid = savgol_filter(valid, 5, 3, axis=1)
    # percentile_1 = np.percentile(train, 1, axis=1)
    # mean = np.median(percentile_1, axis=0)
    # train = train / (1 - mean[np.newaxis, np.newaxis, :])
    # valid = valid / (1 - mean[np.newaxis, np.newaxis, :])
    return train, valid