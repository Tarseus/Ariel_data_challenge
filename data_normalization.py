import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf 
import random 
import os
from matplotlib.ticker import ScalarFormatter
import pandas as pd

data_folder = './data'
auxiliary_folder = './auxiliary'
normalized_data_folder = './normalized_data'
data_train = np.load(f'{data_folder}/data_train.npy')
data_train_FGS = np.load(f'{data_folder}/data_train_FGS.npy')
print(data_train.shape)
print(data_train[0].shape)
print(data_train_FGS.shape)
print(data_train_FGS[0].shape)

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


def norm_star_spectrum (signal) : 
    img_star = signal[:,:50].mean(axis = 1) + signal[:,-50:].mean(axis = 1)
    return signal/img_star[:,np.newaxis,:]

def set_output_dir(output_dir='./output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created output directory {output_dir}')
    else:
        print(f'Output directory {output_dir} already exists')
    
def normalize_data():
    train_solution = np.loadtxt(f'{auxiliary_folder}/train_labels.csv', delimiter = ',', skiprows = 1)
    targets = train_solution[:,1:]
    targets_mean = targets[:,1:].mean(axis = 1) # used for the 1D-CNN to extract the mean value, only AIRS wavelengths as the FGS point is not used in the white curve
    N = targets.shape[0]
    signal_AIRS_diff_transposed_binned, signal_FGS_diff_transposed_binned  = data_train, data_train_FGS
    FGS_column = signal_FGS_diff_transposed_binned.sum(axis = 2)
    dataset = np.concatenate([signal_AIRS_diff_transposed_binned, FGS_column[:,:, np.newaxis,:]], axis = 2)
    # shape: 673, 187, 283, 32; num_samples, num_time_step, num_wavelength, num_spatial_dim
    dataset = dataset.sum(axis=3)
    dataset_norm = norm_star_spectrum(dataset)
    dataset_norm = np.transpose(dataset_norm,(0,2,1))
    cut_inf, cut_sup = 39, 321 # we have previously cut the data along the wavelengths to remove the edges, this is to match with the targets range in the make data file
    l = cut_sup - cut_inf + 1 
    wls = np.arange(l)
    N_train = 8*N//10
    train_obs, valid_obs, list_index_train = split(dataset_norm, N_train)
    train_targets, valid_targets = targets[list_index_train], targets[~list_index_train]
    signal_AIRS_diff_transposed_binned = signal_AIRS_diff_transposed_binned.sum(axis=3)
    wc_mean = signal_AIRS_diff_transposed_binned.mean(axis=1).mean(axis=1)
    white_curve = signal_AIRS_diff_transposed_binned.sum(axis=2)/ wc_mean[:, np.newaxis]
    # Split the light curves and targets 
    train_wc, valid_wc = white_curve[list_index_train], white_curve[~list_index_train]
    train_targets_wc, valid_targets_wc = targets_mean[list_index_train], targets_mean[~list_index_train]

    # Normalize the wlc
    train_wc, valid_wc = normalise_wlc(train_wc, valid_wc)

    # Normalize the targets 
    train_targets_wc_norm, valid_targets_wc_norm, min_train_valid_wc, max_train_valid_wc = normalize(train_targets_wc, valid_targets_wc)
    return train_wc, valid_wc, train_targets_wc_norm, valid_targets_wc_norm, min_train_valid_wc, max_train_valid_wc

if __name__ == '__main__':
    set_output_dir()
    train_wc, valid_wc, train_targets_wc_norm, valid_targets_wc_norm, min_train_valid_wc, max_train_valid_wc =normalize_data()
    np.save(f'{normalized_data_folder}/train_wc.npy', train_wc)
    np.save(f'{normalized_data_folder}/valid_wc.npy', valid_wc)
    np.save(f'{normalized_data_folder}/train_targets_wc_norm.npy', train_targets_wc_norm)
    np.save(f'{normalized_data_folder}/valid_targets_wc_norm.npy', valid_targets_wc_norm)
    np.save(f'{normalized_data_folder}/min_train_valid_wc.npy', min_train_valid_wc)
    np.save(f'{normalized_data_folder}/max_train_valid_wc.npy', max_train_valid_wc)