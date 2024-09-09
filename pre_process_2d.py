import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf 
import random 
import os
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from utils import *

data_folder = './data/w_lin_corr'
auxiliary_folder = './auxiliary'
normalized_data_folder = './pre_processed_2d'
data_train = np.load(f'{data_folder}/data_train.npy')
data_train_FGS = np.load(f'{data_folder}/data_train_FGS.npy')

def supress_mean(targets, mean):
    res = targets - np.repeat(mean.reshape((mean.shape[0], 1)), repeats=targets.shape[1], axis=1)
    return res

def suppress_out_transit(data):
    bound_data = np.loadtxt(f'{anxiliary_folder}/breakpoints.csv', delimiter=',', skiprows=1, usecols=(-3, -2)).astype(np.int16)
    ingress = bound_data[:, 0].max()
    egress = bound_data[:, 1].min()
    data_in = data[:, ingress:egress, :]
    return data_in

def substract_data_mean(data):
    data_mean = np.zeros(data.shape)
    for i in range(data.shape[0]):
        data_mean[i] = data[i] - data[i].mean()
    return data_mean

def data_norm(data1, data2):
    data_min = data1.min()
    data_max = data1.max()
    data_abs_max = np.max([data_min, data_max])
    data1 = data1/data_abs_max
    data2 = data2/data_abs_max
    return data1, data2, data_abs_max

def targets_norm_back(data, data_obs_max):
    data = data*data_obs_max
    return data

def targets_normalization (data1, data2) : 
    data_min = data1.min()
    data_max = data1.max()
    data_abs_max = np.max([data_min, data_max])  
    data1 = data1/data_abs_max
    data2 = data2/data_abs_max
    return data1, data2, data_abs_max

def targets_norm_back (data, data_abs_max) : 
    return data * data_abs_max

def pre_process_2d():
    signal_AIRS_diff_transposed_binned, signal_FGS_diff_transposed_binned  = data_train, data_train_FGS
    FGS_column = signal_FGS_diff_transposed_binned.sum(axis = 2)
    # print(signal_AIRS_diff_transposed_binned.shape, FGS_column[:, :, np.newaxis, :].shape)
    dataset = np.concatenate([signal_AIRS_diff_transposed_binned, FGS_column[:,:, np.newaxis,:]], axis = 2)
    dataset = dataset.sum(axis=3)
    
    train_solution = np.loadtxt(f'{auxiliary_folder}/train_labels.csv', delimiter = ',', skiprows = 1)
    targets = train_solution[:,1:]
    targets_mean = targets[:,1:].mean(axis = 1) # used for the 1D-CNN to extract the mean value, only AIRS wavelengths as the FGS point is not used in the white curve
    N = targets.shape[0]
    N_train = 8*N//10
    # N_train = N
    # # Validation and train data split
    cut_inf, cut_sup = 39, 321 # we have previously cut the data along the wavelengths to remove the edges, this is to match with the targets range in the make data file
    l = cut_sup - cut_inf + 1 
    wls = np.arange(l)
    
    dataset_norm = norm_star_spectrum(dataset)
    dataset_norm = np.transpose(dataset_norm,(0,2,1))
    train_obs, valid_obs, list_index_train = split(dataset_norm, N_train)
    
    train_targets, valid_targets = targets[list_index_train], targets[~list_index_train]
    train_targets_shift = supress_mean(train_targets, targets_mean[list_index_train])
    valid_targets_shift = supress_mean(valid_targets, targets_mean[~list_index_train])
    train_targets_norm, valid_targets_norm, targets_abs_max = targets_normalization(train_targets_shift, valid_targets_shift)
    
    plt.figure(figsize=(15,5))
    for i in range (240) :
        plt.plot(wls, train_targets_norm[i], 'g-', alpha = 0.5)
    plt.plot([], [], 'g-', alpha=0.5, label='Train targets')
    for i in range (60) : 
        plt.plot(wls, valid_targets_norm[i], 'r-', alpha = 0.7)
    plt.plot([], [], 'r-', alpha=0.5, label='Valid targets (true mean)')
    plt.legend()
    plt.ylabel(f'$(R_p/R_s)^2$')
    plt.title('All targets after substracting the mean value and normalization')
    plt.show()
    
    train_obs = train_obs.transpose(0,2,1)
    valid_obs = valid_obs.transpose(0,2,1)
    print('train_targets:', train_targets.shape)
    print('train_targets_shift:', train_targets_shift.shape)
    print('train_targets_norm:', train_targets_norm.shape)
    print('train_obs:', train_obs.shape)
    train_obs, valid_obs = denoise1(train_obs, valid_obs)
    # exit()
    train_obs_in = suppress_out_transit(train_obs)
    valid_obs_in = suppress_out_transit(valid_obs)
    print('train_obs_in:', train_obs_in.shape)
    train_obs_2d_mean = substract_data_mean(train_obs_in)
    print('train_obs_2d_mean:', train_obs_2d_mean.shape)
    print('train_obs_2d.mean:', train_obs_2d_mean.mean())
    valid_obs_2d_mean = substract_data_mean(valid_obs_in)
    
    train_obs_norm, valid_obs_norm, data_abs_max = data_norm(train_obs_2d_mean, valid_obs_2d_mean)
    train_obs_norm_mean = train_obs_norm.mean()
    train_obs_norm_max = train_obs_norm.max()
    train_obs_norm_min = train_obs_norm.min()

    print('train_obs_norm mean:', train_obs_norm_mean)
    print('train_obs_norm max:', train_obs_norm_max)
    print('train_obs_norm min:', train_obs_norm_min)
    plt.figure(figsize=(15,5))
    for i in range (train_obs.shape[0]) :
        plt.plot(wls, train_obs_norm[i,10], 'g-', alpha = 0.5)
    plt.plot([], [], 'g-', alpha=0.5, label='Train targets')
    for i in range (valid_obs.shape[0]) : 
        plt.plot(wls, valid_obs_norm[i,10], 'r-', alpha = 0.7)
    plt.plot([], [], 'r-', alpha=0.5, label='Valid targets (true mean)')

    plt.legend()
    plt.ylabel(f'$(R_p/R_s)^2$')
    plt.title('Train and Valid data after substracting the mean value and normalization')
    plt.show()
    
    return train_obs_norm, valid_obs_norm, train_targets_norm, valid_targets_norm, targets_abs_max

def set_normalization_output_dir(output_dir='./normalized_data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created output directory {output_dir}')
    else:
        print(f'Output directory {output_dir} already exists')

if __name__ == '__main__':
    set_normalization_output_dir('./pre_processed_2d')
    train_obs_norm, valid_obs_norm, train_targets_norm, valid_targets_norm, targets_abs_max = pre_process_2d()
    np.save(f'{normalized_data_folder}/train_obs_norm.npy', train_obs_norm)
    np.save(f'{normalized_data_folder}/valid_obs_norm.npy', valid_obs_norm)
    np.save(f'{normalized_data_folder}/train_targets_norm.npy', train_targets_norm)
    np.save(f'{normalized_data_folder}/valid_targets_norm.npy', valid_targets_norm)
    np.save(f'{normalized_data_folder}/targets_abs_max.npy', targets_abs_max)