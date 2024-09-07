import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf 
import random 
import os
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from utils import *

data_folder = './data'
auxiliary_folder = './auxiliary'
normalized_data_folder = './pre_processed_2d'
data_train = np.load(f'{data_folder}/data_train.npy')
data_train_FGS = np.load(f'{data_folder}/data_train_FGS.npy')

def supress_mean(targets, mean):
    res = targets - np.repeat(mean.reshape((mean.shape[0], 1)), repeats=targets.shape[1], axis=1)
    return res

def suppress_out_transit(data, ingress, egress):
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

def pre_process_2d():
    signal_AIRS_diff_transposed_binned, signal_FGS_diff_transposed_binned  = data_train, data_train_FGS
    FGS_column = signal_FGS_diff_transposed_binned.sum(axis = 2)
    # print(signal_AIRS_diff_transposed_binned.shape, FGS_column[:, :, np.newaxis, :].shape)
    dataset = np.concatenate([signal_AIRS_diff_transposed_binned, FGS_column[:,:, np.newaxis,:]], axis = 2)
    train_solution = np.loadtxt(f'{auxiliary_folder}/train_labels.csv', delimiter = ',', skiprows = 1)
    targets = train_solution[:,1:]
    targets_mean = targets[:,1:].mean(axis = 1) # used for the 1D-CNN to extract the mean value, only AIRS wavelengths as the FGS point is not used in the white curve
    N = targets.shape[0]
    N_train = 8*N//10
    # Validation and train data split
    dataset_norm = norm_star_spectrum(dataset)
    dataset_norm = np.transpose(dataset_norm,(0,2,1))
    train_obs, valid_obs, list_index_train = split(dataset_norm, N_train)
    train_targets, valid_targets = targets[list_index_train], targets[~list_index_train]
    train_targets_shift = supress_mean(train_targets, targets_mean[list_index_train])
    valid_targets_shift = supress_mean(valid_targets, targets_mean[~list_index_train])
    train_targets_norm, valid_targets_norm, targets_abs_max = data_norm(train_targets_shift, valid_targets_shift)
    train_obs = train_obs.transpose(0,2,1)
    valid_obs = valid_obs.transpose(0,2,1)
    print(train_obs.shape)
    ingress, egress = 75, 115
    train_obs_in = suppress_out_transit(train_obs, ingress, egress)
    valid_obs_in = suppress_out_transit(valid_obs, ingress, egress)
    train_obs_2d_mean = substract_data_mean(train_obs_in)
    valid_obs_2d_mean = substract_data_mean(valid_obs_in)
    train_obs_norm, valid_obs_norm, data_abs_max = data_norm(train_obs_2d_mean, valid_obs_2d_mean)


if __name__ == '__main__':
    pre_process_2d()