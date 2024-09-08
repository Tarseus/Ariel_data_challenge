import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf 
import random 
import os
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from utils import *
import imageio
import shutil
from tqdm import tqdm

data_folder = './data'
auxiliary_folder = './auxiliary'
normalized_data_folder = './pre_processed_1d'
data_train = np.load(f'{data_folder}/data_train.npy')
data_train_FGS = np.load(f'{data_folder}/data_train_FGS.npy')
print(data_train.shape)
print(data_train[0].shape)
print(data_train_FGS.shape)
print(data_train_FGS[0].shape)

def set_output_dir(output_dir='./output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created output directory {output_dir}')
    else:
        print(f'Output directory {output_dir} already exists')
    
def set_normalization_output_dir(output_dir='./normalized_data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created output directory {output_dir}')
    else:
        print(f'Output directory {output_dir} already exists')
    
    
def save_video(dataset, train_wc, tmp_dir='./tmp'):
    bounds = np.loadtxt(f"{auxiliary_folder}/breakpoints.csv", delimiter=',', skiprows=1, usecols=(1, 2, 3, 4)).astype(np.int16)
    os.makedirs(tmp_dir, exist_ok=True)
    plt.figure()
    file_names = []
    for i in tqdm(range(train_wc.shape[0])):
    # for i in tqdm(range(10)):
        data, norm_data, bound = dataset[i], train_wc[i], bounds[i]
        # data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data1 = data / np.max(data)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        ax1.plot(data1, color='#4E79A7', alpha=0.7, label="Data")
        ax2.plot(norm_data, color='#F28E2B', label="Normalized Data")
        # ax1.set_title('Light-curves from the train set') 
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        for j in range(2):
            ax1.axvline(x=bound[2*j + 0], color="r", linestyle="--")
            ax1.axvline(x=bound[2*j + 1], color="r", linestyle="--")
            ax1.axvspan(bound[2*j + 0], bound[2*j + 1], color="gray", alpha=0.3)
            ax2.axvline(x=bound[2*j + 0], color="r", linestyle="--")
            ax2.axvline(x=bound[2*j + 1], color="r", linestyle="--")
            ax2.axvspan(bound[2*j + 0], bound[2*j + 1], color="gray", alpha=0.3)
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Data')
        ax2.set_ylabel('Normalized Data')
        
        fig.tight_layout()
        fig.suptitle('Light-curve from the train set')
        plt.savefig(f'tmp/{i}.png')
        plt.close()
        file_names.append(f'tmp/{i}.png')
    with imageio.get_writer('norm_light_curve.mp4', mode='I', fps=1.25) as writer:
        for filename in file_names:
            image = imageio.imread(filename)
            writer.append_data(image)
    shutil.rmtree('tmp')
def pre_process_1d():
    train_solution = np.loadtxt(f'{auxiliary_folder}/train_labels.csv', delimiter = ',', skiprows = 1)
    targets = train_solution[:,1:]
    targets_mean = targets[:,1:].mean(axis = 1) # used for the 1D-CNN to extract the mean value, only AIRS wavelengths as the FGS point is not used in the white curve
    N = targets.shape[0]
    signal_AIRS_diff_transposed_binned, signal_FGS_diff_transposed_binned  = data_train, data_train_FGS
    FGS_column = signal_FGS_diff_transposed_binned.sum(axis = 2)
    dataset = np.concatenate([signal_AIRS_diff_transposed_binned, FGS_column[:,:, np.newaxis,:]], axis = 2)
    # shape: 673, 187, 283, 32; num_samples, num_time_step, num_wavelength, num_spatial_dim
    dataset = dataset.sum(axis=3)
    dataset_2d = dataset.sum(axis=2)
    # shape: 673, 187, 283; num_samples, num_time_step, num_wavelength
    dataset_norm = norm_star_spectrum(dataset)
    dataset_norm = np.transpose(dataset_norm,(0,2,1))
    # N_train = 8*N//10
    N_train = N
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
    
    save_video(dataset_2d, train_wc)
    return train_wc, valid_wc, train_targets_wc_norm, valid_targets_wc_norm, min_train_valid_wc, max_train_valid_wc

if __name__ == '__main__':
    set_output_dir()
    set_normalization_output_dir('./pre_processed_1d')
    (train_wc, valid_wc, train_targets_wc_norm, valid_targets_wc_norm, min_train_valid_wc, max_train_valid_wc) = pre_process_1d()
    np.save(f'{normalized_data_folder}/train_wc.npy', train_wc)
    np.save(f'{normalized_data_folder}/valid_wc.npy', valid_wc)
    np.save(f'{normalized_data_folder}/train_targets_wc_norm.npy', train_targets_wc_norm)
    np.save(f'{normalized_data_folder}/valid_targets_wc_norm.npy', valid_targets_wc_norm)
    np.save(f'{normalized_data_folder}/min_train_valid_wc.npy', min_train_valid_wc)
    np.save(f'{normalized_data_folder}/max_train_valid_wc.npy', max_train_valid_wc)