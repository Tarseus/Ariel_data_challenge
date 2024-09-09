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

data_folder = './data/w_lin_corr'
auxiliary_folder = './auxiliary'
normalized_data_folder = './pre_processed_1d'
data_train = np.load(f'{data_folder}/data_train.npy')
data_train_FGS = np.load(f'{data_folder}/data_train_FGS.npy')
capture_frames = -1

def vis_star_norm():
    signal_AIRS_diff_transposed_binned, signal_FGS_diff_transposed_binned = data_train, data_train_FGS
    signal_AIRS_diff_transposed_binned = signal_AIRS_diff_transposed_binned.sum(axis=3)
    # signal_AIRS_diff_transposed_binned , _ = normalise_wlc(signal_AIRS_diff_transposed_binned,
    #                                                        signal_AIRS_diff_transposed_binned)
    signal_AIRS_diff_transposed_binned = norm_star_spectrum(signal_AIRS_diff_transposed_binned)
    # signal_norm = np.transpose(signal_norm, (0, 2, 1))
    # signal_denoise, _ = denoise(signal_norm, signal_norm)
    
    white_curve = signal_AIRS_diff_transposed_binned.mean(axis=2)
    save_video(white_curve)

def save_video(dataset, tmp_dir='./tmp'):
    bounds = np.loadtxt(f"{auxiliary_folder}/breakpoints.csv", delimiter=',', skiprows=1, usecols=(1, 2, 3, 4)).astype(np.int16)
    os.makedirs(tmp_dir, exist_ok=True)
    plt.figure()
    file_names = []
    if capture_frames == -1:
        num_frame = dataset.shape[0]
    else:
        num_frame = capture_frames
    # for i in tqdm(range(train_wc.shape[0])):
    
    for i in tqdm(range(num_frame)):
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
        data, bound = dataset[i], bounds[i]
        ax1.plot(data, alpha=0.7)  # 不设置颜色和标签
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Data')
        for j in range(2):
            ax1.axvline(x=bound[2*j + 0], color="r", linestyle="--")
            ax1.axvline(x=bound[2*j + 1], color="r", linestyle="--")
            ax1.axvspan(bound[2*j + 0], bound[2*j + 1], color="gray", alpha=0.3)
        
        fig.tight_layout()
        fig.suptitle('Light-curve from the train set')
        plt.savefig(f'tmp/{i}.png')
        plt.close()
        file_names.append(f'tmp/{i}.png')
    with imageio.get_writer('norm_star.mp4', mode='I', fps=3) as writer:
        for filename in file_names:
            image = imageio.imread(filename)
            writer.append_data(image)
    shutil.rmtree('tmp')

if __name__ == '__main__':
    vis_star_norm()