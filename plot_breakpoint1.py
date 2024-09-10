from algo.breakpoint import *
import pandas as pd
import imageio
import shutil

auxiliary_folder = './auxiliary'
data_folder = './data'

if __name__ == "__main__":
    data_train = np.load(f'{data_folder}/data_train.npy')
    data_train_FGS = np.load(f'{data_folder}/data_train_FGS.npy')
    signal_AIRS_diff_transposed_binned, signal_FGS_diff_transposed_binned  = data_train, data_train_FGS
    FGS_column = signal_FGS_diff_transposed_binned.sum(axis = 2)
    dataset = np.concatenate([signal_AIRS_diff_transposed_binned, FGS_column[:,:, np.newaxis,:]], axis = 2)
    # shape: 673, 187, 283, 32; num_samples, num_time_step, num_wavelength, num_spatial_dim
    dataset = dataset.sum(axis=3)
    # shape: 673, 187, 283; num_samples, num_time_step, num_wavelength
    dataset = dataset.sum(axis=2)
    # shape: 673, 187; num_samples, num_time_step
    train_labels = pd.read_csv('auxiliary/train_labels.csv', index_col="planet_id")
    file_names = []
    for IDX in range(len(dataset)):
    # for IDX in range(10):
        data = dataset[IDX]
        find_and_plot_breakpoints(data, IDX, train_labels, verbose=True)
        file_names.append(f'tmp/breakpoint_{IDX}.png')
    
    with imageio.get_writer('breakpoints1.mp4', mode='I', fps=1) as writer:
        for filename in file_names:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    shutil.rmtree('tmp')