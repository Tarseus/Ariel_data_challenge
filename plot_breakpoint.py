from algo.breakpoint1 import *
from algo.breakpoint2 import *
import pandas as pd
import imageio
import shutil

auxiliary_folder = './auxiliary'
data_folder = './data'
if_plot = True
export_csv = True

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
    
    results = pd.DataFrame(columns=['planet_id', 'bound1', 'bound2', 'bound3', 'bound4'])
    for IDX in range(len(dataset)):
    # for IDX in range(630, 640):
        data = dataset[IDX]
        bounds = find_derivative(data, IDX, train_labels, verbose=True, plot=if_plot)
        if if_plot:
            file_names.append(f'tmp/breakpoint_{IDX}.png')
        planet_id = train_labels.index[IDX]
        new_row = pd.DataFrame({
        'planet_id': [planet_id],
        'bound1': [bounds[0]],
        'bound2': [bounds[1]],
        'bound3': [bounds[2]],
        'bound4': [bounds[3]],
        })
        results = pd.concat([results, new_row], ignore_index=True)
    results.to_csv('auxiliary/breakpoints.csv', index=False)
    
    if len(file_names) > 0:
        with imageio.get_writer('breakpoints.mp4', mode='I', fps=2.5) as writer:
            for filename in file_names:
                image = imageio.imread(filename)
                writer.append_data(image)
        shutil.rmtree('tmp')