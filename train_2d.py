from model.CNN1d import create_model, scheduler
from keras.models import Model, load_model
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

checkpoint_filepath = 'output/model_2dcnn.h5'
def train(model, train_data, train_label, valid_data, valid_label):
    model_ckt2 = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        save_freq='epoch',
    )
    history = model.fit(
        x=train_data,
        y=train_label,
        epochs=200,
        batch_size=32,
        validation_data=(valid_data, valid_label),
        callbacks=[model_ckt2]
    )
    return history

if __name__ == '__main__':
    model = create_model()
    train_dir = './pre_processed_2d'
    train_data = np.load(f'{train_dir}/train_obs_norm.npy')
    train_label = np.load(f'{train_dir}/train_targets_norm.npy')
    valid_data = np.load(f'{train_dir}/valid_obs_norm.npy')
    valid_label = np.load(f'{train_dir}/valid_targets_norm.npy')
    train(model, train_data, train_label, valid_data, valid_label)