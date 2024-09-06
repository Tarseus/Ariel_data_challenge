from model.CNN1d import create_model, scheduler
from keras.models import Model, load_model
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

checkpoint_filepath = 'output/model_1dcnn.h5'
def train(model, train_data, train_label, valid_data, valid_label):
    callback = LearningRateScheduler(scheduler)
    model_ckt = ModelCheckpoint(
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
        epochs=1200,
        batch_size=32,
        validation_data=(valid_data, valid_label),
        callbacks=[callback, model_ckt]
    )
    return history

if __name__ == '__main__':
    model = create_model()
    train_dir = './normalized_data'
    train_data = np.load(f'{train_dir}/train_wc.npy')
    train_label = np.load(f'{train_dir}/train_targets_wc_norm.npy')
    valid_data = np.load(f'{train_dir}/valid_wc.npy')
    valid_label = np.load(f'{train_dir}/valid_targets_wc_norm.npy')
    # train(model, train_data, train_label, valid_data, valid_label)