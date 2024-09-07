from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Reshape, Dropout, BatchNormalization, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanAbsoluteError
import tensorflow as tf
import numpy as np

def create_model():
    input_obs = Input((40, 283, 1))
    x = Conv2D(32, (3,1), activation='relu', padding='same')(input_obs)
    x = MaxPooling2D((2,1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,1), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,1))(x)
    x = Conv2D(128, (3,1), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,1))(x)
    x = Conv2D(256, (3,1), activation='relu', padding='same')(x)
    x = Conv2D(32, (1,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((1,2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (1,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((1,2))(x)
    x = Conv2D(128, (1,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((1,2))(x)
    x = Conv2D(256, (1,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((1,2))(x)
    x = Flatten()(x)
    
    x = Dense(700, activation='relu')(x)
    x = Dropout(0.2)(x, training=True)
    output = Dense(283, activation='linear')(x)
    
    model = Model(inputs=[input_obs], outputs=output)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=[MeanAbsoluteError()])
    model.summary()
    return model