from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Concatenate, AveragePooling1D
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.losses import MeanAbsoluteError

def scheduler(epoch, lr):
    decay_rate = 0.2
    decay_step = 200
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

def create_model():
    imput_wc = Input(shape=(187,1))
    x = Conv1D(32, 3, activation='relu')(imput_wc)
    x = MaxPooling1D()(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D()(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D()(x)
    x = Conv1D(256, 3, activation='relu')(x)
    x = MaxPooling1D()(x)
    x = Flatten()(x)

    x = Dense(500, activation='relu')(x)
    x = Dropout(0.2)(x, training=True)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.1)(x, training=True)
    output_wc = Dense(1, activation='linear')(x)
    model = Model(inputs=imput_wc, outputs=output_wc)

    optimizer = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=optimizer, loss='mse', metrics=[MeanAbsoluteError()])
    model.summary()
    return model