import pandas as pd
import numpy as np
from tensorflow.keras import layers, optimizers, Sequential, Model, optimizers, metrics
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping



###

def initialize_model():

    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation = 'relu', padding = 'same', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (2, 2), activation = 'relu', padding = 'same'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4, activation = 'softmax'))

    return model



def compile_model(model):
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizers.Adam(learning_rate = 0.01),
                  metrics = ['accuracy'])
    return model


def train_model(
        model,
        data_train: pd.DataFrame,
        batch_size,
        patience,
        epochs,
        ):


    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=2
    )

    history = model.fit(
        data_train,
        validation_split=0.3,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    return model, history


def model_train(data_train: pd.DataFrame, batch_size: int, patience:int, epochs:int):
    model = initialize_model()
    model = compile_model(model)
    history = train_model(model, data_train, batch_size=batch_size, patience=patience, epochs=epochs)

    return history
