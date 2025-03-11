from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image
import requests
import os
import glob
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, optimizers, callbacks

from tensorflow.keras.utils import image_dataset_from_directory

from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import models
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import numpy as np

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
        model: Model,
        train_set: pd.DataFrame,
        batch_size=32,
        patience=2,
        validation_data=pd.DataFrame, # overrides validation_split
        validation_split=None):
    """
    Fit the model and return a tuple (fitted_model, history)
    """

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=2
    )

    history = model.fit(
        train_set = train_set,
        validation_data = validation_data,
        validation_split=validation_split,
        epochs=50,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    return model, history


def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('accuracy')
    ax[1].plot(history.epoch, history.history["accuracy"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()


'''
model = initialize_model()
model = compile_model(model)

es = EarlyStopping(patience = 5, verbose = 2)
history = model.fit(
        train_ds,
        epochs=50,
        validation_data=val_ds,
        callbacks = [es],
        verbose = 2)
'''
