import data as data

import ml_logic.data_augment as data_augment
import pandas as pd
import utils
import shutil
import os

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics

import matplotlib.pyplot as plt




################################## VBinary ##################################



### Rearranging the train directories in a Binary split ###
def moving_and_splitting_train(path_train_prepro: str):
    print('🧮 Splitting the no tumor and tumor data')

    path_binary_train = 'data_parent/C_data_binary/train_binary'

    train_glioma= path_train_prepro + "/glioma"
    train_meningioma= path_train_prepro + "/meningioma"
    train_pituitary= path_train_prepro + "/pituitary"
    train_binary_tumor= path_binary_train + "/tumor"


    # Merging the different type of tumor
    shutil.copytree(train_glioma, train_binary_tumor, dirs_exist_ok=True)
    shutil.copytree(train_meningioma, train_binary_tumor, dirs_exist_ok=True)
    shutil.copytree(train_pituitary, train_binary_tumor, dirs_exist_ok=True)

    train_notumor= path_train_prepro + "/notumor"
    train_binary_notumor= path_binary_train + "/notumor"

    shutil.copytree(train_notumor, train_binary_notumor, dirs_exist_ok=True)

    print('💾 Splitting train done')
    return path_binary_train


### Rearranging the test directories in a Binary split ###
def moving_and_splitting_test(path_test_prepro: str):
    print('🧮 Splitting the test no tumor and tumor data')

    path_binary_test = 'data_parent/C_data_binary/test_binary'

    test_glioma= path_test_prepro + "/glioma"
    test_meningioma= path_test_prepro + "/meningioma"
    test_pituitary= path_test_prepro + "/pituitary"
    test_binary_tumor= path_binary_test + "/tumor"


    # Merging the different type of tumor
    shutil.copytree(test_glioma, test_binary_tumor, dirs_exist_ok=True)
    shutil.copytree(test_meningioma, test_binary_tumor, dirs_exist_ok=True)
    shutil.copytree(test_pituitary, test_binary_tumor, dirs_exist_ok=True)

    test_notumor= path_test_prepro + "/notumor"
    test_binary_notumor= path_binary_test + "/notumor"

    shutil.copytree(test_notumor, test_binary_notumor, dirs_exist_ok=True)

    print('💾 Splitting done')
    return path_binary_test



def data_augmentation_notumor(nbr_img: int, path_binary: str):

    '''Complete the dataset train with new picture for data augmentation'''
    print('🧮 Making augmented pictures')
    ### Variable for data augmentation ###

    n = nbr_img # -> nbr of picture generated of non-tumor
    # directory to store the images if it does not exist it will be created
    augdir=r'data_parent/D_data_aug'
    path_binary_no_tumor = path_binary + '/notumor'
    df_binary_no_tumor = utils.load_precise_dataframe(path_binary_no_tumor) # -> Df where preprocessed binary no tumor is


    ### Data augmentation ###

    data_augment.make_and_store_images(df_binary_no_tumor, augdir, n)

    ### Mix the augmented data with the rest of tumor ###

    shutil.copytree(augdir, path_binary_no_tumor, dirs_exist_ok=True)

    print('💾 Merging augmented picture')



def data_suppression(nbr_img: int, path_binary: str):

    ''' Delete some files in the tumor train set to be equal to non tumor train'''
    print("🧮 Start equalize tumor and no tumor")
    path_binary_tumor = path_binary + '/tumor'
    df_binary_no_tumor = utils.load_precise_dataframe(path_binary_tumor)
    delta = len(df_binary_no_tumor) - nbr_img # -> the difference between tumor and no tumor

    df_to_remove = df_binary_no_tumor.sample(delta)

    for i in df_to_remove['images_paths']:
        os.remove(i)
    print('💾 Delete completed, equalize finished')



def make_tensor_train(path_train_binary: str, img_size):
    ''' Make a tensor dataset train to feed the model'''
    ds_train = image_dataset_from_directory(
        path_train_binary,
        labels="inferred",
        class_names=["notumor", 'tumor'],
        label_mode="binary",
        seed=123,
        validation_split=0.25,
        subset='both',
        image_size=img_size,
        batch_size=32)

    return ds_train


def make_tensor_test(path_test_binary: str, img_size):
    ''' Make a tensor dataset train to feed the model'''
    ds_test = image_dataset_from_directory(
        path_test_binary,
        labels="inferred",
        class_names=["notumor", 'tumor'],
        label_mode="binary",
        seed=123,
        image_size=img_size,
        batch_size=32)

    return ds_test


def load_model():
    model = VGG16(weights="imagenet", include_top=False, input_shape=(150,150,3))
    return model


def set_nontrainable_layers(model):
    model.trainable = False
    return model


def add_last_layers(model):
    '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
    non_trainable_model = set_nontrainable_layers(model)

    # Convolutional Layers

    new_model = Sequential()
    new_model.add(non_trainable_model)
    new_model.add(layers.Input((150, 150, 3)))
    new_model.add(layers.Rescaling(1./255))
    new_model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"))
    new_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same") )

    new_model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"))
    new_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same") )

    new_model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))
    new_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same") )

    new_model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", padding="same"))
    new_model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same") )

    new_model.add(layers.Flatten())

    new_model.add(layers.Dense(64, activation="relu"))

    new_model.add(layers.Dropout(0.5))
    new_model.add(layers.Dense(1, activation="sigmoid"))

    return new_model


def build_model():

    ''' Initialize the model and build it'''
    model = load_model()
    model = add_last_layers(model)

    opt = optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=[metrics.Recall()])
    return model



def train_model_bin(path_train_prepro: str, path_test_prepro : str, nbr_img: int, img_size, patience: int, epochs: int, batch_size: int):
    ''' Starting the process for training the model'''
    print("🧮 Start Binary Model")

    # Splitting the dataset train between tumor and no tumor
    path_binary_train = moving_and_splitting_train(path_train_prepro)

    # Splitting the dataset train between tumor and no tumor
    path_binary_test = moving_and_splitting_test(path_test_prepro)

    # Making new augmented picture
    data_augmentation_notumor(nbr_img, path_binary_train)

    # Equalize nbr of tumor and no tumor
    data_suppression(nbr_img, path_binary_train)

    # Making a tensor of the binary dataset train
    ds_train = make_tensor_train(path_binary_train, img_size)

    # Making a tensor of the dataset test
    ds_test = make_tensor_test(path_binary_test, img_size)

    print('🧮 Training of the binary model')

    # Initialize the model
    model=build_model()

    # Params for early stopping
    es = EarlyStopping(
        patience=patience,
        restore_best_weights=True,
        verbose=2
    )

    # Start to train the model

    history = model.fit(
            ds_train[0],
            epochs=epochs,
            callbacks=[es],
            batch_size=batch_size,
            validation_data=ds_train[1]
            )

    print('🧮 Evalutation of the model on test')
    scores = model.evaluate(ds_test)

    print('⭐ Return of the results')

    return history, scores, model
