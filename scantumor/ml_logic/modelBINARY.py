import data as data
import modelCNN as modelCNN
import preprocessor as prepro
import modelvgg16 as modelvgg
import data_augment as data_augment
import pandas as pd
import utils
import shutil
import splitfolders
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, optimizers, callbacks, models
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt




################################## VBinary ##################################



### Rearranging the directories in a Binary split ###
def moving_and_splitting(path_train: str):
    print('🧮 Splitting the no tumor and tumor data')

    path_binary = 'data_parent/C_data_binary'

    train_glioma= path_train + "/glioma"
    train_meningioma= path_train + "/meningioma"
    train_pituitary= path_train + "/pituitary"
    train_binary_tumor= path_binary + "/tumor"

    shutil.copytree(train_glioma, train_binary_tumor, dirs_exist_ok=True)
    shutil.copytree(train_meningioma, train_binary_tumor, dirs_exist_ok=True)
    shutil.copytree(train_pituitary, train_binary_tumor, dirs_exist_ok=True)

    train_notumor= path_train + "/notumor"
    train_binary_notumor= path_binary + "/notumor"

    shutil.copytree(train_notumor, train_binary_notumor, dirs_exist_ok=True)

    print('💾 Splitting done')
    return path_binary



def data_augmentation_notumor(nbr_img: int, augdir):

    '''Complete the dataset train with new picture for data augmentation'''

    ### Variable for data augmentation ###

    n = nbr_img # -> nbr of picture generated of non-tumor
    augdir=r"/home/landsberg/code/blarflelsouf/scan-tumor/raw_data/archive/augmented_binary_train_split" # directory to store the images if it does not exist it will be created
    img_size = (150, 150) # -> choose the size of picture

    ### Data augmentation ###

    utils.load_data_dataframe(binary_train_split)
    data_augment.make_and_store_images(binary_train_split,augdir,n)

                    ######### Mix the original and preprocessed data #########

    completed_binary_train="/home/landsberg/code/blarflelsouf/scan-tumor/raw_data/archive/Completed_Binary"

    shutil.copytree(binary_train_split, completed_binary_train, dirs_exist_ok=True)
    shutil.copytree(augdir, completed_binary_train, dirs_exist_ok=True)











                ######### directories into trainable datasets #########

batch_size=8

train_ds = image_dataset_from_directory(
    completed_binary_train,
    labels="inferred",
    class_names=["notumor","tumor"],
    label_mode="binary",
    seed=123,
    image_size=(150, 150),
    batch_size=batch_size)


test_ds = image_dataset_from_directory(
    test_binary_dir,
    labels="inferred",
    class_names=["notumor","tumor"],
    label_mode="binary",
    seed=123,
    image_size=(150, 150),
    batch_size=batch_size)


                ######### Train a model #########

'''Train a model and report his history. You can choose between:
    - the batch size of training
    - the number of epochs
'''

batch_size = 8
epochs = 8
model=modelvgg.build_model()
history=modelvgg.train_model(train_ds, epochs, batch_size)



def train_model_bin(path_train_prepro: str):
    print("🧮 Start Binary Model")

    path_binary = moving_and_splitting(path_train_prepro)
