import os
import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import image_dataset_from_directory

'''
If local is True:
    - Load the data from a local path train and test if local variable is true
        If dataframe true:
            - Return 2 dataframe with path of all pictures and theirs labels
            - Or return 2 dataset with pictures and theirs labels
'''

def load_data_dataframe(path_data: str):
    '''
    If type_data is dataframe:
        - return a dataframe to process
    '''
    train_data_dir=path_data

    # First step with datatrain
    folders_train = os.listdir(train_data_dir)
    images_paths_train =[]
    labels_train = []

    # Path of each image in each folder
    for folder in folders_train:
        folder_path_train = os.path.join(train_data_dir, folder)
        list_image_train = os.listdir(folder_path_train)
        for image in list_image_train :
            image_path = os.path.join(folder_path_train, image)
            images_paths_train.append(image_path)
            labels_train.append(folder)

    #Df with path and labels
    img_path_train = pd.Series(images_paths_train, name='images_paths')
    label_train = pd.Series(labels_train, name='labels')
    df = pd.concat([img_path_train, label_train], axis=1)

    return df # Return a dataframe



def load_data_dataset(path_train: str,
                        batch_size: int =64,
                        image_size: tuple[int, int] =(150, 150),
                        crop_flag: bool =True) -> "tf.data.Dataset" :
    """
    Pre-requesites :
    *   Each class of images must be stored in their own subdirectory.
    *   Subdirectory names must have a name matching the class
    Images will be uploaded in a `Tensorflow Datasets` object (https://www.tensorflow.org/api_docs/python/tf/data/Dataset).
    Sub-directory names will be used as labels for classification.
    This will allow us to:
    *   Grab images from our directory batch by batch, we won't load ALL the data at the same time
    *   Reshape all the images to our desired input shape
    *   crop image if needed
    """
    dataset = image_dataset_from_directory(directory=path_train,
                                            labels="inferred",
                                            label_mode="categorical",
                                            seed=42,
                                            batch_size=batch_size,
                                            image_size=image_size,
                                            crop_to_aspect_ratio = crop_flag)

    return dataset






'''
If local is True:
    - Load the data from a local path train and test if local variable is true
        If dataframe true:
            - Return 2 dataframe with path of all pictures and theirs labels
            - Or return 2 dataset with pictures and theirs labels
'''

def load_data(local: bool, path: str, type_data: bool):
    if local is True:
        # If you already download the dataset:
        print('File is found on local!')

        if type_data is True:
            df = load_data_dataframe(path)
            print('⭐ dataframe completed ⭐')
            return df # Return 1 df

        else:
            ds = load_data_dataset(path)
            print('⭐ dataset completed ⭐')
            return ds # Return 1 df
