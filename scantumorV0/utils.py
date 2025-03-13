import os
import pandas as pd

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
