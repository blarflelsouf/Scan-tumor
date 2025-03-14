from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import cv2
import os
import shutil

def make_square_with_padding(image: np.ndarray, dest_img_size) -> np.ndarray:
    """
    Transform a rectangular image into a square image by adding padding.
    """

    color= (0, 0, 0) #Black
    height, width = image.shape[:2]
    size = max(height, width)

    # Compute the padding sizes
    top = (size - height) // 2
    bottom = size - height - top
    left = (size - width) // 2
    right = size - width - left

    # # Apply padding
    squared_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    image_resize = cv2.resize(squared_image,dest_img_size)


    return image_resize

def preprocess_create_dir(src_img_df, root_dest_dir):
    """
    CAUTION :
    *   Existing repositories/files in {root_dest_dir}/E_data_to_process WILL BE CLEARED by this function
    and replaced by the images generated

    Input params :
    *   src_img_df is a custom dataframe from the project having the full path/name of the image in column 1
    and its class in column 2.
    *   root_dest_dir is the directory in which squared images will be created with the hardecode.

    Output :
    *   No output params
    *   Classified images from the dataframe will be resized to a squared image with padding of the requested size. Then
    they are saved in a new "E_data_to_process" directory in the root_dest_dir provided as an input.
    Each class will have its own sub-directory.
    Each file with be named as the source image.
    """
    #################################### REPOSITORY PREPARATION #######################################
    '''
    if os.path.split(root_dest_dir)[1] != "data_parent":
        print(f"ERROR - {os.path.split(root_dest_dir)[1]} - Destination repository should be in raw_data project repository \
              to be sure that cleaning of 'E_data_to_process' sub-directory does not delete another existing repository")
        return None
    '''

    img_preprocessed_dir = root_dest_dir #repository where files will be written

    # Clean destination directory and then create it if needed
    if os.path.isdir(img_preprocessed_dir):# start with an empty directory
        print(f"Cleaning of the directory {img_preprocessed_dir}")
        shutil.rmtree(img_preprocessed_dir)
    os.mkdir(img_preprocessed_dir)

    # Create label sub-repositories
    for label in src_img_df['labels'].unique():
            classpath=os.path.join(img_preprocessed_dir,label)
            os.mkdir(classpath)

    print('💾 Repository created!')

    return img_preprocessed_dir



def preprocess_write_image(src_img_df: pd.DataFrame,
                        dest_img_size: tuple[int, int],
                        img_preprocessed_dir):

    ##################################### RESIZING IMAGES AND WRITE THEM INTO REPOSITORY ##################
    # Create squared images with a black padding
    print('🧮 Images are in padding and resize process')
    nb_image = 0
    for index, row in src_img_df.iterrows():
        nb_image += 1
        image = cv2.imread(row[0])
        image_name = os.path.split(row[0])[1] #Tail = image name
        square_image = make_square_with_padding(image, dest_img_size) # Squares and resizes image
        cv2.imwrite(f"{img_preprocessed_dir}/{row[1]}/{image_name}", square_image)  # Save the output image
    print(f"{nb_image} images resized to squared image with padding in repository {img_preprocessed_dir}")


    print('⭐ Images created')

    return None




def preprocess_write_squared_image_to_dir(src_img_df: pd.DataFrame,
                                          dest_img_size: tuple[int, int],
                                          root_dest_dir: str) -> None :
    img_preprocessed_dir = preprocess_create_dir(src_img_df, root_dest_dir)
    preprocess_write_image(src_img_df, dest_img_size, img_preprocessed_dir)

    print('⭐ Fin du preprocessing')
