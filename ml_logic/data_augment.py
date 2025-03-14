import os
import pandas as pd
import shutil
import tensorflow as tf
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def make_and_store_images(df_binary_no_tumor, augdir, n, color_mode='rgb', save_prefix='aug-',save_format='jpg'):
    #augdir is the full path where augmented images will be stored
    #n is the number of augmented images that will be created for each class that has less than n image samples
    # img_size  is a tupple(height,width) that specifies the size of the augmented images
    # color_mode is 'rgb by default'
    # save_prefix is the prefix augmented images are identified with by default it is 'aug-'
    #save_format is the format augmented images will be save in, by default it is 'jpg'
    # see documentation of ImageDataGenerator at https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator for details



    total=0
    # in ImageDateGenerator select the types of augmentation you desire  below are some examples
    gen=ImageDataGenerator(horizontal_flip=True,  rotation_range=20, width_shift_range=.1,
                                  height_shift_range=.1, zoom_range=.1)

    sample_count = len(df_binary_no_tumor)   # determine how many pic of no tumor
    if sample_count< n: # if the class has less than target number of images
            aug_img_count=0
            delta=n - sample_count  # number of augmented images to create

            im = cv2.imread(df_binary_no_tumor['images_paths'][0])
            img_size = im.shape[:2]

            aug_gen=gen.flow_from_dataframe(df_binary_no_tumor,  x_col='images_paths', y_col=None, target_size=img_size,
                                            class_mode=None, batch_size=1, shuffle=False,
                                            save_to_dir=augdir, save_prefix=save_prefix, color_mode=color_mode,
                                            save_format=save_format)
            while aug_img_count<delta:
                images=next(aug_gen)
                aug_img_count += len(images)
            total +=aug_img_count

    print('💾 Total Augmented images created= ', total)
    return None
