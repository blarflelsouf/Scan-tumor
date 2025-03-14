import os
import pandas as pd
import shutil
import tensorflow as tf
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def make_and_store_images(data_train, augdir, n, color_mode='rgb', save_prefix='aug-',save_format='jpg'):
    #augdir is the full path where augmented images will be stored
    #n is the number of augmented images that will be created for each class that has less than n image samples
    # img_size  is a tupple(height,width) that specifies the size of the augmented images
    # color_mode is 'rgb by default'
    # save_prefix is the prefix augmented images are identified with by default it is 'aug-'
    #save_format is the format augmented images will be save in, by default it is 'jpg'
    # see documentation of ImageDataGenerator at https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator for details
    df_augm=data_train
    if os.path.isdir(augdir):# start with an empty directory
        shutil.rmtree(augdir)
    os.mkdir(augdir)  # if directory does not exist create it
    for label in df_augm['labels'].unique():
        classpath=os.path.join(augdir,label)
        os.mkdir(classpath) # make class directories within aug directory
    # create and store the augmented images
    total=0
    # in ImageDateGenerator select the types of augmentation you desire  below are some examples
    gen=ImageDataGenerator(horizontal_flip=True,  rotation_range=20, width_shift_range=.1,
                                  height_shift_range=.1, zoom_range=.1)
    groups=df_augm.groupby('labels') # group by class
    for label in df_augm['labels'].unique():  # for every class
        classdir=os.path.join(augdir, label)
        group=groups.get_group(label)  # a dataframe holding only rows with the specified label
        sample_count=len(group)   # determine how many samples there are in this class
        if sample_count< n: # if the class has less than target number of images
            aug_img_count=0
            delta=n - sample_count  # number of augmented images to create
            msg='{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', label, str(delta))
            im = cv2.imread(group['images_paths'][0])
            img_size = im.shape[:2]
            print(msg, '\r', end='') # prints over on the same line
            aug_gen=gen.flow_from_dataframe(group,  x_col='images_paths', y_col=None, target_size=img_size,
                                            class_mode=None, batch_size=1, shuffle=False,
                                            save_to_dir=classdir, save_prefix=save_prefix, color_mode=color_mode,
                                            save_format=save_format)
            while aug_img_count<delta:
                images=next(aug_gen)
                aug_img_count += len(images)
            total +=aug_img_count
    return print('Total Augmented images created= ', total)


path = 'data_parent/A_raw_data/'
data_to_augment = path + 'Training/notumor'



    data_augment.make_and_store_images(data_train = utils.load_precise_dataframe(data_to_augment),
                        augdir = 'data_parent/B_data_augmented',
                        n = 1696, # -> Nbr of non tumor to augmented, if i want 2k pic of non-tumor, i put n=2000
                        color_mode='rgb',
                        save_format='png')
