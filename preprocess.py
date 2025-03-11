from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image
import requests
import os
import glob
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


# Get the data
glioma_train = glob.glob('data_test/glioma_trainV2/*')
meningioma_train = glob.glob('data_test/meningioma_trainV2/*')
no_tumor_train = glob.glob('data_test/no_tumor_trainV2/*')
pituitary_train = glob.glob('data_test/pituitary_trainV2/*')

# Load and clean the images
def load_clean_image(image):
    img = load_img(image)
    img_array = img_to_array(img)
    img_array = tf.image.resize(img_array, [224, 224])
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255
    return img_array


# Put images in a list

list_glioma = []
for image in glioma_train:
    list_glioma.append(load_clean_image(image))

list_meningioma = []
for image in meningioma_train:
    list_meningioma.append(load_clean_image(image))

list_no_tumor = []
for image in no_tumor_train:
    list_no_tumor.append(load_clean_image(image))


df_glioma = pd.DataFrame(list_glioma)
df_glioma['target'] = 'glioma'

df_meningioma = pd.DataFrame(list_meningioma)
df_meningioma['target'] = 'meningioma'

df_no_tumor = pd.DataFrame(list_no_tumor)
df_no_tumor['target'] = 'no_tumor'

df_total = pd.concat([df_glioma, df_meningioma, df_no_tumor])

df_total.to_csv('data_test/df_total.csv', index=False)
