from ml_logic import preprocessor
import io
import numpy as np
import pandas as pd
from PIL import Image
from ml_logic.registry import load_model
from ml_logic.preprocessor import make_square_with_padding
from ml_logic.params import *
from interface import test_demo

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2


#pa = 'data_parent/A_raw_data/Testing/glioma/Te-gl_0010.jpg'
pa = 'data_parent/A_raw_data/Testing/glioma/Te-glTr_0004.jpg'
im = cv2.imread(pa)

img_size = (150,150)
model_binary = load_model(LOCAL_REGISTRY_PATH_BINARY)
model_cats = load_model(LOCAL_REGISTRY_PATH_CLASS)

image_processed = make_square_with_padding(im, img_size)
image_processed = np.expand_dims(image_processed, axis=0)

y_pred = test_demo.predict(image_processed,model_binary,model_cats)
print(float(y_pred[0][0][0]))
print(type(y_pred[0][0][0]))
y_pred_cats = y_pred[1]
cat = pd.DataFrame({
    'Type': ["notumor","glioma", 'meningioma', 'pituitary'],
    'Proba': y_pred_cats[0]
})

print(cat)
