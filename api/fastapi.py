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
import base64



app = FastAPI()
model_binary = load_model(LOCAL_REGISTRY_PATH_BINARY)
model_cats = load_model(LOCAL_REGISTRY_PATH_CLASS)

#Load models in app.state for predict-yolo
app.state.model_binary = load_model(LOCAL_REGISTRY_PATH_BINARY)
app.state.model_cats = load_model(LOCAL_REGISTRY_PATH_CLASS)
app.state.model_yolo = load_model(LOCAL_REGISTRY_PATH_YOLO)

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
                    )


img_size = (150,150)

@app.post("/predict")
async def Scan_img(file: UploadFile = File(...)) -> dict:
    # image processing
    img = Image.open(io.BytesIO(await file.read()))

    if len(np.array(img).shape) == 2:  # If image is grayscale
        img = img.convert("RGB")



    # Convert the pic in array and process it
    image_array = np.asarray(img)
    image_processed = make_square_with_padding(image_array, img_size)
    image_processed = np.expand_dims(image_processed, axis=0)


    # models prediction

    y_pred = test_demo.predict(image_processed,model_binary,model_cats)


    # model binary
    y_pred_binary = y_pred[0][0]
    if y_pred_binary<0.25:
        y_bin = False
    else:
        y_bin = True

    recall = round(float(y_pred_binary), 3)

    # model VGG
    y_pred_cats = y_pred[1]
    cat = pd.DataFrame({
    'Type': ["glioma", 'meningioma', 'pituitary'],
    'Proba': y_pred_cats[0]
})
    cat_max = cat.loc[cat['Proba'].idxmax()]
    acc = round(float(cat_max['Proba']), 3)

    return {"tumor" : y_bin, # -> Bool
            "recall" : recall, # -> Float
            "tumor_type" : cat_max['Type'], # -> label class
            "precision": acc # -> Float
            }
@app.post("/predict-yolo")
async def Scan_img(file: UploadFile = File(...)) -> dict:
    # image processing
    img = Image.open(io.BytesIO(await file.read()))

    if len(np.array(img).shape) == 2:  # If image is grayscale
        img = img.convert("RGB")

    # Convert the pic in array and process it
    image_array = np.asarray(img)
    image_processed = make_square_with_padding(image_array, img_size)
    image_processed = np.expand_dims(image_processed, axis=0)


    # models prediction

    y_pred = test_demo.predict(image_processed,app.state.model_binary,app.state.model_cats)


    # model binary
    y_pred_binary = y_pred[0][0]
    if y_pred_binary<0.25:
        y_bin = False
    else:
        y_bin = True

    recall = round(float(y_pred_binary), 3)

    # model VGG
    y_pred_cats = y_pred[1]
    cat = pd.DataFrame({
    'Type': ["glioma", 'meningioma', 'pituitary'],
    'Proba': y_pred_cats[0]
})
    cat_max = cat.loc[cat['Proba'].idxmax()]
    acc = round(float(cat_max['Proba']), 3)

    # Yolo model
    img_source_path = 'api/img_source_yolo/tmp_scan_img_for_yolo.jpg' # Path where to stock the pic for yolo
    img.save(img_source_path) # Save locally the pic for yolo

    img_pred_yolo = test_demo.predict_yolo(img_source_path,app.state.model_yolo)
    _, buffer = cv2.imencode('.jpg', img_pred_yolo)
    img_pred_yolo = base64.b64encode(buffer).decode('utf-8')

    return {"tumor" : y_bin, # -> Bool
            "recall" : recall, # -> Float
            "tumor_type" : cat_max['Type'], # -> label class
            "precision": acc, # -> Float
            "img_pred_yolo" : img_pred_yolo
            }

@app.get("/")
def root():
    return dict(api_status="API up")
