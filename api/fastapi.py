import io
import numpy as np
import pandas as pd
from PIL import Image
from ml_logic.registry import load_model
from ml_logic.preprocessor import make_square_with_padding
from ml_logic.params import *
from interface import main,test_demo

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()
app.state.model.binary = load_model(LOCAL_REGISTRY_PATH_BINARY)
app.state.model.cats = load_model(LOCAL_REGISTRY_PATH_CLASS)

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
                    )


img_size = (224,224)

@app.post("/predict")
async def Scan_img(file: UploadFile = File(...)) -> dict:
    # image processing
    img = Image.open(io.BytesIO(await file.read()))
    image_array = np.asarray(img)
    image_processed = make_square_with_padding(image_array,(img_size))
    image_processed = np.expand_dims(image_processed, axis=0)


    # models prediction
    model_binary = app.state.model.binary
    model_cats = app.state.model.cats
    y_pred = test_demo.predict(image_processed,model_binary,model_cats)


    # model binary
    y_pred_binary = y_pred[0][0]
    recall = main.histo_bin_test

    # model VGG
    y_pred_cats = y_pred[1][0]
    cat = pd.DataFrame({
    'Type': ["notumor","glioma", 'meningioma', 'pituitary'],
    'Proba': y_pred_cats[0]
})
    cat = cat.max()

    return {"tumor" : y_pred_binary, # -> Bool
            "recall" : recall, # -> Float
            "tumor_type" : cat['Type'], # -> label class
            "precision": cat['Proba']} # -> Float



@app.get("/")
def root():
    return dict(greeting="JC")
