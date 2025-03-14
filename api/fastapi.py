import io
import numpy as np
from PIL import Image
from ml_logic.registry import load_model
from ml_logic.preprocessor import make_square_with_padding
from ml_logic.params import *

from fastapi import FastAPI
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

def transform_image(IMG: bytes) -> np.array:
    image_string = open(IMG, 'rb').read()
    img = Image.open(io.BytesIO(image_string))
    arr = np.asarray(img)
    return arr


img_size = (224,224)

@app.post("/predict")
def Scan_img(IMG: bytes) -> dict:
    # image processing
    image_string = open(IMG, 'rb').read()
    img = Image.open(io.BytesIO(image_string))
    X_arr = np.asarray(img)
    X_processed = make_square_with_padding(X_arr,(img_size))

    # model binary prediction
    model_binary = app.state.model.binary
    y_pred_binary = model_binary.predict(X_processed)
    tumor = y_pred_binary[0]
    recall = y_pred_binary[1]

    # VGG model prediction
    model_cats = app.state.model.cats
    y_pred_cats = model_cats.predict(X_processed)
    tumor_type = y_pred_cats[0]
    precision =  y_pred_cats[1]

    return {"tumor" : tumor,
            "recall" : recall,
            "tumor_type" : tumor_type,
            "precision": precision}



@app.get("/")
def root():
    return dict(greeting="JC")
