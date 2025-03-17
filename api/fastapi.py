import io
import numpy as np
from PIL import Image
from ml_logic.registry import load_model
from ml_logic.preprocessor import make_square_with_padding
from ml_logic.params import *

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

def transform_image(IMG: bytes) -> np.array:
    image_string = open(IMG, 'rb').read()
    img = Image.open(io.BytesIO(image_string))
    arr = np.asarray(img)
    return arr


img_size = (224,224)

@app.post("/predict")
async def Scan_img(file: UploadFile = File(...)) -> dict:
    # image processing
    img = Image.open(io.BytesIO(await file.read()))
    image_array = np.asarray(img)
    image_processed = make_square_with_padding(image_array,(img_size))

    # model binary prediction --> A FINALISER
    model_binary = app.state.model.binary
    y_pred_binary = model_binary.predict(image_processed)
    #tumor = y_pred_binary[0]
    #recall = y_pred_binary[1]

    # VGG model prediction -> A FINALISER
    model_cats = app.state.model.cats
    y_pred_cats = model_cats.predict(image_processed)
    #tumor_type = y_pred_cats[0]
    #precision =  y_pred_cats[1]



    tumor = True
    recall = 0.98
    tumor_type = 'glioma'
    precision = 0.97



    return {"tumor" : tumor,
            "recall" : recall,
            "tumor_type" : tumor_type,
            "precision": precision}



@app.get("/")
def root():
    return dict(greeting="JC")
