
import io
import numpy as np
from PIL import Image
from ml_logic import modelvgg16
from ml_logic import preprocessor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# launch the API -> api.fast:app


app = FastAPI()
# app.state.model = load_model()

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






@app.post("/predict")
def Scan_img(IMG: bytes) -> dict:
    image_string = open(IMG, 'rb').read()
    img = Image.open(io.BytesIO(image_string))
    X_arr = np.asarray(img)


    #X_processed = preprocessor.preprocess_features(X_arr) -> TO DO LATER WITH LOUIS FUNCTION

    # app.state.model = load_model()

    # y_pred = model.predict(X_processed) -> FROM LOAD MODEL FUNCTION

    # transform y_pred to the dictionary we want to return

    return {"tumor" : True,
            "recall" : 0.9,
            "tumor_type" : "glioma",
            "precision": 0.8}



@app.get("/")
def root():
    return dict(greeting="JC")
