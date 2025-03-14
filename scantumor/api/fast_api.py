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
