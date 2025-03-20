from ml_logic import preprocessor as prepro
from ml_logic import modelVGG as modelvgg
from ml_logic import modelBINARY as modelbin
from ml_logic import modelYOLO
from ml_logic import data_preparation as prepa
import utils
from ml_logic import registry
from ml_logic.params import *
from ml_logic.preprocessor import *
from api import fastapi

import io
from PIL import Image

# Predict function (both binary & vgg)

def predict(X_pred, binary_model_to_load, vgg_model_to_load):
    binary_model = binary_model_to_load
    y_pred_binary= binary_model.predict(X_pred)
    vgg_model = vgg_model_to_load
    y_pred_VGG= vgg_model.predict(X_pred)

    return y_pred_binary,y_pred_VGG

def predict_yolo(X_image_path,yolo_model):
    image_yolo_pred = modelYOLO.predict(X_image_path,yolo_model)

    return image_yolo_pred
