
from ml_logic.params import *
from ml_logic.preprocessor import *

# Predict function (both binary & vgg)

def predict(X_pred, binary_model_to_load, vgg_model_to_load):
    binary_model = binary_model_to_load
    y_pred_binary= binary_model.predict(X_pred)
    vgg_model = vgg_model_to_load
    y_pred_VGG= vgg_model.predict(X_pred)

    return y_pred_binary,y_pred_VGG
