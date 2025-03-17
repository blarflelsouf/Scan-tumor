from   tensorflow import keras
import os
import glob
import time
import pickle
from ml_logic.params import *


timestamp = time.strftime("%Y%m%d-%H%M%S")

# path -> CHECK PARAMETERS

def save_model(model, path) -> None:

    if model is not None:
        model_path = os.path.join(path, "model", timestamp + ".pickle")
        model.save(model_path)
    print("✅ Model saved locally")

    return None


def load_model(path) -> keras.Model :

    local_model_directory = os.path.join(path, "models")
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    latest_model = keras.models.load_model(most_recent_model_path_on_disk)

    print("✅ Model loaded from local disk")

    return latest_model



'''

def save_model(params: dict, metrics: dict, model, path) -> None:

    # Save the parameters locally

    if params is not None:
        params_path = os.path.join(path, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(path, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)


    if model is not None:
        model_path = os.path.join(path, "model", timestamp + ".pickle")
        model.save(model_path)
    print("✅ Results saved locally")


    '''
