from   tensorflow import keras
import os
import glob
import time
import pickle
from ml_logic.params import *
from colorama import Fore, Style


timestamp = time.strftime("%Y%m%d-%H%M%S")

# path -> CHECK PARAMETERS

def save_model(params: dict, metrics: dict, path) -> None:

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

    print("✅ Results saved locally")




def load_model(path) -> keras.Model :
    print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    local_model_directory = os.path.join(path, "models")
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    latest_model = keras.models.load_model(most_recent_model_path_on_disk)

    print("✅ Model loaded from local disk")

    return latest_model
