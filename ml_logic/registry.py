import pickle
from ml_logic.params import *


def save_model(model, path) -> None:
   pickle.dump(model, open(path, 'wb'))
   print("✅ Model saved locally")



def load_model(path):
    load_model = pickle.load(open(path, 'rb'))
    print("✅ Model loaded from local disk")
    return load_model
