import pickle
from ml_logic.params import *


def save_model(model, path):

    pickle.dump(model, open(path, 'wb'))
    print("✅ Model saved locally")


def load_model(path):

    model = pickle.load(open(path, "rb"))
    print("✅ Model loaded from local disk")

    return model
