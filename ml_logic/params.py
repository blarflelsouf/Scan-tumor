import os

LOCAL_DATA_PATH = None
LOCAL_REGISTRY_PATH_BINARY =  'ml_logic/model_pickles/binary_model.pkl'
LOCAL_REGISTRY_PATH_CLASS = 'ml_logic/model_pickles/vgg_model.pkl'
LOCAL_REGISTRY_PATH_YOLO = 'ml_logic/model_pickles/yolo_model.pkl'



GCP_REGION = os.environ.get("GCP_REGION")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
