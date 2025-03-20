from roboflow import Roboflow
import supervision as sv
import cv2
from ml_logic import registry, params
import os

def import_yolo_model():
    api_key = os.environ.get("API_KEY_ROBOFLOW")
    print(api_key)
    print(f"api_key retrieved")
    rf = Roboflow(api_key)
    project = rf.workspace().project("tomour_detection")
    model2 = project.version(1).model
    print(f"model retrieved")
    yolo_model_saved = registry.save_model(
                                    model = model2,
                                    path = params.LOCAL_REGISTRY_PATH_YOLO)
    print(f"model saved")
    return None

def predict(X_image_path,yolo_model):
    result = yolo_model.predict(X_image_path, confidence=40).json()
    detections = sv.Detections.from_inference(result)
    mask_annotator = sv.MaskAnnotator()
    mask_annotator2 = sv.BoxAnnotator()
    image = cv2.imread(X_image_path)
    annotated_image = mask_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = mask_annotator2.annotate(
        scene=image, detections=detections)
    image=annotated_image

    return image


if __name__ == "__main__":
    import_yolo_model()
