from roboflow import Roboflow
import supervision as sv
import cv2


def model_yolo(path):



    rf = Roboflow(api_key="W59EOYGjHqoyOe7PNPwP")
    project = rf.workspace().project("tomour_detection")
    model2 = project.version(1).model


    result = model2.predict(path, confidence=40).json()
    # labels = [item["class"] for item in result["predictions"]]
    detections = sv.Detections.from_inference(result)
    # label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    mask_annotator2 = sv.BoxAnnotator()
    image = cv2.imread(path)
    annotated_image = mask_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = mask_annotator2.annotate(
        scene=image, detections=detections)
    image=annotated_image

    return image
