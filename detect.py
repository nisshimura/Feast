from ultralytics import YOLO
from getLabel import get_label
import os
import shutil
import datetime
def predict(img_path):
    from roboflow import Roboflow
    rf = Roboflow(api_key="SvoHMJJPTP96xfhtZcMk")
    project = rf.workspace().project("aicook-lcv4d")
    model = project.version(2).model

    out_dir = "."
    out_name = "predict"
    detection_result = model.predict(img_path, confidence=40, overlap=30).json()

    # Extract the 'class' from each prediction
    detected_classes = [prediction['class'] for prediction in detection_result['predictions']]
    # Remove duplicate classes
    labels = list(set(detected_classes))
    return labels

if __name__ == "__main__":
    img_path = "dataset/test/images/DSC_5941_JPG_jpg.rf.7f34ef03affd2f952f6519e8506d8cdc.jpg"
    labels = predict(img_path)
    print(labels)
