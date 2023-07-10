from math import e
from ultralytics import YOLO

model = YOLO('yolov8x.pt')
model.train(data='/home/initial/workspace/kikaichino/data.yaml',epochs=3)
