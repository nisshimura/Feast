from ultralytics import YOLO
from getLabel import get_label
import os
import shutil
import datetime
def predict(img_path):
    model = YOLO('./weight/best.pt')  # pretrained YOLOv8n model

    out_dir = "."
    out_name = "predict"
    model.predict(img_path,project=out_dir,name=out_name,save=True, conf=0.5,save_txt=True)

    filename_with_ext = os.path.basename(img_path)
    filename, _ = os.path.splitext(filename_with_ext)

    labels = get_label(os.path.join(out_dir,out_name,"labels",f"{filename}.txt"))
    #directoryをコピー
    shutil.copytree(os.path.join(out_dir,out_name),os.path.join(out_dir,"output",str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))))
    shutil.rmtree(os.path.join(out_dir,out_name))
    return labels

if __name__ == "__main__":
    img_path = "dataset/test/images/DSC_5941_JPG_jpg.rf.7f34ef03affd2f952f6519e8506d8cdc.jpg"
    labels = predict(img_path)
    print(labels)
