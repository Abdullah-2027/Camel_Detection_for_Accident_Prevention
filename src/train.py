"""Train YOLOv8 on camel dataset."""

from ultralytics import YOLO
from multiprocessing import freeze_support

def camel_detection():
    
# Load pretrained YOLOv8 nano model
    model_detect = YOLO("yolov8n.pt")

# Train on data.yaml with fixed settings
    model_detect.train(
    data="C:\\University\\Computer vision project\\camel.yaml",
    epochs=50,
    imgsz=960,
    batch=8,
    device=0,      # GPU
    workers=2,      # you can increase later
    lr0=0.01        # start with a higher learning rate
    )

def road_segmentation():
    model = YOLO("yolo26n-seg.pt")

    model.train(
        data="C:\\University\\Computer vision project\\road.yaml",
        epochs=50,
        imgsz=960,
        batch=8,
        device=0,      # GPU
        workers=2,      # you can increase later
        lr0=0.03        # start with a higher learning rate
    )
    
    model.predict(
        source="C:\\University\\Computer vision project\\dataset-roads\\images\\val",
        save=True,
        conf=0.25,
        imgsz=960
        )
    
    model.predict(
        source="C:\\University\\Computer vision project\\testing_dataset",
        save=True,
        conf=0.25,
        imgsz=960
        )

if __name__ == "__main__":
    freeze_support()
    # road_segmentation()
    camel_detection()
