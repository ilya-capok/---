from ultralytics import YOLO
import opendatasets as od
od.download("https://www.kaggle.com/datasets/ultralytics/taco")
with open('taco/data.yaml', 'w') as f:
    f.write("""
path: taco
train: train/images
val: valid/images
nc: 1
names:
  - garbage
""")
model = YOLO('yolov8n.pt')
model.train(
    data='taco/data.yaml',
    epochs=30,
    imgsz=640,
    batch=16
)
