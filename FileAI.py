Открыть в Google Colab
!pip install ultralytics opendatasets -q

import opendatasets as od
from ultralytics import YOLO
from google.colab import files

od.download("https://www.kaggle.com/datasets/ultralytics/taco")

with open('/content/taco/data.yaml', 'w') as f:
    f.write("""
path: /content/taco
train: train/images
val: valid/images
nc: 60
names: ['garbage']
""")

model = YOLO('yolov8n.pt')
model.train(data='/content/taco/data.yaml', epochs=30, imgsz=640, batch=16)

files.download('/content/garbage_model/taco_detector/weights/best.pt')