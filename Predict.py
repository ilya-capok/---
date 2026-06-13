from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model.predict(
    source="test.jpg",
    save=True,
    conf=0.5
)

print(results)