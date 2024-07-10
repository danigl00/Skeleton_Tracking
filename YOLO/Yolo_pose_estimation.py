from ultralytics import YOLO

model = YOLO('YOLO/yolov8l-pose.pt')  # Load model
source = ''

results = model.preditct(source, stream=True)