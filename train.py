from ultralytics import YOLO

if __name__ == '__main__':
    # Load the model
    #model = YOLO("yolov8n.pt") 
    model = YOLO("E:/Codes/runs/detect/train-4/weights/last.pt")

    # Run the training
    #results = model.train(data="data.yaml", epochs=50, imgsz=640, device=0)
    results = model.train(resume=True, workers=2)
