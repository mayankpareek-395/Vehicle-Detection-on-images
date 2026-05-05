from ultralytics import YOLO

if __name__ == '__main__':
    # Load your model
    #model = YOLO("yolov8n.pt") 
    model = YOLO("E:/Codes/runs/detect/train-4/weights/last.pt")

    # Run the training
    results = model.train(data="data.yaml", epochs=50, imgsz=640, device=0)

    # The training was interrupted after 40 epoches due to RAM overload, the mAP50 (mean average precision) was 0.682.
    # At Epoch 40, YOLOv8 switches off "Mosaic Augmentation" (it calls this "Closing dataloader mosaic"). 
    # This is a standard strategy to fine-tune the model on realistic images for the last 10 epochs.
    # However, because we are using 8 workers and our dataset is fairly large (~6,000 images),
    # Windows ran out of RAM trying to "re-serialize" all that data to the background workers for this new phase.
    
    # ReRun the training
    #results = model.train(data="data.yaml", epochs=50, imgsz=640, device=0)
    results = model.train(resume=True, workers=2)