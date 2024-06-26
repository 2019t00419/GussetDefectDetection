from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)
    results = model.train(data="F:/UOC/Research/Programs/Test program for edge detection/BalanceOutDetection/data", epochs=1, imgsz=64)

if __name__ == "__main__":
    train_model()
