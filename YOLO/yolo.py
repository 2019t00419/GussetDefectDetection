from ultralytics import YOLO

def train_model():
    # Path to your data.yaml file
    data_path = '../data.yaml'

    # Initialize the model
    model = YOLO('yolov8n.pt')

    # Train the model
    model.train(data=data_path, epochs=200, imgsz=1000)

if __name__ == '__main__':
    train_model()

