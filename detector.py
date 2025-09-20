from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path="models/yolov8n-face.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        faces = []
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1,y1,x2,y2))
        return faces
