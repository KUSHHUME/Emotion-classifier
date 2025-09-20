import cv2
from detector import FaceDetector
from classifier import EmotionClassifier

def run_demo(cam=0):
    detector = FaceDetector("models/yolov8n-face.pt")
    clf = EmotionClassifier(model_path="models/emotion_model.pth")

    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        for (x1,y1,x2,y2) in faces:
            face = frame[y1:y2, x1:x2]
            if face.size == 0: continue
            label, prob = clf.predict(face)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {prob:.2f}", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("YOLO + Emotion Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord("q")]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()
