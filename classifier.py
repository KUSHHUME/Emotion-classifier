import cv2
import torch
import torch.nn as nn
import numpy as np

EMOTIONS = ['angry','disgust','fear','happy','sad','surprise','neutral']

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class EmotionClassifier:
    def __init__(self, model_path="models/emotion_model.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionCNN(num_classes=len(EMOTIONS)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def preprocess(self, face_bgr):
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (48,48))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)  # channel
        face = np.expand_dims(face, axis=0)  # batch
        return torch.tensor(face, dtype=torch.float32).to(self.device)

    def predict(self, face_bgr):
        x = self.preprocess(face_bgr)
        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)
            prob, pred = torch.max(probs, 1)
            return EMOTIONS[pred.item()], prob.item()
