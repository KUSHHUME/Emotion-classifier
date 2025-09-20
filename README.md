````
# YOLO-Emotion-Recognition

This project combines a **YOLOv8** face detector with a custom **Convolutional Neural Network (CNN)** to perform real-time emotion recognition from a live camera feed.

## Features

-   **Real-time Face Detection:** Uses a lightweight YOLOv8 model to accurately detect faces in each frame.
-   **Emotion Classification:** A custom-trained CNN model classifies the detected faces into one of seven emotions: angry, disgust, fear, happy, sad, surprise, and neutral.
-   **Live Demonstration:** The `main.py` script provides a live camera demo, drawing bounding boxes around faces and displaying the predicted emotion and its probability.

---

## Project Structure

-   `main.py`: The primary script to run the real-time emotion recognition demo. It orchestrates the face detection and emotion classification processes.
-   `detector.py`: Contains the `FaceDetector` class, which uses the `ultralytics` library to perform face detection with a pre-trained YOLOv8 model.
-   `classifier.py`: Houses the `EmotionClassifier` and `EmotionCNN` classes. This file handles preprocessing the detected face regions and predicting the emotion using the custom-trained CNN. It also defines the CNN architecture.
-   `train.py`: A script for training the emotion classification model (`EmotionCNN`) on the **FER2013 dataset**. It includes data loading, model training, and saving the final model weights.
-   `requirements.txt`: Lists all the necessary Python packages to run the project.

---

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://your-repository-url.git](https://your-repository-url.git)
    cd YOLO-Emotion-Recognition
    ```

2.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    [cite_start]The required packages include `torch`, `ultralytics`, `opencv-python`, `numpy`, and `scikit-learn`[cite: 1].

3.  **Download Pre-trained Models:**
    This project requires two pre-trained models. Create a `models` directory and place the files inside.
    
    -   **YOLOv8 Face Detection Model:** Download `yolov8n-face.pt` from the official repository or a reliable source and place it in the `models/` directory.
    -   **Emotion Recognition Model:** The `emotion_model.pth` is generated after running the `train.py` script. If you don't want to train the model yourself, you can download a pre-trained version.

    Your `models` directory should look like this:
    ```
    models/
    ├── emotion_model.pth
    └── yolov8n-face.pt
    ```

---

## Usage

### Training the Emotion Model

If you want to train the emotion classifier yourself, you'll need the FER2013 dataset.

1.  **Download FER2013 Dataset:**
    Place the `fer2013.csv` file inside a `datasets/` folder.
    ```
    datasets/
    └── fer2013.csv
    ```

2.  **Run the Training Script:**
    ```bash
    python train.py
    ```
    This script will train the `EmotionCNN` model and save the weights as `emotion_model.pth` in the `models/` directory.

### Running the Demo

To start the real-time emotion recognition demo, execute the main script:
```bash
python main.py
````

This will open a window showing your camera feed with real-time face and emotion detection. Press `Esc` or `Q` to quit.

## Model Details

### Face Detector

  - **Model:** YOLOv8 (specifically `yolov8n-face.pt`)
  - **Framework:** `ultralytics`
  - **Function:** Detects bounding boxes for faces in an image or video frame.

### Emotion Classifier

  - **Architecture:** A simple Convolutional Neural Network (`EmotionCNN`) with two convolutional layers, max-pooling, a fully connected layer, and a dropout layer.
  - **Input:** Grayscale 48x48 pixel images of a face.
  - **Output:** Probabilities for seven emotions (`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`).

<!-- end list -->

```
```
