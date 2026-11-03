# Realtime Facial Emotion Recognition 🎭

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0+-red.svg)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A real-time deep learning pipeline that detects faces and classifies human facial expressions using your webcam feed. Powered by **OpenCV Haar Cascades** for face detection and a custom **Convolutional Neural Network (CNN)** built with **TensorFlow / Keras**.

---

## 📌 Features

- **⚡ Real-Time Inference:** Low-latency video capture and frame-by-frame emotion prediction.
- **🎯 7 Emotion Categories:**
  - 😠 Angry
  - 🤢 Disgust
  - 😨 Fear
  - 😊 Happy
  - 😐 Neutral
  - 😢 Sad
  - 😲 Surprise
- **🧠 Custom CNN Architecture:** 4-stage convolutional feature extractor with dropout regularization for robust classification.
- **🔄 Dual Model Compatibility:** Automatically loads modern `.keras` format or legacy `.h5` model weights.
- **📓 End-to-End Notebook:** Complete Jupyter notebook included for training, evaluation, and data preparation.

---

## 🛠️ Architecture & Pipeline

```text
[ Webcam Feed ] ──> [ OpenCV Haar Cascade ] ──> [ Crop Face ROI (48x48 Grayscale) ]
                                                            │
                                                            ▼
                                                [ Deep CNN Model ]
                                                            │
                                                            ▼
[ Annotated Output Window ] <── [ Draw Bounding Box & Emotion Label ]
```

### CNN Architecture Summary
- **Input:** `48 x 48 x 1` (Grayscale facial crop)
- **Conv Block 1:** Conv2D (128 filters, 3x3, ReLU) + MaxPooling2D (2x2) + Dropout (0.4)
- **Conv Block 2:** Conv2D (256 filters, 3x3, ReLU) + MaxPooling2D (2x2) + Dropout (0.4)
- **Conv Block 3:** Conv2D (512 filters, 3x3, ReLU) + MaxPooling2D (2x2) + Dropout (0.4)
- **Conv Block 4:** Conv2D (512 filters, 3x3, ReLU) + MaxPooling2D (2x2) + Dropout (0.4)
- **Fully Connected:** Flatten + Dense (512, ReLU) + Dropout (0.4) + Dense (256, ReLU) + Dropout (0.3)
- **Output:** Dense (7, Softmax)

---

## 📂 Project Structure

```bash
Realtime_facial_emotion_recognition/
├── .gitignore                    # Ignored files (venv, caches, etc.)
├── facialemotionmodel.h5         # Trained model weights (H5 format)
├── facialemotionmodel.json       # Model architecture JSON
├── facialemotionmodel.keras      # Trained model (Keras format)
├── realtimedetection.py          # Real-time webcam inference script
├── requirements.txt              # Project dependencies
├── trainmodel.ipynb              # Training & dataset exploration notebook
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or 3.11 recommended
- A working webcam

### 2. Clone the Repository
```bash
git clone https://github.com/pabitralenka07/Realtime_facial_emotion_recognition.git
cd Realtime_facial_emotion_recognition
```

### 3. Create & Activate Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate on macOS / Linux
source .venv/bin/activate

# Activate on Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🎮 Running Real-Time Detection

With your virtual environment activated, run:

```bash
python realtimedetection.py
```

### Controls:
- **`q`** or **`ESC`**: Exit the video feed and close the window.

> **Note for macOS Users:** On first run, grant camera permissions if macOS displays an access prompt.

---

## 🏋️ Training the Model

To train the model on the FER-2013 dataset or your own custom dataset:

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook trainmodel.ipynb
   ```
2. Place training and test images in the expected structure:
   ```text
   images/
   ├── train/
   │   ├── angry/
   │   ├── disgust/
   │   ├── fear/
   │   ├── happy/
   │   ├── neutral/
   │   ├── sad/
   │   └── surprise/
   └── test/
       └── ...
   ```
3. Run all cells in `trainmodel.ipynb` to preprocess the data, compile, train, and save the updated weights.

---

## ⚙️ Troubleshooting

<details>
<summary><b>1. Camera not opening or immediate exit</b></summary>

- Verify your webcam is not in use by another application.
- On macOS, check **System Settings > Privacy & Security > Camera** and ensure your Terminal / IDE has permission.
- If using an external webcam, change the index in `realtimedetection.py`:
  ```python
  webcam = cv2.VideoCapture(1)  # Change from 0 to 1 or 2
  ```
</details>

<details>
<summary><b>2. OpenCV / Protobuf error on macOS</b></summary>

If you see errors relating to `libprotobuf` or broken system OpenCV:
- Always run the script using the dedicated virtual environment interpreter:
  ```bash
  ./.venv/bin/python realtimedetection.py
  ```
</details>

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
