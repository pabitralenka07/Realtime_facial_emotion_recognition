import os
import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input

# 1. Load trained emotion detection model
if os.path.exists("facialemotionmodel.keras"):
    model = load_model("facialemotionmodel.keras")
elif os.path.exists("facialemotionmodel.h5"):
    model = Sequential([
        Input(shape=(48, 48, 1)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        Conv2D(256, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        Conv2D(512, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        Conv2D(512, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(7, activation='softmax')
    ])
    model.load_weights("facialemotionmodel.h5")
else:
    raise FileNotFoundError("Could not find facialemotionmodel.keras or facialemotionmodel.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image, dtype=np.float32)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

print("Starting webcam... Press 'q' or 'ESC' on the video window to quit.")

while True:
    ret, im = webcam.read()
    if not ret or im is None:
        print("Camera feed ended or webcam not accessible.")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img, verbose=0)
            prediction_label = labels[pred.argmax()]
            cv2.putText(im, f'{prediction_label}', (p, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2)

        cv2.imshow("Facial Emotion Recognition - Press 'q' to Quit", im)
        key = cv2.waitKey(27) & 0xFF
        if key == ord('q') or key == 27:
            break
    except cv2.error as e:
        print(f"OpenCV error: {e}")

webcam.release()
cv2.destroyAllWindows()