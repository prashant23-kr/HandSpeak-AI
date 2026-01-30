import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from threading import Thread
import pyttsx3
import time
import math
import numpy as np
from deepface import DeepFace
import csv
import os

LANGUAGE_MODE = "english"  
LOG_FILE = "gesture_emotion_log.csv"

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    voices = engine.getProperty('voices')

    if LANGUAGE_MODE == "hindi":
        for v in voices:
            if "hindi" in v.name.lower() or "hi" in v.id.lower():
                engine.setProperty('voice', v.id)
                break
    else:
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id)

    engine.say(text)
    engine.runAndWait()

# Logging setup
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Gesture", "Emotion"])

def log_interaction(gesture, emotion):
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), gesture, emotion])

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # allow 2 hands
classifier = Classifier(
    "C:/Users/ASUS/OneDrive/Desktop/converted_keras (4)/keras_model.h5",
    "C:/Users/ASUS/OneDrive/Desktop/converted_keras (4)/labels.txt"
)

offset = 20
imgSize = 300
gesture_labels = ["Hello", "Iloveyou", "No", "Please", "Thank you", "Yes"]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

speech_interval = 1.0
last_gesture_speech_time = 0
last_spoken_gesture = None

mood_interval = 2.5
last_mood_speech_time = 0
last_spoken_emotion = None

while True:
    success, img = cap.read()
    if not success:
        print("Cannot read frame from camera")
        break

    imgOutput = img.copy()
    all_detected_gestures = []
    all_detected_emotions = []

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(imgOutput, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        face_roi = img[fy:fy + fh, fx:fx + fw]

        if face_roi.size != 0:
            try:
                face_resized = cv2.resize(face_roi, (224, 224))
                res = DeepFace.analyze(face_resized, actions=['emotion'], enforce_detection=False)

                mood_label = res[0]['dominant_emotion'] if isinstance(res, list) else res.get("dominant_emotion")
                all_detected_emotions.append(mood_label)

                cv2.putText(imgOutput, mood_label, (fx, fy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                now = time.time()
                if (now - last_mood_speech_time) > mood_interval and mood_label != last_spoken_emotion:
                    last_mood_speech_time = now
                    last_spoken_emotion = mood_label
                    Thread(target=speak, args=(mood_label,)).start()

            except Exception as e:
                print("Emotion detection failed:", e)
   
    hands, _ = detector.findHands(img)
    for hand in hands:
        x, y, w, h = hand['bbox']
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hGap + hCal, :] = imgResize
            except:
                imgResize = None

            if imgResize is not None:
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                detected_label = gesture_labels[index]
                all_detected_gestures.append(detected_label)

                cv2.putText(imgOutput, f"{detected_label} {int(prediction[index]*100)}%",
                            (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (0, 255, 0), 3)

                now = time.time()
                if (now - last_gesture_speech_time) > speech_interval and detected_label != last_spoken_gesture:
                    last_gesture_speech_time = now
                    last_spoken_gesture = detected_label
                    Thread(target=speak, args=(detected_label,)).start()

    combined_text = ""
    if all_detected_gestures:
        combined_text += ", ".join(all_detected_gestures)
    if all_detected_emotions:
        combined_text += " (" + ", ".join(all_detected_emotions) + ")"

    if combined_text:
        cv2.putText(imgOutput, combined_text, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        log_interaction(
            ", ".join(all_detected_gestures) if all_detected_gestures else "-",
            ", ".join(all_detected_emotions) if all_detected_emotions else "-"
        )

    # Show
    cv2.imshow("Sign+Emotion System", imgOutput)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
