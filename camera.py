import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import pandas as pd
import time
from collections import Counter

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

emotion_model = tf.keras.models.load_model(
    "fer_efficientnet_b3_70pct.keras",
    compile=False
)

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

music_dist = {
    0: "songs/angry.csv",
    1: "songs/disgusted.csv",
    2: "songs/fearful.csv",
    3: "songs/happy.csv",
    4: "songs/neutral.csv",
    5: "songs/sad.csv",
    6: "songs/surprised.csv"
}

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        time.sleep(1.0)
        self.df1 = None

        self.emotion_buffer = []
        self.locked_emotion = None
        self.lock_start_time = None
        self.LOCK_DURATION = 10        # Adjust to keep an emotion for longer time
        self.BUFFER_TIME = 5           # Adjust to provide more time for detection
        self.last_buffer_time = time.time()

    def get_frame(self):
        ret, image = self.cap.read()
        if not ret:
            return None, None

        image = cv2.resize(image, (600, 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        current_time = time.time()

        for (x, y, w, h) in faces:
            roi = image[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            roi = cv2.resize(roi, (224, 224))
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = roi.astype("float32")
            roi = preprocess_input(roi)
            roi = np.expand_dims(roi, axis=0)

            pred = emotion_model.predict(roi, verbose=0)
            idx = int(np.argmax(pred))

            if self.locked_emotion is None:
                self.emotion_buffer.append(idx)

                if current_time - self.last_buffer_time >= self.BUFFER_TIME:
                    dominant = Counter(self.emotion_buffer).most_common(1)[0][0]
                    self.locked_emotion = dominant
                    self.lock_start_time = current_time
                    self.emotion_buffer.clear()
                    self.last_buffer_time = current_time
            else:
                if current_time - self.lock_start_time >= self.LOCK_DURATION:
                    self.locked_emotion = None
                    self.emotion_buffer.clear()
                    self.last_buffer_time = current_time

            display_emotion = self.locked_emotion if self.locked_emotion is not None else idx
            confidence = float(np.max(pred) * 100)

            cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(
                image,
                f"{emotion_dict[display_emotion]}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255,255,255),
                2
            )

        if self.locked_emotion is not None:
            self.df1 = pd.read_csv(music_dist[self.locked_emotion])[
                ['Name', 'Album', 'Artist']
            ].head(15)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes(), self.df1
