import cv2
import numpy as np
import pickle

# Загрузка моделей
with open('face_model.pkl', 'rb') as file:
    face_model = pickle.load(file)

with open('emotion_model.pkl', 'rb') as file:
    emotion_model = pickle.load(file)

emotions = {
    0: 'angry',
    1: 'disgusted',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised'
}


def detect_face(frame):
    """
    Предсказание координат лица
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h_original, w_original = gray.shape

    img = cv2.resize(gray, (150, 150))
    img = img.astype(np.float32) / 255.0
    img = img.reshape(1, 150, 150, 1)

    bbox = face_model.predict(img)

    x1, y1, x2, y2 = bbox[0].astype(int)

    # Перевод координат обратно к размеру кадра
    x_scale = w_original / 150
    y_scale = h_original / 150

    x1 = int(x1 * x_scale)
    x2 = int(x2 * x_scale)
    y1 = int(y1 * y_scale)
    y2 = int(y2 * y_scale)

    return x1, y1, x2, y2


def predict_emotion(face_crop):
    """
    Предсказание эмоции
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (48, 48))
    gray = gray.astype(np.float32) / 255.0

    gray = gray.reshape(1, 48, 48, 1)

    prediction = emotion_model.predict(gray)

    emotion_idx = np.argmax(prediction)

    return emotions[emotion_idx]


# Запуск камеры
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    try:
        x1, y1, x2, y2 = detect_face(frame)

        # Защита от выхода за границы изображения
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        face = frame[y1:y2, x1:x2]

        if face.size > 0:
            emotion = predict_emotion(face)

            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 0),
                          2)

            cv2.putText(frame,
                        emotion,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2)

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Emotion Recognition", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()