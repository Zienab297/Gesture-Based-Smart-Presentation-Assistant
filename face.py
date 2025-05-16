import cv2
import os
from deepface import DeepFace

AUTHORIZED_DIR = r"C:\Users\Sohyla\Downloads\Computer Vision Project-1\Computer Vision Project\authorized_users"  # parent folder
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Identify user by comparing with each person's folder of photos
def identify_user(frame):
    for person_folder in os.listdir(AUTHORIZED_DIR):
        person_path = os.path.join(AUTHORIZED_DIR, person_folder)
        if not os.path.isdir(person_path):
            continue
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            try:
                result = DeepFace.verify(frame, image_path, enforce_detection=False)
                if result['verified']:
                    return person_folder  # Return the folder name as identity
            except:
                continue
    return "Unknown"

# Initialize camera
cap = cv2.VideoCapture(0)
paused = False
pause_counter = 0
pause_duration = 100  # number of frames to pause
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if not paused:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Detect emotion every 30 frames
        if frame_count % 30 == 0:
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant_emotion = analysis[0]['dominant_emotion']
                print("Emotion Detected:", dominant_emotion)

                if dominant_emotion in ['surprise', 'fear', 'disgust', 'confused']:
                    paused = True
                    pause_counter = 0
            except Exception as e:
                print("Emotion detection failed:", e)

        # Draw boxes and names
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            name = identify_user(face_crop)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    else:
        pause_counter += 1
        cv2.putText(frame, "Paused: Confused/Surprised Detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if pause_counter > pause_duration:
            paused = False

    cv2.imshow("Face & Emotion Detection", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
