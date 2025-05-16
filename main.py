import cv2
import os
import numpy as np
from deepface import DeepFace
from HandTracker import HandDetector
from dottedline import drawrect

# ========== FACE & EMOTION DETECTION SETUP ==========
#Configuration
AUTHORIZED_DIR = r"D:\EJUST\Senior year\Spring\Computer Vision\Computer Vision Project\authorized_users"
ACCESS_GRANTED_TIMEOUT = 3  # Seconds to display access message
RECOGNITION_THRESHOLD = 0.5  # Adjust this threshold as needed (0.0-1.0)
EMOTION_CONFIDENCE_THRESHOLD = 0.5  # Only analyze emotions when recognition confidence > 50%

# Initialize face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# Try to load alternative cascade as well
alt_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")


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
                    return person_folder
            except:
                continue
    return "Unknown"

# ========== HAND GESTURE SETUP ==========
width, height = 1280, 720
frames_folder = r"D:\EJUST\Senior year\Spring\Computer Vision\Computer Vision Project\Hand-Gesture-Presentation-System-master\Images"
slide_num = 0
hs, ws = int(120 * 1.2), int(213 * 1.2)
ge_thresh_y = 400
ge_thresh_x = 750
gest_done = False
gest_counter = 0
delay = 15
annotations = [[]]
annot_num = 0
annot_start = False
gesture_name = ""

path_imgs = sorted(os.listdir(frames_folder), key=len)

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detector = HandDetector(detectionCon=0.8, maxHands=1)

paused = False
pause_counter = 0
pause_duration = 100
frame_count = 0
unauthorized = False  # 🔒 Flag for unauthorized person
authorized_user_detected = False

# ========== MAIN LOOP ==========
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    pathFullImage = os.path.join(frames_folder, path_imgs[slide_num])
    slide_current = cv2.imread(pathFullImage)
    slide_current = cv2.resize(slide_current, (1280, 720))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    name = "Unknown"

    if frame_count % 30 == 0:  # Check every 30 frames to reduce load
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            name = identify_user(face_crop)

            # Draw face box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if name == "Unknown":
                unauthorized = True
                authorized_user_detected = False
                break
            else:
                unauthorized = False
                authorized_user_detected = True

        if not unauthorized:
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant_emotion = analysis[0]['dominant_emotion']
                print("Emotion Detected:", dominant_emotion)

                if dominant_emotion in ['surprise', 'fear', 'disgust', 'confused']:
                    paused = True
                    pause_counter = 0
            except Exception as e:
                print("Emotion detection failed:", e)

    # If neither paused by emotion nor unauthorized, allow control
    if not paused and not unauthorized:
        # ==== Gesture Control ====
        hands, frame = detector.findHands(frame)
        drawrect(frame, (width, 0), (ge_thresh_x, ge_thresh_y), (0, 255, 0), 5, 'dotted')

        if hands and not gest_done:
            hand = hands[0]
            cx, cy = hand["center"]
            lm_list = hand["lmList"]
            fingers = detector.fingersUp(hand)

            x_val = int(np.interp(lm_list[8][0], [width//2, width], [0, width]))
            y_val = int(np.interp(lm_list[8][1], [150, height - 150], [0, height]))
            index_fing = x_val, y_val

            if cy < ge_thresh_y and cx > ge_thresh_x:
                annot_start = False

                if fingers == [1, 0, 0, 0, 0]:  # previous
                    gesture_name = "Previous Slide"
                    if slide_num > 0:
                        gest_done = True
                        slide_num -= 1
                        annotations = [[]]
                        annot_num = 0

                elif fingers == [0, 0, 0, 0, 1]:  # next
                    gesture_name = "Next Slide"
                    if slide_num < len(path_imgs) - 1:
                        gest_done = True
                        slide_num += 1
                        annotations = [[]]
                        annot_num = 0

                elif fingers == [0, 0, 0, 0, 0]:  # begin
                    gesture_name = "First Slide"
                    annot_start = False
                    slide_num = 0

                elif fingers == [1, 1, 1, 1, 1]:  # end
                    gesture_name = "Last Slide"
                    annot_start = False
                    slide_num = len(path_imgs) - 1

            elif fingers == [0, 1, 1, 0, 0]:  # pointer
                gesture_name = "Pointer"
                cv2.circle(slide_current, index_fing, 4, (0, 0, 255), cv2.FILLED)
                annot_start = False

            elif fingers == [0, 1, 0, 0, 0]:  # draw
                gesture_name = "Pen"
                if not annot_start:
                    annot_start = True
                    annot_num += 1
                    annotations.append([])
                annotations[annot_num].append(index_fing)
                cv2.circle(slide_current, index_fing, 4, (0, 0, 255), cv2.FILLED)

            elif fingers == [0, 1, 1, 1, 0]:  # erase
                gesture_name = "Erase"
                if annotations:
                    if annot_num >= 0:
                        annotations.pop(-1)
                        annot_num -= 1
                        gest_done = True

        if gest_done:
            gest_counter += 1
            if gest_counter > delay:
                gest_counter = 0
                gest_done = False

        for annotation in annotations:
            for j in range(len(annotation)):
                if j != 0:
                    cv2.line(slide_current, annotation[j - 1], annotation[j], (0, 0, 255), 6)

    elif unauthorized:
        cv2.putText(slide_current, "Access Denied: Unauthorized User", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    elif paused:
        pause_counter += 1
        cv2.putText(slide_current, "Paused: Confused/Surprised Detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if pause_counter > pause_duration:
            paused = False

    # Insert camera view on slide
    img_small = cv2.resize(frame, (ws, hs))
    h, w, _ = slide_current.shape
    slide_current[h-hs:h, w-ws:w] = img_small

    text = f'Gesture: {gesture_name}'
    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
    cv2.putText(slide_current, text, (1280 - text_width - 50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)


    # Show final window
    cv2.imshow("Smart Presentation System", slide_current)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
