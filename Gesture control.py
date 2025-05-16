import cv2
import os
import numpy as np
from HandTracker import HandDetector
from deepface import DeepFace
from dottedline import drawrect

# --- Constants & Setup ---
width, height = 1280, 720
frames_folder = r"D:\EJUST\Senior year\Spring\Computer Vision\Computer Vision Project-1 (2)\Computer Vision Project-1\Computer Vision Project\Hand-Gesture-Presentation-System-master\Hand-Gesture-Presentation-System-master\Images"
AUTHORIZED_DIR = r"D:\EJUST\Senior year\Spring\Computer Vision\Computer Vision Project-1 (2)\Computer Vision Project-1\Computer Vision Project\authorized_users"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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
presenter_name = "Unknown"
frame_count = 0

# --- Load slides ---
path_imgs = sorted(os.listdir(frames_folder), key=len)

# --- Camera and Hand Detector ---
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
detector = HandDetector(detectionCon=0.8, maxHands=1)


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


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    pathFullImage = os.path.join(frames_folder, path_imgs[slide_num])
    slide_current = cv2.imread(pathFullImage)
    slide_current = cv2.resize(slide_current, (1280, 720))

    # --- Identify presenter every 60 frames (~2s) ---
    if frame_count % 60 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            presenter_name = identify_user(face_crop)
            break  # Only use the first detected face

    frame_count += 1

    # --- Detect hands and gestures ---
    hands, frame = detector.findHands(frame)
    drawrect(frame, (width, 0), (ge_thresh_x, ge_thresh_y), (0, 255, 0), 5, 'dotted')
    gesture_name = ""

    if hands and not gest_done:
        hand = hands[0]
        cx, cy = hand["center"]
        lm_list = hand["lmList"]
        fingers = detector.fingersUp(hand)

        # Interpolate fingertip
        x_val = int(np.interp(lm_list[8][0], [width // 2, width], [0, width]))
        y_val = int(np.interp(lm_list[8][1], [150, height - 150], [0, height]))
        index_fing = x_val, y_val

        if cy < ge_thresh_y and cx > ge_thresh_x:
            annot_start = False

            if fingers == [1, 0, 0, 0, 0]:
                gesture_name = "Previous Slide"
                if slide_num > 0:
                    gest_done = True
                    slide_num -= 1
                    annotations = [[]]
                    annot_num = 0

            elif fingers == [0, 0, 0, 0, 1]:
                gesture_name = "Next Slide"
                if slide_num < len(path_imgs) - 1:
                    gest_done = True
                    slide_num += 1
                    annotations = [[]]
                    annot_num = 0

            elif fingers == [0, 0, 0, 0, 0]:
                gesture_name = "First Slide"
                slide_num = 0
                annotations = [[]]
                annot_num = 0

            elif fingers == [1, 1, 1, 1, 1]:
                gesture_name = "Last Slide"
                slide_num = len(path_imgs) - 1
                annotations = [[]]
                annot_num = 0

        # Pointer gesture
        if fingers == [0, 1, 1, 0, 0]:
            gesture_name = "Pointer"
            cv2.circle(slide_current, index_fing, 4, (0, 0, 255), cv2.FILLED)
            annot_start = False

        # Pen gesture
        elif fingers == [0, 1, 0, 0, 0]:
            gesture_name = "Pen"
            if not annot_start:
                annot_start = True
                annot_num += 1
                annotations.append([])
            annotations[annot_num].append(index_fing)
            cv2.circle(slide_current, index_fing, 4, (0, 0, 255), cv2.FILLED)
        else:
            annot_start = False

        # Erase gesture
        if fingers == [0, 1, 1, 1, 0]:
            gesture_name = "Erase"
            if annotations:
                annot_start = False
                if annot_num >= 0:
                    annotations.pop(-1)
                    annot_num -= 1
                    gest_done = True

    else:
        annot_start = False

    # --- Gesture delay handling ---
    if gest_done:
        gest_counter += 1
        if gest_counter > delay:
            gest_counter = 0
            gest_done = False

    # --- Draw annotations ---
    for annotation in annotations:
        for j in range(1, len(annotation)):
            cv2.line(slide_current, annotation[j - 1], annotation[j], (0, 0, 255), 6)

    # --- Add webcam preview in slide ---
    img_small = cv2.resize(frame, (ws, hs))
    h, w, _ = slide_current.shape
    slide_current[h - hs:h, w - ws:w] = img_small

    # --- Display gesture + presenter name ---
    display_text = f'Gesture: {gesture_name} | Presenter: {presenter_name}'
    cv2.putText(slide_current, display_text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # --- Show Slide ---
    cv2.imshow("Slides", slide_current)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
