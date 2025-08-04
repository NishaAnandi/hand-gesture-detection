import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def finger_is_extended(lm, tip_id, pip_id):
    # Returns True if the finger is extended (tip is above pip joint)
    return lm[tip_id].y < lm[pip_id].y

def detect_gesture(hand_landmarks):
    lm = hand_landmarks.landmark
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = lm[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = lm[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = lm[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = lm[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = lm[mp_hands.HandLandmark.PINKY_PIP]
    wrist = lm[mp_hands.HandLandmark.WRIST]

    # Determine which fingers are up
    index_up = finger_is_extended(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
    middle_up = finger_is_extended(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
    ring_up = finger_is_extended(lm, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
    pinky_up = finger_is_extended(lm, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    thumb_up = thumb_tip.y < wrist.y and abs(thumb_tip.x - wrist.x) < 0.15

    fingers_up = [index_up, middle_up, ring_up, pinky_up]

    if thumb_up and not any(fingers_up):
        return "Thumbs Up"
    if sum(fingers_up) == 1 and index_up and not middle_up and not ring_up and not pinky_up:
        return "1 Finger Up"
    if sum(fingers_up) == 2 and index_up and middle_up and not ring_up and not pinky_up:
        return "2 Fingers Up"
    if sum(fingers_up) == 3 and index_up and middle_up and ring_up and not pinky_up:
        return "3 Fingers Up"
    if sum(fingers_up) == 4 and all(fingers_up):
        return "4 Fingers Up"
    if index_up and pinky_up and not middle_up and not ring_up:
        return "Rock Sign"
    if thumb_up and pinky_up and not index_up and not middle_up and not ring_up:
        return "Call Me"
    return "Unknown"

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    gesture = "None"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(handLms)

    cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("Hand Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

