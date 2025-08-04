import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time

st.set_page_config(page_title="Hand Gesture Detection", layout="wide")
st.title("Real-Time Hand Gesture Detection & Gesture Recognition")

@st.cache_resource
def get_hand_module():
    mp_hands = mp.solutions.hands
    return mp_hands, mp_hands.Hands(max_num_hands=1), mp.solutions.drawing_utils

mp_hands, hands, mp_draw = get_hand_module()

run = st.checkbox('Run Hand Detection', value=False)
FRAME_WINDOW = st.empty()
gesture_label = st.empty()
gesture_description = st.empty()

GESTURE_DESCRIPTIONS = {
    # ... (same descriptions as in your previous code, omitted for brevity)
    "👍 Thumbs Up": "Thumb raised upwards, all other fingers folded. Means 'Like' or 'Approve'.",
    # (add all other gesture descriptions from your previous code here)
}

def finger_is_extended(hand_landmarks, tip_id, pip_id):
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

def detect_gesture(hand_landmarks):
    lm = hand_landmarks.landmark
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = lm[mp_hands.HandLandmark.THUMB_IP]
    index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = lm[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = lm[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = lm[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = lm[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = lm[mp_hands.HandLandmark.PINKY_PIP]
    wrist = lm[mp_hands.HandLandmark.WRIST]

    index_up = finger_is_extended(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
    middle_up = finger_is_extended(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
    ring_up = finger_is_extended(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
    pinky_up = finger_is_extended(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    thumb_up = thumb_tip.y < wrist.y and abs(thumb_tip.x - wrist.x) < 0.12
    thumb_down = thumb_tip.y > wrist.y and abs(thumb_tip.x - wrist.x) < 0.12

    fingers_up = [index_up, middle_up, ring_up, pinky_up]

    # Add the full gesture detection logic here (identical to previous code)
    # ... (Counting fingers, rock, spock, call me, etc.)

    # Example:
    if sum(fingers_up) == 1 and index_up:
        return "☝️ 1 Finger Up"
    if sum(fingers_up) == 2 and index_up and middle_up:
        return "✌️ 2 Fingers Up"
    if sum(fingers_up) == 3 and index_up and middle_up and ring_up:
        return "🤟 3 Fingers Up"
    if sum(fingers_up) == 4 and all(fingers_up):
        return "🖖 4 Fingers Up"
    if all(fingers_up):
        return "🖐️ 5 Fingers Up"
    if thumb_up and not any(fingers_up):
        return "👍 Thumbs Up"
    if thumb_down and not any(fingers_up):
        return "👎 Thumbs Down"
    if index_up and pinky_up and not middle_up and not ring_up:
        return "🤘 Rock Sign"
    if (index_up and middle_up and ring_up and pinky_up and
        abs(index_tip.x - middle_tip.x) < 0.03 and
        abs(ring_tip.x - pinky_tip.x) < 0.03 and
        abs(middle_tip.x - ring_tip.x) > 0.05):
        return "🖖 Spock Sign"
    if (thumb_up or (thumb_tip.y < index_tip.y and thumb_tip.x < wrist.x)) and pinky_up and not (index_up or middle_up or ring_up):
        return "🤙 Call Me"
    # OK sign
    index_thumb_dist = np.hypot((thumb_tip.x - index_tip.x), (thumb_tip.y - index_tip.y))
    ok_sign = index_thumb_dist < 0.06 and not (middle_up or ring_up or pinky_up)
    if ok_sign:
        return "👌 OK Sign"
    if not any([thumb_up, thumb_down, index_up, middle_up, ring_up, pinky_up]):
        return "✊ Fist"
    if index_up and middle_up and not ring_up and not pinky_up and not thumb_up and not thumb_down:
        return "✌️ Peace / Victory"

    return "No recognized gesture"

if run:
    camera = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                st.warning("Could not access webcam.")
                break
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            gesture = "No hand detected"
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                    gesture = detect_gesture(handLms)

            FRAME_WINDOW.image(frame, channels="BGR")
            gesture_label.markdown(f"**Gesture:** {gesture}")
            gesture_description.info(GESTURE_DESCRIPTIONS.get(
                gesture, GESTURE_DESCRIPTIONS["No recognized gesture"]))
            time.sleep(0.05)

            if not st.session_state.get("run_hand_detection", True):
                break
    finally:
        camera.release()
        st.session_state["run_hand_detection"] = False
else:
    FRAME_WINDOW.info('👆 Check "Run Hand Detection" to start.')
    gesture_label.empty()
    gesture_description.empty()
    st.session_state["run_hand_detection"] = False
