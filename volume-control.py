import cv2
import mediapipe as mp
import numpy as np
import pickle
from ctypes import cast, POINTER
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import comtypes

# Load model
data = pickle.load(open("model.pkl", "rb"))
model = data["model"]
encoder = data["encoder"]

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Volume API (WORKS with 20230407)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    comtypes.CLSCTX_ALL,
    None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol = volume.GetVolumeRange()[:2]

def set_volume(percent):
    level = np.interp(percent, [0, 100], [minVol, maxVol])
    volume.SetMasterVolumeLevel(level, None)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        row = []
        for lm in hand.landmark:
            row.extend([lm.x, lm.y, lm.z])

        pred = model.predict([row])[0]
        gesture = encoder.inverse_transform([pred])[0]

        if gesture == "ThumbsUp":
            set_volume(100)
        elif gesture == "ThumbsDown":
            set_volume(10)

        cv2.putText(frame, gesture, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 3)

    cv2.imshow("Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
