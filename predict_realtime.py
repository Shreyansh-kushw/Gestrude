"""Predicts the gestures of hand captured from webcam in real time"""

import cv2
import mediapipe as mp
import numpy as np
import pickle

data = pickle.load(open("model.pkl", "rb")) # opening the trained model saved in the pkl file
model = data["model"] # defining the model (neural network)
encoder = data["encoder"] # defining the labelencoder file in the pkl file

NONE_THRESHOLD = 0.75   # probability of prediction lesser than this will be marked as NONE

mp_hands = mp.solutions.hands 
hands = mp_hands.Hands( # creating the hands object of the hands.Hands class of the mp solutions API
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    isTrue, frame = cap.read()
    if not isTrue:
        break

    frame = cv2.flip(frame,1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # collecting the information on the 21 points obtained on palm detection
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            row = np.array(row).reshape(1, -1) # converts 1D list row into a 2D numpy array -> apparently sklearn needs this shape

            prob = model.predict_proba(row)[0] # returns probabilities for every class your model was trained on for the given input. returns a 2D numpy array [[0.1, 0.7,..]]
            max_prob = np.max(prob) # finding the open with the maximum probability

            if max_prob < NONE_THRESHOLD:
                gesture = "None"
            else:
                prediction = np.argmax(prob) # returns the corresponding encoded gesture for the prediction with the highest probability
                gesture = encoder.inverse_transform([prediction])[0] # basically converts the encoded code obtained for the predicted gesture into human readable string.

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # putting up the text for the gesture prediction near the wrists
            h, w, _ = frame.shape # getting the shapes of the images -> height, width, channels (color channels - generally 3)
            wrist = hand_landmarks.landmark[0] # getting the normalized coordinates for the 0th landmark (the landmark of the wrist)
            x, y = int(wrist.x * w), int(wrist.y * h) # getting the absolute coordinates of the wrist using the dimensions of the image
            cv2.putText(frame, gesture, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3) # putting the text.

    cv2.imshow("Multi-Hand Gesture Recognition", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
