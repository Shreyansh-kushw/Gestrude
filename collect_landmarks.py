"""Collects data sets for training the model, and stores it in dataset.csv"""

# importing modules
import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands # creating an object for the hands class of the mediapipe solutions API
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1) # Staic_image_mode = False -> Detection is ran on first few frames then hands are tracked and detector is only run when mp loses track of the hand
mp_draw = mp.solutions.drawing_utils # object creation for drawing class of mp solutions API

gesture_name = input("Enter gesture name: ") 

f = open("dataset.csv", "a", newline="")
writer = csv.writer(f) # creating a write object for writing the opened csv file

cap = cv2.VideoCapture(0)
print("Press 's' to save a sample, 'q' to quit.")

while True:
    isTrue, frame = cap.read()
    if not isTrue:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # image -> bgr to rgb
    results = hands.process(img_rgb) # processing the rgb image using mp hands object.

    if results.multi_hand_landmarks: # checking is there are hands in the frame.
        # collect 63 values (x,y,z for 21 points)
        row = []
        for lm in hands.landmark:
            row.extend([lm.x, lm.y, lm.z]) # extending the list (row) by adding the following elements in order
            # append vs extend -> append creates nested lists while extend adds the elements.

        # drawing landmarks
        mp_draw.draw_landmarks(frame, hands, mp_hands.HAND_CONNECTIONS) 

        cv2.putText(frame, "Press 's' to save", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Collecting datasets", frame)

        if cv2.waitKey(20) & 0xFF == ord('s'):
            writer.writerow([gesture_name] + row) # writing the row in the csv dataset file
            print("Saved sample.")

        elif cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        # when there is no hand in the frame
        cv2.imshow("Collecting datasets", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
f.close()
print("Done.")
