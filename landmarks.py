import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

model = load_model('mp_hand_gesture')
# Load class names

classNames = ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']
print(classNames)
i = 0
cap = cv2.VideoCapture(0)
while True:
  if i == 1:
       i = 0
       continue
  i += 1
  # Read each frame from the webcam
  _, frame = cap.read()
  x , y, c = frame.shape
  # Flip the frame vertically
  frame = cv2.flip(frame, 1)
  # Show the final output
  framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # Get hand landmark prediction
  result = hands.process(framergb)
  className = ''
  # post process the result
  if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, 
mpHands.HAND_CONNECTIONS)
            prediction = model.predict([landmarks])
            print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]
  # show the prediction on the frame
  cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0,0,255), 2, cv2.LINE_AA)
  cv2.imshow("Output", frame)
  if cv2.waitKey(1) == ord('q'):
            break
# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()