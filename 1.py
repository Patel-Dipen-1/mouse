import cv2
import mediapipe as mp
import autopy
import numpy as np

# Set screen width and height for mouse control
screen_width, screen_height = autopy.screen.size()

# Initialize mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Main loop for hand tracking and mouse control
cap = cv2.VideoCapture(0)  # Set the webcam index accordingly (0 for default webcam)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Unable to read from webcam.")
        break

    # Flip the frame horizontally for a more intuitive view
    frame = cv2.flip(frame, 1)

    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                # Extract landmark positions
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                
                # Scale coordinates for mouse control
                x = np.interp(x, [0, w], [0, screen_width])
                y = np.interp(y, [0, h], [0, screen_height])

                # Check if the coordinates are within the screen bounds
                if 0 <= x < screen_width and 0 <= y < screen_height:
                    # Move the mouse cursor
                    autopy.mouse.move(x, y)

            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame
    cv2.imshow("Hand Tracking", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
