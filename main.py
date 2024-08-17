import cv2
import mediapipe as mp

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to count fingers
def count_fingers(image, landmarks):
    count = 0

    # Tip ids for fingers: [4, 8, 12, 16, 20]
    finger_tip_ids = [4, 8, 12, 16, 20]

    for tip_id in finger_tip_ids:
        # Compare the tip with its preceding landmark to check if the finger is open
        if tip_id != 4:
            if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                count += 1
        else:
            if landmarks[tip_id].x > landmarks[tip_id - 2].x:
                count += 1

    return count

# Start capturing video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for natural (selfie-view) visualization
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands and landmarks
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmark positions
            landmarks = hand_landmarks.landmark

            # Count fingers
            fingers = count_fingers(frame, landmarks)
            
            # Display the number of fingers
            cv2.putText(frame, f'{fingers}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Finger Counter", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
