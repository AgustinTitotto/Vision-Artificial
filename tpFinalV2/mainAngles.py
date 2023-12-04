import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_pose = mp.solutions.pose  # Mediapipe Solutions

# Choose handedness ('right' or 'left')
selected_hand = 'right'


# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Function to calculate distance between two points
def distance_to(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.linalg.norm(a - b)


# Function to check if elbow angle is invalid
def invalidElbowAngle(elbow_angle, wrist_to_mouth_distance):
    return elbow_angle < 20 or elbow_angle > 40 or wrist_to_mouth_distance > 0.1


# Function to check if arm angle is invalid
def invalidArmAngle(angle):
    return angle < 145 or angle > 170


# Function to check if leg angles are invalid
def invalidLegAngle(left, right):
    return left < 175 or right < 175 or left > 180 or right > 180


# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow('ACERTA')

# Initiate holistic model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = pose.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                                  )

        try:
            # Extract Pose landmarks
            landmarks = results.pose_landmarks.landmark

            leftMouth = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x,
                         landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]

            rightMouth = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x,
                          landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]

            leftShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            leftElbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            leftWrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            leftKnee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            leftAnkle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            rightShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            rightElbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

            rightWrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            rightKnee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            rightAnkle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Get coordinates based on handedness
            if selected_hand == 'right':
                wrist_to_mouth_distance = distance_to(leftWrist, leftMouth)
                elbow_angle = calculate_angle(leftShoulder, leftElbow, leftWrist)
                arm_angle = calculate_angle(rightShoulder, rightElbow, rightWrist)
                leg_angle_left = calculate_angle(leftAnkle, leftKnee, leftShoulder)
                leg_angle_right = calculate_angle(rightAnkle, rightKnee, rightShoulder)
                incorrect_arm_position = rightWrist
                incorrect_elbow_position = leftElbow

            else:  # Use left-handed landmarks
                wrist_to_mouth_distance = distance_to(rightWrist, rightMouth)
                elbow_angle = calculate_angle(rightShoulder, rightElbow, rightWrist)
                arm_angle = calculate_angle(leftShoulder, leftElbow, leftWrist)
                leg_angle_left = calculate_angle(rightAnkle, rightKnee, rightShoulder)
                leg_angle_right = calculate_angle(leftAnkle, leftKnee, leftShoulder)
                incorrect_arm_position = leftWrist
                incorrect_elbow_position = rightElbow

            # Posture detection
            if invalidArmAngle(arm_angle):
                cv2.putText(image, "WRONG ARM POSITION",
                            tuple(np.multiply(incorrect_arm_position,
                                              [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                                          )

            if invalidElbowAngle(elbow_angle, wrist_to_mouth_distance):
                cv2.putText(image, "WRONG ELBOW POSITION",
                            tuple(np.multiply(incorrect_elbow_position,
                                              [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                                          )

            if invalidLegAngle(leg_angle_left, leg_angle_right):
                cv2.putText(image, "WRONG LEG POSITION",
                            tuple(np.multiply(leftAnkle,
                                              [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                                          )

        except:
            pass

        # Display selected hand information
        cv2.putText(image, f"Selected Hand: {selected_hand.capitalize()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "Press 'r' to change",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('ACERTA', image)
        # Check for key press to change selected hand
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Toggle selected hand between 'right' and 'left'
            selected_hand = 'left' if selected_hand == 'right' else 'right'

cap.release()
cv2.destroyAllWindows()
