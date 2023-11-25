import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_pose = mp.solutions.pose  # Mediapipe Solutions

cap = cv2.VideoCapture(0)
cv2.namedWindow('img')


# Initiate holistic model
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def invalidElbowAngle(angle):
    return angle < 20 or angle > 40


def invalidArmAngle(angle):
    return angle < 145 or angle > 170

def invalidLegAngle(left, right):
    return left < 175 or right < 175 or left > 180 or right > 180


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

            # Get coordinates
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

            # Calculate angles
            leftElbowAngle = calculate_angle(leftShoulder, leftElbow, leftWrist)
            rightArmAngle = calculate_angle(rightShoulder, rightElbow, rightWrist)
            leftLegAngle = calculate_angle(leftAnkle, leftKnee, leftShoulder)
            rightLegAngle = calculate_angle(rightAnkle, rightKnee, rightShoulder)

            # Posture detection
            if invalidArmAngle(rightArmAngle):
                cv2.putText(image, "WRONG ARM POSITION",
                            tuple(np.multiply(rightShoulder, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                                          )

            if invalidElbowAngle(leftElbowAngle):
                # change the color of the lines to red
                cv2.putText(image, "WRONG ELBOW POSITION",
                            tuple(np.multiply(leftWrist, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                                          )

            if invalidLegAngle(leftLegAngle, rightLegAngle):
                cv2.putText(image, "WRONG LEG POSITION",
                            tuple(np.multiply(leftAnkle, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                                          )

        except:
            pass

        cv2.imshow('img', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
