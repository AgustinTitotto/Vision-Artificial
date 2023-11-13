import cv2
import numpy as np
import mediapipe as mp
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained RandomForestClassifier model
model_filename = 'model.pkl'
loaded_model = joblib.load(model_filename)

# Load the image you want to identify the pose from
image_path = './dataset/correctPosture/24.png'
img = cv2.imread(image_path)

cap = cv2.VideoCapture(0)  # Use laptop camara
cv2.namedWindow('img1')  # Create Window
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    flip_frame = cv2.flip(frame, 1)

    # Preprocess the image and extract pose landmarks

    results = pose.process(flip_frame)

    # Extract pose landmarks and format them
    if results.pose_landmarks is not None:
        positions = results.pose_landmarks.landmark
        pose_data = []
        for landmark in positions:
            x = landmark.x
            y = landmark.y
            z = landmark.z
            v = landmark.visibility
            pose_data.append([x, y, z, v])

        # Reshape the pose data
        pose_data = np.array(pose_data).reshape(1, -1)

        # Use the trained model to make predictions
        predicted_label = loaded_model.predict(pose_data)[0]

        # Decode the label (if needed)
        text = 'help'
        print(predicted_label)

        if predicted_label == 1.0:
            text = 'correctPosture'
        elif predicted_label == 2.0:
            text = 'incorrect arm posture'
        elif predicted_label == 3.0:
            text = 'incorrect elbow posture'
        elif predicted_label == 4.0:
            text = 'incorrect leg posture'
        else:
            text = 'nothing'

        mp_draw.draw_landmarks(flip_frame, results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                               mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                               )

        # Display the image with the prediction
        cv2.putText(flip_frame, f'Predicted Pose: {text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Image with Prediction', flip_frame)

    if cv2.waitKey(1) == ord('z'):  # Waits () amount of time, if the key 'z' is pressed, it stops the loopz
        break

cap.release()
cv2.destroyAllWindows()
