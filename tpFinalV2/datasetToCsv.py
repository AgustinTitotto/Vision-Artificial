import os
import numpy as np
import mediapipe as mp
import cv2
import pandas as pd
import csv

image_path = []
label = []
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

# guardo la direccion y la etiqueta de cada imagen en el dataset
for dirname, _, filenames in os.walk('./dataset'):
    for filename in filenames:
        image_path.append(os.path.join(dirname, filename))
        label.append(os.path.split(dirname)[1])

image_path = pd.Series(image_path, name='path')
label = pd.Series(label, name='label')
df = pd.concat([image_path, label], axis=1)

# creo un archivo csv con las coordenadas de cada landmark de cada imagen
image = cv2.imread(df['path'][0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image.flags.writeable = False

# Make Detections
results = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5).process(image)
landmarks = ['label']
num_coords = len(results.pose_landmarks.landmark)
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

# Creo el csv
with open('coords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

# Guardo las coordenadas de cada landmark de cada imagen en el csv

for i in range(len(df)):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image = cv2.imread(df['path'][i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = holistic.process(image)

        # Export coordinates
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Concate rows
            row = pose_row

            # Append class name
            row.insert(0, df['label'][i])

            # Export to CSV
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
        except:
            pass

