import numpy as np  # linear algebra
import pandas as pd  # data processing)
import os
import IPython  # display picture
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image  # load picture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

seed = 333
image_path = []
label = []

for dirname, _, filenames in os.walk('./dataset'):
    for filename in filenames:
        image_path.append(os.path.join(dirname, filename))
        label.append(os.path.split(dirname)[1])



image_path = pd.Series(image_path, name='path')
label = pd.Series(label, name='label')
df = pd.concat([image_path,label], axis=1)

labels=[]
X=[]

for i in range(len(df)):
    img = Image.open(df['path'][i])
    img = img.resize((128, 128))
    img = np.array(img)
    if len(img.shape)>2 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif len(img.shape)<3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        X.append(img)
        labels.append(df['label'][i])



label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(labels)

#print labels from label_encoded
for i in range(len(label_encoded)):
    print(label_encoded[i])

for i in range(len(label)):
    print(label[i])

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()


x_position = np.zeros((len(X),33,4))
y_position = np.zeros(label_encoded.shape)
x_position.shape, y_position.shape


for i in range(len(X)):
    results = pose.process(X[i])
    if results.pose_landmarks is not None:
        positions = results.pose_landmarks.landmark
        j=0
        for landmark in positions:
            x = landmark.x
            y = landmark.y
            z = landmark.z
            v = landmark.visibility
            x_position[i,j] = [x,y,z,v]
            y_position[i] = label_encoded[i]+1
            j+=1

x_position = x_position.reshape(x_position.shape[0],x_position.shape[1]*x_position.shape[2])



X_train, X_test, y_train, y_test = train_test_split(x_position, y_position,
                                                    test_size=0.2, shuffle=True,
                                                    random_state=seed)



smote = SMOTE(random_state=seed, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

forest = RandomForestClassifier(n_estimators=51,random_state=seed)
forest.fit(X_resampled, y_resampled)

joblib.dump(forest, "model.pkl")







