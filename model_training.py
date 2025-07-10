import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from PIL import Image
from tkinter import Tk, filedialog

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATA_DIR = "c:/Users/bhanu/OneDrive/Desktop/Hackerthon/abc"
IMAGE_SIZE = (128, 128)

rows = []
for plant in os.listdir(DATA_DIR):
    plant_path = os.path.join(DATA_DIR, plant)
    if os.path.isdir(plant_path):
        for img_file in os.listdir(plant_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                disease = os.path.splitext(img_file)[0].lower()
                severity = 0 if "healthy" in disease else 50
                rows.append({
                    "image_path": os.path.join(plant_path, img_file),
                    "plant_name": plant.lower(),
                    "disease_name": disease,
                    "severity": severity
                })

df = pd.DataFrame(rows)
df.to_csv("labels.csv", index=False)
print(f"✅ labels.csv generated with {len(df)} entries")

X, y = [], []
for _, row in df.iterrows():
    img = cv2.imread(row["image_path"])
    if img is None:
        print(f"⚠️ Could not read image: {row['image_path']}")
        continue
    img = cv2.resize(img, IMAGE_SIZE)
    X.append(img)
    y.append(row["disease_name"])

X = np.array(X) / 255.0
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)

model.save("plant_disease_model.h5")
np.save("classes.npy", le.classes_)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("plant_disease_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model trained and saved as H5 and TFLite.")
