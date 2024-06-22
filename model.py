import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import os
import ast

# Load CSV file
data_df = pd.read_csv('synthetic_defect_data.csv')

# Function to convert flattened image data back to 3D arrays
def convert_to_image_array(flattened_data, img_size=(150, 150, 3)):
    return np.array(flattened_data).reshape(img_size)

# Prepare data and labels
data = []
labels = []
for idx, row in data_df.iterrows():
    # Safely parse the string representation of the list
    img_array = convert_to_image_array(ast.literal_eval(row['image']))
    data.append(img_array)
    labels.append(row['label'])

data = np.array(data)
labels = np.array(labels)

# Normalize data
data = data / 255.0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=10,
    validation_data=(X_test, y_test)
)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the model
model.save('defect_detection_model.h5')
