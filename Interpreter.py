import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import openai

# Load the trained defect detection model
cnn_model = tf.keras.models.load_model('defect_detection_model.h5')

# Set your OpenAI API key
openai.api_key = 'API'

# Function to generate textual explanations using OpenAI API
def generate_explanation_with_openai(confusion_matrix, classification_report):
    prompt = f"Confusion Matrix:\n{confusion_matrix}\n\nClassification Report:\n{classification_report}\n\nProvide a detailed interpretation of the model's performance based on the above results."

    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use "gpt-3.5-turbo" if you don't have access to GPT-4
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7,
    )

    explanation = response.choices[0].message["content"].strip()
    return explanation

# Assuming X_test and y_test are available from the previous steps
y_pred = cnn_model.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)

# Generate classification report and confusion matrix
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Generate textual explanation
explanation = generate_explanation_with_openai(conf_matrix, class_report)
print(explanation)
