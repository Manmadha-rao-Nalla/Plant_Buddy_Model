from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)

# Load model and class labels
interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
classes = np.load("classes.npy")

IMAGE_SIZE = (128, 128)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', result="❌ No file uploaded.")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', result="❌ No file selected.")

    img = Image.open(file).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    disease_name = classes[class_index]
    health_pct = round((1 - confidence) * 100, 2) if "healthy" not in disease_name else 100
    disease_pct = round(confidence * 100, 2) if "healthy" not in disease_name else 0

    result = f"Prediction: {disease_name} | Health: {health_pct}% | Disease Severity: {disease_pct}%"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)