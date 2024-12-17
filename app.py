from flask import Flask, request, render_template, send_file
import cv2
import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the YOLO model
model = YOLO("C:\\Users\\PC\\Desktop\\stenosis_detection\\stenosis_detection\\best(2).pt")

# Function to calculate volume (using a similar assumption for depth)
def calculate_stenosis_volume(width, height, depth=None):
    if depth is None:
        depth = (width + height) / 2  # Approximate depth
    volume = width * height * depth  # Volume in cubic pixels
    volume_cm3 = volume * (0.1 ** 3)  # Convert to cubic cm (assuming 1 pixel = 0.1 cm)
    return volume_cm3

# Function to annotate stenosis areas
def annotate_image_with_stenosis(img, model):
    # Perform inference using the YOLO model
    results = model(img)

    # Get the original image
    orig_img = results[0].orig_img

    # Iterate through detected objects
    for result in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = result

        # Calculate bounding box dimensions
        width = x2 - x1
        height = y2 - y1

        # Calculate stenosis volume
        volume = calculate_stenosis_volume(width, height)

        # Set annotation details
        class_name = "Stenosis"  # Adjust if necessary
        color = (0, 0, 255)  # Red for stenosis

        # Draw bounding box
        cv2.rectangle(orig_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Annotate volume
        label = f"{class_name} V:{volume:.2f}cm^3"
        cv2.putText(orig_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return orig_img, results[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    if file:
        # Read the image file
        in_memory_file = BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)
        file_bytes = np.asarray(bytearray(in_memory_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Process the image
        annotated_img, model_output = annotate_image_with_stenosis(img, model)

        # Convert the annotated image to a format Flask can serve
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_image_bytes = BytesIO(buffer)

        return send_file(annotated_image_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)