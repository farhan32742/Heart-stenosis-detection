# YOLO Stenosis Detection and Volume Calculation

## Overview
This project leverages a YOLO (You Only Look Once) model to detect stenosis in medical images and calculate their approximate volume. The application is built using Flask and provides a simple web interface to upload images for analysis.

Key features include:
- **Stenosis detection** using a pre-trained YOLO model.
- **Volume estimation** for detected stenosis regions.
- Annotations directly on the uploaded image, displaying bounding boxes and calculated volumes.

## Prerequisites
Before running the application, ensure you have the following installed:

1. **Python** (>= 3.8)
2. Required Python libraries:
   - `Flask`
   - `cv2` (OpenCV)
   - `ultralytics`
   - `numpy`
   - `matplotlib`
   - `Pillow`
3. A trained YOLO model file (`best(2).pt`) located in the specified directory.

## File Structure
```
project_directory/
│
├── stenosis_detection/
│   ├── best(2).pt       # YOLO model file
│   └── ...              # Other related files
├── app.py               # Main Flask application
├── templates/
│   └── index.html       # HTML template for the web interface
└── static/              # (Optional) For storing static assets like CSS/JS
```

## Setting Up the Environment
1. Clone the repository or create a new directory for the project.
2. Navigate to the project directory and install the required dependencies:
   ```bash
   pip install flask ultralytics opencv-python-headless numpy matplotlib pillow
   ```

3. Place the trained YOLO model file (`best(2).pt`) in the `stenosis_detection` directory.

## Running the Application
1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open a web browser and navigate to `http://127.0.0.1:5000` to access the application.

## Usage
1. On the web interface, upload a medical image for analysis.
2. The server processes the image and:
   - Detects stenosis regions using the YOLO model.
   - Calculates the volume of each detected region.
   - Annotates the image with bounding boxes and volume information.
3. The processed image is displayed and can be downloaded.

## Key Functions
### `calculate_stenosis_volume`
Calculates the volume of stenosis regions using the formula:
```
Volume = Width × Height × Depth
```
- The depth is approximated as the average of the width and height.
- Converts volume from cubic pixels to cubic centimeters (cm³) assuming a scale of 1 pixel = 0.1 cm.

### `annotate_image_with_stenosis`
- Runs the YOLO model on the input image.
- Draws bounding boxes around detected stenosis regions.
- Adds volume annotations on the image.

### Flask Routes
- `/`: Serves the web interface (index.html).
- `/upload`: Handles image uploads, processes the image, and returns the annotated image.

## Example Input/Output
### Input
Upload a medical image in `.jpg` or `.png` format.

### Output
The annotated image will include:
- Red bounding boxes around detected stenosis regions.
- Volume displayed as `Stenosis V:<calculated_volume>cm³`.

## Notes
- The YOLO model must be trained and fine-tuned specifically for stenosis detection to ensure accurate results.
- Adjustments may be required for different image resolutions or pixel-to-centimeter scale ratios.

## Future Improvements
- Add support for batch processing of images.
- Enhance the volume estimation algorithm by incorporating depth information from 3D imaging techniques.
- Include an option to download annotated images directly from the interface.


