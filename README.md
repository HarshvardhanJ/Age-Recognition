# Age Estimation using OpenCV and Tkinter

This project uses OpenCV for face detection and the AgeGenderNet model for age estimation. It provides a simple GUI using Tkinter to display the live video feed with age estimation.

## Models Used

### Face Detection Model
- **Model**: Pre-trained Haar Cascade Classifier for face detection from OpenCV.
- **Description**: The face detection model is based on a Haar Cascade Classifier provided by OpenCV. Haar Cascades are machine learning-based object detection methods that use patterns of pixel intensity to identify objects.
### Age Estimation Model
- **Model**: AgeGenderNet
- **Description**: A pre-trained Convolutional Neural Network (CNN) model for estimating age and gender.

## How to Use

1. Clone the repository:

    ```bash
    git clone https://github.com/HarshvardhanJ/Age-Recognition
    ```

2. Install required dependencies:

    ```bash
    pip install opencv-python pillow
    ```

    Make sure that `age_deploy.prototxt` and `age_net.caffemodel` are in the same folder as `face_reconition.py`. 

3. Run the application:

    ```bash
    python face_recognition.py
    ```

4. Press the "Start" button to begin the age estimation from the live video feed.
5. Press the "Stop" button to stop the video feed.
