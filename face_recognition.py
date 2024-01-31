import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import math

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')

la = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'] 

# Function to detect faces and estimate age in a given frame
def detect_faces_and_estimate_age(frame):
    # Convert the frame to grayscale (required for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Get the region of interest (ROI) for age detection
        face_roi = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face_roi, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Make age prediction
        age_net.setInput(blob)
        age_preds = age_net.forward()

        # Get the predicted age
        age = la[age_preds[0].argmax()] 

        # Display the age on the frame
        cv2.putText(frame, f'Age: {age}', (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return frame

# Function to update the video feed
def update_feed():
    ret, frame = cap.read()
    if ret:
        # Detect faces and estimate age in the frame
        frame_with_faces_and_age = detect_faces_and_estimate_age(frame)

        # Convert the frame to RGB format for Tkinter
        img = cv2.cvtColor(frame_with_faces_and_age, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        # Update the label image
        label.img = img
        label.config(image=img)

    # Repeat the update after a delay (in milliseconds)
    label.after(10, update_feed)

# Function to start the video feed
def start_video_feed():
    global cap
    cap = cv2.VideoCapture(0)
    update_feed()

# Function to stop the video feed
def stop_video_feed():
    cap.release()

# Create the main Tkinter window
root = tk.Tk()
root.title("Age Estimation")

# Create Start and Stop buttons
start_button = ttk.Button(root, text="Start", command=start_video_feed)
start_button.pack(pady=10)

stop_button = ttk.Button(root, text="Stop", command=stop_video_feed)
stop_button.pack(pady=10)

# Create a label to display the video feed
label = ttk.Label(root)
label.pack()

# Run the Tkinter event loop
root.mainloop()

