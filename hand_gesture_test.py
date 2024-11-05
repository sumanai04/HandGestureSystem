import cv2
import time
import numpy as np
import tensorflow as tf
import os

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="mobilenetv2_gesture_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Update the input shape based on the model's requirements
input_shape = input_details[0]['shape']
input_height = input_shape[1]
input_width = input_shape[2]

# Define the function to preprocess images for the TFLite model.
def preprocess_image(image):
    image = cv2.resize(image, (input_width, input_height))  # Resize to match the model's expected input size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension and convert to float32
    return image

# Define the function to capture and process an image.
def capture_and_predict():
    # Access the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    ret, frame = cap.read()  # Capture a single frame
    cap.release()  # Release the webcam
    
    if ret:
        # Preprocess and run inference
        preprocessed_image = preprocess_image(frame)
        interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
        interpreter.invoke()
        
        # Get the prediction result
        prediction = interpreter.get_tensor(output_details[0]['index'])
        class_label = 'on' if prediction[0][0] > 0.5 else 'off'
        print(f"Predicted status: {class_label}")

        # Save and delete the image (for demonstration, saved temporarily)
        img_path = "temp_image.jpg"
        cv2.imwrite(img_path, frame)
        os.remove(img_path)  # Delete the image after processing

# Loop to capture images every 5 seconds
try:
    while True:
        capture_and_predict()
        time.sleep(5)  # Wait for 5 seconds before capturing the next image
except KeyboardInterrupt:
    print("Stopped by user.")
