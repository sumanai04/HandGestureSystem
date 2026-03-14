import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="mobilenetv2_gesture_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Update the input shape based on the model's requirements
input_shape = input_details[0]['shape']
input_height = input_shape[1]
input_width = input_shape[2]

print("Starting live video feed... Press 'q' to quit.")

# Access the webcam ONCE before the loop
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        # Capture frame-by-frame continuously
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # --- Preprocessing ---
        # Resize to match the model's expected input size
        resized_image = cv2.resize(frame, (input_width, input_height))
        # Convert BGR (OpenCV default) to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        normalized_image = rgb_image / 255.0
        # Add batch dimension and convert to float32
        input_tensor = np.expand_dims(normalized_image, axis=0).astype(np.float32)

        # --- Inference ---
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        
        # Get the prediction result
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        # Determine label and text color
        if prediction > 0.5:
            class_label = "ON"
            color = (0, 255, 0) # Green
        else:
            class_label = "OFF"
            color = (0, 0, 255) # Red

        # --- Display ---
        # Draw the text directly onto the live video frame
        cv2.putText(frame, f"Status: {class_label} ({prediction:.2f})", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        # Show the live feed window
        cv2.imshow('TFLite Hand Gesture Recognition', frame)

        # Wait for 1 millisecond and check if 'q' was pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user.")
            break

except KeyboardInterrupt:
    print("Stopped by user via console.")

finally:
    # Always release the webcam and destroy windows when done
    cap.release()
    cv2.destroyAllWindows()