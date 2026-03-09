import cv2
import numpy as np
import tensorflow as tf
import winsound  # For Windows alert sounds (use `playsound` for other OS)
import os

MODEL_PATH = os.path.join("models", "driver_action_model.h5")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please train the model first by running src/train.py.")
        return

    print("Loading the trained model...")
    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Recompile the model to ensure metrics are available
    model.compile(
        optimizer=tf.keras.optimizers.Adam(), 
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
    )
    print("Model successfully loaded and compiled!")

    # Define class labels matching the training folders (alphabetical order)
    class_labels = ["Safe Driving", "Adjusting Radio", "Eating", "Looking Back", 
                    "Makeup", "Talking Passenger", "Phone Left", "Phone Right",
                    "Texting Left", "Texting Right"]

    # In the original 20-class setup, indices mapped differently. Assuming a clean 10-class model:
    # c0_safe, c1_texting_right, c2_phone_right, c3_eating, c4_looking_back, 
    # c5_makeup, c6_talking_passenger, c7_texting_left, c8_phone_left, c9_adjust_radio
    # Note: Keras flow_from_directory sorts folders alphabetically:
    # 0 -> c0_safe
    # 1 -> c1_texting_right
    # 2 -> c2_phone_right
    # 3 -> c3_eating
    # 4 -> c4_looking_back
    # 5 -> c5_makeup
    # 6 -> c6_talking_passenger
    # 7 -> c7_texting_left
    # 8 -> c8_phone_left
    # 9 -> c9_adjust_radio
    
    class_labels = [
        "Safe Driving", 
        "Texting Right", 
        "Phone Right", 
        "Eating", 
        "Looking Back", 
        "Makeup", 
        "Talking Passenger", 
        "Texting Left",
        "Phone Left", 
        "Adjusting Radio"
    ]

    # Initialize webcam
    print("Initializing webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, change if using external cam

    # Set warnings for dangerous behavior
    dangerous_actions = ["Texting Right", "Texting Left", "Phone Right", "Phone Left", 
                         "Looking Back", "Makeup", "Adjusting Radio"]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera. Exiting...")
            break

        # Preprocess frame for model
        # The model expects RGB images. OpenCV captures in BGR.
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224)) / 255.0  # Resize & normalize
        img_expanded = np.expand_dims(img_resized, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_expanded, verbose=0)
        predicted_class = np.argmax(prediction)
        
        # Handle case where original model with 20 classes is used
        if prediction.shape[1] == 20:
             # Just map up to 10
             if predicted_class < 10:
                 action = class_labels[predicted_class]
             else:
                 action = "Unknown"
        else:
             action = class_labels[predicted_class]

        # Display action on screen
        color = (0, 255, 0) if action == "Safe Driving" else (0, 0, 255)
        cv2.putText(frame, f"Action: {action}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Add alert for dangerous behavior
        if action in dangerous_actions:
            winsound.Beep(1000, 200)  # Short alert sound for Windows

        # Show the frame
        cv2.imshow("Driver Action Detection", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
