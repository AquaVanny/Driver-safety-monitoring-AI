import os
import shutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Configuration
# Assuming data is placed in the `data/` folder of the project root
DATA_DIR = os.path.join("data", "statefarm_data", "imgs", "train")
MODEL_SAVE_PATH = os.path.join("models", "driver_action_model.h5")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

def organize_dataset(base_dir):
    """Organize the dataset into class folders if not already done."""
    if not os.path.exists(base_dir):
        print(f"Data directory {base_dir} not found.")
        print("Please extract the state farm dataset so the path exists.")
        return False
        
    categories = ["c0_safe", "c1_texting_right", "c2_phone_right", "c3_eating",
                  "c4_looking_back", "c5_makeup", "c6_talking_passenger",
                  "c7_texting_left", "c8_phone_left", "c9_adjust_radio"]

    # Create directories for categories
    for category in categories:
        os.makedirs(os.path.join(base_dir, category), exist_ok=True)

    # Move images to respective category folders
    organized = False
    for category in categories:
        category_label = category.split("_")[0]  # Extract category number, e.g., 'c0'
        src_folder = os.path.join(base_dir, category_label)
        dst_folder = os.path.join(base_dir, category)
        
        if os.path.exists(src_folder) and src_folder != dst_folder:
            for file in os.listdir(src_folder):
                shutil.move(os.path.join(src_folder, file), os.path.join(dst_folder, file))
            
            # Remove the empty original folders to prevent Keras from thinking there are 20 classes
            try:
                os.rmdir(src_folder)
            except OSError:
                pass
            organized = True
            
    if organized:
        print("Dataset organized into descriptive class folders.")
    return True

def create_model(num_classes=10):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2,2),
        
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),  # Prevent Overfitting
        Dense(num_classes, activation='softmax')  # Output layer
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model

def main():
    print("Checking dataset...")
    if not organize_dataset(DATA_DIR):
        print("Ensure dataset is placed correctly before training.")
        return

    # Data Augmentation & Preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,         # Normalize pixel values
        rotation_range=20,      # Random rotation
        width_shift_range=0.2,  # Horizontal shift
        height_shift_range=0.2, # Vertical shift
        shear_range=0.2,        # Shear transformation
        zoom_range=0.2,         # Zoom in/out
        horizontal_flip=True,   # Flip images
        validation_split=0.2    # 80% Train, 20% Validation Split
    )

    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    print("Loading validation data...")
    val_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    num_classes = len(train_generator.class_indices)
    print(f"Number of Classes Detected: {num_classes}")

    model = create_model(num_classes=num_classes)
    model.summary()

    # Train the model
    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        verbose=1
    )

    # Save the Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved successfully to {MODEL_SAVE_PATH}!")

    # Plot Accuracy & Loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title("Model Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Model Loss")

    plt.savefig(os.path.join("models", "training_history.png"))
    print("Training history plot saved to models/training_history.png.")

if __name__ == "__main__":
    main()
