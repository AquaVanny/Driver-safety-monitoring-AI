# 🚗 Driver Action Detection

A deep learning project to detect distracted driving behaviors in real-time. This project uses a Convolutional Neural Network (CNN) trained on the [State Farm Distracted Driver Detection dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection), along with OpenCV to deploy a live detection system using your webcam.

## 📌 Features
- **Data Preprocessing & Augmentation**: Automatically organizes images into class folders and applies data augmentation.
- **CNN Model**: A custom Convolutional Neural Network built with TensorFlow/Keras to classify 10 different driver actions.
- **Live Webcam Inference**: Uses OpenCV to capture real-time webcam feed and plays an alert sound when dangerous behavior (e.g., texting, looking back) is detected.

## 📂 Project Structure
```text
driver-action-detection/
│
├── data/                   # Directory to store dataset
├── models/                 # Saved trained models (.h5) & plots
├── notebooks/              # Jupyter notebooks for experimentation
├── src/                    # Source code for training and inference
│   ├── train.py            # Script to preprocess data and train the CNN
│   └── live_inference.py   # Script for real-time webcam detection
├── .gitignore              # Files and folders to ignore in Git
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/driver-action-detection.git
cd driver-action-detection
```

### 2. Install Dependencies
Make sure you have Python 3.8+ installed. Install the required packages via pip:
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
1. Download the dataset from [Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection).
2. Extract the `statefarm_dataset.zip` into the `data/` folder. The structure should look like `data/statefarm_data/imgs/train/`.

### 4. Train the Model
Run the training script. It will automatically organize the dataset and begin training the CNN.
```bash
python src/train.py
```
*Note: The script fixes an issue where the original notebook recognized 20 classes instead of 10 by cleaning up old empty folders after organization. Your output model will appropriately use 10 outputs instead of 20.*

### 5. Live Webcam Detection
Once the model is trained and saved in the `models/` directory, run the live inference script:
```bash
python src/live_inference.py
```
Press `q` to quit the webcam feed.

## 📖 Classes Recognized
- `c0`: Safe Driving
- `c1`: Texting - Right
- `c2`: Talking on the Phone - Right
- `c3`: Eating
- `c4`: Looking Back
- `c5`: Applying Makeup
- `c6`: Talking to Passenger
- `c7`: Texting - Left
- `c8`: Talking on the Phone - Left
- `c9`: Adjusting Radio

## 🔧 Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📜 License

