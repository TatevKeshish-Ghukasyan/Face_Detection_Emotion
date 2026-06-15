# 😀 Face Detection & Emotion Recognition

A deep learning project that detects human faces and classifies facial emotions using Convolutional Neural Networks (CNNs).

The model is trained on the FER-2013 facial emotion dataset and can recognize seven different emotional states from facial images.

## 📌 Project Overview

Facial Emotion Recognition (FER) is a computer vision task that identifies a person's emotional state based on facial expressions.

This project combines:

* Face detection
* Image preprocessing
* Emotion classification
* Deep learning with CNNs
* Real-time or image-based emotion prediction

The system analyzes facial features and predicts one of the supported emotions.

## 😊 Supported Emotions

The model classifies faces into the following categories:

* Angry 😠
* Disgust 😖
* Fear 😨
* Happy 😄
* Neutral 😐
* Sad 😢
* Surprise 😲

The FER-2013 dataset contains 35,685 grayscale facial images (48×48 pixels) distributed across these seven emotion classes.

## 📊 Dataset

Dataset used:

**Emotion Detection (FER-2013)**

Dataset Source:

[Kaggle FER-2013 Dataset](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer?utm_source=chatgpt.com)

Dataset characteristics:

* 35,685 facial images
* 48×48 grayscale images
* 7 emotion classes
* Pre-divided into training and testing sets

## 🚀 Features

* Automatic face detection
* Emotion classification
* Deep learning-based recognition
* Training visualization
* Custom image prediction
* Model evaluation metrics
* TensorFlow/Keras implementation

## 📂 Project Structure

```text
Face_Detection_Emotion/
│
├── dataset/                # FER-2013 dataset
├── models/                 # Saved trained models
├── notebooks/              # Jupyter notebooks
├── images/                 # Sample test images
├── train.py                # Model training
├── predict.py              # Emotion prediction
├── requirements.txt        # Dependencies
└── README.md
```

## 🛠 Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/TatevKeshish-Ghukasyan/Face_Detection_Emotion.git
cd Face_Detection_Emotion
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📥 Download Dataset

Download the FER-2013 dataset from Kaggle and place it inside the project directory:

[Download FER-2013 Dataset](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer?utm_source=chatgpt.com)

## 🏋️ Model Training

Run:

```bash
python train.py
```

The training pipeline:

1. Loads FER-2013 images
2. Preprocesses and normalizes data
3. Splits data into training and validation sets
4. Trains the CNN model
5. Saves the trained model

## 🔍 Prediction

Predict emotion from an image:

```bash
python predict.py --image path/to/image.jpg
```

Example output:

```text
Detected Face

Emotion: Happy
Confidence: 96.3%
```

## 📈 Model Evaluation

The model is evaluated using:

* Accuracy
* Loss
* Validation Accuracy
* Validation Loss
* Confusion Matrix

Training history can be visualized to monitor learning performance and detect overfitting.

## 🧠 Model Architecture

The CNN architecture includes:

* Convolutional Layers
* ReLU Activation
* Max Pooling Layers
* Dropout Layers
* Dense Layers
* Softmax Output Layer

The output layer predicts probabilities for the seven emotion classes.

## 🎯 Applications

Facial emotion recognition can be applied in:

* Human-Computer Interaction
* Healthcare Monitoring
* Educational Systems
* Customer Experience Analysis
* Smart Assistants
* Behavioral Research

## 📷 Example Predictions

| Face | Prediction |
| ---- | ---------- |
| 😀   | Happy      |
| 😢   | Sad        |
| 😠   | Angry      |
| 😲   | Surprise   |

## 🔮 Future Improvements

* Real-time webcam emotion detection
* Transfer Learning (ResNet, EfficientNet)
* Mobile deployment
* Multi-face emotion recognition
* Emotion analytics dashboard
* Improved accuracy with data augmentation

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

## 👩‍💻 Author

**Tatev Keshish-Ghukasyan**

GitHub Repository:

https://github.com/TatevKeshish-Ghukasyan/Face_Detection_Emotion

## 📄 License

This project is intended for educational, research, and learning purposes.
