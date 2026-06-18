# Face Detection & Emotion Recognition

## Overview

This project is a Deep Learning and Computer Vision application that detects human faces and recognizes facial emotions from images or real-time video streams. Using OpenCV for face detection and TensorFlow/Keras for emotion classification, the system analyzes facial expressions and predicts the emotional state of detected individuals.

The model is capable of classifying the following emotions:

* Happy
* Sad
* Angry
* Fear
* Surprise
* Disgust
* Neutral

Facial emotion recognition is widely used in human-computer interaction, healthcare, customer analytics, education, and intelligent surveillance systems.

---

## Features

* Real-time face detection using OpenCV
* Facial emotion recognition with Deep Learning
* Image and webcam-based prediction
* TensorFlow/Keras model implementation
* User-friendly prediction visualization
* Support for multiple faces in a frame

---

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Matplotlib

---

## Dataset

The model is trained on facial expression datasets containing labeled images of different emotions. Images are preprocessed, normalized, and augmented to improve model performance and generalization. Similar emotion recognition projects commonly use the FER-2013 dataset, which contains seven emotion categories.

---

## Project Structure

```text
Face_Detection_Emotion/
│
├── dataset/                 # Training and testing data
├── models/                  # Saved trained models
├── notebooks/               # Jupyter notebooks
├── images/                  # Sample images
├── train.py                 # Model training script
├── predict.py               # Emotion prediction script
├── requirements.txt         # Dependencies
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/TatevKeshish-Ghukasyan/Face_Detection_Emotion.git
cd Face_Detection_Emotion
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the Model

```bash
python train.py
```

### Predict Emotion from an Image

```bash
python predict.py --image path/to/image.jpg
```

### Real-Time Webcam Detection

```bash
python predict.py --webcam
```

---

## Model Architecture

The emotion recognition model is based on Convolutional Neural Networks (CNNs), which automatically learn facial features and patterns associated with different emotions. The architecture includes:

* Convolutional Layers
* Max Pooling Layers
* Dropout Layers
* Dense Layers
* Softmax Output Layer

CNN-based approaches have shown strong performance in facial emotion recognition tasks.

---

## Results

The trained model successfully detects faces and classifies emotions with high accuracy on validation data. Predictions are displayed directly on the detected faces, providing an intuitive real-time visualization of emotional states.

---

## Future Improvements

* Improve model accuracy with larger datasets
* Deploy as a web application
* Mobile application integration
* Multi-face emotion tracking
* Emotion analytics dashboard

---

## Author

**Tatev Keshish-Ghukasyan**

Machine Learning & Artificial Intelligence Engineer

GitHub: https://github.com/TatevKeshish-Ghukasyan
