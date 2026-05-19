# lenet5-digit-recognition
# LeNet-5 Digit Recognition

## Overview
This project implements an End-to-End Numeral Extraction System using LeNet-5 CNN architecture.

The system:
- Detects handwritten digits
- Recognizes numerals using Deep Learning
- Uses the MNIST dataset

---

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- CNN (LeNet-5)

---

## Dataset
MNIST handwritten digit dataset

- 70,000 digit images
- 28×28 grayscale images
- Digits from 0–9

---

## Project Structure

lenet5-digit-recognition/
│
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── lenet5_model.h5

---

## How to Run

### Install dependencies

pip install -r requirements.txt

### Train model

python train.py

### Predict digit

python predict.py

---

## Features
- CNN-based digit recognition
- LeNet-5 architecture
- Image preprocessing
- High accuracy on MNIST

---

## Future Improvements
- Real-time OCR
- Vehicle number plate recognition
- Utility bill digit extraction