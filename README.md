# 🐟 Fish Classification

A deep learning image classification project to identify different fish species using TensorFlow and Keras.

---

## 🚀 Project Overview

This repository contains code and trained models for classifying fish species from images. It includes:

- Multiple pretrained CNN models converted and saved in Keras format
- Utilities for loading and preprocessing images
- A simple Streamlit app for interactive image classification (under development)
- Scripts for training and converting models

---

## 📁 Repository Structure

Fish/
├── models/ # Source model training/conversion scripts
├── models_converted/ # Pretrained Keras model files (.keras, .h5)
├── images.cv_*/ # Dataset folders containing fish images
├── app.py # Streamlit app for inference
├── requirements.txt # Python dependencies
└── README.md # Project documentation (this file)


---

## ⚙️ Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/yourusername/fish-classification.git
cd fish-classification
```
## (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

## Install dependencies:

```bash
pip install -r requirements.txt
```
## Running the Streamlit app (experimental)

```bash
streamlit run app.py
```

## 🐠 Models Included
EfficientNetB0

InceptionV3

MobileNet

MobileNetV2

ResNet50

VGG16

CNN from scratch (basic custom CNN)

All models have been converted to Keras .keras or .h5 format and are stored in the models_converted/ directory.
