# Breast-cancer-detection

This project involves training a neural network model to detect breast cancer based on microscopic images of tumor tissue. We utilized a Kaggle dataset containing 7,909 images—2,480 of benign tumors and 5,429 of malignant tumors.

The dataset was preprocessed, including image resizing and normalization, before being fed into a Convolutional Neural Network (CNN). The model architecture includes multiple convolutional layers, max pooling, and dense layers to optimize feature extraction and classification.

The model was trained to achieve high accuracy in distinguishing between benign and malignant tumors, providing a valuable tool for early detection.

For more details, you can access the dataset here: [Kaggle Breast Cancer Prediction Dataset.](https://www.kaggle.com/datasets/merishnasuwal/breast-cancer-prediction-dataset/data).

# Key Features
Python 3.x

TensorFlow

Keras

OpenCV

NumPy

# Code Overview

**Data Preparation:** Efficient loading and preprocessing of images using OpenCV and NumPy.

**Model Architecture:** Utilizes a multi-layer CNN with convolutional, pooling, and dense layers to extract features and classify images.

**Training and Validation:** Model trained on a specified dataset with a validation split to monitor performance.

**Evaluation Metrics:** Outputs loss and accuracy to gauge model effectiveness.

**Model Persistence:** Saves the trained model for future inference or further training.
