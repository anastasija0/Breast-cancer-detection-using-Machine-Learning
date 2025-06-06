# Breast Cancer Detection using Deep Learning

This project implements a Convolutional Neural Network (CNN) to classify breast cancer images as either benign or malignant.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Data Organization:
Place your dataset in the following structure:
```
C:/Datasets/BreastCancer/
├── training/
│   ├── benign/
│   └── malignant/
└── testing/
    ├── benign/
    └── malignant/
```

3. Run the model:
```bash
python app.py
```

## Model Architecture
- Input: Grayscale images (120x120 pixels)
- 2 Convolutional layers with MaxPooling
- Dense layers for classification
- Binary output (benign/malignant)

## Performance Metrics
The model evaluates accuracy and loss on the test set.
