import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np 
import os
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Configuration
SEED = 42
IMG_SIZE = 120
BATCH_SIZE = 32
EPOCHS = 20
np.random.seed(SEED)
tf.random.set_seed(SEED)

traindir = "C:/Datasets/BreastCancer/training"
testdir = "C:/Datasets/BreastCancer/testing"
categories = ["benign","malignant"]

def create_data(directory):
    data = []
    for category in categories:
        path = os.path.join(directory, category)
        class_num = categories.index(category)
        
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
            except Exception as e:
                print(f"Error loading image {img}: {str(e)}")
    
    return data

# Create data
print("Loading training data...")
training_data = create_data(traindir)
print("Loading testing data...")
testing_data = create_data(testdir)

# Shuffle data
random.shuffle(training_data)
random.shuffle(testing_data)

# Prepare arrays
def prepare_data(data):
    x = []
    y = []
    for features, label in data:
        x.append(features)
        y.append(label)
    x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x = x / 255.0
    return x, np.array(y)

x_train, y_train = prepare_data(training_data)
x_test, y_test = prepare_data(testing_data)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Build model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

# Callbacks
callbacks = [
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3, monitor='val_loss'),
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
]

# Create and train model
model = build_model()
print(model.summary())

# Train with data augmentation
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=callbacks
)

# Evaluate model
loss, accuracy, auc = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc:.4f}")

# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'])
    
    # Loss plot
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

plot_training_history(history)

# Save model
model.save('breast_cancer_model.h5')
print("Model saved as 'breast_cancer_model.h5'")

# Function for making predictions on new images
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    img = img / 255.0
    
    prediction = model.predict(img)
    probability = prediction[0][0]
    class_name = categories[1] if probability > 0.5 else categories[0]
    
    return class_name, probability

print("\nModel is ready for predictions. Use predict_image(image_path) to classify new images.")
