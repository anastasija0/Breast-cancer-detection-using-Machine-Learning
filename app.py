import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import numpy as np 
import os
import cv2
import random
from tqdm import tqdm


traindir = "C:/Datasets/BreastCancer/training"
testdir = "C:/Datasets/BreastCancer/testing"
categories = ["benign","malignant"]

IMG_SIZE = 120

#trening
training_data = []

def create_training_data():
    for category in categories:  # benign i malignant

        path = os.path.join(traindir,category)  
        class_num = categories.index(category) #benign - 0 i malignant - 1 

        for img in tqdm(os.listdir(path)):  
            try:
                img_niz = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  
                new_niz = cv2.resize(img_niz, (IMG_SIZE, IMG_SIZE))  
                training_data.append([new_niz, class_num]) 
            except Exception as e:
                pass

create_training_data()
random.shuffle(training_data)

x_train = [] #slike
y_train = [] #label

for features,label in training_data:
    x_train.append(features)
    y_train.append(label)

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_train= x_train/255

#testiranje
testing_data = []
def create_testing_data():
    for category in categories:  # benign i malignant

        path = os.path.join(testdir,category)  
        class_num = categories.index(category) #benign - 0 i malignant - 1 

        for img in tqdm(os.listdir(path)):  
            try:
                img_niz = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  
                new_niz = cv2.resize(img_niz, (IMG_SIZE, IMG_SIZE))  
                testing_data.append([new_niz, class_num]) 
            except Exception as e:
                pass

create_testing_data()
random.shuffle(testing_data)

x_test = [] 
y_test = [] 

for features,label in testing_data:
    x_test.append(features)
    y_test.append(label)

x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test= x_test/255

#model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=3, validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test)  
print(f"loss je :{loss}, tacnost je:{accuracy}")

model.save('cancer detection.model')
