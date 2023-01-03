# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 12:35:45 2022

@author: romain corseri
"""

import os
from os import path
import random
import math

from tensorflow.keras.preprocessing.image import load_img,img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend, models
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, RandomFlip, RandomRotation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical #Image generator used for transformation to categorical
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Define directory and bird species for the classification task
BASE_DIR = "content"

TRAIN_DIR = os.path.join(BASE_DIR, 'train')
#TRAIN_DIR = os.path.join("content/train/")
print(TRAIN_DIR)
VALIDATION_DIR = os.path.join(BASE_DIR, 'valid')
TEST_DIR = os.path.join(BASE_DIR, 'test')

data_generator = ImageDataGenerator(rescale=1./255, )

classes = [
  'AFRICAN FIREFINCH',
  'ALBATROSS',
  'ALTAMIRA YELLOWTHROAT',
  'APAPANE',
  'BALTIMORE ORIOLE',
  'BALD IBIS',
  'BAR-TAILED GODWIT',
  'CAPE LONGCLAW',
  'CAPPED HERON',
  'DOUBLE BARRED FINCH',
  'DOUBLE BRESTED CORMARANT',
  'GREAT GRAY OWL',
  'HELMET VANGA',
  'HORNBILL',
  'HOUSE SPARROW',
  'INLAND DOTTEREL',
  'JANDAYA PARAKEET',
  'KIWI',
  'LONG-EARED OWL',
  'NORTHERN FLICKER'
  ]
n_classes = len(classes)
print('Number of classes:', n_classes)


#Load an image and determine image shape for analysis.

fig = plt.figure(figsize=(18,9))

for i, bird in enumerate(classes):
  IMAGE = load_img(os.path.join(TRAIN_DIR, f"{bird}/001.jpg"))
  ax = fig.add_subplot(4, 5, i+1)
  ax.set_title(bird)
  imgplot = plt.imshow(IMAGE)
  plt.axis("off")

plt.show()

IMAGEDATA = img_to_array(IMAGE)
SHAPE = IMAGEDATA.shape
print('Figures are ', SHAPE)



target_size = (100,100)
train_data = data_generator.flow_from_directory(TRAIN_DIR, classes=classes, target_size=target_size)
validation_data = data_generator.flow_from_directory(VALIDATION_DIR, classes=classes, target_size=target_size)
test_data = data_generator.flow_from_directory(TEST_DIR, classes=classes, target_size=target_size)

# Build convolutional neural network using Keras implementation

backend.clear_session()

model = Sequential(name='bird_classifier')#, RandomFlip('horizontal'), RandomRotation(0.2))


# Convert the 100x100 image into a flat vector of 100x100 = 100000 values
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
          
# Create a "hidden" layer with 256 neurons and apply the ReLU non-linearity
model.add(Dense(256,activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.35))
          
# Create another hidden layer with 256 neurons
model.add(Dense(256,activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.35))
model.add(Dense(20,activation='softmax',kernel_initializer='glorot_normal'))
# Create an "output layer" with 20 neurons

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True),
    loss="categorical_crossentropy",
    metrics=["accuracy"])

model.build(input_shape=(None, target_size[0], target_size[1], 3)) # (batch_size, width, height, channels)
model.summary()

history = model.fit(
    train_data,
    batch_size=128,
    epochs = 20,
    validation_data = validation_data,
    verbose = 1,
    callbacks = [
                 EarlyStopping(monitor='val_accuracy', patience = 5, restore_best_weights = True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)])

#plot accuracy vs epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss values vs epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate against test data.
scores = model.evaluate(test_data, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])