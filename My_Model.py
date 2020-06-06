#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 00:07:48 2020

@author: rahul
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense,Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

#Creating the model for classification
model_classifier=Sequential()

# will use 4 Blocks of layers of Convolution followed by MaxPooling 

#block 1
model_classifier.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',kernel_initializer='glorot_uniform',activation='relu',input_shape=(224,224,3)))
model_classifier.add(MaxPooling2D(pool_size=(2,2)))

#block 2
model_classifier.add(Conv2D(64,(3,3),activation='relu'))
model_classifier.add(MaxPooling2D(pool_size=(2,2)))

#block 3
model_classifier.add(Conv2D(128,(3,3),activation='relu'))
model_classifier.add(MaxPooling2D(pool_size=(2,2)))
model_classifier.add(Dropout(0.25))

#block 4
model_classifier.add(Conv2D(256,(3,3),activation='relu'))
model_classifier.add(MaxPooling2D(pool_size=(2,2)))


model_classifier.add(Flatten())
model_classifier.add(Dense(64,activation='relu',kernel_initializer='glorot_uniform'))
model_classifier.add(Dropout(0.5))
model_classifier.add(Dense(1,activation='sigmoid'))

# Compilation of model
model_classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



# Data Augmentation
train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
)
test_datagen=image.ImageDataGenerator(rescale=1./255)

#Here, train_datagen and test_datagen is object
training_data=train_datagen.flow_from_directory('/COVID Prediction/Dataset/Train',
                                                batch_size=32,
                                                target_size=(224,224),
                                                class_mode='binary')
testing_data=test_datagen.flow_from_directory('/COVID Prediction/Dataset/Test',
                                              batch_size=32,
                                              target_size=(224,224),
                                             class_mode='binary')


model_classifier.fit_generator(training_data,
                        steps_per_epoch=8,
                        epochs=8,
                        validation_data=testing_data,
                        validation_steps=2)

#Prediction
predict_image=image.load_img('/COVID Prediction/Dataset/test/1_002.png',target_size=(224,224))
predict_image=image.img_to_array(predict_image)
predict_image=np.expand_dims(predict_image,axis=0)
result=model_classifier.predict(predict_image)
print(result)
if result==1:
    print("COVID")
else:
    print("COVID Free")

training_data.class_indices





