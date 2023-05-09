# -*- coding: utf-8 -*-
'''
Author: Allana Pracuccio Freitas
Date: 01/01/2023
Title of Program: Weed segmentation using U-Net architecture with transfer learning in multispectral images obtained by drone
Code version: 1.0
UNESP GRADUATE PROGRAM ON CARTOGRAPHIC SCIENCES
'''

"""
*** INSTALL LIBRARIES ***
"""

import sys
sys.path

#TensorFlow: is an open source software library for high performance numerical computation
pip install tensorflow
#Patchify: a library that helps you split image into small, overlappable patches, and merge patches back into the original image
pip install patchify
#Segmentation Models: image segmentation models with pre-trained backbones with Keras
pip install segmentation-models==1.0.1
#Tifffile: read and write TIFF files
pip install tifffile


import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
import matplotlib.pyplot as plt
import segmentation_models as sm
sm.set_framework('tf.keras')
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD

#Check Tensorflow Version
tf.__version__

"""
*** LOSS FUNCTION *** 
Dice Loss
"""
#Credits: https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution/blob/master/losses.py
#Credits: https://www.kaggle.com/questions-and-answers/330689

from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

"""
*** METRIC ***
F1 Score 
"""
#Credits: https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

import tensorflow.keras.backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

"""
*** IMPORT DATASET *** 
"""

#Import Orthomosaic and Binary Mask
orto_image = tiff.imread('D:/Allana/Mestrado/image/Sequoia_10cm_NDVI_REC.tif')
orto_mask = tiff.imread ('D:/Allana/Mestrado/mask/binary_mask_mult.tif')

print(orto_image.shape)
print(orto_mask.shape)
print(orto_image.dtype)

#Generation of Patches (64x64) 
def make_tiles(_img, _tilesize):
  tiles=[]
  _shape = _img.shape
  nrl = _shape[0]//_tilesize
  nrc = _shape[1]//_tilesize
  if len(_shape) >2:
    l=0
    for i in range(nrl):
      c=0
      for j in range(nrc):
        tiles.append(_img[l:l+_tilesize,c:c+_tilesize,:])
        c = c+_tilesize
      l=l+_tilesize
  else:
    l=0
    for i in range(nrl):
      c=0
      for j in range(nrc):
        tiles.append(_img[l:l+_tilesize,c:c+_tilesize])
        c = c+_tilesize
      l=l+_tilesize
  return np.array(tiles)

masks = make_tiles(orto_mask, 64)
masks = np.expand_dims(masks, -1)

images = make_tiles(orto_image, 64)
images = np.stack((images,)*3, axis=-1)

#Show Array and Digital Number (DN)
print("Images: ", images.shape)
print("Masks: ", masks.shape)
print("DN of the binary mask pixels: ", np.unique(masks)) #Background = 0 e Planta Daninha = 1.

#Delete patches so classes (background and weeds) are not unbalanced
a=[]
for i in range(masks.shape[0]):
  if np.unique(masks[i,]).shape[0]==1:
    a.append(i)

def limpa_lixo (a,image_mask):
  aux = np.delete(image_mask, a, axis=0)
  return aux

masks = limpa_lixo(a,masks)
images = limpa_lixo(a,images)

t=110
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(images[t])
plt.subplot(122)
plt.imshow(masks[t])
plt.show()

"""
*** DEFINE THE MODEL ***
"""

BACKBONE = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE)
images1=preprocess_input1(images)
print('\nTotal Dataset')
print(images1.shape)

#Define Test Data = 30%
X_train, X_test, y_train, y_test = train_test_split(images1, masks, test_size = 0.30, random_state = 42)
print('\nTrain Dataset')
print(X_train.shape)
print('\nTest Dataset')
print(X_test.shape)

#Sanity Check (view few images)
#It is important to check if the patches match with their respective binary mask before train
import random
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(X_train[image_number, :,:, 0], cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (64, 64)), cmap='gray')
plt.show()

"""
*** DATA AUGMENTATION *** 
"""
#New generator with rotation and shear where interpolation that comes with rotation and shear are thresholded in masks
#This gives a binary mask rather than a mask with interpolated values

seed=24 #Use the same number for images and masks, as both went through this process. The only way to check is to keep the seed the same
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')

mask_data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect',
                     preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again.

image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_data_generator.fit(X_train, augment=True, seed=seed)

image_generator = image_data_generator.flow(X_train, seed=seed)
valid_img_generator = image_data_generator.flow(X_test, seed=seed)

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_data_generator.fit(y_train, augment=True, seed=seed)
mask_generator = mask_data_generator.flow(y_train, seed=seed)
valid_mask_generator = mask_data_generator.flow(y_test, seed=seed)

def my_image_mask_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

my_generator = my_image_mask_generator(image_generator, mask_generator)

validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)

#Sanity Check for Data Augmentation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

x = image_generator.next()
y = mask_generator.next()
for i in range(0,1):
    image = x[i]
    mask = y[i]
    plt.subplot(1,2,1)
    plt.imshow(image[:,:,0], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask[:,:,0])
    plt.show()

"""
*** MODEL TRANSFER LEARNING *** 
ImageNet
"""

#Backbone: ResNet-34; Dropout: 20%; Transfer Learning: ImageNet; Otmizer: Adam; Loss: Dice Loss; Metric: F1 Score, IOU, Accuracy
model = sm.Unet(BACKBONE, encoder_weights='imagenet', encoder_freeze=True)
'''sgd = SGD(learning_rate=0.0001)'''
'''model.compile(optimizer='sgd', loss=dice_loss, metrics=[f1, sm.metrics.iou_score, 'accuracy'])'''

'''model = sm.Unet(BACKBONE, encoder_weights=None, encoder_freeze=True)'''
model.compile(optimizer='Adam', loss=dice_loss, metrics=[f1, sm.metrics.iou_score, 'accuracy'])
model.trainable = True
print(model.summary())

batch_size=8
steps_per_epoch = (len(X_train))//batch_size

#Fit Model on Training Data
history = model.fit(my_generator, 
                    validation_data=validation_datagen, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_steps=steps_per_epoch, 
                    epochs=10)

"""
*** PLOT THE RESULTS *** 
"""

#Convert Test Dataset to Float32
X_testX = X_test.astype(np.float32)
y_testy = y_test.astype(np.float32)

# Evaluate the model on the test data using evaluate
print("Evaluate on test data")
results = model.evaluate(X_testX, y_testy, batch_size=8)
print("test loss, test acc:", results)

#Plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#ACCURACY 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#IOU
acc = history.history['iou_score']
val_acc = history.history['val_iou_score']
plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and Validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()

#Predicting the Test set results IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

#F1 SCORE
acc = history.history['f1']
val_acc = history.history['val_f1']
plt.plot(epochs, acc, 'y', label='Training F1 Score')
plt.plot(epochs, val_acc, 'r', label='Validation F1 Score')
plt.title('Training and Validation F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.show()

#Plot the Results
test_img_number = random.randint(0, len(X_test)-1)
test_img = X_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)
ground_truth=y_test[test_img_number]
prediction = model.predict(test_img_input)
prediction = prediction[0,:,:,0]

print("Patch number:", test_img_number)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()