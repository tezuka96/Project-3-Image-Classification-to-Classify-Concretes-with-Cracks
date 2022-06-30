# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:50:21 2022

@author: Ryzen

Project 3
Image Classification to Classify Concretes with Cracks
"""

#1. Import packages
import matplotlib.pyplot as plt
import numpy as np
import os, cv2, datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, layers
import pathlib as path

#2. Data preparation
root_path = r"C:\Users\Ryzen\Documents\tensorflow\GitHub\Project-3-Image-Classification-to-Classify-Concretes-with-Cracks\Concrete Crack Images for Classification"
data_dir = path.Path(root_path)

SEED = 12345
IMG_SIZE = (160,160)
BATCH_SIZE = 32

train_dataset = keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.3, subset='training', seed=SEED, shuffle=True,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_dataset = keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.3, subset='validation', seed=SEED, shuffle=True,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE)

#%%
#Further split validation dataset into validation-test split
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

#%%
# Create prefetch dataset for all 3 splits
AUTOTUNE = tf.data.AUTOTUNE
pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_validation = validation_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

#Data is prepared mostly....

#%%
#3. Create data augmentation pipeline
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

#%%
for images,labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')

#%%
#Create a layer for data preprocessing
preprocess_input = applications.mobilenet_v2.preprocess_input
#Create the base model by using MobileNetV2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

#Apply layer freezing
for layer in base_model.layers[:100]:
    layer.trainable = False
    
base_model.summary()

#%%
#Create classification layer
class_names = train_dataset.class_names
nClass = len(class_names)

global_avg_pooling = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(nClass, activation='softmax')

#%%
#Use functional API to construct the entire model
inputs = keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x)
x = global_avg_pooling(x)
outputs = output_layer(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

#Use in iPython console 
tf.keras.utils.plot_model(model, show_shapes=True)

#%%
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
loss = keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

#%%
#Perform model training
EPOCHS = 100
base_log_path = r"C:\Users\Ryzen\Documents\tensorflow\GitHub\Project-3-Image-Classification-to-Classify-Concretes-with-Cracks\tb_logs"
log_path= os.path.join(base_log_path, datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '__Project_3')
tb = keras.callbacks.TensorBoard(log_dir=log_path)
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)

history = model.fit(pf_train, validation_data=pf_validation, epochs=EPOCHS, callbacks=[es,tb])

#%%
#Deploy model to make prediction
test_loss, test_accuracy = model.evaluate(pf_test)
print('---------------------Test Result---------------------')
print(f'Loss = {test_loss}')
print(f'Accuracy = {test_accuracy}')

#%%
image_batch, label_batch = pf_test.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
class_predictions = np.argmax(predictions, axis=1)

#%%
#7. Show some prediction results
plt.figure(figsize=(10,10))

for i in range(4):
    axs = plt.subplot(2,2,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    current_prediction = class_names[class_predictions[i]]
    current_label = class_names[label_batch[i]]
    plt.title(f"Prediction: {current_prediction}, Actual: {current_label}")
    plt.axis('off')
    
save_path = r"C:\Users\Ryzen\Documents\tensorflow\GitHub\Project-3-Image-Classification-to-Classify-Concretes-with-Cracks\img"
plt.savefig(os.path.join(save_path,"result.png"),bbox_inches='tight')
plt.show()

#%%
from numba import cuda 
device = cuda.get_current_device()
device.reset()
