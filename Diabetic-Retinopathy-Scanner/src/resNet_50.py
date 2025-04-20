import numpy as np
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from dataExtraction import training_dataset,validation_dataset
from tensorflow.python.keras.layers import Dense, Flatten
from keras import layers, models, Input,preprocessing
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

resNet_50= tf.keras.applications.ResNet50V2(include_top=False,
input_shape=(224,224,3),
pooling='max',classes=5,
weights="imagenet")

for layer in resNet_50.layers:
    layer.trainable = False

inputs =  keras.Input(shape=(224, 224, 3))

norm_layer = keras.layers.Normalization(axis=None)
x = norm_layer(inputs)
x = resNet_50(x,training = False)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(2048, activation = "relu")(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(5,activation = "softmax")(x)
model = keras.Model(inputs, outputs)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  
    factor=0.8,          
    patience=5,   
    mode='auto',             
    verbose=1, 
    cooldown=5      
)

# Early stopping in case the model overfits
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,   
    mode='min',    
    verbose=1,      
    restore_best_weights=True
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001,), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

history =model.fit(
    training_dataset,
    validation_data=validation_dataset,
    callbacks=[reduce_lr, early_stopping],
    epochs=5,
)

for layer in resNet_50.layers[42:190]:
    layer.trainable = True

history =model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=5,
    callbacks=[reduce_lr, early_stopping]
)
    

hist_df = pd.DataFrame(history.history)
hist_df.to_csv('resNet_history.csv')

model.save("new_resNet_50.keras")
