import tensorflow as tf
from tensorflow import keras
from keras import layers, models, Input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from dataExtraction import test_dataset

try:
    model = keras.models.load_model('resNet_model.keras')
    score = model.evaluate(test_dataset,batch_size=32,
    verbose='auto',)
except Exception as e:
    print(e)