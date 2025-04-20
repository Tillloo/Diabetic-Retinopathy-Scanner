import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, preprocessing
import os
import matplotlib.pyplot as plt

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Parent folder
DATA_DIR = os.path.join(BASE_DIR, 'data')  # 'data' folder
TRAIN_DIR = os.path.join(DATA_DIR, 'train')  # 'train' folder
TEST_DIR = os.path.join(DATA_DIR,'test')

# Check if dataset exists
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

# Read labels
LABELS_FILE = os.path.join(DATA_DIR, 'new_trainLabels.csv')  # Assuming the CSV is in 'data'
if not os.path.exists(LABELS_FILE):
    raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")

labels = pd.read_csv(LABELS_FILE, dtype=str)  # Read labels
labels['image'] = labels['image'] + '.jpeg'  # Append '.jpeg' extension

OLD_LABELS_FILE = os.path.join(DATA_DIR, 'trainLabels.csv')  
old_labels = pd.read_csv(OLD_LABELS_FILE, dtype=str)
old_labels['image'] = old_labels['image'] + '.jpeg'
# Filter labels for images that exist
fnames = os.listdir(TRAIN_DIR)
mask = labels['image'].isin(fnames)
filtered_labels = labels[mask].sort_values('image')

# Dataset Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Create a TensorFlow Dataset for training and validation data
# Initalize the training dataset
training_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123,
    labels=filtered_labels['level'].astype(int).to_list(),
    label_mode='int',
    validation_split=0.25,
    subset="training",
    color_mode='rgb'
)

# Initialize the validation data set
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123,
    labels=filtered_labels['level'].astype(int).to_list(),
    label_mode='int',
    validation_split=0.25,
    subset="validation",
    color_mode='rgb'
)

tfnames = os.listdir(TEST_DIR)
tmask = old_labels['image'].isin(tfnames)
tfiltered_labels = old_labels[tmask].sort_values('image')


test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    labels=tfiltered_labels['level'].astype(int).to_list(),
    label_mode='int',
    color_mode='rgb'
)


# Data Augmentation Pipeline
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),  
    layers.RandomRotation(0.2),                 
    layers.RandomZoom(0.2),                       
    layers.RandomBrightness(factor=0.1),        
    layers.RandomContrast(factor=0.2),
    layers.Lambda(lambda x: tf.image.random_saturation(x, lower=0.75, upper=1.5)),
    layers.Lambda(lambda x: tf.image.random_hue(x, max_delta=0.15)),                
])

# Apply data augmentation to the training dataset
training_dataset = training_dataset.map(lambda x, y: (data_augmentation(x), y))
