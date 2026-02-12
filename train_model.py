
import os
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
# 1. Environment Setup and Data Acquisition
# --------------------------------------------------------------------------------
# Note: For local execution, ensure kagglehub is installed: pip install kagglehub
import kagglehub

print("Downloading dataset...")
path_raw = kagglehub.dataset_download('hamdallak/the-iqothnccd-lung-cancer-dataset')
DATASET_PATH = os.path.join(path_raw, "The IQ-OTHNCCD lung cancer dataset")
print(f"Data Source Import Complete. Files are at: {DATASET_PATH}")

# --------------------------------------------------------------------------------
# 2. Dataset Configuration and Loading
# --------------------------------------------------------------------------------
BATCH_SIZE = 62
IMAGE_SIZE = 256
EPOCHS = 15
CHANNELS = 3

print("Loading dataset from directory...")
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
print(f"Detected Classes: {class_names}")

# --------------------------------------------------------------------------------
# 3. Data Preprocessing (Splitting and Caching)
# --------------------------------------------------------------------------------
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

print(f"Training Batches: {len(train_ds)}")
print(f"Validation Batches: {len(val_ds)}")
print(f"Testing Batches: {len(test_ds)}")

# Optimization
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Data Augmentation Layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# Rescaling Layer
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1./255),
])

# --------------------------------------------------------------------------------
# 4. Model Architecture
# --------------------------------------------------------------------------------
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)
model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# --------------------------------------------------------------------------------
# 5. Training
# --------------------------------------------------------------------------------
print(f"Starting training for {EPOCHS} epochs...")
history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS,
)

# --------------------------------------------------------------------------------
# 6. Evaluation and Saving
# --------------------------------------------------------------------------------
print("Evaluating on test set...")
scores = model.evaluate(test_ds)
print(f"Test Accuracy: {scores[1]*100:.2f}%")

model_save_path = "model_notebook.h5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
