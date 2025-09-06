
# ------------------------------------------------
# Import required modules and set seeds for reproducibility
# ------------------------------------------------
import os
import random
import shutil
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Set random seeds to ensure that the results are reproducible across different runs.
# Any operation with a stochastic element (like data shuffling or model weight
# initialization) will produce the same output every time the code is executed.
SEED = 42
os.environ["SEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------
# Verify GPU availability
# ------------------------------------------------
# This check confirms if a GPU is available and recognized by TensorFlow.
# Training deep learning models is computationally intensive, and using a GPU
# can accelerate the process by orders of magnitude compared to a CPU.
print("\n--- GPU Verification ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Use only the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"GPU Device: {gpus[0].name} is available and configured.")
    except RuntimeError as e:
        print("RuntimeError:", e)
else:
    print("No GPU detected. The notebook will run on CPU.")
print("--------------------------\n")


# @title 1.2. The BDPapayaLeaf Dataset: Acquisition and Profiling
# ------------------------------------------------
# Download and Extract the Dataset
# ------------------------------------------------
# The dataset is downloaded from its Mendeley Data repository using a direct link.
# The 'wget' command fetches the zip archive, and the '-q' flag suppresses download progress output.
# The 'unzip' command extracts the contents into the Colab environment.
print("--- Dataset Acquisition ---")
# ------------------------------------------------
# Profile the Dataset (Exploratory Data Analysis)
# ------------------------------------------------
# The dataset documentation indicates that the images are stored in the
# 'BDPapayaLeaf/Original Images' directory. We will verify this structure
# and count the number of images in each class subdirectory.
original_data_dir = pathlib.Path('BDPapayaLeaf/Original Images')

# List the class names by getting the names of the subdirectories.
class_names = sorted([item.name for item in original_data_dir.glob('*') if item.is_dir()])
print(f"Found {len(class_names)} classes: {class_names}\n")

# Count images per class and store in a dictionary.
image_counts = {class_name: len(list(original_data_dir.glob(f'{class_name}/*.jpg'))) for class_name in class_names}

# Create a pandas DataFrame for a clean, tabular display of the class distribution.
df_counts = pd.DataFrame(list(image_counts.items()), columns=['Class Name', 'Image Count'])
total_images = df_counts['Image Count'].sum()
total_row = pd.DataFrame()
df_counts = pd.concat([df_counts, total_row], ignore_index=True)

print("--- BDPapayaLeaf Dataset Class Distribution ---")
print(df_counts.to_string(index=False))
print("---------------------------------------------\n")

# ------------------------------------------------
# Visualize Sample Images from Each Class
# ------------------------------------------------
# This provides a qualitative look at the dataset, helping to understand the
# visual characteristics and challenges of the classification task.
print("--- Sample Images ---")
plt.figure(figsize=(12, 8))
for i, class_name in enumerate(class_names):
    ax = plt.subplot(2, 3, i + 1)
    # Get the first image from each class directory for display.
    image_path = next((original_data_dir / class_name).glob('*.jpg'))
    image = mpimg.imread(image_path)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis("off")
plt.tight_layout()
plt.show()
print("---------------------\n")

# @title 1.3. A Reproducible Data Splitting Protocol
# ------------------------------------------------
# This cell implements a robust, manual data splitting strategy by creating
# separate directories for train, validation, and test sets. This prevents
# data leakage and ensures a fair and reproducible evaluation.
# ------------------------------------------------
print("--- Data Splitting Protocol ---")
split_base_dir = pathlib.Path('papaya_data_split')

# Remove the directory if it already exists to ensure a clean split.
if split_base_dir.exists():
    shutil.rmtree(split_base_dir)
os.makedirs(split_base_dir, exist_ok=True)

# Define split ratios
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Create train, validation, and test directories
train_dir = split_base_dir / 'train'
val_dir = split_base_dir / 'validation'
test_dir = split_base_dir / 'test'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Iterate over each class to perform a stratified split
for class_name in class_names:
    print(f"Splitting class: {class_name}")
    # Create class-specific subdirectories in each split folder
    os.makedirs(train_dir / class_name, exist_ok=True)
    os.makedirs(val_dir / class_name, exist_ok=True)
    os.makedirs(test_dir / class_name, exist_ok=True)

    # Get all image file paths for the current class
    class_dir = original_data_dir / class_name
    image_files = list(class_dir.glob('*.jpg'))
    
    # Shuffle the files with a fixed seed for reproducibility
    random.shuffle(image_files)
    
    # Calculate split indices
    n_images = len(image_files)
    train_split_idx = int(n_images * train_ratio)
    val_split_idx = int(n_images * (train_ratio + val_ratio))
    
    # Partition the file list
    train_files = image_files[:train_split_idx]
    val_files = image_files[train_split_idx:val_split_idx]
    test_files = image_files[val_split_idx:]
    
    # Copy files to their new destination directories
    for file in train_files:
        shutil.copy(file, train_dir / class_name)
    for file in val_files:
        shutil.copy(file, val_dir / class_name)
    for file in test_files:
        shutil.copy(file, test_dir / class_name)

print("\nData splitting complete.")
print(f"Total training images: {len(list(train_dir.glob('*/*.jpg')))}")
print(f"Total validation images: {len(list(val_dir.glob('*/*.jpg')))}")
print(f"Total test images: {len(list(test_dir.glob('*/*.jpg')))}")
print("-----------------------------\n")# @title 1.4. High-Performance Data Ingestion Pipelines
# ------------------------------------------------
# Create tf.data.Dataset objects for each split using a Keras utility.
# This utility automatically infers class labels from the directory structure.
# ------------------------------------------------
print("--- Creating Data Pipelines ---")
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Create datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    interpolation='bilinear',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    color_mode="rgb"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    interpolation='bilinear',
    batch_size=BATCH_SIZE,
    shuffle=False,
    color_mode="rgb"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    interpolation='bilinear',
    batch_size=BATCH_SIZE,
    shuffle=False,
    color_mode="rgb"
)

print("Data pipelines created successfully.\n")

# ✅ Save class names BEFORE wrapping
class_names = train_ds.class_names

# Optimize
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Verify
print("--- Pipeline Verification ---")
print(f"Class names inferred by the pipeline: {class_names}")

for image_batch, labels_batch in train_ds.take(1):
    print(f"\nShape of one image batch: {image_batch.shape}")
    print(f"Shape of one label batch: {labels_batch.shape}")
    print(f"Data type of image batch: {image_batch.dtype}")
    print(f"Data type of label batch: {labels_batch.dtype}")

print("\n--- Creating Data Pipelines ---")

# --- Constants for Data Loading ---
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# Create TensorFlow Dataset objects from the directories
# These act as generators that feed data to the model in batches.

train_generator = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True,
    color_mode="rgb"
)

val_generator = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False,
    color_mode="rgb"
)

test_generator = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False,
    color_mode="rgb"
)

print("Data pipelines created successfully.")


# --- Pipeline Verification ---
print("\n--- Pipeline Verification ---")
print(f"Class names inferred by the pipeline: {train_generator.class_names}")

# Let's look at a single batch to verify shapes
for images, labels in train_generator.take(1):
    print(f"Shape of one image batch: {images.shape}")
    print(f"Shape of one label batch: {labels.shape}")
    print(f"Data type of image batch: {images.dtype}")
    print(f"Data type of label batch: {labels.dtype}")
    
# STEP 3: MODEL ARCHITECTURE DEFINITION & TRANSFER LEARNING
# ==========================================================
print("\n--- Step 3: Model Architecture Definition & Transfer Learning ---")

import tensorflow as tf
from tensorflow.keras.applications import VGG16, EfficientNetB3, MobileNetV3Large
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- Constants ---
IMG_SIZE = (224, 224)
NUM_CLASSES = len(class_names)
LEARNING_RATE = 0.0001
EPOCHS = 25 # Increased for potentially better convergence

def build_vgg16_model(num_classes):
    """
    Builds a VGG16 model for transfer learning.

    Args:
        num_classes (int): The number of output classes.

    Returns:
        A compiled Keras Model.
    """
    # Load the VGG16 base model with pre-trained ImageNet weights
    # include_top=False means we don't include the final fully-connected layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))

    # Freeze the layers of the base model so they are not updated during the first training phase
    base_model.trainable = False

    # Add a custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x) # Dropout for regularization
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("VGG16 model built successfully.")
    return model

# Instantiate the model
vgg16_model = build_vgg16_model(NUM_CLASSES)

# Display the model architecture
print("\n--- VGG16 Model Summary ---")
vgg16_model.summary()

# STEP 4: MODEL TRAINING
# ======================
print("\n--- Step 4: Model Training (VGG16) ---")

# We will use EarlyStopping to prevent overfitting. It stops training when the
# validation loss has not improved for a certain number of epochs ('patience').
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True # Restores model weights from the epoch with the best value
)

print("Starting training for VGG16 model...")
# Note: Since no GPU is detected, this will be slow.
# For a full run, 25 epochs might take a significant amount of time.
# You can reduce EPOCHS for a quick test.

history_vgg16 = vgg16_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

vgg16_model.save("papaya_disease_vgg16.keras")

print("VGG16 model training complete.")

# --- Visualize Training History ---
def plot_history(history, model_name):
    """
    Plots the training and validation accuracy and loss.

    Args:
        history: The training history object from model.fit().
        model_name (str): Name of the model for titles.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)


    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title(f'Training and Validation Loss for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

print("\n--- Visualizing VGG16 Training History ---")
plot_history(history_vgg16, "VGG16")
