import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Step 1: Set base directory (download & extract manually or via your own code if zipped)
base_dir = 'plantVillage/PlantVillage'  # Correct dataset path

# Step 2: Display a sample image
# Loads and displays one random image from the dataset.
# Helps visually confirm the dataset is correctly loaded.
sample_img_path = None

#Gathers all subdirectories (disease/health classes)
class_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

#Loops through and selects the first image it finds from any class folde
for sample_class in class_dirs:
    image_files = os.listdir(os.path.join(base_dir, sample_class))
    if image_files:
        sample_img_path = os.path.join(base_dir, sample_class, image_files[0])
        break

#Loads and shows one sample image to confirm that the dataset is correctly set up and readable.
#Before training, always verify the structure, format, and quality of your data.
# if sample_img_path:
    # img = image.load_img(sample_img_path)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title(f"Sample: {sample_class}")
    # plt.show()

# Step 3: Data Generators

#All images are resized to a fixed size: 200x200 pixels with 3 color channels (RGB).
image_size = (200, 200)

# What it does:
#rescale=1./255: Pixel values are scaled to [0,1] range, improving training stability.
#validation_split=0.2: Splits data into 80% training, 20% validation.
#shear_range, zoom_range, horizontal_flip: These are data augmentation techniques.

# Why it matters:
#Augmentation increases dataset variability without adding new images
#Prevents overfitting by helping the model generalize better.
#Validation data must not be augmented-it should represent real-world unseen data.

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

#Reads images from folders and automatically labels them.
#Prepares batches of size 32 for efficient training.
#class_mode='categorical': Since this is a multi-class problem, one-hot encoding is used for labels.

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=image_size,
    batch_size=32,
    subset='training',
    seed=42,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    base_dir,
    target_size=image_size,
    batch_size=32,
    subset='validation',
    seed=42,
    class_mode='categorical'
)

#Retrieves number of output classes from the folder names.
num_classes = len(train_generator.class_indices)
print(f"Total classes in dataset: {num_classes}")

# Step 4: CNN Model

#A CNN uses convolutional layers to learn spatial hierarchies of features.
#Convolutional Layers detect patterns like edges, corners, textures.
#Pooling Layers reduce spatial dimensions.
# Fully Connected Layers (Dense) classify based on extracted features.

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(num_classes, activation='softmax')
])

#adam: Combines the advantages of RMSProp and SGD. Great for deep learning.
#categorical_crossentropy: Used for multi-class classification with softmax outputs.
#accuracy: Tracks how well the model is doing.

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
#What it does:
#Monitors validation loss.
#If the loss does not improve for 3 consecutive epochs, training stops early.

#Why it matters:
#Prevents overfitting.
#Saves training time and resources.
callback = EarlyStopping(monitor='val_loss', patience=3)

#What it does:
#Trains the model for up to 30 epochs.
#Uses real-time data loading and augmentation from generators.
#Uses EarlyStopping to avoid wasting time.

#Internally:
#In each epoch:
#A batch of 32 images is fetched.
#CNN updates weights using backpropagation.
#Metrics (loss and accuracy) are calculated for both training and validation.

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[callback]
)

# Step 6: Save the trained model
model.save("mango_leaf_disease_model.h5")
print("Model saved as mango_leaf_disease_model.h5")
