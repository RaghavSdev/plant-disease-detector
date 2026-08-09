import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tf2onnx

data_dir = 'plantVillage/PlantVillage'

print("Initializing Data Generators...")
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    rotation_range=20
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=(200, 200),
    batch_size=64,
    subset='training',
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    data_dir,
    target_size=(200, 200),
    batch_size=64,
    subset='validation',
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_gen.class_indices)
print(f"Classes found ({num_classes}): {list(train_gen.class_indices.keys())}")

# Base model with MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

# Unfreeze last 30 layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("Starting training for up to 30 Epochs...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks
)

# Load best weights
model.load_weights('best_model.h5')
model.save('mango_leaf_disease_model.h5')
print("Model trained and saved as mango_leaf_disease_model.h5!")

print("Converting trained model to ONNX format...")
spec = (tf.TensorSpec((None, 200, 200, 3), tf.float32, name='input_layer'),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path='model.onnx')
print("ONNX conversion completed successfully!")
