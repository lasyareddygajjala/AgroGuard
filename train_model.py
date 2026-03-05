import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os

# Paths
train_dir = 'pest/train'
test_dir = 'pest/test'

print("Loading data...")
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
).flow_from_directory(train_dir, target_size=(224,224), batch_size=32, class_mode='categorical')

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(test_dir, target_size=(224,224), batch_size=32, class_mode='categorical')

print(f"Found {train_gen.samples} training images in {train_gen.num_classes} classes")

# Build model
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training model... (this may take 5-10 minutes)")
model.fit(train_gen, epochs=10, validation_data=test_gen, verbose=1)

# Save model
os.makedirs('model', exist_ok=True)
model.save('model/pest_detector.h5')
print("Model saved as model/pest_detector.h5")