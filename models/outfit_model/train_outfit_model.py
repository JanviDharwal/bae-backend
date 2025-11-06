import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============== CONFIGURATION ==============
CSV_PATH = "BAE--Bringing-Aesthetics-to-Emotions\data\outfit_data\Fashion Product Images (Small)\filtered_top_bottom.csv"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SAVE_DIR = "BAE--Bringing-Aesthetics-to-Emotions\models\outfit_model"
os.makedirs(SAVE_DIR, exist_ok=True)
# ===========================================

# Load filtered CSV
df = pd.read_csv(CSV_PATH)

# Encode labels (Topwear=0, Bottomwear=1) and convert to string
df['label'] = df['subCategory'].apply(lambda x: '0' if x == 'Topwear' else '1') # Convert to string

# Split into train/test
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42) # Stratify by string labels


# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_full_path', # Use image_full_path
    y_col='label',
    target_size=IMG_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col='image_full_path', # Use image_full_path
    y_col='label',
    target_size=IMG_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============== MODEL DEFINITION ==============
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
for layer in base_model.layers[:-100]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
checkpoint = ModelCheckpoint(os.path.join(SAVE_DIR, "mobilenetv2_top_bottom.h5"),
                             save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint, early_stop])

# Evaluate model
val_loss, val_acc = model.evaluate(val_gen)
print(f"\nValidation Accuracy: {val_acc*100:.2f}%")

# Plot training performance
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("MobileNetV2 - Topwear vs Bottomwear Accuracy")
plt.show()