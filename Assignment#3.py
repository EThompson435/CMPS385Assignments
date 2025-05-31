import os
import zipfile
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# === STEP 1: UNZIP DATA ===
zip_path = os.path.expanduser('~/Downloads/plantvillage.zip')
extract_path = 'plantvillage_data'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

base_dir = os.path.join(extract_path, 'plantvillage')
train_dir = 'plantvillage_split/train'
test_dir = 'plantvillage_split/test'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# === STEP 2: SPLIT INTO TRAIN/TEST ===
class_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

for class_name in class_folders:
    class_path = os.path.join(base_dir, class_name)
    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(images) == 0:
        print(f"[WARNING] No images found in class folder: {class_name}. Skipping...")
        continue

    img_paths = [os.path.join(class_path, img) for img in images]
    
    train_imgs, test_imgs = train_test_split(img_paths, train_size=0.8, random_state=42)

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    for img in train_imgs:
        shutil.copy(img, os.path.join(train_dir, class_name))
    for img in test_imgs:
        shutil.copy(img, os.path.join(test_dir, class_name))

# === STEP 3: LOAD DATA ===
img_size = 128
batch_size = 32

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1  # 10% of training used for validation
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# === STEP 4: CNN MODEL ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# === STEP 5: TRAIN ===
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[checkpoint]
)

# === STEP 6: PLOT ===
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# === STEP 7: EVALUATE ===
loss, acc = model.evaluate(test_data)
print(f"Test Accuracy: {acc * 100:.2f}%")

# === STEP 8: CLASSIFICATION REPORT ===
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)
true_labels = test_data.classes
class_names = list(test_data.class_indices.keys())

print("Confusion Matrix")
print(confusion_matrix(true_labels, y_pred))
print("\nClassification Report")
print(classification_report(true_labels, y_pred, target_names=class_names))
