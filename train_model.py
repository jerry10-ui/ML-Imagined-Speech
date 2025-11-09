import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# -------------------- LOAD DATA --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X = np.load(os.path.join(BASE_DIR, "X_segments.npy"), allow_pickle=True)
y = np.load(os.path.join(BASE_DIR, "y_segments.npy"), allow_pickle=True)

print(f"✅ Loaded data: X={X.shape}, y={y.shape}")

# -------------------- PREPROCESS --------------------
# Shuffle to ensure random distribution
X, y = shuffle(X, y, random_state=42)

# Normalize (reliable normalization)
X = (X - np.mean(X)) / np.std(X)

# Light noise augmentation
noise = np.random.normal(0, 0.01, X.shape)
X_aug = np.concatenate([X, X + noise])
y_aug = np.concatenate([y, y])

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_aug)
label_classes = label_encoder.classes_
np.save(os.path.join(BASE_DIR, "label_classes.npy"), label_classes)

# Split into train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(X_aug, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ⚠️ DO NOT ADD extra channel dimension here — Conv1D expects (39, 256)
print(f"📊 Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# -------------------- MODEL --------------------
model = models.Sequential([
    layers.Conv1D(64, 3, activation='relu', input_shape=(39, 256)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),

    layers.Conv1D(128, 3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),

    layers.Conv1D(256, 3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling1D(),

    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.4),
    layers.Dense(len(label_classes), activation='softmax')
])

# -------------------- COMPILE --------------------
optimizer = optimizers.Adam(learning_rate=3e-4)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# -------------------- CALLBACKS --------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6)
]

# -------------------- TRAIN --------------------
print("\n🚀 Training model (this may take a few minutes)...\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# -------------------- EVALUATE --------------------
print("\n🧠 Evaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# -------------------- SAVE MODEL & HISTORY --------------------
np.save(r"C:\Coding\ML Imagined Speech\history.npy", history.history)
model.save(os.path.join(BASE_DIR, "eeg_model.h5"))
print("💾 Model saved as eeg_model.h5")

