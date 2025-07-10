import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm

# Paths
DATA_DIR = os.path.join("dataset")  # 'FAKE' and 'REAL' are inside here
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Parameters
n_mfcc = 26
window_size = 5
batch_size = 32
epochs = 10

def extract_mfcc_windows(file_path, label):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
        if mfcc.shape[0] < window_size:
            return [], []
        windows = [mfcc[i:i+window_size] for i in range(len(mfcc) - window_size + 1)]
        labels = [label] * len(windows)
        return windows, labels
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], []

# Step 1: Load and extract features
X, y = [], []
for label in ['REAL', 'FAKE']:
    folder = os.path.join(DATA_DIR, label)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Expected folder not found: {folder}")
    print(f"Loading {label} files from {folder}...")
    for filename in tqdm(os.listdir(folder), desc=f"Processing {label}"):
        if filename.lower().endswith(('wav', 'mp3', 'ogg')):
            file_path = os.path.join(folder, filename)
            windows, labels = extract_mfcc_windows(file_path, label)
            X.extend(windows)
            y.extend(labels)

X = np.array(X)
y = np.array(y)

print(f"✅ Total samples: {len(X)}")

# Step 2: Preprocessing
X_flat = X.reshape(-1, n_mfcc)

scaler = StandardScaler()
X_scaled_flat = scaler.fit_transform(X_flat)

X_scaled = X_scaled_flat.reshape(-1, window_size, n_mfcc)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Step 4: Model Building
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(window_size, n_mfcc)),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Step 5: Train
model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    verbose=1
)

# Step 6: Save model and preprocessors
model.save(os.path.join(MODEL_DIR, 'model.keras'))

with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"\n🎉 Model training complete. Files saved in '{MODEL_DIR}':")
print(" - model.keras")
print(" - scaler.pkl")
print(" - label_encoder.pkl")
