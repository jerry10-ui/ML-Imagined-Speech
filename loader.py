import os
import glob
import mne
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = input("Enter the path to your EEG dataset folder: ").strip()
edf_files = glob.glob(os.path.join(dataset_path, "*.edf"))

if not edf_files:
    raise FileNotFoundError("⚠️ No .edf files found in the specified folder!")

print(f"✅ Found {len(edf_files)} EEG files.")

X = []
y = []
min_channels = None
min_samples = None

# ------------------------------------------------
# 1️⃣ First pass: find smallest common shape
# ------------------------------------------------
for file in edf_files:
    raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
    data = raw.get_data()
    ch, t = data.shape
    if min_channels is None or ch < min_channels:
        min_channels = ch
    if min_samples is None or t < min_samples:
        min_samples = t

print(f"🧩 Common shape -> Channels: {min_channels}, Samples: {min_samples}")

# ------------------------------------------------
# 2️⃣ Second pass: process and reshape all EEGs
# ------------------------------------------------
for file in edf_files:
    print(f"📂 Loading file: {os.path.basename(file)}")
    raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
    raw.filter(1., 40., fir_design='firwin', verbose=False)
    data = raw.get_data()[:min_channels, :min_samples]

    # Normalize per channel
    data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)

    # Extract label from filename
    filename = os.path.basename(file)
    label = filename.split('_')[-1].replace('.edf', '')

    X.append(data)
    y.append(label)

# Convert to NumPy array (now all shapes are same)
X = np.array(X)
y = np.array(y)

y = np.array([str(label) for label in y], dtype=str)

# Print clean label summary
unique_labels = set(map(str, np.unique(y)))
print(f"✅ Final data shape: {X.shape} | Labels: {unique_labels}")

np.save(os.path.join(BASE_DIR, "X.npy"), X)
np.save(os.path.join(BASE_DIR, "y.npy"), y)

print("💾 Saved X.npy and y.npy successfully!")
