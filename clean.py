# clean_segments.py
import numpy as np, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, "X_segments.npy")

X = np.load(path, allow_pickle=True)
print(f"🔍 Loaded X_segments.npy | Shape: {X.shape}")

nan_count = np.isnan(X).sum()
print(f"Before cleaning: {nan_count} NaNs")

# ✅ Replace NaNs using per-sample mean normalization
for i in range(len(X)):
    sample = X[i]
    if np.isnan(sample).any():
        mean_val = np.nanmean(sample)
        if np.isnan(mean_val):  # all NaN in this sample
            X[i] = np.zeros_like(sample)
        else:
            X[i] = np.nan_to_num(sample, nan=mean_val)

# ✅ Optional normalization (zero mean, unit std per sample)
X = (X - np.mean(X, axis=(1,2), keepdims=True)) / (np.std(X, axis=(1,2), keepdims=True) + 1e-8)

np.save(path, X)
print("✅ Cleaned & normalized X_segments.npy and saved back.")
print("NaNs remaining:", np.isnan(X).sum())
print("Example range after clean:", X.min(), "to", X.max())
