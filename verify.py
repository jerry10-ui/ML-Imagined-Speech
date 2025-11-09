import numpy as np, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X = np.load(os.path.join(BASE_DIR, "X_segments.npy"), allow_pickle=True)
print("Total NaNs:", np.isnan(X).sum())
print("Any NaN present?:", np.isnan(X).any())
print("Per-sample NaN ratio:", np.isnan(X).sum() / X.size)
