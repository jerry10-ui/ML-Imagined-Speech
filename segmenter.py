import os
import numpy as np

# ✅ Always use the ML Project folder as the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def clean_labels(y):
    """Ensure all labels are plain Python strings."""
    cleaned = []
    for label in y:
        # Convert np.str_ → str
        if isinstance(label, np.str_):
            label = str(label)
        # Handle weird repr cases like "np.str_('Apple')"
        if label.startswith("np.str_('") and label.endswith("')"):
            label = label[len("np.str_('"):-2]
        cleaned.append(label)
    return np.array(cleaned, dtype=object)

def segment_eeg_data(X, y, window_size=256, step_size=128):
    X_segments = []
    y_segments = []

    for i in range(len(X)):
        signal = X[i]
        label = str(y[i]).strip().capitalize()  # force label to plain string
        for start in range(0, signal.shape[1] - window_size + 1, step_size):
            segment = signal[:, start:start+window_size]
            X_segments.append(segment)
            y_segments.append(label)

    X_segments = np.array(X_segments)
    y_segments = np.array(y_segments, dtype=object)

    unique_labels = sorted(set(y_segments))
    print(f"✅ Segmented EEG shape: {X_segments.shape}, Labels: {unique_labels}")
    return X_segments, y_segments


if __name__ == "__main__":
    # ✅ Load data from ML Project folder
    X_path = os.path.join(BASE_DIR, "X.npy")
    y_path = os.path.join(BASE_DIR, "y.npy")

    X = np.load(X_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)

    # ✅ Clean labels automatically
    y = clean_labels(y)

    # ✅ Segment data
    X_segments, y_segments = segment_eeg_data(X, y)

    # ✅ Save clean segmented arrays
    np.save(os.path.join(BASE_DIR, "X_segments.npy"), X_segments)
    np.save(os.path.join(BASE_DIR, "y_segments.npy"), y_segments)

    print("💾 Saved segmented arrays: X_segments.npy and y_segments.npy")
