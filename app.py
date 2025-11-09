import streamlit as st
import numpy as np
import tensorflow as tf
from gtts import gTTS
import tempfile
import matplotlib.pyplot as plt
import random
import os

# -------------------------
# Config / paths
# -------------------------
BASE_DIR = r""
MODEL_PATH = os.path.join(BASE_DIR, "eeg_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "label_classes.npy")
Y_SEGMENTS_PATH = os.path.join(BASE_DIR, "y_segments.npy")

# -------------------------
# Load model
# -------------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Cannot load model at {MODEL_PATH}: {e}")
    st.stop()

# -------------------------
# Load or build labels
# -------------------------
labels = None
if os.path.exists(LABELS_PATH):
    try:
        labels = np.load(LABELS_PATH, allow_pickle=True)
        labels = labels.tolist()
        st.sidebar.success("✅ Loaded labels from label_classes.npy")
    except Exception:
        labels = None

if labels is None and os.path.exists(Y_SEGMENTS_PATH):
    try:
        y = np.load(Y_SEGMENTS_PATH, allow_pickle=True)
        labels = list(np.unique(y))
        st.sidebar.success("✅ Loaded labels from y_segments.npy (unique values)")
    except Exception:
        labels = None

if labels is None:
    # fallback: will be replaced after we know model output size
    st.sidebar.warning("⚠️ No label file found — using generic class names")
    labels = None

# -------------------------
# TTS helper (threaded)
# -------------------------
def speak_text_streamlit(text):
    try:
        # Generate speech using gTTS
        tts = gTTS(text)
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_path.name)
        # Play the generated audio in Streamlit
        st.audio(temp_path.name, format="audio/mp3", start_time=0)
    except Exception as e:
        st.error(f"TTS error: {e}")

# -------------------------
# Streamlit UI
# -------------------------
st.title("🧠 EEG Imagined Speech Classifier (robust app)")
st.write("Upload a single EEG segment (.npy) or the whole dataset (.npy). The app will pick/reshape a sample for prediction.")

uploaded = st.file_uploader("Upload .npy file (single (39,256) or dataset (N,39,256) allowed)", type=["npy"])

# -------------------------
# Utility: reshape/choose sample
# -------------------------
def prepare_sample(arr):
    """
    Accepts numpy array loaded from uploaded file.
    Returns sample in shape (1, 39, 256, 1).
    Raises ValueError on unsupported shapes.
    """
    arr = np.asarray(arr)
    if arr.ndim == 4:
        # possible shapes: (N,39,256,1) or (N,39,256,channels)
        if arr.shape[0] == 0:
            raise ValueError("Uploaded array has zero samples.")
        idx = random.randint(0, arr.shape[0] - 1)
        sample = arr[idx]
        # ensure sample shape (39,256) or (39,256,1)
        if sample.ndim == 3 and sample.shape[-1] == 1:
            return np.expand_dims(sample, axis=0)  # (1,39,256,1)
        elif sample.ndim == 2:
            return np.expand_dims(np.expand_dims(sample, -1), 0)
        else:
            # If channels >1, collapse or take first channel
            if sample.ndim == 3:
                sample2 = sample[..., 0]
                return np.expand_dims(np.expand_dims(sample2, -1), 0)
            raise ValueError(f"Unsupported sample shape after selecting index: {sample.shape}")

    elif arr.ndim == 3:
        # could be (N,39,256) or (39,256,1)
        if arr.shape[0] == 39 and arr.shape[1] == 256:
            # it's (39,256,?) actually 3 dims but first dim 39 -> ambiguous
            # treat as (39,256,channels) -> if channels==1 or >1
            if arr.shape[-1] == 1:
                return np.expand_dims(arr, axis=0)
            else:
                # if shape is (39,256,channels) but uploaded as 3D where first dim 39
                # We assume this is (39,256,channels) meaning single sample
                return np.expand_dims(arr, axis=0)
        # else treat as dataset (N,39,256)
        if arr.shape[1] == 39 and arr.shape[2] == 256:
            # arr is (N,39,256)
            idx = random.randint(0, arr.shape[0] - 1)
            sample = arr[idx]
            return np.expand_dims(np.expand_dims(sample, -1), 0)

        # handle common case: (39,256,1) (single sample)
        if arr.shape[0] == 39 and arr.shape[1] == 256:
            # arr is (39,256,channels)
            if arr.shape[-1] == 1:
                return np.expand_dims(arr, axis=0)
            else:
                # take first channel if multiple
                sample = arr[..., 0]
                return np.expand_dims(np.expand_dims(sample, -1), 0)

        raise ValueError(f"Unsupported 3D array shape: {arr.shape}")

    elif arr.ndim == 2:
        # (39,256) single sample -> make (1,39,256,1)
        if arr.shape[0] == 39 and arr.shape[1] == 256:
            return np.expand_dims(np.expand_dims(arr, -1), 0)
        else:
            raise ValueError(f"Unsupported 2D array shape: {arr.shape}")

    else:
        raise ValueError(f"Unsupported array dimensions: {arr.ndim} (shape {arr.shape})")

# -------------------------
# Main: handle upload and predict
# -------------------------
if uploaded is not None:
    try:
        data = np.load(uploaded, allow_pickle=True)
        st.write(f"📐 Uploaded array shape: {data.shape}")

        # prepare a (1,39,256,1) sample
        sample = prepare_sample(data)  # may raise ValueError if unsupported
        st.write(f"✅ Using sample shape: {sample.shape} for prediction")

        # model prediction
        preds = model.predict(sample, verbose=0)
        if preds.ndim == 2:
            preds_vec = preds[0]
        elif preds.ndim == 1:
            preds_vec = preds
        else:
            raise ValueError(f"Unexpected model output shape: {preds.shape}")

        num_classes = preds_vec.shape[0]

        # If labels were not loaded earlier, create generic ones now
        if labels is None:
            labels = [f"class_{i}" for i in range(num_classes)]
            st.sidebar.info("Using generated generic labels because no label file was found.")

        # If label length mismatches model output, handle gracefully
        if len(labels) != num_classes:
            st.sidebar.warning(f"Label count ({len(labels)}) != model output size ({num_classes}). Adjusting labels to match.")
            # try to truncate or pad labels
            if len(labels) > num_classes:
                labels = labels[:num_classes]
            else:
                # pad with generic names
                extra = [f"class_{i}" for i in range(len(labels), num_classes)]
                labels = list(labels) + extra

        # safe indexing
        predicted_idx = int(np.argmax(preds_vec))
        predicted_label = labels[predicted_idx]

        st.subheader(f"🔮 Predicted: {predicted_label}  ({preds_vec[predicted_idx]*100:.2f}%)")

        # top 3
        top_idx = np.argsort(preds_vec)[-3:][::-1]
        st.write("### Top 3 predictions")
        for i in top_idx:
            st.write(f"- {labels[i]} : {preds_vec[i]*100:.2f}%")

        # bar chart
        fig, ax = plt.subplots(figsize=(8,3))
        ax.bar(labels, preds_vec, color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Probability")
        st.pyplot(fig)

        # heatmap of the sample (39 x 256)
        st.write("### EEG segment (heatmap)")
        eeg2d = sample[0, :, :, 0]
        fig2, ax2 = plt.subplots(figsize=(8,3))
        ax2.imshow(eeg2d, aspect='auto', cmap='viridis')
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Channels")
        st.pyplot(fig2)

        # speak button (thread-safe)
        if st.button("🔊 Predict"):
            try:
                st.info("🔈 Speaking prediction...")
                speak_text_streamlit(predicted_label)
            except Exception as e:
                st.error(f"TTS failed: {e}")

    except Exception as e:
        st.error(f"❌ Error while predicting: {e}")
else:
    st.info("Upload a .npy EEG segment (or dataset) to predict.")

