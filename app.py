# streamlit app that loads the saved model and classifies uploaded audio
# usage: streamlit run app.py

import json
import os
import tempfile

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from features import SR, load_audio, extract_features, split_windows

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

st.set_page_config(page_title="Urban Sound Classifier", layout="wide")
st.title("Urban Sound Classifier")
st.write("Classical ML model built on hand-crafted audio features. Upload a clip and it predicts "
         "**dog_bark**, **drilling**, or **street_music**.")


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "audio_model.joblib"))
    metrics = None
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            metrics = json.load(f)
    return model, metrics


if not os.path.exists(os.path.join(MODEL_DIR, "audio_model.joblib")):
    st.error("No trained model found. Run `python train_model.py` first.")
    st.stop()

model, metrics = load_model()

# sidebar shows how the model did on the held out folds
with st.sidebar:
    st.header("Model info")
    if metrics:
        st.write(f"Model: `{metrics['best_model']}`")
        st.metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
        st.caption(f"Trained on {metrics['n_train']} clips, tested on {metrics['n_test']} clips "
                   f"from held out folds {metrics['test_folds']}")
        cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path)


def predict_file(y):
    # long files get split into 4 s windows (same length as the training clips) and the probabilities are averaged
    windows = split_windows(y)
    X = np.vstack([extract_features(w) for w in windows])
    probs = model.predict_proba(X).mean(axis=0)
    return dict(zip(model.classes_, probs)), len(windows)


def plot_audio(y):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    librosa.display.waveshow(y, sr=SR, ax=axes[0])
    axes[0].set_title("Waveform")
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR), ref=np.max)
    librosa.display.specshow(mel, sr=SR, x_axis="time", y_axis="mel", ax=axes[1])
    axes[1].set_title("Mel spectrogram")
    fig.tight_layout()
    return fig


uploads = st.file_uploader("Upload audio files (wav or mp3)", type=["wav", "mp3"], accept_multiple_files=True)

if uploads:
    summary = []
    for up in uploads:
        st.divider()
        st.subheader(up.name)
        st.audio(up)

        # librosa needs a real file path, so write the upload to a temp file
        suffix = os.path.splitext(up.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(up.getvalue())
            tmp_path = tmp.name
        try:
            y = load_audio(tmp_path)
        finally:
            os.remove(tmp_path)

        probs, n_win = predict_file(y)
        pred = max(probs, key=probs.get)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.success(f"Prediction: **{pred}**")
            st.write(f"Confidence: {probs[pred]:.1%}")
            st.caption(f"{len(y) / SR:.1f} s long, averaged over {n_win} window(s)")
            st.bar_chart(pd.Series(probs, name="probability"), height=250)
        with col2:
            st.pyplot(plot_audio(y))

        summary.append({"file": up.name, "prediction": pred, "confidence": f"{probs[pred]:.1%}",
                        **{c: round(float(p), 3) for c, p in probs.items()}})

    st.divider()
    st.subheader("Summary")
    st.dataframe(pd.DataFrame(summary), hide_index=True)
