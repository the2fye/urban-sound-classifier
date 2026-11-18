# feature extraction shared by train_model.py and app.py, so training and prediction always match

import numpy as np
import librosa

SR = 22050          # resample everything to the same rate
CLIP_SECONDS = 4.0  # UrbanSound8K clips are at most 4 s, so longer files get split into 4 s windows
N_MFCC = 40


def load_audio(path):
    # mono + resample to SR so every file is on the same footing
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def clean_signal(y):
    # trim leading/trailing silence, then peak normalize so recording volume doesn't matter
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    if len(y_trim) > 0:
        y = y_trim
    # some clips are very short, pad up to 0.5 s so the fft frames still work
    min_len = int(0.5 * SR)
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))
    if np.max(np.abs(y)) > 0:
        y = librosa.util.normalize(y)
    return y


def mean_std(x):
    # summarize a (n_features, n_frames) matrix into one vector per clip
    return np.concatenate([np.mean(x, axis=1), np.std(x, axis=1)])


def extract_features(y):
    y = clean_signal(y)
    stft = np.abs(librosa.stft(y))

    # cepstral domain
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)

    # frequency domain
    centroid = librosa.feature.spectral_centroid(S=stft, sr=SR)
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=SR)
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=SR)
    flatness = librosa.feature.spectral_flatness(S=stft)
    contrast = librosa.feature.spectral_contrast(S=stft, sr=SR)
    chroma = librosa.feature.chroma_stft(S=stft, sr=SR)

    # time domain
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    feats = np.concatenate([
        mean_std(mfcc),          # 80
        np.mean(mfcc_delta, 1),  # 40
        mean_std(centroid),      # 2
        mean_std(bandwidth),     # 2
        mean_std(rolloff),       # 2
        mean_std(flatness),      # 2
        np.mean(contrast, 1),    # 7
        np.mean(chroma, 1),      # 12
        mean_std(zcr),           # 2
        mean_std(rms),           # 2
    ])                           # 151 total
    return feats.astype(np.float32)


def feature_names():
    names = [f"mfcc{i}_mean" for i in range(N_MFCC)] + [f"mfcc{i}_std" for i in range(N_MFCC)]
    names += [f"mfcc{i}_delta_mean" for i in range(N_MFCC)]
    for f in ["centroid", "bandwidth", "rolloff", "flatness"]:
        names += [f"{f}_mean", f"{f}_std"]
    names += [f"contrast{i}_mean" for i in range(7)]
    names += [f"chroma{i}_mean" for i in range(12)]
    names += ["zcr_mean", "zcr_std", "rms_mean", "rms_std"]
    return names


def split_windows(y):
    # split a long recording into 4 s windows (the last piece is kept if it's at least 1 s)
    win = int(CLIP_SECONDS * SR)
    if len(y) <= win:
        return [y]
    chunks = [y[i:i + win] for i in range(0, len(y), win)]
    return [c for c in chunks if len(c) >= SR] or [y[:win]]
