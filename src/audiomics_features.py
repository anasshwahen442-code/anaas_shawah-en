"""
audiomics_features.py
----------------------
Smartphone-based cough/exhalation acoustic feature extraction for the
COPD exacerbation prediction framework.

Given a mono audio signal (a single cough or exhalation clip), this module
extracts a compact, clinically-motivated feature set:
  - MFCCs (13 coefficients) — timbral/spectral envelope shape
  - Spectral centroid — "brightness" of the sound (wet/harsh coughs skew higher)
  - Zero-crossing rate — noisiness / turbulence of the airflow
  - Spectral bandwidth & RMS energy — supporting descriptors

Each feature is summarized by its mean and std across the clip, giving a
fixed-length vector regardless of clip duration -- required to feed a
tabular classifier (LogisticRegression / RandomForest) alongside the
clinical/biomarker variables.
"""

import numpy as np
import librosa


N_MFCC = 13


def extract_cough_features(y: np.ndarray, sr: int) -> dict:
    """Extract a fixed-length acoustic feature vector from one audio clip.

    Parameters
    ----------
    y : np.ndarray
        Mono audio waveform, values in [-1, 1].
    sr : int
        Sample rate in Hz.

    Returns
    -------
    dict of feature_name -> float
    """
    if y.size == 0 or np.allclose(y, 0):
        # Silent/empty clip -- return NaNs so it's obviously flagged upstream
        # rather than silently injecting zeros that look like real signal.
        feats = {f"mfcc{i+1}_mean": np.nan for i in range(N_MFCC)}
        feats.update({f"mfcc{i+1}_std": np.nan for i in range(N_MFCC)})
        feats.update({
            "spectral_centroid_mean": np.nan, "spectral_centroid_std": np.nan,
            "zcr_mean": np.nan, "zcr_std": np.nan,
            "spectral_bandwidth_mean": np.nan, "rms_mean": np.nan,
        })
        return feats

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    feats = {}
    for i in range(N_MFCC):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))

    feats["spectral_centroid_mean"] = float(np.mean(centroid))
    feats["spectral_centroid_std"] = float(np.std(centroid))
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"] = float(np.std(zcr))
    feats["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    feats["rms_mean"] = float(np.mean(rms))
    return feats


def extract_patient_features(clip_paths: list[str]) -> dict:
    """Aggregate features across multiple cough clips for one patient
    (patients typically provide 2-5 cough recordings per session).
    Averages each feature across clips -- median would be more robust to
    one bad recording, swap in np.median if that matters more to you.
    """
    import soundfile as sf

    per_clip = []
    for path in clip_paths:
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)  # downmix to mono
        per_clip.append(extract_cough_features(y, sr))

    keys = per_clip[0].keys()
    return {k: float(np.nanmean([f[k] for f in per_clip])) for k in keys}


if __name__ == "__main__":
    # Smoke test with a synthetic clip so this module is verifiably runnable
    # without needing a real recording on disk.
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    fake_cough = 0.3 * np.random.randn(sr) * np.exp(-3 * t)  # decaying noise burst
    result = extract_cough_features(fake_cough.astype(np.float32), sr)
    for k, v in result.items():
        print(f"{k}: {v:.4f}")
