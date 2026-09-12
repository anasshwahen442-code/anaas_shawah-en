"""
Generates 3 synthetic cough-like clips per patient. This is a STAND-IN for
real smartphone recordings -- it exists so the multimodal pipeline (audio +
clinical + biomarkers) can be run and verified end-to-end today. Replace the
`synthesize_cough()` body with real .wav loading once recordings exist; the
downstream feature extraction and model code do not need to change.

Cough acoustic character is loosely linked to FEV1 / mMRC so the synthetic
signal is a plausible stand-in (worse airway obstruction -> lower-pitched,
noisier, longer cough) -- NOT a claim about what real cough acoustics look
like, just enough structure to exercise the pipeline honestly.
"""
import numpy as np
import soundfile as sf
import pandas as pd
import os

SR = 16000
OUT_DIR = "/home/claude/synthetic_cough_audio"
os.makedirs(OUT_DIR, exist_ok=True)


def synthesize_cough(severity: float, rng: np.random.Generator) -> np.ndarray:
    """severity in [0,1]: higher = more obstructed/harsher cough."""
    duration = 0.35 + 0.25 * severity + rng.normal(0, 0.03)
    duration = max(0.15, duration)
    n = int(SR * duration)
    t = np.linspace(0, duration, n)

    # Broadband turbulent noise, more low-frequency energy at higher severity
    noise = rng.normal(0, 1, n)
    b = [1.0]
    a = [1.0, -(0.2 + 0.5 * severity)]  # simple low-pass leak: more severity -> more low-freq energy
    from scipy.signal import lfilter
    shaped = lfilter(b, a, noise)

    # Explosive onset + decay envelope typical of a cough burst
    envelope = np.exp(-t / (0.08 + 0.05 * severity)) * (t < duration)
    envelope[: int(0.01 * SR)] *= np.linspace(0, 1, int(0.01 * SR))  # fast attack

    clip = shaped * envelope
    clip = clip / (np.max(np.abs(clip)) + 1e-8) * 0.8
    return clip.astype(np.float32)


def main():
    rng = np.random.default_rng(7)
    dataset = pd.read_csv("/home/claude/copd_synthetic_dataset.csv")

    manifest_rows = []
    for _, row in dataset.iterrows():
        # Map clinical severity proxy (low FEV1, high mMRC) to a 0-1 severity score
        severity = np.clip(
            0.5 * (1 - row["FEV1_pct_predicted"] / 100) + 0.5 * (row["mMRC_Dyspnea"] / 4)
            + rng.normal(0, 0.08),
            0, 1,
        )
        for clip_idx in range(3):
            clip = synthesize_cough(severity, rng)
            fname = f"{row['Patient_ID']}_cough{clip_idx+1}.wav"
            path = os.path.join(OUT_DIR, fname)
            sf.write(path, clip, SR)
            manifest_rows.append({"Patient_ID": row["Patient_ID"], "clip_path": path})

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv("/home/claude/cough_audio_manifest.csv", index=False)
    print(f"Wrote {len(manifest)} clips for {dataset.shape[0]} patients to {OUT_DIR}")


if __name__ == "__main__":
    main()
