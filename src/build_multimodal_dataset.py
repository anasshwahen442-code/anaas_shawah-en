import pandas as pd
import numpy as np
from audiomics_features import extract_patient_features

manifest = pd.read_csv("/home/claude/cough_audio_manifest.csv")
clinical = pd.read_csv("/home/claude/copd_synthetic_dataset.csv")

rows = []
for pid, group in manifest.groupby("Patient_ID"):
    feats = extract_patient_features(group["clip_path"].tolist())
    feats["Patient_ID"] = pid
    rows.append(feats)

audio_df = pd.DataFrame(rows)
print(f"Extracted {audio_df.shape[1]-1} audio features for {audio_df.shape[0]} patients")

multimodal = clinical.merge(audio_df, on="Patient_ID", how="left")
multimodal.to_csv("/home/claude/copd_multimodal_dataset.csv", index=False)
print("Saved copd_multimodal_dataset.csv, shape:", multimodal.shape)
print(multimodal.filter(regex="mfcc1_mean|spectral_centroid_mean|zcr_mean|Exacerbation").head())
