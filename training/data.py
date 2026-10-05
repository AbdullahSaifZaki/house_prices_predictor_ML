"""Deterministic data rules; learned preprocessing belongs in the pipeline."""
from pathlib import Path
import numpy as np
import pandas as pd
from Backend.features import FEATURES, AGE_LABELS

ROOT = Path(__file__).resolve().parents[1]


def clean_data(raw):
    frame = raw.drop_duplicates().copy()
    # Fixed scope rules inherited from the original project, applied equally
    # to all models. Preserve missing inputs so train-only imputation can learn.
    mask = (frame.Net_Metrekare.between(50, 500, inclusive='neither')
            & frame['Brüt_Metrekare'].between(50, 600, inclusive='neither')
            & (frame['Oda_Sayısı'].between(0, 9, inclusive='neither') | frame['Oda_Sayısı'].isna())
            & (frame['Banyo_Sayısı'].between(0, 5, inclusive='neither') | frame['Banyo_Sayısı'].isna())
            & frame.Fiyat.between(1_000_000, 20_000_000, inclusive='neither')
            & (frame['Binanın_Kat_Sayısı'] > 0))
    return frame.loc[mask].copy()


def feature_frame(cleaned):
    features = cleaned[FEATURES].copy()
    features['Binanın_Yaşı'] = features['Binanın_Yaşı'].map(AGE_LABELS)
    if features['Binanın_Yaşı'].isna().any():
        raise ValueError('Unexpected building-age label in raw data')
    return features


def export_snapshots():
    raw = pd.read_csv(ROOT / 'data/raw/home_price.csv')
    cleaned = clean_data(raw)
    cleaned.to_csv(ROOT / 'data/processed/cleaned_data.csv', index=False)
    engineered = feature_frame(cleaned)
    engineered['Fiyat'] = np.log1p(cleaned.Fiyat)
    engineered.to_csv(ROOT / 'data/processed/cleaned_data_after_eda.csv', index=False)
    return raw, cleaned


if __name__ == '__main__':
    raw, cleaned = export_snapshots()
    print(f'{len(raw)} raw rows -> {len(cleaned)} eligible rows')
