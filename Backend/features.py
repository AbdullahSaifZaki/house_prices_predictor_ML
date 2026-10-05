"""Shared, serializable preprocessing for training and inference."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

FEATURES = ['Net_Metrekare', 'Oda_Sayısı', 'Bulunduğu_Kat', 'Binanın_Yaşı',
            'Isıtma_Tipi', 'Şehir', 'Binanın_Kat_Sayısı', 'Kullanım_Durumu', 'Banyo_Sayısı']
CATEGORICAL = ['Bulunduğu_Kat', 'Isıtma_Tipi', 'Şehir', 'Kullanım_Durumu']
NUMERIC = [column for column in FEATURES if column not in CATEGORICAL]
AGE_LABELS = {'0 (Yeni)': 0, '1': 1, '2': 1, '3': 1, '4': 1,
              '5-10': 2, '11-15': 3, '16-20': 4, '21 Ve Üzeri': 5}


def age_group(years):
    """Map an actual age in years to the dataset's six age bands."""
    return int(np.searchsorted([0, 4, 10, 15, 20], years, side='left'))


class HousePreprocessor(TransformerMixin, BaseEstimator):
    """Fit medians and frequent heating labels on training features only.

    Categorical missingness is explicit rather than a guessed category. The
    target is never used. This transformer stays inside every model pipeline.
    """
    def __init__(self, heating_min_count=500):
        self.heating_min_count = heating_min_count

    def fit(self, X, y=None):
        frame = X.loc[:, FEATURES]
        self.feature_names_in_ = np.array(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        self.medians_ = frame[NUMERIC].median()
        if self.medians_.isna().any():
            raise ValueError('Training data must contain values for every numeric feature')
        counts = frame['Isıtma_Tipi'].value_counts()
        self.heating_categories_ = set(counts[counts >= self.heating_min_count].index)
        return self

    def transform(self, X):
        check_is_fitted(self, ['medians_', 'heating_categories_'])
        frame = X.loc[:, FEATURES].copy()
        frame[NUMERIC] = frame[NUMERIC].astype(float).fillna(self.medians_)
        for column in CATEGORICAL:
            frame[column] = frame[column].fillna('Bilinmiyor').astype(str)
        frame['Isıtma_Tipi'] = frame['Isıtma_Tipi'].where(
            frame['Isıtma_Tipi'].isin(self.heating_categories_), 'Diğer')
        return frame

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURES, dtype=object)
