"""Convert the public request to the model's feature contract."""
import pickle
import pandas as pd
from Backend.features import FEATURES, age_group


def prepare_features(data):
    return pd.DataFrame([{
        'Net_Metrekare': data.net_area, 'Oda_Sayısı': data.rooms,
        'Bulunduğu_Kat': data.floor, 'Binanın_Yaşı': age_group(data.building_age),
        'Isıtma_Tipi': data.heating, 'Şehir': data.city,
        'Binanın_Kat_Sayısı': data.total_floors,
        'Kullanım_Durumu': data.usage_status, 'Banyo_Sayısı': data.bathrooms,
    }], columns=FEATURES)


def load_model(model_path):
    # Only the artifact shipped with this repository is loaded.
    with open(model_path, 'rb') as file:
        return pickle.load(file)
