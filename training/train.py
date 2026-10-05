"""Reproducible comparison; choose by validation RMSE before inspecting test."""
import hashlib
import importlib.metadata
import json
import pickle
import platform
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBRegressor
from Backend.features import FEATURES, CATEGORICAL, NUMERIC, HousePreprocessor
from training.data import ROOT, export_snapshots, feature_frame


def models():
    encode = ColumnTransformer([
        ('numeric', RobustScaler(), NUMERIC),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)])
    from sklearn.base import clone
    estimators = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=2),
        'XGBoost': XGBRegressor(n_estimators=400, max_depth=5, learning_rate=.05,
                               subsample=.8, colsample_bytree=.8, reg_lambda=10, random_state=42, n_jobs=2),
        'CatBoost': CatBoostRegressor(iterations=600, depth=6, learning_rate=.05,
                                     loss_function='RMSE', cat_features=CATEGORICAL,
                                     random_seed=42, verbose=False, thread_count=2, allow_writing_files=False)}
    return {name: Pipeline([('preprocess', HousePreprocessor())]
                           + ([] if name == 'CatBoost' else [('encode', clone(encode))])
                           + [('model', model)]) for name, model in estimators.items()}


def split_indices(X):
    # Identical model-input profiles must not occur on both sides of a split.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, groups=groups))
    train_local, val_local = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=43).split(X.iloc[dev], groups=groups[dev]))
    return dev[train_local], dev[val_local], test, groups


def metrics(y, pred):
    true_try, pred_try = np.expm1(y), np.expm1(pred)
    return {'rmse_log': float(np.sqrt(mean_squared_error(y, pred))),
            'mae_log': float(mean_absolute_error(y, pred)), 'r2_log': float(r2_score(y, pred)),
            'mae_try': float(mean_absolute_error(true_try, pred_try)),
            'rmse_try': float(np.sqrt(mean_squared_error(true_try, pred_try))),
            'r2_try': float(r2_score(true_try, pred_try)),
            'median_ape_pct': float(np.median(np.abs(pred_try-true_try)/true_try)*100)}


def main():
    raw, cleaned = export_snapshots()
    X = feature_frame(cleaned)
    y = np.log1p(cleaned.Fiyat)
    train, val, test, groups = split_indices(X)
    candidates = models()
    validation = {}
    for name, pipeline in candidates.items():
        start = time.perf_counter()
        pipeline.fit(X.iloc[train], y.iloc[train])
        validation[name] = metrics(y.iloc[val], pipeline.predict(X.iloc[val]))
        print(name, 'validation RMSE', validation[name]['rmse_log'], 'seconds', round(time.perf_counter()-start,1), flush=True)
    selected = min(validation, key=lambda name: validation[name]['rmse_log'])
    print('Selected on validation:', selected, flush=True)
    development = np.concatenate([train, val])
    results = {}
    final_model = None
    for name, pipeline in models().items():
        fitted = pipeline.fit(X.iloc[development], y.iloc[development])
        pred = fitted.predict(X.iloc[test])
        results[name] = metrics(y.iloc[test], pred)
        results[name]['train_r2_log'] = float(fitted.score(X.iloc[development], y.iloc[development]))
        if name == selected:
            final_model = fitted
        print(name, 'test', results[name], flush=True)
    target = ROOT / 'Model/best_model.pkl'
    target.write_bytes(pickle.dumps(final_model, protocol=5))
    options = {field: sorted(cleaned[column].dropna().unique().tolist()) for field, column in
               {'city':'Şehir', 'floor':'Bulunduğu_Kat', 'heating':'Isıtma_Tipi', 'usage_status':'Kullanım_Durumu'}.items()}
    (ROOT / 'Model/input_options.json').write_text(json.dumps(options, ensure_ascii=False, indent=2)+'\n')
    manifest = {'seed': 42, 'validation_seed':43, 'raw_rows':len(raw), 'exact_duplicates':int(raw.duplicated().sum()),
                'eligible_rows':len(X), 'cities':int(X['Şehir'].nunique()),
                'train_rows':len(train), 'validation_rows':len(val), 'test_rows':len(test),
                'unique_feature_profiles':int(len(set(groups))),
                'selected_model':selected, 'selection_metric':'validation rmse_log',
                'features':FEATURES, 'validation':validation, 'test':results,
                'raw_sha256':hashlib.sha256((ROOT/'data/raw/home_price.csv').read_bytes()).hexdigest(),
                'model_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'split_row_ids': {'train': cleaned.index[train].tolist(), 'validation': cleaned.index[val].tolist(), 'test': cleaned.index[test].tolist()},
                'python':platform.python_version(),
                'versions':{**{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','catboost']}, 'xgboost': __import__('xgboost').__version__},
                'split_group_overlap':{a+'_'+b:len(set(groups[i]) & set(groups[j])) for a,i,b,j in
                                       [('train',train,'validation',val),('train',train,'test',test),('validation',val,'test',test)]}}
    (ROOT / 'Model/metrics.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
