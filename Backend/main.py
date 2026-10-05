"""Single-origin FastAPI app, runnable locally or as a Vercel function."""
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from Backend.schemas import HouseInput, PredictionOutput, OPTIONS
from Backend.utils import load_model, prepare_features

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title='EstateEstimate', version='2.0.0')
model = load_model(ROOT / 'Model/best_model.pkl')


@app.get('/', include_in_schema=False)
def homepage():
    return FileResponse(ROOT / 'Frontend/index.html')


@app.get('/health')
def health():
    return {'message': 'ML API is running'}


@app.get('/options')
def options():
    return OPTIONS


@app.post('/predict', response_model=PredictionOutput)
def predict(data: HouseInput):
    log_price = model.predict(prepare_features(data))[0]
    with np.errstate(over='ignore'):
        price = float(np.expm1(log_price))
    if not np.isfinite(price) or price <= 0:
        raise HTTPException(status_code=500, detail='The model could not produce a valid estimate')
    return {'prediction': price}


# The same paths work under Uvicorn and Vercel's static-file promotion.
app.mount('/assets', StaticFiles(directory=ROOT / 'Frontend'), name='assets')
