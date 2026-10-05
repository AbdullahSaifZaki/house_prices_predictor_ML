# Türkiye House Price Prediction

An end-to-end machine learning project that estimates house prices in Türkiye from nine property features. It includes data preparation, exploratory analysis, model comparison, a FastAPI backend, and a responsive HTML/CSS/JavaScript frontend.

## Algorithms Explored

- Linear Regression
- Random Forest
- XGBoost — selected for the API using validation error
- CatBoost

## Dataset and Processing

The raw file contains **20,326 records across 53 cities**. Removing duplicates and applying fixed property/price bounds leaves **17,292 records**. The model uses net area, rooms, floor, age group, heating, city, total floors, occupancy, and bathrooms.

Prices use `log1p` during training and `expm1` for predictions. Numeric medians and heating-category grouping are fitted inside each model's pipeline. Identical feature profiles stay in the same split to avoid train/test overlap.

## Results

All models use the same held-out test set of 3,455 records. Selection uses a separate validation set; the saved model is fitted on training plus validation data.

| Model | Log RMSE | Log R² | MAE (TRY) |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.324 | 0.615 | 806,299 |
| Random Forest | 0.312 | 0.644 | 767,432 |
| XGBoost | 0.300 | 0.671 | 742,473 |
| CatBoost | 0.303 | 0.664 | 745,324 |

See the [six-page technical report](technical-report.pdf) and [machine-readable results](Model/metrics.json).

## Directory Structure

```text
├── app.py              
├── Backend/             
├── Frontend/          
├── Model/              
├── data/               
├── training/           
├── notebooks/       
└── technical-report.pdf 
```





