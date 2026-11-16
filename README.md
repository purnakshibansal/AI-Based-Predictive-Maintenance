# AI-Based Predictive Maintenance

A machine learning system that estimates the failure risk of a machine from six
readings, with a seven-page Streamlit dashboard.

> The dataset is **synthetic** and the failure score is a **risk indicator, not a
> guaranteed probability**. Predictions support inspection decisions; they do not
> replace them.

## Live app

https://ai-based-predictive-maintenance-5u7aff34e4t6mtluwnr2zo.streamlit.app

## What it does

- Predicts failure risk (Low / Medium / High) for one machine or a whole CSV.
- Compares four models: Logistic Regression, Decision Tree, Random Forest, XGBoost.
- Explains the data, the model results and the documented failure rules.

## Results (2,000 held-out test rows, 68 real failures)

Final model: **XGBoost**, chosen by cross-validated PR-AUC before the test rows were used.

| Measure | Value |
|---|---|
| Failures found | 54 of 68 (recall 79.4%) |
| Failures missed | 14 |
| False alarms | 34 (precision 61.4%) |
| PR-AUC | 0.827 |
| ROC-AUC | 0.966 |

Accuracy is not used to judge models: 96.6% of rows are normal, so a model that
always says "no failure" would already score 96.6%.

## How it works

1. `src/data_loader.py` loads the CSV (BOM-safe) and defines the column lists.
2. `src/preprocessing.py` keeps only Type plus five sensors as inputs. Failure-mode
   flags and IDs are blocked to avoid target leakage. Stratified 80/20 split, seed 42.
3. `src/train_model.py` tunes four models with 5-fold cross-validation on training
   rows only, using class weights for the rare failure class.
4. `src/evaluate_model.py` picks each alert threshold on training data (F2 score),
   then reports test-set results once.
5. `src/finalize_model.py` saves `models/best_model.joblib` and `models/model_info.json`.
6. `src/predict.py` validates input and predicts; `app.py` and `views/` are the dashboard.

## Run it

Python 3.14 was used for development.

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements-dev.txt
    python -m pytest -q
    streamlit run app.py

To retrain from scratch:

    python -m src.train_model
    python -m src.evaluate_model
    python -m src.finalize_model

## Dashboard pages

Home, Prediction, Batch Prediction (CSV upload), Model Performance, Dataset Explorer,
Maintenance Insights and About.

## Project structure

    app.py                 Streamlit entry point
    views/                 dashboard pages
    src/                   data loading, preprocessing, training, evaluation, prediction
    tests/                 pytest tests
    data/raw/              original dataset (unmodified)
    models/                final saved model and its info file
    reports/               metrics JSON and figures

## Limitations

- Synthetic data; real machines may behave differently.
- The model still misses some failures (14 of 68 in the test rows).
- The test set has only 68 failures, so results can shift with a different split.
- The failure rules on the Insights page describe how this dataset's labels were
  generated. They are not real-world safety limits.
- Random failures (RNF) cannot be predicted from sensor values.

## Data source and licence

AI4I 2020 Predictive Maintenance Dataset, S. Matzka (2020), UCI Machine Learning
Repository. DOI: 10.24432/C5HS5C. Licensed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). The CSV is included unmodified.
