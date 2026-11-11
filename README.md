# Patient Outcomes

This repository contains analytical and predictive modelling workflows for patient-level outcomes, including hospital length of stay (HLOS), mortality risk prediction, cohort analysis, and synthetic data generation.

The project brings together:
- Python-based machine learning pipelines
- model evaluation and explainability workflows
- clinical and administrative dataset processing
- table generation in R
- synthetic data generation for privacy-aware experimentation

## Project focus

### 1. Hospital length of stay modelling
The HLOS workflow uses machine learning models to predict length of hospital stay using patient-level clinical and operational data. It includes:
- data preprocessing and feature engineering
- train/test splitting
- hyperparameter tuning using cross-validation and grid search
- comparison of models such as random forest, XGBoost, LightGBM, gradient boosting, Elastic Net, SVR, and CatBoost
- stacked ensemble model training
- joblib model persistence
- SHAP-based explainability and feature importance analysis

Main scripts:
- `HLOS/ML_developement/Scripts/ML_LOS_pipeline.py`
- `HLOS/ML_developement/Scripts/machine_learning_pipeline.py`
- `HLOS/ML_developement/Scripts/model_evaluation.py`
- `HLOS/ML_developement/Scripts/visualizations.py`
- `HLOS/ML_developement/Scripts/test_model.py`

### 2. Mortality modelling
The `Mortality model/` folder contains datasets, model outputs, calibration and validation results, and notebooks related to mortality risk prediction and explainability.

### 3. Table generation
The `Table_data_analysis/` folder contains an R Markdown workflow for producing summary table outputs such as Table 1.

## Environment and dependencies

The project is primarily Python-based and uses the following packages:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- xgboost
- lightgbm
- catboost
- shap
- sdv
- sdmetrics

R is also used for table generation in `Table_data_analysis`.

## Recommended setup

```bash
cd "Patient outcomes"
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost lightgbm catboost shap sdv sdmetrics
```

## Running the HLOS model pipeline

From the scripts directory:

```bash
cd "HLOS/ML_developement/Scripts"
python ML_LOS_pipeline.py
```

This workflow writes trained model artefacts into:

```text
HLOS/ML_developement/Outputs/ML_outputs/
```

Typical outputs include:
- `model_*.joblib`
- `best_model_summary.txt`
- `model_performance_metrics.csv`
- `feature_importances.csv`
- stacked model files and configuration objects


## R table workflow

```bash
cd "Table_data_analysis"
Rscript -e "rmarkdown::render('table1_creation.Rmd')"
```


## Summary

This repository supports patient outcome research and clinical prediction modelling, with a structure tailored to hospital-level data analysis, outcome prediction, and methodological experimentation. It is intended for reproducible analysis, model development, and performance reporting within a research or applied healthcare setting.
