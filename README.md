## Project Purpose
This repository contains the technical planning and pipeline scaffolding 
for a machine-learning workflow that predicts whether a FinTrust transaction 
requires risk review, using the synthetic `Risk_Review_Flag` label. The 
ML Engineering's role is not building the predictive model itself, 
but designing the reproducible workflow, technical requirements, and 
testing strategy that a model can be integrated into.

## Repository Structure

| Folder | Purpose |
|---|---|
| `data/raw/` | Original, unmodified source data (FinTrust customer & transaction files) |
| `data/processed/` | Cleaned and feature-engineered data, ready for modelling |
| `notebooks/` | Exploratory data analysis|
| `src/` | Reusable, tested pipeline code (validation, preprocessing, features, model) |
| `tests/` | Unit tests for everything in `src/` |
| `models/` | Saved/serialized trained model artifacts |
| `reports/` | Generated outputs, evaluation results, figures |


## Status
Week 2 complete: data validation and preprocessing functions built and tested 
(both customer and transaction datasets), datasets joined, baseline Logistic 
Regression model trained and evaluated under both random and time-based splits, 
5 technical tests passed. Migration of pipeline logic from notebooks into 
standalone src/ modules planned for Week 3.
