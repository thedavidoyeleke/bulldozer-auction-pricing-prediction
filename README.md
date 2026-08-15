# 🚜 Blue Book for Bulldozers: Predicting Sale Prices with Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![ML](https://img.shields.io/badge/Task-Regression-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

This project tackles the classic Kaggle machine learning challenge: **Blue Book for Bulldozers**. The core objective is to predict the auction sale price of heavy equipment based on its usage, equipment type, and configuration parameters.

---

## 📌 Project Overview

* **Problem Type:** Regression (Supervised Learning)
* **Evaluation Metric:** Root Mean Squared Logarithmic Error (**RMSLE**)
* **Goal:** Build and optimize end-to-end machine learning pipelines to accurately estimate equipment auction values.

---

## 📂 Dataset Structure

The data for this project is sourced from Kaggle's [Blue Book for Bulldozers Competition](https://www.kaggle.com/c/bluebook-for-bulldozers/data). It consists of three primary files:

| File | Description |
| :--- | :--- |
| `Train.csv` | Historical bulldozer sales data up through the end of 2011. |
| `Valid.csv` | Validation dataset containing sales records from January 1, 2012, to April 30, 2012. |
| `ValidSolution.csv` | The true target values (`SalePrice`) corresponding to the validation set. |

---

## ⚙️ Methodology & Pipeline Architecture

1. **Feature Engineering**
   Extracted rich datetime attributes (years, months, days, day of week, etc.) from transaction sale dates.

2. **Preprocessing & Encoding**
   Handled missing values through automated imputation strategies and transformed high-cardinality categorical variables into structured numerical formats.

3. **Model Exploration & Training**
   Tested and iterated across multiple robust regression algorithms:
   * **Random Forest Regressor** — Baseline & strong performer
   * **LightGBM Regressor** — Gradient boosted trees with hyperparameter tuning
   * **CatBoost Regressor** — Native categorical handling

4. **Model Blending (Ensembling)**
   Combined predictions across diverse models via weighted averaging to mitigate individual model blind spots and optimize overall error bounds.

---

## 📊 Validation Results

Model performance was evaluated on the validation set (`Valid.csv` matched against `ValidSolution.csv`):

| Model / Strategy | R² Score | MAE ($) | RMSLE |
| :--- | :---: | :---: | :---: |
| Random Forest | 0.8827 | 5,895.78 | 0.2431 |
| Tuned LightGBM | 0.8838 | 5,955.42 | 0.2508 |
| CatBoost | 0.8680 | 6,431.60 | 0.2740 |
| **Blended Ensemble (80/10/10)** | **0.8849** | **5,843.27** | **0.2415** |

> 🏆 The blended ensemble delivered the best overall performance across all three metrics.

---

## 🗂️ Repository Structure

```text
├── valid_predictions.csv         # Random Forest validation predictions
├── valid_predictions_lgb.csv     # LightGBM validation predictions
├── valid_predictions_cat.csv     # CatBoost validation predictions
├── valid_predictions_blended.csv # Final optimized ensemble predictions
└── README.md                     # Project documentation
```

---

## 📎 Source

Data and competition details available on Kaggle: [Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers/data)
