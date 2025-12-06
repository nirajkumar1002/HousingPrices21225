# 🏡 House Prices Prediction – Advanced ML Pipeline (Ames Housing Dataset)

This project builds a complete end-to-end machine learning pipeline to predict house prices for the Ames, Iowa dataset, part of the Kaggle competition **"House Prices: Advanced Regression Techniques"**.

We progressively improve model performance from a baseline to an advanced multi-model ensemble, achieving a final **Kaggle score of 13,801**, which is a strong result for this competition without exhaustive hyperparameter search.

---

## 🚀 Project Highlights

### ✔ Clean, modular ML pipeline  
### ✔ Strong Feature Engineering  
Both basic and advanced:
- Total square footage  
- Total bathrooms  
- Porch area  
- House age & remodel age  
- Interaction features (e.g., `GrLivArea × OverallQual`)  
- Quality ordinal scores (Ex/Gd/TA/Fa/Po)  
- Rare-category grouping  

### ✔ Multiple ML Models  
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  

### ✔ Final 4-Model Ensemble  
The final blended model significantly outperforms any single model.

---

## 📊 Final Kaggle Performance

| Model | Validation RMSE (log) | Kaggle Score |
|-------|------------------------|--------------|
| Random Forest | 0.146 | ~17,000 |  
| XGBoost (baseline) | 0.134 | ~14,200 |  
| XGBoost + FE | **0.1297** | ~14,160 |  
| 3-model blend | – | ~14,051 |  
| **4-model blend (Final)** | – | **13,801** |  

---
## 📁 Repository Structure

├── notebooks/
│ └── house_prices_final.ipynb # Final notebook
├── data/
│ ├── train.csv
│ ├── test.csv
│ └── submission.csv
├── README.md
└── report.md

*(Note: `data/` is usually excluded from GitHub — add to .gitignore)*

---

## 🧠 Approach Summary

### 1️⃣ Data Exploration  
- Distribution of SalePrice  
- Log-transforming targets  
- Missing value analysis  
- Numeric vs categorical breakdown  

### 2️⃣ Feature Engineering  
- Basic: TotalSF, TotalBath, Age features  
- Advanced interactions: `TotalSF × Qual`, `GarageCars × GarageArea`  
- Ordinal encodings for quality features  
- Rare category grouping  

### 3️⃣ Modeling  
Models trained:
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  

Evaluation metric: **RMSE on log(SalePrice)**.

### 4️⃣ Blending  
Weighted ensemble:



---

## 🧪 Requirements

python 3.9+
numpy
pandas
scikit-learn
xgboost
lightgbm
catboost
matplotlib
seaborn


---

## ▶ How to Run

1. Download the dataset from Kaggle.  
2. Place `train.csv` and `test.csv` in a `data/` folder.  
3. Open the notebook:
jupyter notebook notebooks/house_prices_final.ipynb
4. Run all cells to generate:
submission.csv
5. Upload to Kaggle.

---

## 🏆 Final Thoughts & Next Steps

Future improvements:
- K-fold cross-validation with out-of-fold stacking  
- Bayesian hyperparameter optimization  
- Neighborhood-level domain feature engineering  
- Weighted ensembling tuned by grid search or Optuna  

This project demonstrates the full ML pipeline for tabular problems and provides a strong foundation for competitive modeling.

---

## 👤 Author
Niraj Kumar  
*Data Science & ML Enthusiast*  


