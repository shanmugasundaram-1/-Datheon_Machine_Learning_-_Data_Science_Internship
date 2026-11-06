# Datheon AI Internship – Machine Learning Task Pack

Four end-to-end machine learning projects built as Jupyter notebooks for the Datheon AI Systems & Automation training programme (Application ID: `DT-2026-GHN2Q`).

| # | Task | Notebook | Key Result |
|---|------|----------|------------|
| 1 | Multi-Class Benchmark Classification (Iris) | `iris_classification_pipeline.ipynb` | 93.3% accuracy (Logistic Regression & Decision Tree) |
| 2 | Real Estate Housing Price Prediction (OLS / Ridge / Lasso) | `housing_regression_ridge_lasso.ipynb` | R² = 0.608, RMSE ≈ 71.7k |
| 3 | Predictive Customer Churn Classifier (Random Forest & XGBoost) | `churn_classifier_rf_xgboost.ipynb` | ROC-AUC = 0.839 (target > 0.80 met) |
| 4 | NLP Sentiment Classification (TF-IDF) | `nlp_sentiment_tfidf.ipynb` | 83.5% accuracy (Logistic Regression) |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── iris_classification_pipeline.ipynb
├── housing_regression_ridge_lasso.ipynb
├── churn_classifier_rf_xgboost.ipynb
├── nlp_sentiment_tfidf.ipynb
└── sentiment_reviews.csv          # starter file for Task 4 (place next to the notebook)
```

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn nltk jupyter
jupyter notebook
```

`requirements.txt` content:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
nltk
jupyter
```

An internet connection is required: Tasks 1–3 load their datasets directly from GitHub URLs, and Task 4 downloads NLTK resources. In each notebook use **Kernel → Restart & Run All**.

---

## Task 1 – Multi-Class Benchmark Classification Pipeline (Iris)

**Dataset:** [Iris](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv) (150 samples, 3 classes)

**Approach**
1. Load data with Pandas and explore with a Seaborn pairplot
2. Normalise features with `StandardScaler` (fit on training data only)
3. Stratified 80/20 train/test split
4. Train Logistic Regression and Decision Tree
5. Evaluate accuracy, precision, recall, F1; plot confusion matrix heatmaps
6. Plot decision boundaries (petal length vs petal width)

**Results (test set, 30 samples)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 |
| Decision Tree | 0.933 | 0.933 | 0.933 | 0.933 |

Setosa is classified perfectly; all errors occur between versicolor and virginica, which overlap slightly.

---

## Task 2 – Real Estate Housing Price Prediction (Ridge & Lasso Regression)

**Dataset:** [California Housing](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv) (20,640 rows)

**Approach**
1. Impute 207 missing `total_bedrooms` values (median, fitted on training data only)
2. Correlation heatmap and geographic exploration
3. Feature engineering: `rooms_per_household`, `bedrooms_per_room`, `population_per_household`, `income_x_rooms` (interaction)
4. Multicollinearity handling via VIF: `households` (28.3), `total_bedrooms` (26.9) and `total_rooms` (12.1) were replaced by ratio features, bringing their VIF below 2
5. Standardise numeric features, one-hot encode `ocean_proximity` (all inside a Pipeline)
6. Fit OLS, Ridge (L2) and Lasso (L1); tune alpha with 5-fold `GridSearchCV` (best alpha = 1 for both)
7. Plot predicted vs actual values, residual distributions and residuals vs predictions

**Results (test set)**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| OLS | 71,651 | 51,568 | 0.608 |
| Ridge (L2) | 71,658 | 51,575 | 0.608 |
| Lasso (L1) | 71,652 | 51,569 | 0.608 |

The three models perform almost identically because the feature cleanup already removed the worst multicollinearity. Lasso set the `NEAR BAY` coefficient to zero. Residual plots show a ceiling effect caused by the dataset capping house values at $500,000.

---

## Task 3 – Predictive Customer Churn Classifier (Random Forest & XGBoost)

**Dataset:** [Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) (7,043 customers, 26.5% churn)

**Approach**
1. Convert `TotalCharges` to numeric; the 11 blank values belong to `tenure = 0` customers and are filled with 0
2. One-hot encode categorical features, scale numeric features
3. Handle class imbalance with **SMOTE**, placed inside an `imblearn` Pipeline so only training folds are oversampled (no leakage)
4. Train Random Forest and XGBoost with 5-fold stratified cross-validation
5. Plot ROC curves, confusion matrices and the top 10 feature importances
6. Compare results with and without SMOTE

**Results (test set)**

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Random Forest + SMOTE | 0.835 | 0.768 | 0.553 | 0.655 | 0.600 |
| XGBoost + SMOTE | 0.839 | 0.780 | 0.578 | 0.631 | 0.604 |

Both models exceed the required ROC-AUC of 0.80. SMOTE leaves ROC-AUC almost unchanged but raises recall on churners from about 0.51–0.53 to 0.63–0.66. Top drivers: month-to-month contract, no online security / tech support, fiber-optic internet, electronic-check payment.

---

## Task 4 – NLP Sentiment Classification on Customer Reviews (TF-IDF)

**Datasets**
- `sentiment_reviews.csv` – the provided starter file (10 reviews: 5 positive, 4 negative, 1 neutral)
- NLTK `movie_reviews` – 2,000 labelled reviews, used for the benchmark

**Why two datasets:** the starter file is too small to benchmark a classifier (a 80/20 split leaves 2 test reviews, and Leave-One-Out accuracy is 11% because the reviews share almost no vocabulary). The notebook therefore:
- **Part A** demonstrates the full pipeline on the starter file (the single neutral review is excluded since the task is positive vs negative)
- **Part B** trains on 1,600 movie reviews plus the 9 product reviews, and benchmarks on 400 unseen hold-out movie reviews

**Approach**
1. Clean text: remove HTML tags, lowercase, strip punctuation, remove stopwords, lemmatise (NLTK). Negations (`not`, `no`, `nor`, `never`) are kept because they flip sentiment
2. `TfidfVectorizer(max_features=5000)`
3. Train Multinomial Naive Bayes and Logistic Regression
4. Evaluate with accuracy, precision, recall, F1 and confusion matrices; plot the most influential words
5. `predict_sentiment(text)` helper that returns the label and confidence; set `INTERACTIVE = True` in the last helper cell to type your own reviews

**Results (400 hold-out reviews)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Multinomial NB | 0.810 | 0.830 | 0.780 | 0.804 |
| Logistic Regression | 0.835 | 0.816 | 0.865 | 0.840 |

A TF-IDF bag-of-words model ignores word order and sarcasm, so predictions on very short custom phrases should be treated as indicative. Possible improvements: bigrams, `GridSearchCV` tuning, or transformer embeddings.

---

## Tools & Libraries

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-Learn · XGBoost · imbalanced-learn (SMOTE) · NLTK · Jupyter Notebook

## Author

**Shanmuga Sundaram A** (Broly)
B.Tech Artificial Intelligence & Data Science, Paavai Engineering College (Autonomous)

- GitHub: [github.com/shanmugasundaram-1](https://github.com/shanmugasundaram-1)
- LinkedIn: [linkedin.com/in/shanmuga-sundaram-0aa178420](https://linkedin.com/in/shanmuga-sundaram-0aa178420)
