# 🌧️ Rainfall Prediction with Logistic Regression

This project focuses on predicting whether it will rain tomorrow using historical weather data from Australia.
The main objective is to build a **robust baseline classification model** while properly handling **missing values** and **class imbalance**.

---

## 📌 Problem Definition

- **Task:** Binary classification  
- **Target variable:** `RainTomorrow` (Yes / No)  
- **Main challenge:** The dataset is **imbalanced**, with significantly fewer rainy days compared to non-rainy days.

---

## 📊 Dataset

- Source: *Rain in Australia* dataset  
- Rows after cleaning: ~142,000  
- Features include temperature, humidity, pressure, wind direction/speed, cloud coverage, and rainfall indicators.

```markdown
> Note: The dataset file (`weatherAUS.csv`) is not included in this repository.  
> Please download it from the original source and place it under `data/weatherAUS.csv`.

---

## 🧠 Methodology

The project is implemented as an end-to-end machine learning pipeline using `scikit-learn`.

### 1️⃣ Data Preprocessing
- Removed invalid and inconsistent target labels
- Handled missing values:
  - **Numerical features:** Median imputation (robust to outliers)
  - **Categorical features:** Most frequent value
- Applied **One-Hot Encoding** to categorical variables
- Scaled numerical features using **StandardScaler**

### 2️⃣ Model
- **Logistic Regression**
- `class_weight="balanced"` to address class imbalance
- Implemented using `Pipeline` and `ColumnTransformer` to prevent data leakage

---

## 📈 Results

| Metric | Value |
|------|------|
| Accuracy | **0.81** |
| F1-score (Rain = Yes) | **0.65** |
| Recall (Rain = Yes) | **0.79** |

> The model prioritizes recall for rainy days, which is desirable in weather-related risk prediction scenarios.

---

## 📂 Project Structure

rainfall-prediction/
├── data/
│ └── weatherAUS.csv
├── src/
│ └── train.py
├── models/
│ └── model_RainTomorrow.joblib
├── results/
│ └── metrics.json
├── requirements.txt
└── README.md

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/train.py

