# 💳 Credit Card Fraud Detection

> **Data Analyst Internship Project** — Machine Learning-based fraud detection on real-world transaction data using Random Forest and SMOTE.

---

## 📌 Project Overview

Credit card fraud costs the global financial industry billions of dollars annually. This project builds an end-to-end machine learning pipeline to **detect fraudulent transactions** from a highly imbalanced dataset of 284,807 credit card transactions.

The project covers the full data science workflow: exploratory data analysis, handling class imbalance, feature engineering, model training, and performance evaluation.

---

## 🗂️ Repository Structure

```
credit-card-fraud-detection/
│
├── creditcardfraud.ipynb        # Main Jupyter Notebook (full pipeline)
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencie

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Rows** | 284,807 transactions |
| **Features** | 30 (V1–V28 PCA components + `Time` + `Amount`) |
| **Target** | `Class` — `0` = Legitimate, `1` = Fraud |
| **Fraud Rate** | **0.17%** (highly imbalanced) |

> ⚠️ The dataset is not included in this repository due to size. Download it from Kaggle and place `creditcard.csv` in the project root.

---

## 🔍 Exploratory Data Analysis (EDA)

Key findings from EDA:

- **Severe Class Imbalance**: 99.83% legitimate vs. 0.17% fraudulent transactions
- **Transaction Amount**: Right-skewed distribution with significant outliers — boxplot reveals extreme high-value anomalies
- **Feature Correlations**: Heatmap of V1–V28 shows low inter-feature correlation (expected, as these are PCA-transformed)
- **Fraud Patterns**: Fraudulent transactions exhibit distinct distributions in several V-features

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Normalized `Amount` column using `StandardScaler`
- Dropped `Time` column (not informative after normalization)

### 2. Handling Class Imbalance — SMOTE
Used **Synthetic Minority Oversampling Technique (SMOTE)** to balance the dataset before training:

```
Before SMOTE → Class 0: 284,315  |  Class 1: 492
After SMOTE  → Class 0: 284,315  |  Class 1: 284,315
```

### 3. Model — Random Forest Classifier
```python
RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
```

- Trained on 80% of SMOTE-resampled data
- Tested on held-out 20%

---

## 📈 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.99% |
| **Precision** | 99.98% |
| **Recall** | 99.998% |
| **F1 Score** | 99.99% |
| **AUC-ROC** | ≈ 1.0000 |

> 🔑 **High Recall** is the most critical metric in fraud detection — it means almost zero fraudulent transactions were missed (minimizing false negatives).

### Confusion Matrix
The confusion matrix confirmed near-perfect separation between fraud and legitimate classes after SMOTE balancing.

### ROC Curve
The ROC curve hugged the upper-left corner with AUC ≈ 1.0, demonstrating excellent discriminative ability of the model.

### Top 15 Feature Importances
The most influential features for fraud prediction were primarily PCA-transformed components V17, V14, V12, V10, and V11 — consistent with domain knowledge that these components capture the most fraud-discriminative signal.

---

## 💡 Key Insights

1. **Class Imbalance is Critical**: Without SMOTE, a naive model would predict "legitimate" for everything and still achieve 99.83% accuracy — making raw accuracy a misleading metric.

2. **SMOTE Effectiveness**: Balancing the dataset dramatically improved the model's ability to learn fraud patterns, pushing Recall from ~80% to ~99.998%.

3. **Random Forest Robustness**: The ensemble approach handles the noisy, PCA-transformed features well and provides reliable feature importance scores.

4. **Real-world Applicability**: The model's high recall makes it suitable for integration into real-time transaction monitoring systems, where catching fraud is more important than the occasional false alarm.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Visualizations |
| Scikit-learn | ML models & metrics |
| imbalanced-learn | SMOTE oversampling |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root, or update the file path in the notebook:
```python
df = pd.read_csv("creditcard.csv")
```

### 4. Run the notebook
```bash
jupyter notebook creditcardfraud.ipynb
```

---

## 📋 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
jupyter
```

---

## 📝 Conclusion

This project demonstrates that machine learning — specifically a **Random Forest classifier with SMOTE balancing** — can effectively detect credit card fraud with near-perfect accuracy. The pipeline addresses the core real-world challenge of class imbalance and delivers a model suitable for deployment in financial fraud monitoring systems.

---

## 🙋 Author

**PILLA MADHAVI LATHA**
Data Analyst Intern
[LinkedIn](https://linkedin.com/in/yourprofile) • [GitHub](https://github.com/yourusername)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
