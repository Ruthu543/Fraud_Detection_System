# Credit Card Fraud Detection Using Machine Learning

An end-to-end Machine Learning project that detects potentially fraudulent credit card transactions using **Random Forest, XGBoost, and Isolation Forest**. The trained models are integrated into an interactive **Streamlit web application** for transaction-level fraud prediction.

## Project Overview

Credit card fraud is a major challenge for financial institutions. Since fraudulent transactions represent a very small percentage of total transactions, detecting them accurately is a highly imbalanced classification problem.

This project uses supervised learning and unsupervised anomaly detection techniques to identify suspicious transactions and compare their performance.

## Objectives

- Analyze an imbalanced credit card transaction dataset.
- Perform data preprocessing and exploratory data analysis (EDA).
- Handle class imbalance while preventing data leakage.
- Build and evaluate multiple machine learning models.
- Compare model performance using appropriate evaluation metrics.
- Detect potentially fraudulent and anomalous transactions.
- Deploy trained models through an interactive Streamlit application.

## Dataset

**Dataset:** Credit Card Fraud Detection

**Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Description | Value |
|---|---:|
| Total transactions | 284,807 |
| Normal transactions | 284,315 |
| Fraudulent transactions | 492 |
| Total features | 30 |
| Target column | Class |

The dataset contains anonymized transaction features (`V1` to `V28`), transaction time (`Time`), transaction amount (`Amount`), and the target variable (`Class`).

- `0` – Normal transaction
- `1` – Fraudulent transaction

**Note:** The dataset is not included in this repository. Download it from Kaggle and place `creditcard.csv` inside the `dataset/` folder.

## Technologies Used

- **Programming Language:** Python
- **Development Environment:** Jupyter Notebook, VS Code
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn, XGBoost
- **Imbalanced Data Handling:** SMOTE
- **Model Persistence:** Joblib
- **Web Application:** Streamlit

## Machine Learning Models

Three models are implemented:

### 1. Random Forest
A supervised ensemble learning algorithm used to classify transactions as normal or fraudulent.

### 2. XGBoost
A gradient boosting algorithm used for fraud classification, with class imbalance handling through `scale_pos_weight`.

### 3. Isolation Forest
An unsupervised anomaly detection algorithm trained using normal transactions to identify unusual transaction patterns.

**Model distinction:** Random Forest and XGBoost perform supervised fraud classification, while Isolation Forest identifies anomalies. An anomaly is not necessarily a fraudulent transaction.

## Project Workflow

1. **Data Collection:** Obtain the credit card fraud dataset from Kaggle.
2. **Exploratory Data Analysis:** Examine dataset structure, missing values, duplicates, feature distributions, and class imbalance.
3. **Data Preprocessing:** Remove duplicate records and prepare features and target labels.
4. **Data Splitting:** Split the data into training, validation, and testing sets using stratified sampling.
5. **Feature Scaling:** Apply appropriate scaling to transaction features.
6. **Imbalance Handling:** Apply SMOTE only to the supervised training data to avoid data leakage.
7. **Model Training:** Train Random Forest, XGBoost, and Isolation Forest.
8. **Model Evaluation:** Evaluate predictions using classification metrics and confusion matrices.
9. **Threshold Optimization:** Select classification thresholds using validation data.
10. **Model Saving:** Save trained models, scalers, and feature-column information using Joblib.
11. **Deployment:** Integrate the saved models into a Streamlit application.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC (Precision-Recall AUC)
- Confusion Matrix
- Classification Report

**Why these metrics?**

Accuracy alone is not sufficient for highly imbalanced datasets. Precision and recall help evaluate false fraud alerts and missed fraudulent transactions. PR-AUC is particularly useful for assessing performance on the minority fraud class.

### Model Performance Comparison


| Model | Accuracy | Precision | Recall | F1-score | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest |  100.0 % | 96.36% | 71.62% | 82.17%	 | 81.69% |
| XGBoost | 99.98 % | 88.24% |81.08%	 | 84.51%	 | 83.40% |
| Isolation Forest | 96.37 % | 23.00%	 | 24.00%	|23.53% | 12.75% |



## Streamlit Web Application

The project includes an interactive Streamlit application that allows users to enter transaction details and obtain predictions from the trained models.

### Application Features

- Interactive transaction input.
- Random Forest fraud prediction.
- XGBoost fraud prediction and fraud probability.
- Isolation Forest anomaly detection.
- Model prediction comparison.
- Fraud probability and decision threshold display.

### Application Screenshots

actual Streamlit screenshots here:

**1. Application Dashboard**
<img width="1894" height="795" alt="image" src="https://github.com/user-attachments/assets/44c1e1bf-b8f2-46a4-b713-3a23729eb630" />


`![Application Dashboard](screenshots/dashboard.png)`

**2. Normal Transaction Prediction**

`![Normal Transaction Prediction](screenshots/normal_prediction.png)`

**3. Fraudulent Transaction Prediction**

`![Fraudulent Transaction Prediction](screenshots/fraud_prediction.png)`



## Project Structure

```text
Fraud-Detection-Using-Machine-Learning 
│
├── dataset/
│   └── creditcard.csv
│
├── models/
│   ├── if_model.pkl
│   ├── if_scaler.pkl
│   ├── isolation_feature_columns.pkl
│   ├── rf_model.pkl
│   ├── supervised_feature_columns.pkl
│   ├── supervised_scaler.pkl
│   └── xgb_model.pkl
│
├── notebook/
│   └── fraud_detection.ipynb
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

The dataset and virtual environments are excluded from the GitHub repository. The model files are stored in the `models/` directory.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ruthu543/Fraud_Detection_System.git
```

### 2. Navigate to the project directory

```bash
cd Fraud-Detection-Using-Machine-Learning
```

### 3. Create a virtual environment

```bash
python -m venv .venv_streamlit
```

### 4. Activate the environment

**Windows PowerShell:**

```powershell
.venv_streamlit\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the dataset

Download `creditcard.csv` from Kaggle and place it inside the `dataset/` folder if you want to run the Jupyter Notebook.

### 7. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## Handling Unseen Transactions

The trained models are evaluated on transactions that were not used during training.

Random Forest and XGBoost use learned classification patterns to predict unseen transactions. Isolation Forest identifies transactions that differ from the normal transaction distribution learned during training.

Although anomaly detection can help identify unusual patterns, it cannot guarantee detection of every previously unseen fraud technique.

## Challenges and Limitations

- Highly imbalanced transaction data.
- Reducing false positives while maintaining high fraud recall.
- Selecting suitable classification thresholds.
- Detecting previously unseen
